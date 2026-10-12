"""智能分诊服务：急症拦截 → 结构化提取 → 证据召回 → 科室推荐。

设计原则（对应产品对「AI 工程化」的要求）：

1. **急症拦截不走模型**。命中急症关键词直接给出急诊引导，LLM 配额耗尽、
   供应商宕机、模型幻觉都不能延误真正的急症患者——这是医疗安全的红线，
   所以它是第 0 步而不是和 LLM 并行的一路。
2. **AI 是辅助路径，不是必需路径**。LLM 降级 / 结构化校验失败 / 内容审核
   拦截时，退回规则分诊（症状 × 科室关键词匹配打分），流程照常出结论，
   只在响应里标注 decision_source=rule 供后续评估依赖度。
3. **RAG 证据可选**。知识库尚未向量化、检索异常、零结果都不阻塞分诊；
   检索到就把出处附在 evidence 里给前端展示，提升患者信任。
4. **同步主路径 + 异步补偿**。HTTP 走同步分诊；run_triage_workflow 任务
   用于批量评估与「规则结果的事后 AI 补偿」，两条路径共用本模块的实现体。

分诊结果落 TriageResult：converted_appointment_id / actual_department_id 由
后续阶段回填，用于计算分诊转化率与准确率（见 models/ai.py 注释）。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.ai import TriageResult
from app.models.auth import Department
from app.models.patient import PatientProfile
from app.services.llm.service import get_llm_service
from app.services.llm.types import ChatMessage, LLMRequest, LLMScenario
from app.services.rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)

# ==================== 急症关键词 ====================

# 词表面向中文口语主诉（患者不会写医学术语）。分两级：
# - 高危词：单独出现即视为急症（如「意识不清」没有良性解读场景）
# - 组合词：需与部位/程度词共现才触发（如「头痛」常见于感冒，
#   「剧烈头痛+呕吐」才是颅高压征象）——第一版全部按高危处理，
#   误报的代价是急诊科多接一个电话，漏报的代价是命。安全不对称，
#   所以宁可敏感。词表扩充走配置化注释，见 EXTENSION_NOTE。
EMERGENCY_KEYWORDS: tuple[str, ...] = (
    "呼吸困难",
    "喘不上气",
    "气促加重",
    "胸痛",
    "胸口压榨",
    "心口疼得厉害",
    "意识不清",
    "昏倒",
    "昏迷",
    "叫不醒",
    "抽搐",
    "痉挛发作",
    "偏瘫",
    "一侧肢体无力",
    "嘴歪眼斜",
    "说话不清突然",
    "大出血",
    "出血不止",
    "咯血",
    "呕血",
    "便血黑色",
    "剧烈头痛",
    "头痛伴呕吐",
    "高热惊厥",
    "婴幼儿高热",
    "婴儿发烧抽搐",
    "中毒",
    "服毒",
    "药物过量",
    "过敏性休克",
    "全身起疹伴呼吸困难",
    "车祸",
    "工伤",
    "严重外伤",
    "骨折外露",
    "烫伤面积大",
)

# 词表刻意保持「宁可误报」的取向：胸痛既可能是心梗也可能是肋间神经痛，
# 但把胸痛分诊到门诊而非急诊的漏报不可接受。误报由急诊分诊台兜底，
# 这在分诊准确率统计里会算成「推荐科室与实际不符」，评估时应剔除急症分支。
# 若后续要做组合词语法，把元组升级为 (pattern, require_terms) 结构即可，
# 调用点只有 _match_emergency 一处。

_EMERGENCY_PATTERN = re.compile("|".join(re.escape(k) for k in EMERGENCY_KEYWORDS))


def _match_emergency(text: str) -> list[str]:
    """返回命中的急症关键词（去重保序）。"""
    found: list[str] = []
    for kw in EMERGENCY_KEYWORDS:
        if kw in text and kw not in found:
            found.append(kw)
    return found


# ==================== 结构化提取 schema ====================


class SymptomExtract(BaseModel):
    """LLM 从主诉中提取的结构化症状。

    字段设计对着 TriageResult.extracted_symptoms 与规则匹配的需要来：
    症状词用于 RAG 检索与科室关键词匹配，duration/severity 供紧急度判断，
    department_hint 是模型对科室的直觉判断——只作为融合的加分项，
    不单独定结论（模型不认识本系统的科室 code 是常态，不能依赖它）。
    """

    symptoms: list[str] = Field(default_factory=list, description="症状短语列表，每项 2-8 字")
    duration: str | None = Field(default=None, description="病程描述，如 '3天'、'一周余'")
    severity: int = Field(default=3, ge=1, le=5, description="主观严重程度 1 最轻 5 最重")
    accompanying: list[str] = Field(default_factory=list, description="伴随症状")
    department_hint: str | None = Field(
        default=None, description="模型猜测的科室名或 code，允许为空"
    )


TRIAGE_EXTRACT_PROMPT = """你是医院智能分诊助手。根据患者主诉提取结构化症状信息。

要求：
1. 只提取患者描述中出现的内容，不要臆造症状；不确定的字段留空
2. 把口语症状规范为医学短语（如"肚子疼"→"腹痛"，"喘不上气"→"呼吸困难"）
3. severity 依据：影响日常活动为 4-5，可忍受但明显不适为 2-3，轻微为 1
4. 严格输出 JSON，不要任何解释文字，格式如下：
{"symptoms": ["..."], "duration": "...", "severity": 1-5, "accompanying": ["..."], "department_hint": "..."}

患者主诉：
{chief_complaint}"""

TRIAGE_DEPT_PROMPT = """你是医院智能分诊助手。根据患者症状，从以下科室列表中选择最合适的就诊科室。

可选科室（名称 / 常见症状关键词）：
{departments}

判断规则：
1. 症状与哪个科室的关键词重合最多且最能解释主要不适，就选哪个
2. 只能从上述列表中选择，不要发明科室
3. 严格输出 JSON，格式：{"department_name": "...", "reason": "..."}

患者症状：{symptoms}
原始主诉：{chief_complaint}"""


# ==================== 数据结构 ====================


@dataclass(slots=True)
class EvidenceItem:
    """一条 RAG 证据（给前端展示「推荐依据」用）。"""

    title: str
    source: str
    source_tier: str
    snippet: str
    rrf_score: float


@dataclass(slots=True)
class TriageOutcome:
    """分诊的内部结果对象（service 层产物，路由负责转 schema）。"""

    triage_id: uuid.UUID
    urgency: str  # routine / urgent / stat
    is_emergency: bool
    matched_emergency_keywords: list[str]
    department_id: uuid.UUID | None
    department_name: str | None
    department_code: str | None
    confidence: float | None
    reason: str
    advice: str | None
    decision_source: str  # ai / rule / hybrid / emergency
    degraded: bool
    extracted: dict[str, Any] | None
    evidence: list[EvidenceItem] = field(default_factory=list)
    llm_model: str | None = None
    trace_id: str | None = None


# ==================== 规则分诊 ====================


async def _load_departments(db: AsyncSession) -> list[Department]:
    return list((await db.execute(select(Department))).scalars().all())


def _score_departments(
    terms: list[str], departments: list[Department]
) -> list[tuple[Department, int, list[str]]]:
    """症状词 × 科室 symptom_keywords 的双向包含匹配打分。

    为什么双向包含：中文分词粒度不一致，"胸痛"匹配关键词"胸部疼痛"，
    反过来"腹痛腹泻"也可能整体命中症状短语。只做一个方向会漏。
    命中权重按「症状词命中」计，同分时关键词数量多的科室不加分——
    大而全的科室（内科）关键词长尾多，靠数量堆分它会吸走所有分诊，
    这是规则分诊最常见的失真，刻意不做数量加成。
    """
    scored: list[tuple[Department, int, list[str]]] = []
    norm_terms = [t.strip() for t in terms if t and t.strip()]
    for dept in departments:
        keywords = dept.symptom_keywords or []
        hits: list[str] = []
        for term in norm_terms:
            for kw in keywords:
                if not kw:
                    continue
                if kw in term or term in kw:
                    if kw not in hits:
                        hits.append(kw)
                    break
        scored.append((dept, len(hits), hits))
    scored.sort(key=lambda x: (-x[1], x[0].code))
    return scored


def _rule_decide(
    terms: list[str], departments: list[Department], chief_complaint: str
) -> tuple[Department | None, float, str]:
    """纯规则兜底：症状词 + 原始主诉一起匹配。返回 (科室, 置信度, 理由)。"""
    scored = _score_departments(terms + [chief_complaint], departments)
    if not scored or scored[0][1] == 0:
        return None, 0.0, "症状无法匹配到任何科室关键词，建议先到内科或全科门诊分诊"
    dept, score, hits = scored[0]
    # 置信度：单科室明确命中给 0.6 封顶（规则法天然不如模型理解语境），
    # 并列第一（其他科室同分）时下调给 0.4 并在理由中列出备选。
    ties = [d.code for d, s, _ in scored[1:] if s == score]
    confidence = 0.4 if ties else min(0.6, 0.2 + 0.1 * score)
    reason = f"症状命中{dept.name}的关键词：{'、'.join(hits)}"
    if ties:
        tie_names = [d.name for d, s, _ in scored[1:] if s == score]
        reason += f"；备选科室：{'、'.join(tie_names)}"
    return dept, confidence, reason


# ==================== LLM 路径 ====================


async def _extract_symptoms(
    chief_complaint: str, *, trace_id: str | None, user_id: uuid.UUID | None
) -> tuple[SymptomExtract | None, str | None, bool]:
    """第 1 步：结构化症状提取。返回 (结果, 模型名, 是否降级)。"""
    service = get_llm_service()
    request = LLMRequest(
        scenario=LLMScenario.TRIAGE,
        messages=[
            ChatMessage(
                role="system",
                content="你是严谨的医院分诊助手，只输出符合要求的 JSON。",
            ),
            ChatMessage(
                role="user",
                content=TRIAGE_EXTRACT_PROMPT.replace("{chief_complaint}", chief_complaint),
            ),
        ],
        trace_id=trace_id,
        user_id=user_id,
        workflow_node="triage_extract",
    )
    try:
        parsed, response = await service.structured_chat(request, SymptomExtract)
    except Exception as exc:  # noqa: BLE001 - 分诊主路径不允许模型故障中断
        logger.warning("分诊症状提取异常，转规则兜底：%s", exc)
        return None, None, True
    if parsed is None:
        return None, response.model, True
    return parsed, response.model, response.is_degraded


async def _ask_department(
    extract: SymptomExtract,
    chief_complaint: str,
    departments: list[Department],
    *,
    trace_id: str | None,
    user_id: uuid.UUID | None,
) -> tuple[str | None, str | None, str | None, bool]:
    """第 3 步（AI 路）：让模型在候选科室里做选择。

    返回 (科室名, 理由, 模型名, 是否降级)。科室名可能不在列表里（幻觉），
    由调用方验证后回落规则路——模型只建议，规则做合法性校验，
    这是「AI 输出进业务前必须过结构关」的落地形态。
    """

    class DepartmentChoice(BaseModel):
        department_name: str | None = None
        reason: str | None = None

    dept_lines = "\n".join(
        f"- {d.name}（code={d.code}）：{'、'.join((d.symptom_keywords or [])[:8])}"
        for d in departments
    )
    symptom_text = "、".join([*extract.symptoms, *extract.accompanying]) or chief_complaint
    prompt = (
        TRIAGE_DEPT_PROMPT.replace("{departments}", dept_lines)
        .replace("{symptoms}", symptom_text)
        .replace("{chief_complaint}", chief_complaint)
    )
    service = get_llm_service()
    request = LLMRequest(
        scenario=LLMScenario.TRIAGE,
        messages=[
            ChatMessage(role="system", content="你是严谨的医院分诊助手，只输出 JSON。"),
            ChatMessage(role="user", content=prompt),
        ],
        trace_id=trace_id,
        user_id=user_id,
        workflow_node="triage_department",
    )
    try:
        parsed, response = await service.structured_chat(request, DepartmentChoice)
    except Exception as exc:  # noqa: BLE001
        logger.warning("分诊科室推荐异常，转规则兜底：%s", exc)
        return None, None, None, True
    if parsed is None:
        return None, None, response.model, True
    return parsed.department_name, parsed.reason, response.model, response.is_degraded


async def _gather_evidence(
    db: AsyncSession,
    query_text: str,
    *,
    trace_id: str | None,
    user_id: uuid.UUID | None,
) -> list[EvidenceItem]:
    """第 2 步：RAG 证据召回。任何失败都吞掉返回空列表。"""
    try:
        response = await RAGRetriever().retrieve(
            db, query_text, trace_id=trace_id, user_id=user_id
        )
    except Exception as exc:  # noqa: BLE001 - 证据是加分项，故障不反噬主路径
        logger.warning("分诊 RAG 证据检索失败，按无证据处理：%s", exc)
        return []
    items: list[EvidenceItem] = []
    for r in response.results[: settings.rag_result_count]:
        items.append(
            EvidenceItem(
                title=r.title,
                source=r.source,
                source_tier=r.source_tier,
                snippet=r.content[:120] + ("…" if len(r.content) > 120 else ""),
                rrf_score=round(r.rrf_score, 5),
            )
        )
    return items


# ==================== 主入口 ====================


async def assess(
    db: AsyncSession,
    *,
    chief_complaint: str,
    user_id: uuid.UUID | None = None,
    patient_id: uuid.UUID | None = None,
    ai_task_id: uuid.UUID | None = None,
    trace_id: str | None = None,
    allow_rule_update: bool = False,
    existing: TriageResult | None = None,
    rule_only: bool = False,
) -> TriageOutcome:
    """完整分诊流程。同步 HTTP 与 Celery 补偿共用。

    Args:
        rule_only: 跳过全部 LLM 节点只走规则路。批量评估先落规则结果，
            随后由 run_triage_workflow 逐条带 existing 重跑覆盖为 AI 结论——
            规则先行保证「评估者立刻能查」，异步补偿提升质量且不阻塞请求。
        allow_rule_update/existing：异步补偿场景——payload 里带已有 TriageResult，
        规则分诊先行落库，LLM 恢复后重跑本函数覆盖结论，但 decision_source
        从 rule 升级为 ai/hybrid 并保留规则理由快照（写进 advice 前缀），
        不新增历史表：补偿是低频路径，字段级快照够用。
    """
    trace_id = trace_id or uuid.uuid4().hex
    text = chief_complaint.strip()

    # ---------- 第 0 步：急症拦截 ----------
    matched = _match_emergency(text)
    if matched:
        emergency_dept = await db.execute(
            select(Department).where(Department.code == "EMERGENCY")
        )
        dept = emergency_dept.scalar_one_or_none()
        record = existing or TriageResult(
            patient_id=patient_id,
            user_id=user_id,
            trace_id=trace_id,
            chief_complaint=text,
        )
        record.is_emergency = True
        record.matched_emergency_keywords = matched
        record.urgency = "stat"
        record.decision_source = "emergency"
        record.recommended_department_id = dept.id if dept else None
        record.recommended_department_name = dept.name if dept else "急诊科"
        record.advice = (
            f"主诉命中急症警示词（{'、'.join(matched)}），请立即前往急诊科就诊"
            "或拨打 120；不要等待门诊号源。"
        )
        record.confidence = 1.0
        if record.id is None:
            db.add(record)
        await db.flush()
        await db.commit()
        return TriageOutcome(
            triage_id=record.id,
            urgency="stat",
            is_emergency=True,
            matched_emergency_keywords=matched,
            department_id=dept.id if dept else None,
            department_name=dept.name if dept else "急诊科",
            department_code="EMERGENCY",
            confidence=1.0,
            reason="急症关键词拦截，未调用模型",
            advice=record.advice,
            decision_source="emergency",
            degraded=False,
            extracted=None,
            trace_id=trace_id,
        )

    # ---------- 第 1 步：结构化提取 ----------
    departments = await _load_departments(db)
    if rule_only:
        # 批量评估的规则先行路径：整条 LLM 链路跳过（提取与科室推荐都不调），
        # 症状词直接用原始主诉做规则匹配，零 token 消耗。
        extract, model_name, degraded_extract = None, None, False
    else:
        extract, model_name, degraded_extract = await _extract_symptoms(
            text, trace_id=trace_id, user_id=user_id
        )

    rule_dept, rule_conf, rule_reason = _rule_decide(
        extract.symptoms + extract.accompanying if extract else [],
        departments,
        text,
    )

    decision_source = "rule"
    dept: Department | None = rule_dept
    confidence = rule_conf
    reason = rule_reason
    degraded = degraded_extract
    llm_model = model_name

    # ---------- 第 3 步：科室推荐（AI 路 + 融合） ----------
    if extract is not None and not degraded_extract:
        ai_dept_name, ai_reason, dept_model, degraded_dept = await _ask_department(
            extract, text, departments, trace_id=trace_id, user_id=user_id
        )
        llm_model = llm_model or dept_model
        ai_dept: Department | None = None
        if ai_dept_name:
            ai_dept = _match_department(dept_name_or_code=ai_dept_name, departments=departments)
        if degraded_dept:
            degraded = True
        if ai_dept is not None and rule_dept is not None and ai_dept.id == rule_dept.id:
            # 两路一致：融合加权上调置信度（这是最可信的分诊形态）
            confidence = min(0.95, max(confidence, 0.75) + 0.1)
            reason = f"模型与规则一致指向{ai_dept.name}：{ai_reason or rule_reason}"
            decision_source = "hybrid"
        elif ai_dept is not None and not degraded_dept:
            # 模型给出合法科室但与规则分歧：以模型为准、规则作备选展示
            dept = ai_dept
            confidence = 0.65
            reason = (
                f"模型建议{ai_dept.name}：{ai_reason or '（模型未给出理由）'}；"
                f"规则备选{rule_dept.name if rule_dept else '无'}（{rule_reason}）"
            )
            decision_source = "ai"
        # 非法科室名 / 降级：保持规则结果，decision_source 仍 rule

    # ---------- 第 2 步：RAG 证据（非阻塞加分） ----------
    # rule_only 同样跳过：证据检索要经 embed 消耗模型配额，
    # 批量评估的规则先行阶段要求整条链路零 LLM 调用。
    evidence: list[EvidenceItem] = []
    if not rule_only:
        evidence_query = "、".join(extract.symptoms[:6]) if extract else text
        evidence = await _gather_evidence(
            db, evidence_query, trace_id=trace_id, user_id=user_id
        )

    # ---------- 落库 ----------
    urgency = "urgent" if (extract and extract.severity >= 4) else "routine"
    record = existing or TriageResult(
        patient_id=patient_id,
        user_id=user_id,
        trace_id=trace_id,
        chief_complaint=text,
    )
    record.user_id = user_id
    record.patient_id = patient_id
    record.trace_id = trace_id
    record.extracted_symptoms = (
        extract.model_dump() if extract is not None else {"fallback_terms": [text[:50]]}
    )
    record.recommended_department_id = dept.id if dept else None
    record.recommended_department_name = dept.name if dept else None
    record.urgency = urgency
    record.is_emergency = False
    record.advice = _build_advice(dept, evidence, degraded)
    record.confidence = confidence
    record.decision_source = decision_source
    record.ai_task_id = ai_task_id
    record.llm_model = llm_model
    if existing is not None and existing.decision_source == "rule" and decision_source != "rule":
        # 补偿覆盖：规则结论快照保留在 advice 前缀，零新增字段
        snapshot = existing.advice or existing.chief_complaint
        record.advice = f"（规则分诊快照：{snapshot}）{record.advice}"
    if record.id is None:
        db.add(record)
    await db.flush()
    await db.commit()

    return TriageOutcome(
        triage_id=record.id,
        urgency=urgency,
        is_emergency=False,
        matched_emergency_keywords=[],
        department_id=dept.id if dept else None,
        department_name=dept.name if dept else None,
        department_code=dept.code if dept else None,
        confidence=confidence,
        reason=reason,
        advice=record.advice,
        decision_source=decision_source,
        degraded=degraded,
        extracted=extract.model_dump() if extract else None,
        evidence=evidence,
        llm_model=llm_model,
        trace_id=trace_id,
    )


def _match_department(*, dept_name_or_code: str, departments: list[Department]) -> Department | None:
    """把模型返回的科室名/code 映射回本系统科室；映射不到视为幻觉。

    宽松匹配（包含/反包含）刻意只用于**名称**，code 必须精确相等：
    模型爱输出「心血管内科」「心内科」「CARDIO」三种形态，
    名称做包含匹配，code 直接等值，都不中就返回 None 交规则路兜底。
    """
    name = dept_name_or_code.strip()
    for d in departments:
        if d.code == name:
            return d
    for d in departments:
        if name and (name in d.name or d.name in name):
            return d
    return None


def _build_advice(
    dept: Department | None, evidence: list[EvidenceItem], degraded: bool
) -> str:
    parts: list[str] = []
    if dept is not None:
        parts.append(f"建议挂{dept.name}号")
    else:
        parts.append("症状描述不足以定位科室，建议先到内科或全科门诊分诊")
    if evidence:
        refs = "；".join(f"《{e.title}》（来源：{e.source}）" for e in evidence[:2])
        parts.append(f"健康科普参考：{refs}")
    if degraded:
        parts.append("（当前智能服务为降级模式，建议仅供参考，请以现场分诊为准）")
    parts.append("出现急症表现请直接前往急诊或拨打 120。")
    return "。".join(parts) + "。"


async def list_results(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    role_name: str,
    patient_id: uuid.UUID | None = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[TriageResult], int]:
    """分诊历史分页。患者只看自己的；医生/管理员按矩阵可读全部，
    这里传 patient_id 则收窄到单个患者。行级隔离在服务层强制。"""
    stmt = select(TriageResult)
    if role_name == "patient":
        if patient_id is None:
            # 患者自己的档案 id：注册已同步建档
            profile = await db.execute(
                select(PatientProfile.id).where(PatientProfile.user_id == user_id)
            )
            own = profile.scalar_one_or_none()
            if own is None:
                return [], 0
            stmt = stmt.where(TriageResult.patient_id == own)
        else:
            stmt = stmt.where(TriageResult.patient_id == patient_id)
    elif patient_id is not None:
        stmt = stmt.where(TriageResult.patient_id == patient_id)

    from sqlalchemy import func

    total = (
        await db.execute(
            select(func.count()).select_from(stmt.subquery())
        )
    ).scalar_one()
    rows = await db.execute(
        stmt.order_by(TriageResult.created_at.desc())
        .offset((max(1, page) - 1) * page_size)
        .limit(page_size)
    )
    return list(rows.scalars().all()), int(total)


async def get_result(db: AsyncSession, triage_id: uuid.UUID) -> TriageResult:
    from app.core.exceptions import ResourceNotFoundError

    row = await db.execute(select(TriageResult).where(TriageResult.id == triage_id))
    record = row.scalar_one_or_none()
    if record is None:
        raise ResourceNotFoundError("分诊记录不存在")
    return record


__all__ = [
    "EMERGENCY_KEYWORDS",
    "SymptomExtract",
    "TriageOutcome",
    "assess",
    "get_result",
    "list_results",
    "_match_emergency",
]
