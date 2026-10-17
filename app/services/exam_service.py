"""检验科业务服务：申请单接收、报告录入、危急值判定与 AI 报告解读工作流。

阶段 4 的检验侧边界：**ExamRequest 从 ``received`` 起归本模块**。
``ordered``（医生开单）由医生侧写入，这里只消费它——所有函数都假定传入的申请单
已经存在，绝不创建或修改 ordered 之前的字段。

============================ 三条贯穿全模块的设计原则 ============================

1. **确定性规则优先于 AI**（见 ``evaluate_abnormalities`` / ``CRITICAL_RULES``）
   异常旗标与危急值全部由「实测值 × 参考区间」的规则比对得出，一次 LLM 都不调。
   理由不是省 token，而是医疗安全：模型幻觉、降级话术、供应商宕机都可能让
   「血钾 2.8」被说成「基本正常」，而低钾血症可致恶性心律失常。危急值是抢救
   指征，它的判定必须可复算、可审计、可在断网时仍然生效。AI 只负责把规则已经
   判出来的异常**解释给患者听**，不负责判断它是否异常。

2. **AI 是辅助路径，绝不能阻塞医疗数据录入**（见 ``submit_report``）
   报告录入接口先把 raw_content / structured_data / abnormal_flags / conclusion
   落库并提交，之后才尝试派发解读任务。broker 挂了就把 AITask 标 failed，
   接口照样返回 201。反过来（先派发后落库）会让检验科在 Redis 故障时录不进报告，
   等于让一个锦上添花的功能卡死临床主流程。

3. **AI 永不直接下诊断结论**（见 ``ReportInterpretation`` 的 need_doctor_review）
   该字段恒为 True，且校验层拒绝 False。输出的措辞口径是「提示/建议复查」而不是
   「诊断为」。医生的审核动作（reported→reviewed）才是报告的生效条件。

============================ 异步交互模型 ============================

录入 → 立即返回（ai_status=pending）→ 前端轮询 GET /exams/reports/{id}
看 ai_status 从 pending → processing → completed/failed/skipped。
同步等 LLM 会挂住 HTTP 请求十几秒并诱发重复提交，详见路由 docstring。
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.core.exceptions import (
    InvalidStateTransitionError,
    ResourceNotFoundError,
)
from app.models.ai import AITask, AITaskStatus
from app.models.clinical import (
    ExamReport,
    ExamRequest,
    ExamRequestStatus,
    Urgency,
    Visit,
)
from app.models.patient import PatientProfile
from app.services.llm.service import get_llm_service
from app.services.llm.types import ChatMessage, LLMRequest, LLMScenario
from app.services.observability import (
    build_step,
    finish_workflow_trace_sync,
    start_workflow_trace,
    sum_usage,
)
from app.services.rag.retriever import RAGRetriever
from app.services.state_machine import assert_transition

logger = logging.getLogger(__name__)

# ==================== 常量 ====================

# workflow_traces.workflow_name 取值（models/ai.py 注释登记的三种之一）。
# 字面量与 AITask.task_type 常量 TASK_TYPE_REPORT_INTERPRET 相同，但刻意分列：
# 一个是追踪口径、一个是任务路由口径，改一个不该顺手动另一个。
WORKFLOW_NAME_REPORT_INTERPRET = "report_interpret"
# 解读工作流的计划节点数：规则判定 / 证据召回 / prompt 组装 / LLM 调用 / 结果回写。
REPORT_INTERPRET_NODE_COUNT = 5

# AITask.task_type 取值。models/ai.py 注释里登记的正是这个字符串，
# 派发侧与 worker 侧必须共用常量，否则按 task_type 查重的索引语义会失效。
TASK_TYPE_REPORT_INTERPRET = "report_interpret"

# 解读任务的 Celery 全限定名。写错一个字符 worker 就永远收不到消息，
# 而且 send_task 对不存在的任务不会报错（消息只是躺在队列里），
# 因此把它收敛成常量并由测试静态断言。
INTERPRET_TASK_NAME = "app.tasks.ai_tasks.interpret_exam_report"

# ExamReport.ai_status 取值集合（models/clinical.py 列注释登记的五种）。
# 没有 CHECK 约束兜底，写入一律走常量，避免大小写漂移导致前端轮询认不出来。
AI_STATUS_PENDING = "pending"
AI_STATUS_PROCESSING = "processing"
AI_STATUS_COMPLETED = "completed"
AI_STATUS_FAILED = "failed"
AI_STATUS_SKIPPED = "skipped"

# 「进行中」的 ai_status 集合。派发侧只用 processing 做重复拦截
# （pending 必须允许重投，理由见 dispatch_interpretation），
# 这个常量给路由与前端判断「要不要显示重新解读按钮」用。
ACTIVE_AI_STATUSES: frozenset[str] = frozenset(
    {AI_STATUS_PENDING, AI_STATUS_PROCESSING}
)

# 终态 ai_status：completed 表示已成功（重触发无意义，幂等直接返回），
# failed/skipped 允许人工重新触发。
TERMINAL_AI_STATUSES: frozenset[str] = frozenset(
    {AI_STATUS_COMPLETED, AI_STATUS_FAILED, AI_STATUS_SKIPPED}
)

# 检验科工作台要处理的申请单状态。reported/reviewed 不在其中——
# 出完报告的单子在工作台留着只会让人误以为还没做。
LAB_QUEUE_STATUSES: tuple[str, ...] = (
    ExamRequestStatus.RECEIVED.value,
    ExamRequestStatus.PROCESSING.value,
)

# urgency 排序权重。STAT 排在最前是「立即」（危急值复测），URGENT 次之。
_URGENCY_RANK: dict[str, int] = {
    Urgency.STAT.value: 0,
    Urgency.URGENT.value: 1,
    Urgency.ROUTINE.value: 2,
}

# ==================== 危急值表 ====================

# 危急值（critical value）= 检验结果危及生命、必须立即干预的阈值。
#
# **为什么用独立常量表而不是复用参考区间**：参考区间是「人群分布」口径
# （95% 健康人的波动范围），危急值是「临床风险」口径（低于此值可能致死）。
# 两者数值经常重合但语义完全不同：血钾参考下限 3.5，危急下限 3.0；
# 血红蛋白参考下限 115，危急下限 60。混用会把「轻度偏低」误报成需要叫抢救团队的
# 红色警报，也会漏掉真正要命的值。所以这里单独一张表，且刻意比参考区间更窄。
#
# **演示口径声明**：下列阈值取自国内三甲医院公开的《检验危急值项目表》常见取值
# 区间的中位水平（不同医院、不同检测平台、成人/新生儿口径差异可达 20%），
# 本项目用于工程演示，**不得作为真实临床决策依据**。落地生产必须由医务部
# 按本院 LIS 的实际危急值目录逐条核定，并把这张表迁到配置或数据库字典表，
# 让它能被有资质的人改而不用发版。
#
# code 匹配用「归一化后精确相等」而非包含匹配：K 会被 "PK"、"KIM-1" 这类
# code 误命中，而误命中一条「血钾危急值」触发的是全院通报级别的告警，
# 代价远大于漏配几个别名。别名统一在 _normalize_code 里折叠。
CRITICAL_RULES: tuple[dict[str, Any], ...] = (
    {"code": "K", "name": "血钾", "low": 3.0, "high": 6.0, "unit": "mmol/L"},
    {"code": "GLU", "name": "血糖", "low": 2.8, "high": 22.2, "unit": "mmol/L"},
    {"code": "HGB", "name": "血红蛋白", "low": 60.0, "unit": "g/L"},
    {"code": "PLT", "name": "血小板计数", "low": 50.0, "unit": "10^9/L"},
    {"code": "WBC", "name": "白细胞计数", "low": 2.0, "high": 30.0, "unit": "10^9/L"},
    {"code": "CRT", "name": "血肌酐", "high": 530.0, "unit": "umol/L"},
    {"code": "UREA", "name": "血尿素氮", "high": 21.4, "unit": "mmol/L"},
    {"code": "CA", "name": "血钙", "low": 1.75, "high": 3.37, "unit": "mmol/L"},
    {"code": "NA", "name": "血钠", "low": 120.0, "high": 160.0, "unit": "mmol/L"},
    {"code": "CL", "name": "血氯", "low": 90.0, "high": 110.0, "unit": "mmol/L"},
    {"code": "CO2", "name": "二氧化碳结合力", "low": 13.0, "unit": "mmol/L"},
    {"code": "INR", "name": "国际标准化比值", "high": 5.0, "unit": ""},
    {"code": "PT", "name": "凝血酶原时间", "high": 30.0, "unit": "s"},
    {"code": "APTT", "name": "活化部分凝血活酶时间", "high": 70.0, "unit": "s"},
    {"code": "AMY", "name": "血淀粉酶", "high": 350.0, "unit": "U/L"},
)

# 常见别名 → 标准 code。LIS 厂商字段名千奇百怪（Serum K / 血钾 / Potassium），
# 只在这一处折叠，新增别名不影响任何判定逻辑。
_CODE_ALIASES: dict[str, str] = {
    "POTASSIUM": "K",
    "SERUMK": "K",
    "KPLUS": "K",
    "BLDGLU": "GLU",
    "GLUCOSE": "GLU",
    "BG": "GLU",
    "FBG": "GLU",
    "HB": "HGB",
    "HGB": "HGB",
    "HEMOGLOBIN": "HGB",
    "BLC": "HGB",
    "PLTC": "PLT",
    "TPC": "PLT",
    "WBCCOUNT": "WBC",
    "CRE": "CRT",
    "CREAT": "CRT",
    "CREATININE": "CRT",
    "BUN": "UREA",
    "UREA": "UREA",
    "CA2": "CA",
    "CALCIUM": "CA",
    "NAPLUS": "NA",
    "SODIUM": "NA",
    "CLPLUS": "CL",
    "CHLORIDE": "CL",
    "CO2CP": "CO2",
    "TCO2": "CO2",
    "PROTHROMBIN": "PT",
}


def _normalize_code(raw: Any) -> str:
    """把指标 code 归一化成用于匹配的形态。

    去掉空格/连字符/下划线/点，转大写，再折叠别名。
    ``"serum k"`` / ``"Serum-K"`` / ``"血清钾"`` 三种写法都能落到同一个判定上。
    """
    text = str(raw or "").strip().upper()
    for ch in (" ", "-", "_", ".", "/"):
        text = text.replace(ch, "")
    return _CODE_ALIASES.get(text, text)


# ==================== structured_data 契约（阶段 5 / 前端以此为准） ====================

STRUCTURED_DATA_CONTRACT = """{
  "items": [
    {"name": "血钾", "code": "K", "value": 2.8, "unit": "mmol/L",
     "ref_low": 3.5, "ref_high": 5.3}
  ]
}"""

# 契约说明放在代码注释里而不是只写文档：改动解析逻辑时必须同时看到这段话。
# - items 必填，每项至少要有 name（或 code）与 value
# - name：中文显示名，缺失时用 code 顶上
# - code：英文代号，用于匹配危急值表；缺失则只做区间比对不做危急值
# - value：数值或数字字符串（"2.8" 等价 2.8）；无法解析成数的项跳过判定
#   但仍保留在 structured_data 里（定性项如「尿蛋白 +/-」不该被丢掉）
# - unit：单位，仅展示用，不做换算
# - ref_low / ref_high：参考区间端点，任一可缺（单项参考只有上限的情况常见）
#   两端都缺的项视为「无法判定」，不出旗标
# - 顶层多余的键（如 collected_at / instrument）原样保留，不校验、不清洗


@dataclass(slots=True)
class AbnormalItem:
    """一条越界指标的判定明细（落库只存文本旗标，明细留给响应与日志）。"""

    name: str
    code: str
    value: float
    unit: str
    ref_low: float | None
    ref_high: float | None
    direction: str  # high / low

    def to_flag(self) -> str:
        """生成人读旗标文本，如「血钾 2.8 ↓(参考 3.5-5.3)」。"""
        arrow = "↑" if self.direction == "high" else "↓"
        return f"{self.name} {_fmt_number(self.value)} {arrow}{self._ref_text()}"

    def _ref_text(self) -> str:
        if self.ref_low is not None and self.ref_high is not None:
            return f"(参考 {_fmt_number(self.ref_low)}-{_fmt_number(self.ref_high)})"
        if self.ref_high is not None:
            return f"(参考 ≤{_fmt_number(self.ref_high)})"
        if self.ref_low is not None:
            return f"(参考 ≥{_fmt_number(self.ref_low)})"
        return ""


@dataclass(slots=True)
class CriticalHit:
    """一条命中危急值的指标。"""

    name: str
    code: str
    value: float
    unit: str
    threshold_text: str

    def to_line(self) -> str:
        """一行可直接展示的告警文本。"""
        unit_part = f" {self.unit}" if self.unit else ""
        return (
            f"危急值：{self.name} {_fmt_number(self.value)}{unit_part}"
            f"（{self.threshold_text}）——请立即通知接诊医生并复核样本"
        )


def _fmt_number(value: float | int | Decimal | None) -> str:
    """数值格式化：整数值不带小数点（2.0 → "2"，2.8 → "2.8"）。

    检验科习惯把细胞计数写成 50 而不是 50.0；带小数点会让人以为是浮点误差。
    """
    if value is None:
        return "-"
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _to_float(value: Any) -> float | None:
    """把 value/ref_low/ref_high 容错解析成 float。

    混合容错是刚需：LIS 导出既有 JSON number 也有字符串（"2.8"、" 3.5 "），
    还有 Decimal（Numeric 列读出来就是它）。解析不了返回 None，
    由调用方决定「跳过判定但保留原值」，绝不抛异常——
    一份报告里一个脏值不能让整批录入失败。
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _coerce_items(structured: Any) -> list[dict[str, Any]]:
    """从 structured_data 里取出合法的 item 字典列表。

    容忍三种畸形输入：整个对象不是 dict、items 不是 list、list 元素不是 dict。
    这些都是外部录入/历史脏数据的现实形态，静默过滤比抛 500 好——
    原始 raw_content 始终完整保存，结构化只是附赠能力。
    """
    if not isinstance(structured, dict):
        return []
    raw_items = structured.get("items")
    if not isinstance(raw_items, list):
        return []
    return [item for item in raw_items if isinstance(item, dict)]


def evaluate_abnormalities(structured: Any) -> list[AbnormalItem]:
    """规则法异常判定：逐项与参考区间比对。

    **不调用 LLM**，理由见模块 docstring 第 1 条。判定口径：
    - 严格越界才算异常（等于端点不算），与「超出参考范围」的临床表述一致
    - 只给了上限时，超过上限才算高值；只给了下限时同理
    - 两端都没有（或未给）的项无法判定，不出旗标
    - 定性项（value 解析不出数）跳过判定，但不会被丢弃
    """
    results: list[AbnormalItem] = []
    for item in _coerce_items(structured):
        value = _to_float(item.get("value"))
        if value is None:
            continue
        ref_low = _to_float(item.get("ref_low"))
        ref_high = _to_float(item.get("ref_high"))
        code = _normalize_code(item.get("code"))
        name = str(item.get("name") or code or "未命名指标").strip()
        unit = str(item.get("unit") or "").strip()

        if ref_high is not None and value > ref_high:
            results.append(
                AbnormalItem(
                    name=name,
                    code=code,
                    value=value,
                    unit=unit,
                    ref_low=ref_low,
                    ref_high=ref_high,
                    direction="high",
                )
            )
        elif ref_low is not None and value < ref_low:
            results.append(
                AbnormalItem(
                    name=name,
                    code=code,
                    value=value,
                    unit=unit,
                    ref_low=ref_low,
                    ref_high=ref_high,
                    direction="low",
                )
            )
    return results


def detect_critical_values(structured: Any) -> list[CriticalHit]:
    """危急值筛查。命中即返回，供上层写 WARNING 日志与响应提示。

    与 ``evaluate_abnormalities`` 分开实现而不是复用其结果：异常判定看
    「有没有出参考区间」，危急值看「有没有出生命体征红线」，两套阈值本来
    就该独立演进。合并成一个函数会导致将来调整参考区间口径时不小心动到危急值。
    """
    hits: list[CriticalHit] = []
    for item in _coerce_items(structured):
        code = _normalize_code(item.get("code"))
        if not code:
            continue
        rule = next((r for r in CRITICAL_RULES if r["code"] == code), None)
        if rule is None:
            continue
        value = _to_float(item.get("value"))
        if value is None:
            continue
        low = rule.get("low")
        high = rule.get("high")
        name = str(item.get("name") or rule["name"]).strip() or str(rule["name"])
        unit = str(item.get("unit") or rule.get("unit") or "").strip()

        if low is not None and value < float(low):
            hits.append(
                CriticalHit(
                    name=name,
                    code=code,
                    value=value,
                    unit=unit,
                    threshold_text=f"危急下限 {_fmt_number(low)}",
                )
            )
        elif high is not None and value > float(high):
            hits.append(
                CriticalHit(
                    name=name,
                    code=code,
                    value=value,
                    unit=unit,
                    threshold_text=f"危急上限 {_fmt_number(high)}",
                )
            )
    return hits


# ==================== AI 结构化输出 schema ====================


class ReportInterpretation(BaseModel):
    """LLM 报告解读的结构化输出。

    ``need_doctor_review`` 是**医疗红线字段**：默认 True 且校验器拒绝 False。
    AI 的输出永远是「待医生审核的建议」，不允许模型自己宣布「无需复核」。
    之所以还留在 schema 里而不是干脆删掉这个字段，是为了让契约显式存在——
    下游（前端/阶段 5）据此渲染「本报告需医生确认后方可作为诊疗依据」的提示条，
    并且当模型试图输出 false 时我们能拿到一次明确的校验失败记录，
    而不是静默地把越权结论当成合法结果。

    severity 用三档而不是连续分：患者能理解「偏高一点」和「明显异常」的区别，
    理解不了 0.73 是什么。confidence 保留连续值，但那是给统计用的，不给患者看。
    """

    summary: str = Field(
        max_length=120,
        description="面向患者的通俗解读摘要，不超过 120 字",
    )
    severity: str = Field(description="low / medium / high")
    suggestions: list[str] = Field(
        default_factory=list,
        description="后续建议，每条一句话，最多 5 条",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="模型自评置信度 0-1")
    need_doctor_review: bool = Field(default=True)

    @field_validator("severity")
    @classmethod
    def _check_severity(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        if v not in allowed:
            raise ValueError(f"severity 只能是 {'/'.join(sorted(allowed))}")
        return v

    @field_validator("need_doctor_review")
    @classmethod
    def _must_need_review(cls, v: bool) -> bool:
        if v is not True:
            raise ValueError("AI 解读必须标记为需医生复核（need_doctor_review 恒为 true）")
        return True

    @field_validator("summary")
    @classmethod
    def _strip_summary(cls, v: str) -> str:
        return v.strip()

    @field_validator("suggestions")
    @classmethod
    def _cap_suggestions(cls, v: list[str]) -> list[str]:
        cleaned = [s.strip() for s in v if s and s.strip()]
        return cleaned[:5]


INTERPRET_SYSTEM_PROMPT = (
    "你是医院的检查报告解读助手，服务对象是没有任何医学背景的患者。"
    "你的职责是把检验指标讲成人话，帮助患者理解要不要复诊，"
    "而不是给出诊断。严格遵守："
    "1) 不下诊断结论，不使用「确诊」「患有」「可以排除」这类表述；"
    "2) 只依据提供的指标与参考资料说话，资料里没有的病名不要提；"
    "3) 异常项一律引导患者找开单医生复核，不要建议自行用药或停药；"
    "4) 语气中性，不制造恐慌也不淡化风险；明显危险值要明确写出「尽快就医」；"
    "5) 只输出符合要求的 JSON，不要任何解释文字。"
)

INTERPRET_USER_PROMPT_TEMPLATE = """请解读下面这份检查报告，输出面向患者的通俗说明。

【患者基本情况】{patient_profile}

【检查项目】{exam_name}

【异常指标（已由系统规则判定，你不需要重新判断是否异常）】
{abnormal_lines}

【危急值告警（最高优先级，必须在建议中体现「尽快就医」）】
{critical_lines}

【全部指标明细】
{item_lines}

【检验科结论】
{conclusion}

【医学知识参考资料（来自院内知识库，可能为空；为空就只依据上面的指标说话）】
{evidence}

【输出要求】
严格按以下 JSON 结构输出，字段名与取值范围不可更改：
{{"summary": "≤120字的患者可读摘要",
  "severity": "low|medium|high",
  "suggestions": ["建议一", "建议二"],
  "confidence": 0.0-1.0,
  "need_doctor_review": true}}

注意：need_doctor_review 必须为 true，这是系统硬性要求。"""


# ==================== 工作流内部数据结构 ====================


@dataclass(slots=True)
class InterpretOutcome:
    """一次 AI 解读的结果快照（service 出口，路由/任务据此组装响应）。

    ``status`` 与 ExamReport.ai_status 同取值域，两边必须一致，
    否则前端轮询看到的和任务返回的会打架。
    """

    report_id: uuid.UUID
    status: str  # completed / failed / skipped
    summary: str | None = None
    severity: str | None = None
    suggestions: list[str] = field(default_factory=list)
    confidence: float | None = None
    model: str | None = None
    provider: str | None = None
    error: str | None = None
    evidence_count: int = 0
    degraded: bool = False
    latency_ms: int = 0


@dataclass(slots=True)
class DispatchResult:
    """自动派发的结果，供响应如实告知「AI 解读到底排上没有」。"""

    dispatched: bool
    ai_task_id: uuid.UUID | None = None
    celery_task_id: str | None = None
    reason: str | None = None


# ==================== 会话抽象 ====================
#
# 本模块的函数刻意写成「接受 AsyncSession 或 Session」两种形态：
# HTTP 请求走 AsyncSession，Celery worker 走同步 Session（进程模型见
# app/tasks/celery_app.py 模块 docstring）。两侧共用同一套业务实现体，
# 避免出现「worker 里的解读逻辑和 API 里的不一样」这种最难查的分叉。
# 具体查询语句各写一遍（同步/异步 API 本就不同形），但判定、prompt 构造、
# 回写字段这些**业务规则**只有一份。


def _is_async_session(session: Any) -> bool:
    return isinstance(session, AsyncSession)


# 证据检索的**可替换点**：interpret_async 一律经由本模块属性调用，
# 于是 ``monkeypatch.setattr(exam_service, "retrieve_evidence", stub)`` 一定生效。
# 直接写裸名 ``retrieve_evidence(...)`` 在 CPython 里同样查 globals，但语义不显式，
# 后续若有人把它改成局部闭包 / 从别处转调，桩就会静默失效——而契约是
# 「RAG 抖动不得取消一份已录好报告的解读资格」，这条只能靠显式间接层守住。
_SELF = sys.modules[__name__]


def _retrieve_evidence(
    session: AsyncSession | Session, query_text: str, **kwargs: Any
) -> Any:
    """按当前模块属性分派到 retrieve_evidence（保持可打桩）。"""
    return _SELF.retrieve_evidence(session, query_text, **kwargs)


# ==================== 申请单状态流转 ====================


async def _load_request(db: AsyncSession, request_id: uuid.UUID) -> ExamRequest:
    req = await db.get(ExamRequest, request_id)
    if req is None:
        raise ResourceNotFoundError("检查申请单不存在")
    return req


async def receive_request(db: AsyncSession, request_id: uuid.UUID) -> ExamRequest:
    """检验科接收申请单：ordered → received。

    已 received 的申请单再次接收返回幂等结果而不是 409：检验科双人复核时
    两个人都可能点接收，第二个人不该看到一个红色报错以为自己操作失败了。
    其他状态（processing/reported/reviewed）说明单子已经往前走过去了，
    这时改回 received 会造成「正在做的检查凭空退回」，必须硬拒。
    """
    req = await _load_request(db, request_id)
    if req.status == ExamRequestStatus.RECEIVED.value:
        logger.info("申请单 %s 已是 received，幂等返回", req.id)
        return req
    req.status = assert_transition(
        "exam_request", req.status, ExamRequestStatus.RECEIVED.value
    )
    await db.commit()
    await db.refresh(req)
    logger.info("检查申请单已接收 id=%s patient_id=%s", req.id, req.patient_id)
    return req


async def start_processing(db: AsyncSession, request_id: uuid.UUID) -> ExamRequest:
    """开始检查：received → processing，并按转换矩阵允许 ordered → processing 直跳。

    直跳不是我们放宽了规则，而是 ``EXAM_REQUEST_TRANSITIONS`` 本来就允许
    ordered 的两个后继（received / processing）：样本上机后 LIS 会自动把结果
    推过来，中间那次「人工点接收」在流程上是可省略的。矩阵是唯一合法来源，
    这里不额外加限制，也不自己发明 ordered→processing 之外的捷径。
    """
    req = await _load_request(db, request_id)
    if req.status == ExamRequestStatus.PROCESSING.value:
        return req
    req.status = assert_transition(
        "exam_request", req.status, ExamRequestStatus.PROCESSING.value
    )
    await db.commit()
    await db.refresh(req)
    logger.info("检查申请单进入执行 id=%s from=%s", req.id, req.status)
    return req


# ==================== 报告录入 ====================


async def submit_report(
    db: AsyncSession,
    request_id: uuid.UUID,
    *,
    lab_tech_id: uuid.UUID,
    raw_content: str,
    structured_data: dict[str, Any] | None = None,
    conclusion: str | None = None,
    auto_dispatch: bool = True,
) -> tuple[ExamReport, DispatchResult]:
    """录入报告并推进状态：processing → reported（矩阵亦允许 received → reported
    的场景不存在，因此必须先 processing），随后自动派发 AI 解读。

    **事务顺序刻意为「业务数据先提交，派发失败不回滚」**：
    1. 状态机校验 + 建 ExamReport + commit —— 到这里医疗数据已经安全落库
    2. 建 AITask + commit
    3. dispatch_task —— broker 故障时 ``dispatch_task`` 返回 None（它内部吞异常）

    第 3 步失败只把 AITask 标 failed 并记日志，**不回滚报告**。
    理由：AI 解读是辅助功能，检验科录报告是临床必需动作。Redis 抖动一下就让
    报告录不进去，等于把一个可选增强变成了单点故障源。失败的任务可以由
    医生/检验科事后用 POST /exams/reports/{id}/interpret 手动补触发，
    响应里 dispatched=false 与 reason 会如实告诉调用方「没排上，需要手动触发」。

    异常旗标在本函数内由规则算出（``evaluate_abnormalities``），
    不接受调用方传入——如果允许前端传旗标，就等于允许绕过服务端判定，
    危急值可能被客户端一个 bug 悄悄抹掉。

    ``auto_dispatch=False`` 供测试与「批量导入历史报告」场景关闭派发。
    """
    req = await _load_request(db, request_id)

    # 一张申请单只能有一份报告（uq_exam_report_request）。
    # 提前查出来给出干净的错误，而不是等 IntegrityError 冒成 500。
    existing = await db.execute(
        select(ExamReport.id).where(ExamReport.request_id == req.id)
    )
    if existing.scalar_one_or_none() is not None:
        raise InvalidStateTransitionError(
            "该检查申请单已有报告，不能重复录入",
            current=req.status,
            target=ExamRequestStatus.REPORTED.value,
            entity="exam_request",
        )

    req.status = assert_transition(
        "exam_request", req.status, ExamRequestStatus.REPORTED.value
    )

    abnormalities = evaluate_abnormalities(structured_data)
    critical_hits = detect_critical_values(structured_data)
    flags = [item.to_flag() for item in abnormalities]

    report = ExamReport(
        request_id=req.id,
        patient_id=req.patient_id,
        lab_tech_id=lab_tech_id,
        raw_content=raw_content,
        structured_data=structured_data,
        abnormal_flags=flags,
        conclusion=conclusion,
        ai_status=AI_STATUS_PENDING,
    )
    db.add(report)
    await db.commit()
    await db.refresh(report)

    logger.info(
        "检查报告已录入 id=%s request_id=%s 异常项=%d 危急值命中=%d",
        report.id,
        req.id,
        len(flags),
        len(critical_hits),
    )
    if critical_hits:
        # 医疗红线：危急值必须进 WARNING 日志（生产接告警通道），
        # 且不因为「AI 也在解读」而推迟——这条日志此刻就是唯一的通知手段。
        for hit in critical_hits:
            logger.warning(
                "危急值命中 report_id=%s patient_id=%s %s",
                report.id,
                report.patient_id,
                hit.to_line(),
            )

    dispatch = DispatchResult(dispatched=False, reason="未启用自动派发")
    if auto_dispatch:
        dispatch = await dispatch_interpretation(db, report.id, trigger="auto")

    return report, dispatch


# ==================== AI 解读派发 ====================


async def dispatch_interpretation(
    db: AsyncSession,
    report_id: uuid.UUID,
    *,
    trigger: str = "manual",
) -> DispatchResult:
    """建 AITask 并派发解读任务。自动派发与手动重触发共用本函数。

    **先落库再派发**：派发成功但库里没记录的话，前端查不到进度、
    僵尸清理也无从下手；反过来（先落库、派发失败）只多一条 pending 记录，
    损失更小。这与 knowledge_service.start_vectorization 同口径。

    重复派发防护看 ExamReport.ai_status 而不是看「有没有同类 running 任务」：
    解读任务是 per-report 的，全局查重会把两个不同报告的正常并发也拦掉。

    **状态回写的时序是幂等的关键**：入队成功后 ai_status 仍是 pending
    （「已排队、尚无人认领」），``processing`` 只能由 worker 的 CAS 认领写入
    （见 ``claim_for_interpret``）。worker 据此判断「这条消息是不是第一次被
    真正消费」：pending → 认领并开跑；processing/completed → 重复投递，跳过。
    派发方绝不能抢先把状态写成 processing——那会让 worker 把自己发的第一条
    消息认成重复投递，解读永远没人跑。
    """
    report = await _load_report(db, report_id)

    # AI_STATUS_* 本身就是 str 常量（不是 Enum 成员），没有 .value。
    # 写成 AI_STATUS_COMPLETED.value 会在自动派发路径上抛
    # AttributeError: 'str' object has no attribute 'value'，把整条录入接口打挂。
    if report.ai_status == AI_STATUS_COMPLETED:
        return DispatchResult(
            dispatched=False,
            ai_task_id=report.ai_task_id,
            reason="该报告已完成 AI 解读，无需重复触发",
        )
    # 只有 processing 才是「正在跑、别重复提交」。pending 不拦：
    # 派发返回 None（broker 故障）时我们已把 ai_status 标 failed，
    # 而「建了任务但 worker 从没接手」这种 pending 恰恰需要能被重新推一次，
    # 否则单子会永久卡死且没有任何人工补救入口。
    if report.ai_status == AI_STATUS_PROCESSING:
        raise InvalidStateTransitionError(
            "该报告的 AI 解读正在进行中，请等待完成后再试",
            current=report.ai_status,
            target=AI_STATUS_PROCESSING,
            entity="exam_report_ai",
        )

    task = AITask(
        task_type=TASK_TYPE_REPORT_INTERPRET,
        ref_id=report.id,
        status=AITaskStatus.PENDING.value,
        payload={
            "report_id": str(report.id),
            "trigger": trigger,
            "total": 1,
            "done": 0,
        },
        progress=0,
        progress_message="已排队，等待解读 worker 接手",
        idempotency_key=f"{TASK_TYPE_REPORT_INTERPRET}:{report.id.hex}",
    )
    db.add(task)
    # 主键 default=uuid.uuid4 在 **flush 时**才求值，构造时 task.id 是 None。
    # 不先 flush 就把 task.id 赋给 report.ai_task_id，写进去的就是 NULL
    # （阶段2注册建档同款坑）。flush 后 id 才可用，再 commit。
    await db.flush()
    report.ai_task_id = task.id
    report.ai_status = AI_STATUS_PENDING
    await db.commit()
    # expire_on_commit=False（见 core/database 的工厂配置），commit 后 ORM 对象
    # 仍可安全访问属性，不需要 refresh：refresh 会多发一条 SELECT，
    # 而对刚插入的本行没有额外信息量。id 由应用侧 default=uuid.uuid4 生成，
    # 构造时已有值（不依赖数据库返回）。

    # 延迟导入：service 模块会被 Web 进程导入，顶层 import tasks 包
    # 等于让 Web 进程加载整套 Celery 配置（broker 连接参数等），没必要。
    from app.tasks.celery_app import QUEUE_INTERPRET, dispatch_task

    async_result = dispatch_task(
        INTERPRET_TASK_NAME,
        str(report.id),
        str(task.id),
        queue=QUEUE_INTERPRET,
        task_id=str(task.id),
    )

    if async_result is None:
        # broker 不可用：把刚建的任务标 failed，否则它永远占着 pending 位置，
        # 而 ai_status 若仍留 pending 会让前端一直转圈。这里同样标 failed，
        # 于是「重新触发」能走通（failed 不在 ACTIVE_AI_STATUSES 里）。
        task.status = assert_transition(
            "ai_task", task.status, AITaskStatus.FAILED.value
        )
        task.error_message = "任务队列不可用，未能派发；可由医生或检验科手动重新触发解读"
        task.completed_at = utcnow_zoned()
        report.ai_status = AI_STATUS_FAILED
        await db.commit()
        logger.error(
            "报告解读派发失败 report_id=%s ai_task_id=%s（报告数据已落库，不受影响）",
            report.id,
            task.id,
        )
        return DispatchResult(
            dispatched=False,
            ai_task_id=task.id,
            reason="任务队列暂时不可用，报告已录入；请稍后手动触发 AI 解读",
        )

    celery_id = str(getattr(async_result, "id", task.id))[:64]
    task.celery_task_id = celery_id
    task.progress_message = "已入队，等待 worker 接手"
    # AITask 同步推进到 running：与 knowledge_tasks 的入口语义一致
    # （业务侧状态由派发方/worker 共同维护，running 才有 started_at，
    # 僵尸清理 reap_stale_tasks 正是靠 started_at + 超时判断卡死任务）。
    task.status = assert_transition("ai_task", task.status, AITaskStatus.RUNNING.value)
    task.started_at = utcnow_zoned()
    # ai_status 在入队成功后**保持 pending**，绝不提前置 processing。
    # 「processing」是 worker 认领凭证，由 claim_for_interpret 的 CAS
    # （WHERE ai_status IN (pending, failed, skipped)）唯一写入方。
    # 派发方若替 worker 占下 processing，worker 第一次消费就会被自己的
    # 幂等闸门判定为重复投递直接跳过——解读永远没人跑（阶段4容器实测
    # 正是这样卡死的，e2e 里 ai_status 停在 processing、worker 日志
    # "跳过重复消费 reason=in_progress"）。派发成功与否用 AITask 状态
    # 和 celery_task_id 表达，与「谁在算」解耦。
    report.ai_status = AI_STATUS_PENDING
    await db.commit()
    logger.info(
        "报告解读已派发 report_id=%s ai_task_id=%s celery_id=%s trigger=%s",
        report.id,
        task.id,
        celery_id,
        trigger,
    )
    return DispatchResult(
        dispatched=True, ai_task_id=task.id, celery_task_id=celery_id
    )


async def trigger_interpretation(
    db: AsyncSession,
    report_id: uuid.UUID,
    *,
    requester_user_id: uuid.UUID | None = None,
) -> DispatchResult:
    """医生/检验科手动触发（或失败后重试）AI 解读。

    允许的起点 ai_status：pending（派发过但 worker 一直没接手，可重投）、
    failed（重试）、skipped（审核拦截后人工确认要重来）。
    processing → 409（InvalidStateTransitionError），completed → 幂等拒绝。
    """
    result = await dispatch_interpretation(db, report_id, trigger="manual")
    logger.info(
        "手动触发报告解读 report_id=%s user_id=%s dispatched=%s",
        report_id,
        requester_user_id,
        result.dispatched,
    )
    return result


# ==================== 工作台与详情 ====================


async def list_queue(
    db: AsyncSession,
    *,
    status_filter: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[ExamRequest], int]:
    """检验科工作台：待办申请单，按紧急度优先、同级按下单时间先后。

    排序不能用 created_at 单键：STAT 单子是插进来的，时间最新却最该先做。
    紧急度权重放第一键、created_at 升序放第二键（先到先做），
    这样 routine 队列保持 FIFO，urgent/stat 自动插队。

    ``status_filter`` 只能收窄到 received/processing 之内：传 reported 之类
    不属于工作台的狀態时返回空集而不是报错——工作台是待办视图，
    查历史报告走 reports 列表接口。
    """
    if status_filter is not None and status_filter not in LAB_QUEUE_STATUSES:
        # 越界的过滤值代表「一个不属于待办视图的切片」，必须整体为空。
        # 若退回到默认集合，会把「过滤没生效」伪装成「有待办」，
        # 前端在 reported 视图里看到 routine 单子是最难查的错位。
        return [], 0
    allowed = [status_filter] if status_filter else list(LAB_QUEUE_STATUSES)
    conditions = [ExamRequest.status.in_(allowed)]

    stmt = select(ExamRequest).where(*conditions)
    total = int(
        (
            await db.execute(select(func.count()).select_from(stmt.subquery()))
        ).scalar_one()
    )

    rows = list((await db.execute(stmt)).scalars().all())
    # 紧急度权重在 Python 侧排序而不是 SQL CASE：urgency 只有三个取值，
    # 且这一页最多 100 行，写进 SQL 的可读性代价不值。
    # 但总数 total 必须来自 SQL count，否则分页码会错。
    rows.sort(key=lambda r: (_URGENCY_RANK.get(r.urgency, 9), r.created_at, r.id))
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    start = (page - 1) * page_size
    return rows[start : start + page_size], total


async def list_ordered_requests(
    db: AsyncSession,
    *,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[ExamRequest], int]:
    """待接收（ordered）申请单列表，供检验科「接单」视图使用。

    ``list_queue`` 刻意不含 ordered（工作台是「已在手上」的视图，接单前
    的单子混进来会让「接收」动作对已接收单重复出现）。但医生开单后
    单子停在 ordered，检验科没有任何只读入口能发现它们——接收动作就无从
    发起。这里补上这个缺口：排序与 list_queue 同一口径（紧急度优先、
    同级先到先做），危急单在接单视图里同样插队。
    """
    conditions = [ExamRequest.status == ExamRequestStatus.ORDERED.value]
    stmt = select(ExamRequest).where(*conditions)
    total = int(
        (
            await db.execute(select(func.count()).select_from(stmt.subquery()))
        ).scalar_one()
    )
    rows = list((await db.execute(stmt)).scalars().all())
    rows.sort(key=lambda r: (_URGENCY_RANK.get(r.urgency, 9), r.created_at, r.id))
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    start = (page - 1) * page_size
    return rows[start : start + page_size], total


async def _load_report(db: AsyncSession, report_id: uuid.UUID) -> ExamReport:
    report = await db.get(ExamReport, report_id)
    if report is None:
        raise ResourceNotFoundError("检查报告不存在")
    return report


async def get_report_detail(
    db: AsyncSession,
    report_id: uuid.UUID,
    *,
    requester_user_id: uuid.UUID,
    role_name: str,
) -> ExamReport:
    """报告详情。患者只能读自己的报告（越权返回 404 而不是 403）。

    404 而非 403 是刻意选择：403 等于向探测者确认「这个 ID 存在且是别人的报告」，
    而医疗记录的 ID 本身就是敏感信息。与 appointment/triage 的越权口径一致。

    检验科与医生持有 read_all/read，可按矩阵放行全量读取；
    这里不再细分「医生只能读自己开的单子」——医生协作会诊时会跨医生看报告，
    收窄到开单人会挡住正常的多学科场景，那类审计应交给 audit_log 而不是行级过滤。
    """
    report = await _load_report(db, report_id)
    if role_name == "patient":
        own = await _own_patient_id(db, requester_user_id)
        if own is None or report.patient_id != own:
            raise ResourceNotFoundError("检查报告不存在")
    return report


async def _own_patient_id(db: AsyncSession, user_id: uuid.UUID) -> uuid.UUID | None:
    """当前登录用户的患者档案 id（主体口径与挂号/分诊链路一致）。"""
    row = await db.execute(
        select(PatientProfile.id).where(PatientProfile.user_id == user_id)
    )
    return row.scalar_one_or_none()


async def list_reports(
    db: AsyncSession,
    *,
    requester_user_id: uuid.UUID,
    role_name: str,
    patient_id: uuid.UUID | None = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[ExamReport], int]:
    """报告分页列表。患者只看自己的，检验科/医生按矩阵可读全部或指定患者。

    行级隔离在服务层强制：患者传别人的 patient_id 会被**忽略**并按自己的档案过滤
    （返回空集而不是 403——同 get_report_detail 的口径，不向外确认他人记录是否存在）。
    """
    conditions: list[Any] = []
    if role_name == "patient":
        own = await _own_patient_id(db, requester_user_id)
        if own is None:
            return [], 0
        conditions.append(ExamReport.patient_id == own)
    elif patient_id is not None:
        conditions.append(ExamReport.patient_id == patient_id)

    stmt = select(ExamReport).where(*conditions) if conditions else select(ExamReport)
    total = int(
        (await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
    )
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    rows = await db.execute(
        stmt.order_by(ExamReport.created_at.desc(), ExamReport.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(rows.scalars().all()), total


async def attach_request_info(
    db: AsyncSession, reports: Sequence[ExamReport]
) -> dict[uuid.UUID, tuple[str | None, str | None, str | None, str]]:
    """批量补齐报告对应的申请单信息，返回 ``{report_id: (type, name, urgency, status)}``。

    一次 IN 查询而不是逐条 JOIN：报告列表一页最多 100 行，N+1 查询在 PG 上
    会变成 100 次往返，列表接口就是这么一步步变慢的。
    """
    ids = [r.request_id for r in reports]
    if not ids:
        return {}
    rows = await db.execute(
        select(
            ExamRequest.id,
            ExamRequest.exam_type,
            ExamRequest.exam_name,
            ExamRequest.urgency,
            ExamRequest.status,
        ).where(ExamRequest.id.in_(ids))
    )
    by_request = {row[0]: (row[1], row[2], row[3], row[4]) for row in rows}
    return {r.id: by_request.get(r.request_id, (None, None, None, None)) for r in reports}


async def queue_report_map(
    db: AsyncSession, requests: Sequence[ExamRequest]
) -> dict[uuid.UUID, tuple[uuid.UUID, str]]:
    """给工作台行补上「已出报告的 id 与解读状态」。

    没有这一步，检验科在工作台看不到「这份单子报告出了但 AI 失败了」，
    也就找不到手动重触发的入口。
    """
    ids = [r.id for r in requests]
    if not ids:
        return {}
    rows = await db.execute(
        select(ExamReport.request_id, ExamReport.id, ExamReport.ai_status).where(
            ExamReport.request_id.in_(ids)
        )
    )
    return {row[0]: (row[1], row[2]) for row in rows}


# ==================== AI 解读工作流（核心） ====================


def utcnow_zoned() -> datetime:
    """时区感知的当前时间。与 models/base.utcnow 同口径，此处单独成函数便于测试冻结。"""
    return datetime.now(UTC)


def _describe_patient(profile: PatientProfile | None) -> str:
    """把档案压成一行给模型看的患者画像。

    只给年龄性别，**刻意不给姓名/身份证/手机号**：喂给第三方模型的上下文越少越好，
    解读需要的只是「成年女性」这类生理特征。慢病与过敏史对解读有帮助，
    但属于更高敏感级别的数据，留作后续批次按同意策略开关。
    """
    if profile is None:
        return "未提供患者档案（按成人通用参考口径解读）"
    bits: list[str] = []
    age = profile.age
    if age is not None:
        bits.append(f"{age}岁")
    gender_map = {"M": "男", "F": "女"}
    if profile.gender in gender_map:
        bits.append(gender_map[profile.gender])
    chronic = [c for c in (profile.chronic_diseases or []) if c]
    if chronic:
        bits.append(f"既往慢病：{'、'.join(chronic[:5])}")
    if not bits:
        return "档案信息不完整（按成人通用参考口径解读）"
    return "、".join(bits)


def _build_prompt(
    *,
    exam_name: str,
    profile: PatientProfile | None,
    abnormalities: Sequence[AbnormalItem],
    critical_hits: Sequence[CriticalHit],
    items: Sequence[dict[str, Any]],
    conclusion: str | None,
    evidence: Sequence[Any],
) -> str:
    abnormal_lines = (
        "\n".join(f"- {a.to_flag()}" for a in abnormalities) or "（无异常项）"
    )
    critical_lines = (
        "\n".join(f"- {c.to_line()}" for c in critical_hits) or "（未命中危急值）"
    )
    item_lines = "\n".join(
        f"- {item.get('name') or item.get('code') or '未命名'}"
        f" {item.get('value')}{(' ' + str(item.get('unit'))) if item.get('unit') else ''}"
        f"（参考 {item.get('ref_low', '-')}~{item.get('ref_high', '-')}）"
        for item in items[:40]
    ) or "（无结构化指标，仅有原始文本）"

    if evidence:
        ev_blocks = []
        for idx, e in enumerate(evidence, start=1):
            ev_blocks.append(
                f"[{idx}] 《{e.title}》（来源：{e.source}，可信度：{e.source_tier}）\n{e.content[:400]}"
            )
        evidence_text = "\n\n".join(ev_blocks)
    else:
        evidence_text = "（知识库未检索到相关条目）"

    return INTERPRET_USER_PROMPT_TEMPLATE.format(
        patient_profile=_describe_patient(profile),
        exam_name=exam_name,
        abnormal_lines=abnormal_lines,
        critical_lines=critical_lines,
        item_lines=item_lines,
        conclusion=conclusion or "（检验科未填写结论）",
        evidence=evidence_text,
    )


def build_retrieval_query(
    abnormalities: Sequence[AbnormalItem], critical_hits: Sequence[CriticalHit]
) -> str:
    """把异常项拼成检索 query。

    用「指标名 + 方向词」而不是直接把 code 串起来：知识库切块是中文科普正文，
    里面写的是「血钾偏低」而不是 "K low"。危急值命中的项排在前面，
    让它主导召回——最需要解释的正是最危险的那几项。
    """
    parts: list[str] = []
    direction_word = {"low": "偏低", "high": "偏高"}
    for hit in critical_hits:
        parts.append(f"{hit.name} 危急值")
    for ab in abnormalities:
        parts.append(f"{ab.name} {direction_word.get(ab.direction, '异常')}")
    if not parts:
        return "检查报告 结果解读 参考区间"
    return "、".join(dict.fromkeys(parts))


async def retrieve_evidence(
    session: AsyncSession | Session,
    query_text: str,
    *,
    trace_id: str | None = None,
    user_id: uuid.UUID | None = None,
    top_k: int | None = None,
) -> list[Any]:
    """RAG 证据召回。**零结果不算失败**，任何异常也吞掉返回空列表。

    知识库还没向量化、pgvector 不可用、embedding 服务抖动，都不该让一份
    已经录好的报告失去解读资格。证据是加分项：有就引用出处提升可信度，
    没有就只依据指标本身说话（prompt 里明确写了「为空就只看指标」）。
    """
    if not _is_async_session(session):
        # 同步侧（worker）没法在这里 await 异步 retriever，
        # 由 run_interpretation_sync 通过 run_async 桥接进来，故本分支不应到达。
        logger.debug("retrieve_evidence 收到同步会话，跳过证据召回")
        return []
    try:
        response = await RAGRetriever().retrieve(
            session, query_text, trace_id=trace_id, user_id=user_id
        )
    except Exception as exc:  # noqa: BLE001 - 证据失败不反噬解读主路径
        logger.warning("报告解读 RAG 检索失败，按无证据处理：%s", exc)
        return []
    results = list(response.results)
    limit = top_k if top_k is not None else len(results)
    return results[: max(0, limit)]


async def interpret_async(
    db: AsyncSession,
    report_id: uuid.UUID,
    *,
    ai_task_id: uuid.UUID | None = None,
    trace_id: str | None = None,
) -> InterpretOutcome:
    """AI 报告解读工作流（异步侧，FastAPI 与集成测试可用）。

    六步编排，与 worker 侧 ``run_interpretation_sync`` 共享全部业务规则：
    1. 读报告 + 患者档案（年龄性别）
    2. 异常项 → 检索 query → RAG 取 top 证据（零结果继续）
    3. structured_chat + Pydantic 输出模型
    4. 降级透明：is_degraded / 校验失败 → ai_status=failed，报告数据不动
    5. moderation_blocked → ai_status=skipped
    6. 成功回写全部 ai_* 字段 + interpreted_at，推进 reported→reviewed

    **为什么报告数据在任何失败分支都不动**：ai_* 是一组独立列，
    失败时只把它们标 failed 并保留原 raw_content/structured_data/abnormal_flags。
    医生看原始数值就能判读，AI 没了不等于报告没了。
    """
    started = datetime.now(UTC)
    report = await _load_report(db, report_id)
    req = await _load_request(db, report.request_id)
    profile = await db.get(PatientProfile, report.patient_id)

    abnormalities = evaluate_abnormalities(report.structured_data)
    critical_hits = detect_critical_values(report.structured_data)
    items = _coerce_items(report.structured_data)

    trace_id = trace_id or uuid.uuid4().hex
    query = build_retrieval_query(abnormalities, critical_hits)
    # 证据检索是「锦上添花」：pgvector 抖动、embed 配额耗尽都不能取消一份
    # 已录好报告的解读资格——报告数值本身就是给 LLM 的主料，证据只是补充。
    # 契约由 try/except 兑现，而不是只写在 retrieve_evidence 的 docstring 里：
    # 调用方若绕过它直接 patch retrieve_evidence 抛错，这里也必须兜住。
    try:
        evidence = await _retrieve_evidence(
            db, query, trace_id=trace_id, user_id=report.patient_id
        )
    except Exception:  # noqa: BLE001 - 见上，非致命依赖失败降级为空证据
        logger.exception("证据检索异常，本次解读不带知识库证据 report_id=%s", report.id)
        evidence = []

    prompt = _build_prompt(
        exam_name=req.exam_name,
        profile=profile,
        abnormalities=abnormalities,
        critical_hits=critical_hits,
        items=items,
        conclusion=report.conclusion,
        evidence=evidence,
    )

    request = LLMRequest(
        scenario=LLMScenario.REPORT_INTERPRET,
        messages=[
            ChatMessage(role="system", content=INTERPRET_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        ],
        trace_id=trace_id,
        workflow_node="report_interpret",
        ai_task_id=ai_task_id,
    )

    try:
        parsed, response = await get_llm_service().structured_chat(
            request, ReportInterpretation
        )
    except Exception as exc:  # noqa: BLE001 - 解读失败不得冒泡成 500
        logger.exception("报告解读 LLM 调用异常 report_id=%s", report.id)
        outcome = InterpretOutcome(
            report_id=report.id,
            status=AI_STATUS_FAILED,
            error=f"解读调用异常：{exc}"[:500],
            evidence_count=len(evidence),
            latency_ms=int((datetime.now(UTC) - started).total_seconds() * 1000),
        )
        await _write_back_async(db, report, outcome, reviewed=False)
        return outcome

    latency_ms = int((datetime.now(UTC) - started).total_seconds() * 1000)

    # ---------- 审核拦截：skipped ----------
    if getattr(response, "moderation_blocked", False):
        outcome = InterpretOutcome(
            report_id=report.id,
            status=AI_STATUS_SKIPPED,
            error="解读内容未通过安全审核，已跳过",
            model=getattr(response, "model", None),
            provider=getattr(response, "provider", None),
            evidence_count=len(evidence),
            degraded=True,
            latency_ms=latency_ms,
        )
        await _write_back_async(db, report, outcome, reviewed=False)
        logger.warning("报告解读被审核拦截 report_id=%s", report.id)
        return outcome

    # ---------- 降级 / 校验失败：failed ----------
    if parsed is None:
        outcome = InterpretOutcome(
            report_id=report.id,
            status=AI_STATUS_FAILED,
            error=(
                f"结构化输出校验失败（error_type={getattr(response, 'error_type', None)}）"
            ),
            model=getattr(response, "model", None),
            provider=getattr(response, "provider", None),
            evidence_count=len(evidence),
            degraded=True,
            latency_ms=latency_ms,
        )
        await _write_back_async(db, report, outcome, reviewed=False)
        return outcome

    if _is_degraded_response(response):
        outcome = InterpretOutcome(
            report_id=report.id,
            status=AI_STATUS_FAILED,
            error=(
                "AI 服务处于降级模式，未采用其解读结果"
                f"（degradation_level={getattr(getattr(response, 'degradation_level', None), 'value', None)}）"
            ),
            model=getattr(response, "model", None),
            provider=getattr(response, "provider", None),
            evidence_count=len(evidence),
            degraded=True,
            latency_ms=latency_ms,
        )
        await _write_back_async(db, report, outcome, reviewed=False)
        logger.warning(
            "报告解读处于降级模式，标记 failed 而不采用 report_id=%s level=%s",
            report.id,
            getattr(response, "degradation_level", None),
        )
        return outcome

    # ---------- 成功 ----------
    try:
        confidence = float(parsed.confidence)
    except (TypeError, ValueError):
        confidence = None

    outcome = InterpretOutcome(
        report_id=report.id,
        status=AI_STATUS_COMPLETED,
        summary=parsed.summary,
        severity=parsed.severity,
        suggestions=list(parsed.suggestions),
        confidence=confidence,
        model=getattr(response, "model", None),
        provider=getattr(response, "provider", None),
        evidence_count=len(evidence),
        latency_ms=latency_ms,
    )
    await _write_back_async(db, report, outcome, reviewed=True)
    logger.info(
        "报告解读完成 report_id=%s severity=%s confidence=%s evidence=%d",
        report.id,
        parsed.severity,
        confidence,
        len(evidence),
    )
    return outcome


def _is_degraded_response(response: Any) -> bool:
    """是否处于降级模式。

    除了 ``is_degraded``（degradation_level != PRIMARY），还要看 status 字段：
    chat 在预算耗尽时返回 status="budget_exceeded" 而 degradation_level
    仍是 CANNED_RESPONSE，两条路径都要拦住——拿兜底话术当解读结果写进病历，
    比没有解读更糟。
    """
    if getattr(response, "is_degraded", False):
        return True
    return getattr(response, "status", None) in ("degraded", "budget_exceeded", "blocked")


async def _write_back_async(
    db: AsyncSession,
    report: ExamReport,
    outcome: InterpretOutcome,
    *,
    reviewed: bool,
) -> None:
    """异步侧回写。字段口径与 ``_write_back_sync`` 必须完全一致。"""
    _apply_ai_fields(report, outcome)
    if reviewed:
        req = await _load_request(db, report.request_id)
        if req.status == ExamRequestStatus.REPORTED.value:
            req.status = assert_transition(
                "exam_request", req.status, ExamRequestStatus.REVIEWED.value
            )
    await db.commit()


def _apply_ai_fields(report: ExamReport, outcome: InterpretOutcome) -> None:
    """把解读结果写进 ExamReport 的 ai_* 列（两侧共用，唯一的字段映射点）。"""
    report.ai_status = outcome.status
    if outcome.status == AI_STATUS_COMPLETED:
        report.ai_interpretation = outcome.summary
        report.ai_summary = outcome.summary
        report.ai_suggestions = list(outcome.suggestions)
        report.ai_confidence = (
            Decimal(str(round(outcome.confidence, 3)))
            if outcome.confidence is not None
            else None
        )
        report.ai_model = outcome.model
        report.ai_provider = outcome.provider
        report.interpreted_at = utcnow_zoned()
    else:
        # 失败/跳过：清掉可能残留的旧解读文本，避免「ai_status=failed 但
        # ai_summary 还写着上一轮结论」这种自相矛盾的读数。
        # 原始数据（raw_content/structured_data/abnormal_flags/conclusion）
        # 一个字都不碰——那是医疗事实，与 AI 成败无关。
        report.ai_summary = None
        report.ai_interpretation = None
        report.ai_suggestions = []
        report.ai_confidence = None
        report.ai_model = outcome.model
        report.ai_provider = outcome.provider
        report.interpreted_at = None


def interpret_report_payload(outcome: InterpretOutcome) -> dict[str, Any]:
    """解读结果 → 可落 AITask.result 的统计字典（worker 与测试共用）。"""
    return {
        "report_id": str(outcome.report_id),
        "status": outcome.status,
        "severity": outcome.severity,
        "confidence": outcome.confidence,
        "suggestion_count": len(outcome.suggestions),
        "evidence_count": outcome.evidence_count,
        "model": outcome.model,
        "provider": outcome.provider,
        "latency_ms": outcome.latency_ms,
        "degraded": outcome.degraded,
        "error": outcome.error,
    }


# ==================== worker 侧（同步 Session）====================


def _load_report_sync(session: Session, report_id: uuid.UUID) -> ExamReport | None:
    return session.get(ExamReport, report_id)


def _load_request_sync(session: Session, request_id: uuid.UUID) -> ExamRequest | None:
    return session.get(ExamRequest, request_id)


@dataclass(slots=True)
class ClaimResult:
    """worker 认领结果。"""

    claimed: bool
    reason: str
    ai_status: str | None = None


def claim_for_interpret(session: Session, report_id: uuid.UUID) -> ClaimResult:
    """worker 侧幂等闸门：把报告从「排队中」认领为「解读中」。

    ``acks_late=True`` 意味着消息可能在 worker 崩溃后被重新投递，同一个
    report_id 会被投给两次。不加这道闸的话第二次会再调一遍 LLM——多花一次配额是小事，
    两次回写交错覆盖 ai_* 字段（第一次的 summary 配第二次的 confidence）才是脏数据。

    判定口径（看 ExamReport.ai_status，不看 AITask.status）：
    - completed → 已出结果，跳过（不重复烧配额，也不覆盖已有解读）
    - processing → 另一个副本正在跑（或上一轮崩在中间），跳过并留人工重触发；
      自动接管会让两个进程同时写同一行
    - pending / failed / skipped → 允许开跑：pending 是首次消费，
      failed/skipped 是用户手动重试派发的新消息
    - 报告不存在 → 跳过（数据被删了不该让任务无限重试）

    认领动作是一条带状态谓词的 UPDATE（CAS）而不是「先 SELECT 再改」：
    后者在两个副本同时读到 pending 时会双双通过检查，两条都去调 LLM。
    ``WHERE ai_status IN (...)`` 让数据库在同一语句里完成判与改，
    rowcount 就是「我抢到了没有」的唯一可信答案——与 appointment_service
    扣号源用的是同一个手法。SQLite 也支持条件 UPDATE，因此这条语义在单测里能真跑到。
    """
    target = session.get(ExamReport, report_id)
    if target is None:
        return ClaimResult(claimed=False, reason="not_found", ai_status=None)
    current = target.ai_status

    if current == AI_STATUS_COMPLETED:
        return ClaimResult(claimed=False, reason="already_completed", ai_status=current)
    if current == AI_STATUS_PROCESSING:
        return ClaimResult(claimed=False, reason="in_progress", ai_status=current)

    result = session.execute(
        update(ExamReport)
        .where(
            ExamReport.id == report_id,
            ExamReport.ai_status.in_(
                (AI_STATUS_PENDING, AI_STATUS_FAILED, AI_STATUS_SKIPPED)
            ),
        )
        .values(ai_status=AI_STATUS_PROCESSING)
    )
    session.commit()
    if int(getattr(result, "rowcount", 0) or 0) == 0:
        # 被别人抢先一步推进了状态，本次消息按重复投递处理
        session.refresh(target)
        return ClaimResult(
            claimed=False, reason="lost_race", ai_status=target.ai_status
        )
    return ClaimResult(claimed=True, reason="claimed", ai_status=AI_STATUS_PROCESSING)


def run_interpretation_sync(
    session: Session,
    report_id: uuid.UUID,
    *,
    ai_task_id: uuid.UUID | None = None,
    trace_id: str | None = None,
    llm_service: Any | None = None,
    evidence_fn: Callable[[str, str | None], Sequence[Any]] | None = None,
) -> InterpretOutcome:
    """AI 报告解读工作流（同步侧，Celery worker 用）。

    与 ``interpret_async`` 是同一套业务规则的两种会话形态，判定/prompt/回写字段
    全部复用同一个纯函数（``evaluate_abnormalities`` / ``_build_prompt`` /
    ``_apply_ai_fields``），只有 DB 读写语句各写一遍。

    可注入参数是为可测性留的口子，生产路径全部为 None 走全局单例：
    - ``llm_service``：真 LLM 要网络，单测必须能塞计数桩
    - ``evidence_fn``：真检索要 embed，单测必须能打桩
      （签名 ``(query_text, trace_id) -> 证据序列``，同步侧由外层 run_async 桥接）

    **为什么同步侧不自己 asyncio.run**：worker 的事件循环生命周期由
    ``celery_app.run_async`` 统一管理（它会在新循环里跑完协程再显式关闭
    adapter 客户端，避免 finalizer 落在已关闭的循环上刷出成片
    "Event loop is closed"）。任务体里再套一层 asyncio.run 会撞上
    「嵌套事件循环」RuntimeError，也会绕过那套收尾逻辑。
    """
    started = utcnow_zoned()
    # 契约前置：本函数只应在无活循环的线程（Celery prefork 子进程）里跑。
    # 必须在 try/except 外面探——若让下面 except Exception 接住这个
    # RuntimeError，调用方会收到一个静默的 failed outcome，把「函数调错了」
    # 伪装成「AI 服务故障」，掩盖真正的接线问题。
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "run_interpretation_sync 处在已运行的事件循环中，应改调 exam_service.interpret_async"
        )
    report = _load_report_sync(session, report_id)
    if report is None:
        raise ResourceNotFoundError("检查报告不存在")
    req = _load_request_sync(session, report.request_id)
    if req is None:
        raise ResourceNotFoundError("检查申请单不存在")
    profile = session.get(PatientProfile, report.patient_id)

    abnormalities = evaluate_abnormalities(report.structured_data)
    critical_hits = detect_critical_values(report.structured_data)
    items = _coerce_items(report.structured_data)

    trace_id = trace_id or uuid.uuid4().hex

    # ---------- 工作流追踪：入口 ----------
    # **只加追踪，不动 CAS 认领**：claim_for_interpret 的 pending→processing 幂等语义
    # 由调用方（Celery 任务体）在进入本函数之前完成，本函数不碰 ai_status 的判定；
    # 追踪记录写在独立会话上，不参与业务事务，因此也不会影响 rowcount 的可信度。
    # start_workflow_trace 是异步函数且内部自开 async_session_factory：此处已确认
    # 无活循环（上面的契约前置），用 asyncio.run 桥接一次，符合项目 worker 侧约定。
    trace_row_id: int | None = None
    try:
        trace_row_id = asyncio.run(
            start_workflow_trace(
                trace_id,
                WORKFLOW_NAME_REPORT_INTERPRET,
                user_id=report.patient_id,
                ref_id=report.id,
                node_count=REPORT_INTERPRET_NODE_COUNT,
            )
        )
    except Exception:  # noqa: BLE001 - 追踪失败不得影响解读
        logger.warning("创建报告解读工作流追踪失败", exc_info=True)
    wf_steps: list[dict[str, Any]] = []
    wf_start = time.perf_counter()

    step_start = time.perf_counter()
    abnormalities = evaluate_abnormalities(report.structured_data)
    critical_hits = detect_critical_values(report.structured_data)
    items = _coerce_items(report.structured_data)
    wf_steps.append(
        build_step(
            "rule_evaluation",
            "rule",
            latency_ms=int((time.perf_counter() - step_start) * 1000),
            output_snapshot={
                "abnormal_count": len(abnormalities),
                "critical_count": len(critical_hits),
                "item_count": len(items),
                "critical_codes": [h.code for h in critical_hits[:5]],
            },
        )
    )

    query = build_retrieval_query(abnormalities, critical_hits)
    step_start = time.perf_counter()
    if evidence_fn is not None:
        evidence = list(evidence_fn(query, trace_id) or [])
    else:
        evidence = []
        logger.info("worker 侧未注入证据检索函数，本次解读不带知识库证据")
    wf_steps.append(
        build_step(
            "evidence_retrieval",
            "retrieval",
            latency_ms=int((time.perf_counter() - step_start) * 1000),
            input_snapshot=query,
            output_snapshot={"evidence_count": len(evidence)},
        )
    )

    step_start = time.perf_counter()
    prompt = _build_prompt(
        exam_name=req.exam_name,
        profile=profile,
        abnormalities=abnormalities,
        critical_hits=critical_hits,
        items=items,
        conclusion=report.conclusion,
        evidence=evidence,
    )
    request = LLMRequest(
        scenario=LLMScenario.REPORT_INTERPRET,
        messages=[
            ChatMessage(role="system", content=INTERPRET_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        ],
        trace_id=trace_id,
        workflow_node="report_interpret",
        ai_task_id=ai_task_id,
    )

    service = llm_service if llm_service is not None else get_llm_service()

    step_start = time.perf_counter()
    try:
        parsed, response = _call_structured_chat(service, request)
    except Exception as exc:  # noqa: BLE001
        logger.exception("worker 报告解读 LLM 调用异常 report_id=%s", report.id)
        outcome = InterpretOutcome(
            report_id=report.id,
            status=AI_STATUS_FAILED,
            error=f"解读调用异常：{exc}"[:500],
            evidence_count=len(evidence),
            latency_ms=int((utcnow_zoned() - started).total_seconds() * 1000),
        )
        _apply_ai_fields(report, outcome)
        session.commit()
        wf_steps.append(
            build_step(
                "llm_interpret",
                "llm",
                latency_ms=int((time.perf_counter() - step_start) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        # 异常出口同样要收尾，否则这条 trace 永远停在 running，看板上表现为
        # 「无限多的进行中工作流」——这是最难发现的监控失真。
        _finish_report_trace_sync(
            trace_row_id,
            trace_id,
            wf_steps,
            wf_start,
            status=AI_STATUS_FAILED,
            branch_taken="llm_exception",
            outcome=outcome,
            responses=[],
            error_message=outcome.error,
        )
        return outcome

    latency_ms = int((utcnow_zoned() - started).total_seconds() * 1000)

    wf_steps.append(
        build_step(
            "llm_interpret",
            "llm",
            latency_ms=int((time.perf_counter() - step_start) * 1000),
            input_snapshot={"prompt_chars": len(prompt), "evidence_count": len(evidence)},
            output_snapshot={
                "model": getattr(response, "model", None),
                "provider": getattr(response, "provider", None),
                "parsed_ok": parsed is not None,
                "moderation_blocked": bool(getattr(response, "moderation_blocked", False)),
            },
        )
    )

    if getattr(response, "moderation_blocked", False):
        outcome = InterpretOutcome(
            report_id=report.id,
            status=AI_STATUS_SKIPPED,
            error="解读内容未通过安全审核，已跳过",
            model=getattr(response, "model", None),
            provider=getattr(response, "provider", None),
            evidence_count=len(evidence),
            degraded=True,
            latency_ms=latency_ms,
        )
    elif parsed is None:
        outcome = InterpretOutcome(
            report_id=report.id,
            status=AI_STATUS_FAILED,
            error=(
                "结构化输出校验失败"
                f"（error_type={getattr(response, 'error_type', None)}）"
            ),
            model=getattr(response, "model", None),
            provider=getattr(response, "provider", None),
            evidence_count=len(evidence),
            degraded=True,
            latency_ms=latency_ms,
        )
    elif _is_degraded_response(response):
        outcome = InterpretOutcome(
            report_id=report.id,
            status=AI_STATUS_FAILED,
            error=(
                "AI 服务处于降级模式，未采用其解读结果"
                f"（degradation_level={getattr(getattr(response, 'degradation_level', None), 'value', None)}）"
            ),
            model=getattr(response, "model", None),
            provider=getattr(response, "provider", None),
            evidence_count=len(evidence),
            degraded=True,
            latency_ms=latency_ms,
        )
    else:
        try:
            confidence: float | None = float(parsed.confidence)
        except (TypeError, ValueError):
            confidence = None
        outcome = InterpretOutcome(
            report_id=report.id,
            status=AI_STATUS_COMPLETED,
            summary=parsed.summary,
            severity=parsed.severity,
            suggestions=list(parsed.suggestions),
            confidence=confidence,
            model=getattr(response, "model", None),
            provider=getattr(response, "provider", None),
            evidence_count=len(evidence),
            latency_ms=latency_ms,
        )

    _apply_ai_fields(report, outcome)
    if outcome.status == AI_STATUS_COMPLETED and req.status == ExamRequestStatus.REPORTED.value:
        req.status = assert_transition(
            "exam_request", req.status, ExamRequestStatus.REVIEWED.value
        )
    session.commit()

    wf_steps.append(
        build_step(
            "write_back",
            "db_write",
            latency_ms=0,
            output_snapshot={
                "ai_status": outcome.status,
                "request_status": req.status,
                "severity": outcome.severity,
            },
        )
    )
    # 工作流追踪：出口。status 直接沿用 ai_status（completed/failed/skipped 三态
    # 与 workflow_traces.status 语义天然对齐），branch_taken 记录走了哪个判定分支，
    # 让「审核拦截 / 校验失败 / 降级 / 正常」四种结局在看板上可分别统计。
    _finish_report_trace_sync(
        trace_row_id,
        trace_id,
        wf_steps,
        wf_start,
        status=outcome.status,
        branch_taken=_interpret_branch(outcome, response),
        outcome=outcome,
        responses=[response],
        error_message=outcome.error,
    )

    logger.info(
        "worker 报告解读结束 report_id=%s status=%s evidence=%d",
        report.id,
        outcome.status,
        len(evidence),
    )
    return outcome


def _interpret_branch(outcome: InterpretOutcome, response: Any) -> str:
    """把解读结局归一成 branch_taken 标识（String(40) 内）。"""
    if getattr(response, "moderation_blocked", False):
        return "moderation_blocked"
    if outcome.status == AI_STATUS_COMPLETED:
        return "interpreted"
    if outcome.status == AI_STATUS_SKIPPED:
        return "skipped"
    return "failed"


def _finish_report_trace_sync(
    trace_row_id: int | None,
    trace_id: str,
    steps: list[dict[str, Any]],
    wf_start: float,
    *,
    status: str,
    branch_taken: str,
    outcome: InterpretOutcome,
    responses: Sequence[Any],
    error_message: str | None,
) -> None:
    """报告解读工作流的收尾写入（同步封装 + 外层再兜一层异常）。

    **绝不破坏 CAS 幂等**：本函数只写 workflow_traces，不读也不改 ExamReport.ai_status，
    且跑在 observability 自己的独立会话上——阶段 4 那套 pending→processing→终态的
    rowcount 判定完全不受影响。定位优先用 start 返回的主键 id，拿不到时退化为
    trace_id（重复投递会留下多条 running，收尾语义是「写到最新一条」）。
    """
    try:
        total_tokens, total_cost_usd = sum_usage(responses)
        finish_workflow_trace_sync(
            trace_row_id if trace_row_id is not None else trace_id,
            status=status,
            steps=steps,
            total_latency_ms=int((time.perf_counter() - wf_start) * 1000),
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd if total_cost_usd else None,
            branch_taken=branch_taken,
            final_output={
                "report_id": str(outcome.report_id),
                "ai_status": outcome.status,
                "severity": outcome.severity,
                "confidence": outcome.confidence,
                "evidence_count": outcome.evidence_count,
                "model": outcome.model,
                "latency_ms": outcome.latency_ms,
            },
            error_message=error_message,
        )
    except Exception:  # noqa: BLE001 - 追踪失败不得影响解读结果返回
        logger.warning("收尾报告解读工作流追踪失败", exc_info=True)


def _call_structured_chat(
    service: Any, request: LLMRequest
) -> tuple[ReportInterpretation | None, Any]:
    """在同步上下文里驱动一次 structured_chat。

    **为什么不能直接 await**：``LLMService.structured_chat`` 是协程，而本函数处在
    Celery 的同步 Session 世界里。模块边界对此有明确约定（见 celery_app.run_async
    docstring）：客户端生命周期与事件循环绑定，收尾必须在同一个活循环内完成，
    所以**任务体内禁止自己再套 asyncio.run**——嵌套会抛
    ``RuntimeError: cannot be called from a running event loop``，
    还会绕过 run_async 里那段 aclose_adapter_clients 收尾，
    实测后果是 worker 日志被成片 "Event loop is closed" 刷脏。

    处理策略按「是否已有活循环」分支：
    - 无运行循环（Celery prefork 子进程的常态）：用 asyncio.run 驱动这一次调用，
      与 knowledge_service.embed_texts 同一口径（那里也是同步侧收口 embedding）。
    - 已有运行循环（说明调用方本该走 ``interpret_async``）：立刻抛错而不是想办法绕，
      绕法要么阻塞整个 loop，要么静默丢结果——两者都比显式失败危险。
    """
    coro_factory: Callable[[], Any] | None = None
    hook = getattr(service, "structured_chat", None)
    if callable(hook):
        coro_factory = lambda: hook(request, ReportInterpretation)  # noqa: E731
    if coro_factory is None:
        raise TypeError(
            f"{type(service).__name__} 未提供 structured_chat，无法执行报告解读"
        )

    import asyncio

    coro = coro_factory()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        "run_interpretation_sync 处在已运行的事件循环中，应改调 exam_service.interpret_async"
    )


__all__ = [
    "AI_STATUS_COMPLETED",
    "AI_STATUS_FAILED",
    "AI_STATUS_PENDING",
    "AI_STATUS_PROCESSING",
    "AI_STATUS_SKIPPED",
    "ACTIVE_AI_STATUSES",
    "CRITICAL_RULES",
    "CriticalHit",
    "AbnormalItem",
    "DispatchResult",
    "INTERPRET_TASK_NAME",
    "LAB_QUEUE_STATUSES",
    "ReportInterpretation",
    "InterpretOutcome",
    "STRUCTURED_DATA_CONTRACT",
    "TASK_TYPE_REPORT_INTERPRET",
    "attach_request_info",
    "build_retrieval_query",
    "claim_for_interpret",
    "ClaimResult",
    "detect_critical_values",
    "dispatch_interpretation",
    "evaluate_abnormalities",
    "get_report_detail",
    "interpret_async",
    "interpret_report_payload",
    "list_queue",
    "list_reports",
    "queue_report_map",
    "receive_request",
    "run_interpretation_sync",
    "start_processing",
    "submit_report",
    "trigger_interpretation",
    "utcnow_zoned",
]
