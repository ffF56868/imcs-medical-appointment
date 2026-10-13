"""处方域业务逻辑（阶段 5 处方侧）。

链路位置：``visit(in_progress/documented) → prescription(draft)
→ confirmed → paid(收费侧) → dispensed(发药侧)``。
本模块只负责 **draft / confirmed / cancelled** 三段，
paid 与 dispensed 属收费域和发药域，这里刻意不写入。

============================ 为什么安全校验是「纯规则」而不是 AI ============================

与 ``exam_service`` 的危急值判定同一个理由：**医疗安全相关的判断必须可复算、
可审计、断网也能生效**。模型幻觉或供应商降级都可能让「对青霉素过敏的人被开了
阿莫西林」被说成「无冲突」。过敏史冲突是明确的致敏风险，重复用药是明确的剂量叠加，
两者都能用一张规则表判出来，没有理由把它们交给一次不确定的推理。

规则表是**演示级**的（见下方三张表的声明）：命中口径为名称前缀匹配，
覆盖的是种子数据里那几味药，不是完整药品字典。落地生产必须由药学部逐条核定
并把表迁到配置或数据库字典表，让有资质的人改而不用发版。

============================ override 语义：医师处方权优先 ============================

阻断项（level=block）存在时默认拒绝确认开具，返回 422 +
``prescription_allergy_conflict`` / ``prescription_drug_interaction``。
但**规则引擎不能取代医师处方权**——这是医疗业务的既有秩序：

- 过敏史可能是陈旧或错误的记录（患者早年自述，实际已脱敏）；
- 某些场景下获益明确大于风险（感染科指南推荐的替代方案本就跨类）；
- 药师审核环节（dispensing:approve）才是这道风险的最终把关人。

因此请求携带非空 ``safety_override_reason`` 时允许医生带理由强行开具，
理由原样落 ``safety_override_reason`` 列。**留痕不是形式主义**：没有这一列，
「医生复核后主动放行」与「规则根本没跑」在数据上完全不可区分，
事后追责与质控抽样都失去依据。warning 级（剂量超限）不需要 override，
它本来就只提示不拦。

============================ 金额口径：整数分 ============================

``prescriptions.total_amount`` 是 Integer，单位**分**（见 models/pharmacy.py 列注释）。
计算复用 ORM 自带的 ``Prescription.recalculate_total()``：明细 ``unit_price`` 存的是元
（Numeric(10,2)），逐项 ``Decimal(单价) × Decimal(数量)`` 得元，乘 100 后用
ROUND_HALF_UP 量化到整分再求和。走 Decimal 而不是 float 的理由写在模型注释里：
``0.1 + 0.2 != 0.3`` 这类误差在收费场景不可接受。schema 层不做除以 100 的换算，
格式化展示交给前端。

============================ 与其它域的边界 ============================

- **不依赖药品主数据表**：``drug_code`` / ``drug_name`` 由医生提交，本域不回查字典。
  旧版 ``app/legacy/models.Drug`` 属 legacy 包，按约定不 import。
- **不碰收费与发药**：不写 ``charge_orders``，也不建 ``dispensing_records``。
  paid/dispensed 由对应域推进，本模块只在读取时如实投影当前状态。
- **不改状态机与权限矩阵**：迁移一律走 ``state_machine.assert_transition``，
  授权位一律取 ``rbac_matrix`` 既有定义。
"""

from __future__ import annotations

import logging
import re
import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    AllergyConflictError,
    DrugInteractionError,
    InvalidStateTransitionError,
    ResourceNotFoundError,
)
from app.models.auth import RoleName, User
from app.models.base import utcnow
from app.models.clinical import Visit, VisitStatus
from app.models.patient import PatientProfile
from app.models.pharmacy import Prescription, PrescriptionItem, PrescriptionStatus
from app.schemas.prescription import (
    PrescriptionConfirmRequest,
    PrescriptionCreateRequest,
    PrescriptionUpdateRequest,
)
from app.services.state_machine import allowed_targets, assert_transition, validate_status_value

logger = logging.getLogger(__name__)


# ==================== 常量 ====================

# 可开处方的 visit 状态白名单。与 consult_service.OPENABLE_VISIT_STATUSES 同口径：
# closed 不在其中——结案后再开药属于「另起一次就诊」，否则处方会挂到一段
# 已归档病历下，质控抽样时会被当成漏诊处理。
OPENABLE_VISIT_STATUSES: tuple[str, ...] = (
    VisitStatus.IN_PROGRESS.value,
    VisitStatus.DOCUMENTED.value,
)

# 实体名，状态机与错误 detail 里都用它，避免各处硬编码字符串漂移。
ENTITY = "prescription"

# 单日剂量校验用到的频次换算。表外频次（如「必要时」「每周一次」）不参与
# 剂量判定 —— 宁可漏报一条 warning，也不能把 q_wan 猜成每日 1 次后误报。
FREQUENCY_PER_DAY: dict[str, int] = {
    "qd": 1,
    "bid": 2,
    "tid": 3,
    "qid": 4,
}

# ==================== 演示级规则表 ====================
#
# **这三张表都是演示口径，不得作为真实临床决策依据。**
# 数值与命中范围取自常见门诊用药的量级，仅用于跑通「开药 → 拦截 → 带理由放行」
# 这条工程闭环。落地生产须由药学部按本院处方审核规则库（通常几千条）逐条替换。

# a) 药物 → 过敏类别。键是**药名通用名前缀**（小写比较），值是过敏原类别名。
#    之所以按前缀而不是精确匹配：剂型后缀千变万化
#    （阿莫西林胶囊 / 阿莫西林干混悬剂 / 阿莫西林克拉维酸钾片），
#    精确匹配会让同一味药换个剂型就绕过过敏拦截——那是最危险的漏报方向。
#    命中方式反过来也保守：只在前缀层面匹配，不做包含匹配，
#    避免「克林霉素」里含「林可」这类字面巧合引发误报。
ALLERGY_CLASS_MAP: dict[str, str] = {
    "阿莫西林": "青霉素类",
    "氨苄西林": "青霉素类",
    "哌拉西林": "青霉素类",
    "青霉素v": "青霉素类",
    "复方新诺明": "磺胺类",
    "磺胺嘧啶": "磺胺类",
    "磺胺甲噁唑": "磺胺类",
    "头孢呋辛": "头孢菌素类",
    "苯妥英": "乙内酰脲类",
}

# 过敏原别名 → 档案里的标准写法。患者档案常写成「青霉素」「盘尼西林」「磺胺药」，
# 规则表统一折叠到标准类别名后才比对，否则同一过敏原有三种记法就得配三条规则。
ALLERGY_ALIAS_MAP: dict[str, str] = {
    "盘尼西林": "青霉素类",
    "青霉素类": "青霉素类",
    "青霉素": "青霉素类",
    "磺胺类": "磺胺类",
    "磺胺药": "磺胺类",
    "磺胺": "磺胺类",
    "头孢类": "头孢菌素类",
    "头孢菌素": "头孢菌素类",
    "乙内酰脲": "乙内酰脲类",
    "乙内酰脲类": "乙内酰脲类",
    "苯妥英": "乙内酰脲类",
    "苯妥英钠": "乙内酰脲类",
}

# drug_code 维度也参与过敏判定（编码稳定、名称易变），前缀同上口径。
ALLERGY_CODE_PREFIX_MAP: dict[str, str] = {
    "AMOX": "青霉素类",
    "AMPX": "青霉素类",
    "SMEA": "磺胺类",
    "SADZ": "磺胺类",
}

# b) 每日最大剂量（mg）。键同样是通用名前缀。
#    只有落在表内的药才做剂量校验；单次剂量从 ``dosage`` 文本里解析 mg/g。
DAILY_MAX_DOSE_MG: dict[str, Decimal] = {
    "阿莫西林": Decimal("4000"),  # 口服上限 4g/日
    "氨苄西林": Decimal("6000"),
    "对乙酰氨基酚": Decimal("2000"),  # 肝毒性约束，国内说明书普遍收紧到 2g/日
    "复方新诺明": Decimal("2000"),  # 以 SMZ 计
    "头孢呋辛": Decimal("3000"),
}

# c) 剂型后缀。归一化时逐个剥离，用于「同通用名不同厂牌」的重复用药检测。
#     顺序敏感：先剥长的（「缓释胶囊」要在「胶囊」之前剥掉），否则会留下「缓释」。
DOSAGE_FORM_SUFFIXES: tuple[str, ...] = (
    "缓释胶囊",
    "缓释片",
    "肠溶胶囊",
    "肠溶片",
    "分散片",
    "咀嚼片",
    "泡腾片",
    "含片",
    "胶囊",
    "片",
    "颗粒",
    "干混悬剂",
    "混悬液",
    "口服液",
    "注射液",
    "注射用",
    "软膏",
    "乳膏",
    "滴眼液",
    "喷雾剂",
    "气雾剂",
    "栓",
)

# 厂牌/商品名前缀标记：形如「阿莫西林胶囊(0.25g*24片)」的括号说明整体剥离。
_PARENTHESES_RE = re.compile(r"[（(][^（）()]*[)）]")
_TRAILING_NOISE_RE = re.compile(r"[\s·\-—_/]+")

# 校验级别。block 拦得住开具，warning 只写进 safety_checks。
LEVEL_BLOCK = "block"
LEVEL_WARNING = "warning"

RULE_ALLERGY = "allergy_conflict"
RULE_DUPLICATE = "duplicate_therapy"
RULE_DAILY_DOSE = "daily_dose_excess"


# ==================== 内部工具 ====================


def _stringify(value: object | None) -> str | None:
    """UUID → str，None 保持 None（响应契约与后端方言无关）。"""
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _norm(name: object) -> str:
    """药名归一化：去括号说明、去空白与连接符、转小写、全角转半角无需处理。"""
    text = str(name or "").strip().lower()
    text = _PARENTHESES_RE.sub("", text)
    text = _TRAILING_NOISE_RE.sub("", text)
    return text


def _generic_name(drug_name: object) -> str:
    """剥掉剂型后缀，得到「通用名」，用于重复用药比对。

    「阿莫西林胶囊」与「阿莫西林片」是同一成分不同厂牌/剂型，
    同时开两支就是剂量翻倍——临床上这是最常见的重复用药形态。
    长后缀排在前面先剥，避免「缓释胶囊」被「胶囊」截断成「缓释」。
    **每轮只剥一个后缀**：反复剥离会把「布洛芬片」一路削成「布洛」，
    再和别的药名误判成同一成分——那是假阳性，代价是让医生对所有提示免疫。
    """
    core = _norm(drug_name)
    for suffix in DOSAGE_FORM_SUFFIXES:
        lowered = suffix.lower()
        if core.endswith(lowered) and len(core) > len(lowered):
            core = core[: -len(lowered)]
            break
    return core


def _match_prefix_key(text: str, table: dict[str, Any]) -> str | None:
    """在规则表里找「text 以某键开头」的那个键。

    前缀匹配而不是精确匹配的理由见 ALLERGY_CLASS_MAP 上方注释。
    表为空串时跳过，防止一个脏键把所有药都命中。
    """
    if not text:
        return None
    for key in table:
        normalized = _norm(key)
        if normalized and text.startswith(normalized):
            return key
    return None


def _standardize_allergen(raw: object) -> str:
    """把档案里的过敏原写法折叠成标准类别名。"""
    text = _norm(raw)
    if not text:
        return ""
    alias = _norm(ALLERGY_ALIAS_MAP.get(text, ""))
    if alias:
        return alias
    # 「青霉素过敏」「对磺胺类过敏」这类自由表述很常见：去掉常见噪声词后再比一次。
    for noise in ("过敏", "禁用", "药物", "品"):
        text = text.replace(noise, "")
    text = text.strip()
    return _norm(ALLERGY_ALIAS_MAP.get(text, text))


def _parse_mg(dosage: object) -> Decimal | None:
    """从「单次剂量」文本里解析出毫克数。

    支持 ``500mg`` / ``0.5g`` / ``250 mg`` / ``1片``。解析不出数（如「1片」这种
    无质量信息的写法）返回 None，调用方跳过剂量校验——**猜一个数会产生假告警**，
    而假告警看多了医生就会对所有告警免疫，这比漏报更危险。
    """
    text = str(dosage or "").strip().lower()
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(mg|g|微克|ug|μg)", text)
    if not match:
        return None
    value = Decimal(match.group(1))
    unit = match.group(2)
    if unit == "g":
        return value * Decimal("1000")
    if unit in ("微克", "ug", "μg"):
        return value / Decimal("1000")
    return value


def _per_day(frequency: object) -> int | None:
    """频次 → 每日次数。未知频次返回 None（不参与剂量判定）。"""
    text = str(frequency or "").strip().lower()
    return FREQUENCY_PER_DAY.get(text)


async def _resolve_profile_id(db: AsyncSession, user_id: uuid.UUID) -> uuid.UUID | None:
    """账号 id → 患者档案 id。查不到返回 None，由调用方决定报错口径。

    ``prescriptions.patient_id`` 外键指向 ``patient_profiles.id`` 而非 ``users.id``，
    与 visit / exam_request 同源。
    """
    row = await db.execute(
        select(PatientProfile.id).where(PatientProfile.user_id == user_id)
    )
    return row.scalar_one_or_none()


async def _doctor_name(db: AsyncSession, doctor_id: uuid.UUID) -> str:
    """单列 SELECT 取医生姓名。

    **不读 prescription.doctor 关系**：它是 ``lazy="selectin"``，只在查询时预取；
    对经 ``db.get()`` 主键取回的对象首次访问关系属性会触发隐式同步 IO，
    AsyncSession 下抛 MissingGreenlet，把一次正常读取变成 500
    （consult_service._doctor_name 同款坑）。
    """
    row = await db.execute(select(User.real_name).where(User.id == doctor_id))
    return str(row.scalar_one_or_none() or "")


async def _load_prescription(db: AsyncSession, rx_id: uuid.UUID) -> Prescription:
    rx = await db.get(Prescription, rx_id)
    if rx is None:
        raise ResourceNotFoundError("处方不存在")
    return rx


# ==================== 合理用药校验（纯规则） ====================


def check_allergy_conflicts(
    items: list[dict[str, Any]], allergy_history: list[str] | None
) -> list[dict[str, Any]]:
    """a) 过敏史校验：命中即 block。

    双向比对：既看「药品名/编码命中的过敏类别」是否出现在患者过敏史里，
    也把过敏史的自由写法折叠成标准类别（``盘尼西林`` → ``青霉素类``）。
    两个方向都做是因为任一侧都可能是录入者随手写的中文俗名。
    """
    allergens = {_standardize_allergen(a) for a in (allergy_history or [])}
    allergens.discard("")
    if not allergens:
        return []

    findings: list[dict[str, Any]] = []
    for item in items:
        name = _norm(item.get("drug_name"))
        code = str(item.get("drug_code") or "").strip().upper()
        category = None
        matched_by = None
        hit_key = _match_prefix_key(name, ALLERGY_CLASS_MAP)
        if hit_key is not None:
            category = ALLERGY_CLASS_MAP[hit_key]
            matched_by = f"药名前缀「{hit_key}」"
        else:
            for prefix, mapped in ALLERGY_CODE_PREFIX_MAP.items():
                if code.startswith(prefix.upper()):
                    category = mapped
                    matched_by = f"药品编码前缀「{prefix}」"
                    break
        if category is None or category not in allergens:
            continue
        findings.append(
            {
                "rule": RULE_ALLERGY,
                "level": LEVEL_BLOCK,
                "drug_code": _stringify(item.get("drug_code")),
                "drug_name": item.get("drug_name"),
                "message": (
                    f"{item.get('drug_name')} 属于{category}，与患者过敏史冲突"
                    f"（{matched_by}命中）"
                ),
                "detail": {"allergen": category, "matched_by": matched_by},
            }
        )
    return findings


def check_duplicate_therapy(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """c) 重复用药：同通用名不同厂牌/剂型 → block。

    判据是「剥掉剂型后缀后的通用名相同」，而不是 drug_code 相同
    （同 code 的唯一性已由协议层与服务层前置拦住）。
    两张不同厂牌的「阿莫西林」同时开等于剂量叠加，抗菌药尤其危险，
    因此这一条定级 block 而不是 warning。

    **只报一次**：三个同成分制剂聚在同一桶里时输出一条 finding（detail 列出全部），
    而不是两两配对输出三条——重复告警只会稀释医生对提示的注意力。
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        buckets.setdefault(_generic_name(item.get("drug_name")), []).append(item)

    findings: list[dict[str, Any]] = []
    for generic, group in buckets.items():
        if len(group) < 2 or not generic:
            continue
        names = [str(g.get("drug_name")) for g in group]
        findings.append(
            {
                "rule": RULE_DUPLICATE,
                "level": LEVEL_BLOCK,
                "drug_code": _stringify(group[-1].get("drug_code")),
                "drug_name": names[-1],
                "message": f"重复用药：{'、'.join(names)} 同为「{group[0].get('drug_name')}」的同类制剂",
                "detail": {"generic": generic, "drugs": names},
            }
        )
    return findings


def check_daily_dose(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """b) 单日剂量上限：超限 → warning（不阻断）。

    算法：``每日次数(FREQUENCY_PER_DAY[frequency]) × 单次剂量(mg) × 疗程天数``
    得到疗程总量，与「日上限 × 天数」比较——等价于直接比日均剂量，
    但把总量一并写进 message，医生核对时不必心算。

    定级 warning 而不是 block 的理由：剂量决策高度依赖个体（肾功能、体重、年龄），
    规则表只有一个成人通用阈值。把它做成硬拦会让肾功能不全患者的正常减量方案
    也被拦下，反而逼出更多无理由的 override，稀释这个通道真正的信号价值。
    """
    findings: list[dict[str, Any]] = []
    for item in items:
        name = _norm(item.get("drug_name"))
        hit_key = _match_prefix_key(name, DAILY_MAX_DOSE_MG)
        if hit_key is None:
            continue
        per_day = _per_day(item.get("frequency"))
        single_mg = _parse_mg(item.get("dosage"))
        if per_day is None or single_mg is None:
            continue
        max_daily = DAILY_MAX_DOSE_MG[hit_key]
        actual_daily = single_mg * Decimal(per_day)
        if actual_daily <= max_daily:
            continue
        days_raw = item.get("duration_days")
        days = int(days_raw) if isinstance(days_raw, int) and days_raw > 0 else 1
        findings.append(
            {
                "rule": RULE_DAILY_DOSE,
                "level": LEVEL_WARNING,
                "drug_code": _stringify(item.get("drug_code")),
                "drug_name": item.get("drug_name"),
                "message": (
                    f"{item.get('drug_name')} 日均剂量 {actual_daily}mg 超过演示规则上限 "
                    f"{max_daily}mg（{days} 天疗程合计 {actual_daily * Decimal(days)}mg）"
                ),
                "detail": {
                    "single_dose_mg": str(single_mg),
                    "times_per_day": per_day,
                    "daily_dose_mg": str(actual_daily),
                    "max_daily_dose_mg": str(max_daily),
                    "duration_days": days,
                    "total_dose_mg": str(actual_daily * Decimal(days)),
                },
            }
        )
    return findings


def run_safety_checks(
    items: list[dict[str, Any]], allergy_history: list[str] | None
) -> list[dict[str, Any]]:
    """跑完三条规则，返回结果列表（写入 prescriptions.safety_checks）。

    顺序刻意为 过敏 → 重复用药 → 剂量：阻断项排在前面，
    医生与药师在弹窗里第一眼看到的就是最该处理的那条。
    """
    findings: list[dict[str, Any]] = []
    findings.extend(check_allergy_conflicts(items, allergy_history))
    findings.extend(check_duplicate_therapy(items))
    findings.extend(check_daily_dose(items))
    # 稳定排序把 block 顶到前面：三条规则各自内部仍保持
    # 「过敏 → 重复 → 剂量」的追加顺序，只是级别分组优先。
    # 不排序会出现「前 N-1 条都是 block」不成立的序列
    # （如过敏2条+剂量1条+重复1条时，末位是 block）。
    findings.sort(key=lambda f: 0 if f.get("level") == LEVEL_BLOCK else 1)
    return findings


def _blocking_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [f for f in findings if f.get("level") == LEVEL_BLOCK]


def _raise_for_blocking(findings: list[dict[str, Any]]) -> None:
    """把阻断项翻译成既有异常类，不新增异常类型。

    过敏冲突走 ``AllergyConflictError``，其余阻断项（重复用药）走
    ``DrugInteractionError``——两者都是 422，且 code 已在 ErrorCode 里登记，
    前端可按 code 分支渲染「改药」还是「填理由」。
    """
    blocking = _blocking_findings(findings)
    if not blocking:
        return
    allergy = [f for f in blocking if f["rule"] == RULE_ALLERGY]
    picked = allergy or blocking
    messages = "；".join(str(f["message"]) for f in picked)
    detail = {"findings": picked, "override_required": True}
    if allergy:
        raise AllergyConflictError(f"处方与患者过敏史冲突：{messages}", detail=detail)
    raise DrugInteractionError(f"处方存在用药安全风险：{messages}", detail=detail)


def _items_as_dicts(rx: Prescription) -> list[dict[str, Any]]:
    """ORM 明细 → 规则引擎消费的普通字典。

    规则函数刻意不接受 ORM 对象：纯函数的入参是 dict，
    单元测试就能直接构造用例而不必建库建表（本文件的规则用例全部走这条路）。

    **必须 ``list(rx.items)`` 复制一份**：AsyncSession 下首次访问 selectin 关系
    会触发隐式同步 IO 并抛 MissingGreenlet。对经 ``db.get()`` 取回的处方对象，
    items 此刻可能尚未加载——这不是理论风险，而是 confirm 路径的必经一步。
    """
    return [
        {
            "drug_code": item.drug_code,
            "drug_name": item.drug_name,
            "dosage": item.dosage,
            "frequency": item.frequency,
            "duration_days": item.duration_days,
        }
        for item in list(rx.items)
    ]


# ==================== 序列化 ====================


def _coerce_checks(raw: Any) -> list[dict[str, Any]]:
    """safety_checks 列容错成字典列表（历史脏数据不能让详情接口 500）。"""
    if not isinstance(raw, list):
        return []
    return [f for f in raw if isinstance(f, dict)]


async def serialize_prescription(db: AsyncSession, rx: Prescription) -> dict[str, object]:
    """Prescription ORM → 可直接喂给 ``PrescriptionOut`` 的字典。"""
    checks = _coerce_checks(rx.safety_checks)
    return {
        "id": _stringify(rx.id),
        "visit_id": _stringify(rx.visit_id),
        "patient_id": _stringify(rx.patient_id),
        "doctor_id": _stringify(rx.doctor_id),
        "doctor_name": await _doctor_name(db, rx.doctor_id),
        "status": rx.status,
        "diagnosis": rx.diagnosis,
        "notes": rx.notes,
        "total_amount": rx.total_amount,
        "items": [
            {
                "id": _stringify(item.id),
                "drug_code": item.drug_code,
                "drug_name": item.drug_name,
                "specification": item.specification,
                "dosage": item.dosage,
                "frequency": item.frequency,
                "route": item.route,
                "duration_days": item.duration_days,
                "quantity": str(item.quantity),
                "unit": item.unit,
                "unit_price": str(item.unit_price),
                "usage_instructions": item.usage_instructions,
            }
            # 同 _items_as_dicts：复制一份再迭代，避免异步会话下的隐式关系加载。
            for item in list(rx.items)
        ],
        "safety_checks": checks,
        "blocked_count": sum(1 for f in checks if f.get("level") == LEVEL_BLOCK),
        "warning_count": sum(1 for f in checks if f.get("level") == LEVEL_WARNING),
        "safety_override_reason": rx.safety_override_reason,
        "confirmed_at": rx.confirmed_at,
        "cancelled_at": rx.cancelled_at,
        "created_at": rx.created_at,
        "updated_at": rx.updated_at,
        "allowed_transitions": allowed_targets(ENTITY, rx.status),
    }


# ==================== 创建草稿 ====================


async def create_draft(
    db: AsyncSession,
    doctor_id: uuid.UUID,
    payload: PrescriptionCreateRequest,
) -> Prescription:
    """开一张处方草稿（status=draft），明细至少一条，drug_code 唯一。

    **visit 归属与状态双闸门**：visit 必须属于当前医生（越权 404，不泄露
    「这条病历存在」），且状态未 closed。这里刻意沿用 consult_service 的口径
    而不是要求 visit 必须是 documented——现实中医生常常边写病历边开药，
    把开药卡在「病历定稿之后」会让诊室里多一次无意义的点击。

    **drug_code 唯一性提前给出友好错误**：表上有 ``uq_rx_item_drug``，
    撞上去只会冒成 IntegrityError → 500；协议层已经拦过一次，
    这里是第二层（防止有人绕过 Pydantic 直接调 service）。

    新建草稿**不跑安全校验**：草稿是医生的工作台便签，此刻就把他拦住
    只会打断思路；校验发生在「确认开具」这一有法律效力的动作上。
    代价是 draft 行可能带着明知冲突的内容存在一段时间，这是刻意的取舍。

    初始态走 ``validate_status_value`` 而不是 ``assert_transition``：
    新建一张单子不是一次状态迁移（没有 from 状态），与 order_exam 同一做法。
    """
    visit = await db.get(Visit, uuid.UUID(str(payload.visit_id)))
    if visit is None or visit.doctor_id != doctor_id:
        raise ResourceNotFoundError("就诊记录不存在")

    if visit.status not in OPENABLE_VISIT_STATUSES:
        raise InvalidStateTransitionError(
            f"当前就诊状态「{visit.status}」不允许开具处方",
            current=visit.status,
            target=PrescriptionStatus.DRAFT.value,
            entity="visit",
        )

    codes = [item.drug_code for item in payload.items]
    duplicated = sorted({code for code in codes if codes.count(code) > 1})
    if duplicated:
        raise DrugInteractionError(
            f"同一处方内药品编码重复：{'、'.join(duplicated)}",
            detail={"duplicate_drug_codes": duplicated},
        )

    rx = Prescription(
        visit_id=visit.id,
        patient_id=visit.patient_id,
        doctor_id=visit.doctor_id,
        status=validate_status_value(ENTITY, PrescriptionStatus.DRAFT.value),
        diagnosis=payload.diagnosis,
        notes=payload.notes,
    )
    rx.items = [_build_item(item) for item in payload.items]
    rx.recalculate_total()

    db.add(rx)
    # 主键 default=uuid.uuid4 在 **flush 时**才求值，构造时 rx.id 是 None。
    # recalculate_total 只读 items 不读 id，因此这里 flush 的目的在于让
    # cascade 插入子行时 prescription_id 已有值（与 exam/consult 侧同一时序）。
    await db.flush()
    await db.commit()
    await db.refresh(rx)

    logger.info(
        "处方草稿已创建 id=%s visit=%s doctor=%s patient=%s items=%d total=%d分",
        rx.id,
        visit.id,
        doctor_id,
        visit.patient_id,
        len(rx.items),
        rx.total_amount,
    )
    return rx


def _build_item(item: Any) -> PrescriptionItem:
    """入参明细 → ORM 行。quantity/unit_price 交给 Numeric 列自己处理。"""
    return PrescriptionItem(
        drug_code=item.drug_code,
        drug_name=item.drug_name,
        specification=item.specification,
        dosage=item.dosage,
        frequency=item.frequency,
        route=item.route,
        duration_days=item.duration_days,
        quantity=item.quantity,
        unit=item.unit,
        unit_price=item.unit_price,
        usage_instructions=item.usage_instructions,
    )


# ==================== 更新草稿 ====================


async def update_draft(
    db: AsyncSession,
    rx_id: uuid.UUID,
    doctor_id: uuid.UUID,
    payload: PrescriptionUpdateRequest,
) -> Prescription:
    """修改草稿：仅 draft 可改，明细整体替换，金额重算。

    **非 draft 一律 409**：confirmed 之后的处方已经有法律效力并进入收费流程，
    此时改明细会让票据与医嘱不一致；cancelled 是终态。矩阵里没有
    confirmed→draft 这条回退边，所以这里也不需要「先退回再改」。

    同样只允许开单医生本人修改（越权 404）。药师持 read_all/approve 但
    **刻意没有 update**（rbac_matrix 注释原话：有问题应驳回让医生重开），
    因此路由层的 prescription:update 已经把药师挡在门外。

    明细替换靠 ``cascade="all, delete-orphan"``：重建 ``rx.items`` 即可让旧行删除、
    新行插入，同事务提交，不会出现「删了旧的没插新的」。金额重算必须排在替换之后
    （见函数体内注释）。
    """
    rx = await _load_prescription(db, rx_id)
    if rx.doctor_id != doctor_id:
        raise ResourceNotFoundError("处方不存在")

    if rx.status != PrescriptionStatus.DRAFT.value:
        # 这里**不调 assert_transition**：用户此刻想做的是「改内容」而不是「改状态」，
        # 伪造一次 draft→draft 的迁移会产出一条误导性的 current_status/target_status，
        # 仿佛系统认为存在这样一条状态边。与 consult_service.save_medical_record
        # 对 closed 的处理是同一取舍：闸门管字段写入，矩阵管状态合法性，两层不混用。
        raise InvalidStateTransitionError(
            "只有草稿状态的处方可以修改",
            current=rx.status,
            target=PrescriptionStatus.DRAFT.value,
            entity=ENTITY,
        )

    codes = [item.drug_code for item in payload.items]
    duplicated = sorted({code for code in codes if codes.count(code) > 1})
    if duplicated:
        raise DrugInteractionError(
            f"同一处方内药品编码重复：{'、'.join(duplicated)}",
            detail={"duplicate_drug_codes": duplicated},
        )

    # **必须先删旧行再插新行**：SQLAlchemy 的 unit of work 在同一 flush 里
    # 对 delete-orphan 与新增行没有「先删后插」的保证，而新旧行在
    # uq_rx_item_drug（prescription_id + drug_code）上可以取到相同的 drug_code，
    # 于是 INSERT 先落地就撞 ``UNIQUE constraint failed:
    # prescription_items.prescription_id, prescription_items.drug_code``。
    # clear + flush 把 DELETE 显式提前落库，之后的 INSERT 不再有针对性冲突。
    rx.items.clear()
    await db.flush()

    rx.items = [_build_item(item) for item in payload.items]
    if payload.diagnosis is not None:
        rx.diagnosis = payload.diagnosis
    if payload.notes is not None:
        rx.notes = payload.notes
    # **必须在明细替换之后**重算：recalculate_total 读的是此刻挂在关系上的行，
    # 顺序颠倒会把旧明细的钱写进新明细的单子。
    rx.recalculate_total()

    await db.flush()
    await db.commit()
    await db.refresh(rx)

    logger.info("处方草稿已更新 id=%s doctor=%s items=%d total=%d分", rx.id, doctor_id, len(rx.items), rx.total_amount)
    return rx


# ==================== 确认开具 ====================


async def confirm_prescription(
    db: AsyncSession,
    rx_id: uuid.UUID,
    doctor_id: uuid.UUID,
    payload: PrescriptionConfirmRequest | None = None,
) -> tuple[Prescription, bool]:
    """确认开具：draft → confirmed，执行纯规则合理用药校验。

    返回 ``(rx, overridden)``：overridden 如实告知本次是否存在
    「被医生理由压过的阻断项」，前端据此提示药师将看到该理由。

    **校验时机在状态迁移之前**：先把三条规则跑完、把结果算清楚，
    再决定要不要动 status。若先迁移后校验，一旦拦截就得回滚，
    而半途 commit 过的会话回滚不干净的风险远大于多算一遍。

    **override 语义（模块 docstring 第 2 节）**：存在阻断项时默认 422 拒绝；
    请求带非空 ``safety_override_reason`` 才放行，理由落库。
    这是「医师处方权优先」——规则引擎提供信息，最终裁量权在人。
    校验结果无论是否被压过都**完整写入 safety_checks**，
    包括被压过的那几条：药师审核时要看得见医生忽略了什么。

    剂量超限是 warning，不进 override 通道，永远只是提示。

    幂等性：已是 confirmed 的处方再次确认会得到 409（confirmed→confirmed
    不是矩阵里的边）。刻意不做幂等静默返回——重复开具意味着重复进入收费流程，
    让医生明确看到「这张单子已经开过了」比让他以为成功了安全。

    **非 draft 一律拒**：cancelled/dispensed/paid 都到不了 confirmed，
    assert_transition 会自然给出带 current_status 的 409；这里不额外加白名单，
    因为矩阵本来就是这些边的唯一合法性来源。

    **空明细无法被开具**：``draft`` 状态下没有任何后继能绕过 items 为空这一事实——
    理论上 items 只可能来自创建/更新两条入口，两者都强制至少一条，
    因此不再在这里重复校验（多一道判据就多一处会漂移的地方）。
    """
    rx = await _load_prescription(db, rx_id)
    if rx.doctor_id != doctor_id:
        raise ResourceNotFoundError("处方不存在")

    # 先探一次迁移是否合法，再跑规则：非法起点此刻就该 409，
    # 不该让人以为「是过敏拦住了我」。
    assert_transition(ENTITY, rx.status, PrescriptionStatus.CONFIRMED.value)

    profile = await db.get(PatientProfile, rx.patient_id)
    allergy_history = list(profile.allergy_history or []) if profile is not None else []

    findings = run_safety_checks(_items_as_dicts(rx), allergy_history)
    blocking = _blocking_findings(findings)
    reason = payload.safety_override_reason if payload is not None else None

    if blocking and not (reason and reason.strip()):
        _raise_for_blocking(findings)

    # 上面已探过合法性，这里正式落笔；两次判定之间不写库，因此不存在半途状态。
    rx.status = assert_transition(ENTITY, rx.status, PrescriptionStatus.CONFIRMED.value)
    rx.safety_checks = findings
    rx.safety_override_reason = reason.strip() if reason else None
    rx.confirmed_at = utcnow()

    try:
        await db.flush()
        await db.commit()
    except Exception:
        await db.rollback()
        raise

    await db.refresh(rx)

    logger.info(
        "处方已开具 id=%s doctor=%s status=%s 阻断项=%d 提示项=%d overridden=%s",
        rx.id,
        doctor_id,
        rx.status,
        len(blocking),
        len(findings) - len(blocking),
        bool(blocking),
    )
    if blocking:
        # 医疗红线：被压过的阻断项必须进 WARNING 日志（生产接药学部告警通道）。
        for finding in blocking:
            logger.warning("处方风险被医生理由压过 id=%s %s", rx.id, finding["message"])

    return rx, bool(blocking)


# ==================== 取消 ====================


async def cancel_prescription(
    db: AsyncSession,
    rx_id: uuid.UUID,
    doctor_id: uuid.UUID,
    reason: str | None = None,
) -> Prescription:
    """作废处方：draft/confirmed → cancelled。

    **paid 不可取消**（矩阵里 paid 的后继只有 dispensed）。缴费后的撤药
    属于退费流程，要冲正收费单、可能要追回已发出的药，那是收费域的动作，
    在这里开一个「取消」口子会留下一张已收钱却被作废的处方。
    非法起点由 assert_transition 自然 409，detail 带 current/target。

    作废原因追加进 ``notes`` 而不是丢弃：表上没有独立的 cancel_reason 列，
    而「谁因为什么作废了一张处方」是处方点评要看的信息，
    塞进备注列的成本远低于为此新增迁移。

    终态 cancelled 再次取消会得到 409，不做静默成功——与 confirm 同一取舍。
    """
    rx = await _load_prescription(db, rx_id)
    if rx.doctor_id != doctor_id:
        raise ResourceNotFoundError("处方不存在")

    rx.status = assert_transition(ENTITY, rx.status, PrescriptionStatus.CANCELLED.value)
    rx.cancelled_at = utcnow()
    if reason:
        note = f"[作废原因] {reason}"
        rx.notes = f"{rx.notes}\n{note}" if rx.notes else note

    await db.flush()
    await db.commit()
    await db.refresh(rx)

    logger.info("处方已作废 id=%s doctor=%s reason=%s", rx.id, doctor_id, reason)
    return rx


# ==================== 读取与行级隔离 ====================


async def get_prescription_detail(
    db: AsyncSession,
    rx_id: uuid.UUID,
    *,
    requester_user_id: uuid.UUID,
    requester_role: str,
) -> Prescription:
    """按角色取处方详情，行级隔离在服务层强制。

    - 医生：只能看自己开的（越权 404）
    - 患者：只能看自己档案下的（越权 404）
    - pharmacist / cashier：持 read_all，可读全部（核对待发药清单、核对药费）
    - **admin 同样 404**：矩阵刻意不给管理员临床记录读写，
      理由与 consult_service.get_visit_detail 一致——不能用一个函数后门
      把「管理员不应能篡改诊疗记录」这条设计决策抹掉。
      ``require_permission`` 对 admin 的依赖层放行只代表「允许调用端点」，
      可见行范围仍由本函数说了算。

    404 而不是 403：403 等于承认「这张处方存在，只是不属于你」，
    攻击者可据此枚举全院处方 ID。
    """
    rx = await _load_prescription(db, rx_id)

    if requester_role == RoleName.DOCTOR.value:
        if rx.doctor_id != requester_user_id:
            raise ResourceNotFoundError("处方不存在")
        return rx

    if requester_role == RoleName.PATIENT.value:
        profile_id = await _resolve_profile_id(db, requester_user_id)
        if profile_id is None or rx.patient_id != profile_id:
            raise ResourceNotFoundError("处方不存在")
        return rx

    if requester_role in (RoleName.PHARMACIST.value, RoleName.CASHIER.value):
        return rx

    raise ResourceNotFoundError("处方不存在")


async def list_prescriptions(
    db: AsyncSession,
    *,
    requester_user_id: uuid.UUID,
    requester_role: str,
    status_filter: str | None = None,
    visit_id: uuid.UUID | None = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[Prescription], int]:
    """处方列表：医生看自己开的，pharmacist/cashier 看全部，患者只看自己的。

    行级隔离按角色强制注入过滤条件，客户端传任何参数都越不出自己的范围：
    患者即使知道别人的 visit_id 也查不到别人的处方
    （patient_id 条件恒在，且 ``visit_id`` 对它只做收窄）。

    count 用子查询包一层，与 consult_service.list_my_visits 同一做法：
    条件复杂时裸 ``func.count()`` 有漏 WHERE 的风险。

    排序按创建时间倒序 + id 兜底：id 作第二键保证分页稳定，
    否则同一秒内创建的多张处方在翻页时会重复出现或漏掉。
    """
    conditions: list[Any] = []

    if requester_role == RoleName.DOCTOR.value:
        conditions.append(Prescription.doctor_id == requester_user_id)
    elif requester_role == RoleName.PATIENT.value:
        profile_id = await _resolve_profile_id(db, requester_user_id)
        if profile_id is None:
            return [], 0
        conditions.append(Prescription.patient_id == profile_id)
    elif requester_role not in (RoleName.PHARMACIST.value, RoleName.CASHIER.value):
        # lab_tech / admin 与任何其他角色：矩阵里没有处方读权限，
        # 走到这里说明依赖层被绕过，返回空集而不是抛权限错（不泄露角色信息）。
        return [], 0

    if status_filter:
        conditions.append(Prescription.status == status_filter)
    if visit_id is not None:
        conditions.append(Prescription.visit_id == visit_id)

    stmt = select(Prescription).where(*conditions)
    total = int((await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one())

    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    rows = (
        await db.execute(
            stmt.order_by(Prescription.created_at.desc(), Prescription.id)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).scalars().all()
    return list(rows), total


async def list_visit_prescriptions(
    db: AsyncSession, visit_id: uuid.UUID
) -> list[Prescription]:
    """某次就诊下的全部处方（供病历页展开）。归属校验由调用方完成。"""
    rows = (
        await db.execute(
            select(Prescription)
            .where(Prescription.visit_id == visit_id)
            .order_by(Prescription.created_at.asc(), Prescription.id)
        )
    ).scalars().all()
    return list(rows)


__all__ = [
    "ALLERGY_ALIAS_MAP",
    "ALLERGY_CLASS_MAP",
    "DAILY_MAX_DOSE_MG",
    "ENTITY",
    "FREQUENCY_PER_DAY",
    "LEVEL_BLOCK",
    "LEVEL_WARNING",
    "OPENABLE_VISIT_STATUSES",
    "RULE_ALLERGY",
    "RULE_DAILY_DOSE",
    "RULE_DUPLICATE",
    "cancel_prescription",
    "check_allergy_conflicts",
    "check_daily_dose",
    "check_duplicate_therapy",
    "confirm_prescription",
    "create_draft",
    "get_prescription_detail",
    "list_prescriptions",
    "list_visit_prescriptions",
    "run_safety_checks",
    "serialize_prescription",
    "update_draft",
]
