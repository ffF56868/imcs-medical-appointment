"""处方域的请求/响应模型（阶段 5 处方侧）。

字段口径对齐 ``app/models/pharmacy.py`` 的 ``Prescription`` / ``PrescriptionItem``。
约定沿用前序批次（``schemas/consult.py``、``schemas/exam.py``）：

1. **响应模型显式列字段**，不用 ``from_attributes`` 全量导出 ORM。
   ``Prescription.doctor`` 指向 ``User``，而 User 带 ``password_hash``；
   处方详情一旦全量展开关系，就等于把口令哈希挂到药师工作台上。
2. **金额对外保持整数「分」**：``total_amount`` 与 ``fee_amount``/``registration_fee``
   同源，schema 层不做除以 100 的换算——精度决策留在模型层（Decimal 计算 +
   ROUND_HALF_UP 取整），格式化展示是前端的事。明细里的 ``unit_price`` 例外：
   它在表里就是「元」的 Numeric(10,2)，照实暴露为字符串以便前端原样比对。
3. **UUID 对外一律字符串**：SQLite 测试环境与 PG 驱动返回类型不同，
   在边界统一成 str 让响应契约与后端方言无关。
4. **枚举取值复用模型层常量**（见文件末尾的 import 期断言）：
   状态集合同时是 service 的写入依据和前端的下拉数据源，两处各写一遍必然漂移。

**为什么请求体里没有 patient_id / doctor_id**：医生身份取自 JWT，患者身份由
``visits.patient_id`` 带出（``start_consultation`` 从预约复制）。这两个字段一旦
可由客户端指定，就等于任何人都能替别的医生开处方、或把处方挂到别人身上，
归属校验会失去意义。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.pharmacy import PrescriptionStatus

# ==================== 取值常量 ====================

# 处方五态，唯一真相来源是 pharmacy.PrescriptionStatus。
PRESCRIPTION_STATUS_VALUES: tuple[str, ...] = tuple(item.value for item in PrescriptionStatus)

# 新建处方的初始态。本模块只写入 draft，confirmed/paid/dispensed/cancelled
# 由确认开具 / 收费侧 / 发药侧推进。
PRESCRIPTION_INITIAL_STATUS: str = PrescriptionStatus.DRAFT.value

# 明细条数上限。一张门诊处方几十味药已经是极小概率事件，
# 不设上限会让「一次请求塞进一千行」变成可用的 DoS 面。
MAX_ITEMS = 50

# 强制开具理由的长度上限。够写清「已与患者沟通并复核过敏史」这类临床说明，
# 又不至于让 override 字段变成自由文本倾倒场。
MAX_OVERRIDE_REASON_LENGTH = 500


def _as_uuid(value: object, field_name: str) -> uuid.UUID:
    """字符串 → UUID，失败抛 ValueError（协议层 → 422）。

    与 consult 侧同一取舍：**请求体里**的非法 UUID 是「你请求写错了」，
    而路径参数里的非法 UUID 走路由层 ``_parse_uuid`` 报 404（语义是定位资源）。
    """
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field_name} 必须是合法的 UUID 字符串") from exc


def _clean_text(value: str | None) -> str | None:
    """可选文本统一去空白；空串归一为 None。

    把 "" 和 None 混着存进 Text 列，会让「医生到底填没填医嘱备注」这种判断
    在读取侧多一处 ``or None`` 补偿，而补偿代码最容易在不同入口之间漏掉。

    长度不在这里再查一遍：字段声明上的 ``max_length`` 已由 Pydantic 先拦，
    重复实现只会多出两处会漂移的上限。
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


# ==================== 明细 ====================


class PrescriptionItemIn(BaseModel):
    """处方明细行入参。

    ``drug_code`` 必填而不是由服务端按名称反查：药品字典属另一套主数据，
    本域刻意不依赖它（见 service 模块 docstring 的边界声明）。
    ``quantity`` 用 ``float`` 承载但语义上是「两位小数内的数」，
    落库列是 Numeric(10,2)；真正的金额计算全程走 Decimal，不碰这个 float。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    drug_code: str = Field(min_length=1, max_length=30, description="药品编码，如 D0001")
    drug_name: str = Field(min_length=1, max_length=100, description="药品全称，含剂型")
    specification: str | None = Field(default=None, max_length=100, description="规格，如 0.25g*24片")
    dosage: str | None = Field(default=None, max_length=50, description="单次剂量，如 1片 / 500mg")
    frequency: str | None = Field(
        default=None,
        max_length=30,
        description="用药频次 qd/bid/tid/qid（剂量校验按此换算每日次数）",
    )
    route: str | None = Field(default=None, max_length=30, description="给药途径，如 口服")
    duration_days: int | None = Field(default=None, ge=1, le=365, description="疗程天数")
    quantity: float = Field(gt=0, le=10000, description="开药数量，可为 1.5 这类两位小数")
    unit: str = Field(min_length=1, max_length=20, description="单位，如 盒/瓶/支")
    unit_price: float = Field(ge=0, le=1000000, description="单价，单位元")
    usage_instructions: str | None = Field(default=None, max_length=1000, description="用药指导")

    @field_validator("drug_code")
    @classmethod
    def _normalize_drug_code(cls, value: str) -> str:
        """编码大写去空格：``d0001`` 与 ``D0001`` 必须落到同一个唯一约束上。

        数据库的 UniqueConstraint 是大小写敏感的，不归一化就会让同一味药
        以两种写法双双入库，重复用药检测也随之失效。
        """
        return value.strip().upper()


def _assert_unique_codes(items: list["PrescriptionItemIn"]) -> list["PrescriptionItemIn"]:
    """同一处方内 drug_code 唯一的公共判据（三个请求模型共用一份实现）。

    写成模块级函数而不是每个模型各抄一遍校验体：两处判据一旦分叉，
    就会出现「创建拦得住、更新拦不住」这种只在生产才暴露的漏洞。
    """
    seen: set[str] = set()
    dup: list[str] = []
    for item in items:
        if item.drug_code in seen and item.drug_code not in dup:
            dup.append(item.drug_code)
        seen.add(item.drug_code)
    if dup:
        raise ValueError(f"同一处方内药品编码重复：{'、'.join(dup)}")
    return items


# ==================== 请求 ====================


class PrescriptionCreateRequest(BaseModel):
    """创建处方草稿。

    ``items`` 必传且至少一条：空明细的处方在临床上不存在——它连一张纸都不该打印。
    更新草稿同样要求全量提交而不是提供增删行的三个端点：草稿阶段的改药在临床上
    就是「重开一张单子」，逐行 PATCH 会让前端自己维护行 id 集合，
    也让「明细至少一条」这条不变量在多次调用之间无法保证。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    visit_id: str = Field(description="所属就诊 id；处方必须挂在医生自己的有效就诊上")
    diagnosis: str | None = Field(default=None, max_length=1000, description="处方对应的临床诊断")
    notes: str | None = Field(default=None, max_length=2000, description="医嘱备注")
    items: list[PrescriptionItemIn] = Field(
        min_length=1, max_length=MAX_ITEMS, description="处方明细，至少一条"
    )

    @field_validator("visit_id")
    @classmethod
    def _validate_visit_id(cls, value: str) -> str:
        return str(_as_uuid(value, "visit_id"))

    @field_validator("items")
    @classmethod
    def _check_unique_drug_code(cls, value: list[PrescriptionItemIn]) -> list[PrescriptionItemIn]:
        return _assert_unique_codes(value)


class PrescriptionUpdateRequest(BaseModel):
    """修改处方草稿。

    只有 draft 可改（service 层闸门），且语义是**整体替换**：
    ``items`` 必传，diagnosis/notes 传了才覆盖。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    diagnosis: str | None = Field(default=None, max_length=1000)
    notes: str | None = Field(default=None, max_length=2000)
    items: list[PrescriptionItemIn] = Field(
        min_length=1, max_length=MAX_ITEMS, description="替换后的完整明细"
    )

    @field_validator("items")
    @classmethod
    def _check_unique_drug_code(cls, value: list[PrescriptionItemIn]) -> list[PrescriptionItemIn]:
        return _assert_unique_codes(value)


class PrescriptionConfirmRequest(BaseModel):
    """确认开具（draft → confirmed）。

    ``safety_override_reason`` 是**医师处方权优先**这条医疗业务规则的接口：
    规则引擎拦下的过敏/重复用药风险，默认拒绝开具；医生若已在临床上复核过
    风险（例如过敏史记录有误、或脱敏治疗期间仍需用药），可以带一句非空理由
    强行开具，理由原样落 ``safety_override_reason`` 供药师审核与事后追责。
    留痕不是形式主义：没有这一列，强制开具就与「规则没跑」不可区分。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    safety_override_reason: str | None = Field(
        default=None,
        max_length=MAX_OVERRIDE_REASON_LENGTH,
        description="存在阻断项时必填；无阻断项时可留空",
    )

    @field_validator("safety_override_reason")
    @classmethod
    def _strip_reason(cls, value: str | None) -> str | None:
        # 只传一个空格串等于「想绕过但没有理由」，视为未提供而不是空理由，
        # 否则 service 会把 "" 当成合法的非空 override 凭证。
        return _clean_text(value)


class PrescriptionCancelRequest(BaseModel):
    """作废处方（draft/confirmed → cancelled）。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    reason: str | None = Field(default=None, max_length=500, description="作废原因（写入 notes 留痕）")

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str | None) -> str | None:
        return _clean_text(value)


# ==================== 响应 ====================


class PrescriptionItemOut(BaseModel):
    """处方明细行。

    ``quantity`` / ``unit_price`` 以字符串回显而不是 float：Numeric(10,2) 读出来是
    Decimal，而 0.1 这类值经 JSON 浮点往返会出现 0.1000000000000000055——
    药师核对药费、收费侧比对票据时必须拿到与库里完全一致的表示。
    前端要展示成数字由它自己解析，服务端不替它做有损转换。
    """

    id: str
    drug_code: str
    drug_name: str
    specification: str | None = None
    dosage: str | None = None
    frequency: str | None = None
    route: str | None = None
    duration_days: int | None = None
    quantity: str = Field(description="数量，Numeric 原样字符串")
    unit: str
    unit_price: str = Field(description="单价，单位元，Numeric 原样字符串")
    usage_instructions: str | None = None


class SafetyCheckOut(BaseModel):
    """一条合理用药校验结果（纯规则产出，不含任何 AI 判断）。"""

    rule: str = Field(description="allergy_conflict / duplicate_therapy / daily_dose_excess")
    level: str = Field(description="block 会拦住开具；warning 只提示不拦")
    drug_code: str | None = None
    drug_name: str | None = None
    message: str = Field(description="人读结论，直接可用于前端弹窗")
    detail: dict[str, Any] = Field(default_factory=dict)


class PrescriptionOut(BaseModel):
    """处方详情 / 列表行通用。"""

    id: str
    visit_id: str
    patient_id: str = Field(description="患者档案 UUID（patient_profiles.id），非账号 id")
    doctor_id: str
    doctor_name: str
    status: str
    diagnosis: str | None = None
    notes: str | None = None
    total_amount: int = Field(description="处方总金额，单位分（整数口径，见模块 docstring）")
    items: list[PrescriptionItemOut] = Field(default_factory=list)
    safety_checks: list[SafetyCheckOut] = Field(default_factory=list)
    blocked_count: int = Field(default=0, description="阻断级风险条数，>0 时未带理由无法开具")
    warning_count: int = Field(default=0, description="提示级风险条数，不阻断开具")
    safety_override_reason: str | None = Field(
        default=None, description="医生带理由强行开具时填写的理由；为空表示未被拦截或无风险"
    )
    confirmed_at: datetime | None = None
    cancelled_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    allowed_transitions: list[str] = Field(
        default_factory=list, description="当前状态可迁移到的目标状态（矩阵投影）"
    )


class PrescriptionListResponse(BaseModel):
    """处方分页响应。page/page_size 回显实际生效值。"""

    items: list[PrescriptionOut]
    total: int
    page: int
    page_size: int


class PrescriptionConfirmResponse(BaseModel):
    """确认开具的回执。

    ``overridden`` 如实告知这次是否用了强制开具通道：前端据此提示
    「已带理由开具，药师将看到你的理由」，而不必自己比对 safety_checks。
    """

    prescription: PrescriptionOut
    overridden: bool = Field(description="本次是否存在被医生理由压过的阻断项")


class PrescriptionCancelResponse(BaseModel):
    """作废回执。

    包一层对象而不是直接返 ``PrescriptionOut``：与 confirm 的回执形态对称，
    前端处理「状态推进类」动作时只需认一种响应结构（``{prescription: ...}``）。
    """

    prescription: PrescriptionOut


# ==================== 取值一致性守卫 ====================


def _assert_literals_match_constants() -> None:
    """import 期比对 schema 侧常量与模型枚举，任一侧漂移立即抛错。"""
    assert set(PRESCRIPTION_STATUS_VALUES) == {item.value for item in PrescriptionStatus}
    assert PRESCRIPTION_INITIAL_STATUS == PrescriptionStatus.DRAFT.value


_assert_literals_match_constants()


__all__ = [
    "MAX_ITEMS",
    "MAX_OVERRIDE_REASON_LENGTH",
    "PRESCRIPTION_INITIAL_STATUS",
    "PRESCRIPTION_STATUS_VALUES",
    "PrescriptionCancelRequest",
    "PrescriptionCancelResponse",
    "PrescriptionConfirmRequest",
    "PrescriptionConfirmResponse",
    "PrescriptionCreateRequest",
    "PrescriptionItemIn",
    "PrescriptionItemOut",
    "PrescriptionListResponse",
    "PrescriptionOut",
    "PrescriptionUpdateRequest",
    "SafetyCheckOut",
]
