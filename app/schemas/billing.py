"""收费接口的请求/响应模型（阶段 5 收费域）。

字段口径对齐 ``ChargeOrder``（models/billing.py）：``items`` 是 JSONB 列表，
每张收费单聚合一次就诊里的若干费用项。约定与前序批次
（``schemas/consult.py``、``schemas/exam.py``）保持一致：

1. **金额一律整数「分」**，schema 层不做除以 100 的换算。模型层注释已写明
   「避免浮点误差」，一旦在这里转成 float，精度决策就被推给二进制浮点。
   处方侧 ``Prescription.total_amount`` 同为 int 分，两边口径一致。
2. **UUID 对外一律字符串**：SQLite 测试环境与 PG 驱动返回类型不同，
   在边界统一成 str 让响应契约与后端方言无关。
3. **响应显式列字段**，不用 ``from_attributes`` 全量导出 ORM——``ChargeOrder.cashier``
   指向带 ``password_hash`` 的 User，全量展开等于把口令哈希挂到收费员页面。

============================ items 的三类形态 ============================

请求体里 ``items`` 用判别式联合（按 ``type`` 分支）而不是一个宽松 dict：

- ``{"type": "prescription", "prescription_id": ...}`` —— 关联一张处方，
  金额由服务端从 ``Prescription.total_amount`` 取，**不接受客户端传 amount**。
- ``{"type": "exam", "exam_request_id": ...}`` —— 关联一张检查申请单，
  金额取 ``ExamRequest.fee_amount``。
- ``{"type": "other", "name": ..., "amount": 分}`` —— 自定义项（挂号费、材料费等），
  金额由客户端给出但必须是正整数分。

之所以对 prescription / exam 两类**禁止客户端传金额**：这两类费用对应真实业务单据，
若允许前端传价，就能开出一张「关联了 300 元处方却写着 0 元」的收费单，
应收与实收对不上，退费时也无法回溯。金额必须由服务端从源单据求值，
客户端传的 name/amount 只用于展示回填。``other`` 类没有源单据，才由客户端定价。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ==================== 常量 ====================

# 收费单状态取值，唯一真相来源是 billing.ChargeOrderStatus。
CHARGE_STATUS_VALUES: tuple[str, ...] = (
    "pending",
    "paid",
    "refunded",
    "cancelled",
)

# payment_channel 允许值。本项目只实现 mock（不接真实支付渠道），
# 但列定义留了 cash / insurance，故这里三值都放行以兼容演示与后续接入。
PAYMENT_CHANNEL_VALUES: tuple[str, ...] = ("mock", "cash", "insurance")

# 单张收费单的费用项上限。临床上一次就诊不会几十项；设上限防自由文本滥用，
# 也让 items 保持可枚举、可在票据上打印。
MAX_CHARGE_ITEMS = 50


def _as_uuid(value: object, field_name: str) -> uuid.UUID:
    """字符串 → UUID，失败抛 ValueError（协议层 → 422）。

    与 consult / exam schema 同款：请求体里的 ID 形态非法属于「请求本身写错」，
    在进入业务逻辑前拒掉；路径参数里的非法 UUID 走路由层报 404（语义是定位资源）。
    """
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field_name} 必须是合法的 UUID 字符串") from exc


# ==================== items 判别式结构 ====================


class PrescriptionItemIn(BaseModel):
    """关联一张处方。金额由服务端从处方 total_amount 求值，入参不含 amount。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    type: Literal["prescription"]
    prescription_id: str = Field(min_length=1, description="处方 id")

    @field_validator("prescription_id")
    @classmethod
    def _check_rx(cls, v: str) -> str:
        _as_uuid(v, "prescription_id")
        return v


class ExamItemIn(BaseModel):
    """关联一张检查申请单。金额由服务端从 fee_amount 求值，入参不含 amount。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    type: Literal["exam"]
    exam_request_id: str = Field(min_length=1, description="检查申请单 id")

    @field_validator("exam_request_id")
    @classmethod
    def _check_exam(cls, v: str) -> str:
        _as_uuid(v, "exam_request_id")
        return v


class OtherItemIn(BaseModel):
    """自定义费用项（无源单据），由客户端定价，单位分且必须为正整数。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    type: Literal["other"]
    name: str = Field(min_length=1, max_length=100, description="项目名称，如 挂号费")
    # ge=1：不收零元项。金额为 0 的 other 项没有业务含义，只会污染应收汇总；
    # gt 上界防「输入一亿」这种误操作被当成合法高价项落库。
    amount: int = Field(ge=1, le=100_000_000, description="金额，单位分（正整数）")


# 判别式联合：Pydantic 按 type 字面量路由到对应分支，未知 type 直接 422。
ChargeItemIn = Annotated[
    Union[PrescriptionItemIn, ExamItemIn, OtherItemIn],
    Field(discriminator="type"),
]


# ==================== 请求 ====================


class ChargeOrderCreateRequest(BaseModel):
    """医生为某次就诊创建收费单（status=pending）。

    刻意**不包含 visit_id / patient_id**：就诊归属与患者身份应由 service 从
    Visit 反查得到（见 billing_service.create_charge_order），
    而非由客户端指定——否则任何人都能把收费单挂到别人的就诊或患者身上，
    行级隔离会失去意义。请求体只需给出 visit_id 定位「哪次就诊要收费」，
    patient_id 从该 visit 带出。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    visit_id: str = Field(min_length=1, description="就诊 id")
    items: list[ChargeItemIn] = Field(
        min_length=1,
        max_length=MAX_CHARGE_ITEMS,
        description="费用项列表，至少一项；处方/检查项金额由服务端求值",
    )

    @field_validator("visit_id")
    @classmethod
    def _check_visit(cls, v: str) -> str:
        _as_uuid(v, "visit_id")
        return v


class PayRequest(BaseModel):
    """收费员确认收款（pending→paid）。

    本项目不接真实支付渠道：``channel`` 默认 mock，接口直接把 paid_amount
    置为应收全额并生成小票号。金额不接受客户端传入——实收恒等于应收全额，
    部分缴费场景不在本演示范围内（见 billing_service.pay_charge_order docstring）。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    channel: str = Field(
        default="mock",
        description=f"支付渠道，取值 {'/'.join(PAYMENT_CHANNEL_VALUES)}",
    )

    @field_validator("channel")
    @classmethod
    def _check_channel(cls, v: str) -> str:
        allowed = set(PAYMENT_CHANNEL_VALUES)
        if v not in allowed:
            raise ValueError(f"channel 非法，允许值：{'、'.join(sorted(allowed))}")
        return v


class RefundRequest(BaseModel):
    """收费员退费（paid→refunded）。reason 必填：资金流出必须留痕。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    reason: str = Field(
        min_length=1,
        max_length=500,
        description="退费原因（必填，写入 refund_reason 供审计）",
    )


# ==================== 响应 ====================


class ChargeItemOut(BaseModel):
    """费用项出参。保留源单据 id 与规范化后的金额，便于前端逐行显示。"""

    type: str
    prescription_id: str | None = None
    exam_request_id: str | None = None
    name: str | None = None
    amount: int = Field(description="该项金额，单位分")


class ChargeOrderOut(BaseModel):
    """收费单（列表行 / 详情通用）。"""

    id: str
    visit_id: str
    patient_id: str
    status: str = Field(description="pending/paid/refunded/cancelled")
    items: list[ChargeItemOut] = Field(default_factory=list)
    total_amount: int = Field(description="应收合计，单位分")
    paid_amount: int = Field(description="实收合计，单位分；退费后为 0")
    payment_channel: str | None = None
    cashier_id: str | None = None
    paid_at: datetime | None = None
    refunded_at: datetime | None = None
    refund_reason: str | None = None
    receipt_no: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    allowed_transitions: list[str] = Field(
        default_factory=list,
        description="当前状态可迁移到的目标状态，供前端禁用不可能成功的按钮",
    )


class ChargeOrderListResponse(BaseModel):
    items: list[ChargeOrderOut]
    total: int
    page: int
    page_size: int


__all__ = [
    "ChargeItemIn",
    "ChargeItemOut",
    "ChargeOrderCreateRequest",
    "ChargeOrderListResponse",
    "ChargeOrderOut",
    "ExamItemIn",
    "OtherItemIn",
    "PayRequest",
    "PrescriptionItemIn",
    "RefundRequest",
    "CHARGE_STATUS_VALUES",
    "PAYMENT_CHANNEL_VALUES",
]
