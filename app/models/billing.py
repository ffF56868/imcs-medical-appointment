"""收费模型。

收费单聚合一次就诊产生的所有费用项（检查费 + 药费）。
本项目为面试演示定位，**不接真实支付渠道**：支付通过 ``mock_pay``
接口直接置为已支付，并在 ``payment_channel`` 字段标记为模拟支付。
金额一律用「分」为单位存整数，避免浮点误差。
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.auth import User
from app.models.base import Base, TimestampMixin, UUIDPrimaryKey
from app.models.types import JSONBText


class ChargeOrderStatus(str, enum.Enum):
    """收费单状态。"""

    PENDING = "pending"  # 待缴费
    PAID = "paid"  # 已缴费
    REFUNDED = "refunded"  # 已退费
    CANCELLED = "cancelled"  # 已作废


CHARGE_ORDER_TRANSITIONS: dict[str, set[str]] = {
    ChargeOrderStatus.PENDING.value: {
        ChargeOrderStatus.PAID.value,
        ChargeOrderStatus.CANCELLED.value,
    },
    # 已缴费只能退费，不能直接作废
    ChargeOrderStatus.PAID.value: {ChargeOrderStatus.REFUNDED.value},
    ChargeOrderStatus.REFUNDED.value: set(),
    ChargeOrderStatus.CANCELLED.value: set(),
}


class ChargeOrder(UUIDPrimaryKey, TimestampMixin, Base):
    """收费单。一次就诊可有多张（如检查与药品分开收费）。"""

    __tablename__ = "charge_orders"
    __table_args__ = (
        Index("idx_charge_visit", "visit_id"),
        Index("idx_charge_patient_status", "patient_id", "status"),
        # 收费员工作台按「待缴费」筛选，最高频查询
        Index(
            "idx_charge_pending",
            "created_at",
            postgresql_where=text(f"status = '{ChargeOrderStatus.PENDING.value}'"),
        ),
        CheckConstraint("total_amount >= 0", name="ck_charge_amount_non_negative"),
    )

    visit_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("visits.id", ondelete="CASCADE"),
        nullable=False,
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("patient_profiles.id"),
        nullable=False,
    )
    # 费用明细：[{type: 'exam'|'drug'|'registration', ref_id, name, amount}]
    items: Mapped[list] = mapped_column(JSONBText(), nullable=False, default=list)
    # 总金额，单位分
    total_amount: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 实收金额，单位分。退费后为 0
    paid_amount: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(
        String(20),
        default=ChargeOrderStatus.PENDING.value,
        nullable=False,
    )
    # mock / cash / insurance，本项目只实现 mock
    payment_channel: Mapped[str | None] = mapped_column(String(20))
    cashier_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id"),
    )
    paid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    refunded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    refund_reason: Mapped[str | None] = mapped_column(Text)
    receipt_no: Mapped[str | None] = mapped_column(String(40), unique=True)

    cashier: Mapped["User | None"] = relationship(lazy="selectin")
