"""处方与发药模型。

处方安全校验（过敏史冲突、药物相互作用）在 service 层执行，
拦截结果通过 ``AllergyConflictError`` / ``DrugInteractionError`` 抛出，
而不是仅靠数据库约束——这类校验需要联合患者档案与药品知识库判断。
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.auth import User
from app.models.base import Base, TimestampMixin, UUIDPrimaryKey
from app.models.types import JSONBText


class PrescriptionStatus(str, enum.Enum):
    """处方状态。"""

    DRAFT = "draft"  # 医生草稿
    CONFIRMED = "confirmed"  # 已开具，待收费
    PAID = "paid"  # 已缴费，待发药
    DISPENSED = "dispensed"  # 已发药
    CANCELLED = "cancelled"  # 已作废


PRESCRIPTION_TRANSITIONS: dict[str, set[str]] = {
    PrescriptionStatus.DRAFT.value: {
        PrescriptionStatus.CONFIRMED.value,
        PrescriptionStatus.CANCELLED.value,
    },
    PrescriptionStatus.CONFIRMED.value: {
        PrescriptionStatus.PAID.value,
        PrescriptionStatus.CANCELLED.value,
    },
    # 已缴费后不能直接作废，需走退费流程
    PrescriptionStatus.PAID.value: {PrescriptionStatus.DISPENSED.value},
    PrescriptionStatus.DISPENSED.value: set(),
    PrescriptionStatus.CANCELLED.value: set(),
}


class Prescription(UUIDPrimaryKey, TimestampMixin, Base):
    """处方主单。"""

    __tablename__ = "prescriptions"
    __table_args__ = (
        Index("idx_prescription_visit", "visit_id"),
        Index("idx_prescription_status_created", "status", "created_at"),
        # 药师工作台最高频查询是「已缴费待发药」列表。做成部分索引后
        # 索引只包含 paid 状态的行，体积远小于覆盖全表的复合索引。
        Index(
            "idx_prescription_pending_dispense",
            "created_at",
            postgresql_where=text(f"status = '{PrescriptionStatus.PAID.value}'"),
        ),
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
    doctor_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default=PrescriptionStatus.DRAFT.value,
        nullable=False,
    )
    diagnosis: Mapped[str | None] = mapped_column(Text, comment="处方对应的临床诊断")
    notes: Mapped[str | None] = mapped_column(Text, comment="医嘱备注")
    # 处方总金额，单位分。由明细汇总，收费时校验一致性
    total_amount: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="处方总金额，单位分"
    )
    # 安全校验记录：拦截了哪些风险、是否被医生强制确认
    safety_checks: Mapped[dict | None] = mapped_column(
        JSONBText(), comment="过敏史与相互作用校验结果"
    )
    safety_override_reason: Mapped[str | None] = mapped_column(
        Text, comment="医生强制开具时填写的理由"
    )
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    doctor: Mapped["User"] = relationship(lazy="selectin")
    items: Mapped[list["PrescriptionItem"]] = relationship(
        back_populates="prescription",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    dispensing: Mapped["DispensingRecord | None"] = relationship(
        back_populates="prescription",
        uselist=False,
    )

    def recalculate_total(self) -> int:
        """按明细重算总金额，返回并写入单位为「分」的整数。

        金额一律用 ``Decimal`` 计算再取整到分，不用 float——
        0.1 + 0.2 != 0.3 这类浮点误差在收费场景是不可接受的。
        ``unit_price`` 存的是元，乘 100 转分，**逐项**四舍五入到整分。
        取整必须发生在 quantize 这一步（quantum=1 分）：早先版本先量化到
        0.01 分再 ``int()`` 截断，0.495 元会得到 49 分而不是 50 分，
        与 docstring 承诺的「四舍五入到整分」不符——收费金额差一分都对不上账。
        """
        one_cent = Decimal("1")
        total = Decimal(0)
        for item in self.items:
            # 单价（元）× 数量（可含小数，如 1.5 盒）→ 元，再转分
            amount_yuan = Decimal(str(item.unit_price)) * Decimal(str(item.quantity))
            total += (amount_yuan * 100).quantize(one_cent, rounding=ROUND_HALF_UP)
        self.total_amount = int(total)
        return self.total_amount


class PrescriptionItem(UUIDPrimaryKey, Base):
    """处方明细行。"""

    __tablename__ = "prescription_items"
    __table_args__ = (
        Index("idx_prescription_item_rx", "prescription_id"),
        CheckConstraint("quantity > 0", name="ck_rx_item_quantity_positive"),
        # 同一处方内不应重复开同一药品
        UniqueConstraint("prescription_id", "drug_code", name="uq_rx_item_drug"),
    )

    prescription_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("prescriptions.id", ondelete="CASCADE"),
        nullable=False,
    )
    drug_code: Mapped[str] = mapped_column(String(30), nullable=False)
    drug_name: Mapped[str] = mapped_column(String(100), nullable=False)
    specification: Mapped[str | None] = mapped_column(String(100), comment="规格，如 0.25g*24片")
    # 单次剂量，如 "1片"、"5ml"
    dosage: Mapped[str | None] = mapped_column(String(50))
    # 用药频次，如 tid（每日三次）、bid（每日两次）
    frequency: Mapped[str | None] = mapped_column(String(30))
    # 用药途径，如 口服 / 静脉滴注
    route: Mapped[str | None] = mapped_column(String(30))
    duration_days: Mapped[int | None] = mapped_column(Integer)
    quantity: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False, comment="单位，如 盒/瓶/支")
    # 单价，单位元。用 Decimal 避免浮点误差
    unit_price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False, default=0)
    usage_instructions: Mapped[str | None] = mapped_column(Text, comment="用药指导")

    prescription: Mapped["Prescription"] = relationship(back_populates="items")


class DispensingRecord(UUIDPrimaryKey, TimestampMixin, Base):
    """发药记录。药师核对处方后写入。"""

    __tablename__ = "dispensing_records"
    __table_args__ = (
        UniqueConstraint("prescription_id", name="uq_dispensing_prescription"),
        Index("idx_dispensing_pharmacist", "pharmacist_id"),
    )

    prescription_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("prescriptions.id", ondelete="CASCADE"),
        nullable=False,
    )
    pharmacist_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )
    # 药师是否完成了处方审核（四查十对）
    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    verification_notes: Mapped[str | None] = mapped_column(Text)
    dispensed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)

    prescription: Mapped["Prescription"] = relationship(back_populates="dispensing")
    pharmacist: Mapped["User"] = relationship(lazy="selectin")
