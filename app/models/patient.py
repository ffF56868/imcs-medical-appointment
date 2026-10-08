"""患者档案模型。"""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import (
    CheckConstraint,
    Date,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.auth import User
from app.models.base import Base, TimestampMixin, UUIDPrimaryKey
from app.models.types import JSONBText, TextArray


class PatientProfile(UUIDPrimaryKey, TimestampMixin, Base):
    """患者健康档案。

    与 ``User`` 一对一分离的原因：账号信息（登录名、密码）与健康信息
    （过敏史、慢性病）的访问控制级别不同，分表后便于对健康数据做
    更严格的字段级脱敏与审计。

    ``allergy_history`` 与 ``chronic_diseases`` 用数组类型而非逗号分隔字符串，
    因为处方安全校验要做精确集合匹配，字符串切分会在值含分隔符时产生歧义。
    """

    __tablename__ = "patient_profiles"
    __table_args__ = (
        CheckConstraint(
            "gender IS NULL OR gender IN ('M', 'F')",
            name="ck_patient_gender_valid",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    gender: Mapped[str | None] = mapped_column(String(1))
    birth_date: Mapped[date | None] = mapped_column(Date)
    # 身份证号属于敏感信息，展示时必须脱敏，落库前应加密
    id_card_no: Mapped[str | None] = mapped_column(String(18))
    blood_type: Mapped[str | None] = mapped_column(String(5))
    allergy_history: Mapped[list[str] | None] = mapped_column(TextArray())
    chronic_diseases: Mapped[list[str] | None] = mapped_column(TextArray())
    # 长期用药、手术史等非结构化补充信息
    medical_notes: Mapped[str | None] = mapped_column(Text)
    emergency_contact: Mapped[dict | None] = mapped_column(JSONBText())

    user: Mapped["User"] = relationship(lazy="selectin")

    @property
    def age(self) -> int | None:
        """按出生日期计算周岁年龄。分诊与用药剂量校验会用到。"""
        if self.birth_date is None:
            return None
        today = date.today()
        years = today.year - self.birth_date.year
        if (today.month, today.day) < (self.birth_date.month, self.birth_date.day):
            years -= 1
        return years
