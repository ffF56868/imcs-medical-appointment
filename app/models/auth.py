"""身份、角色与权限模型（RBAC）。

权限模型：``Role`` - ``Permission`` 多对多，通过 ``RolePermission`` 关联表。
六类角色固定由种子数据写入，不支持运行时动态增删——医疗系统的角色边界
应当是稳定的，动态角色只会让权限审计变得困难。

行级数据隔离（医生只能看自己患者、药师只能看待发药处方）在 service 层实现，
不用 PostgreSQL Row-Level Security：RLS 安全性更高但调试困难，面试项目里
service 层过滤配合单元测试更直观、更易讲解。
"""

from __future__ import annotations

import enum
import uuid
from datetime import date

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKey
from app.models.types import TextArray


class RoleName(str, enum.Enum):
    """系统六类角色。值即数据库中 roles.name 存储的内容。"""

    PATIENT = "patient"
    DOCTOR = "doctor"
    LAB_TECH = "lab_tech"  # 检验科
    CASHIER = "cashier"  # 收费员
    PHARMACIST = "pharmacist"  # 药师
    ADMIN = "admin"  # 超级管理员


class Department(UUIDPrimaryKey, TimestampMixin, Base):
    """科室。分诊结果指向科室，医生隶属于科室。"""

    __tablename__ = "departments"

    name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    # 短代码用于分诊 prompt 与前端展示，如 CARDIO / NEURO / LAB
    code: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    # 常见症状关键词，供规则兜底分诊与召回使用
    symptom_keywords: Mapped[list[str] | None] = mapped_column(TextArray())
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("departments.id"),
    )


class Role(Base):
    """角色。主键用固定小整数，便于种子数据与权限矩阵可读性。"""

    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(30), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)

    users: Mapped[list["User"]] = relationship(back_populates="role")
    permissions: Mapped[list["Permission"]] = relationship(
        secondary="role_permissions",
        back_populates="roles",
        lazy="selectin",
    )


class Permission(Base):
    """权限项：资源 + 动作，如 (appointment, create)。"""

    __tablename__ = "permissions"
    __table_args__ = (UniqueConstraint("resource", "action", name="uq_permission_resource_action"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    resource: Mapped[str] = mapped_column(String(50), nullable=False)
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    roles: Mapped[list["Role"]] = relationship(
        secondary="role_permissions",
        back_populates="permissions",
    )


class RolePermission(Base):
    """角色-权限关联表。"""

    __tablename__ = "role_permissions"

    role_id: Mapped[int] = mapped_column(
        ForeignKey("roles.id", ondelete="CASCADE"),
        primary_key=True,
    )
    permission_id: Mapped[int] = mapped_column(
        ForeignKey("permissions.id", ondelete="CASCADE"),
        primary_key=True,
    )


class User(UUIDPrimaryKey, TimestampMixin, Base):
    """统一用户表。六类角色共用一张表 + role_id 外键。

    不做多表继承：角色间共享字段远多于差异字段（用户名、密码、姓名、手机号），
    拆表只会带来 JOIN 成本。角色特有信息放在专门的档案表里
    （患者 -> PatientProfile，医生 -> DoctorSchedule）。
    """

    __tablename__ = "users"
    __table_args__ = (
        Index("idx_users_role_active", "role_id", "is_active"),
        Index("idx_users_department", "department_id"),
    )

    username: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    real_name: Mapped[str] = mapped_column(String(50), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(20), unique=True)
    email: Mapped[str | None] = mapped_column(String(200), unique=True)
    role_id: Mapped[int] = mapped_column(
        ForeignKey("roles.id"),
        nullable=False,
    )
    # 医生/检验科/药师必填；患者与管理员为空
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("departments.id"),
    )
    # 医生职称，用于患者选医生时展示
    title: Mapped[str | None] = mapped_column(String(50))
    specialty: Mapped[str | None] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    role: Mapped["Role"] = relationship(back_populates="users", lazy="selectin")
    department: Mapped["Department | None"] = relationship(lazy="selectin")


class DoctorSchedule(UUIDPrimaryKey, TimestampMixin, Base):
    """医生排班与号源。

    并发控制用乐观锁 ``version`` 字段而非 SELECT FOR UPDATE：
    高并发下悲观锁容易造成阻塞与死锁，乐观锁冲突时重试即可。
    扣减语句形如::

        UPDATE doctor_schedules
           SET booked_slots = booked_slots + 1, version = version + 1
         WHERE id = :id AND version = :version AND booked_slots < total_slots

    受影响行数为 0 说明版本已变或号已满，由 service 层决定重试或返回满号。
    """

    __tablename__ = "doctor_schedules"
    __table_args__ = (
        UniqueConstraint(
            "doctor_id",
            "schedule_date",
            "time_slot",
            name="uq_schedule_doctor_date_slot",
        ),
        Index("idx_schedule_date_status", "schedule_date", "status"),
        CheckConstraint("booked_slots >= 0", name="ck_booked_slots_non_negative"),
        CheckConstraint("total_slots > 0", name="ck_total_slots_positive"),
        CheckConstraint("booked_slots <= total_slots", name="ck_booked_within_total"),
    )

    doctor_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    department_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("departments.id"),
        nullable=False,
    )
    schedule_date: Mapped[date] = mapped_column(Date, nullable=False)
    # 时段文本，如 "08:00-08:30"；用文本而非起止两列，便于按半天/整段排班
    time_slot: Mapped[str] = mapped_column(String(20), nullable=False)
    slot_period: Mapped[str] = mapped_column(
        String(10),
        default="am",
        nullable=False,
        comment="am / pm，便于前端按上下午分组展示",
    )
    total_slots: Mapped[int] = mapped_column(Integer, default=30, nullable=False)
    booked_slots: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # 乐观锁版本号
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    # open / full / closed
    status: Mapped[str] = mapped_column(String(10), default="open", nullable=False)
    registration_fee: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="挂号费，单位分，避免浮点误差",
    )

    doctor: Mapped["User"] = relationship(lazy="selectin")

    @property
    def remaining_slots(self) -> int:
        return max(0, self.total_slots - self.booked_slots)
