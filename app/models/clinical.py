"""临床业务模型：预约、就诊、检查申请与检查报告。

这四个实体构成核心闭环的前半段：
挂号(appointment) -> 接诊(visit) -> 开检查(exam_request) -> 出报告(exam_report)

状态机流转合法性在 service 层用 ``StatusMachine`` 校验，不在数据库做触发器。
每个状态枚举旁边都定义了 ``TRANSITIONS`` 转换矩阵，是唯一的合法流转来源。
"""

from __future__ import annotations

import enum
import uuid
from datetime import date, datetime

from sqlalchemy import (
    Date,
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

from app.models.auth import DoctorSchedule, User
from app.models.base import Base, TimestampMixin, UUIDPrimaryKey
from app.models.types import JSONBText, TextArray


# ==================== 状态枚举与转换矩阵 ====================


class AppointmentStatus(str, enum.Enum):
    """挂号/预约单状态。"""

    BOOKED = "booked"  # 已预约
    CHECKED_IN = "checked_in"  # 已到院签到
    IN_PROGRESS = "in_progress"  # 接诊中
    COMPLETED = "completed"  # 已完成
    CANCELLED = "cancelled"  # 已取消
    NO_SHOW = "no_show"  # 爽约


# 唯一合法流转来源。service 层的状态机校验以此为准。
APPOINTMENT_TRANSITIONS: dict[str, set[str]] = {
    AppointmentStatus.BOOKED.value: {
        AppointmentStatus.CHECKED_IN.value,
        AppointmentStatus.CANCELLED.value,
        AppointmentStatus.NO_SHOW.value,
    },
    AppointmentStatus.CHECKED_IN.value: {
        AppointmentStatus.IN_PROGRESS.value,
        AppointmentStatus.NO_SHOW.value,
        AppointmentStatus.CANCELLED.value,
    },
    AppointmentStatus.IN_PROGRESS.value: {AppointmentStatus.COMPLETED.value},
    AppointmentStatus.COMPLETED.value: set(),
    AppointmentStatus.CANCELLED.value: set(),
    AppointmentStatus.NO_SHOW.value: set(),
}


class VisitStatus(str, enum.Enum):
    """就诊记录状态。"""

    IN_PROGRESS = "in_progress"  # 接诊中，病历未定稿
    DOCUMENTED = "documented"  # 病历已书写
    CLOSED = "closed"  # 就诊结束


VISIT_TRANSITIONS: dict[str, set[str]] = {
    VisitStatus.IN_PROGRESS.value: {
        VisitStatus.DOCUMENTED.value,
        VisitStatus.CLOSED.value,
    },
    VisitStatus.DOCUMENTED.value: {VisitStatus.CLOSED.value},
    VisitStatus.CLOSED.value: set(),
}


class ExamRequestStatus(str, enum.Enum):
    """检查申请单状态。"""

    ORDERED = "ordered"  # 医生已开单
    RECEIVED = "received"  # 检验科已接收
    PROCESSING = "processing"  # 检查进行中
    REPORTED = "reported"  # 报告已出
    REVIEWED = "reviewed"  # 报告已审核（AI 解读完成后）


EXAM_REQUEST_TRANSITIONS: dict[str, set[str]] = {
    ExamRequestStatus.ORDERED.value: {
        ExamRequestStatus.RECEIVED.value,
        ExamRequestStatus.PROCESSING.value,
    },
    ExamRequestStatus.RECEIVED.value: {ExamRequestStatus.PROCESSING.value},
    ExamRequestStatus.PROCESSING.value: {ExamRequestStatus.REPORTED.value},
    ExamRequestStatus.REPORTED.value: {ExamRequestStatus.REVIEWED.value},
    ExamRequestStatus.REVIEWED.value: set(),
}


class Urgency(str, enum.Enum):
    """检查/就诊紧急度。分诊工作流的输出之一。"""

    ROUTINE = "routine"  # 常规
    URGENT = "urgent"  # 紧急
    STAT = "stat"  # 立即（危急值）


# ==================== 预约 ====================


class Appointment(UUIDPrimaryKey, TimestampMixin, Base):
    """挂号/预约单。一次挂号锁定一个排班时段的一个号源。"""

    __tablename__ = "appointments"
    __table_args__ = (
        Index("idx_appt_patient_date", "patient_id", "appointment_date"),
        Index("idx_appt_doctor_date_status", "doctor_id", "appointment_date", "status"),
        # 同一患者同一时段只能有一个有效预约，防止重复挂号。
        # 用部分唯一索引：只约束未取消的预约，取消后可再挂。
        Index(
            "uq_appt_patient_slot_active",
            "patient_id",
            "schedule_id",
            unique=True,
            postgresql_where=text("status NOT IN ('cancelled', 'no_show')"),
        ),
    )

    patient_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("patient_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    doctor_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )
    department_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("departments.id"),
        nullable=False,
    )
    schedule_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("doctor_schedules.id"),
        nullable=False,
    )
    appointment_date: Mapped[date] = mapped_column(Date, nullable=False)
    time_slot: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        default=AppointmentStatus.BOOKED.value,
        nullable=False,
    )
    # self（患者自主挂号）/ triage_referral（智能分诊引导挂号）
    source: Mapped[str] = mapped_column(String(20), default="self", nullable=False)
    # 分诊摘要，来源可追溯：记录 AI 建议的科室与紧急度依据
    triage_summary: Mapped[str | None] = mapped_column(Text)
    # 关联的分诊结果 ID，用于评估分诊到挂号的转化与准确率
    triage_result_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    chief_complaint: Mapped[str | None] = mapped_column(Text, comment="患者主诉")
    registration_fee: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="挂号费，单位分"
    )
    checked_in_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    doctor: Mapped["User"] = relationship(lazy="selectin")
    schedule: Mapped["DoctorSchedule"] = relationship(lazy="selectin")


# ==================== 就诊 ====================


class Visit(UUIDPrimaryKey, TimestampMixin, Base):
    """就诊记录 / 病历。一次接诊对应一条，与预约一对一。"""

    __tablename__ = "visits"
    __table_args__ = (
        Index("idx_visit_doctor_status", "doctor_id", "status"),
        Index("idx_visit_patient", "patient_id"),
    )

    appointment_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("appointments.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
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
    # 病历核心字段：主诉、现病史、既往史、体格检查、诊断
    chief_complaint: Mapped[str] = mapped_column(Text, nullable=False)
    present_illness: Mapped[str | None] = mapped_column(Text)
    past_history: Mapped[str | None] = mapped_column(Text)
    physical_exam: Mapped[str | None] = mapped_column(Text)
    diagnosis: Mapped[str | None] = mapped_column(Text)
    # ICD-10 编码，用数组支持多诊断
    icd_codes: Mapped[list[str] | None] = mapped_column(TextArray())
    treatment_advice: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(20),
        default=VisitStatus.IN_PROGRESS.value,
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    doctor: Mapped["User"] = relationship(lazy="selectin")


# ==================== 检查申请与报告 ====================


class ExamRequest(UUIDPrimaryKey, TimestampMixin, Base):
    """检查申请单。医生开出，检验科执行。"""

    __tablename__ = "exam_requests"
    __table_args__ = (
        Index("idx_exam_request_visit", "visit_id"),
        Index("idx_exam_request_status", "status"),
        Index("idx_exam_request_lab_status", "status", "created_at"),
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
    # 检查项目类型，如 blood_routine / liver_function / ct_chest
    exam_type: Mapped[str] = mapped_column(String(50), nullable=False)
    exam_name: Mapped[str] = mapped_column(String(100), nullable=False)
    body_part: Mapped[str | None] = mapped_column(String(100))
    clinical_info: Mapped[str | None] = mapped_column(Text, comment="临床诊断信息，供检验科参考")
    urgency: Mapped[str] = mapped_column(
        String(10),
        default=Urgency.ROUTINE.value,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default=ExamRequestStatus.ORDERED.value,
        nullable=False,
    )
    # 是否已生成收费项
    is_charged: Mapped[bool] = mapped_column(default=False, nullable=False)
    fee_amount: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="检查费，单位分"
    )

    visit: Mapped["Visit"] = relationship(lazy="selectin")


class ExamReport(UUIDPrimaryKey, TimestampMixin, Base):
    """检查报告。检验科录入原始结果，AI 生成解读。

    ``structured_data`` 存结构化指标（如 {白细胞: {value, unit, ref_range, flag}}），
    ``abnormal_flags`` 存异常项标记数组，供 AI 报告解读工作流消费。
    AI 解读结果与其使用的模型、置信度一并落库，保证可追溯。
    """

    __tablename__ = "exam_reports"
    __table_args__ = (
        # 一张申请单只能有一份报告
        UniqueConstraint("request_id", name="uq_exam_report_request"),
        Index("idx_exam_report_lab_tech", "lab_tech_id"),
    )

    request_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("exam_requests.id", ondelete="CASCADE"),
        nullable=False,
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("patient_profiles.id"),
        nullable=False,
    )
    lab_tech_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )
    # 原始报告文本 / 结构化录入内容
    raw_content: Mapped[str] = mapped_column(Text, nullable=False)
    # 结构化指标，JSONB 存储，可按内部字段过滤
    structured_data: Mapped[dict | None] = mapped_column(JSONBText())
    # 异常项标记，如 ["白细胞偏高", "血红蛋白偏低"]
    abnormal_flags: Mapped[list[str] | None] = mapped_column(TextArray())
    conclusion: Mapped[str | None] = mapped_column(Text, comment="检验科结论")

    # ---- AI 解读结果 ----
    ai_interpretation: Mapped[str | None] = mapped_column(Text)
    ai_summary: Mapped[str | None] = mapped_column(Text, comment="AI 解读摘要")
    ai_suggestions: Mapped[list[str] | None] = mapped_column(
        TextArray(), comment="AI 给出的后续建议"
    )
    ai_confidence: Mapped[float | None] = mapped_column(Numeric(4, 3))
    ai_model: Mapped[str | None] = mapped_column(String(50))
    ai_provider: Mapped[str | None] = mapped_column(String(20))
    # 关联的 AI 任务，用于查解读链路与 token 消耗
    ai_task_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    ai_status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False,
        comment="pending / processing / completed / failed / skipped",
    )
    interpreted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    request: Mapped["ExamRequest"] = relationship(lazy="selectin")
    lab_tech: Mapped["User"] = relationship(lazy="selectin")
