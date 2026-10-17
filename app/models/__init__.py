"""IMCS 数据模型包（SQLAlchemy 2.0 声明式）。

按业务域拆分模块，全部继承同一个 ``Base``，Alembic 通过 ``Base.metadata``
发现所有表。主键统一用 UUID，避免自增 ID 暴露业务量且便于分布式生成。
"""

from app.models.base import Base, TimestampMixin, UUIDPrimaryKey
from app.models.auth import (
    Department,
    DoctorSchedule,
    Permission,
    Role,
    RolePermission,
    User,
)
from app.models.patient import PatientProfile
from app.models.clinical import (
    Appointment,
    AppointmentStatus,
    ExamReport,
    ExamRequest,
    ExamRequestStatus,
    Visit,
    VisitStatus,
)
from app.models.pharmacy import (
    DispensingRecord,
    Prescription,
    PrescriptionItem,
    PrescriptionStatus,
)
from app.models.billing import ChargeOrder, ChargeOrderStatus
from app.models.knowledge import (
    KnowledgeChunk,
    KnowledgeDocument,
    VectorEmbedding,
)
from app.models.evaluation import (
    EvalCase,
    EvalKind,
    EvalRun,
    EvalRunStatus,
)
from app.models.health_qa import HealthQALog
from app.models.ai import (
    AIMetric,
    AITask,
    AITaskStatus,
    LLMCallLog,
    RAGRetrievalLog,
    TriageResult,
    WorkflowTrace,
)

__all__ = [
    "Base",
    "TimestampMixin",
    "UUIDPrimaryKey",
    "Department",
    "Permission",
    "Role",
    "RolePermission",
    "User",
    "DoctorSchedule",
    "PatientProfile",
    "Appointment",
    "AppointmentStatus",
    "ExamReport",
    "ExamRequest",
    "ExamRequestStatus",
    "Visit",
    "VisitStatus",
    "DispensingRecord",
    "Prescription",
    "PrescriptionItem",
    "PrescriptionStatus",
    "ChargeOrder",
    "ChargeOrderStatus",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "VectorEmbedding",
    "EvalCase",
    "EvalKind",
    "EvalRun",
    "EvalRunStatus",
    "HealthQALog",
    "AIMetric",
    "AITask",
    "AITaskStatus",
    "LLMCallLog",
    "RAGRetrievalLog",
    "TriageResult",
    "WorkflowTrace",
]
