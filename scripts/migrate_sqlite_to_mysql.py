"""Copy the existing SQLite data into MySQL without deleting the source file."""

import os
import sys
from pathlib import Path

# Allow both `python scripts/...` and `python -m scripts...` inside the container.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlmodel import Session, SQLModel, select

from app.database import build_engine, create_db_and_tables
from app.legacy.models import (
    AnswerFeedback,
    ChatMessage,
    Condition,
    Drug,
    KnowledgeDocument,
    KnowledgeIndexState,
    KnowledgeRebuildJob,
    KnowledgeReviewLog,
    KnowledgeSnapshot,
    LoginAttempt,
    RAGEvaluationCase,
    RAGEvaluationRun,
    RAGASAutoEvaluationRun,
    RAGQualityEvaluationRun,
    RAGRequestMetric,
    SecurityAuditLog,
    User,
    UserSession,
)


SOURCE_DATABASE_URL = os.getenv(
    "SOURCE_DATABASE_URL",
    "sqlite:////app/data/medical_health.db",
)
TARGET_DATABASE_URL = os.getenv(
    "TARGET_DATABASE_URL",
    "mysql+pymysql://medical_user:medical_password@mysql:3306/medical_health?charset=utf8mb4",
)

MIGRATION_MODELS = [
    Condition,
    Drug,
    KnowledgeDocument,
    ChatMessage,
    AnswerFeedback,
    KnowledgeIndexState,
    KnowledgeSnapshot,
    KnowledgeRebuildJob,
    KnowledgeReviewLog,
    RAGEvaluationCase,
    RAGEvaluationRun,
    RAGASAutoEvaluationRun,
    RAGQualityEvaluationRun,
    RAGRequestMetric,
    User,
    UserSession,
    LoginAttempt,
    SecurityAuditLog,
]


def copy_table(source_session: Session, target_session: Session, model: type[SQLModel]) -> int:
    rows = source_session.exec(select(model)).all()
    for row in rows:
        target_session.merge(model.model_validate(row.model_dump()))
    target_session.commit()
    return len(rows)


def main() -> None:
    source_engine = build_engine(SOURCE_DATABASE_URL)
    target_engine = build_engine(TARGET_DATABASE_URL)
    create_db_and_tables(source_engine)
    create_db_and_tables(target_engine)

    counts: dict[str, int] = {}
    with Session(source_engine) as source_session, Session(target_engine) as target_session:
        for model in MIGRATION_MODELS:
            counts[model.__tablename__] = copy_table(
                source_session,
                target_session,
                model,
            )

    print({"source_preserved": True, "copied_rows": counts})


if __name__ == "__main__":
    main()
