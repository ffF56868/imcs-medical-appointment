from datetime import UTC, datetime

from sqlalchemy import Text
from sqlmodel import Field, SQLModel


class Condition(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, max_length=100)
    symptoms: str = Field(sa_type=Text)
    treatment: str = Field(sa_type=Text)
    source: str = Field(default="未标注来源", max_length=200)
    source_url: str | None = Field(default=None, max_length=2000)
    source_tier: str = Field(default="unverified", max_length=20)
    knowledge_base_id: str = Field(default="global", index=True, max_length=100)
    visibility: str = Field(default="public", index=True, max_length=20)
    updated_at: datetime | None = Field(default=None)


class Drug(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, max_length=100)
    effects: str = Field(sa_type=Text)
    instructions: str = Field(sa_type=Text)
    source: str = Field(default="未标注来源", max_length=200)
    source_url: str | None = Field(default=None, max_length=2000)
    source_tier: str = Field(default="unverified", max_length=20)
    knowledge_base_id: str = Field(default="global", index=True, max_length=100)
    visibility: str = Field(default="public", index=True, max_length=20)
    updated_at: datetime | None = Field(default=None)


class KnowledgeDocument(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    title: str = Field(index=True, max_length=200)
    content: str = Field(sa_type=Text)
    source: str = Field(default="未标注来源", max_length=200)
    source_url: str | None = Field(default=None, max_length=2000)
    source_tier: str = Field(default="unverified", max_length=20)
    owner_user_id: int | None = Field(default=None, index=True)
    knowledge_base_id: str = Field(default="global", index=True, max_length=100)
    visibility: str = Field(default="public", index=True, max_length=20)
    page_number: int | None = Field(default=None, index=True, ge=1)
    updated_at: datetime | None = Field(default=None)


class ChatMessage(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int | None = Field(default=None, index=True)
    conversation_id: str = Field(index=True, max_length=100)
    role: str = Field(max_length=20)
    content: str = Field(sa_type=Text)
    response_metadata_json: str = Field(default="{}", sa_type=Text)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
    )


class AnswerFeedback(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    assistant_message_id: int = Field(index=True, unique=True)
    helpful: bool
    reason: str | None = Field(default=None, max_length=1000)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
    )


class KnowledgeIndexState(SQLModel, table=True):
    id: int = Field(default=1, primary_key=True)
    content_hash: str = Field(max_length=64)
    document_count: int
    vector_store_type: str = Field(default="milvus", max_length=30)
    vector_count: int = Field(default=0)
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class KnowledgeSnapshot(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    content_hash: str = Field(index=True, max_length=64)
    document_count: int
    reason: str = Field(default="rebuild", max_length=50)
    payload_json: str = Field(sa_type=Text)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
    )


class KnowledgeRebuildJob(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    retry_of_job_id: int | None = Field(default=None, index=True)
    status: str = Field(default="pending", max_length=20, index=True)
    document_count: int = 0
    chunk_count: int = 0
    error_message: str | None = Field(default=None, max_length=2000)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
    )
    started_at: datetime | None = None
    completed_at: datetime | None = None


class KnowledgeReviewLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    record_type: str = Field(max_length=20, index=True)
    record_id: int = Field(index=True)
    record_title: str = Field(max_length=200)
    source: str = Field(max_length=200)
    source_url: str = Field(max_length=2000)
    source_tier: str = Field(max_length=20)
    action: str = Field(default="batch_review", max_length=50)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
    )


class RAGEvaluationCase(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    question: str = Field(max_length=1000)
    expected_name: str = Field(max_length=200)
    expected_type: str = Field(max_length=20)
    category: str = Field(default="自定义", max_length=50)
    alternative_names_json: str = Field(default="[]", max_length=2000)
    answer_keywords_json: str = Field(default="[]", sa_type=Text)
    citation_names_json: str = Field(default="[]", sa_type=Text)
    expected_refusal: bool = Field(default=False)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
    )


class ConversationMemoryState(SQLModel, table=True):
    """Compressed state for one user's conversation."""

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    conversation_id: str = Field(index=True, max_length=100)
    summary: str = Field(default="", sa_type=Text)
    summarized_message_count: int = Field(default=0)
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
    )


class UserMemory(SQLModel, table=True):
    """Explicit, user-owned long-term memory items."""

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    memory_key: str = Field(max_length=50, index=True)
    content: str = Field(sa_type=Text)
    importance: float = Field(default=0.0, index=True)
    embedding_json: str = Field(default="[]", sa_type=Text)
    active: bool = Field(default=True, index=True)
    source_conversation_id: str | None = Field(default=None, max_length=100)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
    )
    last_accessed_at: datetime | None = None
    access_count: int = Field(default=0)


class RAGEvaluationRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    total_count: int
    passed_count: int
    pass_rate: float
    preset_count: int
    custom_count: int
    retrieval_strategy: str = Field(default="hybrid-rerank", max_length=30)
    knowledge_document_count: int
    knowledge_hash: str | None = Field(default=None, max_length=64)
    results_json: str = Field(default="[]", sa_type=Text)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
    )


class RAGQualityEvaluationRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    total_count: int
    answer_correct_count: int
    answer_accuracy: float
    citation_correct_count: int
    citation_accuracy: float
    refusal_correct_count: int
    refusal_accuracy: float
    refusal_expected_count: int
    refusal_observed_count: int
    refusal_rate: float
    retrieval_strategy: str = Field(default="hybrid-rerank", max_length=30)
    knowledge_document_count: int
    knowledge_hash: str | None = Field(default=None, max_length=64)
    results_json: str = Field(default="[]", sa_type=Text)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
    )


class RAGASAutoEvaluationRun(SQLModel, table=True):
    """Persisted background task and per-question scores for a RAGAS run."""

    id: int | None = Field(default=None, primary_key=True)
    status: str = Field(default="pending", max_length=20, index=True)
    sample_size: int = Field(default=10)
    retrieval_strategy: str = Field(default="hybrid-rerank", max_length=30)
    total_count: int = Field(default=0)
    completed_count: int = Field(default=0)
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    knowledge_document_count: int = Field(default=0)
    knowledge_hash: str | None = Field(default=None, max_length=64)
    results_json: str = Field(default="[]", sa_type=Text)
    error_message: str | None = Field(default=None, sa_type=Text)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
    )
    started_at: datetime | None = None
    completed_at: datetime | None = None


class RAGRequestMetric(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int | None = Field(default=None, index=True)
    assistant_message_id: int | None = Field(default=None, index=True)
    endpoint: str = Field(default="ask/stream", max_length=30)
    success: bool = Field(default=True, index=True)
    status_code: int = Field(default=200)
    request_type: str = Field(default="rag", max_length=30, index=True)
    processing_path: str = Field(default="unknown", max_length=120)
    retrieved_count: int = Field(default=0)
    latency_ms: int = Field(default=0)
    error_type: str | None = Field(default=None, max_length=120)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
    )


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    account: str = Field(index=True, unique=True, max_length=200)
    password_hash: str = Field(max_length=300)
    is_active: bool = Field(default=True)
    is_admin: bool = Field(default=False, index=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
    )


class UserSession(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    token_hash: str = Field(index=True, unique=True, max_length=128)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime
    revoked_at: datetime | None = None


class LoginAttempt(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    account: str = Field(index=True, max_length=200)
    source_ip: str = Field(max_length=64)
    failed_count: int = Field(default=0)
    locked_until: datetime | None = None
    last_failed_at: datetime | None = None


class SecurityAuditLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int | None = Field(default=None, index=True)
    account: str = Field(max_length=200)
    method: str = Field(max_length=10)
    path: str = Field(max_length=300)
    status_code: int
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        index=True,
    )
