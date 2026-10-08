from datetime import datetime
from urllib.parse import urlparse

from pydantic import field_validator, model_validator
from sqlmodel import Field, SQLModel

from app.security import normalize_account, validate_password
from app.source_metadata import SOURCE_TIERS


def strip_required_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("内容不能为空")
    return value.strip()


class UserRead(SQLModel):
    id: int
    account: str
    is_admin: bool
    created_at: datetime


class SecurityAuditLogRead(SQLModel):
    id: int
    user_id: int | None = None
    account: str
    method: str
    path: str
    status_code: int
    created_at: datetime


class SecurityAuditLogListResponse(SQLModel):
    total_count: int
    logs: list[SecurityAuditLogRead] = Field(default_factory=list)


class AuthRegisterRequest(SQLModel):
    account: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=1, max_length=100)
    confirm_password: str = Field(min_length=1, max_length=100)

    _normalize_account = field_validator("account", mode="before")(
        normalize_account
    )
    _validate_password = field_validator("password", mode="before")(
        validate_password
    )

    @model_validator(mode="after")
    def passwords_must_match(self):
        if self.password != self.confirm_password:
            raise ValueError("两次输入的密码不一致")
        return self


class AuthLoginRequest(SQLModel):
    account: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=1, max_length=100)

    _normalize_account = field_validator("account", mode="before")(
        normalize_account
    )


class AuthResponse(SQLModel):
    access_token: str
    token_type: str
    expires_at: datetime
    user: UserRead


def validate_source_tier(value: str) -> str:
    normalized_value = strip_required_text(value)
    if normalized_value not in SOURCE_TIERS:
        raise ValueError("可信度等级必须是 authority、professional、general 或 unverified")
    return normalized_value


def normalize_source_url(value: str | None) -> str | None:
    if value is None:
        return None
    normalized_value = value.strip()
    if not normalized_value:
        return None
    parsed_url = urlparse(normalized_value)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("来源链接必须是有效的 http:// 或 https:// 地址")
    return normalized_value


class SourceMetadataCreate(SQLModel):
    source: str = Field(default="未标注来源", min_length=1, max_length=200)
    source_url: str | None = Field(default=None, max_length=2000)
    source_tier: str = Field(default="unverified", min_length=1, max_length=20)

    _strip_source = field_validator("source", mode="before")(strip_required_text)
    _normalize_source_url = field_validator("source_url", mode="before")(
        normalize_source_url
    )
    _validate_source_tier = field_validator("source_tier", mode="before")(
        validate_source_tier
    )


class ConditionCreate(SourceMetadataCreate):
    name: str = Field(min_length=1, max_length=100)
    symptoms: str = Field(min_length=1, max_length=5000)
    treatment: str = Field(min_length=1, max_length=5000)

    _strip_name = field_validator("name", mode="before")(strip_required_text)
    _strip_symptoms = field_validator("symptoms", mode="before")(
        strip_required_text
    )
    _strip_treatment = field_validator("treatment", mode="before")(
        strip_required_text
    )


class ConditionRead(ConditionCreate):
    id: int
    updated_at: datetime | None = None


class DrugCreate(SourceMetadataCreate):
    name: str = Field(min_length=1, max_length=100)
    effects: str = Field(min_length=1, max_length=5000)
    instructions: str = Field(min_length=1, max_length=5000)

    _strip_name = field_validator("name", mode="before")(strip_required_text)
    _strip_effects = field_validator("effects", mode="before")(
        strip_required_text
    )
    _strip_instructions = field_validator("instructions", mode="before")(
        strip_required_text
    )


class DrugRead(DrugCreate):
    id: int
    updated_at: datetime | None = None


class KnowledgeDocumentCreate(SourceMetadataCreate):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=20000)
    knowledge_base_id: str = Field(default="global", min_length=1, max_length=100)
    visibility: str = Field(default="public", min_length=1, max_length=20)
    page_number: int | None = Field(default=None, ge=1)

    _strip_title = field_validator("title", mode="before")(strip_required_text)
    _strip_content = field_validator("content", mode="before")(strip_required_text)

    @field_validator("knowledge_base_id", mode="before")
    @classmethod
    def normalize_knowledge_base_id(cls, value: str) -> str:
        return strip_required_text(value)

    @field_validator("visibility", mode="before")
    @classmethod
    def validate_visibility(cls, value: str) -> str:
        normalized_value = strip_required_text(value)
        if normalized_value not in {"public", "private"}:
            raise ValueError("可见范围必须是 public 或 private")
        return normalized_value


class KnowledgeDocumentRead(KnowledgeDocumentCreate):
    id: int
    updated_at: datetime | None = None


class DocumentUploadItem(SQLModel):
    filename: str
    title: str
    document_id: int
    created_document_count: int = 1


class DocumentUploadError(SQLModel):
    filename: str
    detail: str


class DocumentUploadBatchResponse(SQLModel):
    created_count: int
    created_document_count: int = 0
    failed_count: int
    items: list[DocumentUploadItem] = Field(default_factory=list)
    errors: list[DocumentUploadError] = Field(default_factory=list)


class DocumentWebImportRequest(SQLModel):
    url: str = Field(min_length=1, max_length=2000)
    title: str | None = Field(default=None, max_length=200)

    _normalize_url = field_validator("url", mode="before")(normalize_source_url)

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("资料标题必须是文本")
        return value.strip() or None


KNOWLEDGE_TYPES = {"all", "condition", "drug", "document"}
SOURCE_FILTERS = {"all", "reviewed"}


def validate_knowledge_type(value: str) -> str:
    normalized_value = strip_required_text(value)
    if normalized_value not in KNOWLEDGE_TYPES:
        raise ValueError("检索范围必须是 all、condition、drug 或 document")
    return normalized_value


def validate_concrete_knowledge_type(value: str) -> str:
    normalized_value = validate_knowledge_type(value)
    if normalized_value == "all":
        raise ValueError("资料类型必须是 condition、drug 或 document")
    return normalized_value


def validate_source_filter(value: str) -> str:
    normalized_value = strip_required_text(value)
    if normalized_value not in SOURCE_FILTERS:
        raise ValueError("资料可信度筛选必须是 all 或 reviewed")
    return normalized_value


class AskRequest(SQLModel):
    question: str = Field(min_length=1, max_length=1000)
    conversation_id: str = Field(default="default", min_length=1, max_length=100)
    knowledge_type: str = Field(default="all", min_length=1, max_length=20)
    source_filter: str = Field(default="all", min_length=1, max_length=20)

    _strip_question = field_validator("question", mode="before")(
        strip_required_text
    )
    _strip_conversation_id = field_validator("conversation_id", mode="before")(
        strip_required_text
    )
    _validate_knowledge_type = field_validator("knowledge_type", mode="before")(
        validate_knowledge_type
    )
    _validate_source_filter = field_validator("source_filter", mode="before")(
        validate_source_filter
    )


class ReferenceRead(SQLModel):
    name: str
    type: str
    record_id: int | None = None
    source: str | None = None
    source_url: str | None = None
    source_kind: str = "知识文档"
    location: str = "全文"
    citation: str = ""
    page_number: int | None = None
    chunk_index: int | None = None
    chunk_count: int | None = None
    source_tier: str = "unverified"
    updated_at: datetime | None = None
    needs_review: bool = True
    excerpt: str
    relevance_score: float
    initial_score: float | None = None
    rerank_score: float | None = None
    rerank_text_score: float | None = None
    rerank_title_score: float | None = None
    retrieval_method: str = "vector"
    vector_score: float | None = None
    keyword_score: float | None = None


class AskResponse(SQLModel):
    question: str
    answer: str
    source: str
    conversation_id: str
    assistant_message_id: int
    references: list[ReferenceRead] = Field(default_factory=list)
    processing_path: str
    retrieval_scope: str
    source_filter: str
    retrieved_count: int
    latency_ms: int


class ChatMessageRead(SQLModel):
    id: int
    conversation_id: str
    role: str
    content: str
    response_metadata: dict = Field(default_factory=dict)
    created_at: datetime


class ConversationSummary(SQLModel):
    conversation_id: str
    preview: str
    message_count: int
    updated_at: datetime


class UserMemoryRead(SQLModel):
    id: int
    memory_key: str
    content: str
    importance: float
    source_conversation_id: str | None = None
    access_count: int
    created_at: datetime
    updated_at: datetime


class MemoryMessageRead(SQLModel):
    role: str
    content: str
    created_at: datetime


class MemoryOverview(SQLModel):
    conversation_id: str
    short_term_summary: str = ""
    summarized_message_count: int = 0
    short_term_message_count: int = 0
    recent_message_limit: int
    recent_messages: list[MemoryMessageRead] = Field(default_factory=list)
    long_term_memories: list[UserMemoryRead] = Field(default_factory=list)


class MemoryDeleteResponse(SQLModel):
    deleted: bool
    memory_id: int


def normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip() or None


class FeedbackCreate(SQLModel):
    assistant_message_id: int = Field(gt=0)
    helpful: bool
    reason: str | None = Field(default=None, max_length=1000)

    _normalize_reason = field_validator("reason", mode="before")(
        normalize_optional_text
    )


class FeedbackRead(FeedbackCreate):
    id: int
    created_at: datetime


class FeedbackSummary(SQLModel):
    total_count: int
    helpful_count: int
    not_helpful_count: int
    helpful_rate: float | None = None


class FeedbackDetail(SQLModel):
    id: int
    assistant_message_id: int
    conversation_id: str
    question: str | None = None
    answer: str
    helpful: bool
    reason: str | None = None
    created_at: datetime


class FeedbackGroup(SQLModel):
    text: str
    count: int


class FeedbackImprovementSuggestions(SQLModel):
    not_helpful_count: int
    common_reasons: list[FeedbackGroup] = Field(default_factory=list)
    repeated_questions: list[FeedbackGroup] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


class KnowledgeSearchItem(SQLModel):
    type: str
    record_id: int
    title: str
    source: str | None = None
    source_url: str | None = None
    source_tier: str = "unverified"
    updated_at: datetime | None = None
    needs_review: bool = True
    matched_fields: list[str]
    excerpt: str


class KnowledgeSearchResponse(SQLModel):
    query: str
    total_count: int
    results: list[KnowledgeSearchItem] = Field(default_factory=list)


class KnowledgeReviewItem(SQLModel):
    type: str
    record_id: int
    title: str
    source: str
    source_url: str | None = None
    source_tier: str
    updated_at: datetime | None = None
    review_reasons: list[str] = Field(default_factory=list)


class KnowledgeReviewResponse(SQLModel):
    total_count: int
    results: list[KnowledgeReviewItem] = Field(default_factory=list)


class KnowledgeReviewTarget(SQLModel):
    type: str = Field(min_length=1, max_length=20)
    record_id: int = Field(gt=0)

    _validate_type = field_validator("type", mode="before")(
        validate_concrete_knowledge_type
    )


class KnowledgeReviewBatchUpdate(SQLModel):
    targets: list[KnowledgeReviewTarget] = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, max_length=200)
    source_url: str = Field(min_length=1, max_length=2000)
    source_tier: str = Field(min_length=1, max_length=20)

    _strip_source = field_validator("source", mode="before")(strip_required_text)
    _normalize_source_url = field_validator("source_url", mode="before")(
        normalize_source_url
    )
    _validate_source_tier = field_validator("source_tier", mode="before")(
        validate_source_tier
    )

    @model_validator(mode="after")
    def validate_reviewed_targets(self):
        if self.source_tier == "unverified":
            raise ValueError("批量审核不能把资料标记为待核实")
        target_keys = {(target.type, target.record_id) for target in self.targets}
        if len(target_keys) != len(self.targets):
            raise ValueError("不能重复选择同一条资料")
        return self


class KnowledgeReviewBatchUpdateResponse(SQLModel):
    updated_count: int
    items: list[KnowledgeReviewItem] = Field(default_factory=list)


class KnowledgeReviewLogRead(SQLModel):
    id: int
    record_type: str
    record_id: int
    record_title: str
    source: str
    source_url: str
    source_tier: str
    action: str
    created_at: datetime


class KnowledgeReviewLogResponse(SQLModel):
    total_count: int
    logs: list[KnowledgeReviewLogRead] = Field(default_factory=list)


class RAGEvaluationCaseResult(SQLModel):
    case_id: str
    case_source: str
    question: str
    expected_name: str
    expected_type: str
    category: str
    alternative_names: list[str] = Field(default_factory=list)
    passed: bool
    expected_rank: int | None = None
    matched_name: str | None = None
    top_name: str | None = None
    top_type: str | None = None
    top_score: float | None = None
    retrieved_count: int = 0
    relevant_count: int = 0


class RetrievalStrategyResult(SQLModel):
    passed: bool
    expected_rank: int | None = None
    matched_name: str | None = None
    top_name: str | None = None
    top_type: str | None = None
    top_score: float | None = None
    retrieved_count: int = 0
    relevant_count: int = 0


class RetrievalEvaluationMetrics(SQLModel):
    top1_correct_count: int
    top1_accuracy: float
    recalled_count: int
    recall_at_3: float
    relevant_result_count: int
    retrieved_result_count: int
    precision_at_3: float


class RetrievalComparisonCaseResult(SQLModel):
    case_id: str
    case_source: str
    question: str
    expected_name: str
    expected_type: str
    category: str
    alternative_names: list[str] = Field(default_factory=list)
    baseline: RetrievalStrategyResult
    current: RetrievalStrategyResult
    change: str


class RetrievalStrategySummary(SQLModel):
    passed_count: int
    pass_rate: float
    metrics: RetrievalEvaluationMetrics


class RetrievalComparisonResponse(SQLModel):
    total_count: int
    preset_count: int
    custom_count: int
    baseline_strategy: str
    current_strategy: str
    baseline: RetrievalStrategySummary
    current: RetrievalStrategySummary
    pass_rate_delta: float
    top1_accuracy_delta: float
    recall_at_3_delta: float
    precision_at_3_delta: float
    improved_count: int
    regressed_count: int
    unchanged_count: int
    results: list[RetrievalComparisonCaseResult] = Field(default_factory=list)


class RetrievalDiagnosticCandidate(SQLModel):
    rank: int
    name: str
    type: str
    relevance_score: float


class RetrievalDiagnosticCaseResult(SQLModel):
    case_id: str
    case_source: str
    question: str
    expected_name: str
    expected_type: str
    category: str
    alternative_names: list[str] = Field(default_factory=list)
    passed: bool
    expected_rank: int | None = None
    matched_name: str | None = None
    diagnostic_level: str
    diagnostic: str
    suggested_action: str
    candidates: list[RetrievalDiagnosticCandidate] = Field(default_factory=list)


class RetrievalDiagnosticResponse(SQLModel):
    total_count: int
    healthy_count: int
    attention_count: int
    failed_count: int
    results: list[RetrievalDiagnosticCaseResult] = Field(default_factory=list)


class RAGEvaluationCategoryMetric(SQLModel):
    category: str
    total_count: int
    passed_count: int
    pass_rate: float


class RAGEvaluationQualityGate(SQLModel):
    status: str
    compared_history_id: int | None = None
    message: str
    regressed_questions: list[str] = Field(default_factory=list)
    improved_questions: list[str] = Field(default_factory=list)
    new_questions: list[str] = Field(default_factory=list)
    rank_regressed_questions: list[str] = Field(default_factory=list)
    rank_improved_questions: list[str] = Field(default_factory=list)


class RAGEvaluationResponse(SQLModel):
    history_id: int
    total_count: int
    passed_count: int
    pass_rate: float
    preset_count: int
    custom_count: int
    retrieval_strategy: str
    retrieval_metrics: RetrievalEvaluationMetrics
    category_metrics: list[RAGEvaluationCategoryMetric] = Field(default_factory=list)
    quality_gate: RAGEvaluationQualityGate
    results: list[RAGEvaluationCaseResult] = Field(default_factory=list)


class RAGEvaluationHistoryRead(SQLModel):
    id: int
    total_count: int
    passed_count: int
    pass_rate: float
    preset_count: int
    custom_count: int
    retrieval_strategy: str
    knowledge_document_count: int
    knowledge_hash: str | None = None
    created_at: datetime


class RAGEvaluationHistoryResponse(SQLModel):
    total_count: int
    runs: list[RAGEvaluationHistoryRead] = Field(default_factory=list)


EVALUATION_EXPECTED_TYPES = {"condition", "drug", "document"}


def validate_evaluation_expected_type(value: str) -> str:
    normalized_value = strip_required_text(value)
    if normalized_value not in EVALUATION_EXPECTED_TYPES:
        raise ValueError("目标资料类型必须是 condition、drug 或 document")
    return normalized_value


def normalize_alternative_names(value: list[str] | None) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("可接受资料名称必须是列表")

    names: list[str] = []
    for name in value:
        normalized_name = strip_required_text(name)
        if normalized_name not in names:
            names.append(normalized_name)

    if len(names) > 10:
        raise ValueError("最多可填写 10 个可接受资料名称")
    return names


class RAGEvaluationCaseCreate(SQLModel):
    question: str = Field(min_length=1, max_length=1000)
    expected_name: str = Field(min_length=1, max_length=200)
    expected_type: str = Field(min_length=1, max_length=20)
    category: str = Field(default="自定义", min_length=1, max_length=50)
    alternative_names: list[str] = Field(default_factory=list)
    answer_keywords: list[str] = Field(default_factory=list)
    citation_names: list[str] = Field(default_factory=list)
    expected_refusal: bool = False

    _strip_question = field_validator("question", mode="before")(strip_required_text)
    _strip_expected_name = field_validator("expected_name", mode="before")(
        strip_required_text
    )
    _validate_expected_type = field_validator("expected_type", mode="before")(
        validate_evaluation_expected_type
    )
    _strip_category = field_validator("category", mode="before")(strip_required_text)
    _normalize_alternative_names = field_validator(
        "alternative_names", mode="before"
    )(normalize_alternative_names)
    _normalize_answer_keywords = field_validator(
        "answer_keywords", mode="before"
    )(normalize_alternative_names)
    _normalize_citation_names = field_validator(
        "citation_names", mode="before"
    )(normalize_alternative_names)


class RAGEvaluationCaseRead(RAGEvaluationCaseCreate):
    id: int
    created_at: datetime


class RAGEvaluationCaseListResponse(SQLModel):
    total_count: int
    cases: list[RAGEvaluationCaseRead] = Field(default_factory=list)


class RAGQualityCaseResult(SQLModel):
    case_id: str
    case_source: str
    question: str
    category: str
    expected_refusal: bool
    answer_correct: bool
    citation_correct: bool
    refusal_observed: bool
    refusal_correct: bool
    answer_keyword_count: int
    answer_match_count: int
    missing_answer_keywords: list[str] = Field(default_factory=list)
    expected_citation_names: list[str] = Field(default_factory=list)
    cited_names: list[str] = Field(default_factory=list)
    missing_citation_names: list[str] = Field(default_factory=list)
    citation_has_location: bool
    retrieved_count: int
    diagnostic: str


class RAGQualityMetrics(SQLModel):
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


class RAGQualityResponse(SQLModel):
    history_id: int | None = None
    metrics: RAGQualityMetrics
    preset_count: int
    custom_count: int
    retrieval_strategy: str
    results: list[RAGQualityCaseResult] = Field(default_factory=list)


class RAGASMetricScores(SQLModel):
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


class RAGASQuestionResult(SQLModel):
    case_id: str
    case_source: str
    question: str
    expected_name: str
    expected_type: str
    category: str
    answer: str
    processing_path: str
    retrieved_count: int
    context_titles: list[str] = Field(default_factory=list)
    retrieved_contexts: list[str] = Field(default_factory=list)
    reference_available: bool
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    error_message: str | None = None


class RAGASAutoEvaluationRunRead(SQLModel):
    id: int
    status: str
    sample_size: int
    retrieval_strategy: str
    total_count: int
    completed_count: int
    metrics: RAGASMetricScores | None = None
    knowledge_document_count: int
    knowledge_hash: str | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    results: list[RAGASQuestionResult] = Field(default_factory=list)


class RAGASAutoEvaluationRunListResponse(SQLModel):
    total_count: int
    runs: list[RAGASAutoEvaluationRunRead] = Field(default_factory=list)


class MonitoringPathMetric(SQLModel):
    processing_path: str
    request_count: int
    success_count: int
    failure_count: int
    average_latency_ms: float
    retrieval_hit_rate: float | None = None


class MonitoringFailureRead(SQLModel):
    endpoint: str
    status_code: int
    request_type: str
    processing_path: str
    error_type: str | None = None
    latency_ms: int
    created_at: datetime


class MonitoringQualitySnapshot(SQLModel):
    retrieval_pass_rate: float | None = None
    answer_accuracy: float | None = None
    citation_accuracy: float | None = None
    refusal_accuracy: float | None = None
    evaluated_at: datetime | None = None


class MonitoringFeedbackSnapshot(SQLModel):
    total_count: int
    helpful_count: int
    not_helpful_count: int
    helpful_rate: float | None = None


class MonitoringSummary(SQLModel):
    window_hours: int
    window_started_at: datetime
    generated_at: datetime
    request_count: int
    successful_request_count: int
    failed_request_count: int
    success_rate: float
    failure_rate: float
    retrieval_attempt_count: int
    retrieval_hit_count: int
    retrieval_hit_rate: float | None = None
    average_latency_ms: float | None = None
    p50_latency_ms: int | None = None
    p95_latency_ms: int | None = None
    quality: MonitoringQualitySnapshot
    feedback: MonitoringFeedbackSnapshot
    path_metrics: list[MonitoringPathMetric] = Field(default_factory=list)
    recent_failures: list[MonitoringFailureRead] = Field(default_factory=list)


class KnowledgeRebuildResponse(SQLModel):
    message: str
    document_count: int
    chunk_count: int
    snapshot_id: int
    snapshot_created: bool


class KnowledgeRebuildJobRead(SQLModel):
    id: int
    retry_of_job_id: int | None = None
    status: str
    document_count: int
    chunk_count: int
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class KnowledgeRebuildJobListResponse(SQLModel):
    total_count: int
    jobs: list[KnowledgeRebuildJobRead] = Field(default_factory=list)


class KnowledgeVersionRead(SQLModel):
    id: int
    document_count: int
    reason: str
    created_at: datetime
    is_current: bool


class KnowledgeVersionListResponse(SQLModel):
    total_count: int
    versions: list[KnowledgeVersionRead] = Field(default_factory=list)


class KnowledgeRestoreResponse(SQLModel):
    message: str
    restored_version_id: int
    backup_version_id: int
    document_count: int
    chunk_count: int


class KnowledgeStatusResponse(SQLModel):
    is_current: bool
    document_count: int
    chunk_count: int
    indexed_document_count: int | None = None
    vector_store_type: str = "milvus"
    vector_count: int | None = None
    indexed_at: datetime | None = None
    text_cleaning_version: str | None = None
    chunking_strategy: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class HealthResponse(SQLModel):
    status: str
    service: str
    database: str
    cache: str
    knowledge_base_current: bool
    document_count: int
    chunk_count: int
