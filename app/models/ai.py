"""AI 可观测性模型：任务、LLM 调用日志、检索日志、工作流追踪、指标聚合。

这五张表支撑「AI 运行监控」与「RAG 日志可回放」两个需求。
核心设计是一次业务请求生成一个 ``trace_id``，贯穿工作流每个节点、
每次 LLM 调用、每次检索，事后可以按 trace_id 完整还原一次 AI 决策链路。
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKey
from app.models.types import JSONBText


class AITaskStatus(str, enum.Enum):
    """异步 AI 任务状态。演进自旧项目的 KnowledgeRebuildJob。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class AITask(UUIDPrimaryKey, TimestampMixin, Base):
    """异步 AI 任务。Celery 任务的业务侧状态记录。

    Celery 自身的 result backend 只保留 ``result_expires``（1 小时）内的结果，
    不适合做长期审计。这张表是任务状态的持久化真相来源，
    与 Celery task_id 关联但生命周期独立。
    """

    __tablename__ = "ai_tasks"
    __table_args__ = (
        Index("idx_ai_task_type_status", "task_type", "status"),
        Index("idx_ai_task_ref", "task_type", "ref_id"),
        Index("idx_ai_task_celery_id", "celery_task_id"),
    )

    # knowledge_vectorize / report_interpret / triage / ragas_evaluate / metrics_aggregate
    task_type: Mapped[str] = mapped_column(String(40), nullable=False)
    # 关联的业务对象 ID，如报告 ID、知识库重建批次 ID
    ref_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    # Celery 任务 ID，用于关联 Flower 里的执行详情
    celery_task_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        String(20),
        default=AITaskStatus.PENDING.value,
        nullable=False,
    )
    # 进度百分比，长任务（如知识库向量化）用于前端展示进度条
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress_message: Mapped[str | None] = mapped_column(String(200))
    payload: Mapped[dict | None] = mapped_column(JSONBText(), comment="任务入参快照")
    result: Mapped[dict | None] = mapped_column(JSONBText(), comment="任务产出")
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # 幂等键：同一 key 的重复任务直接复用已有结果，不重复调用 LLM
    idempotency_key: Mapped[str | None] = mapped_column(String(128), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # 排队时长与执行时长分开记，才能区分「队列积压」和「任务本身慢」
    queued_ms: Mapped[int | None] = mapped_column(Integer)
    duration_ms: Mapped[int | None] = mapped_column(Integer)


class LLMCallLog(Base):
    """LLM 调用日志。Token 预算、成本核算、降级率统计的数据来源。

    写入路径是**内存缓冲 + 批量异步刷盘**，不在请求链路里同步 INSERT，
    否则会拖慢流式响应的首 token 时间。
    """

    __tablename__ = "llm_call_logs"
    __table_args__ = (
        # 主键用 BIGSERIAL：调用日志是高频写入表，自增整数的 B-tree 插入
        # 比 UUID 顺序性更好，不会造成索引页随机写放大
        Index("idx_llm_log_created", "created_at"),
        Index("idx_llm_log_scenario", "scenario", "created_at"),
        Index("idx_llm_log_provider_model", "provider", "model"),
        Index("idx_llm_log_trace", "trace_id"),
        Index("idx_llm_log_status", "status", "created_at"),
        Index("idx_llm_log_user", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 贯穿一次业务请求的链路 ID，与工作流追踪、检索日志共享
    trace_id: Mapped[str | None] = mapped_column(String(36), index=True)
    # chat / triage / report_interpret / embedding / rerank / moderation / summary
    scenario: Mapped[str] = mapped_column(String(30), nullable=False)
    provider: Mapped[str] = mapped_column(String(20), nullable=False, comment="openai / dashscope")
    model: Mapped[str] = mapped_column(String(50), nullable=False)
    # 请求的目标模型；降级后 provider_model 与 requested_model 不同，据此统计降级率
    requested_model: Mapped[str | None] = mapped_column(String(50))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # 由数据库计算，避免应用侧漏更新导致 token 统计对不上账
    total_tokens: Mapped[int] = mapped_column(
        Integer,
        Computed("prompt_tokens + completion_tokens", persisted=True),
        nullable=False,
    )
    cost_usd: Mapped[float | None] = mapped_column(Numeric(10, 6))
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # 首 token 延迟。流式体验的关键指标，与总延迟分开统计
    first_token_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="success", nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(60))
    http_status: Mapped[int | None] = mapped_column(Integer)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # 降级链层级：0=主模型，1=备用模型，2=语义缓存，3=确定性规则，4=兜底话术
    degradation_level: Mapped[int | None] = mapped_column(Integer)
    # 是否触发内容审核拦截
    moderation_blocked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # 预算检查是否被拒
    budget_exceeded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_stream: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    user_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    # 工作流节点名，定位是哪个节点消耗了 token
    workflow_node: Mapped[str | None] = mapped_column(String(60))
    ai_task_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class RAGRetrievalLog(Base):
    """RAG 检索日志。记录各路召回与 RRF 融合的贡献度。

    ``vector_hit_count`` / ``keyword_hit_count`` / ``both_hit_count`` 三个字段
    用于回答「哪条召回路径真正起作用」：若 both_hit_count 长期为 0，
    说明两路召回几乎不重叠，融合带来的收益有限，需要重新审视分词或向量模型。
    """

    __tablename__ = "rag_retrieval_logs"
    __table_args__ = (
        Index("idx_rag_log_created", "created_at"),
        Index("idx_rag_log_trace", "trace_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trace_id: Mapped[str | None] = mapped_column(String(36), index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    query_text: Mapped[str | None] = mapped_column(Text)
    # 改写后的检索问题。多轮对话里指代消解后的查询与原始问题不同
    rewritten_query: Mapped[str | None] = mapped_column(Text)
    vector_hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    keyword_hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # 两路都命中的数量，衡量 RRF 融合的交叉验证效果
    both_hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    merged_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    final_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # 零结果意味着知识库缺口，是补充资料的信号
    is_zero_result: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    top_score: Mapped[float | None] = mapped_column(Numeric(8, 6))
    vector_latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    keyword_latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rrf_latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rerank_latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # rerank 走的通道：cross_encoder（DashScope gte-rerank）/ rule（规则降级）/ none
    rerank_channel: Mapped[str | None] = mapped_column(String(20))
    rrf_k: Mapped[int] = mapped_column(Integer, default=60, nullable=False)
    # 融合前后 top-k 明细，含各路 rank 与 rrf_score，用于回放
    candidates: Mapped[list | None] = mapped_column(JSONBText())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class WorkflowTrace(Base):
    """工作流执行追踪。一次工作流一条主记录 + 每节点一条 step 记录。

    存 ``input_snapshot`` / ``output_snapshot`` 的价值在于可回放：
    当医生质疑某份报告的 AI 解读时，能还原当时喂给模型的中间状态，
    而不是只看到最终输出。
    """

    __tablename__ = "workflow_traces"
    __table_args__ = (
        Index("idx_wf_trace_workflow", "workflow_name", "created_at"),
        Index("idx_wf_trace_trace_id", "trace_id"),
        Index("idx_wf_trace_status", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(36), nullable=False)
    # triage / report_interpret / prescription_review
    workflow_name: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="running", nullable=False)
    # 各节点执行明细：[{node, type, latency_ms, tokens, input, output, error}]
    steps: Mapped[list | None] = mapped_column(JSONBText())
    node_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    executed_node_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_cost_usd: Mapped[float | None] = mapped_column(Numeric(10, 6))
    # 走了哪个分支。分诊工作流的急诊分支与挂号引导分支要能区分统计
    branch_taken: Mapped[str | None] = mapped_column(String(40))
    final_output: Mapped[dict | None] = mapped_column(JSONBText())
    error_message: Mapped[str | None] = mapped_column(Text)
    user_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    ref_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AIMetric(Base):
    """AI 指标分钟级聚合。监控看板直接查这张表，不扫明细日志。

    指标存 PostgreSQL 而不引入 Prometheus/Grafana：本项目日均调用量在千级，
    PostgreSQL 窗口函数足以算 P50/P95/P99，多一套时序系统只会增加运维成本。
    ``tags`` 用 JSONB 存放维度（如 model、provider、scenario），
    避免为每个维度组合建一张表。
    """

    __tablename__ = "ai_metrics"
    __table_args__ = (
        Index("idx_ai_metric_name_time", "metric_name", "bucket_time"),
        Index("idx_ai_metric_bucket", "bucket_time"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    bucket_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="分钟级时间桶，向下取整到分钟",
    )
    # llm_call_count / llm_error_rate / llm_p95_latency / llm_first_token_p95
    # / token_usage / token_cost / degradation_rate / moderation_block_rate
    # / retrieval_zero_result_rate / retrieval_p95_latency / task_queue_backlog
    metric_name: Mapped[str] = mapped_column(String(50), nullable=False)
    metric_value: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tags: Mapped[dict | None] = mapped_column(JSONBText())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class TriageResult(UUIDPrimaryKey, TimestampMixin, Base):
    """智能分诊结果。

    独立成表而不是塞进 appointments 的理由：分诊未必转化成挂号
    （患者可能只是咨询，或被引导去急诊）。要把「分诊建议科室」与
    「患者实际挂号科室」做对比来评估分诊准确率，必须保留未转化的记录。
    """

    __tablename__ = "triage_results"
    __table_args__ = (
        Index("idx_triage_patient", "patient_id", "created_at"),
        Index("idx_triage_urgency", "urgency", "created_at"),
    )

    patient_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    user_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    trace_id: Mapped[str | None] = mapped_column(String(36), index=True)
    chief_complaint: Mapped[str] = mapped_column(Text, nullable=False)
    # LLM 提取出的症状列表，结构化存储便于统计高频症状
    extracted_symptoms: Mapped[list | None] = mapped_column(JSONBText())
    recommended_department_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("departments.id"),
    )
    recommended_department_name: Mapped[str | None] = mapped_column(String(50))
    # routine / urgent / stat
    urgency: Mapped[str] = mapped_column(String(10), default="routine", nullable=False)
    # 是否触发急诊警示分支
    is_emergency: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    matched_emergency_keywords: Mapped[list | None] = mapped_column(JSONBText())
    advice: Mapped[str | None] = mapped_column(Text, comment="给患者的就医建议")
    confidence: Mapped[float | None] = mapped_column(Numeric(4, 3))
    # ai / rule / hybrid：记录本次分诊由哪条路径给出，评估 LLM 依赖度
    decision_source: Mapped[str] = mapped_column(String(20), default="ai", nullable=False)
    ai_task_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    llm_model: Mapped[str | None] = mapped_column(String(50))
    # 患者是否采纳建议并完成了挂号，用于计算分诊转化率
    converted_appointment_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    # 医生接诊后回填的实际科室，与推荐科室对比得出分诊准确率
    actual_department_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    feedback_rating: Mapped[int | None] = mapped_column(Integer, comment="患者对分诊的评分 1-5")
