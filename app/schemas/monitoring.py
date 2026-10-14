"""AI 监控接口的请求/响应模型（阶段 6 批次 A）。

字段口径对齐 ``AIMetric`` / ``LLMCallLog`` / ``RAGRetrievalLog`` /
``WorkflowTrace`` / ``TriageResult``（models/ai.py）。约定与前序批次
（``schemas/billing.py``、``schemas/exam.py``）保持一致：

1. **UUID 对外一律字符串**：SQLite 与 PG 驱动返回类型不同，在协议层统一成 str，
   响应契约与后端方言无关。
2. **响应显式列字段**，不用 ``from_attributes`` 全量导出 ORM。回放接口尤其重要：
   ``llm_call_logs.user_id``、``workflow_traces.ref_id`` 都是可关联到自然人/病历的
   标识，管理员看板需要的是运行指标而不是患者身份，逐字段列出等于把这条边界写死。
3. **数值统一 float**：``metric_value`` / ``cost_usd`` 在库里是 Numeric，
   直接吐 Decimal 会被 JSON 序列化成字符串，前端要做一次 parseFloat 才能画曲线；
   在 schema 层转 float，让「时间序列点」是个可直接喂图表的数字。

============================ 为什么不做严格枚举校验 ============================

``status`` / ``scenario`` / ``provider`` 这些过滤参数保持自由字符串：
监控接口的职责是「按你给的条件捞数据」，取值集合会随降级链、供应商接入而增加。
在协议层用 Literal 卡住，等于每加一个 provider 就要改 schema——
而查询条件写错了最坏结果是空列表，不是错误决策，不值得为此牺牲可扩展性。
真正需要钉死取值的是**写入路径**（聚合侧的指标名常量），那在 metrics_service 里。
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ==================== 常量 ====================

# 手动触发聚合时允许回溯的最大分钟数。补数据是运维刚需（Beat 停摆一段时间后要回填），
# 但不设上限就能一次请求要求扫几十万行明细，把 API 进程变成批处理机。
MAX_AGGREGATE_LOOKBACK_MINUTES = 180

# overview 窗口上限（分钟）= 6 小时。更长的区间属于曲线查询（/metrics）的职责。
MAX_OVERVIEW_WINDOW_MINUTES = 360


# ==================== 时间序列 ====================


class MetricPointOut(BaseModel):
    """单个时间桶上的一个指标读数。"""

    bucket_time: datetime = Field(description="分钟桶起点（向下取整到分钟）")
    metric_name: str = Field(description="指标名，取值见 metrics_service.METRIC_*")
    metric_value: float = Field(description="指标值；比率类为 0~1 的小数")
    sample_count: int = Field(description="该读数的样本量，用于判断可信度")
    tags: dict[str, Any] = Field(
        default_factory=dict, description="维度标签，如 {\"model\": \"qwen-max\"}"
    )


class MetricSeriesResponse(BaseModel):
    """GET /monitoring/metrics 响应。"""

    items: list[MetricPointOut]
    count: int = Field(description="返回点数，便于前端判断是否被 limit 截断")
    limit: int
    metric_names: list[str] = Field(default_factory=list, description="本次查询的指标名")


# ==================== 调用明细 ====================


class LLMCallOut(BaseModel):
    """一次 LLM 调用的明细行（下钻用）。"""

    id: int
    trace_id: str | None = None
    scenario: str
    provider: str
    model: str
    requested_model: str | None = Field(
        default=None, description="请求的目标模型；与 model 不同即发生了降级"
    )
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None = None
    latency_ms: int
    first_token_ms: int | None = None
    status: str = Field(description="success / degraded / error / timeout")
    error_type: str | None = None
    http_status: int | None = None
    retry_count: int = 0
    degradation_level: int | None = Field(
        default=None, description="0=主模型 1=备用 2=语义缓存 3=规则 4=兜底"
    )
    moderation_blocked: bool = False
    budget_exceeded: bool = False
    is_stream: bool = False
    user_id: str | None = None
    workflow_node: str | None = None
    ai_task_id: str | None = None
    created_at: datetime | None = None


class LLMCallListResponse(BaseModel):
    items: list[LLMCallOut]
    total: int
    page: int
    page_size: int


# ==================== 全链路回放 ====================


class RetrievalFragmentOut(BaseModel):
    """回放片段：一次 RAG 检索。"""

    id: int
    trace_id: str | None = None
    user_id: str | None = None
    query_text: str | None = None
    rewritten_query: str | None = None
    vector_hit_count: int = 0
    keyword_hit_count: int = 0
    both_hit_count: int = 0
    merged_count: int = 0
    final_count: int = 0
    is_zero_result: bool = False
    top_score: float | None = None
    vector_latency_ms: int = 0
    keyword_latency_ms: int = 0
    rrf_latency_ms: int = 0
    rerank_latency_ms: int = 0
    rerank_channel: str | None = None
    rrf_k: int = 60
    candidates: list[Any] = Field(default_factory=list)
    created_at: datetime | None = None


class WorkflowFragmentOut(BaseModel):
    """回放片段：一次工作流执行。"""

    id: int
    trace_id: str | None = None
    workflow_name: str
    status: str
    steps: list[Any] = Field(default_factory=list)
    node_count: int = 0
    executed_node_count: int = 0
    total_latency_ms: int = 0
    total_tokens: int = 0
    total_cost_usd: float | None = None
    branch_taken: str | None = None
    final_output: dict[str, Any] | None = None
    error_message: str | None = None
    user_id: str | None = None
    ref_id: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None


class TriageFragmentOut(BaseModel):
    """回放片段：分诊业务产出。"""

    id: str | None = None
    trace_id: str | None = None
    patient_id: str | None = None
    user_id: str | None = None
    chief_complaint: str | None = None
    extracted_symptoms: list[Any] = Field(default_factory=list)
    recommended_department_id: str | None = None
    recommended_department_name: str | None = None
    urgency: str | None = None
    is_emergency: bool = False
    matched_emergency_keywords: list[Any] = Field(default_factory=list)
    advice: str | None = None
    confidence: float | None = None
    decision_source: str | None = None
    ai_task_id: str | None = None
    llm_model: str | None = None
    converted_appointment_id: str | None = None
    actual_department_id: str | None = None
    feedback_rating: int | None = None
    created_at: datetime | None = None


class TraceSummaryOut(BaseModel):
    """回放汇总：整条链路的资源消耗与质量概况。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    call_count: int = 0
    error_count: int = 0
    degraded_count: int = 0
    max_latency_ms: int | None = None
    p95_latency_ms: int | None = Field(
        default=None, description="本链路内调用延迟 P95（最近秩法，与聚合同口径）"
    )
    started_at: datetime | None = None
    ended_at: datetime | None = None
    providers: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    scenarios: list[str] = Field(default_factory=list)


class TraceCountsOut(BaseModel):
    """各段日志的行数。缺段就是 0，排障时先看这里判断哪一环没记上。"""

    llm_calls: int = 0
    retrievals: int = 0
    workflows: int = 0
    triage_results: int = 0


class TimelineEntryOut(BaseModel):
    """时间线条目：四类来源按时刻合并后的单条事件。"""

    kind: str = Field(description="llm_call / retrieval / workflow / triage_result")
    at: datetime | None = None
    summary: str = Field(description="一行可读摘要，供前端直接渲染")
    ref: dict[str, Any] = Field(default_factory=dict, description="回指原始片段的定位信息")


class TraceReplayResponse(BaseModel):
    """GET /monitoring/traces/{trace_id} 响应：一次问答的全链路回放。"""

    trace_id: str
    counts: TraceCountsOut
    summary: TraceSummaryOut
    llm_calls: list[LLMCallOut] = Field(default_factory=list)
    retrievals: list[RetrievalFragmentOut] = Field(default_factory=list)
    workflows: list[WorkflowFragmentOut] = Field(default_factory=list)
    triage_results: list[TriageFragmentOut] = Field(default_factory=list)
    timeline: list[TimelineEntryOut] = Field(default_factory=list)


# ==================== 看板首屏 ====================


class OverviewMetricsOut(BaseModel):
    """窗口内各项指标的当前读数。"""

    llm_call_count: float | None = None
    llm_error_rate: float | None = Field(
        default=None, description="0~1 小数；无样本时为 null 而非 0"
    )
    llm_p95_latency_ms: float | None = None
    token_usage_total: float | None = Field(default=None, description="窗口内 token 合计")
    token_cost_usd: float | None = None
    degradation_rate: float | None = None
    moderation_block_rate: float | None = None
    retrieval_zero_result_rate: float | None = None
    task_queue_backlog: float | None = Field(
        default=None, description="窗口末个桶的积压快照（不求和）"
    )


class OverviewResponse(BaseModel):
    """GET /monitoring/overview 响应。"""

    window_minutes: int
    window_start: datetime
    window_end: datetime
    source: str = Field(
        description="preagg=读预聚合表；live=预聚合缺席，已实时扫明细兜底"
    )
    bucket_count: int = Field(description="窗口内命中的分钟桶数，用于判断聚合是否滞后")
    metrics: OverviewMetricsOut
    queue: dict[str, int] = Field(
        default_factory=dict, description="AITask 按状态的实时计数"
    )


# ==================== 手动聚合 ====================


class AggregateRequest(BaseModel):
    """POST /monitoring/aggregate 请求体。

    三种粒度，优先级从高到低：
    1. 给 ``bucket_start`` → 只重算那一个桶（补某一天某分钟的数据）
    2. 给 ``lookback_minutes`` → 从上一个完整桶往前连续重算 N 个桶
    3. 都不给 → 只聚合上一个完整分钟桶（等价于 Beat 的一次触发）
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    bucket_start: Annotated[datetime | None, Field(default=None)] = Field(
        description="指定要重算的分钟桶起点（会被向下取整到分钟）"
    )
    lookback_minutes: Annotated[
        int | None, Field(default=None, ge=1, le=MAX_AGGREGATE_LOOKBACK_MINUTES)
    ] = Field(
        description=f"从上一个完整桶往前连续补算 N 个桶，上限 {MAX_AGGREGATE_LOOKBACK_MINUTES}"
    )

    @field_validator("bucket_start")
    @classmethod
    def _check_bucket(cls, v: datetime | None) -> datetime | None:
        return v


class AggregateBucketResult(BaseModel):
    """单个桶的聚合结果。"""

    bucket_start: datetime
    bucket_end: datetime
    written: int = 0
    deleted: int = 0
    ok: bool = True
    error: str | None = None


class AggregateResponse(BaseModel):
    """POST /monitoring/aggregate 响应。

    ``ok`` 表示**全部桶都成功**。整体失败不返回 500 而是 200 + ok=false，
    因为聚合是旁路观测操作，调用方（演示脚本/补数据任务）需要的是结构化结果
    而不是异常堆栈；具体原因在 ``error`` / 各桶的 error 里。
    """

    ok: bool
    buckets: list[AggregateBucketResult] = Field(default_factory=list)
    total_written: int = 0
    message: str = ""


__all__ = [
    "AggregateBucketResult",
    "AggregateRequest",
    "AggregateResponse",
    "LLMCallListResponse",
    "LLMCallOut",
    "MAX_AGGREGATE_LOOKBACK_MINUTES",
    "MAX_OVERVIEW_WINDOW_MINUTES",
    "MetricPointOut",
    "MetricSeriesResponse",
    "OverviewMetricsOut",
    "OverviewResponse",
    "RetrievalFragmentOut",
    "TimelineEntryOut",
    "TraceCountsOut",
    "TraceReplayResponse",
    "TraceSummaryOut",
    "TriageFragmentOut",
    "WorkflowFragmentOut",
]
