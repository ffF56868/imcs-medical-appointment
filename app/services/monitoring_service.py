"""AI 监控看板查询侧服务（阶段 6 批次 A，只读）。

============================ 与 metrics_service 的分工 ============================

``metrics_service`` 负责**写**：把明细日志按分钟桶折叠成 ``ai_metrics`` 行。
本模块负责**读**：看板首屏、时间序列曲线、调用明细列表、以及阶段 6 验收的
核心能力——按 trace_id 回放一次问答的完整链路。

分开两个文件而不是一个是因为读写关注点差异极大：写侧是批处理、幂等覆盖、
Python 侧算分位；读侧是分页、过滤、多表拼装。混在一起会让「改一个查询排序」
和「改一个聚合口径」两类变更互相踩脚。

============================ 为什么看板要查预聚合而不是明细 ============================

见 ``metrics_service`` 模块 docstring。一句话：曲线走 ``ai_metrics``（小表），
下钻走明细表（``llm_call_logs`` 等）。用户在看板上看到 P95 突刺，
点进去才扫明细定位是哪几次调用——两层各司其职，响应时间与数据规模解耦。

============================ 时间参数的宽容解析 ============================

``since`` / ``until`` 接受 date 或 datetime：API 层的 ``start_date=2026-10-13``
这种纯日期查询很常见（前端日历控件默认给日粒度）。把它当成当天 00:00 UTC 而不是
要求调用方必须补时分秒，是为了让「看今天一整天的曲线」不需要客户端猜时区。
naive datetime 一律按 UTC 理解，与 models/base.utcnow 的全项目约定一致。

============================ replay_trace 的取舍 ============================

四张表（llm_call_logs / rag_retrieval_logs / workflow_traces / triage_results）
都各自持有 trace_id 但**没有外键关联**——它们是异步写入路径分别落盘的旁路日志，
用外键绑死会让任何一张表的写入失败连带阻塞业务请求。代价是回放只能靠
trace_id 字符串匹配，某一段日志缺失就是缺失（返回空数组而不是报错）。

这恰恰是回放功能最有用的形态：排障时最想看的正是「哪一段没记上」。
因此返回值里每段都带自己的行数，缺段一目了然。
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ResourceNotFoundError
from app.models.ai import (
    AIMetric,
    AITask,
    LLMCallLog,
    RAGRetrievalLog,
    TriageResult,
    WorkflowTrace,
)
from app.services.metrics_service import (
    ERROR_STATUSES,
    METRIC_DEGRADATION_RATE,
    METRIC_LLM_CALL_COUNT,
    METRIC_LLM_ERROR_RATE,
    METRIC_LLM_P95_LATENCY,
    METRIC_MODERATION_BLOCK_RATE,
    METRIC_RETRIEVAL_ZERO_RESULT_RATE,
    METRIC_TASK_QUEUE_BACKLOG,
    METRIC_TOKEN_COST,
    METRIC_TOKEN_USAGE,
    nearest_rank_percentile,
    safe_ratio,
)

logger = logging.getLogger(__name__)

# 单次查询返回的指标点上限。一分钟一个桶，1440 个点是一整天；
# 再大就不是给人看的曲线了，也会把响应体撑肥。
DEFAULT_METRIC_LIMIT = 1440
MAX_METRIC_LIMIT = 5000

# 明细分页上限
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# overview 默认窗口（分钟）。看板首屏看最近一小时足够判断「现在有没有问题」，
# 更长区间交给 /metrics 曲线。
DEFAULT_OVERVIEW_WINDOW_MINUTES = 60


# ==================== 内部工具 ====================


def _as_utc_start(value: datetime | date | None) -> datetime | None:
    """时间下界：date → 当日 00:00 UTC；naive datetime → 按 UTC 附加时区。"""
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC)


def _as_utc_end(value: datetime | date | None) -> datetime | None:
    """时间上界。纯日期取**次日零点**（左闭右开），这样「查到 10-13」含 13 号全天。"""
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, tzinfo=UTC) + timedelta(days=1)
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC)


def _stamp(moment: datetime | None) -> datetime | None:
    """对外时间戳统一成 aware UTC。

    ``DateTime(timezone=True)`` 在 PG 上读写都带时区，但 SQLite 不保存 tzinfo，
    读回来是 naive——同一个字面时刻在两个环境下形态不同。回放接口要按
    created_at 合并四类数据排序、前端要拿它和别处时间做比较，
    naive/aware 混用会直接 ``TypeError`` 或比较恒假。这里统一补 UTC，
    让两种方言的响应契约一致。
    """
    if moment is None:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _uuid_str(value: Any) -> str | None:
    """UUID / str 统一成对外字符串，None 原样传出去。"""
    if value is None:
        return None
    return str(value)


def _parse_uuid(value: str | None) -> uuid.UUID | None:
    """查询参数里的 UUID；形态非法当作「不匹配任何记录」。

    这里不抛 400：监控接口传了个乱码 user_id，最合理的结果是没有这条数据，
    而不是让管理员的看板因为一个拼错的筛选条件整块报错。

    **哨兵必须是合法 UUID**：早先写 ``uuid.UUID(int=-1)`` 想造一个「不可能命中」
    的值，实际却直接抛 ``ValueError: int is out of range``——本意是容错的函数
    反而成了新的崩溃点，恰好违背它 docstring 的承诺。改用 nil UUID：
    业务侧 user_id 全部由 uuid4 生成，``00000000-0000-0000-0000-000000000000``
    不可能出现在数据里，作为「查不到」的哨兵既合法又安全。
    """
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return uuid.UUID(int=0)


def _tags_match(row_tags: dict[str, Any] | None, wanted: dict[str, Any]) -> bool:
    """行 tags 是否包含全部期望键值（子集匹配，不是全等）。

    看板的典型诉求是「token_usage 里 model=qwen-max 那条线」——
    它只关心一个维度，而落库 tags 可能还带 provider 之类。全等匹配会让这种
    最常见的下钻查询永远返回空。
    """
    if not wanted:
        return True
    if not row_tags:
        return False
    for key, expect in wanted.items():
        actual = row_tags.get(key)
        if actual is None:
            return False
        # JSON 文本落库后数字可能变 int 也可能变 str，比较前归一
        if str(actual) != str(expect):
            return False
    return True


def _metric_point(row: AIMetric) -> dict[str, Any]:
    """指标行 → 曲线点。value 转 float 供前端直接消费。"""
    return {
        "bucket_time": _stamp(row.bucket_time),
        "metric_name": row.metric_name,
        "metric_value": float(row.metric_value),
        "sample_count": row.sample_count,
        "tags": row.tags or {},
    }


# ==================== 时间序列查询 ====================


async def query_metrics(
    db: AsyncSession,
    metric_names: Sequence[str] | str | None = None,
    since: datetime | date | None = None,
    until: datetime | date | None = None,
    tags: dict[str, Any] | None = None,
    limit: int = DEFAULT_METRIC_LIMIT,
) -> list[dict[str, Any]]:
    """从 ai_metrics 取时间序列点，按 (bucket_time, metric_name) 升序。

    - ``metric_names``：单个名字、名字列表、或 None（不限）。
    - ``tags``：子集匹配（见 ``_tags_match``）。因为是 Python 侧过滤，
      所以先按时间窗把候选捞出来再筛——PG 上可换 JSONB ``@>`` 下推，
      但 SQLite 没有该运算符，为保持两端行为一致统一在应用层过滤。
    - ``limit``：截断到 MAX_METRIC_LIMIT，超出按上限执行而不是报错。
    """
    names = (
        None
        if metric_names is None
        else [metric_names]
        if isinstance(metric_names, str)
        else list(metric_names)
    )

    statement = select(AIMetric)
    if names:
        statement = statement.where(AIMetric.metric_name.in_(names))
    start = _as_utc_start(since)
    end = _as_utc_end(until)
    if start is not None:
        statement = statement.where(AIMetric.bucket_time >= start)
    if end is not None:
        statement = statement.where(AIMetric.bucket_time < end)
    statement = statement.order_by(AIMetric.bucket_time.asc(), AIMetric.metric_name.asc())

    # 注意不能用 ``limit or DEFAULT``：0 是合法入参（想只要 1 个点），
    # 但 ``0 or DEFAULT`` 会把它当缺省吞掉。只有 None 才是「没传」。
    raw_limit = DEFAULT_METRIC_LIMIT if limit is None else limit
    try:
        requested = int(raw_limit)
    except (TypeError, ValueError):
        requested = DEFAULT_METRIC_LIMIT
    effective_limit = max(1, min(requested, MAX_METRIC_LIMIT))
    rows = list((await db.execute(statement.limit(effective_limit))).scalars().all())

    if tags:
        rows = [row for row in rows if _tags_match(row.tags, tags)]

    return [_metric_point(row) for row in rows]


# ==================== 调用明细 ====================


def _serialize_llm_log(row: LLMCallLog) -> dict[str, Any]:
    """LLM 调用日志 → 明细行。

    刻意逐字段列出而不用 ORM 全量导出：这张表虽然不含口令这类敏感列，
    但 prompt/completion 原文不在这里（模型本身也没存），字段清单稳定
    可以让前端契约不受模型加列影响。
    """
    return {
        "id": row.id,
        "trace_id": row.trace_id,
        "scenario": row.scenario,
        "provider": row.provider,
        "model": row.model,
        "requested_model": row.requested_model,
        "prompt_tokens": row.prompt_tokens,
        "completion_tokens": row.completion_tokens,
        "total_tokens": row.total_tokens,
        "cost_usd": float(row.cost_usd) if row.cost_usd is not None else None,
        "latency_ms": row.latency_ms,
        "first_token_ms": row.first_token_ms,
        "status": row.status,
        "error_type": row.error_type,
        "http_status": row.http_status,
        "retry_count": row.retry_count,
        "degradation_level": row.degradation_level,
        "moderation_blocked": bool(row.moderation_blocked),
        "budget_exceeded": bool(row.budget_exceeded),
        "is_stream": bool(row.is_stream),
        "user_id": _uuid_str(row.user_id),
        "workflow_node": row.workflow_node,
        "ai_task_id": _uuid_str(row.ai_task_id),
        "created_at": _stamp(row.created_at),
    }


async def list_llm_calls(
    db: AsyncSession,
    *,
    scenario: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    status_filter: str | None = None,
    trace_id: str | None = None,
    user_id: str | None = None,
    since: datetime | date | None = None,
    until: datetime | date | None = None,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> tuple[list[dict[str, Any]], int]:
    """LLM 调用明细分页查询，返回 (rows, total)。

    排序按 created_at 倒序（最新在前）+ id 兜底：同一毫秒内的多条日志
    仅按 created_at 排序不稳定，翻页会出现重复行或漏行。
    """
    size = max(1, min(int(page_size or DEFAULT_PAGE_SIZE), MAX_PAGE_SIZE))
    current_page = max(1, int(page or 1))

    conditions: list[Any] = []
    if scenario:
        conditions.append(LLMCallLog.scenario == scenario)
    if provider:
        conditions.append(LLMCallLog.provider == provider)
    if model:
        conditions.append(LLMCallLog.model == model)
    if status_filter:
        conditions.append(LLMCallLog.status == status_filter)
    if trace_id:
        conditions.append(LLMCallLog.trace_id == trace_id)
    parsed_user = _parse_uuid(user_id)
    if user_id and parsed_user is not None:
        conditions.append(LLMCallLog.user_id == parsed_user)
    start = _as_utc_start(since)
    end = _as_utc_end(until)
    if start is not None:
        conditions.append(LLMCallLog.created_at >= start)
    if end is not None:
        conditions.append(LLMCallLog.created_at < end)

    total = int(
        (
            await db.execute(
                select(func.count()).select_from(LLMCallLog).where(*conditions)
            )
        ).scalar_one()
        or 0
    )

    statement = (
        select(LLMCallLog)
        .where(*conditions)
        .order_by(LLMCallLog.created_at.desc(), LLMCallLog.id.desc())
        .offset((current_page - 1) * size)
        .limit(size)
    )
    rows = list((await db.execute(statement)).scalars().all())
    return [_serialize_llm_log(row) for row in rows], total


# ==================== 全链路回放 ====================


def _serialize_rag_log(row: RAGRetrievalLog) -> dict[str, Any]:
    """检索日志 → 回放片段：各路召回数 + 融合后 top-k 明细。"""
    return {
        "id": row.id,
        "trace_id": row.trace_id,
        "user_id": _uuid_str(row.user_id),
        "query_text": row.query_text,
        "rewritten_query": row.rewritten_query,
        "vector_hit_count": row.vector_hit_count,
        "keyword_hit_count": row.keyword_hit_count,
        "both_hit_count": row.both_hit_count,
        "merged_count": row.merged_count,
        "final_count": row.final_count,
        "is_zero_result": bool(row.is_zero_result),
        "top_score": float(row.top_score) if row.top_score is not None else None,
        "vector_latency_ms": row.vector_latency_ms,
        "keyword_latency_ms": row.keyword_latency_ms,
        "rrf_latency_ms": row.rrf_latency_ms,
        "rerank_latency_ms": row.rerank_latency_ms,
        "rerank_channel": row.rerank_channel,
        "rrf_k": row.rrf_k,
        "candidates": row.candidates or [],
        "created_at": _stamp(row.created_at),
    }


def _serialize_workflow_trace(row: WorkflowTrace) -> dict[str, Any]:
    """工作流追踪 → 回放片段：节点 steps + 分支走向。"""
    return {
        "id": row.id,
        "trace_id": row.trace_id,
        "workflow_name": row.workflow_name,
        "status": row.status,
        "steps": row.steps or [],
        "node_count": row.node_count,
        "executed_node_count": row.executed_node_count,
        "total_latency_ms": row.total_latency_ms,
        "total_tokens": row.total_tokens,
        "total_cost_usd": (
            float(row.total_cost_usd) if row.total_cost_usd is not None else None
        ),
        "branch_taken": row.branch_taken,
        "final_output": row.final_output,
        "error_message": row.error_message,
        "user_id": _uuid_str(row.user_id),
        "ref_id": _uuid_str(row.ref_id),
        "created_at": _stamp(row.created_at),
        "completed_at": row.completed_at,
    }


def _serialize_triage_result(row: TriageResult) -> dict[str, Any]:
    """分诊结果 → 回放片段（业务侧最终产出，含推荐科室与紧急度）。"""
    return {
        "id": _uuid_str(row.id),
        "trace_id": row.trace_id,
        "patient_id": _uuid_str(row.patient_id),
        "user_id": _uuid_str(row.user_id),
        "chief_complaint": row.chief_complaint,
        "extracted_symptoms": row.extracted_symptoms or [],
        "recommended_department_id": _uuid_str(row.recommended_department_id),
        "recommended_department_name": row.recommended_department_name,
        "urgency": row.urgency,
        "is_emergency": bool(row.is_emergency),
        "matched_emergency_keywords": row.matched_emergency_keywords or [],
        "advice": row.advice,
        "confidence": float(row.confidence) if row.confidence is not None else None,
        "decision_source": row.decision_source,
        "ai_task_id": _uuid_str(row.ai_task_id),
        "llm_model": row.llm_model,
        "converted_appointment_id": _uuid_str(row.converted_appointment_id),
        "actual_department_id": _uuid_str(row.actual_department_id),
        "feedback_rating": row.feedback_rating,
        "created_at": _stamp(row.created_at),
    }


def _timeline_entry(kind: str, moment: Any, summary: str, ref: dict[str, Any]) -> dict[str, Any]:
    """时间线条目：四类来源共用一个结构，前端可直接按时间渲染。"""
    return {"kind": kind, "at": moment, "summary": summary, "ref": ref}


async def replay_trace(db: AsyncSession, trace_id: str) -> dict[str, Any]:
    """按 trace_id 回放一次问答的全链路（阶段 6 验收核心能力）。

    聚合四类数据，每类都按 created_at 升序（同刻用 id 兜底保证稳定）：

    1. ``llm_calls`` —— 该链路里的**每一次** LLM 调用（一次问答常有提取症状 +
       生成建议两次），含 tokens / latency / degradation / cost。
    2. ``retrievals`` —— RAG 检索日志：原始 query、改写 query、各路命中数、
       融合后 candidates（回答「当时喂给模型的证据是什么」）。
    3. ``workflows`` —— 工作流追踪：节点 steps、branch_taken（走了急诊还是普通分支）。
    4. ``triage_results`` —— 分诊业务产出（若该链路是分诊）。

    额外给出 ``timeline``：把上述四段按时间合并成一条有序事件流，
    排障时看这一个数组就能定位「哪一步开始变慢/失败」。

    **找不到任何记录 → 抛 ResourceNotFoundError（404）**：
    传了个不存在/已清理的 trace_id 与「trace 存在但各段为空」是两回事，
    前者是调用方参数错误，后者是有意义的观测结果。
    """
    if not trace_id or not trace_id.strip():
        raise ResourceNotFoundError("trace_id 不能为空")

    tid = trace_id.strip()

    llm_rows = list(
        (
            await db.execute(
                select(LLMCallLog)
                .where(LLMCallLog.trace_id == tid)
                .order_by(LLMCallLog.created_at.asc(), LLMCallLog.id.asc())
            )
        ).scalars().all()
    )
    rag_rows = list(
        (
            await db.execute(
                select(RAGRetrievalLog)
                .where(RAGRetrievalLog.trace_id == tid)
                .order_by(RAGRetrievalLog.created_at.asc(), RAGRetrievalLog.id.asc())
            )
        ).scalars().all()
    )
    wf_rows = list(
        (
            await db.execute(
                select(WorkflowTrace)
                .where(WorkflowTrace.trace_id == tid)
                .order_by(WorkflowTrace.created_at.asc(), WorkflowTrace.id.asc())
            )
        ).scalars().all()
    )
    triage_rows = list(
        (
            await db.execute(
                select(TriageResult)
                .where(TriageResult.trace_id == tid)
                .order_by(TriageResult.created_at.asc())
            )
        ).scalars().all()
    )

    if not (llm_rows or rag_rows or wf_rows or triage_rows):
        raise ResourceNotFoundError(f"未找到 trace_id={tid} 的任何链路日志")

    llm_calls = [_serialize_llm_log(row) for row in llm_rows]
    retrievals = [_serialize_rag_log(row) for row in rag_rows]
    workflows = [_serialize_workflow_trace(row) for row in wf_rows]
    triage_results = [_serialize_triage_result(row) for row in triage_rows]

    # ---------- 汇总 ----------
    total_prompt = sum(item["prompt_tokens"] for item in llm_calls)
    total_completion = sum(item["completion_tokens"] for item in llm_calls)
    total_cost = sum(item["cost_usd"] or 0.0 for item in llm_calls)
    latencies = [item["latency_ms"] for item in llm_calls]
    error_count = sum(1 for item in llm_calls if (item["status"] or "") in ERROR_STATUSES)
    degraded_count = sum(
        1
        for item in llm_calls
        if (item["status"] or "") == "degraded"
        or int(item["degradation_level"] or 0) > 0
    )

    timeline: list[dict[str, Any]] = []
    for item in llm_calls:
        timeline.append(
            _timeline_entry(
                "llm_call",
                item["created_at"],
                f"{item['scenario']}/{item['model']} · {item['latency_ms']}ms · "
                f"{item['total_tokens']}tok · {item['status']}",
                {"id": item["id"], "workflow_node": item["workflow_node"]},
            )
        )
    for item in retrievals:
        timeline.append(
            _timeline_entry(
                "retrieval",
                item["created_at"],
                f"向量 {item['vector_hit_count']} · 关键词 {item['keyword_hit_count']} · "
                f"融合后 {item['final_count']} 条"
                + ("（零结果）" if item["is_zero_result"] else ""),
                {"id": item["id"]},
            )
        )
    for item in workflows:
        timeline.append(
            _timeline_entry(
                "workflow",
                item["created_at"],
                f"{item['workflow_name']} · {item['status']} · "
                f"分支={item['branch_taken'] or '-'} · 节点 {item['executed_node_count']}/{item['node_count']}",
                {"id": item["id"]},
            )
        )
    for item in triage_results:
        timeline.append(
            _timeline_entry(
                "triage_result",
                item["created_at"],
                f"紧急度 {item['urgency']} · 推荐 {item['recommended_department_name'] or '-'} · "
                f"来源 {item['decision_source']}",
                {"id": item["id"]},
            )
        )
    # 时间线排序：SQLite 读回的 DateTime 可能是 naive，PG 是 aware，
    # 两者混排会 TypeError。统一折算成可比较的键（naive 视作 UTC）。
    timeline.sort(key=_timeline_sort_key)

    first_at = timeline[0]["at"] if timeline else None
    last_at = timeline[-1]["at"] if timeline else None

    return {
        "trace_id": tid,
        "counts": {
            "llm_calls": len(llm_calls),
            "retrievals": len(retrievals),
            "workflows": len(workflows),
            "triage_results": len(triage_results),
        },
        "summary": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "cost_usd": round(total_cost, 6),
            "call_count": len(llm_calls),
            "error_count": error_count,
            "degraded_count": degraded_count,
            "max_latency_ms": max(latencies) if latencies else None,
            "p95_latency_ms": nearest_rank_percentile(latencies, 0.95),
            "started_at": first_at,
            "ended_at": last_at,
            "providers": sorted({item["provider"] for item in llm_calls}),
            "models": sorted({item["model"] for item in llm_calls}),
            "scenarios": sorted({item["scenario"] for item in llm_calls}),
        },
        "llm_calls": llm_calls,
        "retrievals": retrievals,
        "workflows": workflows,
        "triage_results": triage_results,
        "timeline": timeline,
    }


def _timeline_sort_key(entry: dict[str, Any]) -> tuple[float, str]:
    """时间线排序键：时刻 → epoch 秒（naive 视为 UTC），无时刻排最后。"""
    moment = entry.get("at")
    if isinstance(moment, datetime):
        aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
        return (aware.timestamp(), "")
    return (float("inf"), str(moment))


# ==================== 看板首屏 ====================


def _latest_metric_value(points: list[dict[str, Any]]) -> float | None:
    """取时间序列最后一个点的值（曲线当前读数）。"""
    return points[-1]["metric_value"] if points else None


async def overview(db: AsyncSession, window_minutes: int = DEFAULT_OVERVIEW_WINDOW_MINUTES) -> dict[str, Any]:
    """看板首屏汇总。

    **两段数据来源，各有分工**：

    - 曲线型读数（调用量、错误率、P95、成本、降级率…）取自 ``ai_metrics``
      预聚合表——这是看板的主路径，响应时间与明细表规模解耦。
      注意 ``task_queue_backlog`` 取窗口内**最后一个桶**的值而不是求和：
      积压是瞬时快照，把 60 个桶加起来等于把同一个任务数了 60 遍。
    - 明细兜底：当预聚合还没跑到（新部署、Beat 停摆、手动补数据之前），
      曲线区会是空的。此时直接从明细表实时算一遍当前窗口，
      并在返回里标 ``source="live"`` 告诉前端「这些数没经过预聚合」。
      没有这个兜底，刚上线的演示环境打开看板会是一片 0，
      看起来像系统坏了而不是数据没聚合。

    另外始终附带 ``queue``（AITask 实时状态分布）——它是运维关心的即时事实，
    不该有分钟级延迟。
    """
    # 同 query_metrics 的 limit：0 是合法入参（想只看最近一分钟），
    # ``window_minutes or DEFAULT`` 会把 0 当缺省吞成 60。
    raw_window = (
        DEFAULT_OVERVIEW_WINDOW_MINUTES if window_minutes is None else window_minutes
    )
    try:
        minutes = max(1, int(raw_window))
    except (TypeError, ValueError):
        minutes = DEFAULT_OVERVIEW_WINDOW_MINUTES
    now = datetime.now(UTC)
    start = now - timedelta(minutes=minutes)

    names = [
        METRIC_LLM_CALL_COUNT,
        METRIC_LLM_ERROR_RATE,
        METRIC_LLM_P95_LATENCY,
        METRIC_TOKEN_USAGE,
        METRIC_TOKEN_COST,
        METRIC_DEGRADATION_RATE,
        METRIC_MODERATION_BLOCK_RATE,
        METRIC_RETRIEVAL_ZERO_RESULT_RATE,
        METRIC_TASK_QUEUE_BACKLOG,
    ]
    points = await query_metrics(db, names, since=start, until=now, limit=MAX_METRIC_LIMIT)

    by_name: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        by_name.setdefault(point["metric_name"], []).append(point)

    source = "preagg" if by_name else "live"
    live: dict[str, Any] = {}
    if not by_name:
        live = await _overview_from_details(db, start, now)

    token_series = by_name.get(METRIC_TOKEN_USAGE, [])
    token_total = sum(p["metric_value"] for p in token_series)

    metrics_block = {
        "llm_call_count": _latest_metric_value(by_name.get(METRIC_LLM_CALL_COUNT, [])),
        "llm_error_rate": _latest_metric_value(by_name.get(METRIC_LLM_ERROR_RATE, [])),
        "llm_p95_latency_ms": _latest_metric_value(by_name.get(METRIC_LLM_P95_LATENCY, [])),
        "token_usage_total": token_total if token_series else live.get("token_usage_total"),
        "token_cost_usd": _latest_metric_value(by_name.get(METRIC_TOKEN_COST, [])),
        "degradation_rate": _latest_metric_value(by_name.get(METRIC_DEGRADATION_RATE, [])),
        "moderation_block_rate": _latest_metric_value(
            by_name.get(METRIC_MODERATION_BLOCK_RATE, [])
        ),
        "retrieval_zero_result_rate": _latest_metric_value(
            by_name.get(METRIC_RETRIEVAL_ZERO_RESULT_RATE, [])
        ),
        # 积压取最后一个桶的值，理由见函数 docstring
        "task_queue_backlog": _latest_metric_value(
            by_name.get(METRIC_TASK_QUEUE_BACKLOG, [])
        ),
    }
    if source == "live":
        for key, value in live.items():
            if metrics_block.get(key) is None:
                metrics_block[key] = value

    bucket_count = len({p["bucket_time"] for p in points})

    return {
        "window_minutes": minutes,
        "window_start": start,
        "window_end": now,
        "source": source,
        "bucket_count": bucket_count,
        "metrics": metrics_block,
        "queue": await _queue_snapshot(db),
    }


async def _queue_snapshot(db: AsyncSession) -> dict[str, int]:
    """AITask 按状态的实时计数（运维视角的即时事实）。"""
    rows = (
        await db.execute(select(AITask.status, func.count()).group_by(AITask.status))
    ).all()
    return {str(status_name): int(count or 0) for status_name, count in rows}


async def _overview_from_details(
    db: AsyncSession, start: datetime, end: datetime
) -> dict[str, Any]:
    """预聚合缺席时，直接对明细表现算窗口汇总（口径与 metrics_service 一致）。"""
    rows = list(
        (
            await db.execute(
                select(
                    LLMCallLog.status,
                    LLMCallLog.latency_ms,
                    LLMCallLog.total_tokens,
                    LLMCallLog.cost_usd,
                    LLMCallLog.degradation_level,
                    LLMCallLog.moderation_blocked,
                ).where(LLMCallLog.created_at >= start, LLMCallLog.created_at < end)
            )
        ).all()
    )
    call_count = len(rows)
    error_count = sum(1 for r in rows if (r.status or "") in ERROR_STATUSES)
    degraded_count = sum(
        1 for r in rows if (r.status or "") == "degraded" or int(r.degradation_level or 0) > 0
    )
    blocked_count = sum(1 for r in rows if bool(r.moderation_blocked))

    rag_total, rag_zero = await _rag_counts(db, start, end)
    backlog = int(
        (
            await db.execute(
                select(func.count()).select_from(AITask).where(
                    AITask.status.in_(("pending", "running"))
                )
            )
        ).scalar_one()
        or 0
    )

    cost_sum = sum(float(r.cost_usd or 0) for r in rows)
    return {
        "llm_call_count": float(call_count),
        "llm_error_rate": safe_ratio(error_count, call_count),
        "llm_p95_latency_ms": nearest_rank_percentile(
            [int(r.latency_ms or 0) for r in rows], 0.95
        ),
        "token_usage_total": float(sum(int(r.total_tokens or 0) for r in rows)),
        "token_cost_usd": round(cost_sum, 6) if call_count else None,
        "degradation_rate": safe_ratio(degraded_count, call_count),
        "moderation_block_rate": safe_ratio(blocked_count, call_count),
        "retrieval_zero_result_rate": safe_ratio(rag_zero, rag_total),
        "task_queue_backlog": float(backlog),
    }


async def _rag_counts(
    db: AsyncSession, start: datetime, end: datetime
) -> tuple[int, int]:
    """窗口内检索日志的 (总数, 零结果数)。"""
    total = int(
        (
            await db.execute(
                select(func.count()).select_from(RAGRetrievalLog).where(
                    RAGRetrievalLog.created_at >= start, RAGRetrievalLog.created_at < end
                )
            )
        ).scalar_one()
        or 0
    )
    zero = int(
        (
            await db.execute(
                select(func.count()).select_from(RAGRetrievalLog).where(
                    RAGRetrievalLog.created_at >= start,
                    RAGRetrievalLog.created_at < end,
                    RAGRetrievalLog.is_zero_result.is_(True),
                )
            )
        ).scalar_one()
        or 0
    )
    return total, zero


__all__ = [
    "DEFAULT_METRIC_LIMIT",
    "DEFAULT_OVERVIEW_WINDOW_MINUTES",
    "MAX_METRIC_LIMIT",
    "MAX_PAGE_SIZE",
    "list_llm_calls",
    "overview",
    "query_metrics",
    "replay_trace",
]
