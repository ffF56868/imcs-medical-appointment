"""AI 运行监控路由：看板首屏、指标曲线、调用明细、全链路回放与手动聚合。

============================ 权限映射（rbac_matrix 为准，未改动矩阵） ============================

+-------------------------------------+--------------------------------+--------+
| 端点                                | require_permission             | 角色   |
+-------------------------------------+--------------------------------+--------+
| GET  /monitoring/overview           | ai_monitor:read_all            | ADMIN  |
| GET  /monitoring/metrics            | ai_monitor:read_all            | ADMIN  |
| GET  /monitoring/llm-calls          | ai_monitor:read_all            | ADMIN  |
| GET  /monitoring/traces/{trace_id}  | ai_monitor:read_all            | ADMIN  |
| POST /monitoring/aggregate          | ai_monitor:read_all            | ADMIN  |
+-------------------------------------+--------------------------------+--------+

**五个端点全部同一个闸门**：矩阵里 ``ai_monitor`` 只登记了
``("ai_monitor", "read_all")`` 一条，且仅授予 ADMIN（见 rbac_matrix.py，本批禁改）。
因此这里不存在 exam/billing 那种 read/read_all 双门问题——没有面向协作的单数 read，
监控本来就是纯运维视角。

``POST /aggregate`` 也挂 read_all 而不是另立一个动作，是**刻意的取舍**：
它只是把已经在跑的 Beat 任务提前触发一次，幂等覆盖同一分钟桶，不产生新数据、
不改任何业务状态；而新增一个权限动作必须动矩阵（禁改清单）。
副作用面与 GET 查询同级，用同一个闸门读起来最诚实。若日后要开放给值班工程师，
正确做法是在矩阵里补 ``("ai_monitor", "aggregate")`` 而不是在这里绕过权限。

============================ 为什么这些接口不给医生/患者 ============================

``llm_call_logs`` 带 ``user_id``、``workflow_traces`` 带 ``ref_id``、
``triage_results`` 带主诉原文——回放一条 trace 等于读到某位患者的完整 AI 诊疗过程。
这不是「运行指标」而是病历内容，因此授权位必须是全院视角的 read_all，
并且响应侧逐字段列出而非 ORM 全量导出（见 schemas/monitoring.py docstring）。

路由保持薄：解析参数 → 调 monitoring_service / metrics_service → 装配响应。
分位数口径、幂等落库、多表拼装全在服务层。
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, DbSession, require_permission
from app.models.auth import User
from app.schemas.monitoring import (
    MAX_AGGREGATE_LOOKBACK_MINUTES,
    MAX_OVERVIEW_WINDOW_MINUTES,
    AggregateBucketResult,
    AggregateRequest,
    AggregateResponse,
    LLMCallListResponse,
    LLMCallOut,
    MetricPointOut,
    MetricSeriesResponse,
    OverviewMetricsOut,
    OverviewResponse,
    RetrievalFragmentOut,
    TimelineEntryOut,
    TraceCountsOut,
    TraceReplayResponse,
    TraceSummaryOut,
    TriageFragmentOut,
    WorkflowFragmentOut,
)
from app.services import metrics_service as ms
from app.services import monitoring_service as monitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["AI监控"])

# 依赖工厂结果提成模块级私有变量再挂进 Annotated，保证同请求内相同依赖只解析一次
# （FastAPI 按依赖对象身份判定复用；每次现调工厂会对 permissions 表发多次查询）。
_PermissionDeps: dict[tuple[str, str], Any] = {}


def _gate(resource: str, action: str):
    key = (resource, action)
    if key not in _PermissionDeps:
        _PermissionDeps[key] = require_permission(resource, action)
    return _PermissionDeps[key]


CanReadMonitor = Annotated[User, Depends(_gate("ai_monitor", "read_all"))]


def _split_names(metric: str | None, metrics: str | None) -> list[str] | None:
    """合并 metric / metrics 两个参数为指标名列表。

    两个入口都留是因为前端形态不同：单指标曲线（画一张图）传 ``metric``，
    多指标叠加（错误率与降级率放同一坐标系对照）传逗号分隔的 ``metrics``。
    都不给则不限指标名，由 service 返回窗口内全部点。
    """
    names: list[str] = []
    if metric:
        names.append(metric.strip())
    if metrics:
        names.extend(part.strip() for part in metrics.split(",") if part.strip())
    cleaned = [n for n in names if n]
    return cleaned or None


# ==================== 看板首屏 ====================


@router.get("/overview", response_model=OverviewResponse, summary="看板首屏汇总")
async def get_overview(
    db: DbSession,
    current_user: CanReadMonitor,
    window_minutes: Annotated[
        int,
        Query(ge=1, le=MAX_OVERVIEW_WINDOW_MINUTES, description="统计窗口长度（分钟）"),
    ] = 60,
) -> OverviewResponse:
    """当前窗口的调用量、错误率、P95、成本、降级率等读数 + 队列实时快照。

    ``source`` 字段告诉前端数据来自预聚合表还是实时兜底：刚部署或 Beat 停摆时
    预聚合还没跑，看板会退化成现场扫明细（数字仍然对，但 bucket_count=0），
    这个区别必须可见，否则运维会误判「监控挂了」。
    """
    data = await monitor.overview(db, window_minutes)
    return OverviewResponse(
        window_minutes=data["window_minutes"],
        window_start=data["window_start"],
        window_end=data["window_end"],
        source=data["source"],
        bucket_count=data["bucket_count"],
        metrics=OverviewMetricsOut(**data["metrics"]),
        queue=data["queue"],
    )


# ==================== 指标时间序列 ====================


@router.get("/metrics", response_model=MetricSeriesResponse, summary="指标时间序列")
async def get_metrics(
    db: DbSession,
    current_user: CanReadMonitor,
    metric: Annotated[str | None, Query(description="单个指标名")] = None,
    metrics: Annotated[
        str | None, Query(description="多个指标名，逗号分隔；与 metric 可并存")
    ] = None,
    start_date: Annotated[
        datetime | None, Query(description="起始时间（含），naive 按 UTC 理解")
    ] = None,
    end_date: Annotated[datetime | None, Query(description="结束时间（不含）")] = None,
    tag: Annotated[
        str | None, Query(description='维度过滤，形如 model=qwen-max；只支持单个键值')
    ] = None,
    limit: Annotated[int, Query(ge=1, le=monitor.MAX_METRIC_LIMIT)] = 1440,
) -> MetricSeriesResponse:
    """从预聚合表 ``ai_metrics`` 取曲线点，按桶时间升序。

    ``tag`` 用 ``k=v`` 单键形式而不是 JSON：看板下钻的实际需求就是
    「只看某个模型的 token 用量」这一类，多键组合可以重复调用叠加。
    格式非法（没有等号）时忽略该过滤条件并在日志留痕，不让整个曲线因为
    一个拼错的筛选参数报错。
    """
    tags = _parse_tag(tag)
    names = _split_names(metric, metrics)
    points = await monitor.query_metrics(
        db, names, since=start_date, until=end_date, tags=tags, limit=limit
    )
    return MetricSeriesResponse(
        items=[MetricPointOut(**p) for p in points],
        count=len(points),
        limit=limit,
        metric_names=names or [],
    )


def _parse_tag(raw: str | None) -> dict[str, str] | None:
    """``model=qwen-max`` → {"model": "qwen-max"}；无法解析则返回 None。"""
    if not raw or "=" not in raw:
        if raw:
            logger.info("tag 参数格式非法，已忽略：%s", raw)
        return None
    key, _, value = raw.partition("=")
    key = key.strip()
    value = value.strip()
    if not key or not value:
        return None
    return {key: value}


# ==================== 调用明细 ====================


@router.get("/llm-calls", response_model=LLMCallListResponse, summary="LLM 调用明细")
async def list_llm_calls(
    db: DbSession,
    current_user: CanReadMonitor,
    scenario: Annotated[str | None, Query(description="按场景过滤")] = None,
    provider: Annotated[str | None, Query(description="按供应商过滤")] = None,
    model: Annotated[str | None, Query(description="按实际使用模型过滤")] = None,
    status_filter: Annotated[
        str | None, Query(description="按调用状态过滤：success/degraded/error/timeout")
    ] = None,
    trace_id: Annotated[str | None, Query(description="按链路 ID 精确过滤")] = None,
    user_id: Annotated[str | None, Query(description="按调用发起用户过滤")] = None,
    start_date: Annotated[datetime | None, Query(description="起始时间（含）")] = None,
    end_date: Annotated[datetime | None, Query(description="结束时间（不含）")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=monitor.MAX_PAGE_SIZE)] = 20,
) -> LLMCallListResponse:
    """明细下钻：看板上看到 P95 突刺后，用这个接口定位是哪几次调用慢。"""
    rows, total = await monitor.list_llm_calls(
        db,
        scenario=scenario,
        provider=provider,
        model=model,
        status_filter=status_filter,
        trace_id=trace_id,
        user_id=user_id,
        since=start_date,
        until=end_date,
        page=page,
        page_size=page_size,
    )
    return LLMCallListResponse(
        items=[LLMCallOut(**row) for row in rows],
        total=total,
        page=max(1, page),
        page_size=min(page_size, monitor.MAX_PAGE_SIZE),
    )


# ==================== 全链路回放 ====================


@router.get(
    "/traces/{trace_id}",
    response_model=TraceReplayResponse,
    summary="按 trace_id 回放全链路",
)
async def replay_trace(
    trace_id: str,
    db: DbSession,
    current_user: CanReadMonitor,
) -> TraceReplayResponse:
    """阶段 6 验收能力：任意一次问答可按 trace_id 还原完整决策链路。

    返回四段（LLM 调用 × N、RAG 检索、工作流追踪、分诊产出）+ 合并时间线 + 汇总。
    ``counts`` 里某段为 0 说明那一段没写日志——这本身就是排障线索，
    不是错误，所以接口不因缺段报错。**trace 完全不存在才 404**。
    """
    data = await monitor.replay_trace(db, trace_id)
    return TraceReplayResponse(
        trace_id=data["trace_id"],
        counts=TraceCountsOut(**data["counts"]),
        summary=TraceSummaryOut(**data["summary"]),
        llm_calls=[LLMCallOut(**item) for item in data["llm_calls"]],
        retrievals=[RetrievalFragmentOut(**item) for item in data["retrievals"]],
        workflows=[WorkflowFragmentOut(**item) for item in data["workflows"]],
        triage_results=[TriageFragmentOut(**item) for item in data["triage_results"]],
        timeline=[TimelineEntryOut(**item) for item in data["timeline"]],
    )


# ==================== 手动触发聚合 ====================


@router.post(
    "/aggregate",
    response_model=AggregateResponse,
    summary="手动触发指标聚合（补数据/演示）",
)
async def trigger_aggregate(
    payload: AggregateRequest,
    db: DbSession,
    current_user: CanReadMonitor,
) -> AggregateResponse:
    """把 Beat 的聚合逻辑同步跑一遍，便于演示与断档补数据。

    **为什么提供这个端点**：Beat 每分钟只算「上一个完整桶」，演示环境刚启动时
    看板是空的；而且 Beat 停摆一段时间后需要回填。两者都需要一个能立即执行、
    可指定桶、可批量回溯的入口。

    **为什么在当前请求会话里跑而不是派发 Celery**：派发会让接口变成「提交成功，
    结果未知」，演示与补数据恰恰需要同步拿到写了多少行；聚合本身是秒级操作。
    代价是这个请求会占住一条数据库连接直到算完——因此 lookback 上限设到
    ``MAX_AGGREGATE_LOOKBACK_MINUTES``，超出交给 Beat 自然补齐。

    幂等：重复调用同一桶只覆盖不新增（见 metrics_service 模块 docstring）。
    失败不抛 500：返回 200 + ok=false + 每桶 error，调用方按结构化结果判断。
    """
    buckets = _resolve_buckets(payload)
    results: list[AggregateBucketResult] = []
    total_written = 0
    all_ok = True

    for start, end in buckets:
        try:
            outcome = await ms.aggregate_minute_bucket(db, start, end)
            total_written += int(outcome.get("written", 0))
            results.append(
                AggregateBucketResult(
                    bucket_start=start,
                    bucket_end=end,
                    written=int(outcome.get("written", 0)),
                    deleted=int(outcome.get("deleted", 0)),
                    ok=True,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 单桶失败不影响其余桶
            logger.exception("手动聚合失败 bucket=%s", start.isoformat())
            all_ok = False
            results.append(
                AggregateBucketResult(
                    bucket_start=start,
                    bucket_end=end,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}"[:500],
                )
            )

    logger.info(
        "手动聚合完成 operator=%s buckets=%d written=%d ok=%s",
        current_user.id,
        len(results),
        total_written,
        all_ok,
    )
    return AggregateResponse(
        ok=all_ok,
        buckets=results,
        total_written=total_written,
        message=(
            f"已聚合 {len(results)} 个分钟桶，写入 {total_written} 行指标"
            if all_ok
            else f"共 {len(results)} 个桶，部分失败，详见各桶 error"
        ),
    )


def _resolve_buckets(payload: AggregateRequest) -> list[tuple[datetime, datetime]]:
    """把请求体解析成待聚合的桶区间列表（时间升序）。

    优先级：显式 bucket_start > lookback_minutes > 默认「上一个完整桶」。
    显式桶同样向下取整到分钟——传 12:03:47 时聚的是 12:03~12:04，
    让「指定桶」与 Beat 产出的桶对齐，否则会出现一堆非整分的孤儿桶行。
    """
    if payload.bucket_start is not None:
        start = payload.bucket_start
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        floored = start.astimezone(UTC).replace(second=0, microsecond=0)
        return [(floored, floored + timedelta(minutes=1))]

    if payload.lookback_minutes:
        last_end = datetime.now(UTC).replace(second=0, microsecond=0)
        count = max(1, min(int(payload.lookback_minutes), MAX_AGGREGATE_LOOKBACK_MINUTES))
        out: list[tuple[datetime, datetime]] = []
        for offset in range(count, 0, -1):
            end = last_end - timedelta(minutes=offset - 1)
            out.append((end - timedelta(minutes=1), end))
        return out

    start, end = ms.previous_complete_bucket()
    return [(start, end)]


__all__ = ["router"]
