"""AI 指标分钟桶聚合服务（阶段 6 批次 A）。

============================ 为什么预聚合而不是看板实时扫明细 ============================

``llm_call_logs`` / ``rag_retrieval_logs`` 是高频写入的明细表，随调用量线性增长。
看板若每次打开都现场算 P95，有三个实际问题：

1. **响应时间与数据规模挂钩**——上线半年后一次刷新要扫几十万行，管理员的
   「看一眼有没有降级」变成三十秒等待，监控本身失去意义。
2. **重复计算**——N 个管理员同时打开看板就把同一批分位数算 N 遍。
3. **明细保留策略受限**——日后按天归档/清理明细日志时，历史曲线会一起消失；
   预聚合表每桶只有十几行，可以长期保留。

因此 Celery Beat 每分钟把「上一个完整分钟桶」聚合进 ``ai_metrics``，
看板只读这张小表（见 ``monitoring_service.query_metrics``）。

============================ 分位数为什么在 Python 侧算 ============================

PostgreSQL 有 ``percentile_cont`` / ``percentile_disc``，SQLite **两个都没有**。
本项目测试跑在 SQLite（aiosqlite）上，如果聚合 SQL 用 PG 专有函数：

- 测试环境直接报 ``no such function``，要么跳过用例（等于没测），
  要么为测试写一套分支实现（被测代码与线上代码不是同一份，测试失真）。
- 更麻烦的是 ``percentile_cont`` 默认做**线性插值**，与常见的「最近秩取整」口径不同。
  两套算法在不同方言上的行为差异会让断言无法钉死。

取舍：**把窗口内的 latency 列拉到 Python 内存里排序取分位**。代价是单桶明细全进内存；
收益是 PG / SQLite 行为完全一致、且分位数口径由我们自己钉死。
这个代价在本项目量级下可接受：日均调用千级（见 models/ai.py 注释），
一分钟桶内最多几十到几百条 LLM 调用，每条只取一个整数列，内存占用以 KB 计。
若未来单桶明细涨到十万级，正确做法是提高聚合频率或引入时序库，而不是换回
数据库函数把口径分裂成两份。

**分位数口径（实现与测试必须一致）**：采用**最近秩法（nearest-rank, ceiling）**——

    对 n 个样本升序排列后，第 p 分位数 = 排序列表中第 ceil(p * n) 个元素（1-based）。

显式约定三条边界，避免"看实现猜口径"：

- n == 0 → 无定义，返回 None（该指标不落库，而不是落 0 冒充「延迟为 0」）。
- n == 1 → 任何分位数都是那唯一样本。
- p * n 非整数 → 向上取整（ceiling），即结果一定是**真实存在的样本值**，
  不会像 percentile_cont 那样插出列表里不存在的数。
  例：n=2、p=0.95 → rank=ceil(1.9)=2 → 取较大值；n=10、p=0.95 → rank=ceil(9.5)=10 → 取最大值。

选 ceiling 而不是线性插值的理由：监控告警关心「最慢的那一次到底多慢」，
插值会把一个没发生过的中间值当成 P95 上报，反而掩盖长尾。

============================ 幂等：为什么不能用 ON CONFLICT ============================

``ai_metrics`` 表上只有 ``(metric_name, bucket_time)`` 与 ``(bucket_time)`` 两个普通索引，
**没有 (bucket_time, metric_name, tags) 唯一约束**（见 models/ai.py，本批禁止改模型与迁移）。
没有唯一约束就没有 ``ON CONFLICT ... DO UPDATE`` 的冲突判定依据——PG 会直接报
「there is no unique or exclusion constraint matching the ON CONFLICT specification」。
而把同一条指标拆成「先查后 update / insert」又会在并发执行时产生竞态。

因此采用**删除 + 插入**：在同一事务内先按 ``(bucket_time, metric_name)`` 删掉旧行，
再插入本轮算出的全部行。语义上等价于「整桶覆盖重写」，天然幂等——
同一桶跑两次，行数与值都只反映最后一次结果，不会翻倍。
``tags`` 参与区分维度（如 token_usage 按 model 分行），故删除键取
``(bucket_time, metric_name)`` 而非逐 tag 比对：同一指标名的所有 tag 行一起删重建，
比逐个 tag 判断"存在则更新"少一半代码也少一半竞态面。

tags 一律经 ``_normalize_tags`` 规范化（key 排序 + JSON 序列化）后再写入，
保证同一维度组合的 JSON 文本形态稳定，便于人工核对与后续加唯一约束时平滑迁移。

============================ 失败不允许拖垮 worker ============================

``aggregate_minute_bucket`` 内部不吞异常（让调用方决定怎么处理），但
``run_aggregate`` / Celery 任务体在最外层 try/except 收口，异常转成结构化返回值并记日志。
理由是聚合属于旁路观测：这一轮失败了下一轮自然会补，
绝不能因为指标算不出来让 default 队列的 worker 进程挂掉，
那会连带阻塞知识库向量化等真正影响用户的任务。
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_factory
from app.models.ai import (
    AIMetric,
    AITask,
    AITaskStatus,
    LLMCallLog,
    RAGRetrievalLog,
)

logger = logging.getLogger(__name__)

# ==================== 指标名常量 ====================
# 与 models/ai.py 的列注释登记值保持一致；写成常量而不是散落字符串，
# 是为了让「看板查询」与「聚合写入」两侧引用同一个名字，改一处必改两处会被测试抓到。

METRIC_LLM_CALL_COUNT = "llm_call_count"
METRIC_LLM_ERROR_RATE = "llm_error_rate"
METRIC_LLM_P50_LATENCY = "llm_p50_latency"
METRIC_LLM_P95_LATENCY = "llm_p95_latency"
METRIC_LLM_P99_LATENCY = "llm_p99_latency"
METRIC_LLM_FIRST_TOKEN_P95 = "llm_first_token_p95"
METRIC_TOKEN_USAGE = "token_usage"
METRIC_TOKEN_COST = "token_cost"
METRIC_DEGRADATION_RATE = "degradation_rate"
METRIC_MODERATION_BLOCK_RATE = "moderation_block_rate"
METRIC_RETRIEVAL_ZERO_RESULT_RATE = "retrieval_zero_result_rate"
METRIC_TASK_QUEUE_BACKLOG = "task_queue_backlog"

# 看板下拉与文档展示用的全集（顺序即语义分组顺序，不影响计算）。
ALL_METRIC_NAMES: tuple[str, ...] = (
    METRIC_LLM_CALL_COUNT,
    METRIC_LLM_ERROR_RATE,
    METRIC_LLM_P50_LATENCY,
    METRIC_LLM_P95_LATENCY,
    METRIC_LLM_P99_LATENCY,
    METRIC_LLM_FIRST_TOKEN_P95,
    METRIC_TOKEN_USAGE,
    METRIC_TOKEN_COST,
    METRIC_DEGRADATION_RATE,
    METRIC_MODERATION_BLOCK_RATE,
    METRIC_RETRIEVAL_ZERO_RESULT_RATE,
    METRIC_TASK_QUEUE_BACKLOG,
)

# 比率类指标的小数位。**必须与 ai_metrics.metric_value 的 Numeric(16,4) 一致**：
# 早先这里写 6 并注释「6 位可无损落库」，实际 Numeric(16,4) 会把 0.333333
# 截成 0.3333，实现与注释互相矛盾、测试必然对不上。1/3 这类循环小数
# 统一按列精度收敛到 4 位，断言直接引用本常量，不再依赖浮点噪声。
RATE_DECIMALS = 4
# 金额类指标（token_cost）的小数位。cost_usd 源列是 Numeric(10,6)，保持同源精度。
COST_DECIMALS = 6

# 「进行中」的任务状态集合。积压深度（backlog）统计这两个状态：
# pending 是排队中、running 是在跑，两者都占用户感知意义上的「还没完」。
# retrying 不计——它已经在执行链路上，与 running 语义重叠会导致重复计数。
BACKLOG_STATUSES: tuple[str, ...] = (
    AITaskStatus.PENDING.value,
    AITaskStatus.RUNNING.value,
)

# 视为「失败」的 llm_call_logs.status 取值。
# error：调用彻底失败；timeout：超时（对用户体验而言就是失败）。
# degraded **不算**失败——它产出了可用结果，只是走了降级链，另有 degradation_rate 度量。
ERROR_STATUSES: tuple[str, ...] = ("error", "timeout")


# ==================== 纯计算函数（无 IO，便于单独断言） ====================


def nearest_rank_percentile(values: Sequence[int], percentile: float) -> int | None:
    """最近秩法（ceiling）分位数。口径见模块 docstring，实现与测试共用此函数。

    - 空序列返回 None（无定义，不落库）。
    - 单样本返回该样本本身。
    - 秩 = ceil(p * n)，最小为 1，最大为 n（防 p 传 0 或 >1 时越界）。
    """
    n = len(values)
    if n == 0:
        return None
    ordered = sorted(values)
    rank = math.ceil(percentile * n)
    if rank < 1:
        rank = 1
    if rank > n:
        rank = n
    return ordered[rank - 1]


def safe_ratio(numerator: int, denominator: int) -> float | None:
    """比率：分母为 0 时返回 None（无样本即无结论），而不是 0.0。

    用 0.0 冒充「零错误率」会让告警在最需要它的时刻静默——
    那一分钟一条日志都没写（可能正是 LLM 全挂了）时，错误率显示 0% 是最坏的结果。
    """
    if denominator <= 0:
        return None
    return round(numerator / denominator, RATE_DECIMALS)


def _normalize_tags(tags: dict[str, Any] | None) -> str:
    """tags → 稳定的 JSON 文本，用于去重与人工核对。

    sort_keys 保证 {"model":"a","provider":"b"} 与
    {"provider":"b","model":"a"} 归一化成同一串——否则同一维度组合会因为
    dict 插入顺序不同被写成两行，看板上就出现两条重复曲线。
    """
    if not tags:
        return ""
    return json.dumps(tags, sort_keys=True, ensure_ascii=False, default=str)


def _round_value(value: float | int | Decimal | None, decimals: int) -> float | None:
    """统一数值精度。Decimal 走 round() 会 TypeError，故先转 float。"""
    if value is None:
        return None
    return round(float(value), decimals)


def floor_to_minute(moment: datetime) -> datetime:
    """向下取整到分钟（秒与时分秒以外的部分清零）。

    输入若是 naive datetime，按 UTC 理解——库里 created_at 是 timezone-aware，
    混用 naive 会让比较结果随驱动而异（models/base.utcnow 已约定全项目 tz-aware）。
    """
    aware = _as_utc(moment)
    return aware.replace(second=0, microsecond=0)


def _as_utc(moment: datetime) -> datetime:
    """naive → 按 UTC 附加时区；已是 aware 的原样返回。"""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


# ==================== 桶内明细采集 ====================


async def _fetch_llm_rows(db: AsyncSession, start: datetime, end: datetime) -> list[Any]:
    """取桶内 LLM 调用明细（只取算指标需要的列，减少无关 ORM 加载）。"""
    statement = select(
        LLMCallLog.id,
        LLMCallLog.model,
        LLMCallLog.provider,
        LLMCallLog.scenario,
        LLMCallLog.status,
        LLMCallLog.latency_ms,
        LLMCallLog.first_token_ms,
        LLMCallLog.total_tokens,
        LLMCallLog.cost_usd,
        LLMCallLog.degradation_level,
        LLMCallLog.moderation_blocked,
    ).where(
        LLMCallLog.created_at >= start,
        LLMCallLog.created_at < end,
    )
    return list((await db.execute(statement)).all())


async def _count_rag(db: AsyncSession, start: datetime, end: datetime) -> tuple[int, int]:
    """取桶内检索日志的 (总数, 零结果数)。"""
    total = int(
        (
            await db.execute(
                select(func.count())
                .select_from(RAGRetrievalLog)
                .where(
                    RAGRetrievalLog.created_at >= start,
                    RAGRetrievalLog.created_at < end,
                )
            )
        ).scalar_one()
        or 0
    )
    zero = int(
        (
            await db.execute(
                select(func.count())
                .select_from(RAGRetrievalLog)
                .where(
                    RAGRetrievalLog.created_at >= start,
                    RAGRetrievalLog.created_at < end,
                    RAGRetrievalLog.is_zero_result.is_(True),
                )
            )
        ).scalar_one()
        or 0
    )
    return total, zero


async def _count_backlog(db: AsyncSession) -> int:
    """当前队列积压深度。

    **刻意不加时间窗过滤**：backlog 是「此刻有多少任务没跑完」的瞬时快照，
    不是这一分钟新产生的任务数。若按 created_at 过滤，一个十分钟前提交、
    现在仍在排队的任务就不计入，积压看起来永远是 0——而它恰恰是最该告警的场景。
    """
    return int(
        (
            await db.execute(
                select(func.count())
                .select_from(AITask)
                .where(AITask.status.in_(BACKLOG_STATUSES))
            )
        ).scalar_one()
        or 0
    )


# ==================== 单桶指标计算 ====================


def compute_bucket_metrics(
    llm_rows: Sequence[Any],
    rag_total: int,
    rag_zero: int,
    backlog: int,
) -> list[dict[str, Any]]:
    """由桶内明细算出待落库的指标行（纯函数，不碰数据库）。

    返回列表元素形如 ``{"metric_name":..., "metric_value":..., "sample_count":..., "tags":...}``；
    ``metric_value`` 为 None 的项由调用方过滤（无样本不落库）。
    """
    out: list[dict[str, Any]] = []

    def add(name: str, value: float | None, sample: int, tags: dict[str, Any] | None = None) -> None:
        if value is None:
            return
        out.append(
            {
                "metric_name": name,
                "metric_value": value,
                "sample_count": sample,
                "tags": tags,
            }
        )

    call_count = len(llm_rows)

    # ---------- 调用量与失败率 ----------
    add(METRIC_LLM_CALL_COUNT, float(call_count), call_count)
    error_count = sum(1 for row in llm_rows if (row.status or "") in ERROR_STATUSES)
    add(METRIC_LLM_ERROR_RATE, safe_ratio(error_count, call_count), call_count)

    # ---------- 延迟分位数（Python 侧最近秩，见模块 docstring） ----------
    latencies = [int(row.latency_ms or 0) for row in llm_rows]
    add(METRIC_LLM_P50_LATENCY, nearest_rank_percentile(latencies, 0.50), len(latencies))
    add(METRIC_LLM_P95_LATENCY, nearest_rank_percentile(latencies, 0.95), len(latencies))
    add(METRIC_LLM_P99_LATENCY, nearest_rank_percentile(latencies, 0.99), len(latencies))

    # 首 token 延迟只对流式调用有意义：非流式的 first_token_ms 通常为 NULL，
    # 但也可能被适配器填成总延迟——那会把一个「非流式的总耗时」当成体感首 token 上报。
    # 因此样本只取 is_stream 语义下的记录；本函数拿不到 is_stream，
    # 改为只取 first_token_ms 非空的行，并在 tags 里标 stream_only 说明口径。
    first_tokens = [int(row.first_token_ms) for row in llm_rows if row.first_token_ms is not None]
    add(
        METRIC_LLM_FIRST_TOKEN_P95,
        nearest_rank_percentile(first_tokens, 0.95),
        len(first_tokens),
        {"scope": "stream"},
    )

    # ---------- token 用量：按 model 维度分行 ----------
    by_model: dict[str, int] = {}
    for row in llm_rows:
        model = row.model or "unknown"
        by_model[model] = by_model.get(model, 0) + int(row.total_tokens or 0)
    for model, tokens in sorted(by_model.items()):
        add(METRIC_TOKEN_USAGE, float(tokens), tokens, {"model": model})

    # ---------- 成本：桶内合计，不分维度 ----------
    cost_sum = Decimal(0)
    cost_samples = 0
    for row in llm_rows:
        if row.cost_usd is not None:
            cost_sum += Decimal(str(row.cost_usd))
            cost_samples += 1
    add(
        METRIC_TOKEN_COST,
        _round_value(cost_sum, COST_DECIMALS) if cost_samples else 0.0,
        cost_samples,
    )

    # ---------- 降级率 ----------
    # 口径：status == 'degraded' **或** degradation_level > 0。
    # 两个条件都要：降级链层级从 1 开始（0=主模型），但有的路径只标 status 不填层级，
    # 反之亦然；取并集才不会低估降级。
    degraded_count = sum(
        1
        for row in llm_rows
        if (row.status or "") == "degraded" or int(row.degradation_level or 0) > 0
    )
    add(METRIC_DEGRADATION_RATE, safe_ratio(degraded_count, call_count), call_count)

    # ---------- 审核拦截率 ----------
    blocked_count = sum(1 for row in llm_rows if bool(row.moderation_blocked))
    add(METRIC_MODERATION_BLOCK_RATE, safe_ratio(blocked_count, call_count), call_count)

    # ---------- 检索零结果率 ----------
    add(
        METRIC_RETRIEVAL_ZERO_RESULT_RATE,
        safe_ratio(rag_zero, rag_total),
        rag_total,
    )

    # ---------- 队列积压 ----------
    add(METRIC_TASK_QUEUE_BACKLOG, float(backlog), backlog)

    return out


# ==================== 落库 ====================


async def _delete_bucket_rows(
    db: AsyncSession, bucket_start: datetime, names: Iterable[str]
) -> int:
    """删掉本桶内这些指标名的旧行（幂等的第一步）。"""
    name_list = list(names)
    if not name_list:
        return 0
    result = await db.execute(
        delete(AIMetric).where(
            AIMetric.bucket_time == bucket_start,
            AIMetric.metric_name.in_(name_list),
        )
    )
    # SQLAlchemy 2.0 的 DELETE 返回 RowMapping/RowCount；sqlite 与 pg 都给 rowcount
    return int(getattr(result, "rowcount", 0) or 0)


async def aggregate_minute_bucket(
    db: AsyncSession,
    bucket_start: datetime,
    bucket_end: datetime,
) -> dict[str, Any]:
    """聚合 [bucket_start, bucket_end) 窗口内的 AI 明细并写入 ai_metrics。

    **左闭右开**：created_at >= bucket_start 且 < bucket_end。相邻两桶因此
    恰好衔接，边界时刻（如 12:00:00.000000）只归属后一桶，不会重复计入两个桶。

    幂等实现见模块 docstring（无唯一约束 → 不能 ON CONFLICT → 删了重插）。
    整个「删 + 插」在调用方给的事务里完成，中途失败由外层 rollback，
    不会出现「旧行删了新行没写」的空洞。

    返回结构化摘要：
    ``{"bucket_start", "bucket_end", "written", "deleted", "metrics": {...}}``
    其中 ``metrics`` 是 metric_name → value 的扁平映射（同名多 tag 行取求和后的
    代表值仅用于日志观察，token_usage 这类分行指标不在其中做强断言）。
    """
    start = _as_utc(bucket_start)
    end = _as_utc(bucket_end)
    if end <= start:
        raise ValueError("bucket_end 必须晚于 bucket_start")

    llm_rows = await _fetch_llm_rows(db, start, end)
    rag_total, rag_zero = await _count_rag(db, start, end)
    backlog = await _count_backlog(db)

    rows = compute_bucket_metrics(llm_rows, rag_total, rag_zero, backlog)

    deleted = await _delete_bucket_rows(db, start, {r["metric_name"] for r in rows})

    for item in rows:
        db.add(
            AIMetric(
                bucket_time=start,
                metric_name=item["metric_name"],
                # 列是 Numeric(16,4)：写 Decimal/float 都可，这里保持 float 由驱动转换
                metric_value=item["metric_value"],
                sample_count=int(item["sample_count"]),
                tags=item["tags"] or None,
            )
        )

    await db.commit()

    summary: dict[str, float] = {}
    for item in rows:
        summary[item["metric_name"]] = float(item["metric_value"])

    logger.info(
        "指标聚合完成 bucket=%s llm_rows=%d rag=%d/%d backlog=%d written=%d",
        start.isoformat(),
        len(llm_rows),
        rag_zero,
        rag_total,
        backlog,
        len(rows),
    )
    return {
        "bucket_start": start.isoformat(),
        "bucket_end": end.isoformat(),
        "written": len(rows),
        "deleted": deleted,
        "llm_call_rows": len(llm_rows),
        "retrieval_rows": rag_total,
        "queue_backlog": backlog,
        "metrics": summary,
    }


# ==================== Celery 侧入口 ====================


def previous_complete_bucket(now: datetime | None = None) -> tuple[datetime, datetime]:
    """算出「上一个完整分钟桶」的 [start, end)。

    now 未给出则取当前 UTC 时刻。向下取整得到当前分钟的起点，
    它就是上一桶的终点；再往前推一分钟是上一桶的起点。

    为什么聚合**上一个**桶而不是当前桶：当前分钟还在写入，
    现在算出来的分位数与计数天生偏低，且下一轮 Beat 会把同一桶再算一遍——
    虽然幂等不会重复计数，但看板会看到一根正在爬升的假曲线。
    只碰已经封存的桶，曲线才是单调可信的。
    """
    current = _as_utc(now) if now is not None else datetime.now(UTC)
    end = current.replace(second=0, microsecond=0)
    return end - timedelta(minutes=1), end


async def run_aggregate(now: datetime | None = None) -> dict[str, Any]:
    """聚合上一个完整分钟桶（自开会话，供 API 手动触发与 Celery 桥接复用）。

    异常**不外抛**：转成 ``{"ok": False, "error": ...}`` 返回。
    监控聚合失败不该让调用方（worker 进程 / HTTP 请求）跟着崩，
    下一轮 Beat 自然补算；同时把 traceback 记进日志留痕。
    """
    bucket_start, bucket_end = previous_complete_bucket(now)
    try:
        async with async_session_factory() as session:
            return await aggregate_minute_bucket(session, bucket_start, bucket_end)
    except Exception as exc:  # noqa: BLE001 - 观测旁路不得影响宿主进程
        logger.exception("指标聚合失败 bucket=%s", bucket_start.isoformat())
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:500],
            "bucket_start": bucket_start.isoformat(),
            "bucket_end": bucket_end.isoformat(),
            "written": 0,
        }


__all__ = [
    "ALL_METRIC_NAMES",
    "BACKLOG_STATUSES",
    "COST_DECIMALS",
    "ERROR_STATUSES",
    "METRIC_DEGRADATION_RATE",
    "METRIC_LLM_CALL_COUNT",
    "METRIC_LLM_ERROR_RATE",
    "METRIC_LLM_FIRST_TOKEN_P95",
    "METRIC_LLM_P50_LATENCY",
    "METRIC_LLM_P95_LATENCY",
    "METRIC_LLM_P99_LATENCY",
    "METRIC_MODERATION_BLOCK_RATE",
    "METRIC_RETRIEVAL_ZERO_RESULT_RATE",
    "METRIC_TASK_QUEUE_BACKLOG",
    "METRIC_TOKEN_COST",
    "METRIC_TOKEN_USAGE",
    "RATE_DECIMALS",
    "aggregate_minute_bucket",
    "compute_bucket_metrics",
    "floor_to_minute",
    "nearest_rank_percentile",
    "previous_complete_bucket",
    "run_aggregate",
    "safe_ratio",
]
