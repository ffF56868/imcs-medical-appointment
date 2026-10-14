"""可观测性写入层：RAG 检索日志 + 工作流追踪。

============================ 为什么单独成模块 ============================

阶段 6 之前，检索日志写在各调用方的 session 上、工作流追踪无人落库。散落的
写法有两个实际问题：

1. **异常处理不一致**。每个埋点都要自己记得包 try/except，漏一处就等于
   「因为记日志失败，用户的分诊请求 500」——这是可观测性反噬业务，
   在医疗场景里属于不可接受的设计缺陷。集中到本模块后，「写失败只 warning」
   成为一条实现约束而不是各处自觉。
2. **事务边界错**。把日志 add 进请求传入的 session，等于让日志和主业务共用
   一个事务：主业务回滚时日志也消失（恰恰是最需要日志的那次失败没了），
   而主业务提交时又会被日志的额外往返拖慢。

============================ 为什么一律用独立短事务 ============================

所有写函数都**忽略调用方传入的 session，自己开一条新会话并提交**：

- 检索发生在 SSE 流式响应的生成器里。客户端中途断开时 FastAPI 会取消生成器，
  请求级 session 可能已进入回收；此时复用它写日志会撞上
  ``Session is closed`` / ``InvalidRequestError``，日志永远补不上。
- Celery worker 侧同理：解读任务体持有的是同步 Session，异步协程要落日志
  只能另开异步会话。
- 日志与业务无原子性要求（「日志写了但业务回滚」对排障无害，反过来才有害），
  因此独立提交是纯收益。

代价是一次额外的连接池占用。本项目日均调用量千级（见 models/ai.py 注释），
且日志写都是单行 INSERT/UPDATE，不构成池压力。

============================ trace_id 兜底 ============================

trace_id 缺失时尝试从 ``app.core.trace.get_trace_id()``（contextvar）取当前
链路 id。该模块由并行代理交付，**本文件只 import、不创建**，并且用
``try/except ImportError`` 兜住——它尚未落地时导入期不能炸掉整个服务。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update

from app.core.database import async_session_factory
from app.models.ai import RAGRetrievalLog, WorkflowTrace

logger = logging.getLogger(__name__)

# candidates / steps / final_output 都是 JSONB 明细列，刻意不设数据库长度上限，
# 但也不能无限膨胀：一次检索 LIMIT 20，投影后 top-k 控制在下面这个条数即可
# 覆盖回放所需；超出部分对排障没有增量信息。
MAX_CANDIDATES = 10
# 单条 step 的输入/输出快照截断长度。快照用于还原「当时喂给模型什么」，
# 摘要性质，不需要全文。
SNAPSHOT_CHARS = 500


def _fallback_trace_id(trace_id: str | None) -> str | None:
    """trace_id 为空时回落到 contextvar。

    ``app.core.trace`` 由并行代理新建，本模块只消费不创建；未就绪时按
    「无 trace_id」继续写，绝不因取不到上下文就放弃这条日志。
    """
    if trace_id:
        return trace_id
    try:  # pragma: no cover - 取决于并行代理是否已交付该模块
        from app.core.trace import get_trace_id

        return get_trace_id()
    except Exception:  # noqa: BLE001 - 模块不存在或取值异常都按无 trace 处理
        return None


def _as_uuid(value: Any) -> uuid.UUID | None:
    """把 user_id / ref_id 归一成 uuid.UUID。

    上游传法不统一：HTTP 层是 UUID 对象，Celery kwargs 经 JSON 序列化后是
    字符串。写不进 PGUUID 列的字符串在这里转换，转不动就返回 None——
    日志缺一个关联字段远好过整条日志写失败。
    """
    if value is None or isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None


def _snapshot(value: Any, limit: int = SNAPSHOT_CHARS) -> str | None:
    """把任意对象压成截断字符串快照，供 steps 的 input/output 使用。"""
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    else:
        try:
            text = repr(value)
        except Exception:  # noqa: BLE001 - repr 失败退化为类型名
            text = f"<{type(value).__name__}>"
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def project_candidates(
    results: Sequence[Any],
    *,
    limit: int = MAX_CANDIDATES,
) -> list[dict[str, Any]]:
    """把检索结果投影成精简 candidates 列表。

    **只存元数据、绝不塞正文**：rag_retrieval_logs.candidates 的用途是回放
    「哪些 chunk 以第几名进了融合」，正文已经在 knowledge_chunks 里，复制进
    日志会让这张高频写入表迅速膨胀，也让日志备份变成携带患者可见的知识副本。

    每项字段：chunk_id / rank / rrf_score / title / doc_type / source /
    source_tier / routes（各路 rank+score 明细）。
    """
    projected: list[dict[str, Any]] = []
    for rank, item in enumerate(results[:limit], start=1):
        entry: dict[str, Any] = {
            "chunk_id": str(_get(item, "chunk_id", "")),
            "rank": rank,
            "rrf_score": _round(_get(item, "rrf_score", None) or 0.0),
            "title": _truncate(_get(item, "title", ""), 200),
            "doc_type": _get(item, "doc_type"),
            "source": _truncate(_get(item, "source", ""), 200),
            "source_tier": _get(item, "source_tier"),
        }
        routes = _get(item, "route_details")
        if isinstance(routes, dict) and routes:
            entry["routes"] = {
                src: {
                    "rank": info.get("rank"),
                    "score": _round(info.get("score")),
                }
                for src, info in routes.items()
                if isinstance(info, dict)
            }
        projected.append(entry)
    return projected


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """兼容 dataclass 属性与 dict 键两种取法。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _round(value: Any, digits: int = 6) -> float | None:
    """分数保留 6 位小数；非数值原样返回 None，避免序列化报错。"""
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _truncate(value: Any, limit: int) -> Any:
    if not isinstance(value, str):
        return value
    return value if len(value) <= limit else value[:limit] + "…"


# ==================== RAG 检索日志 ====================


async def log_retrieval(**fields: Any) -> None:
    """写一条 ``rag_retrieval_logs``，best-effort。

    Args:
        **fields: 模型字段名 → 值。常用键：trace_id / user_id / query_text /
            rewritten_query / vector_hit_count / keyword_hit_count /
            both_hit_count / merged_count / final_count / is_zero_result /
            top_score / vector_latency_ms / keyword_latency_ms /
            rrf_latency_ms / rerank_latency_ms / rerank_channel / rrf_k /
            candidates。**未知键一律忽略**，这样调用方多传参数不会炸。

    关于 ``session_or_none``：**本函数刻意不接受它**（调用方多传也只会落进
    「未知字段」被忽略）。检索发生在 SSE 流式响应中途，客户端一断开 FastAPI
    就取消生成器、请求 session 可能已进入回收；复用它会撞上 Session is closed，
    日志恰好在该写的时候写不进去。因此一律自开独立短事务、写完立即提交关闭。
    全 kwargs 签名是为了让埋点侧直接透传诊断计数，不必在本模块维护位置参数顺序。
    - **内部吞掉所有异常**，只打 warning。可观测性不能反噬业务：一次日志写
      失败绝不能让用户的检索/分诊请求报错。
    """
    # 调用方若把请求 session 一起传进来（历史上存在这种写法），这里显式剔除，
    # 既避免它落进「未知字段」被静默忽略，也保证不会有人误以为日志跟着它提交。
    fields.pop("session_or_none", None)
    fields.pop("session", None)
    try:
        unknown = set(fields) - _RETRIEVAL_COLUMNS
        if unknown:
            logger.debug("log_retrieval 忽略未知字段：%s", sorted(unknown))
        values = {k: v for k, v in fields.items() if k in _RETRIEVAL_COLUMNS}
        values["trace_id"] = _fallback_trace_id(values.get("trace_id"))
        values["user_id"] = _as_uuid(values.get("user_id"))
        if values.get("candidates") is not None:
            values["candidates"] = project_candidates(values["candidates"])

        async with async_session_factory() as s:
            s.add(RAGRetrievalLog(**values))
            await s.commit()
    except Exception:  # noqa: BLE001 - 日志失败绝不影响主流程
        logger.warning("写入 RAG 检索日志失败", exc_info=True)


_RETRIEVAL_COLUMNS: frozenset[str] = frozenset(
    {
        "trace_id",
        "user_id",
        "query_text",
        "rewritten_query",
        "vector_hit_count",
        "keyword_hit_count",
        "both_hit_count",
        "merged_count",
        "final_count",
        "is_zero_result",
        "top_score",
        "vector_latency_ms",
        "keyword_latency_ms",
        "rrf_latency_ms",
        "rerank_latency_ms",
        "rerank_channel",
        "rrf_k",
        "candidates",
    }
)


# ==================== 工作流追踪 ====================


async def start_workflow_trace(
    trace_id: str | None,
    workflow_name: str,
    *,
    user_id: Any = None,
    ref_id: Any = None,
    node_count: int = 0,
) -> int | None:
    """建一条 status=running 的 ``workflow_traces``，返回其主键 id。

    返回 None 表示没建成（写失败已被吞）。调用方拿到 None 时后续 finish 会
    退化为按 trace_id 更新——匹配不到行就是空操作，同样不影响业务。

    独立短事务的理由同 ``log_retrieval``：工作流入口所在的事务里紧接着还有
    业务写入（分诊落 TriageResult、解读认领 CAS），若追踪行跟它们同生共死，
    「业务回滚 → 追踪记录消失」正好抹掉了最需要留痕的失败样本。
    """
    try:
        row = WorkflowTrace(
            trace_id=_fallback_trace_id(trace_id) or uuid.uuid4().hex,
            workflow_name=workflow_name,
            status="running",
            node_count=int(node_count or 0),
            executed_node_count=0,
            user_id=_as_uuid(user_id),
            ref_id=_as_uuid(ref_id),
        )
        async with async_session_factory() as s:
            s.add(row)
            await s.commit()
            return int(row.id)
    except Exception:  # noqa: BLE001
        logger.warning("创建工作流追踪失败 workflow=%s", workflow_name, exc_info=True)
        return None


async def finish_workflow_trace(
    trace_id_or_id: Any,
    *,
    status: str,
    steps: Sequence[dict[str, Any]] | None = None,
    node_count: int | None = None,
    total_latency_ms: int = 0,
    total_tokens: int = 0,
    total_cost_usd: float | None = None,
    branch_taken: str | None = None,
    final_output: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    """收尾更新一条 ``workflow_traces``，best-effort。

    Args:
        trace_id_or_id: 优先按主键 id 定位（start 返回的那个）；传入字符串
            （trace_id）时按 trace_id 定位最近一条。双路查找是因为 Celery 侧
            常常只有 trace_id 而没有 id。
        status: running / completed / failed / skipped 等业务终态字符串。
        steps: 节点明细列表，结构见 :func:`build_step`。
        branch_taken: 分支标识。分诊的急诊分支必须与常规分支区分统计，
            否则「急症拦截率」和「AI 分诊准确率」会被混在一起。
        node_count: 计划节点数。入口 start 时按「最乐观路径」预估（分诊预估 5），
            走到中途短路（急症拦截只跑 1 个节点）时在这里如实修正——否则
            executed(1)/node_count(5) 会被看板误读成「AI 链路大面积短路」。
            传 None 表示沿用 start 时的预估值。

    同一 trace_id 被多次 finish 时只更新最新一条：重复投递的工作流会留下多条
    running 记录，收尾语义是「把这次跑的结论写到最新那条上」，不做批量回填。

    ``completed_at`` 由本函数写入——它是「工作流跑完」这一事实的时间戳，
    放在这里而不是模型默认值里，才能保证 running 行的该字段为 NULL。
    """
    try:
        # node_count 仅在调用方显式传入时修正；实际执行数永远以 steps 长度为准，
        # 且不得大于计划数——那说明埋点漏记了节点，宁可截齐也不留假异常。
        executed = len(steps or [])
        payload = {
            "status": status,
            "steps": list(steps or []),
            "executed_node_count": executed,
            "total_latency_ms": int(total_latency_ms or 0),
            "total_tokens": int(total_tokens or 0),
            "total_cost_usd": total_cost_usd,
            "branch_taken": branch_taken,
            "final_output": final_output,
            "error_message": _truncate(error_message, 2000),
            "completed_at": datetime.now(timezone.utc),
        }
        if node_count is not None:
            payload["node_count"] = max(int(node_count), executed)
        async with async_session_factory() as s:
            # **必须按 id 倒序取第一条**：同一 trace_id 可能有多条 running 行
            # （重复投递），docstring 承诺「收尾写到最新那条」。不加 order_by 时
            # SQLite/PG 都按插入序返回第一行，实测会把结论写进最早那条、
            # 最新那条永远停在 running——看板上就是一堆假僵尸追踪。
            stmt = (
                select(WorkflowTrace.id)
                .where(_locate_clause(trace_id_or_id))
                .order_by(WorkflowTrace.id.desc())
            )
            target_id = (await s.execute(stmt)).scalars().first()
            if target_id is None:
                logger.debug("工作流追踪记录不存在，跳过收尾：%s", trace_id_or_id)
                return
            await s.execute(
                update(WorkflowTrace).where(WorkflowTrace.id == target_id).values(**payload)
            )
            await s.commit()
    except Exception:  # noqa: BLE001
        logger.warning("收尾工作流追踪失败", exc_info=True)


def _locate_clause(trace_id_or_id: Any):
    """构造定位条件：整数走主键，字符串走 trace_id。"""
    if isinstance(trace_id_or_id, bool):
        return WorkflowTrace.trace_id == str(trace_id_or_id)
    if isinstance(trace_id_or_id, int):
        return WorkflowTrace.id == trace_id_or_id
    return WorkflowTrace.trace_id == str(trace_id_or_id)


def build_step(
    node: str,
    node_type: str,
    *,
    latency_ms: int = 0,
    tokens: int = 0,
    cost_usd: float | None = None,
    input_snapshot: Any = None,
    output_snapshot: Any = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造一条 step 明细（供工作流埋点复用，保证各工作流写入结构一致）。

    字段口径对齐 models/ai.py 里 steps 列的注释：
    ``[{node, type, latency_ms, tokens, input, output, error}]``。
    """
    step: dict[str, Any] = {
        "node": node,
        "type": node_type,
        "latency_ms": int(latency_ms or 0),
        "tokens": int(tokens or 0),
        "input": _snapshot(input_snapshot),
        "output": _snapshot(output_snapshot),
        "error": _truncate(error, 500) if error else None,
    }
    if cost_usd is not None:
        step["cost_usd"] = _round(cost_usd, 6)
    if extra:
        step.update(extra)
    return step


def sum_usage(responses: Sequence[Any]) -> tuple[int, float]:
    """从一组 LLMResponse 累加 (total_tokens, total_cost_usd)。

    用 getattr 逐项安全取值：桩对象经常不带 usage/cost_usd，真实响应才带全。
    """
    tokens = 0
    cost = 0.0
    for resp in responses:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            try:
                tokens += int(getattr(usage, "total_tokens", 0) or 0)
            except (TypeError, ValueError):
                pass
        raw_cost = getattr(resp, "cost_usd", None)
        if raw_cost is not None:
            try:
                cost += float(raw_cost)
            except (TypeError, ValueError):
                pass
    return tokens, round(cost, 6)


# ==================== Celery worker 同步封装 ============================


def _run_coro(coro: Any) -> Any:
    """在同步上下文里驱动一个协程。

    **为什么不能裸调 asyncio.run**：worker 任务体本身常已经处在
    ``celery_app.run_async`` 新建的事件循环里（sync 工作流内部还要桥接异步
    LLM 客户端）。此时再 asyncio.run 会抛
    ``RuntimeError: asyncio.run() cannot be called from a running event loop``，
    而该异常会被 best-effort 语义吞成 warning —— 结果是「日志静默丢失」这种
    最难排查的形态。因此先探测活循环：
    - 无活循环（标准 prefork 任务体）→ ``asyncio.run``，符合项目约定；
    - 有活循环 → 把协程作为后台任务调度到当前循环（fire-and-forget），
      这样 HTTP 侧误用同步封装时写入仍然发生，失败仍由异步版本自己吞掉；
      仅当连调度都失败才告警跳过。
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            return asyncio.run(coro)
        except Exception:  # noqa: BLE001
            logger.warning("可观测性协程执行失败", exc_info=True)
            return None
    try:
        loop.create_task(coro)
        return None
    except Exception:  # noqa: BLE001 - 循环正在关闭等极端情况
        close = getattr(coro, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
        logger.warning(
            "可观测性同步封装无法调度协程，本次写入已跳过（请改调异步版本）",
            exc_info=True,
        )
        return None


def log_retrieval_sync(**fields: Any) -> None:
    """``log_retrieval`` 的同步封装，供 Celery worker 使用。"""
    _run_coro(log_retrieval(**fields))


def finish_workflow_trace_sync(
    trace_id_or_id: Any,
    *,
    status: str,
    steps: Sequence[dict[str, Any]] | None = None,
    total_latency_ms: int = 0,
    total_tokens: int = 0,
    total_cost_usd: float | None = None,
    branch_taken: str | None = None,
    final_output: dict[str, Any] | None = None,
    error_message: str | None = None,
    node_count: int | None = None,
) -> None:
    """``finish_workflow_trace`` 的同步封装，供 Celery worker 使用。"""
    _run_coro(
        finish_workflow_trace(
            trace_id_or_id,
            status=status,
            steps=steps,
            total_latency_ms=total_latency_ms,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            branch_taken=branch_taken,
            final_output=final_output,
            error_message=error_message,
            node_count=node_count,
        )
    )


__all__ = [
    "MAX_CANDIDATES",
    "SNAPSHOT_CHARS",
    "build_step",
    "finish_workflow_trace",
    "finish_workflow_trace_sync",
    "log_retrieval",
    "log_retrieval_sync",
    "project_candidates",
    "start_workflow_trace",
    "sum_usage",
]
