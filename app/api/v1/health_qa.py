"""健康咨询问答路由：SSE 流式端点（阶段 7 批次 0）。

``POST /api/v1/health-qa/stream``。用 POST 而不是 GET：
前端用 ``fetch`` + ``ReadableStream`` 消费（``EventSource`` 不能带
Authorization 头，而本端点必须鉴权），因此可以携带 JSON body。

四个非显然的实现约束，前三条都来自「流式响应一旦开始就无法回头」：

1. **限流必须在返回 StreamingResponse 之前完成**。状态码与响应头在第一次
   yield 时就已发出，之后再想拒绝只能截断流，客户端看到的是「连接断了」
   而不是「409 你已有一次咨询在跑」。所以准入判定在路由函数体里做。
2. **异常不能抛给 Starlette**。同理，流开始后抛异常无法转成 5xx，
   只会让客户端拿到半截响应且不知原因。所有错误都翻译成 ``error`` 事件。
3. **不复用请求级 db 会话**。FastAPI 对「依赖 yield 的资源何时关闭」的语义
   随版本变过（0.106 起依赖退出早于响应发送），而流式响应体是在端点返回
   **之后**才被 Starlette 逐步消费的。把请求会话带进生成器，等于赌版本行为；
   这里改为在生成器内自开一个会话，生命周期完全自控。
4. **落库不能在生成器里做**。见 ``health_qa_service`` 说明；这里用
   ``BackgroundTask`` + 独立短事务，客户端断开也仍会执行。

心跳：每 ``HEARTBEAT_INTERVAL`` 秒发一个 SSE 注释行（``: heartbeat``）。
nginx 侧已设 ``proxy_read_timeout 300s``，但中间可能还有别的代理；
更重要的是前端需要「连接还活着」的信号来区分「模型在想」与「连接已死」。
注释行不会被解析成事件，对前端无副作用。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Annotated, Any, AsyncIterator

from fastapi import APIRouter, BackgroundTasks, Depends, Response
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, require_permission
from app.core.cache import cache_client
from app.core.database import async_session_factory
from app.core.exceptions import DuplicateTaskError, LLMError
from app.models.auth import User
from app.models.health_qa import HealthQALog
from app.schemas.health_qa import HealthQARequest
from app.services.health_qa_service import (
    EVENT_ERROR,
    QaEvent,
    QaOutcome,
    stream_health_qa,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health-qa", tags=["健康咨询"])

# 与既有模块同一写法：require_permission 返回的是依赖函数，必须用
# Annotated + Depends 包一层才能作为参数注解，直接赋值给别名是错的。
CanAsk = Annotated[User, Depends(require_permission("health_qa", "ask"))]

# 单用户并发上限 1：同一账号同时只允许一条咨询流。
# 健康咨询是重活（检索 + 流式生成，全程占一个 uvicorn worker），
# 不限流的话一个人开十几个标签页就能把 worker 池吃光。
CONCURRENCY_KEY_PREFIX = "health_qa:active:"
CONCURRENCY_TTL_SECONDS = 120

# 单进程并发上限。超过就拒绝而不是排队：排队会让用户对着空白屏幕等，
# 体验比直接告知「稍后再试」更差，且队列本身也占内存与 worker。
#
# **注意这是「每个 uvicorn worker 进程」的上限，不是全系统上限**：
# Semaphore 是进程内对象，实际总并发 = 本值 × IMCS_API_WORKERS。
# 当前 compose 默认 4 个 worker，即系统上限 32 条并发流。
# 若将来要按真实总量限流，得换成 Redis 计数器（跨进程可见）；
# 现在不这么做是因为它带来的复杂度（计数泄漏、崩溃后的键残留）
# 高于收益——真正的硬瓶颈是 worker 数，调 worker 数就能调容量。
GLOBAL_CONCURRENCY_LIMIT = 8
_global_semaphore = asyncio.Semaphore(GLOBAL_CONCURRENCY_LIMIT)
GLOBAL_ACQUIRE_TIMEOUT = 5.0

# 单条流的最长存活时间。LLM 生成通常几秒到几十秒，90s 已足够宽裕；
# 设上限是为了防止某个 chunk 卡住导致 worker 被永久占用。
MAX_STREAM_SECONDS = 90

# 心跳间隔。要明显小于任何中间代理的 idle timeout（nginx 是 300s）。
HEARTBEAT_INTERVAL = 15.0

# SSE 注释行，不是事件，前端解析器会忽略
HEARTBEAT_FRAME = ": heartbeat\n\n"


def _sse_frame(event: QaEvent) -> str:
    """把一个领域事件翻译成 SSE 帧。

    格式与旧系统 ``ask.py`` 的 ``format_sse_event`` 一致
    （``event: X\\ndata: {json}\\n\\n``），沿用同名协议是为了让前端与运维
    不必为新旧系统维护两套解析逻辑。``ensure_ascii=False`` 让中文按原文
    传输而不是 ``\\uXXXX`` 转义——SSE 是 UTF-8 文本协议，转义只会白白增大体积。
    """
    payload = json.dumps(event.data, ensure_ascii=False, default=str)
    return f"event: {event.kind}\ndata: {payload}\n\n"


async def _acquire_user_slot(user_id: Any) -> bool:
    """尝试占用单用户并发位。

    Returns:
        True  = 占到了，调用方负责释放；
        False = **Redis 不可用**，降级为不做单用户限流（调用方不该释放）。

    Raises:
        DuplicateTaskError: 键已存在，说明该用户已有一次咨询在跑。

    把「Redis 挂了」与「真的并发冲突」区分开是关键：前者应放行
    （旁路组件不可用不该让核心功能整体失效），后者必须拒。
    若都返回 False，Redis 抖动时用户会被莫名 409。
    """
    key = f"{CONCURRENCY_KEY_PREFIX}{user_id}"
    try:
        # SET NX + TTL 原子占位。TTL 是兜底：进程被强杀没走到释放时
        # 键会自动过期，不至于把用户永久锁死。
        ok = await cache_client.set(key, "1", nx=True, ex=CONCURRENCY_TTL_SECONDS)
    except Exception:  # noqa: BLE001
        logger.warning("健康咨询限流键写入失败，本次跳过单用户限流", exc_info=True)
        return False
    if ok:
        return True
    raise DuplicateTaskError(
        "你已有一次咨询正在进行，请等待它结束", task_type="health_qa"
    )


async def _release_user_slot(user_id: Any, held: bool) -> None:
    """释放单用户并发位。``held=False`` 时是空操作（避免误删别人的键）。"""
    if not held:
        return
    with contextlib.suppress(Exception):
        await cache_client.delete(f"{CONCURRENCY_KEY_PREFIX}{user_id}")


async def _persist_log(outcome: QaOutcome, user_id: Any) -> None:
    """把一次问答写入 ``health_qa_logs``。独立短事务，best-effort。

    在 BackgroundTask 里执行，因此**不能复用请求会话**：那时请求已结束，
    会话可能已归还连接池。自开一个 session 写完即关。

    失败只记日志不外抛：这是旁路审计，且此时响应其实已经发完，
    抛出去也无法让用户看到。
    """
    try:
        async with async_session_factory() as session:
            session.add(
                HealthQALog(
                    trace_id=outcome.trace_id or None,
                    user_id=user_id,
                    question=outcome.question,
                    answer_text=outcome.answer_text or None,
                    ref_sources=outcome.ref_sources or None,
                    prompt_tokens=outcome.prompt_tokens,
                    completion_tokens=outcome.completion_tokens,
                    first_token_ms=outcome.first_token_ms,
                    latency_ms=outcome.latency_ms,
                    model=outcome.model,
                    degradation_level=outcome.degradation_level,
                    urgent_keywords=outcome.urgent_keywords or None,
                    is_urgent_short_circuit=outcome.is_urgent_short_circuit,
                    is_zero_result=outcome.is_zero_result,
                    client_disconnected=outcome.client_disconnected,
                    error_message=outcome.error_message,
                )
            )
            await session.commit()
    except Exception:  # noqa: BLE001 - 审计日志失败不反噬业务
        logger.warning(
            "健康咨询日志落库失败 trace=%s", outcome.trace_id, exc_info=True
        )


async def _iter_with_heartbeat(events: AsyncIterator[QaEvent]) -> AsyncIterator[str]:
    """把事件流包上心跳与总时长上限。

    不能简单 ``async for``：那样在等下一个事件期间无法插入心跳。
    这里把 ``__anext__`` 包成 task，用 ``asyncio.wait(timeout=...)`` 轮询——
    超时就发一帧心跳再继续等，拿到结果就转发。

    task 必须跨轮复用而不是重建：若心跳超时后丢掉旧 task 重新 ``__anext__``，
    会漏掉那个正处于「即将返回」状态的事件，表现为回答中间莫名少一段。
    """
    iterator = events.__aiter__()
    pending: asyncio.Task | None = None
    started = time.perf_counter()
    try:
        while True:
            if time.perf_counter() - started > MAX_STREAM_SECONDS:
                yield _sse_frame(
                    QaEvent(
                        EVENT_ERROR,
                        {
                            "code": "stream_timeout",
                            "message": "回答生成超时，请重新提问。",
                        },
                    )
                )
                return
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=HEARTBEAT_INTERVAL)
            if not done:
                yield HEARTBEAT_FRAME
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                return
            finally:
                pending = None
            yield _sse_frame(event)
    finally:
        # 客户端断开（CancelledError/GeneratorExit）时收尾：取消未完成的
        # __anext__ 并关闭生成器，避免底层 LLM 连接与 DB 会话泄漏。
        if pending is not None and not pending.done():
            pending.cancel()
            with contextlib.suppress(Exception):
                await pending
        with contextlib.suppress(Exception):
            await iterator.aclose()


@router.post(
    "/stream",
    summary="流式健康咨询问答（SSE）",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "SSE 事件流"},
        403: {"description": "缺少 health_qa:ask 权限"},
        409: {"description": "该账号已有一次咨询正在进行"},
        503: {"description": "并发已满，AI 服务暂时不可用"},
    },
)
async def stream_health_qa_endpoint(
    payload: HealthQARequest,
    background: BackgroundTasks,
    current_user: CanAsk,
) -> Response:
    """发起一次流式健康咨询。

    事件顺序：``metadata``（引用来源，先于正文以便前端先渲染出处）
    → ``token``*N（增量文本）→ ``done``（用量与免责声明）。
    出错时以 ``error`` 事件结束，HTTP 状态码仍是 200（流已开始，改不了）。

    急症与心理危机问句会被安全守卫短路：不调 LLM，直接返回固定就医指引，
    ``metadata.guard`` 标明类型（emergency / crisis）供前端差异化呈现。
    """
    user_id = current_user.id
    question = payload.question.strip()
    history = [turn.model_dump() for turn in payload.history]

    # ---------- 准入：顺序是「先占用户位，再抢全局位」----------
    # 反过来写会有一条容易漏的释放路径：全局位抢到后用户位冲突，
    # 就必须记得把全局位还回去；一旦某分支漏还，信号量会被永久吃掉一格，
    # 表现为「并发上限越用越小」这种极难排查的故障。
    # 先占用户位（冲突时直接抛 409，此刻尚未持有任何资源）就没这个问题。
    user_slot_held = await _acquire_user_slot(user_id)

    try:
        await asyncio.wait_for(
            _global_semaphore.acquire(), timeout=GLOBAL_ACQUIRE_TIMEOUT
        )
    except asyncio.TimeoutError:
        await _release_user_slot(user_id, user_slot_held)
        # 503 而非 429：这是服务端容量问题，不是用户超额
        raise LLMError("当前咨询人数较多，请稍后再试") from None

    outcome = QaOutcome(trace_id="", question=question)

    async def event_source() -> AsyncIterator[str]:
        """响应体生成器。异常、超时、断开都在这里收口。"""
        disconnected = False
        # 自开会话，不用请求级 db：见模块 docstring 第 3 条。
        async with async_session_factory() as session:
            try:
                events = stream_health_qa(
                    session,
                    question,
                    user_id=user_id,
                    history=history,
                    outcome=outcome,
                )
                async for frame in _iter_with_heartbeat(events):
                    yield frame
            except asyncio.CancelledError:
                # 客户端断开。Starlette 通过取消任务中止流，这是正常路径
                # 不是错误，但要如实记进审计日志（区分「用户走了」与「生成失败」）。
                disconnected = True
                raise
            except Exception as exc:  # noqa: BLE001 - 流已开始，只能发 error 事件
                logger.exception("健康咨询流式响应异常 trace=%s", outcome.trace_id)
                outcome.error_message = f"{type(exc).__name__}: {exc}"[:500]
                yield _sse_frame(
                    QaEvent(
                        EVENT_ERROR,
                        {
                            "code": "internal_error",
                            "message": "回答生成失败，请稍后再试。",
                        },
                    )
                )
            finally:
                outcome.client_disconnected = disconnected
                _global_semaphore.release()

    # 释放与落库都挂 background：即使客户端在流开始前就断开、
    # 生成器一次都没被迭代（finally 不会执行），这两件事仍会发生。
    # _release_user_slot 幂等，与生成器内的释放重复调用无副作用。
    background.add_task(_release_user_slot, user_id, user_slot_held)
    background.add_task(_persist_log, outcome, user_id)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            # 禁用中间代理缓冲。nginx 侧已配 proxy_buffering off，
            # 这里再声明一次是为了让端点在别的反代下也能正确流式。
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
        background=background,
    )


__all__ = ["router"]
