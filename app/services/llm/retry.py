"""重试与超时策略。

两个核心决策：

**1. 哪些错误重试。** 只对「重试有机会成功」的错误重试：
429（限流）、5xx（服务端故障）、连接错误、超时。
400/401/403/404 是请求本身有问题，重试只会浪费时间与配额——
尤其 401 重试会加速触发供应商的暴力破解防护。

**2. 流式超时拆成两段。** 单一超时无法区分两种完全不同的故障：
- 首 token 迟迟不来：通常是排队或服务过载，应快速失败并降级
- chunk 间隔变长：可能只是模型在生成长文本，不该误杀

因此首 token 超时（10s）明显短于 chunk 间隔超时（15s），
且后者是「相邻两片之间」的滑动上限，不是总时长上限。

SDK 自带重试在 provider 层已关闭（``max_retries=0``），
避免「SDK 重试 × 我们的重试」叠加放大请求量。
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAIError,
)

from app.core.config import settings
from app.services.llm.types import StreamChunk

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 可重试的 HTTP 状态码：限流与服务端故障
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
# 明确不可重试：请求本身有问题，重试只是浪费配额
NON_RETRYABLE_STATUS_CODES = frozenset({400, 401, 402, 403, 404, 422})


def _resolve_error_type(error_type: str, cause: BaseException | None) -> str:
    """确定错误分类名。

    优先级：显式传入的 error_type > 原始异常类名 > unknown。
    这里必须显式判断，不能写成 ``error_type or type(cause).__name__ if cause else ...``
    ——那样会因运算符优先级变成「无 cause 就丢弃 error_type」，
    导致流式超时这类只传分类不传 cause 的场景全部记成 unknown。
    """
    if error_type:
        return error_type
    if cause is not None:
        return type(cause).__name__
    return "unknown"


class LLMRetryableError(Exception):
    """可重试的 LLM 调用失败。携带原始异常与错误分类。"""

    def __init__(self, message: str, *, cause: BaseException | None = None, error_type: str = "") -> None:
        super().__init__(message)
        self.cause = cause
        self.error_type = _resolve_error_type(error_type, cause)


class LLMNonRetryableError(Exception):
    """不可重试的 LLM 调用失败。立即进入降级链，不做无谓重试。"""

    def __init__(self, message: str, *, cause: BaseException | None = None, error_type: str = "") -> None:
        super().__init__(message)
        self.cause = cause
        self.error_type = _resolve_error_type(error_type, cause)


def classify_exception(exc: BaseException) -> tuple[bool, str]:
    """判断异常是否可重试，并返回错误分类名。

    返回 ``(is_retryable, error_type)``。error_type 落库到
    ``llm_call_logs.error_type``，用于统计各类故障占比。
    """
    # 超时：我们自己用 wait_for 包的，以及 SDK 抛的
    if isinstance(exc, asyncio.TimeoutError):
        return True, "timeout"
    if isinstance(exc, APITimeoutError):
        return True, "api_timeout"
    if isinstance(exc, APIConnectionError):
        # 连接层故障（DNS、网络中断、代理不可达），重试有意义
        return True, "connection_error"

    if isinstance(exc, APIStatusError):
        status_code = getattr(exc, "status_code", None)
        if status_code in NON_RETRYABLE_STATUS_CODES:
            return False, f"http_{status_code}"
        if status_code in RETRYABLE_STATUS_CODES:
            return True, f"http_{status_code}"
        # 未预期的状态码：保守起见不重试，避免放大未知故障
        return False, f"http_{status_code}"

    if isinstance(exc, OpenAIError):
        # SDK 内部错误（响应解析失败等），通常重试也无效
        return False, "sdk_error"

    # 非 OpenAI 异常（如我们自己的校验失败）不重试
    return False, type(exc).__name__


def calculate_backoff(attempt: int) -> float:
    """指数退避 + 抖动。

    加抖动（jitter）是必需的：多个客户端同时失败后若按固定间隔重试，
    会形成同步的重试洪峰，把刚恢复的服务再次打垮。
    """
    base = settings.llm_retry_backoff_base
    cap = settings.llm_retry_backoff_cap
    # full jitter：在 [0, 指数上限] 区间内随机取值
    exponential = min(cap, base * (2**attempt))
    return random.uniform(0, exponential)  # noqa: S311 - 非安全用途，仅用于退避抖动


@dataclass(slots=True)
class RetryOutcome:
    """重试执行结果。"""

    value: object | None = None
    attempts: int = 0
    succeeded: bool = False
    last_error: BaseException | None = None
    last_error_type: str = ""


async def execute_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int | None = None,
    timeout: float | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> RetryOutcome:
    """带超时与退避重试地执行一个异步操作。

    ``on_retry`` 回调用于记录每次重试（落日志、埋点），
    在 sleep 之前调用，便于统计重试次数分布。
    """
    retries = settings.llm_max_retries if max_retries is None else max_retries
    read_timeout = settings.llm_read_timeout if timeout is None else timeout
    outcome = RetryOutcome()

    for attempt in range(retries + 1):
        outcome.attempts = attempt + 1
        try:
            outcome.value = await asyncio.wait_for(operation(), timeout=read_timeout)
            outcome.succeeded = True
            return outcome
        except BaseException as exc:  # noqa: BLE001 - 需要分类所有异常类型
            is_retryable, error_type = classify_exception(exc)
            outcome.last_error = exc
            outcome.last_error_type = error_type

            if not is_retryable:
                # 请求本身有问题（400/401/403 等），重试只会浪费配额
                raise LLMNonRetryableError(
                    f"LLM 调用失败，不可重试（{error_type}）",
                    cause=exc,
                    error_type=error_type,
                ) from exc

            if attempt >= retries:
                # 可重试但次数已用尽。仍标记为 retryable，
                # 让调用方能区分「参数错误」与「服务暂时不可用」——
                # 后者才应该继续往降级链走，前者直接返回错误更合理。
                raise LLMRetryableError(
                    f"LLM 调用重试 {retries} 次后仍失败（{error_type}）",
                    cause=exc,
                    error_type=error_type,
                ) from exc

            delay = calculate_backoff(attempt)
            logger.warning(
                "LLM 调用失败，第 %d 次重试，%.2fs 后重试：%s",
                attempt + 1,
                delay,
                error_type,
            )
            if on_retry is not None:
                on_retry(attempt + 1, exc, delay)
            await asyncio.sleep(delay)

    return outcome


async def stream_with_timeouts(
    chunk_iterator: AsyncIterator[StreamChunk],
    *,
    first_token_timeout: float | None = None,
    idle_timeout: float | None = None,
) -> AsyncIterator[StreamChunk]:
    """给流式迭代加上首 token 与 chunk 间隔两段超时。

    实现方式是把底层迭代器包成 ``__anext__`` 任务，对每次取片单独
    ``wait_for``。首片用较短超时（过载应快速降级），后续片用间隔超时
    （长文本生成不该误杀）。

    超时后必须关闭底层迭代器，否则供应商侧的 HTTP 连接会泄漏。
    """
    first_timeout = (
        settings.llm_stream_first_token_timeout
        if first_token_timeout is None
        else first_token_timeout
    )
    idle = (
        settings.llm_stream_idle_timeout if idle_timeout is None else idle_timeout
    )

    received_first = False
    try:
        while True:
            timeout = idle if received_first else first_timeout
            try:
                chunk = await asyncio.wait_for(chunk_iterator.__anext__(), timeout=timeout)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                stage = "chunk_interval" if received_first else "first-token"
                raise LLMRetryableError(
                    f"流式响应{stage}超时", cause=exc, error_type=f"stream_{stage}_timeout"
                ) from exc

            # 只有真正的文本片段才算「首 token 已到」。
            # usage-only 的 final chunk 不带文本，不能用它解除首片超时。
            if chunk.text:
                received_first = True
            yield chunk
    finally:
        # 无论正常结束还是异常退出，都要释放底层连接
        aclose = getattr(chunk_iterator, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001 - 清理失败不应掩盖原始异常
                logger.debug("关闭流式迭代器时出错", exc_info=True)


__all__ = [
    "LLMRetryableError",
    "LLMNonRetryableError",
    "RetryOutcome",
    "RETRYABLE_STATUS_CODES",
    "NON_RETRYABLE_STATUS_CODES",
    "classify_exception",
    "calculate_backoff",
    "execute_with_retry",
    "stream_with_timeouts",
]
