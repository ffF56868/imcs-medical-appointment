"""LLM 调用日志：内存缓冲 + 批量异步落库。

**为什么不在请求链路里同步写库**：流式响应的首 token 延迟是用户体感最敏感的指标，
在 yield 第一片之前插一次 INSERT，等于给每个请求凭空加上一次数据库往返。
日志是可观测性数据而非业务数据，允许短暂延迟落库，因此改为：

- 攒够 ``llm_log_batch_size``（默认 50 条）立即刷盘
- 或距上次刷盘超过 ``llm_log_flush_interval``（默认 5s）由后台任务刷盘
- 应用关闭时强制刷盘，避免丢失尾部日志

代价是进程崩溃时缓冲区里的日志会丢。这是有意的取舍：
监控数据丢几条可以接受，拖慢每一次 AI 调用不行。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import insert

from app.core.config import settings
from app.models.ai import LLMCallLog

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMCallRecord:
    """一条待落库的调用记录。

    用 dataclass 而非直接构造 ORM 对象：缓冲期间不持有 ORM 实例，
    避免对象被意外绑定到已关闭的 session 上。
    """

    scenario: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    status: str = "success"
    trace_id: str | None = None
    requested_model: str | None = None
    cost_usd: float | None = None
    first_token_ms: int | None = None
    error_type: str | None = None
    http_status: int | None = None
    retry_count: int = 0
    degradation_level: int | None = None
    moderation_blocked: bool = False
    budget_exceeded: bool = False
    is_stream: bool = False
    user_id: Any | None = None
    workflow_node: str | None = None
    ai_task_id: Any | None = None


class LLMCallLogger:
    """调用日志缓冲器。

    线程安全由 asyncio 单事件循环保证；Celery worker 内通过
    ``asyncio.run()`` 进入事件循环，同样是单线程语义。
    刷盘用独立 session，不复用请求的 session——请求可能已经结束，
    而缓冲区里的日志属于更早的请求。
    """

    def __init__(
        self,
        *,
        batch_size: int | None = None,
        flush_interval: float | None = None,
    ) -> None:
        self._batch_size = batch_size or settings.llm_log_batch_size
        self._flush_interval = flush_interval or settings.llm_log_flush_interval
        self._buffer: list[LLMCallRecord] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._stopped = False
        # 统计刷盘失败次数，避免数据库故障时无限重试刷日志
        self._consecutive_failures = 0

    # ---------- 生命周期 ----------

    def start(self) -> None:
        """启动定时刷盘后台任务。应用 lifespan 里调用。"""
        if self._flush_task is None or self._flush_task.done():
            self._stopped = False
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """停止定时刷盘并强制清空缓冲区。应用关闭时调用。"""
        self._stopped = True
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._flush_task = None
        # 关闭前把尾部日志刷出去
        await self.flush()

    # ---------- 写入 ----------

    async def log(self, record: LLMCallRecord) -> None:
        """写入一条日志。攒够批量阈值就立即刷盘。"""
        async with self._lock:
            self._buffer.append(record)
            should_flush = len(self._buffer) >= self._batch_size
        if should_flush:
            await self.flush()

    def log_nowait(self, record: LLMCallRecord) -> None:
        """同步入队，不触发刷盘。

        流式响应的中途埋点用这个：不能为了记日志而在 yield 之间插入 await，
        那会打断流式输出的节奏。缓冲满时由下一次 log 或定时任务刷盘。
        """
        self._buffer.append(record)

    async def flush(self) -> int:
        """把缓冲区写入数据库，返回成功落库的条数。"""
        async with self._lock:
            if not self._buffer:
                return 0
            batch = self._buffer
            self._buffer = []

        try:
            await self._write_batch(batch)
            self._consecutive_failures = 0
            return len(batch)
        except Exception as exc:  # noqa: BLE001 - 日志失败不能影响主链路
            self._consecutive_failures += 1
            logger.warning(
                "LLM 调用日志落库失败（连续第 %d 次），丢弃 %d 条：%s",
                self._consecutive_failures,
                len(batch),
                exc,
            )
            # 连续失败说明数据库不可用，此时不再把日志塞回缓冲区，
            # 否则缓冲区会无限增长吃满内存。监控数据可丢，进程不能死。
            return 0

    async def _write_batch(self, batch: list[LLMCallRecord]) -> None:
        """批量 INSERT。用 Core insert 而非 ORM add_all，减少对象开销。"""
        from app.core.database import async_session_factory

        rows = [asdict(record) for record in batch]
        async with async_session_factory() as session:
            await session.execute(insert(LLMCallLog), rows)
            await session.commit()

    # ---------- 定时刷盘 ----------

    async def _flush_loop(self) -> None:
        """周期性刷盘。间隔可配，默认 5s。"""
        while not self._stopped:
            try:
                await asyncio.sleep(self._flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - 循环不能因单次失败而退出
                logger.exception("LLM 日志定时刷盘循环出错")
                await asyncio.sleep(1)

    # ---------- 内省 ----------

    @property
    def pending_count(self) -> int:
        """缓冲区待落库条数。监控接口用它暴露日志积压情况。"""
        return len(self._buffer)

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures


# 全局单例。整个进程共享一个缓冲区才能攒够批量。
call_logger = LLMCallLogger()


def record_llm_call(
    *,
    scenario: str,
    provider: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: int = 0,
    status: str = "success",
    trace_id: str | None = None,
    requested_model: str | None = None,
    cost_usd: float | None = None,
    first_token_ms: int | None = None,
    error_type: str | None = None,
    http_status: int | None = None,
    retry_count: int = 0,
    degradation_level: int | None = None,
    moderation_blocked: bool = False,
    budget_exceeded: bool = False,
    is_stream: bool = False,
    user_id: Any | None = None,
    workflow_node: str | None = None,
    ai_task_id: Any | None = None,
) -> LLMCallRecord:
    """构造一条调用日志记录。

    拆出这个工厂函数是为了让调用方用关键字参数写，
    而不是面对一个二十个位置参数的 dataclass 构造器。
    """
    return LLMCallRecord(
        scenario=scenario,
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        status=status,
        trace_id=trace_id,
        requested_model=requested_model,
        cost_usd=cost_usd,
        first_token_ms=first_token_ms,
        error_type=error_type,
        http_status=http_status,
        retry_count=retry_count,
        degradation_level=degradation_level,
        moderation_blocked=moderation_blocked,
        budget_exceeded=budget_exceeded,
        is_stream=is_stream,
        user_id=user_id,
        workflow_node=workflow_node,
        ai_task_id=ai_task_id,
    )


__all__ = ["LLMCallLogger", "LLMCallRecord", "call_logger", "record_llm_call"]
