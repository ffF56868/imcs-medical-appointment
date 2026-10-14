"""IMCS 链路追踪（trace_id）与 LLM 调用日志落库链路测试。

覆盖四件事，对应阶段 6 批次 A 的交付物：
1. ``app/core/trace.py`` 的 contextvar 语义：set/get/reset 正确复原，
   并发 task 之间互不串值——这是整套链路能按 trace 回放的前提。
2. ``use_trace_id`` 上下文管理器的进出语义，含「传 None 沿用外层值」这一条
   （Celery 任务里嵌套使用时靠它打断链路）。
3. ``TraceIdMiddleware``：响应头带 X-Trace-Id、外部传入时被沿用、
   超长值被截断到 String(36) 列宽以内。
4. ``call_logger`` 生命周期接入：``start()`` 建定时刷盘任务、
   ``stop()`` 取消任务并把尾部日志刷出去、lifespan 启停时确实调用了它们。

为什么不需要 PostgreSQL：本文件只测 contextvar、中间件与缓冲区状态机，
落库这一步用 monkeypatch 替换 ``_write_batch`` 拦下来；因此不依赖 psycopg，
也不依赖 aiosqlite，纯内存即可跑。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from httpx import ASGITransport, AsyncClient

from app.core.trace import (
    TRACE_HEADER,
    get_trace_id,
    new_trace_id,
    reset_trace_id,
    set_trace_id,
    use_trace_id,
)
from app.core.trace_middleware import MAX_TRACE_ID_LEN, TraceIdMiddleware, sanitize_trace_id
from app.services.llm.logger import LLMCallLogger, record_llm_call


# ==================== contextvar 基础语义 ====================


def test_new_trace_id_is_uuid4_string_of_36_chars() -> None:
    """trace_id 必须是 36 字符字符串，且两次生成不相等。

    长度不是洁癖：llm_call_logs.trace_id 是 String(36)，超宽在 PG 上会报错或截断。
    """
    value = new_trace_id()
    assert isinstance(value, str)
    assert len(value) == 36
    uuid.UUID(value)  # 非法格式会在这里抛 ValueError
    assert new_trace_id() != value


def test_default_trace_id_is_none() -> None:
    """未进入任何链路时读到 None，业务代码据此区分「无链路」与「有链路」。"""
    assert get_trace_id() is None


def test_set_get_reset_roundtrip() -> None:
    """reset 必须回到 set 之前的值，而不是简单置空。"""
    token = set_trace_id("outer")
    try:
        assert get_trace_id() == "outer"
        inner = set_trace_id("inner")
        try:
            assert get_trace_id() == "inner"
        finally:
            reset_trace_id(inner)
        # 关键断言：内层 reset 后恢复的是 outer，不是 None
        assert get_trace_id() == "outer"
    finally:
        reset_trace_id(token)
    assert get_trace_id() is None


async def test_concurrent_tasks_do_not_leak_trace_ids() -> None:
    """两个并发 task 各设各的 trace_id，互不串。

    这是 contextvars 相对全局变量的唯一存在理由：若实现退化成模块级变量，
    高并发下 A 请求的 LLM 日志会被打上 B 请求的 trace_id，
    链路回放直接错乱，而且这种 bug 在单线程串行测试里发现不了。
    """
    ids = [new_trace_id(), new_trace_id()]
    seen: dict[str, str] = {}

    async def worker(index: int) -> None:
        token = set_trace_id(ids[index])
        try:
            await asyncio.sleep(0.01 * (2 - index))  # 让两个 task 的区间真正交错
            seen[f"own_{index}"] = get_trace_id() or ""
        finally:
            reset_trace_id(token)
        seen[f"after_{index}"] = get_trace_id() or ""

    await asyncio.gather(worker(0), worker(1))

    assert seen["own_0"] == ids[0]
    assert seen["own_1"] == ids[1]
    # 退出后都回到 None，说明没有把别人的值留在自己的上下文里
    assert seen["after_0"] == ""
    assert seen["after_1"] == ""


async def test_child_task_inherits_parent_trace_id() -> None:
    """create_task 继承创建时的上下文快照。

    流式响应的收尾日志、后台派生任务能自动带上链路 ID，靠的就是这条语义。
    """
    value = new_trace_id()
    observed: list[str | None] = []

    async def reader() -> None:
        observed.append(get_trace_id())

    with use_trace_id(value):
        # 子 task 与同上下文里的直接 await 都要能读到父级 trace_id
        await asyncio.create_task(reader())
        await reader()

    assert observed == [value, value]


# ==================== use_trace_id ====================


def test_use_trace_id_binds_and_restores() -> None:
    value = new_trace_id()
    assert get_trace_id() is None
    with use_trace_id(value) as bound:
        assert bound == value
        assert get_trace_id() == value
    assert get_trace_id() is None


def test_use_trace_id_with_none_keeps_outer_value() -> None:
    """传 None 表示沿用外层链路，不能把已有 trace 清空。

    Celery 任务里常见「有则复用、无则不绑」的写法，若这里退化成了
    set(None)，嵌套一层就会把整条链路抹掉。
    """
    outer = new_trace_id()
    with use_trace_id(outer):
        with use_trace_id(None) as bound:
            assert bound == outer
            assert get_trace_id() == outer
        assert get_trace_id() == outer
    assert get_trace_id() is None


def test_use_trace_id_resets_on_exception() -> None:
    """异常路径同样要复原，否则一个失败请求会污染后续同上下文的调用。"""
    with pytest.raises(RuntimeError):
        with use_trace_id(new_trace_id()):
            raise RuntimeError("boom")
    assert get_trace_id() is None


# ==================== 中间件 ====================


def _build_app() -> FastAPI:
    """最小 app：只挂 TraceIdMiddleware，端点回显它在上下文里读到的值。

    其中一个端点在响应返回**之后**才读 trace_id，用来锁住 reset 时机这条契约：
    BaseHTTPMiddleware 会在 call_next 返回后另起一个 task 流式写出响应体；
    若中间件在 call_next 一返回就 reset，那个 task 里的代码（真实系统里就是
    流式收尾写的 LLM 日志）会读到 None，链路在最后一步断掉。
    这个回归只有专门断言才拦得住，所以固化成用例。

    不复用 app.imcs_main.app：那会连带 import 全部业务路由与数据库引擎，
    测试环境缺依赖时收集阶段就会失败。中间件行为与被包裹的业务无关，
    用最小 app 反而更能在 CI 上稳定复现。
    """
    mini = FastAPI()
    mini.add_middleware(TraceIdMiddleware)

    @mini.get("/echo")
    async def echo() -> dict[str, str | None]:
        return {"trace_id": get_trace_id()}

    @mini.get("/echo-in-stream")
    async def echo_in_stream() -> StreamingResponse:
        """响应体生成时才读 trace_id，模拟流式输出的收尾阶段。"""

        async def body() -> AsyncIterator[bytes]:
            yield json.dumps({"trace_id": get_trace_id()}).encode()

        return StreamingResponse(body(), media_type="application/json")

    return mini


async def _get(app: FastAPI, headers: dict[str, str] | None = None, path: str = "/echo"):  # noqa: ANN201, ANN202
    """向最小 app 发一次 GET，返回 httpx Response。

    返回值类型不显式标注：本机 Python 环境未装 httpx（依赖在 Linux venv 里），
    ``from httpx import Response`` 做类型别名会在 import 期就失败。
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path, headers=headers or {})


async def test_middleware_issues_trace_id_when_absent() -> None:
    """无入参头时自动生成，响应头与上下文值一致。"""
    response = await _get(_build_app())
    issued = response.headers[TRACE_HEADER]
    assert len(issued) == 36
    assert response.json()["trace_id"] == issued


async def test_middleware_reuses_incoming_trace_id() -> None:
    """上游已给 trace_id 时必须沿用，否则跨系统链路在入口断成两截。"""
    incoming = new_trace_id()
    response = await _get(_build_app(), {TRACE_HEADER: incoming})
    assert response.headers[TRACE_HEADER] == incoming
    assert response.json()["trace_id"] == incoming


async def test_middleware_isolates_concurrent_requests() -> None:
    """并发两个请求各自拿到独立 trace_id，互不覆盖。"""
    first, second = new_trace_id(), new_trace_id()
    r1, r2 = await asyncio.gather(
        _get(_build_app(), {TRACE_HEADER: first}),
        _get(_build_app(), {TRACE_HEADER: second}),
    )
    assert r1.json()["trace_id"] == first
    assert r2.json()["trace_id"] == second


async def test_middleware_clears_context_after_response() -> None:
    """请求结束后上下文回到 None：BaseHTTPMiddleware 之外的协程不该看到残留。"""
    await _get(_build_app())
    assert get_trace_id() is None


async def test_streaming_body_still_sees_trace_id() -> None:
    """响应体写出阶段仍读得到 trace_id。

    这是本次实现最容易写错的地方：若把 reset 放在 call_next 返回处，
    BaseHTTPMiddleware 之后流式写出响应体的 task 会读到 None，
    表现为流式对话的收尾日志（正是 token 与首字延迟那条）缺链路 ID。
    """
    response = await _get(_build_app(), path="/echo-in-stream")
    issued = response.headers[TRACE_HEADER]
    assert response.json()["trace_id"] == issued


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        (" abc ", "abc"),
        ("x" * 100, "x" * MAX_TRACE_ID_LEN),
    ],
)
def test_sanitize_trace_id(raw: str | None, expected: str | None) -> None:
    """空白视为缺失，超长截断到列宽；不做格式校验以免拒掉合法的自定义 ID。"""
    assert sanitize_trace_id(raw) == expected


async def test_middleware_truncates_overlong_header() -> None:
    """端到端确认截断生效（写进响应头和上下文的都是截断后的值）。"""
    response = await _get(_build_app(), {TRACE_HEADER: "y" * 100})
    assert len(response.headers[TRACE_HEADER]) == MAX_TRACE_ID_LEN
    assert response.json()["trace_id"] == "y" * MAX_TRACE_ID_LEN


# ==================== call_logger 生命周期 ====================


class FakeSink:
    """替代真实落库：记录每一批被刷出的记录数。"""

    def __init__(self) -> None:
        self.batches: list[int] = []

    async def __call__(self, batch: list[object]) -> None:
        self.batches.append(len(batch))


def _make_logger(**kwargs) -> tuple[LLMCallLogger, FakeSink]:  # noqa: ANN003
    """构造一个把落库换成内存 sink 的 logger，用于观察缓冲区状态机。

    必须自己传 flush_interval：默认值走 settings（5s），测试要等太久；
    显式传入后既快，也顺带覆盖「interval 可注入」这条契约。
    """
    sink = FakeSink()
    logger = LLMCallLogger(**kwargs)
    logger._write_batch = sink  # type: ignore[method-assign]
    return logger, sink


async def test_start_creates_flush_task_and_interval_flushes() -> None:
    """start() 之后定时器到点自动刷盘——这正是本次修复的主问题。

    修复前 lifespan 从不调 start()，低频场景的日志只能一直躺在缓冲区里，
    攒不满 batch(默认 50) 就永远不落库。
    """
    logger, sink = _make_logger(batch_size=100, flush_interval=0.02)
    logger.start()
    assert logger._flush_task is not None
    assert not logger._flush_task.done()

    await logger.log(record_llm_call(scenario="chat", provider="openai", model="m"))
    assert logger.pending_count == 1

    await asyncio.sleep(0.15)
    assert sink.batches == [1]
    assert logger.pending_count == 0
    await logger.stop()


async def test_stop_flushes_tail_and_cancels_task() -> None:
    """stop() 取消定时任务并把尾部日志刷出去，避免关闭丢最后几条。"""
    logger, sink = _make_logger(batch_size=100, flush_interval=3600)
    logger.start()
    for _ in range(3):
        await logger.log(record_llm_call(scenario="chat", provider="openai", model="m"))
    assert logger.pending_count == 3

    await logger.stop()

    assert sink.batches == [3]
    assert logger.pending_count == 0
    assert logger._flush_task is None


async def test_double_start_does_not_spawn_two_loop_tasks() -> None:
    """重复 start 不应产生多个刷盘循环，否则会重复刷同一批日志。"""
    logger, _sink = _make_logger(flush_interval=3600)
    logger.start()
    first = logger._flush_task
    logger.start()
    assert logger._flush_task is first
    await logger.stop()


async def test_batch_threshold_flushes_without_timer() -> None:
    """攒满 batch 立即落库，与定时器无关（interval 设得极大来隔离这条路径）。"""
    logger, sink = _make_logger(batch_size=2, flush_interval=3600)
    await logger.log(record_llm_call(scenario="chat", provider="openai", model="m"))
    assert sink.batches == []
    await logger.log(record_llm_call(scenario="chat", provider="openai", model="m"))
    assert sink.batches == [2]
    await logger.stop()


async def test_write_failure_does_not_raise_or_grow_buffer(monkeypatch) -> None:
    """数据库故障时日志被丢弃而非回灌缓冲区：监控数据可丢，进程不能 OOM。"""

    async def boom(batch: list[object]) -> None:
        raise RuntimeError("db down")

    logger = LLMCallLogger(batch_size=1, flush_interval=3600)
    monkeypatch.setattr(logger, "_write_batch", boom)
    await logger.log(record_llm_call(scenario="chat", provider="openai", model="m"))
    assert logger.pending_count == 0
    assert logger.consecutive_failures == 1
    await logger.stop()


# ==================== lifespan 接线 ====================


async def test_lifespan_starts_and_stops_call_logger(monkeypatch) -> None:
    """应用启动调 start()、关闭调 stop()，且 stop 早于引擎释放。

    顺序错了就等于白做：stop() 要用异步 session 写库，engine 先 dispose 的话
    尾部日志必然丢失，而现象只在滚动重启时才出现，极难定位。

    patch 目标是 ``app.imcs_main`` 模块上的名字而不是源模块：imcs_main 用
    ``from x import y`` 持有函数引用，patch 源模块不会影响已导入的引用。
    """
    import app.imcs_main as imcs_main_module
    from app.imcs_main import lifespan
    from app.services.llm import logger as llm_logger_module

    calls: list[str] = []

    def fake_start() -> None:
        calls.append("start")

    async def fake_stop() -> None:
        calls.append("stop")

    async def fake_close_redis() -> None:
        calls.append("redis")

    async def fake_dispose_async() -> None:
        calls.append("dispose_async")

    def fake_dispose_sync() -> None:
        calls.append("dispose_sync")

    monkeypatch.setattr(llm_logger_module.call_logger, "start", fake_start)
    monkeypatch.setattr(llm_logger_module.call_logger, "stop", fake_stop)
    monkeypatch.setattr(imcs_main_module, "close_redis", fake_close_redis)
    monkeypatch.setattr(imcs_main_module, "dispose_async_engine", fake_dispose_async)
    monkeypatch.setattr(imcs_main_module, "dispose_sync_engine", fake_dispose_sync)

    async with lifespan(FastAPI()):
        assert calls == ["start"]
    assert calls == ["start", "stop", "redis", "dispose_async", "dispose_sync"]


async def test_lifespan_survives_logger_start_failure(monkeypatch) -> None:
    """start() 抛异常只 warning，不阻断启动；stop() 同理不阻断关闭。"""
    import app.imcs_main as imcs_main_module
    from app.imcs_main import lifespan
    from app.services.llm import logger as llm_logger_module

    def explode_start() -> None:
        raise RuntimeError("no event loop")

    async def explode_stop() -> None:
        raise RuntimeError("db gone")

    async def noop_async() -> None:
        return None

    def noop() -> None:
        return None

    monkeypatch.setattr(llm_logger_module.call_logger, "start", explode_start)
    monkeypatch.setattr(llm_logger_module.call_logger, "stop", explode_stop)
    # 关掉真实的资源释放：测试进程里没有 PostgreSQL 连接，dispose 会卡在网络上
    monkeypatch.setattr(imcs_main_module, "close_redis", noop_async)
    monkeypatch.setattr(imcs_main_module, "dispose_async_engine", noop_async)
    # dispose_sync_engine 是同步函数，桩必须同为同步，不能被 await
    monkeypatch.setattr(imcs_main_module, "dispose_sync_engine", noop)

    entered = False
    async with lifespan(FastAPI()):
        entered = True
    assert entered is True


# ==================== service 层 trace 回退 ====================


def test_resolve_trace_id_priority() -> None:
    """显式传入 > contextvar > 新生成。

    第二优先级是这次打通链路的核心：调用方没传也能自动带上 HTTP 层的 trace。
    """
    from app.services.llm.service import LLMService
    from app.services.llm.types import LLMRequest, LLMScenario

    resolve = LLMService._resolve_trace_id
    explicit = LLMRequest(scenario=LLMScenario.CHAT, messages=[], trace_id="own")

    assert get_trace_id() is None
    assert resolve(explicit) == "own"

    implicit = LLMRequest(scenario=LLMScenario.CHAT, messages=[])
    generated = resolve(implicit)
    assert len(generated) == 36  # 无上下文时自造一个，日志不会缺 ID

    with use_trace_id("ctx"):
        assert resolve(implicit) == "ctx"
        assert resolve(explicit) == "own"


# ==================== 注册表一致性 ====================


def test_trace_module_exports_match_api_contract() -> None:
    """防止后续重构误改导出名：并行模块按这些名字引用本模块。"""
    import app.core.trace as trace_module

    assert TRACE_HEADER == "X-Trace-Id"
    for name in (
        "TRACE_HEADER",
        "get_trace_id",
        "new_trace_id",
        "reset_trace_id",
        "set_trace_id",
        "use_trace_id",
    ):
        assert name in trace_module.__all__
        assert hasattr(trace_module, name)
