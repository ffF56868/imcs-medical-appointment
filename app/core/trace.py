"""链路追踪 ID（trace_id）上下文。

**为什么需要它**：一次业务请求（例如患者发起一次智能分诊）在系统里会牵出多步
动作——LLM 调用、向量检索、工作流节点流转、异步 Celery 任务。如果每一步各自
打日志，事后只能看到一堆孤立记录，无法回答「这次请求到底调了几次模型、花了多少
token、在哪一步降级」。因此约定：**一次业务请求对应一个 trace_id**，它贯穿
LLM 调用日志（``llm_call_logs.trace_id``）、检索日志与工作流追踪
（``workflow_traces.trace_id``），事后按同一个 trace_id 即可回放全链路。

**为什么用 contextvars 而不是 ``request.state``**：``request.state`` 只在
FastAPI 请求处理函数里可达。而本系统大量工作在请求之外执行——Celery worker
跑 embedding 与批量任务、``asyncio.create_task`` 派生的后台任务、流式响应的
生成器收尾段——这些地方都拿不到 request 对象。``contextvars`` 由事件循环自动
随协程传播（子任务继承创建时的上下文快照），因此只要在最外层 set 一次，链路上
任意深度的异步代码都能透明读到，不需要把 trace_id 当参数一路透传下去。

使用方式：
- HTTP 入口：由 ``app.core.trace_middleware.TraceIdMiddleware`` 自动注入。
- Celery / 脚本：用 ``use_trace_id(...)`` 上下文管理器临时绑定，退出即复原。
- 业务代码：一般不用管；LLM 服务层会在调用方未显式传 trace_id 时回退读取这里。
"""

from __future__ import annotations

import contextvars
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

# HTTP 头名称。客户端（前端网关、压测脚本、上游 BFF）可以自带该头，
# 服务端沿用其值，从而把链路延伸到系统之外。
TRACE_HEADER = "X-Trace-Id"

# 当前上下文的 trace_id。默认 None 表示「不在任何业务请求链路内」，
# 例如进程启动阶段、未经中间件包裹的单元测试。
_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "imcs_trace_id", default=None
)


def new_trace_id() -> str:
    """生成一个新的 trace_id。

    长度固定 36 字符（uuid4 的规范字符串形式），与
    ``llm_call_logs.trace_id`` / ``workflow_traces.trace_id`` 的
    ``String(36)`` 列宽严格对齐——若改成其他格式必须先确认列宽，
    否则超长值在 PG 上会被静默截断或直接报错。
    """
    return str(uuid.uuid4())


def get_trace_id() -> str | None:
    """读当前上下文的 trace_id，不在链路内时返回 ``None``。"""
    return _trace_id_var.get()


def set_trace_id(value: str | None) -> contextvars.Token:
    """写入当前上下文的 trace_id，返回用于复原的 Token。

    返回值必须交给 ``reset_trace_id`` 使用。不要丢弃 Token：
    contextvar 只有 reset 才能回到上一层的值，直接 set(None)
    会把「父上下文原本有值」这件事覆盖掉。
    """
    return _trace_id_var.set(value)


def reset_trace_id(token: contextvars.Token) -> None:
    """按 Token 复原 trace_id 到 set 之前的值。"""
    _trace_id_var.reset(token)


@contextmanager
def use_trace_id(value: str | None) -> Iterator[str | None]:
    """临时绑定 trace_id 的上下文管理器，退出时自动复原。

    给 Celery 任务和脚本用：这些地方没有中间件兜底，需要自己开一段链路。
    传 ``None`` 表示沿用当前上下文已有的值（不新造、也不清空），
    这样嵌套使用不会打断外层已经建立的 trace。

        with use_trace_id(new_trace_id()):
            await llm_service.chat(request)   # 日志自动带上这个 trace_id
    """
    bound = value if value is not None else get_trace_id()
    token = set_trace_id(bound)
    try:
        yield bound
    finally:
        reset_trace_id(token)


__all__ = [
    "TRACE_HEADER",
    "get_trace_id",
    "new_trace_id",
    "reset_trace_id",
    "set_trace_id",
    "use_trace_id",
]
