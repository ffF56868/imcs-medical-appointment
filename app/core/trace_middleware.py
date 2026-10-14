"""trace_id 注入中间件。

放在中间件栈的**最外层**，这样后续任何一层（安全头、CORS、业务路由、
异常处理器）以及它们派生的异步任务，都能从 contextvar 里读到同一个 trace_id。
反过来若放在内层，外层抛出的异常日志就会缺链路 ID——而恰恰是最需要追踪的
失败请求会丢链路。

设计取舍：
- **沿用外部传入的值**：上游网关/前端已经生成 trace_id 时必须尊重它，
  否则跨系统的链路在入口处被截断，两边各有一套 ID 无法对齐。
- **不校验入参格式**：过度严格会把合法的自定义 ID（如 Jaeger 的 128bit hex）
  拒之门外；这里只做长度截断，防止有人用一个 10KB 的头把日志和数据库撑爆。
- **响应头写回**：客户端报错时能直接拿到 trace_id 反馈给客服/研发，
  这是排查线上问题成本最低的一条路径。
"""

from __future__ import annotations

from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.trace import TRACE_HEADER, get_trace_id, new_trace_id, reset_trace_id, set_trace_id

# 与 llm_call_logs.trace_id / workflow_traces.trace_id 的 String(36) 对齐。
MAX_TRACE_ID_LEN = 36


def sanitize_trace_id(raw: str | None) -> str | None:
    """清洗外部传入的 trace_id：去空白、限长，非法值视为缺失。

    只截断不报错：一个超长 ID 不值得让整个请求 400，
    截断后仍能对上大部分链路，且不会因为写库失败连带丢日志。
    """
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return value[:MAX_TRACE_ID_LEN]


class TraceIdMiddleware(BaseHTTPMiddleware):
    """为每个 HTTP 请求绑定 trace_id，并写回响应头。"""

    async def dispatch(self, request: Request, call_next: Any) -> Response:  # noqa: ANN401
        """进入时 set，响应体写完后 reset。

        ``BaseHTTPMiddleware`` 在每个请求上跑一个新 task，contextvar 的写入
        因此天然隔离，不会串到并发请求；reset 是防御性收尾，避免中间件之外的
        同上下文代码读到上一次请求残留的值。

        **reset 必须放在最外层 finally**：``call_next`` 返回只代表拿到了响应对象，
        真正的响应体（尤其是 StreamingResponse）是由 Starlette 另起的一个 task
        在同上下文里流式写出的。若在 call_next 返回后就 reset，那个 task 里的
        代码会读到 None —— 具体症状是流式对话收尾时写的 LLM 日志缺 trace_id，
        而这条日志恰恰是统计首 token 延迟的唯一来源。

        反过来，写在 finally 里也不会污染下一个请求：BaseHTTPMiddleware 给每个
        请求单独建 task，上下文彼此隔离；这里 reset 只是为了让同一 task 内
        中间件之外的后续代码不残留上一次请求的值。
        """
        trace_id = sanitize_trace_id(request.headers.get(TRACE_HEADER)) or new_trace_id()
        token = set_trace_id(trace_id)
        try:
            response = await call_next(request)
            response.headers[TRACE_HEADER] = trace_id
            return response
        finally:
            reset_trace_id(token)


__all__ = [
    "MAX_TRACE_ID_LEN",
    "TRACE_HEADER",
    "TraceIdMiddleware",
    "get_trace_id",
    "sanitize_trace_id",
]
