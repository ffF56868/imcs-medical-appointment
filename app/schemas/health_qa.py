"""健康咨询问答的 HTTP 契约（阶段 7 批次 0）。

请求模型放这里，响应不走 Pydantic——SSE 的响应体是事件流，
由路由层直接拼 ``text/event-stream`` 文本，用 response_model 反而会误导
（FastAPI 会试图把返回值序列化成 JSON）。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChatTurnIn(BaseModel):
    """一轮历史对话。

    只接受 user/assistant 两种角色：system 由服务端固定下发，
    允许客户端传 system 等于把提示词边界交给调用方，
    那样「不得给出具体处方剂量」这类硬约束就能被绕过。
    """

    model_config = ConfigDict(extra="forbid")

    role: str = Field(pattern=r"^(user|assistant)$")
    content: str = Field(min_length=1, max_length=2000)


class HealthQARequest(BaseModel):
    """一次流式健康咨询的请求体。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=2, max_length=2000)
    # 最近几轮上下文。服务端只取最后 MAX_HISTORY_TURNS 轮，
    # 多传的丢弃而不是报错——前端不必自己裁剪。
    history: list[ChatTurnIn] = Field(default_factory=list, max_length=10)


__all__ = ["ChatTurnIn", "HealthQARequest"]
