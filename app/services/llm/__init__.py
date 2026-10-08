"""统一 LLM 服务层。

这是 IMCS 的 AI 工程化核心。所有业务代码只依赖 ``LLMService``，
不直接调用任何供应商 SDK，从而把超时、重试、Token 预算、内容审核、
失败降级、调用日志这些横切关注点收敛到一处。

模块划分：
- ``types``      请求/响应/用量等数据契约
- ``provider``   供应商适配（OpenAI / DashScope）
- ``token_budget`` Token 预估与预算控制
- ``retry``      超时与重试策略
- ``moderation`` 输入输出内容审核
- ``degradation`` 分级降级链
- ``logger``     调用日志批量落库
- ``cost``       成本核算
- ``service``    对外统一入口
"""

from app.services.llm.service import LLMService, get_llm_service
from app.services.llm.types import (
    ChatMessage,
    DegradationLevel,
    LLMRequest,
    LLMResponse,
    LLMScenario,
    StreamChunk,
    TokenUsage,
)

__all__ = [
    "LLMService",
    "get_llm_service",
    "ChatMessage",
    "DegradationLevel",
    "LLMRequest",
    "LLMResponse",
    "LLMScenario",
    "StreamChunk",
    "TokenUsage",
]
