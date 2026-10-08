"""LLM 服务层数据契约。

所有类型都用 ``dataclass`` 而非 Pydantic：这些对象在服务层内部流转、
不直接暴露给 HTTP 边界，dataclass 的构造开销更低（流式场景每个 chunk
都要构造一次），也不需要校验与序列化能力。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class LLMScenario(str, enum.Enum):
    """调用场景。决定 Token 预算、模型选择与审核策略。

    把场景显式建模而不是散落在各处的字符串常量，是为了让
    「不同场景用不同预算/模型/审核规则」这件事有统一的挂载点。
    """

    CHAT = "chat"  # 患者健康问答
    TRIAGE = "triage"  # 智能分诊
    REPORT_INTERPRET = "report_interpret"  # 检查报告解读
    SUMMARY = "summary"  # 病历/会话摘要压缩
    MODERATION = "moderation"  # 内容审核（用小模型）
    QUERY_REWRITE = "query_rewrite"  # 多轮对话检索问题改写
    EMBEDDING = "embedding"  # 向量化
    RERANK = "rerank"  # 重排序


class ProviderName(str, enum.Enum):
    """LLM 供应商标识。"""

    OPENAI = "openai"
    DASHSCOPE = "dashscope"


class DegradationLevel(int, enum.Enum):
    """降级链层级。数值越大代表降级越深、服务质量越差。

    落库到 ``llm_call_logs.degradation_level``，据此统计降级率——
    降级率持续升高说明主供应商不稳定或预算配置过紧。
    """

    PRIMARY = 0  # 主模型正常返回
    FALLBACK_MODEL = 1  # 备用模型
    SEMANTIC_CACHE = 2  # Redis 语义缓存命中
    DETERMINISTIC = 3  # 确定性规则答案（结构化字段拼接，零 LLM 调用）
    CANNED_RESPONSE = 4  # 兜底话术


@dataclass(slots=True)
class ChatMessage:
    """单条对话消息。"""

    role: str  # system / user / assistant
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class TokenUsage:
    """Token 用量。

    ``is_estimated`` 标记数值来源：流式响应若供应商不支持
    ``stream_options.include_usage``，就只能用 tiktoken 或字符数估算，
    此时成本核算精度下降，需要在看板上区分展示。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    is_estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class LLMRequest:
    """一次 LLM 调用的完整入参。"""

    scenario: LLMScenario
    messages: list[ChatMessage]
    # 期望的模型；为空时由 scenario 决定
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    # 要求返回 JSON。structured_chat 会置为 True
    json_mode: bool = False
    stream: bool = False
    # 贯穿一次业务请求的链路 ID，落库到 llm_call_logs 与 workflow_traces
    trace_id: str | None = None
    user_id: Any | None = None
    # 工作流节点名，定位 token 消耗来源
    workflow_node: str | None = None
    # 关联的异步任务 ID
    ai_task_id: Any | None = None
    # 确定性降级答案的构造器。主模型与备用模型都失败时，
    # 降级链会调用它生成零 LLM 的答案（如药物问题直接拼结构化字段）
    deterministic_builder: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResponse:
    """一次 LLM 调用的结果。"""

    content: str
    usage: TokenUsage
    provider: str
    model: str
    requested_model: str | None = None
    latency_ms: int = 0
    first_token_ms: int | None = None
    status: str = "success"  # success / degraded / failed
    error_type: str | None = None
    retry_count: int = 0
    degradation_level: DegradationLevel = DegradationLevel.PRIMARY
    cost_usd: float = 0.0
    moderation_blocked: bool = False
    budget_exceeded: bool = False
    is_stream: bool = False
    trace_id: str | None = None
    # 结构化输出解析后的对象；structured_chat 填充
    parsed: Any | None = None

    @property
    def is_degraded(self) -> bool:
        return self.degradation_level != DegradationLevel.PRIMARY


@dataclass(slots=True)
class StreamChunk:
    """流式响应的单个片段。

    ``is_final`` 标记最后一个 chunk，它携带真实 usage、实际模型名与首 token
    延迟——这三项是流式场景下调用日志的必要字段，而中间的文本 chunk 不带，
    避免每片都重复构造。

    ``degradation`` 非空表示中途触发了降级，前端应展示提示条，
    而不是静默地把降级内容当正常回答呈现给患者。
    """

    text: str = ""
    usage: TokenUsage | None = None
    is_final: bool = False
    degradation: DegradationLevel | None = None
    error: str | None = None
    # 仅 final chunk 填充
    model: str | None = None
    first_token_ms: int | None = None
