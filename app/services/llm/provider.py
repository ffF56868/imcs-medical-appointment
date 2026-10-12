"""LLM 供应商适配层。

关键事实：**DashScope 提供 OpenAI 兼容端点**
（``https://dashscope.aliyuncs.com/compatible-mode/v1``），
因此 ``DashScopeAdapter`` 直接继承 ``OpenAIAdapter``，只覆写 base_url
与模型名映射，不需要维护两套 SDK 调用逻辑。

两个供应商都用 ``openai`` 官方 SDK 的 ``AsyncOpenAI`` 客户端。
不用 LangChain 抽象层：LangChain 会把 usage、错误类型、首 token 时间
这些我们明确需要统计的信息包在中间层里，取用反而更绕。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.services.llm.types import (
    ChatMessage,
    LLMScenario,
    ProviderName,
    StreamChunk,
    TokenUsage,
)


# 场景 -> 供应商侧模型名。DashScope 与 OpenAI 的模型命名完全不同，
# 由适配器负责映射，业务代码只传场景，不关心具体模型。
OPENAI_MODEL_MAP: dict[LLMScenario, str] = {
    LLMScenario.CHAT: settings.llm_model_chat,
    LLMScenario.TRIAGE: settings.llm_model_triage,
    LLMScenario.REPORT_INTERPRET: settings.llm_model_interpret,
    LLMScenario.SUMMARY: settings.llm_model_chat,
    LLMScenario.MODERATION: settings.llm_model_chat_fallback,
    LLMScenario.QUERY_REWRITE: settings.llm_model_chat_fallback,
}

# 降级用的备用模型（更小更便宜）
OPENAI_FALLBACK_MODEL_MAP: dict[LLMScenario, str] = {
    LLMScenario.CHAT: settings.llm_model_chat_fallback,
    LLMScenario.TRIAGE: settings.llm_model_triage_fallback,
    LLMScenario.REPORT_INTERPRET: settings.llm_model_interpret_fallback,
    LLMScenario.SUMMARY: settings.llm_model_chat_fallback,
    LLMScenario.MODERATION: settings.llm_model_chat_fallback,
    LLMScenario.QUERY_REWRITE: settings.llm_model_chat_fallback,
}


class ProviderAdapter(ABC):
    """供应商适配器抽象基类。"""

    name: ProviderName

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        scenario: LLMScenario,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        use_fallback_model: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> tuple[str, TokenUsage, str, int]:
        """非流式调用。返回 (内容, 用量, 实际模型名, 耗时毫秒)。"""

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        scenario: LLMScenario,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        use_fallback_model: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """流式调用。最后一个 chunk 带 is_final=True 与真实 usage。"""

    @abstractmethod
    async def embed(self, texts: list[str]) -> tuple[list[list[float]], TokenUsage, int]:
        """向量化。返回 (向量列表, 用量, 耗时毫秒)。"""

    def resolve_model(self, scenario: LLMScenario, *, use_fallback: bool) -> str:
        """按场景与是否降级解析出供应商侧模型名。"""
        raise NotImplementedError

    @property
    def is_configured(self) -> bool:
        """该供应商是否已配置 API Key。未配置的供应商不进降级链。"""
        raise NotImplementedError


class OpenAIAdapter(ProviderAdapter):
    """OpenAI 适配器。也是 DashScope 适配器的基类。"""

    name = ProviderName.OPENAI
    model_map = OPENAI_MODEL_MAP
    fallback_model_map = OPENAI_FALLBACK_MODEL_MAP

    def __init__(self) -> None:
        # 客户端复用连接池，不在每次调用时新建。
        # 超时拆成连接与读取两段：连接超时短（快速判断不可达），
        # 读取超时长（容忍长文本生成）。
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=(settings.llm_connect_timeout, settings.llm_read_timeout),
            # SDK 自带重试关掉，重试策略统一由 RetryPolicy 掌控，
            # 否则会出现「SDK 重试 2 次 × 我们重试 3 次 = 6 次」的叠加
            max_retries=0,
        )

    @property
    def api_key(self) -> str:
        return settings.openai_api_key

    @property
    def base_url(self) -> str:
        return settings.openai_base_url

    @property
    def embedding_model(self) -> str:
        return settings.embedding_model

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip())

    def resolve_model(self, scenario: LLMScenario, *, use_fallback: bool) -> str:
        table = self.fallback_model_map if use_fallback else self.model_map
        return table.get(scenario, settings.llm_model_chat)

    def _build_params(
        self,
        messages: list[ChatMessage],
        *,
        scenario: LLMScenario,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
        use_fallback_model: bool,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        resolved_model = model or self.resolve_model(
            scenario, use_fallback=use_fallback_model
        )
        params: dict[str, Any] = {
            "model": resolved_model,
            "messages": [message.to_dict() for message in messages],
            "temperature": (
                settings.llm_temperature if temperature is None else temperature
            ),
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if json_mode:
            # JSON mode 要求 prompt 里明确指示返回 JSON，否则部分模型报错
            params["response_format"] = {"type": "json_object"}
        if extra:
            params.update(extra)
        return params

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        scenario: LLMScenario,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        use_fallback_model: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> tuple[str, TokenUsage, str, int]:
        params = self._build_params(
            messages,
            scenario=scenario,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            use_fallback_model=use_fallback_model,
            extra=extra,
        )
        started = time.perf_counter()
        response = await self._client.chat.completions.create(**params)
        latency_ms = int((time.perf_counter() - started) * 1000)

        raw_usage = response.usage
        usage = TokenUsage(
            prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
            is_estimated=False,
        )
        content = ""
        if response.choices:
            content = response.choices[0].message.content or ""
        return content, usage, str(params["model"]), latency_ms

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        scenario: LLMScenario,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        use_fallback_model: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        params = self._build_params(
            messages,
            scenario=scenario,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            use_fallback_model=use_fallback_model,
            extra=extra,
        )
        # include_usage 让最后一个 chunk 携带真实 usage。
        # OpenAI 与 DashScope 兼容端点都支持，因此不必退化到本地估算。
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}

        started = time.perf_counter()
        first_token_ms: int | None = None
        resolved_model = str(params["model"])
        # 供应商会发一个 choices 为空、只带 usage 的最终事件。
        # 收到它就算流已结束，不能再补发 final chunk，否则调用方会重复结算用量。
        saw_usage_event = False

        stream = await self._client.chat.completions.create(**params)
        async for event in stream:
            # usage-only 的最终事件 choices 为空，需先取 usage 再判断
            raw_usage = getattr(event, "usage", None)
            if raw_usage is not None:
                saw_usage_event = True
                yield StreamChunk(
                    text="",
                    usage=TokenUsage(
                        prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
                        is_estimated=False,
                    ),
                    is_final=True,
                    model=resolved_model,
                    first_token_ms=first_token_ms,
                )
                continue

            if not event.choices:
                continue
            delta = event.choices[0].delta
            piece = getattr(delta, "content", None) or ""
            if piece:
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - started) * 1000)
                yield StreamChunk(text=piece)

        # 供应商没发 usage 事件时补一个 final chunk，让调用方知道流已结束。
        # 此时 usage 为 None，由调用方用 tiktoken 估算兜底。
        if not saw_usage_event:
            yield StreamChunk(
                text="",
                usage=None,
                is_final=True,
                model=resolved_model,
                first_token_ms=first_token_ms,
            )

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], TokenUsage, int]:
        if not texts:
            return [], TokenUsage(), 0
        started = time.perf_counter()
        response = await self._client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        vectors = [item.embedding for item in response.data]
        raw_usage = response.usage
        usage = TokenUsage(
            prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            completion_tokens=0,
            is_estimated=False,
        )
        return vectors, usage, latency_ms


class DashScopeAdapter(OpenAIAdapter):
    """阿里云百炼适配器。

    继承 OpenAI 适配器，只覆写 Key、base_url 与模型映射表。
    DashScope 的 OpenAI 兼容端点接受相同的请求/响应结构，
    包括 ``stream_options.include_usage``。
    """

    name = ProviderName.DASHSCOPE

    # DashScope 模型名与 OpenAI 完全不同，需要独立映射表。
    # qwen-plus 对标 gpt-4.1，qwen-turbo 对标 gpt-4.1-mini。
    model_map = {
        LLMScenario.CHAT: "qwen-turbo",
        LLMScenario.TRIAGE: "qwen-turbo",
        LLMScenario.REPORT_INTERPRET: "qwen-plus",
        LLMScenario.SUMMARY: "qwen-turbo",
        LLMScenario.MODERATION: "qwen-turbo",
        LLMScenario.QUERY_REWRITE: "qwen-turbo",
    }
    fallback_model_map = {
        LLMScenario.CHAT: "qwen-turbo",
        LLMScenario.TRIAGE: "qwen-turbo",
        LLMScenario.REPORT_INTERPRET: "qwen-turbo",
        LLMScenario.SUMMARY: "qwen-turbo",
        LLMScenario.MODERATION: "qwen-turbo",
        LLMScenario.QUERY_REWRITE: "qwen-turbo",
    }

    @property
    def api_key(self) -> str:
        return settings.dashscope_api_key

    @property
    def base_url(self) -> str:
        return settings.dashscope_base_url

    @property
    def embedding_model(self) -> str:
        # text-embedding-v3 输出 1024 维，与 OpenAI text-embedding-3-small 的
        # 1536 维不同。切换供应商的 embedding 必须同步改向量列定义并全量重建，
        # 因此默认仍用 OpenAI 的 embedding 模型名占位，由配置显式指定。
        return settings.embedding_model


_ADAPTER_CACHE: dict[ProviderName, ProviderAdapter] = {}


def get_adapter(provider: ProviderName | str) -> ProviderAdapter:
    """取供应商适配器单例。客户端内含连接池，必须复用。"""
    key = ProviderName(provider) if not isinstance(provider, ProviderName) else provider
    adapter = _ADAPTER_CACHE.get(key)
    if adapter is None:
        adapter = DashScopeAdapter() if key == ProviderName.DASHSCOPE else OpenAIAdapter()
        _ADAPTER_CACHE[key] = adapter
    return adapter


def reset_adapter_clients() -> None:
    """清空适配器缓存，下次 ``get_adapter`` 会重建客户端。

    **为什么必须存在**：``AsyncOpenAI`` 内含 httpx 连接池，而 httpx 的连接
    绑定到**首次使用时正在运行的事件循环**。FastAPI 进程里循环只有一个，
    进程级缓存是免费的；但 Celery prefork worker 每个任务用
    ``asyncio.run()`` 新建并关闭一个循环——第二个任务复用旧客户端时，
    连接池里躺着属于上一个已关闭循环的连接，调用会抛
    ``RuntimeError: Event loop is closed``（或更隐蔽的跨循环挂起）。

    所以 ``celery_app.run_async()`` 在每个任务开始前调用本函数丢弃缓存，
    让任务在自己的循环里重建客户端。代价是每次重建连接池（几十毫秒），
    相对 LLM 调用的秒级耗时可以忽略；旧客户端随引用消失由 GC 回收其
    socket，prefork 进程内不存在跨任务并发，不会堆积。

    刻意不做的优化：按「当前循环 id」做二级缓存。CPython 会复用已释放对象
    的内存地址，新循环完全可能拿到旧循环的 id——用 id 当键会让已关闭循环
    的客户端被误判为可用，那是比现在更难排查的 bug。
    """
    _ADAPTER_CACHE.clear()


def get_configured_providers() -> list[ProviderName]:
    """按主备顺序返回已配置 Key 的供应商。未配置的直接跳过，不参与降级链。"""
    ordered: list[ProviderName] = []
    for raw in (settings.llm_primary_provider, settings.llm_fallback_provider):
        try:
            provider = ProviderName(raw)
        except ValueError:
            continue
        if provider in ordered:
            continue
        if get_adapter(provider).is_configured:
            ordered.append(provider)
    return ordered


__all__ = [
    "ProviderAdapter",
    "OpenAIAdapter",
    "DashScopeAdapter",
    "get_adapter",
    "get_configured_providers",
]
