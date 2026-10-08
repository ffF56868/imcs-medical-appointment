"""Token 预估与预算控制。

两层预算：

1. **请求级**：按场景设上限（分诊 2K、报告解读 8K、问答 1.5K）。
   超限时先截断历史消息，仍超则降级到小模型。
2. **日配额**：按用户 + 全局两级，用 Redis 原子计数。
   超限**不直接拒绝**，而是降级到更便宜的模型——医疗场景下拒绝服务
   的代价高于成本超支。

预估优先用 tiktoken（OpenAI 官方分词器，与计费口径一致）。
tiktoken 对 Qwen 的分词结果不准，因此 DashScope 侧退回字符数估算：
中文 1 token ≈ 1.5 字符，这个系数偏保守（实际约 1.0-1.3），
宁可高估导致提前降级，也不要低估导致超预算。

tiktoken 首次加载需要下载 BPE 文件，离线环境会失败，因此做了
惰性加载 + 异常兜底，任何情况下都不让预估本身成为故障点。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from app.core.cache import decrement, increment_with_ttl
from app.core.config import settings
from app.services.llm.types import LLMScenario, ProviderName

# tiktoken 惰性加载。编码对象构造有成本（需读取 BPE 表），全局复用一份。
_ENCODING = None
_ENCODING_LOAD_FAILED = False

# 一天的秒数，作为日配额键的 TTL
_DAY_SECONDS = 86400

# 场景 -> 请求级 token 上限
SCENARIO_TOKEN_BUDGET: dict[LLMScenario, int] = {
    LLMScenario.TRIAGE: settings.token_budget_triage,
    LLMScenario.REPORT_INTERPRET: settings.token_budget_report_interpret,
    LLMScenario.CHAT: settings.token_budget_chat,
    LLMScenario.SUMMARY: settings.token_budget_default,
    LLMScenario.MODERATION: 800,
    LLMScenario.QUERY_REWRITE: 500,
    LLMScenario.EMBEDDING: settings.token_budget_default,
    LLMScenario.RERANK: settings.token_budget_default,
}


def _get_encoding():
    """惰性加载 tiktoken 编码器。失败返回 None，调用方退回字符估算。"""
    global _ENCODING, _ENCODING_LOAD_FAILED
    if _ENCODING is not None or _ENCODING_LOAD_FAILED:
        return _ENCODING
    try:
        import tiktoken

        _ENCODING = tiktoken.encoding_for_model(settings.llm_model_chat)
    except Exception:  # noqa: BLE001 - 离线环境下载失败、模型名未知都要兜住
        _ENCODING_LOAD_FAILED = True
        _ENCODING = None
    return _ENCODING


def estimate_tokens(text: str, *, provider: ProviderName = ProviderName.OPENAI) -> int:
    """估算文本 token 数。

    OpenAI 走 tiktoken 精确计数；DashScope 走字符估算（系数偏保守）。
    空文本返回 0，不抛异常。
    """
    if not text:
        return 0

    if provider == ProviderName.OPENAI:
        encoding = _get_encoding()
        if encoding is not None:
            try:
                return len(encoding.encode(text, disallowed_special=()))
            except Exception:  # noqa: BLE001 - 特殊 token 序列可能编码失败
                pass

    # 字符估算兜底：按配置的字符/token 比例，向上取整
    ratio = max(0.1, settings.token_char_ratio_fallback)
    return max(1, math.ceil(len(text) / ratio))


def estimate_messages_tokens(
    messages: list,
    *,
    provider: ProviderName = ProviderName.OPENAI,
) -> int:
    """估算一组消息的 token 数。

    每条消息额外计 4 token 的协议开销（role、分隔符等），
    这是 OpenAI 计费的实际口径，漏算会导致长对话系统性低估。
    """
    total = 0
    for message in messages:
        content = getattr(message, "content", "") or ""
        role = getattr(message, "role", "") or ""
        total += estimate_tokens(content, provider=provider) + estimate_tokens(
            role, provider=provider
        ) + 4
    return total


def get_scenario_budget(scenario: LLMScenario) -> int:
    """取场景的请求级 token 上限。"""
    return SCENARIO_TOKEN_BUDGET.get(scenario, settings.token_budget_default)


@dataclass(slots=True)
class BudgetDecision:
    """预算检查结果。

    ``action`` 取值：
    - ``allow``：正常放行
    - ``downgrade_model``：日配额吃紧，降级到小模型继续服务
    - ``reject``：极端情况（全局配额耗尽）才拒绝
    """

    allowed: bool
    action: str
    estimated_tokens: int
    request_budget: int
    exceeded_request_budget: bool
    user_daily_used: int | None = None
    user_daily_limit: int = 0
    global_daily_used: int | None = None
    global_daily_limit: int = 0
    # Redis 不可用时为 True，此时按 fail-open 放行
    counter_unavailable: bool = False
    reason: str | None = None


def _today_key(prefix: str, scope: str) -> str:
    return f"{prefix}:{scope}:{date.today().isoformat()}"


async def check_token_budget(
    scenario: LLMScenario,
    messages: list,
    *,
    user_id: object | None = None,
    provider: ProviderName = ProviderName.OPENAI,
) -> BudgetDecision:
    """执行两层预算检查并**预扣**额度。

    预扣（先 INCRBY 再调用）而非事后记账，是为了防止并发请求同时通过检查
    然后集体超支。若调用最终失败，由 ``release_reserved_tokens`` 回滚。

    Redis 不可用时返回 ``counter_unavailable=True`` 并放行：
    限流组件故障不应该让整个 AI 服务不可用。
    """
    estimated = estimate_messages_tokens(messages, provider=provider)
    request_budget = get_scenario_budget(scenario)
    exceeded_request = estimated > request_budget

    user_key = _today_key("token_budget:user", str(user_id) if user_id else "anonymous")
    global_key = _today_key("token_budget:global", "all")

    user_used = await increment_with_ttl(user_key, estimated, _DAY_SECONDS)
    global_used = await increment_with_ttl(global_key, estimated, _DAY_SECONDS)

    if user_used is None and global_used is None:
        # 计数器完全不可用，fail-open
        return BudgetDecision(
            allowed=True,
            action="allow",
            estimated_tokens=estimated,
            request_budget=request_budget,
            exceeded_request_budget=exceeded_request,
            counter_unavailable=True,
            reason="Redis 不可用，跳过配额校验",
        )

    user_limit = settings.token_daily_limit_per_user
    global_limit = settings.token_daily_limit_global

    # 全局配额耗尽是硬拒绝：这通常意味着成本失控或遭遇刷量
    if global_used is not None and global_used > global_limit:
        await release_reserved_tokens(estimated, user_id=user_id)
        return BudgetDecision(
            allowed=False,
            action="reject",
            estimated_tokens=estimated,
            request_budget=request_budget,
            exceeded_request_budget=exceeded_request,
            user_daily_used=user_used,
            user_daily_limit=user_limit,
            global_daily_used=global_used,
            global_daily_limit=global_limit,
            reason="全局 Token 日配额已耗尽",
        )

    # 用户配额超限只降级，不拒绝
    if user_used is not None and user_used > user_limit:
        return BudgetDecision(
            allowed=True,
            action="downgrade_model",
            estimated_tokens=estimated,
            request_budget=request_budget,
            exceeded_request_budget=exceeded_request,
            user_daily_used=user_used,
            user_daily_limit=user_limit,
            global_daily_used=global_used,
            global_daily_limit=global_limit,
            reason="用户 Token 日配额已超限，降级到小模型",
        )

    return BudgetDecision(
        allowed=True,
        action="allow",
        estimated_tokens=estimated,
        request_budget=request_budget,
        exceeded_request_budget=exceeded_request,
        user_daily_used=user_used,
        user_daily_limit=user_limit,
        global_daily_used=global_used,
        global_daily_limit=global_limit,
    )


async def release_reserved_tokens(estimated: int, *, user_id: object | None = None) -> None:
    """回滚预扣额度。调用失败或审核拦截时使用。"""
    if estimated <= 0:
        return
    user_key = _today_key("token_budget:user", str(user_id) if user_id else "anonymous")
    global_key = _today_key("token_budget:global", "all")
    await decrement(user_key, estimated)
    await decrement(global_key, estimated)


async def settle_actual_tokens(
    estimated: int,
    actual: int,
    *,
    user_id: object | None = None,
) -> None:
    """用真实用量校正预扣值。

    预估与实际总有偏差（尤其是字符估算路径），调用成功后按差值补扣或回滚，
    让日配额计数贴近真实消耗，而不是长期系统性偏高或偏低。
    """
    delta = actual - estimated
    if delta == 0:
        return
    user_key = _today_key("token_budget:user", str(user_id) if user_id else "anonymous")
    global_key = _today_key("token_budget:global", "all")
    if delta > 0:
        await increment_with_ttl(user_key, delta, _DAY_SECONDS)
        await increment_with_ttl(global_key, delta, _DAY_SECONDS)
    else:
        await decrement(user_key, -delta)
        await decrement(global_key, -delta)
