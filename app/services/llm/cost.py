"""LLM 调用成本核算。

单价配置在 settings 里（美元 / 1K token）。这里刻意**不硬编码各模型的价格表**：
模型定价变动频繁，硬编码会悄悄算错成本且不易发现。统一用一组可配置单价，
面试项目规模下精度足够；生产环境应按 model 维度维护价格表并定期核对账单。

成本用 Decimal 计算避免浮点累积误差，落库时转 float 存入 Numeric(10,6)。
"""

from __future__ import annotations

from decimal import Decimal

from app.core.config import settings
from app.services.llm.types import LLMScenario, TokenUsage

# 输入输出分开计价：主流模型的输出 token 单价通常是输入的 3-4 倍，
# 合并成一个均价会让长回答场景的成本被显著低估。
_INPUT_PRICE_PER_TOKEN = Decimal(str(settings.llm_price_input_per_1k)) / 1000
_OUTPUT_PRICE_PER_TOKEN = Decimal(str(settings.llm_price_output_per_1k)) / 1000
_EMBEDDING_PRICE_PER_TOKEN = Decimal(str(settings.embedding_price_per_1k)) / 1000


def calculate_cost(usage: TokenUsage, *, scenario: LLMScenario | None = None) -> float:
    """按用量计算单次调用成本（美元）。

    Embedding 与 rerank 只有输入 token 且单价体系不同，单独走 embedding 价。
    """
    if scenario in (LLMScenario.EMBEDDING, LLMScenario.RERANK):
        total = Decimal(usage.prompt_tokens + usage.completion_tokens)
        return float(total * _EMBEDDING_PRICE_PER_TOKEN)

    cost = (
        Decimal(usage.prompt_tokens) * _INPUT_PRICE_PER_TOKEN
        + Decimal(usage.completion_tokens) * _OUTPUT_PRICE_PER_TOKEN
    )
    return float(cost)


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """不依赖 TokenUsage 对象的便捷估算，供预算阶段试算使用。"""
    usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return calculate_cost(usage)
