"""分级降级链。

主模型 → 备用模型 → 语义缓存 → 确定性规则答案 → 兜底话术。

设计要点：

**降级不是失败，是服务质量的分级。** 每一级都能返回可用结果，
只是可信度与个性化程度递减。降级链的存在让「供应商挂了」从
「服务不可用」变成「服务质量下降」，这是医疗场景必须做到的。

**确定性规则答案这一级最有价值。** 旧项目 ``ask.py`` 里的
``build_deterministic_answer()`` 已经证明：药物用法这类问题可以直接从
结构化字段拼接答案，零 LLM 调用、零延迟、零成本、且不存在幻觉风险。
新架构把它从路由里的特殊分支提升为降级链的正式一级——
它不只是兜底，对于结构化程度高的问题它本来就是**更优**答案。

**每次降级都记录层级**，落库到 ``llm_call_logs.degradation_level``。
降级率是核心运维指标：持续升高说明主供应商不稳定或预算配置过紧。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.services.llm.types import DegradationLevel

logger = logging.getLogger(__name__)


# 兜底话术。所有降级手段都失效时返回，必须诚实说明系统状态，
# 不能伪装成正常回答——患者据此做医疗决策，误导的代价远高于承认失败。
CANNED_FALLBACK_TEXT = (
    "很抱歉，AI 服务当前不可用，无法为你生成回答。\n\n"
    "你可以：\n"
    "1. 稍后重试；\n"
    "2. 直接预约挂号，由医生为你解答；\n"
    "3. 如有紧急症状，请立即拨打 120 或前往最近的急诊科。\n\n"
    "本系统提供健康信息参考，不能替代医生的诊断与治疗建议。"
)

# 报告解读场景的兜底话术。面向医生，措辞不同：
# 医生需要知道「AI 没出结果」以便自行判读，而不是收到一段患者向的安抚文案。
CANNED_INTERPRET_FALLBACK_TEXT = (
    "AI 解读服务当前不可用，未能生成本次报告的自动解读。\n"
    "请依据原始检验数据自行判读，稍后可重新触发解读任务。"
)


@dataclass(slots=True)
class DegradationResult:
    """降级链某一级的执行结果。"""

    content: str | None
    level: DegradationLevel
    succeeded: bool
    error: str | None = None
    # 该级是否可用。语义缓存未命中、确定性构造器未提供，都算不可用，
    # 与「执行了但失败」是不同情况，日志里要能区分。
    attempted: bool = True


# 降级级的构造器签名：无参异步函数，返回文本或 None
LevelBuilder = Callable[[], Awaitable[str | None]]


class DegradationChain:
    """分级降级链。

    每一级是一个 ``(level, name, builder)`` 三元组。builder 返回 None
    表示该级不适用（如无缓存命中），链继续往下走；抛异常表示该级失败，
    同样继续往下走，但会记录错误。

    主模型与备用模型这两级由 ``LLMService`` 直接处理（它们需要完整的
    重试与超时逻辑），本类负责**模型全部失败之后**的三级降级。
    """

    def __init__(
        self,
        *,
        semantic_cache_lookup: LevelBuilder | None = None,
        deterministic_builder: LevelBuilder | None = None,
        canned_text: str = CANNED_FALLBACK_TEXT,
    ) -> None:
        self._semantic_cache_lookup = semantic_cache_lookup
        self._deterministic_builder = deterministic_builder
        self._canned_text = canned_text

    async def run(self, *, error_context: str = "") -> tuple[str, DegradationLevel, list[str]]:
        """依次尝试各级降级，返回 (内容, 最终层级, 各级错误列表)。

        保证一定返回可用内容——最后一级是无条件成功的兜底话术。
        """
        errors: list[str] = []

        # 第 2 级：语义缓存
        if self._semantic_cache_lookup is not None:
            result = await self._try_level(
                DegradationLevel.SEMANTIC_CACHE,
                "semantic_cache",
                self._semantic_cache_lookup,
            )
            if result.succeeded and result.content:
                logger.info("LLM 降级至语义缓存：%s", error_context)
                return result.content, result.level, errors
            if result.error:
                errors.append(f"semantic_cache: {result.error}")

        # 第 3 级：确定性规则答案
        if self._deterministic_builder is not None:
            result = await self._try_level(
                DegradationLevel.DETERMINISTIC,
                "deterministic",
                self._deterministic_builder,
            )
            if result.succeeded and result.content:
                logger.info("LLM 降级至确定性规则答案：%s", error_context)
                return result.content, result.level, errors
            if result.error:
                errors.append(f"deterministic: {result.error}")

        # 第 4 级：兜底话术。无条件成功。
        logger.warning("LLM 降级至兜底话术：%s", error_context)
        return self._canned_text, DegradationLevel.CANNED_RESPONSE, errors

    async def _try_level(
        self,
        level: DegradationLevel,
        name: str,
        builder: LevelBuilder,
    ) -> DegradationResult:
        """执行单级降级，捕获所有异常保证链不中断。"""
        try:
            content = await builder()
            if content is None:
                return DegradationResult(
                    content=None, level=level, succeeded=False, attempted=False
                )
            return DegradationResult(content=content, level=level, succeeded=True)
        except Exception as exc:  # noqa: BLE001 - 降级链任何一级都不能中断整条链
            logger.warning("降级级 %s 执行失败：%s", name, exc, exc_info=True)
            return DegradationResult(
                content=None,
                level=level,
                succeeded=False,
                error=f"{type(exc).__name__}: {exc}",
            )


def get_canned_text(scenario: str) -> str:
    """按场景取兜底话术。面向医生与面向患者的措辞必须不同。"""
    if scenario == "report_interpret":
        return CANNED_INTERPRET_FALLBACK_TEXT
    return CANNED_FALLBACK_TEXT


__all__ = [
    "DegradationChain",
    "DegradationResult",
    "LevelBuilder",
    "CANNED_FALLBACK_TEXT",
    "CANNED_INTERPRET_FALLBACK_TEXT",
    "get_canned_text",
]
