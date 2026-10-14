"""统一 LLM 服务入口。

业务代码只依赖这一个类。它把供应商适配、Token 预算、超时重试、
内容审核、失败降级、调用日志、成本核算串成一条完整链路：

    输入审核 → 预算检查(预扣) → 逐供应商重试调用
        → 输出审核 → 真实用量校正 → 落库调用日志
        → 全部失败则走降级链

**为什么要有这一层**：没有它，超时/重试/日志/降级这些横切关注点会散落在
每个业务路由里，写法各不相同且极易漏项。旧项目 ``ask.py`` 就是反面教材——
裸调 ``get_chat_model().invoke()``，没有超时、没有重试、没有 token 统计，
供应商一抖动整个问答就挂。收敛到一处后，任何 AI 能力的改进都自动覆盖全系统。
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.core.trace import get_trace_id, new_trace_id
from app.services.llm.cost import calculate_cost
from app.services.llm.degradation import DegradationChain, get_canned_text
from app.services.llm.logger import call_logger, record_llm_call
from app.services.llm.moderation import moderator
from app.services.llm.provider import get_adapter, get_configured_providers
from app.services.llm.retry import (
    LLMNonRetryableError,
    LLMRetryableError,
    execute_with_retry,
    stream_with_timeouts,
)
from app.services.llm.token_budget import (
    check_token_budget,
    estimate_tokens,
    release_reserved_tokens,
    settle_actual_tokens,
)
from app.services.llm.types import (
    ChatMessage,
    DegradationLevel,
    LLMRequest,
    LLMResponse,
    LLMScenario,
    ProviderName,
    StreamChunk,
    TokenUsage,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMService:
    """统一 LLM 服务。

    无状态设计（缓冲区在模块级单例里），可安全共享一个实例。
    """

    def __init__(self) -> None:
        self._moderator = moderator
        self._logger = call_logger

    # ==================== 链路 ID ====================

    @staticmethod
    def _resolve_trace_id(request: LLMRequest) -> str:
        """确定本次调用归属的 trace_id。

        优先级：请求显式指定 → HTTP 中间件写入的 contextvar → 新生成。
        中间那一级是关键：业务代码（分诊、报告解读等）大多不会手动传
        ``trace_id``，但 LLM 日志必须能和同一次请求的检索日志、工作流追踪
        对上号，否则事后无法按链路回放。放在这里而不是 logger 内部，是因为
        同一个 trace_id 还要回填进 ``LLMResponse.trace_id`` 返回给调用方。
        """
        return request.trace_id or get_trace_id() or new_trace_id()

    # ==================== 非流式 ====================

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """非流式调用。保证返回结果，永不抛供应商异常。

        这是刻意的设计：调用方（业务 service）不应该被迫处理
        「OpenAI 超时了怎么办」。降级链保证任何情况下都有可用输出，
        调用方只需检查 ``response.is_degraded`` 决定是否向用户披露降级。
        """
        started = time.perf_counter()
        trace_id = self._resolve_trace_id(request)

        # ---------- 1. 输入审核 ----------
        user_text = self._extract_user_text(request.messages)
        input_check = self._moderator.moderate_input(user_text)
        if input_check.blocked:
            await self._log(
                request,
                trace_id=trace_id,
                provider="",
                model=request.model or "",
                status="blocked",
                error_type="input_moderation",
                latency_ms=int((time.perf_counter() - started) * 1000),
                moderation_blocked=True,
            )
            return LLMResponse(
                content=self._input_blocked_reply(input_check.categories),
                usage=TokenUsage(),
                provider="none",
                model=request.model or "",
                latency_ms=int((time.perf_counter() - started) * 1000),
                status="blocked",
                error_type="input_moderation",
                degradation_level=DegradationLevel.CANNED_RESPONSE,
                moderation_blocked=True,
                trace_id=trace_id,
            )

        # ---------- 2. Token 预算检查与预扣 ----------
        providers = get_configured_providers()
        primary_provider = providers[0] if providers else ProviderName.OPENAI
        budget = await check_token_budget(
            request.scenario,
            request.messages,
            user_id=request.user_id,
            provider=primary_provider,
        )
        if not budget.allowed:
            await self._log(
                request,
                trace_id=trace_id,
                provider="",
                model=request.model or "",
                status="budget_exceeded",
                error_type="token_budget",
                latency_ms=int((time.perf_counter() - started) * 1000),
                budget_exceeded=True,
            )
            return LLMResponse(
                content=(
                    "AI 服务今日调用额度已用完，暂时无法回答。"
                    "你可以直接预约挂号由医生解答；如有紧急症状请立即就医。"
                ),
                usage=TokenUsage(),
                provider="none",
                model=request.model or "",
                latency_ms=int((time.perf_counter() - started) * 1000),
                status="budget_exceeded",
                error_type="token_budget",
                degradation_level=DegradationLevel.CANNED_RESPONSE,
                budget_exceeded=True,
                trace_id=trace_id,
            )

        # 预算判定需要降级模型（用户日配额超限）或请求级超预算时，直接用备用模型
        use_fallback_model = (
            budget.action == "downgrade_model" or budget.exceeded_request_budget
        )

        # ---------- 3. 逐供应商调用 ----------
        if not providers:
            logger.error("没有已配置的 LLM 供应商，直接走降级链")
            return await self._degrade(
                request,
                trace_id=trace_id,
                started=started,
                error_context="no_provider_configured",
                budget_estimated=budget.estimated_tokens,
            )

        last_error_type = "unknown"
        total_retries = 0
        requested_model = request.model or get_adapter(primary_provider).resolve_model(
            request.scenario, use_fallback=use_fallback_model
        )

        for provider in providers:
            adapter = get_adapter(provider)
            try:
                outcome = await execute_with_retry(
                    lambda a=adapter: a.chat(
                        request.messages,
                        scenario=request.scenario,
                        model=request.model,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        json_mode=request.json_mode,
                        use_fallback_model=use_fallback_model,
                        extra=request.extra,
                    ),
                    timeout=settings.llm_read_timeout,
                )
                total_retries += max(0, outcome.attempts - 1)
                content, usage, actual_model, latency_ms = outcome.value  # type: ignore[misc]

                # ---------- 4. 输出审核 ----------
                output_check = self._moderator.moderate_output(
                    content, scenario=request.scenario
                )
                final_content = content
                if output_check.blocked:
                    # 输出违规：丢弃模型内容，走降级链拿安全答案。
                    # 不能只是「删掉违规句子」——那样会产出语义破碎的回答。
                    logger.warning(
                        "LLM 输出被审核拦截：%s trace=%s",
                        output_check.categories,
                        trace_id,
                    )
                    await self._log(
                        request,
                        trace_id=trace_id,
                        provider=provider.value,
                        model=actual_model,
                        requested_model=requested_model,
                        usage=usage,
                        status="blocked",
                        error_type="output_moderation",
                        latency_ms=latency_ms,
                        retry_count=total_retries,
                        moderation_blocked=True,
                    )
                    # 释放预扣额度：这次调用的结果没有被采用
                    await release_reserved_tokens(
                        budget.estimated_tokens, user_id=request.user_id
                    )
                    return await self._degrade(
                        request,
                        trace_id=trace_id,
                        started=started,
                        error_context="output_moderation_blocked",
                        budget_estimated=0,
                    )

                if output_check.appended_notice:
                    # 高风险主题：放行但附加危机干预提示
                    final_content = f"{content}\n\n{output_check.appended_notice}"

                cost = calculate_cost(usage, scenario=request.scenario)
                total_latency = int((time.perf_counter() - started) * 1000)

                await self._log(
                    request,
                    trace_id=trace_id,
                    provider=provider.value,
                    model=actual_model,
                    requested_model=requested_model,
                    usage=usage,
                    status="success",
                    latency_ms=total_latency,
                    cost_usd=cost,
                    retry_count=total_retries,
                    degradation_level=DegradationLevel.PRIMARY.value,
                )
                # 用真实用量校正预扣值
                await settle_actual_tokens(
                    budget.estimated_tokens,
                    usage.total_tokens,
                    user_id=request.user_id,
                )

                return LLMResponse(
                    content=final_content,
                    usage=usage,
                    provider=provider.value,
                    model=actual_model,
                    requested_model=requested_model,
                    latency_ms=total_latency,
                    status="success",
                    retry_count=total_retries,
                    degradation_level=DegradationLevel.PRIMARY,
                    cost_usd=cost,
                    trace_id=trace_id,
                )

            except (LLMRetryableError, LLMNonRetryableError) as exc:
                last_error_type = exc.error_type
                total_retries += 1
                logger.warning(
                    "供应商 %s 调用失败（%s），尝试下一个供应商",
                    provider.value,
                    exc.error_type,
                )
                await self._log(
                    request,
                    trace_id=trace_id,
                    provider=provider.value,
                    model=requested_model,
                    requested_model=requested_model,
                    status="failed",
                    error_type=exc.error_type,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    retry_count=total_retries,
                )
                continue
            except Exception as exc:  # noqa: BLE001 - 未预期异常也要进降级链
                last_error_type = type(exc).__name__
                logger.exception("供应商 %s 出现未预期异常", provider.value)
                await self._log(
                    request,
                    trace_id=trace_id,
                    provider=provider.value,
                    model=requested_model,
                    requested_model=requested_model,
                    status="failed",
                    error_type=last_error_type,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    retry_count=total_retries,
                )
                continue

        # ---------- 5. 所有供应商失败，走降级链 ----------
        return await self._degrade(
            request,
            trace_id=trace_id,
            started=started,
            error_context=f"all_providers_failed:{last_error_type}",
            budget_estimated=budget.estimated_tokens,
            retry_count=total_retries,
        )

    # ==================== 流式 ====================

    async def chat_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """流式调用。

        与非流式的关键差异：**降级发生在流开始之后就不能悄悄换内容**。
        因此策略是——在产出第一个文本 chunk 之前若失败，可以完整走降级链，
        把降级内容当作流式输出发出并标记 ``degradation``；一旦已经开始输出
        真实内容再失败，就只能发一个带错误标记的 final chunk，
        让前端展示「回答中断」而不是拼接一段不相干的兜底话术。
        """
        started = time.perf_counter()
        trace_id = self._resolve_trace_id(request)

        # 输入审核：流式同样要做，且在建立连接前做，成本最低
        user_text = self._extract_user_text(request.messages)
        input_check = self._moderator.moderate_input(user_text)
        if input_check.blocked:
            await self._log(
                request,
                trace_id=trace_id,
                provider="",
                model=request.model or "",
                status="blocked",
                error_type="input_moderation",
                latency_ms=int((time.perf_counter() - started) * 1000),
                moderation_blocked=True,
                is_stream=True,
            )
            yield StreamChunk(
                text=self._input_blocked_reply(input_check.categories),
                is_final=True,
                degradation=DegradationLevel.CANNED_RESPONSE,
            )
            return

        providers = get_configured_providers()
        primary_provider = providers[0] if providers else ProviderName.OPENAI
        budget = await check_token_budget(
            request.scenario,
            request.messages,
            user_id=request.user_id,
            provider=primary_provider,
        )
        if not budget.allowed:
            await self._log(
                request,
                trace_id=trace_id,
                provider="",
                model=request.model or "",
                status="budget_exceeded",
                error_type="token_budget",
                latency_ms=int((time.perf_counter() - started) * 1000),
                budget_exceeded=True,
                is_stream=True,
            )
            yield StreamChunk(
                text=(
                    "AI 服务今日调用额度已用完，暂时无法回答。"
                    "你可以直接预约挂号由医生解答；如有紧急症状请立即就医。"
                ),
                is_final=True,
                degradation=DegradationLevel.CANNED_RESPONSE,
            )
            return

        use_fallback_model = (
            budget.action == "downgrade_model" or budget.exceeded_request_budget
        )

        if not providers:
            content, level, _ = await self._run_degradation_chain(
                request, error_context="no_provider_configured"
            )
            yield StreamChunk(text=content, is_final=True, degradation=level)
            await release_reserved_tokens(
                budget.estimated_tokens, user_id=request.user_id
            )
            return

        last_error_type = "unknown"
        total_retries = 0
        requested_model = request.model or get_adapter(primary_provider).resolve_model(
            request.scenario, use_fallback=use_fallback_model
        )

        for provider in providers:
            adapter = get_adapter(provider)
            emitted_any = False
            accumulated: list[str] = []
            final_usage: TokenUsage | None = None
            actual_model = requested_model
            first_token_ms: int | None = None

            try:
                raw_stream = adapter.stream(
                    request.messages,
                    scenario=request.scenario,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    json_mode=request.json_mode,
                    use_fallback_model=use_fallback_model,
                    extra=request.extra,
                )
                async for chunk in stream_with_timeouts(raw_stream):
                    if chunk.text:
                        emitted_any = True
                        accumulated.append(chunk.text)
                        yield chunk
                    if chunk.is_final:
                        final_usage = chunk.usage
                        actual_model = chunk.model or actual_model
                        first_token_ms = chunk.first_token_ms
                # 正常结束
                break

            except (LLMRetryableError, LLMNonRetryableError) as exc:
                last_error_type = exc.error_type
                total_retries += 1
                logger.warning(
                    "流式调用供应商 %s 失败（%s），已输出=%s",
                    provider.value,
                    exc.error_type,
                    emitted_any,
                )
                if emitted_any:
                    # 已经开始输出真实内容，不能中途换成兜底话术——
                    # 那会让患者看到一段前后矛盾的回答。如实告知中断。
                    await self._log(
                        request,
                        trace_id=trace_id,
                        provider=provider.value,
                        model=actual_model,
                        requested_model=requested_model,
                        usage=self._estimated_usage(request.messages, accumulated),
                        status="failed",
                        error_type=exc.error_type,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        first_token_ms=first_token_ms,
                        retry_count=total_retries,
                        is_stream=True,
                    )
                    yield StreamChunk(
                        text="",
                        is_final=True,
                        error="回答生成中断，请重试或改用预约挂号咨询医生",
                    )
                    await release_reserved_tokens(
                        budget.estimated_tokens, user_id=request.user_id
                    )
                    return
                # 还没输出任何内容，可以完整降级或换供应商
                await self._log(
                    request,
                    trace_id=trace_id,
                    provider=provider.value,
                    model=requested_model,
                    requested_model=requested_model,
                    status="failed",
                    error_type=exc.error_type,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    retry_count=total_retries,
                    is_stream=True,
                )
                continue
            except Exception as exc:  # noqa: BLE001
                last_error_type = type(exc).__name__
                logger.exception("流式调用供应商 %s 出现未预期异常", provider.value)
                if emitted_any:
                    yield StreamChunk(
                        text="",
                        is_final=True,
                        error="回答生成中断，请重试",
                    )
                    return
                continue

        else:
            # for 循环正常结束（所有供应商都失败），走降级链
            content, level, _ = await self._run_degradation_chain(
                request, error_context=f"all_providers_failed:{last_error_type}"
            )
            yield StreamChunk(text=content, is_final=True, degradation=level)
            await self._log(
                request,
                trace_id=trace_id,
                provider="none",
                model=requested_model,
                requested_model=requested_model,
                status="degraded",
                error_type=last_error_type,
                latency_ms=int((time.perf_counter() - started) * 1000),
                retry_count=total_retries,
                degradation_level=level.value,
                is_stream=True,
            )
            await release_reserved_tokens(
                budget.estimated_tokens, user_id=request.user_id
            )
            return

        # ---------- 流式成功收尾 ----------
        full_text = "".join(accumulated)
        usage = final_usage or self._estimated_usage(request.messages, accumulated)
        # 流式输出无法在 yield 之后再做整体审核拦截（内容已经发出去了），
        # 因此这里只做**事后记录**：审核结果落库供监控统计，不阻断输出。
        # 真正的前置防护靠 prompt 约束 + 输入审核 + 确定性降级。
        output_check = self._moderator.moderate_output(full_text, scenario=request.scenario)
        if output_check.appended_notice:
            yield StreamChunk(text=f"\n\n{output_check.appended_notice}")

        cost = calculate_cost(usage, scenario=request.scenario)
        await self._log(
            request,
            trace_id=trace_id,
            provider=provider.value,
            model=actual_model,
            requested_model=requested_model,
            usage=usage,
            status="degraded" if output_check.blocked else "success",
            latency_ms=int((time.perf_counter() - started) * 1000),
            first_token_ms=first_token_ms,
            cost_usd=cost,
            retry_count=total_retries,
            degradation_level=DegradationLevel.PRIMARY.value,
            moderation_blocked=output_check.blocked,
            is_stream=True,
        )
        await settle_actual_tokens(
            budget.estimated_tokens, usage.total_tokens, user_id=request.user_id
        )
        yield StreamChunk(text="", usage=usage, is_final=True)

    # ==================== 结构化输出 ====================

    async def structured_chat(
        self,
        request: LLMRequest,
        schema: type[T],
    ) -> tuple[T | None, LLMResponse]:
        """要求 LLM 返回符合 schema 的 JSON。

        用 JSON mode + Pydantic 校验，而非 function calling：
        DashScope 对复杂嵌套 schema 的 function calling 支持不稳定，
        而 JSON mode 两个供应商都稳定支持，且校验失败时可以把
        Pydantic 的具体错误回喂给模型重试，比 function calling 的
        parse error 更可控。

        校验失败重解析一次；仍失败则返回 ``(None, response)``，
        由调用方决定降级策略——分诊拿不到结构化结果时应退回规则分诊，
        而不是抛异常中断挂号流程。
        """
        json_request = LLMRequest(
            scenario=request.scenario,
            messages=request.messages,
            model=request.model,
            temperature=0.0,  # 结构化输出必须低温，减少格式漂移
            max_tokens=request.max_tokens,
            json_mode=True,
            stream=False,
            trace_id=request.trace_id,
            user_id=request.user_id,
            workflow_node=request.workflow_node,
            ai_task_id=request.ai_task_id,
            deterministic_builder=request.deterministic_builder,
            extra=request.extra,
        )

        response = await self.chat(json_request)
        parsed, error = self._parse_structured(response.content, schema)
        if parsed is not None:
            response.parsed = parsed
            return parsed, response

        # 第一次解析失败：把具体错误回喂，要求重新输出
        logger.warning(
            "结构化输出校验失败，重试一次：%s（scenario=%s）",
            error,
            request.scenario.value,
        )
        retry_messages = [
            *json_request.messages,
            ChatMessage(role="assistant", content=response.content),
            ChatMessage(
                role="user",
                content=(
                    f"上面的输出不符合要求的 JSON 格式，错误：{error}\n"
                    f"请严格按以下 schema 重新输出，只输出 JSON，不要任何解释文字：\n"
                    f"{self._schema_hint(schema)}"
                ),
            ),
        ]
        retry_request = LLMRequest(
            scenario=request.scenario,
            messages=retry_messages,
            model=request.model,
            temperature=0.0,
            max_tokens=request.max_tokens,
            json_mode=True,
            stream=False,
            trace_id=response.trace_id,
            user_id=request.user_id,
            workflow_node=request.workflow_node,
            ai_task_id=request.ai_task_id,
            deterministic_builder=request.deterministic_builder,
            extra=request.extra,
        )
        retry_response = await self.chat(retry_request)
        parsed, error = self._parse_structured(retry_response.content, schema)
        if parsed is not None:
            retry_response.parsed = parsed
            return parsed, retry_response

        logger.error(
            "结构化输出两次解析均失败（scenario=%s）：%s",
            request.scenario.value,
            error,
        )
        retry_response.status = "structured_output_invalid"
        retry_response.error_type = "structured_output_invalid"
        return None, retry_response

    # ==================== 向量化 ====================

    async def embed(
        self,
        texts: list[str],
        *,
        trace_id: str | None = None,
        user_id: Any | None = None,
        ai_task_id: Any | None = None,
    ) -> tuple[list[list[float]], TokenUsage]:
        """批量向量化。逐供应商降级，全部失败才抛异常。

        与非流式 chat 不同，embedding 失败**不能降级**——
        返回错误的向量会污染索引且难以发现，因此必须显式失败，
        让 Celery 任务重试或标记为 failed。
        """
        if not texts:
            return [], TokenUsage()

        # embedding 多由 Celery 任务发起，调用方通常不传 trace_id；这里回退读
        # contextvar，让检索入库的向量调用也能并回触发它的那次业务请求。
        # 不在 embed 内新造 ID：Celery 侧没有中间件，凭空造一个只会产生
        # 一条和任何请求都对不上的孤立链路，宁可为空。
        trace_id = trace_id or get_trace_id()

        providers = get_configured_providers()
        if not providers:
            raise LLMNonRetryableError("没有已配置的 LLM 供应商，无法向量化")

        last_error: BaseException | None = None
        for provider in providers:
            adapter = get_adapter(provider)
            started = time.perf_counter()
            try:
                outcome = await execute_with_retry(
                    lambda a=adapter: a.embed(texts),
                    timeout=settings.llm_read_timeout,
                )
                vectors, usage, latency_ms = outcome.value  # type: ignore[misc]
                cost = calculate_cost(usage, scenario=LLMScenario.EMBEDDING)
                await self._logger.log(
                    record_llm_call(
                        scenario=LLMScenario.EMBEDDING.value,
                        provider=provider.value,
                        model=adapter.embedding_model,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        latency_ms=latency_ms,
                        status="success",
                        trace_id=trace_id,
                        cost_usd=cost,
                        user_id=user_id,
                        ai_task_id=ai_task_id,
                    )
                )
                return vectors, usage
            except (LLMRetryableError, LLMNonRetryableError) as exc:
                last_error = exc
                await self._logger.log(
                    record_llm_call(
                        scenario=LLMScenario.EMBEDDING.value,
                        provider=provider.value,
                        model=adapter.embedding_model,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        status="failed",
                        error_type=exc.error_type,
                        trace_id=trace_id,
                        user_id=user_id,
                        ai_task_id=ai_task_id,
                    )
                )
                continue

        raise LLMNonRetryableError(
            "所有供应商的向量化调用均失败", cause=last_error
        )

    # ==================== 内部工具 ====================

    async def _degrade(
        self,
        request: LLMRequest,
        *,
        trace_id: str,
        started: float,
        error_context: str,
        budget_estimated: int,
        retry_count: int = 0,
    ) -> LLMResponse:
        """执行降级链并返回响应。"""
        content, level, errors = await self._run_degradation_chain(
            request, error_context=error_context
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        # 降级意味着预扣的 token 没有真正消耗（除语义缓存外都不调 LLM），
        # 需要回滚，否则用户日配额会被失败请求白白吃掉。
        if budget_estimated > 0 and level != DegradationLevel.SEMANTIC_CACHE:
            await release_reserved_tokens(budget_estimated, user_id=request.user_id)

        await self._log(
            request,
            trace_id=trace_id,
            provider="none",
            model=request.model or "",
            requested_model=request.model,
            status="degraded",
            error_type=error_context,
            latency_ms=latency_ms,
            retry_count=retry_count,
            degradation_level=level.value,
        )
        if errors:
            logger.warning("降级链各级错误：%s", errors)

        return LLMResponse(
            content=content,
            usage=TokenUsage(),
            provider="none",
            model=request.model or "",
            requested_model=request.model,
            latency_ms=latency_ms,
            status="degraded",
            error_type=error_context,
            retry_count=retry_count,
            degradation_level=level,
            trace_id=trace_id,
        )

    async def _run_degradation_chain(
        self,
        request: LLMRequest,
        *,
        error_context: str,
    ) -> tuple[str, DegradationLevel, list[str]]:
        """构造并执行降级链。

        语义缓存这一级当前留空（阶段 2 接入 Redis 向量缓存）；
        确定性构造器由调用方通过 ``request.deterministic_builder`` 注入，
        因为只有业务层知道「这个问题能不能用结构化字段直接回答」。
        """
        chain = DegradationChain(
            semantic_cache_lookup=None,
            deterministic_builder=request.deterministic_builder,
            canned_text=get_canned_text(request.scenario.value),
        )
        return await chain.run(error_context=error_context)

    async def _log(
        self,
        request: LLMRequest,
        *,
        trace_id: str,
        provider: str,
        model: str,
        status: str,
        latency_ms: int,
        usage: TokenUsage | None = None,
        requested_model: str | None = None,
        cost_usd: float | None = None,
        first_token_ms: int | None = None,
        error_type: str | None = None,
        retry_count: int = 0,
        degradation_level: int | None = None,
        moderation_blocked: bool = False,
        budget_exceeded: bool = False,
        is_stream: bool = False,
    ) -> None:
        """写一条调用日志。失败不影响主链路。"""
        try:
            await self._logger.log(
                record_llm_call(
                    scenario=request.scenario.value,
                    provider=provider,
                    model=model,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=latency_ms,
                    status=status,
                    trace_id=trace_id,
                    requested_model=requested_model,
                    cost_usd=cost_usd,
                    first_token_ms=first_token_ms,
                    error_type=error_type,
                    retry_count=retry_count,
                    degradation_level=degradation_level,
                    moderation_blocked=moderation_blocked,
                    budget_exceeded=budget_exceeded,
                    is_stream=is_stream,
                    user_id=request.user_id,
                    workflow_node=request.workflow_node,
                    ai_task_id=request.ai_task_id,
                )
            )
        except Exception:  # noqa: BLE001 - 日志绝不能拖垮业务
            logger.exception("写入 LLM 调用日志失败")

    def _estimated_usage(
        self,
        messages: list[ChatMessage],
        accumulated: list[str],
    ) -> TokenUsage:
        """供应商没返回 usage 时的本地估算兜底。

        标记 ``is_estimated=True``，看板上要区分展示——
        估算值不能用于对账，只能用于趋势观察。
        """
        prompt_tokens = sum(
            estimate_tokens(message.content) + 4 for message in messages
        )
        completion_tokens = estimate_tokens("".join(accumulated))
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            is_estimated=True,
        )

    def _extract_user_text(self, messages: list[ChatMessage]) -> str:
        """取最后一条用户消息作为审核对象。

        只审最后一条：历史消息在之前的请求里已经审过，重复审核
        既浪费也会因为累积的长文本触发长度上限而误拦。
        """
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return ""

    def _input_blocked_reply(self, categories: list[str]) -> str:
        """输入被拦截时的回复。

        区分注入攻击与超长输入：前者不能透露检测细节（否则等于告诉攻击者
        怎么绕过），后者要给出可操作的提示。
        """
        if "too_long" in categories:
            return (
                "你的输入内容过长，请精简后重试。"
                "如果需要咨询复杂病情，建议直接预约挂号由医生详细评估。"
            )
        return (
            "抱歉，我无法处理这个请求。\n\n"
            "我是医疗健康助手，可以为你解答健康问题、提供智能分诊和预约挂号服务。"
            "如有紧急症状，请立即拨打 120 或前往最近的急诊科。"
        )

    def _parse_structured(
        self, content: str, schema: type[T]
    ) -> tuple[T | None, str | None]:
        """解析并校验结构化输出。返回 (对象, 错误信息)。"""
        if not content or not content.strip():
            return None, "模型返回空内容"

        text = content.strip()
        # 容错：模型有时会包一层 ```json ... ``` 代码块，
        # 即使开了 JSON mode 也可能出现。剥掉再解析。
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        # 再容错：截取首个 { 到末个 } 之间的内容，
        # 处理模型在 JSON 前后附加解释文字的情况
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                text = text[start : end + 1]

        try:
            return schema.model_validate_json(text), None
        except ValidationError as exc:
            # 只取前两条错误，避免把冗长的 Pydantic 报错全塞回 prompt
            errors = exc.errors()[:2]
            summary = "; ".join(
                f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in errors
            )
            return None, summary
        except ValueError as exc:
            return None, f"JSON 解析失败：{exc}"

    def _schema_hint(self, schema: type[BaseModel]) -> str:
        """生成给模型看的 schema 提示。截断到合理长度避免吃掉预算。"""
        try:
            hint = str(schema.model_json_schema())
        except Exception:  # noqa: BLE001
            return schema.__name__
        return hint[:1500]


# 全局单例。服务无状态，共享一个实例即可复用内部缓冲与连接。
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """取 LLM 服务单例。也可作为 FastAPI 依赖注入使用。"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


__all__ = ["LLMService", "get_llm_service"]
