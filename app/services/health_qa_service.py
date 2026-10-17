"""健康咨询问答服务：检索 → 拼 prompt → 流式生成（阶段 7 批次 0）。

与 ``eval_answer.answer_for_evaluation`` 是同源不同用途的两条管线：
那条只服务评测（刻意不对外暴露），这条服务真实用户（SSE 流式）。
两者都复用生产同款 ``RAGRetriever`` 与 ``llm_service``，这样评测出来的
faithfulness 才真实反映线上质量。

**关键设计：本模块只产出「事件」，不碰 HTTP。**
生成器 yield 的是领域事件（QaEvent），由路由层翻译成 SSE 帧。理由：
1. 事件流可以被单元测试直接断言（消费 async generator 即可），
   不必起 httpx client 解析 `event:`/`data:` 文本；
2. 将来若要加 WebSocket 或轮询通道，翻译层换掉即可，生成逻辑不动。

**急症与心理危机在检索之前短路**（``app/services/safety/qa_guard``）：
命中即返回固定话术，不检索、不调 LLM。医疗场景下「输出可预测」
比「回答丰富」重要——兜底话术经过审核，模型生成不保证。

**零检索结果不降级为通用建议**：直接说明知识库未收录并引导就诊。
让模型在没有资料时凭常识发挥，是医疗问答里最危险的一类错误。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.trace import get_trace_id, new_trace_id
from app.services.llm.service import get_llm_service
from app.services.llm.types import ChatMessage, LLMRequest, LLMScenario
from app.services.observability import (
    build_step,
    finish_workflow_trace,
    start_workflow_trace,
)
from app.services.rag.retriever import RAGRetriever
from app.services.safety.qa_guard import ZERO_RESULT_RESPONSE, guard_question

logger = logging.getLogger(__name__)

WORKFLOW_NAME = "health_qa"

# 系统提示。三条硬边界写死在这里而不是靠模型自觉：
# 1. 只依据给定资料，资料没覆盖就明说没有（faithfulness 的基础）
# 2. 不做确诊、不给处方剂量——这两件事必须由医生面诊决定
# 3. 急重表现引导就医，不提供家庭处理方案
SYSTEM_PROMPT = (
    "你是医院健康咨询助手，只依据【资料】部分回答用户问题。"
    "资料未覆盖的内容必须明确说明知识库中未收录，禁止推测或编造。"
    "你可以解释疾病常识、就诊科室建议、检查与用药的一般注意事项，"
    "但不得做出确诊结论，不得给出具体处方剂量或调整用药的方案，"
    "遇到这类请求应说明需由医生面诊决定。"
    "引用资料时在相应内容后标注来源，格式为（来源：资料标题）。"
    "若用户描述急重表现，直接建议立即就医或拨打 120，不要给家庭处理建议。"
    "回答用简体中文，条理清晰，长度控制在 400 字以内。"
)

# 上下文字符预算。取 top-5 切块，单块约 500-800 字，6000 字符足够容纳；
# 再大只推高 token 成本，召回窗口本身由检索层控制，与此处截断无关。
CONTEXT_CHAR_BUDGET = 6000
MAX_CONTEXT_ITEMS = 5

# 多轮上下文只取最后 3 轮。健康咨询多是单轮问答，长历史既推高 token
# 成本，也容易让模型被早期话题带偏；3 轮足够支撑「它有什么副作用」这类追问。
MAX_HISTORY_TURNS = 3

# 免责声明。随 done 事件下发，前端必须展示——这是医疗内容合规的硬要求，
# 不能依赖前端"记得加"，所以由服务端统一下发。
DISCLAIMER = "以上内容仅供健康信息参考，不能替代医生的诊断、处方或面诊建议。如有不适请及时就医。"

# 事件类型。与旧系统 ask.py 的四事件协议保持同名，降低前端与运维的认知成本。
EVENT_METADATA = "metadata"
EVENT_TOKEN = "token"
EVENT_DONE = "done"
EVENT_ERROR = "error"


@dataclass(slots=True)
class QaEvent:
    """一个领域事件。``kind`` 取 EVENT_* 常量，``data`` 是可 JSON 序列化的 dict。"""

    kind: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QaOutcome:
    """一次问答的完整结果，供路由层在流结束后落库（BackgroundTask 用）。

    刻意与事件流分开：事件是给前端看的（增量、精简），
    outcome 是给审计用的（全量、含用量与耗时）。
    """

    trace_id: str
    question: str
    answer_text: str = ""
    ref_sources: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    first_token_ms: int | None = None
    latency_ms: int = 0
    model: str | None = None
    degradation_level: int | None = None
    urgent_keywords: list[str] = field(default_factory=list)
    is_urgent_short_circuit: bool = False
    is_zero_result: bool = False
    client_disconnected: bool = False
    error_message: str | None = None


async def _start_trace(trace_id: str, *, user_id: Any = None) -> int | None:
    """start_workflow_trace 的调用侧兜底。

    observability 内部虽然吞了异常，但「可观测旁路绝不反噬业务」这条契约
    不应依赖被调用方的自觉——桩件或底层库换掉后行为就失控。调用侧同样兜底，
    返回 None 时 finish 会退化为按 trace_id 更新（空操作），链路不断。
    """
    try:
        return await start_workflow_trace(
            trace_id, WORKFLOW_NAME, user_id=user_id, node_count=3
        )
    except Exception:  # noqa: BLE001 - 旁路失败只降级，不影响问答
        logger.warning("创建工作流追踪失败 trace=%s", trace_id, exc_info=True)
        return None


async def _finish_trace(trace_ref: Any, **kwargs: Any) -> None:
    """finish_workflow_trace 的调用侧兜底，理由同 ``_start_trace``。"""
    try:
        await finish_workflow_trace(trace_ref, **kwargs)
    except Exception:  # noqa: BLE001
        logger.warning("收尾工作流追踪失败 trace_ref=%s", trace_ref, exc_info=True)


def _project_references(results: list[Any]) -> list[dict[str, Any]]:
    """把检索结果投影成前端展示引用所需的最小字段集。

    不含切块正文：正文可能很长，前端只需要「出自哪份资料的哪一页」，
    点开原文应按 chunk_id 回查，不该在每条引用里重复携带全文。
    """
    projected: list[dict[str, Any]] = []
    for item in results[:MAX_CONTEXT_ITEMS]:
        projected.append(
            {
                "chunk_id": str(item.chunk_id),
                "title": item.title,
                "source": item.source,
                "doc_type": item.doc_type,
                "source_tier": item.source_tier,
                "page_number": item.page_number,
                "rrf_score": round(float(item.rrf_score or 0.0), 6),
            }
        )
    return projected


def _trim_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """只保留最后 ``MAX_HISTORY_TURNS`` 轮，并过滤掉非法角色。

    过滤而不是报错：客户端多传了 system 或脏数据时，静默丢弃比返回 422
    更合适——历史只是辅助上下文，缺了顶多答得糙一点，不该让整次咨询失败。
    """
    if not history:
        return []
    valid = [
        t for t in history
        if t.get("role") in ("user", "assistant") and t.get("content")
    ]
    return valid[-(MAX_HISTORY_TURNS * 2):]


def _build_messages(history: list[dict[str, str]], prompt: str) -> list[ChatMessage]:
    """组装 LLM 消息：system + 裁剪后的历史 + 本轮问题。

    system 永远由服务端下发且放在最前，客户端无法覆盖——这是
    「不得给出具体处方剂量」等硬边界不被绕过的保证。
    """
    messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
    for turn in _trim_history(history):
        messages.append(ChatMessage(role=turn["role"], content=turn["content"]))
    messages.append(ChatMessage(role="user", content=prompt))
    return messages


def _build_context(results: list[Any]) -> str:
    """拼检索上下文，按 rrf_score 降序（检索层已排序）累加至预算上限。

    超预算时从低分端截断而不是截断单条内容：半截资料喂给模型
    容易产生「资料说了但没说全」的误引，宁可少给一条完整的。
    """
    pieces: list[str] = []
    used = 0
    for item in results[:MAX_CONTEXT_ITEMS]:
        piece = f"《{item.title}》（{item.doc_type}）：{item.content}"
        if used + len(piece) > CONTEXT_CHAR_BUDGET and pieces:
            break
        pieces.append(piece)
        used += len(piece)
    return "\n\n".join(pieces)


async def stream_health_qa(
    db: AsyncSession,
    question: str,
    *,
    user_id: uuid.UUID | Any | None = None,
    trace_id: str | None = None,
    history: list[dict[str, str]] | None = None,
    retriever: RAGRetriever | None = None,
    outcome: QaOutcome | None = None,
) -> AsyncIterator[QaEvent]:
    """生成一次流式健康问答的事件序列。

    Args:
        db: 请求级会话。**只用于检索**；落库不走它（见下方说明）。
        history: 最近几轮对话（``[{"role","content"}]``），只取最后
            ``MAX_HISTORY_TURNS`` 轮。用于「它有什么副作用」这类省略主语的追问。
        outcome: 调用方传入的可变结果对象，本函数边生成边填充。
            这样路由层在流结束（或客户端断开）后能拿到完整数据交给
            BackgroundTask 落库，而不必从事件流里反向拼装。

    **为什么不在生成器内写库**：SSE 生成器的生命周期绑在 HTTP 响应上，
    客户端一断开 Starlette 就取消生成器，此时请求 session 可能已进入回收，
    在它上面 commit 会撞 "Session is closed"，或更糟——留下未提交事务。
    旧系统 ask.py 就是在 generator 内 save_message，这是它的架构包袱之一。
    本实现改为：生成器只填 outcome，落库由路由层的 BackgroundTask 用
    独立短事务完成。

    异常处理：LLM 或检索抛错时发 error 事件并结束，不把异常抛给 Starlette
    ——流已经开始（HTTP 200 + headers 已发），此时抛异常无法再改状态码，
    只会让客户端看到一个被截断的响应且不知道为什么。
    """
    resolved_trace = trace_id or get_trace_id() or new_trace_id()
    result = outcome if outcome is not None else QaOutcome(
        trace_id=resolved_trace, question=question
    )
    result.trace_id = resolved_trace
    started = time.perf_counter()
    steps: list[dict[str, Any]] = []
    # guard / retrieve / generate
    trace_row_id = await _start_trace(resolved_trace, user_id=user_id)

    def _elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    try:
        # ---------- 1. 安全守卫：急症与心理危机短路 ----------
        step_start = time.perf_counter()
        verdict = guard_question(question)
        if verdict.blocked:
            result.is_urgent_short_circuit = True
            result.urgent_keywords = list(verdict.matched)
            result.answer_text = verdict.response
            result.latency_ms = _elapsed_ms()
            steps.append(
                build_step(
                    "guard",
                    "rule",
                    latency_ms=int((time.perf_counter() - step_start) * 1000),
                    input_snapshot=question[:500],
                    output_snapshot={"kind": verdict.kind, "matched": verdict.matched},
                )
            )
            yield QaEvent(
                EVENT_METADATA,
                {
                    "trace_id": resolved_trace,
                    "model": None,
                    "references": [],
                    "retrieved_count": 0,
                    "is_zero_result": False,
                    # 前端据此把回答渲染成警示样式而不是普通气泡。
                    # 急症与心理危机的视觉处理不同（后者要避免刺激性配色）。
                    "guard": verdict.kind,
                },
            )
            yield QaEvent(EVENT_TOKEN, {"text": verdict.response})
            result.degradation_level = 4  # CANNED_RESPONSE：零 LLM 调用
            yield QaEvent(
                EVENT_DONE,
                {
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "first_token_ms": None,
                    "latency_ms": result.latency_ms,
                    "degradation": "CANNED_RESPONSE",
                    "disclaimer": DISCLAIMER,
                },
            )
            await _finish_trace(
                trace_row_id if trace_row_id is not None else resolved_trace,
                status="completed",
                steps=steps,
                node_count=1,  # 短路：只跑了守卫这一个节点
                total_latency_ms=result.latency_ms,
                total_tokens=0,
                branch_taken=verdict.kind,
                final_output={"short_circuit": True, "guard": verdict.kind},
            )
            return

        steps.append(
            build_step(
                "guard",
                "rule",
                latency_ms=int((time.perf_counter() - step_start) * 1000),
                input_snapshot=question[:500],
                output_snapshot={"blocked": False},
            )
        )

        # ---------- 2. 检索 ----------
        # 追问常省略主语（「它有什么副作用」），直接拿原句检索会召回不到。
        # 用最近一轮用户问题做确定性拼接补全，而不是调 LLM 改写：
        # 改写要多一次调用（时延与成本翻倍），而健康咨询的追问
        # 绝大多数只是承接上一句的主体，拼接就够用。
        trimmed = _trim_history(history)
        search_query = question
        if trimmed:
            last_user = next(
                (t["content"] for t in reversed(trimmed) if t.get("role") == "user"),
                None,
            )
            if last_user:
                search_query = f"{last_user} {question}"

        step_start = time.perf_counter()
        retriever = retriever or RAGRetriever()
        try:
            response = await retriever.retrieve(
                db,
                question,
                rewritten_query=search_query if search_query != question else None,
                trace_id=resolved_trace,
                user_id=user_id,
            )
        except Exception as exc:  # noqa: BLE001 - 检索失败不应让用户看到 500
            logger.warning("健康咨询检索失败 trace=%s", resolved_trace, exc_info=True)
            result.error_message = f"retrieval_failed: {type(exc).__name__}"
            yield QaEvent(
                EVENT_ERROR,
                {"code": "retrieval_no_result", "message": "知识检索暂时不可用，请稍后再试或前往医院就诊。"},
            )
            await _finish_trace(
                trace_row_id if trace_row_id is not None else resolved_trace,
                status="failed",
                steps=steps,
                total_latency_ms=_elapsed_ms(),
                error_message=result.error_message,
            )
            return

        results = list(response.results or [])
        references = _project_references(results)
        result.ref_sources = references
        result.is_zero_result = bool(response.is_zero_result or not results)
        steps.append(
            build_step(
                "retrieve",
                "rag",
                latency_ms=int((time.perf_counter() - step_start) * 1000),
                input_snapshot=question[:500],
                output_snapshot={
                    "retrieved": len(results),
                    "is_zero_result": result.is_zero_result,
                    "vector_hit_count": response.vector_hit_count,
                    "keyword_hit_count": response.keyword_hit_count,
                },
            )
        )

        yield QaEvent(
            EVENT_METADATA,
            {
                "trace_id": resolved_trace,
                "model": None,  # 真实模型名要等 final chunk，done 事件里给
                "references": references,
                "retrieved_count": len(results),
                "is_zero_result": result.is_zero_result,
                "guard": None,
            },
        )

        # ---------- 3. 零检索结果：不降级为通用建议 ----------
        if not results:
            result.answer_text = ZERO_RESULT_RESPONSE
            result.latency_ms = _elapsed_ms()
            result.degradation_level = 4
            yield QaEvent(EVENT_TOKEN, {"text": ZERO_RESULT_RESPONSE})
            yield QaEvent(
                EVENT_DONE,
                {
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "first_token_ms": None,
                    "latency_ms": result.latency_ms,
                    "degradation": "CANNED_RESPONSE",
                    "disclaimer": DISCLAIMER,
                },
            )
            await _finish_trace(
                trace_row_id if trace_row_id is not None else resolved_trace,
                status="completed",
                steps=steps,
                node_count=2,  # 守卫 + 检索，未进入生成
                total_latency_ms=result.latency_ms,
                total_tokens=0,
                branch_taken="zero_result",
                final_output={"zero_result": True},
            )
            return

        # ---------- 4. 流式生成 ----------
        prompt = (
            "【资料】\n" + _build_context(results) + f"\n\n【问题】{question}"
        )
        request = LLMRequest(
            scenario=LLMScenario.CHAT,
            messages=_build_messages(history, prompt),
            stream=True,
            trace_id=resolved_trace,
            user_id=user_id,
            workflow_node="health_qa",
        )

        llm = get_llm_service()
        gen_start = time.perf_counter()
        collected: list[str] = []
        degradation_name: str | None = None
        stream_broken = False
        async for chunk in llm.chat_stream(request):
            # 中途出错：chat_stream 不抛异常而是发带 error 的 final chunk，
            # 这里如实转成 error 事件，让前端展示「回答中断」而不是静默截断。
            if chunk.error:
                result.error_message = chunk.error
                stream_broken = True
                yield QaEvent(
                    EVENT_ERROR,
                    {"code": "llm_all_providers_failed", "message": chunk.error},
                )
                break
            if chunk.text:
                collected.append(chunk.text)
                yield QaEvent(EVENT_TOKEN, {"text": chunk.text})
            if chunk.is_final:
                usage = chunk.usage
                if usage is not None:
                    result.prompt_tokens = int(usage.prompt_tokens or 0)
                    result.completion_tokens = int(usage.completion_tokens or 0)
                result.model = chunk.model
                result.first_token_ms = chunk.first_token_ms
                if chunk.degradation is not None:
                    result.degradation_level = int(chunk.degradation.value)
                    degradation_name = chunk.degradation.name

        answer = "".join(collected)
        result.answer_text = answer
        result.latency_ms = _elapsed_ms()
        total_tokens = result.prompt_tokens + result.completion_tokens
        steps.append(
            build_step(
                "generate",
                "llm",
                latency_ms=int((time.perf_counter() - gen_start) * 1000),
                tokens=total_tokens,
                input_snapshot={"prompt_chars": len(prompt)},
                output_snapshot={
                    "answer_chars": len(answer),
                    "model": result.model,
                    "degradation": degradation_name,
                    "broken": stream_broken,
                },
                error=result.error_message,
            )
        )

        # 已发 error 就**不再发 done**：两个事件语义互斥，都发的话前端
        # 会把中断的回答当正常结束渲染，用户看到的是一段被悄悄截断的话。
        if stream_broken:
            await _finish_trace(
                trace_row_id if trace_row_id is not None else resolved_trace,
                status="failed",
                steps=steps,
                total_latency_ms=result.latency_ms,
                total_tokens=total_tokens,
                final_output={"answer_chars": len(answer), "broken": True},
                error_message=result.error_message,
            )
            return

        yield QaEvent(
            EVENT_DONE,
            {
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                },
                "first_token_ms": result.first_token_ms,
                "latency_ms": result.latency_ms,
                "model": result.model,
                "degradation": degradation_name,
                "disclaimer": DISCLAIMER,
            },
        )
        await _finish_trace(
            trace_row_id if trace_row_id is not None else resolved_trace,
            status="completed",
            steps=steps,
            total_latency_ms=result.latency_ms,
            total_tokens=total_tokens,
            final_output={
                "answer_chars": len(answer),
                "retrieved_count": len(results),
                "degradation": degradation_name,
            },
        )
    except Exception as exc:  # noqa: BLE001 - 生成器内兜底，不让异常穿透到 Starlette
        logger.exception("健康咨询流式生成异常 trace=%s", resolved_trace)
        result.error_message = f"{type(exc).__name__}: {exc}"[:500]
        result.latency_ms = _elapsed_ms()
        yield QaEvent(
            EVENT_ERROR,
            {"code": "internal_error", "message": "回答生成失败，请稍后再试。"},
        )
        await _finish_trace(
            trace_row_id if trace_row_id is not None else resolved_trace,
            status="failed",
            steps=steps,
            total_latency_ms=result.latency_ms,
            error_message=result.error_message,
        )


__all__ = [
    "DISCLAIMER",
    "EVENT_DONE",
    "EVENT_ERROR",
    "EVENT_METADATA",
    "EVENT_TOKEN",
    "QaEvent",
    "QaOutcome",
    "stream_health_qa",
]
