"""评测专用最小问答管线（阶段 6B RAGAS 的前置环节）。

**为什么要有这个模块**：旧系统的 RAGAS 评测把回答生成委托给
``app/routers/ask.py`` 的问答链路，而 IMCS 的定位是诊疗业务系统、
**没有在线患者问答端点**（旧 ask 属 legacy，不在演进路线上）。
RAGAS 四指标（faithfulness / answer relevancy / context precision /
context recall）需要三元组 (question, answer, contexts)——answer 必须有人生成。

所以这里放一条**只服务评测**的最小链路：检索 → 拼上下文 → LLM 生成。
它刻意不进 ``api/v1``（不对外提供聊天能力），但检索与 LLM 调用
**复用生产同款组件**（RAGRetriever / llm_service），这样评测出来的
faithfulness 才真实反映生产链路的质量——评测管线和生产链路若各走各的，
指标再漂亮也是自欺。

回答生成的 prompt 带医疗边界声明（不确诊、不替代医嘱），与生产
解读场景同一口径：生成越接近线上行为，RAGAS 分数越有迁移价值。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm.service import get_llm_service
from app.services.llm.types import ChatMessage, LLMRequest, LLMScenario
from app.services.rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)

# 评测问答的系统提示。要点：只依据给定资料、资料没有就明说没有
# （拒答题的 faithfulness 依赖这条——模型硬编资料外内容会同时拖低
# faithfulness 与正确拒答率）、带医疗免责边界。
EVAL_SYSTEM_PROMPT = (
    "你是医疗健康知识库问答助手。只依据【资料】部分回答，"
    "资料没有覆盖的内容必须明确说明知识库中未收录，不要推测或编造。"
    "回答仅供健康信息参考，不构成诊断、处方或紧急医疗建议；"
    "出现急重症状应立即就医或拨打 120。"
)
# 上下文字符预算。评测要跑整个题库，单条问题的上下文给到 4000 字符
# 已足够容纳 rag_result_count 条切块；再大只会推高 token 成本，
# 而召回窗口本身由检索层控制，与这里的截断无关。
CONTEXT_CHAR_BUDGET = 4000


@dataclass(slots=True)
class EvalAnswer:
    """一次评测问答的产出，直接喂给 RAGAS 的 SingleTurnSample。"""

    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    context_titles: list[str] = field(default_factory=list)
    retrieved_count: int = 0
    degraded: bool = False
    model: str | None = None


async def answer_for_evaluation(
    db: AsyncSession,
    question: str,
    *,
    retriever: RAGRetriever | None = None,
    trace_id: str | None = None,
) -> EvalAnswer:
    """检索 + 生成一次问答。失败时返回带 degraded 标记的空回答而不是抛错。

    为什么吞异常：一次 RAGAS 运行要跑几十题，第 7 题 LLM 抖动不应该
    让前面 6 题的评测作废。降级题会如实标记，汇总里能区分
    「答得差」与「根本没答上」。
    """
    retriever = retriever or RAGRetriever()
    try:
        response = await retriever.retrieve(db, question, trace_id=trace_id)
    except Exception:  # noqa: BLE001
        logger.warning("评测问答检索失败 question=%.40s", question, exc_info=True)
        return EvalAnswer(question=question, answer="", degraded=True)

    results = list(response.results or [])
    contexts: list[str] = []
    titles: list[str] = []
    used = 0
    for item in results:
        piece = f"《{item.title}》（{item.doc_type}）：{item.content}"
        if used + len(piece) > CONTEXT_CHAR_BUDGET and contexts:
            break
        contexts.append(piece)
        titles.append(item.title)
        used += len(piece)

    if not contexts:
        # 零证据不送 LLM：直接判定「知识库未收录」。省一次调用，
        # 也让拒答题拿到稳定的回答文本——这本身就是被测行为的一部分。
        return EvalAnswer(
            question=question,
            answer="知识库中未收录与该问题直接相关的资料，无法回答；如涉及急重症状请立即就医。",
            retrieved_count=0,
        )

    prompt = "【资料】\n" + "\n\n".join(contexts) + f"\n\n【问题】{question}"
    try:
        llm = get_llm_service()
        result = await llm.chat(
            LLMRequest(
                scenario=LLMScenario.CHAT,
                messages=[
                    ChatMessage(role="system", content=EVAL_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=prompt),
                ],
                trace_id=trace_id,
                workflow_node="eval_answer",
            )
        )
    except Exception:  # noqa: BLE001
        logger.warning("评测问答生成失败 question=%.40s", question, exc_info=True)
        return EvalAnswer(
            question=question,
            answer="",
            contexts=contexts,
            context_titles=titles,
            retrieved_count=len(results),
            degraded=True,
        )

    return EvalAnswer(
        question=question,
        answer=result.content or "",
        contexts=contexts,
        context_titles=titles,
        retrieved_count=len(results),
        degraded=bool(result.is_degraded),
        model=result.model,
    )


__all__ = ["EvalAnswer", "answer_for_evaluation"]
