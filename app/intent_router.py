"""Intent routing for the three-layer memory architecture.

Routes every user input into one of three channels before any retrieval or
generation happens:

  * ``chat``     — greetings, thanks, farewells, etc.  Handled by deterministic
    rule-based replies; no LLM call, no retrieval.
  * ``followup`` — the user is continuing the current conversation (pronouns,
    omitted subjects, chained questions).  RAG retrieval runs, but the retrieval
    query is enriched with short-term memory context for coreference resolution.
  * ``rag``      — the user is asking an objective health question.  Standard
    hybrid retrieval pipeline.

All decisions are rule-based (keywords + regex) so the router adds zero latency
and no extra model calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from sqlmodel import Session

from app.memory import (
    get_long_term_memories,
    get_recent_messages,
    format_long_term_memory,
)
from app.legacy.models import User


class IntentChannel(str, Enum):
    CHAT = "chat"
    FOLLOWUP = "followup"
    RAG = "rag"


@dataclass(frozen=True)
class RouteDecision:
    channel: IntentChannel
    reason: str
    resolved_query: str
    page_number: int | None = None  # 用户明确指定页码时填充


# ── 1. Chat intent ────────────────────────────────────────────────────────────

GREETING_PATTERNS = (
    re.compile(r"^(?:你好|您好|hi|hello|hey|嗨|哈喽|早上好|下午好|晚上好|早安|晚安)[\s!！。.？?，,、]*$", re.I),
    re.compile(r"^(?:在吗|在不在|有人吗|有人在吗)[\s?？。.！!]*$"),
)

FAREWELL_PATTERNS = (
    re.compile(r"^(?:再见|拜拜|bye|byebye|晚安|先走了|告辞|下次见|回头见)[\s!！。.？?]*$", re.I),
)

THANKS_PATTERNS = (
    re.compile(
        r"^(?:谢谢|感谢|多谢|thank\s*(?:you|s)?|thanks|3[qQ]|thx|辛苦了|麻烦了|太感谢"
        r"|非常感谢|多谢了|感恩)[\s!！。.？?]*$",
        re.I,
    ),
)

ACK_PATTERNS = (
    re.compile(r"^(?:好的|嗯|哦|ok|OK|嗯嗯|了解|知道了|明白了|收到|收到啦|好嘞|行|可以)[\s!！。.？?]*$", re.I),
)

CHATTER_PATTERNS = (
    re.compile(r"^(?:你是谁|你叫什么|你是什么|介绍一下你自己)[\s?？。.！!]*$"),
    re.compile(r"^(?:你能做什么|你会什么|你有什么功能|怎么用你)[\s?？。.！!]*$"),
)

CHAT_RULES: list[tuple[tuple, str]] = [
    (GREETING_PATTERNS, "你好！我是医疗健康助手，可以帮你查询病症、药物和健康资料。有什么想了解的？"),
    (FAREWELL_PATTERNS, "再见！如有健康疑问随时回来问我。"),
    (THANKS_PATTERNS, "不客气！还有其他健康问题可以随时问我。"),
    (ACK_PATTERNS, "好的，有需要随时继续问。"),
    (CHATTER_PATTERNS, '我是医疗健康助手，可以帮你查询病症、药物和健康知识。你可以直接输入健康问题，例如"布洛芬有什么作用？"'),
]


def _match_chat_intent(question: str) -> str | None:
    """Return a canned reply when the question is pure chitchat, else None."""
    normalized = question.strip()
    for patterns, reply in CHAT_RULES:
        if any(p.search(normalized) for p in patterns):
            return reply
    return None


# ── 2. Follow-up / context continuation ───────────────────────────────────────

PRONOUN_PATTERNS = (
    re.compile(r"(?:它|这个|那个|这药|这病|这种|这种药|这种病|上面|刚才|之前)"),
    re.compile(r"^(?:还有|另外|此外|同时|而且|并且|但是|不过|可是|然后|那|那么|所以|因此)"),
    re.compile(r"^(?:为什么|什么原因|怎么回事|什么意思|真的吗|确定吗|对吗|是吗|靠谱吗|准确吗)"),
)

OMITTED_SUBJECT_MARKERS = (
    "副作用",
    "怎么用",
    "怎么吃",
    "用法",
    "用量",
    "禁忌",
    "注意事项",
    "适应症",
    "禁忌症",
    "怎么用",
    "多久",
    "疗程",
    "严重吗",
    "要紧吗",
    "能治好吗",
    "会传染吗",
)

CHAIN_QUESTION_STARTERS = (
    "那",
    "那么",
    "所以",
    "也就是说",
    "换句话说",
)

CONTEXT_WINDOW_MESSAGES = 4


def _looks_like_followup(question: str, session: Session, user: User, conversation_id: str) -> bool:
    """Heuristic check: is the user continuing the previous conversation turn?"""
    normalized = question.strip()

    # 先排除页码引用（页码引用走 RAG 通道，不算 followup）
    if _extract_page_number(normalized) is not None:
        return False

    # Explicit pronoun reference
    if any(p.search(normalized) for p in PRONOUN_PATTERNS):
        return True

    # Omitted subject — the question is about properties of something already discussed
    if any(marker in normalized for marker in OMITTED_SUBJECT_MARKERS):
        recent = get_recent_messages(session, user, conversation_id, limit=CONTEXT_WINDOW_MESSAGES)
        if len(recent) >= 2:
            return True

    # Chain question starters
    if any(normalized.startswith(starter) for starter in CHAIN_QUESTION_STARTERS):
        recent = get_recent_messages(session, user, conversation_id, limit=CONTEXT_WINDOW_MESSAGES)
        if len(recent) >= 2:
            return True

    # Very short question after a substantive exchange — likely a follow-up
    if len(normalized) <= 12 and not _looks_like_standalone_question(normalized):
        recent = get_recent_messages(session, user, conversation_id, limit=CONTEXT_WINDOW_MESSAGES)
        if len(recent) >= 2:
            return True

    return False


_STANDALONE_QUESTION_MARKERS = (
    "什么是",
    "是什么",
    "有哪些",
    "怎么办",
    "怎么处理",
    "怎么治疗",
    "什么原因",
    "什么症状",
    "什么药",
    "哪种药",
    "哪个",
    "哪份",
    "布洛芬",
    "阿莫西林",
    "对乙酰氨基酚",
)


def _looks_like_standalone_question(question: str) -> bool:
    return any(marker in question for marker in _STANDALONE_QUESTION_MARKERS)


# ── 2.5 Page reference detection ─────────────────────────────────────────────

_PAGE_PATTERNS = (
    re.compile(r"第\s*(\d+)\s*页"),
    re.compile(r"(\d+)\s*页"),
    re.compile(r"page\s*(\d+)", re.IGNORECASE),
)


def _extract_page_number(question: str) -> int | None:
    """Extract page number from question if user explicitly references a page.
    
    Examples:
        "第50页讲了什么" -> 50
        "50页的内容" -> 50
        "page 50" -> 50
    """
    normalized = question.strip()
    for pattern in _PAGE_PATTERNS:
        match = pattern.search(normalized)
        if match:
            try:
                page_num = int(match.group(1))
                if 1 <= page_num <= 10000:  # 合理范围
                    return page_num
            except (ValueError, IndexError):
                continue
    return None


# ── 3. Public API ─────────────────────────────────────────────────────────────


def route_intent(
    question: str,
    session: Session,
    user: User,
    conversation_id: str,
) -> RouteDecision:
    """Classify the user input and return the channel + resolved query.

    Priority order:
      1. Chat  → deterministic reply, no LLM, no retrieval
      2. Follow-up → RAG retrieval with enriched context
      3. RAG → standard hybrid retrieval
    """
    # 1. Chat channel
    chat_reply = _match_chat_intent(question)
    if chat_reply is not None:
        return RouteDecision(
            channel=IntentChannel.CHAT,
            reason="chat-greeting",
            resolved_query=chat_reply,
        )

    # 2. Follow-up channel
    if _looks_like_followup(question, session, user, conversation_id):
        recent = get_recent_messages(session, user, conversation_id, limit=CONTEXT_WINDOW_MESSAGES)
        context_parts = []
        for msg in recent:
            role_label = "用户" if msg.role == "user" else "助手"
            context_parts.append(f"{role_label}：{msg.content}")
        history_context = "\n".join(context_parts)
        resolved = f"[延续上下文]\n{history_context}\n当前追问：{question}"
        return RouteDecision(
            channel=IntentChannel.FOLLOWUP,
            reason="followup-context",
            resolved_query=resolved,
        )

    # 2.5 Page reference - extract page number and pass to RAG with filter
    page_number = _extract_page_number(question)
    if page_number is not None:
        return RouteDecision(
            channel=IntentChannel.RAG,
            reason="rag-page-reference",
            resolved_query=question,
            page_number=page_number,
        )

    # 3. RAG channel (default)
    return RouteDecision(
        channel=IntentChannel.RAG,
        reason="rag-knowledge-query",
        resolved_query=question,
    )


def build_chat_reply(channel_decision: RouteDecision) -> str:
    """Return the deterministic reply for a chat-channel decision."""
    if channel_decision.channel != IntentChannel.CHAT:
        return ""
    return channel_decision.resolved_query


def build_followup_context(
    session: Session,
    user: User,
    conversation_id: str,
    question: str,
) -> tuple[str, str]:
    """Build enriched context for a follow-up question.

    Returns (prompt_history, retrieval_query) where:
      - prompt_history includes recent conversation turns for the LLM
      - retrieval_query combines recent context + current question for retrieval
    """
    recent = get_recent_messages(session, user, conversation_id, limit=CONTEXT_WINDOW_MESSAGES)
    history_parts = []
    retrieval_parts = []
    for msg in recent:
        role_label = "用户" if msg.role == "user" else "助手"
        history_parts.append(f"{role_label}：{msg.content}")
        retrieval_parts.append(f"{role_label}：{msg.content}")

    # Also inject long-term memory for personalization
    long_term = get_long_term_memories(session, user)
    if long_term:
        history_parts.append(
            "\n用户长期记忆（只用于称呼和回答风格，不作为医疗事实）：\n"
            + format_long_term_memory(long_term)
        )

    prompt_history = "\n".join(history_parts) if history_parts else "暂无历史对话"
    retrieval_query = "\n".join(retrieval_parts) + f"\n当前问题：{question}"
    return prompt_history, retrieval_query
