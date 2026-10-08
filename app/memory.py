"""Layered conversation memory for the medical-health assistant.

Three-layer memory architecture:
  - Working memory: ephemeral task state during a single request (retrieval results,
    intermediate steps). Cleared after response generation.
  - Short-term memory: bounded recent window plus a persisted summary for the current
    conversation. Compressed when exceeding thresholds.
  - Long-term memory: explicit user preferences and facts (allergies, history, etc.)
    persisted across sessions. Conservative extraction via regex patterns.
"""

from __future__ import annotations

import re
import json
import math
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Session, select

from app.legacy.models import ChatMessage, ConversationMemoryState, User, UserMemory


# ── Working Memory ────────────────────────────────────────────────────────────


@dataclass
class WorkingMemory:
    """Ephemeral state for the current request.

    This is NOT persisted. It lives only for the duration of a single request
    and is cleared after the response is generated. Used to track:
      - Retrieved documents and their metadata
      - Intermediate processing steps (e.g., intent classification result)
      - Contextual data needed across pipeline stages
    """

    # Retrieval results
    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)
    retrieval_method: str = ""
    retrieval_count: int = 0

    # Intent routing
    intent_channel: str = ""  # "chat" | "followup" | "rag"
    intent_reason: str = ""

    # Processing metadata
    processing_path: str = ""
    source_filter: str = "all"
    knowledge_type: str = "all"

    # User context (injected from short/long-term memory)
    user_context: str = ""

    def clear(self) -> None:
        """Reset all working memory fields."""
        self.retrieved_documents.clear()
        self.retrieval_method = ""
        self.retrieval_count = 0
        self.intent_channel = ""
        self.intent_reason = ""
        self.processing_path = ""
        self.source_filter = "all"
        self.knowledge_type = "all"
        self.user_context = ""


SHORT_TERM_MESSAGE_LIMIT = 6
SUMMARY_TRIGGER_MESSAGE_COUNT = 8
MAX_SUMMARY_CHARS = 1200
LONG_TERM_IMPORTANCE_THRESHOLD = 0.75
MAX_LONG_TERM_ITEMS = 20

EXPLICIT_MEMORY_PATTERNS = (
    ("preferred_language", re.compile(r"(?:请|以后|回答请|默认请).{0,8}(?:用|使用)(中文|汉语|英文|英语)"), 0.82),
    ("preferred_style", re.compile(r"(?:我喜欢|我偏好|回答请|以后请).{0,12}(简洁|详细|简单|通俗|专业)"), 0.82),
    ("user_name", re.compile(r"(?:我叫|我的名字是|称呼我为)\s*([\u4e00-\u9fffA-Za-z0-9_]{1,30})"), 0.9),
    ("user_age", re.compile(r"(?:我今年|我现在|本人今年)\s*(\d{1,3})\s*岁"), 0.86),
    ("allergy", re.compile(r"(?:我对|本人对)\s*([^，。；;]{1,40})\s*过敏"), 0.98),
    ("medical_history", re.compile(r"(?:我有|本人有)\s*([^，。；;]{1,40})\s*病史"), 0.98),
    ("long_term_medication", re.compile(r"(?:我长期服用|我长期使用|我正在长期服用)\s*([^，。；;]{1,40})"), 0.98),
)

CORRECTION_MARKERS = ("更正", "改成", "改为", "不是", "以后叫", "现在叫")


def _now() -> datetime:
    return datetime.now(UTC)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value.casefold())


def _clip(value: str, length: int) -> str:
    return value.strip()[:length]


def _as_utc(value: datetime | None) -> datetime:
    if value is None:
        return _now()
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _format_messages(messages: list[ChatMessage]) -> str:
    return "\n".join(
        f"{'用户' if message.role == 'user' else '助手'}：{message.content}"
        for message in messages
    )


def get_memory_state(
    session: Session,
    user_id: int,
    conversation_id: str,
) -> ConversationMemoryState | None:
    return session.exec(
        select(ConversationMemoryState).where(
            ConversationMemoryState.user_id == user_id,
            ConversationMemoryState.conversation_id == conversation_id,
        )
    ).first()


def get_recent_messages(
    session: Session,
    user: User,
    conversation_id: str,
    limit: int = SHORT_TERM_MESSAGE_LIMIT,
) -> list[ChatMessage]:
    statement = (
        select(ChatMessage)
        .where(
            ChatMessage.user_id == user.id,
            ChatMessage.conversation_id == conversation_id,
        )
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(limit)
    )
    messages = session.exec(statement).all()
    messages.reverse()
    return messages


def _build_summary(existing: str, older_messages: list[ChatMessage]) -> str:
    """Create a deterministic compact summary without another model call.

    The summary keeps user requests and assistant conclusions, but is bounded so
    it cannot grow forever. The source messages remain available for audit/export.
    """
    lines = [line for line in (existing.strip(), _format_messages(older_messages)) if line]
    return _clip("\n".join(lines), MAX_SUMMARY_CHARS)


def _memory_embedding(text: str) -> list[float]:
    """Use the existing OpenAI embedding client when it is configured.

    Memory failure must never make medical Q&A fail. A missing key, proxy or
    embedding service therefore returns an empty vector and the caller uses the
    deterministic lexical fallback instead.
    """
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return []
    try:
        from app.vector_store import get_embeddings

        return [float(value) for value in get_embeddings().embed_query(text)]
    except Exception:
        return []


def _load_embedding(value: str) -> list[float]:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    try:
        return [float(item) for item in parsed]
    except (TypeError, ValueError):
        return []


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _memory_similarity(
    content: str,
    embedding: list[float],
    existing: UserMemory,
) -> float:
    existing_embedding = _load_embedding(existing.embedding_json)
    if embedding and existing_embedding:
        return _cosine_similarity(embedding, existing_embedding)
    return _similarity(existing.content, content)


def effective_importance(memory: UserMemory, now: datetime | None = None) -> float:
    """Apply gentle time decay while rewarding memories used in conversations."""
    current = _as_utc(now)
    age_days = max(0.0, (current - _as_utc(memory.updated_at)).total_seconds() / 86400)
    decay = 0.98 ** age_days
    access_bonus = min(0.1, memory.access_count * 0.01)
    return memory.importance * decay + access_bonus


def build_memory_context(
    session: Session,
    user: User,
    conversation_id: str,
) -> tuple[str, str]:
    """Return prompt context and the retrieval query context.

    The prompt receives summary + recent turns. Retrieval receives the same
    bounded context plus the new question, preserving follow-up understanding
    without sending the full conversation to the embedding search.
    """
    recent = get_recent_messages(session, user, conversation_id)
    state = get_memory_state(session, int(user.id), conversation_id)
    summary = state.summary if state is not None else ""
    recent_text = _format_messages(recent)
    retrieval_parts = []
    prompt_parts = []
    if summary:
        retrieval_parts.append(f"会话摘要：{summary}")
        prompt_parts.append(f"短期会话摘要：{summary}")
    if recent_text:
        retrieval_parts.append(f"最近对话：{recent_text}")
        prompt_parts.append(f"最近对话：{recent_text}")

    long_term = get_long_term_memories(session, user)
    if long_term:
        prompt_parts.append(
            "已确认的用户长期记忆（只用于称呼和回答风格，不作为医疗事实）：\n"
            + format_long_term_memory(long_term)
        )

    prompt_context = "\n\n".join(prompt_parts) or "暂无历史对话"
    retrieval_context = "\n\n".join(retrieval_parts) or "暂无历史对话"
    return prompt_context, retrieval_context


def finalize_conversation_memory(
    session: Session,
    user: User,
    conversation_id: str,
) -> None:
    """Update the summary after a completed user/assistant turn."""
    update_short_term_memory(session, user, conversation_id)


def update_short_term_memory(
    session: Session,
    user: User,
    conversation_id: str,
) -> ConversationMemoryState | None:
    """Compress old turns after the bounded recent window is exceeded."""
    all_messages = session.exec(
        select(ChatMessage)
        .where(
            ChatMessage.user_id == user.id,
            ChatMessage.conversation_id == conversation_id,
        )
        .order_by(ChatMessage.created_at, ChatMessage.id)
    ).all()
    if len(all_messages) <= SUMMARY_TRIGGER_MESSAGE_COUNT:
        return get_memory_state(session, int(user.id), conversation_id)

    older_messages = all_messages[:-SHORT_TERM_MESSAGE_LIMIT]
    state = get_memory_state(session, int(user.id), conversation_id)
    if state is None:
        state = ConversationMemoryState(
            user_id=int(user.id),
            conversation_id=conversation_id,
        )
        session.add(state)
    if len(older_messages) <= state.summarized_message_count:
        return state
    new_messages = older_messages[state.summarized_message_count:]
    state.summary = _build_summary(state.summary, new_messages)
    state.summarized_message_count = len(older_messages)
    state.updated_at = _now()
    session.add(state)
    return state


def _extract_explicit_memories(question: str) -> list[tuple[str, str, float]]:
    memories: list[tuple[str, str, float]] = []
    normalized_question = question.strip()
    for memory_key, pattern, importance in EXPLICIT_MEMORY_PATTERNS:
        match = pattern.search(normalized_question)
        if match is None:
            continue
        value = _clip(match.group(1), 80)
        if memory_key == "preferred_language":
            content = f"用户偏好使用{value}回答"
        elif memory_key == "preferred_style":
            content = f"用户偏好{value}的回答风格"
        elif memory_key == "user_name":
            content = f"用户希望被称呼为{value}"
        elif memory_key == "user_age":
            content = f"用户今年{value}岁"
        elif memory_key == "allergy":
            content = f"用户明确表示对{value}过敏"
        elif memory_key == "medical_history":
            content = f"用户明确表示有{value}病史"
        else:
            content = f"用户明确表示长期服用或使用{value}"
        memories.append((memory_key, content, importance))
    return memories


def _similarity(left: str, right: str) -> float:
    """Small deterministic token overlap used before persistence.

    It is not an embedding score. Its purpose is duplicate protection when the
    same explicit fact is stated more than once.
    """
    left_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", _normalize(left)))
    right_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", _normalize(right)))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def remember_explicit_user_facts(
    session: Session,
    user: User,
    question: str,
    conversation_id: str,
) -> list[UserMemory]:
    """Upsert explicit high-importance facts and reject ambiguous conflicts."""
    extracted = _extract_explicit_memories(question)
    if not extracted:
        return []

    memories: list[UserMemory] = []
    now = _now()
    for memory_key, content, importance in extracted:
        if importance < LONG_TERM_IMPORTANCE_THRESHOLD:
            continue
        embedding = _memory_embedding(content)
        existing = session.exec(
            select(UserMemory).where(
                UserMemory.user_id == user.id,
                UserMemory.memory_key == memory_key,
                UserMemory.active == True,  # noqa: E712
            )
        ).all()
        # The key already narrows the comparison. A high threshold prevents
        # short Chinese names such as "小明" and "小红" from collapsing into
        # one memory merely because most wrapper text is identical.
        exact = next(
            (
                item
                for item in existing
                if _memory_similarity(content, embedding, item) >= 0.9
            ),
            None,
        )
        if exact is not None:
            exact.last_accessed_at = now
            exact.access_count += 1
            exact.updated_at = now
            session.add(exact)
            memories.append(exact)
            continue

        # A conflicting fact is not silently accepted. The user must explicitly
        # mark the sentence as a correction before the old value is replaced.
        if existing and not any(marker in question for marker in CORRECTION_MARKERS):
            continue

        # A new explicit value for the same key supersedes the old value. This
        # prevents two active names or language preferences from being injected.
        for item in existing:
            item.active = False
            item.updated_at = now
            session.add(item)
        memory = UserMemory(
            user_id=int(user.id),
            memory_key=memory_key,
            content=content,
            importance=importance,
            embedding_json=json.dumps(embedding),
            source_conversation_id=conversation_id,
            last_accessed_at=now,
            access_count=1,
        )
        session.add(memory)
        memories.append(memory)

    active = session.exec(
        select(UserMemory)
        .where(UserMemory.user_id == user.id, UserMemory.active == True)  # noqa: E712
        .order_by(UserMemory.updated_at.desc())
    ).all()
    for stale in active[MAX_LONG_TERM_ITEMS:]:
        stale.active = False
        session.add(stale)
    return memories


def get_long_term_memories(
    session: Session,
    user: User,
    limit: int = MAX_LONG_TERM_ITEMS,
) -> list[UserMemory]:
    memories = session.exec(
        select(UserMemory).where(
            UserMemory.user_id == user.id,
            UserMemory.active == True,  # noqa: E712
        )
    ).all()
    memories.sort(
        key=lambda memory: effective_importance(memory),
        reverse=True,
    )
    memories = memories[:limit]
    now = _now()
    for memory in memories:
        memory.last_accessed_at = now
        memory.access_count += 1
        session.add(memory)
    return memories


def format_long_term_memory(memories: list[UserMemory]) -> str:
    if not memories:
        return "暂无已保存的长期记忆"
    return "\n".join(f"- {memory.content}" for memory in memories)


def delete_user_memory(session: Session, user: User, memory_id: int) -> bool:
    memory = session.exec(
        select(UserMemory).where(
            UserMemory.id == memory_id,
            UserMemory.user_id == user.id,
            UserMemory.active == True,  # noqa: E712
        )
    ).first()
    if memory is None:
        return False
    memory.active = False
    memory.updated_at = _now()
    session.add(memory)
    return True
