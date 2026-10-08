from sqlmodel import Session, select

from app.memory import (
    build_memory_context,
    remember_explicit_user_facts,
    update_short_term_memory,
)
from app.legacy.models import ChatMessage, ConversationMemoryState, User, UserMemory


def make_user() -> User:
    return User(id=1, account="memory@example.com", is_admin=False)


def test_explicit_user_fact_is_saved_once_and_duplicate_is_not_added(test_engine):
    user = make_user()
    with Session(test_engine) as session:
        remember_explicit_user_facts(
            session,
            user,
            "我叫小明，我今年25岁，请记住我以后用中文回答。",
            "memory-conversation",
        )
        remember_explicit_user_facts(
            session,
            user,
            "我叫小明，我今年25岁，请记住我以后用中文回答。",
            "memory-conversation",
        )
        session.commit()

        memories = session.exec(
            select(UserMemory).where(UserMemory.user_id == user.id)
        ).all()

    assert {memory.memory_key for memory in memories} == {
        "user_name",
        "user_age",
        "preferred_language",
    }
    assert len(memories) == 3


def test_new_explicit_value_deactivates_old_value_for_the_same_key(test_engine):
    user = make_user()
    with Session(test_engine) as session:
        remember_explicit_user_facts(session, user, "我叫小明。", "first")
        session.commit()
        remember_explicit_user_facts(session, user, "我更正一下，我叫小红。", "second")
        session.commit()

        memories = session.exec(
            select(UserMemory)
            .where(UserMemory.user_id == user.id)
            .order_by(UserMemory.id)
        ).all()

    assert [memory.content for memory in memories] == [
        "用户希望被称呼为小明",
        "用户希望被称呼为小红",
    ]
    assert memories[0].active is False
    assert memories[1].active is True


def test_short_term_memory_summarizes_old_messages_and_keeps_recent_window(
    test_engine,
):
    user = make_user()
    with Session(test_engine) as session:
        for index in range(10):
            session.add(
                ChatMessage(
                    user_id=user.id,
                    conversation_id="long-conversation",
                    role="user" if index % 2 == 0 else "assistant",
                    content=f"第 {index + 1} 条对话",
                )
            )
        session.commit()

        state = update_short_term_memory(session, user, "long-conversation")
        prompt_context, retrieval_context = build_memory_context(
            session,
            user,
            "long-conversation",
        )

        assert state is not None
        assert state.summarized_message_count == 4
        assert "第 1 条对话" in state.summary
        assert "第 5 条对话" in retrieval_context
        assert "第 10 条对话" in retrieval_context
        assert "短期会话摘要" in prompt_context

        saved_state = session.exec(
            select(ConversationMemoryState).where(
                ConversationMemoryState.conversation_id == "long-conversation"
            )
        ).one()

    assert saved_state.summary


def test_memory_api_returns_current_session_and_can_delete_long_term_memory(
    client,
    test_engine,
):
    user = make_user()
    with Session(test_engine) as session:
        remember_explicit_user_facts(
            session,
            user,
            "我叫小明，我今年25岁。",
            "api-memory-conversation",
        )
        session.add(
            ChatMessage(
                user_id=user.id,
                conversation_id="api-memory-conversation",
                role="user",
                content="我想了解感冒的通用护理建议。",
            )
        )
        session.commit()
        memory_id = session.exec(
            select(UserMemory).where(UserMemory.memory_key == "user_name")
        ).one().id

    response = client.get(
        "/memory",
        params={"conversation_id": "api-memory-conversation"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["short_term_message_count"] == 1
    assert data["recent_messages"][0]["content"].startswith("我想了解感冒")
    assert {item["memory_key"] for item in data["long_term_memories"]} == {
        "user_name",
        "user_age",
    }

    delete_response = client.delete(f"/memory/{memory_id}")

    assert delete_response.status_code == 200
    assert delete_response.json() == {"deleted": True, "memory_id": memory_id}
    remaining = client.get("/memory").json()["long_term_memories"]
    assert all(item["memory_key"] != "user_name" for item in remaining)


def test_memory_api_requires_login(auth_client):
    response = auth_client.get("/memory")

    assert response.status_code == 401
