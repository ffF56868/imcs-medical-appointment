from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.database import get_session
from app.memory import (
    SHORT_TERM_MESSAGE_LIMIT,
    build_memory_context,
    delete_user_memory,
    get_long_term_memories,
    get_memory_state,
    get_recent_messages,
)
from app.legacy.models import User, UserMemory
from app.routers.auth import get_current_user
from app.legacy.schemas import (
    MemoryDeleteResponse,
    MemoryMessageRead,
    MemoryOverview,
    UserMemoryRead,
)


router = APIRouter(
    prefix="/memory",
    tags=["memory"],
    dependencies=[Depends(get_current_user)],
)


def serialize_memory(memory: UserMemory) -> UserMemoryRead:
    return UserMemoryRead(
        id=int(memory.id),
        memory_key=memory.memory_key,
        content=memory.content,
        importance=memory.importance,
        source_conversation_id=memory.source_conversation_id,
        access_count=memory.access_count,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
    )


@router.get("", response_model=MemoryOverview)
def get_memory_overview(
    conversation_id: str = "default",
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    context_state = get_memory_state(session, int(current_user.id), conversation_id)
    recent_messages = get_recent_messages(session, current_user, conversation_id)
    long_term_memories = get_long_term_memories(session, current_user)
    # The endpoint is read-only from the user's point of view; access timestamps
    # are persisted so later diagnostics can show whether memories are useful.
    session.commit()
    return MemoryOverview(
        conversation_id=conversation_id,
        short_term_summary=context_state.summary if context_state else "",
        summarized_message_count=(
            context_state.summarized_message_count if context_state else 0
        ),
        short_term_message_count=len(recent_messages),
        recent_message_limit=SHORT_TERM_MESSAGE_LIMIT,
        recent_messages=[
            MemoryMessageRead(
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            )
            for message in recent_messages
        ],
        long_term_memories=[serialize_memory(item) for item in long_term_memories],
    )


@router.delete(
    "/{memory_id}",
    response_model=MemoryDeleteResponse,
    status_code=status.HTTP_200_OK,
)
def remove_memory(
    memory_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if not delete_user_memory(session, current_user, memory_id):
        raise HTTPException(status_code=404, detail="记忆不存在或无权删除")
    session.commit()
    return MemoryDeleteResponse(deleted=True, memory_id=memory_id)
