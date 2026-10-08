import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlmodel import Session, select

from app.database import get_session
from app.legacy.models import AnswerFeedback, ChatMessage, User
from app.routers.auth import get_current_user
from app.legacy.schemas import ChatMessageRead, ConversationSummary


router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
    dependencies=[Depends(get_current_user)],
)


def parse_response_metadata(message: ChatMessage) -> dict:
    """Old conversation rows did not have answer metadata, so treat them as empty."""
    try:
        metadata = json.loads(message.response_metadata_json or "{}")
    except (TypeError, ValueError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def to_message_read(message: ChatMessage) -> ChatMessageRead:
    return ChatMessageRead(
        id=message.id,
        conversation_id=message.conversation_id,
        role=message.role,
        content=message.content,
        response_metadata=parse_response_metadata(message),
        created_at=message.created_at,
    )


def get_conversation_messages(
    session: Session,
    conversation_id: str,
    current_user: User,
) -> list[ChatMessage]:
    statement = select(ChatMessage).where(
        ChatMessage.conversation_id == conversation_id
    )
    if not current_user.is_admin:
        statement = statement.where(ChatMessage.user_id == current_user.id)
    return session.exec(statement.order_by(ChatMessage.created_at, ChatMessage.id)).all()


def format_references_for_export(references: object) -> list[str]:
    if not isinstance(references, list) or not references:
        return []

    lines = ["### 参考资料"]
    for index, reference in enumerate(references, start=1):
        if not isinstance(reference, dict):
            continue
        name = str(reference.get("name") or "未命名资料")
        knowledge_type = str(reference.get("type") or "未知类型")
        source = str(reference.get("source") or "未标注来源")
        score = reference.get("relevance_score")
        score_text = f"，相关度 {score}" if isinstance(score, (int, float)) else ""
        lines.append(f"{index}. {name}（{knowledge_type}，来源：{source}{score_text}）")
        excerpt = str(reference.get("excerpt") or "").strip()
        if excerpt:
            lines.append(f"   摘要：{excerpt}")
    return lines if len(lines) > 1 else []


def build_conversation_markdown(
    conversation_id: str,
    messages: list[ChatMessage],
) -> str:
    lines = [
        "# 医疗健康助手对话记录",
        "",
        f"导出时间：{datetime.now(UTC).isoformat()}",
        f"会话 ID：{conversation_id}",
        "",
    ]
    for message in messages:
        heading = "用户" if message.role == "user" else "医疗健康助手"
        lines.extend([f"## {heading}", "", message.content, ""])
        if message.role != "assistant":
            continue
        metadata = parse_response_metadata(message)
        if metadata.get("processing_path"):
            lines.extend(
                [
                    "### 本次回答过程",
                    "",
                    f"- 处理路径：{metadata['processing_path']}",
                    f"- 检索范围：{metadata.get('retrieval_scope', 'all')}",
                    f"- 可信度筛选：{metadata.get('source_filter', 'all')}",
                    f"- 命中资料数：{metadata.get('retrieved_count', 0)}",
                    "",
                ]
            )
        reference_lines = format_references_for_export(metadata.get("references"))
        if reference_lines:
            lines.extend([*reference_lines, ""])

    lines.extend(
        [
            "---",
            "内容仅供健康信息参考，不代替医生诊断或处方。",
            "",
        ]
    )
    return "\n".join(lines)


@router.get("", response_model=list[ConversationSummary])
def list_conversations(
    limit: int = Query(default=30, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    statement = select(ChatMessage)
    if not current_user.is_admin:
        statement = statement.where(ChatMessage.user_id == current_user.id)
    messages = session.exec(
        statement.order_by(ChatMessage.created_at, ChatMessage.id)
    ).all()
    grouped_messages: dict[str, list[ChatMessage]] = {}
    for message in messages:
        grouped_messages.setdefault(message.conversation_id, []).append(message)

    summaries: list[ConversationSummary] = []
    for conversation_id, conversation_messages in grouped_messages.items():
        first_user_message = next(
            (
                message.content
                for message in conversation_messages
                if message.role == "user"
            ),
            conversation_messages[0].content,
        )
        latest_message = conversation_messages[-1]
        summaries.append(
            ConversationSummary(
                conversation_id=conversation_id,
                preview=first_user_message[:80],
                message_count=len(conversation_messages),
                updated_at=latest_message.created_at,
            )
        )

    return sorted(
        summaries,
        key=lambda summary: summary.updated_at,
        reverse=True,
    )[:limit]


@router.get("/{conversation_id}/messages", response_model=list[ChatMessageRead])
def list_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    return [
        to_message_read(message)
        for message in get_conversation_messages(
            session, conversation_id, current_user
        )
    ]


@router.get("/{conversation_id}/export")
def export_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    messages = get_conversation_messages(session, conversation_id, current_user)
    if not messages:
        raise HTTPException(status_code=404, detail="会话记录不存在")

    return Response(
        content=build_conversation_markdown(conversation_id, messages),
        media_type="text/markdown",
        headers={
            "Content-Disposition": "attachment; filename=medical-health-conversation.md"
        },
    )


@router.delete("/{conversation_id}/messages", status_code=status.HTTP_204_NO_CONTENT)
def clear_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    statement = select(ChatMessage).where(
        ChatMessage.conversation_id == conversation_id
    )
    if not current_user.is_admin:
        statement = statement.where(ChatMessage.user_id == current_user.id)
    messages = session.exec(statement).all()

    if not messages:
        raise HTTPException(status_code=404, detail="会话记录不存在")

    message_ids = [message.id for message in messages if message.id is not None]
    feedback_items = session.exec(
        select(AnswerFeedback).where(
            AnswerFeedback.assistant_message_id.in_(message_ids)
        )
    ).all()
    for feedback in feedback_items:
        session.delete(feedback)

    for message in messages:
        session.delete(message)

    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
