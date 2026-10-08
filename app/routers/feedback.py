from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from app.database import get_session
from app.legacy.models import AnswerFeedback, ChatMessage, User
from app.routers.auth import get_current_user, require_admin
from app.legacy.schemas import (
    FeedbackCreate,
    FeedbackDetail,
    FeedbackGroup,
    FeedbackImprovementSuggestions,
    FeedbackRead,
    FeedbackSummary,
)


router = APIRouter(
    prefix="/feedback",
    tags=["feedback"],
    dependencies=[Depends(get_current_user)],
)


def get_question_for_answer(
    session: Session,
    assistant_message: ChatMessage,
) -> str | None:
    user_message = session.exec(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == assistant_message.conversation_id)
        .where(ChatMessage.role == "user")
        .where(ChatMessage.id < assistant_message.id)
        .order_by(ChatMessage.id.desc())
        .limit(1)
    ).first()
    return user_message.content if user_message else None


@router.post("", response_model=FeedbackRead)
def create_or_update_feedback(
    feedback_data: FeedbackCreate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    message = session.get(ChatMessage, feedback_data.assistant_message_id)
    if (
        message is None
        or message.role != "assistant"
        or (
            not current_user.is_admin
            and message.user_id != current_user.id
        )
    ):
        raise HTTPException(status_code=404, detail="助手回答记录不存在")

    feedback = session.exec(
        select(AnswerFeedback).where(
            AnswerFeedback.assistant_message_id
            == feedback_data.assistant_message_id
        )
    ).first()
    if feedback is None:
        feedback = AnswerFeedback(**feedback_data.model_dump())
    else:
        feedback.helpful = feedback_data.helpful
        feedback.reason = feedback_data.reason

    session.add(feedback)
    session.commit()
    session.refresh(feedback)
    return feedback


@router.get("/summary", response_model=FeedbackSummary)
def get_feedback_summary(
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    feedback_items = session.exec(select(AnswerFeedback)).all()
    total_count = len(feedback_items)
    helpful_count = sum(item.helpful for item in feedback_items)
    return FeedbackSummary(
        total_count=total_count,
        helpful_count=helpful_count,
        not_helpful_count=total_count - helpful_count,
        helpful_rate=(round(helpful_count / total_count, 3) if total_count else None),
    )


@router.get("/recent", response_model=list[FeedbackDetail])
def list_recent_feedback(
    helpful: bool | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    statement = select(AnswerFeedback).order_by(AnswerFeedback.created_at.desc())
    if helpful is not None:
        statement = statement.where(AnswerFeedback.helpful == helpful)

    feedback_items = session.exec(statement.limit(limit)).all()
    details: list[FeedbackDetail] = []
    for feedback in feedback_items:
        assistant_message = session.get(
            ChatMessage,
            feedback.assistant_message_id,
        )
        if assistant_message is None:
            continue

        details.append(
            FeedbackDetail(
                id=feedback.id,
                assistant_message_id=feedback.assistant_message_id,
                conversation_id=assistant_message.conversation_id,
                question=get_question_for_answer(session, assistant_message),
                answer=assistant_message.content,
                helpful=feedback.helpful,
                reason=feedback.reason,
                created_at=feedback.created_at,
            )
        )

    return details


@router.get(
    "/improvement-suggestions",
    response_model=FeedbackImprovementSuggestions,
)
def get_improvement_suggestions(
    limit: int = Query(default=5, ge=1, le=20),
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    feedback_items = session.exec(
        select(AnswerFeedback)
        .where(AnswerFeedback.helpful == False)  # noqa: E712
        .order_by(AnswerFeedback.created_at.desc())
    ).all()

    reason_counts: Counter[str] = Counter()
    question_counts: Counter[str] = Counter()
    for feedback in feedback_items:
        reason_counts[feedback.reason or "未填写反馈原因"] += 1
        assistant_message = session.get(
            ChatMessage,
            feedback.assistant_message_id,
        )
        if assistant_message is None:
            continue
        question = get_question_for_answer(session, assistant_message)
        if question:
            question_counts[question] += 1

    actions: list[str] = []
    all_reasons = " ".join(reason_counts).lower()
    if not feedback_items:
        actions.append("暂无‘没帮助’反馈；先收集真实问答反馈，再决定优化方向。")
    else:
        if any(word in all_reasons for word in ("资料", "没有", "不足", "没答")):
            actions.append("补充高频问题涉及的权威健康资料，然后重建知识库。")
        if any(word in all_reasons for word in ("错误", "不准", "不准确", "引用")):
            actions.append("检查对应回答的参考资料与引用片段，修正不准确内容。")
        if any(word in all_reasons for word in ("太长", "简短", "看不懂", "格式")):
            actions.append("调整回答提示词，优化篇幅、结构或表达方式。")
        if not actions:
            actions.append("查看‘没帮助’明细，结合原因补资料或调整检索和回答规则。")

    return FeedbackImprovementSuggestions(
        not_helpful_count=len(feedback_items),
        common_reasons=[
            FeedbackGroup(text=text, count=count)
            for text, count in reason_counts.most_common(limit)
        ],
        repeated_questions=[
            FeedbackGroup(text=text, count=count)
            for text, count in question_counts.most_common(limit)
        ],
        recommended_actions=actions,
    )
