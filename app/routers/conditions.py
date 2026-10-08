from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from app.database import get_session
from app.cache import invalidate_knowledge_status_cache
from app.legacy.models import Condition, User
from app.routers.auth import get_current_user, require_admin
from app.legacy.schemas import ConditionCreate, ConditionRead


router = APIRouter(
    prefix="/conditions",
    tags=["conditions"],
    dependencies=[Depends(get_current_user)],
)


@router.post("", response_model=ConditionRead, status_code=201)
def create_condition(
    condition_data: ConditionCreate,
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    existing_condition = session.exec(
        select(Condition).where(Condition.name == condition_data.name)
    ).first()

    if existing_condition is not None:
        raise HTTPException(status_code=409, detail="病症已经存在")

    condition = Condition.model_validate(condition_data)
    condition.updated_at = datetime.now(UTC)
    session.add(condition)
    session.commit()
    invalidate_knowledge_status_cache()
    session.refresh(condition)
    return condition


@router.get("", response_model=list[ConditionRead])
def list_conditions(
    keyword: str | None = Query(default=None, max_length=100),
    session: Session = Depends(get_session),
):
    statement = select(Condition).order_by(Condition.id)

    if keyword and keyword.strip():
        statement = statement.where(Condition.name.contains(keyword.strip()))

    return session.exec(statement).all()


@router.get("/{condition_id}", response_model=ConditionRead)
def get_condition(
    condition_id: int,
    session: Session = Depends(get_session),
):
    condition = session.get(Condition, condition_id)

    if condition is None:
        raise HTTPException(status_code=404, detail="病症不存在")

    return condition


@router.put("/{condition_id}", response_model=ConditionRead)
def update_condition(
    condition_id: int,
    condition_data: ConditionCreate,
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    condition = session.get(Condition, condition_id)

    if condition is None:
        raise HTTPException(status_code=404, detail="病症不存在")

    duplicate = session.exec(
        select(Condition).where(
            Condition.name == condition_data.name,
            Condition.id != condition_id,
        )
    ).first()

    if duplicate is not None:
        raise HTTPException(status_code=409, detail="病症名称已经存在")

    condition.name = condition_data.name
    condition.symptoms = condition_data.symptoms
    condition.treatment = condition_data.treatment
    condition.source = condition_data.source
    condition.source_url = condition_data.source_url
    condition.source_tier = condition_data.source_tier
    condition.updated_at = datetime.now(UTC)
    session.add(condition)
    session.commit()
    invalidate_knowledge_status_cache()
    session.refresh(condition)
    return condition


@router.delete("/{condition_id}", status_code=204)
def delete_condition(
    condition_id: int,
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    condition = session.get(Condition, condition_id)

    if condition is None:
        raise HTTPException(status_code=404, detail="病症不存在")

    session.delete(condition)
    session.commit()
    invalidate_knowledge_status_cache()
