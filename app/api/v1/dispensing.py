"""发药域路由：处方审核、发药与取药记录查询（阶段 5）。

============================ 权限映射（rbac_matrix 为准，未改动矩阵） ============================

+---------------------------------------------------+-------------------------------+--------+
| 端点                                              | require_permission              | 角色   |
+---------------------------------------------------+-------------------------------+--------+
| POST /dispensing/prescriptions/{id}/approve       | prescription:approve            | 药师   |
| POST /dispensing/prescriptions/{id}/dispense      | dispensing:create               | 药师   |
| GET  /dispensing/records                          | dispensing:read                 | 患者/医生
| GET  /dispensing/records/all                      | dispensing:read_all             | 药师   |
| GET  /dispensing/records/{id}                     | dispensing:read/read_all 任一   | 全部   |
+---------------------------------------------------+-------------------------------+--------+

**审核挂 prescription:approve、发药挂 dispensing:create**：矩阵里这两个动作都由
pharmacist 持有，但它们是不同授权位——审核是「对处方的专业判断」，发药是「产生一条
发药记录」。拆成两个端点各挂各的门，与实际写入对齐（见 dispense_service 模块 docstring
关于两步建模的说明）。

**详情用 read/read_all 双门**：患者与医生持 ``dispensing:read``，药师只有
``dispensing:read_all``。只挂 read 会让刚发完药的药师打不开自己的记录，
与 exam/billing 侧的 _gate_any 同源处理。

路由薄：解析参数 → 调 dispense_service → 装配响应。先审后发的闸门、终态断言、
行级隔离都在服务层。
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    CurrentUser,
    DbSession,
    _load_role_permissions,
    get_role_name,
    require_permission,
)
from app.core.exceptions import PermissionDeniedError, ResourceNotFoundError
from app.models.auth import User
from app.schemas.dispensing import (
    ApproveRequest,
    DispenseRequest,
    DispensingListResponse,
    DispensingRecordOut,
)
from app.services import dispense_service as svc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dispensing", tags=["发药"])

_PermissionDeps: dict[tuple[str, str], Any] = {}


def _gate(resource: str, action: str):
    key = (resource, action)
    if key not in _PermissionDeps:
        _PermissionDeps[key] = require_permission(resource, action)
    return _PermissionDeps[key]


def _gate_any(resource: str, *actions: str):
    """组合权限门：持有任一动作即放行，全都没有才 403。见模块 docstring。"""

    async def checker(current_user: CurrentUser, db: DbSession) -> User:
        role_name = get_role_name(current_user)
        if role_name == "admin":
            return current_user
        permissions = await _load_role_permissions(db, current_user.role_id)
        if any((resource, action) in permissions for action in actions):
            return current_user
        raise PermissionDeniedError(
            f"缺少权限：{resource}:{'/'.join(actions)} 任一"
        )

    return checker


CanApproveRx = Annotated[User, Depends(_gate("prescription", "approve"))]
CanCreateDispensing = Annotated[User, Depends(_gate("dispensing", "create"))]
CanReadDispensing = Annotated[User, Depends(_gate("dispensing", "read"))]
CanReadAllDispensing = Annotated[User, Depends(_gate("dispensing", "read_all"))]
CanReadAnyDispensing = Annotated[
    User, Depends(_gate_any("dispensing", "read", "read_all"))
]


def _parse_uuid(value: str, what: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise ResourceNotFoundError(f"{what}不存在") from exc


# ==================== 审核 ====================


@router.post(
    "/prescriptions/{prescription_id}/approve",
    response_model=DispensingRecordOut,
    summary="药师审核处方（四查十对，落地 verified=True）",
)
async def approve_prescription(
    prescription_id: str,
    payload: ApproveRequest,
    db: DbSession,
    current_user: CanApproveRx,
) -> DispensingRecordOut:
    """对已缴费（paid）的处方做审核，建/更新 DispensingRecord 并置 verified=True。
    非 paid 处方返回 409；重复审核幂等返回当前记录。
    """
    record = await svc.approve_prescription(
        db,
        prescription_id=_parse_uuid(prescription_id, "处方"),
        pharmacist_user_id=current_user.id,
        notes=payload.notes,
    )
    logger.info(
        "处方审核操作人 user_id=%s prescription_id=%s",
        current_user.id,
        record.prescription_id,
    )
    return DispensingRecordOut(**await svc.serialize_record(db, record))


# ==================== 发药 ====================


@router.post(
    "/prescriptions/{prescription_id}/dispense",
    response_model=DispensingRecordOut,
    summary="发药（先审后发，处方 paid→dispensed）",
)
async def dispense_prescription(
    prescription_id: str,
    payload: DispenseRequest,
    db: DbSession,
    current_user: CanCreateDispensing,
) -> DispensingRecordOut:
    """实际发药。**必须已通过审核**（存在 verified=True 的发药记录），否则 409。
    成功后写 dispensed_at，把处方推进到 dispensed（终态，二次发药被状态机拒）。
    系统无库存表，本动作不扣任何库存。
    """
    record = await svc.dispense_prescription(
        db,
        prescription_id=_parse_uuid(prescription_id, "处方"),
        pharmacist_user_id=current_user.id,
        notes=payload.notes,
    )
    logger.info(
        "发药操作人 user_id=%s prescription_id=%s",
        current_user.id,
        record.prescription_id,
    )
    return DispensingRecordOut(**await svc.serialize_record(db, record))


# ==================== 取药记录查询 ====================


@router.get("/records", response_model=DispensingListResponse, summary="取药记录（我的）")
async def list_my_records(
    db: DbSession,
    current_user: CanReadDispensing,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> DispensingListResponse:
    """患者只看自己处方的取药记录，医生看自己接诊产生的（行级隔离在服务层强制）。"""
    rows, total = await svc.list_records(
        db,
        requester_user_id=current_user.id,
        role_name=get_role_name(current_user),
        page=page,
        page_size=page_size,
    )
    items = [DispensingRecordOut(**await svc.serialize_record(db, r)) for r in rows]
    return DispensingListResponse(
        items=items,
        total=total,
        page=max(1, page),
        page_size=min(page_size, 100),
    )


@router.get(
    "/records/all", response_model=DispensingListResponse, summary="取药记录（全域）"
)
async def list_all_records(
    db: DbSession,
    current_user: CanReadAllDispensing,
    patient_id: Annotated[str | None, Query(description="按患者过滤")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> DispensingListResponse:
    """药师的全域取药记录视图（dispensing:read_all）。"""
    rows, total = await svc.list_records(
        db,
        requester_user_id=current_user.id,
        role_name=get_role_name(current_user),
        patient_id=_parse_uuid(patient_id, "患者") if patient_id else None,
        page=page,
        page_size=page_size,
    )
    items = [DispensingRecordOut(**await svc.serialize_record(db, r)) for r in rows]
    return DispensingListResponse(
        items=items,
        total=total,
        page=max(1, page),
        page_size=min(page_size, 100),
    )


@router.get(
    "/records/{record_id}", response_model=DispensingRecordOut, summary="取药记录详情"
)
async def dispensing_record_detail(
    record_id: str,
    db: DbSession,
    current_user: CanReadAnyDispensing,
) -> DispensingRecordOut:
    """单条取药记录详情。read/read_all 任一即可进入；患者越权读他人记录返回 404。"""
    record = await svc.get_record_detail(
        db,
        record_id=_parse_uuid(record_id, "取药记录"),
        requester_user_id=current_user.id,
        role_name=get_role_name(current_user),
    )
    return DispensingRecordOut(**await svc.serialize_record(db, record))


__all__ = ["router"]
