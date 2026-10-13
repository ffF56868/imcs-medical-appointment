"""收费域路由：建单、支付、退费、取消与查询（阶段 5）。

============================ 权限映射（rbac_matrix 为准，未改动矩阵） ============================

+--------------------------------------------+---------------------------------+--------+
| 端点                                       | require_permission              | 角色   |
+--------------------------------------------+---------------------------------+--------+
| POST /billing/orders                       | charge_order:create             | 医生   |
| POST /billing/orders/{id}/pay              | charge_order:pay                | 收费员 |
| POST /billing/orders/{id}/refund           | charge_order:update             | 收费员 |
| POST /billing/orders/{id}/cancel           | charge_order:update             | 收费员 |
| GET  /billing/orders/all                   | charge_order:read_all           | 收费/药师
| GET  /billing/orders                       | charge_order:read               | 患者/医生
| GET  /billing/orders/{id}                  | charge_order:read/read_all 任一 | 全部   |
+--------------------------------------------+---------------------------------+--------+

**为什么详情用 read/read_all 双门（_gate_any）**：矩阵里患者与医生持
``charge_order:read``，而收费员与药师只有 ``charge_order:read_all``——它们没有单数 read。
详情接口若只挂 read，收费员刚收完的单子自己打不开；这与 exam.py 的
``_gate_any("exam_report","read","read_all")`` 是同一处理，不是矩阵疏漏。

**列表拆两个端点而不是一个端点判角色**：read（患者/医生，行级隔离）与
read_all（收费/药师，全院视图）是两个授权位，混在一个端点就得靠 role_name 硬编码猜，
正是 rbac_matrix 模块 docstring 要避免的事。service 层仍按 role_name 强制行级隔离，
两道防线叠加。

路由保持薄：解析参数 → 调 billing_service → 装配响应。业务判断（金额求值、状态联动、
退费边界、处方查重）全在服务层。
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
from app.schemas.billing import (
    ChargeOrderCreateRequest,
    ChargeOrderListResponse,
    ChargeOrderOut,
    PayRequest,
    RefundRequest,
)
from app.services import billing_service as svc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["收费"])

# 依赖工厂结果提成模块级私有变量再挂进 Annotated，保证同请求内相同依赖只解析一次
# （FastAPI 按依赖对象身份判定复用；每次现调工厂会对 permissions 表发多次查询）。
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


CanCreateCharge = Annotated[User, Depends(_gate("charge_order", "create"))]
CanPayCharge = Annotated[User, Depends(_gate("charge_order", "pay"))]
CanUpdateCharge = Annotated[User, Depends(_gate("charge_order", "update"))]
CanReadCharge = Annotated[User, Depends(_gate("charge_order", "read"))]
CanReadAllCharge = Annotated[User, Depends(_gate("charge_order", "read_all"))]
CanReadAnyCharge = Annotated[
    User, Depends(_gate_any("charge_order", "read", "read_all"))
]


def _parse_uuid(value: str, what: str) -> uuid.UUID:
    """路径/查询参数转 UUID，非法值 404（与 exam/consult/knowledge 同一口径）。"""
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise ResourceNotFoundError(f"{what}不存在") from exc


def _to_out(order) -> ChargeOrderOut:
    return ChargeOrderOut(**svc.serialize_order(order))


# ==================== 建单 ====================


@router.post(
    "/orders",
    response_model=ChargeOrderOut,
    status_code=status.HTTP_201_CREATED,
    summary="为就诊创建收费单（医生开单）",
)
async def create_charge_order(
    payload: ChargeOrderCreateRequest,
    db: DbSession,
    current_user: CanCreateCharge,
) -> ChargeOrderOut:
    """医生为某次就诊开出待缴费单。

    - 金额由服务端从源单据（处方 total_amount / 检查 fee_amount）求值，
      自定义项取客户端传入的分值；total_amount 为各项之和。
    - 同一处方只能被一张有效收费单关联，重复关联返回 409 + state_transition_invalid。
    - patient_id 从 visit 带出，不接受客户端指定；越权访问他人 visit 返回 404。
    """
    order = await svc.create_charge_order(
        db,
        doctor_user_id=current_user.id,
        visit_id=_parse_uuid(payload.visit_id, "就诊"),
        items_in=payload.items,
    )
    logger.info("建单操作人 user_id=%s order_id=%s", current_user.id, order.id)
    return _to_out(order)


# ==================== 支付 ====================


@router.post(
    "/orders/{order_id}/pay",
    response_model=ChargeOrderOut,
    summary="确认收款（pending→paid，联动处方 confirmed→paid）",
)
async def pay_charge_order(
    order_id: str,
    payload: PayRequest,
    db: DbSession,
    current_user: CanPayCharge,
) -> ChargeOrderOut:
    """收费员收款。**整单同生同灭**：关联处方若非 confirmed，整体回滚，
    收费单停在 pending，绝不出现「钱收了处方没进待发药队列」。
    已 paid 的单再次调用幂等返回当前态。
    """
    order = await svc.pay_charge_order(
        db,
        order_id=_parse_uuid(order_id, "收费单"),
        cashier_user_id=current_user.id,
        channel=payload.channel,
    )
    logger.info("收款操作人 user_id=%s order_id=%s", current_user.id, order.id)
    return _to_out(order)


# ==================== 退费 ====================


@router.post(
    "/orders/{order_id}/refund",
    response_model=ChargeOrderOut,
    summary="退费（paid→refunded）",
)
async def refund_charge_order(
    order_id: str,
    payload: RefundRequest,
    db: DbSession,
    current_user: CanUpdateCharge,
) -> ChargeOrderOut:
    """收费员退费。含处方项的收费单一旦支付即不可退（处方状态机无回退边），
    会被拒绝并给出「请先作废处方/走人工流程」提示；纯 exam/other 单可正常退。
    """
    order = await svc.refund_charge_order(
        db,
        order_id=_parse_uuid(order_id, "收费单"),
        cashier_user_id=current_user.id,
        reason=payload.reason,
    )
    logger.info(
        "退费操作人 user_id=%s order_id=%s 原因=%s",
        current_user.id,
        order.id,
        payload.reason,
    )
    return _to_out(order)


# ==================== 取消 ====================


@router.post(
    "/orders/{order_id}/cancel",
    response_model=ChargeOrderOut,
    summary="作废收费单（pending→cancelled）",
)
async def cancel_charge_order(
    order_id: str,
    db: DbSession,
    current_user: CanUpdateCharge,
) -> ChargeOrderOut:
    """作废未缴的单。已支付的单不能作废（矩阵里 paid 只能→refunded），返回 409。"""
    order = await svc.cancel_charge_order(
        db,
        order_id=_parse_uuid(order_id, "收费单"),
        operator_user_id=current_user.id,
    )
    logger.info("作废操作人 user_id=%s order_id=%s", current_user.id, order.id)
    return _to_out(order)


# ==================== 查询 ====================


@router.get("/orders", response_model=ChargeOrderListResponse, summary="收费单列表（我的）")
async def list_my_orders(
    db: DbSession,
    current_user: CanReadCharge,
    patient_id: Annotated[
        str | None, Query(description="医生可按患者过滤；患者传了会被忽略")
    ] = None,
    status_filter: Annotated[str | None, Query(description="按状态过滤")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ChargeOrderListResponse:
    """患者只看自己的单，医生看自己接诊产生的单（行级隔离在服务层强制）。"""
    rows, total = await svc.list_orders(
        db,
        requester_user_id=current_user.id,
        role_name=get_role_name(current_user),
        patient_id=_parse_uuid(patient_id, "患者") if patient_id else None,
        status_filter=status_filter,
        page=page,
        page_size=page_size,
    )
    return ChargeOrderListResponse(
        items=[_to_out(r) for r in rows],
        total=total,
        page=max(1, page),
        page_size=min(page_size, 100),
    )


@router.get(
    "/orders/all", response_model=ChargeOrderListResponse, summary="收费单列表（全域）"
)
async def list_all_orders(
    db: DbSession,
    current_user: CanReadAllCharge,
    patient_id: Annotated[str | None, Query(description="按患者过滤")] = None,
    status_filter: Annotated[str | None, Query(description="按状态过滤")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ChargeOrderListResponse:
    """收费员/药师的全域收费单视图（charge_order:read_all）。"""
    rows, total = await svc.list_orders(
        db,
        requester_user_id=current_user.id,
        role_name=get_role_name(current_user),
        patient_id=_parse_uuid(patient_id, "患者") if patient_id else None,
        status_filter=status_filter,
        page=page,
        page_size=page_size,
    )
    return ChargeOrderListResponse(
        items=[_to_out(r) for r in rows],
        total=total,
        page=max(1, page),
        page_size=min(page_size, 100),
    )


@router.get(
    "/orders/{order_id}", response_model=ChargeOrderOut, summary="收费单详情"
)
async def charge_order_detail(
    order_id: str,
    db: DbSession,
    current_user: CanReadAnyCharge,
) -> ChargeOrderOut:
    """单张收费单详情。read/read_all 任一即可进入；患者越权读他人单返回 404。"""
    order = await svc.get_order_detail(
        db,
        order_id=_parse_uuid(order_id, "收费单"),
        requester_user_id=current_user.id,
        role_name=get_role_name(current_user),
    )
    return _to_out(order)


__all__ = ["router"]
