"""预约（挂号）路由。

路由层保持薄：解析请求 → 调 ``appointment_service`` → 组装响应。
本模块出现的唯一业务判断是「路径参数是不是合法 UUID」，那属于协议层。

**权限映射与矩阵的对应关系**（rbac_matrix 为准，未做任何改动）：

- ``POST /appointments`` → ``appointment:create``（患者持有；admin 放行）
- ``GET /appointments``、``GET /appointments/{id}`` → ``appointment:read``
  （患者、医生持有）。跨行可见性不在这里控制，由 service 按角色做行级过滤——
  矩阵只回答「能不能做这个动作」，「能看到哪几行」是 service 的职责。
- ``DELETE /appointments/{id}`` → ``appointment:delete``（仅患者持有）
- ``PATCH /appointments/{id}/check-in`` → ``appointment:update``
  （仅医生持有，admin 由依赖直接放行）

check-in 为什么落在 update 上：矩阵的动作清单里没有 checkin 这类细粒度动作，
而签到本质是「把预约状态往前推进一格」，与医生接诊时改 in_progress 用的是
同一个 update 授权位（矩阵注释原话就是「接诊时改为 in_progress」）。
为此新增一个动作要同时改 RESOURCES/ACTIONS/矩阵/种子四处，
而收益只是命名更精确，不值得在本批次动权限矩阵。
收费员不持有 appointment 的任何权限，因此对预约四个端点全部 403。
"""

from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import DbSession, get_role_name, require_permission
from app.core.exceptions import ResourceNotFoundError
from app.models.auth import User
from app.models.clinical import Appointment
from app.schemas.appointment import (
    AppointmentCreate,
    AppointmentListResponse,
    AppointmentOut,
)
from app.services import appointment_service as svc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/appointments", tags=["预约挂号"])

CanCreate = Annotated[User, Depends(require_permission("appointment", "create"))]
# 读取与更新动作各在两个端点上复用，这里必须共享**同一个 checker 实例**：
# ``require_permission(...)`` 每次调用都返回一个新的闭包函数，而 FastAPI 的
# 「同一请求内相同依赖只解析一次」缓存按依赖对象身份判定。两处各调一次工厂，
# 就会对 permissions 表发两次查询；提成模块级变量后只回表一次。
# （签名里直接写 ``CanRead`` 这个 Annotated 别名即可，不要退化成
# ``Depends(CanRead.depends)``——后者绕过了别名所承载的依赖标识。）
_CanReadDep = require_permission("appointment", "read")
_CanUpdateDep = require_permission("appointment", "update")
CanRead = Annotated[User, Depends(_CanReadDep)]
CanDelete = Annotated[User, Depends(require_permission("appointment", "delete"))]
CanUpdate = Annotated[User, Depends(_CanUpdateDep)]


def _parse_uuid(value: str, what: str) -> uuid.UUID:
    """路径参数转 UUID，非法值报 404。

    与 knowledge.py 同一取舍：对客户端而言「ID 格式不对」和「这条记录不存在」
    是同一种失败（它拿不到资源），统一成 404 让前端错误分支只处理一种情况。
    """
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise ResourceNotFoundError(f"{what}不存在") from exc


async def _to_out(db: AsyncSession, appointment: Appointment) -> AppointmentOut:
    """ORM → 响应模型（序列化逻辑在 service，路由只做装配）。"""
    return AppointmentOut(**await svc.serialize_appointment(db, appointment))


# ==================== 创建 ====================


@router.post(
    "",
    response_model=AppointmentOut,
    status_code=status.HTTP_201_CREATED,
    summary="患者挂号（抢占号源）",
)
async def create_appointment(
    payload: AppointmentCreate,
    db: DbSession,
    current_user: CanCreate,
) -> AppointmentOut:
    """对着一个号源下单，成功即占住一格并返回预约单。

    **只能给本人挂号**：患者身份下的 ``patient_id`` 由当前登录用户的档案推出，
    请求体里根本没有「给谁挂」这个字段。这不是省事，而是必须——
    一旦开放 target_patient_id，任何人都能用别人的档案占号，
    而代挂号的正当需求（家属替老人挂号）应当走实名授权的独立流程，
    不是靠一个自由填写的 ID 字段实现。

    并发抢号的三种结局都返回 409，但 code 不同，前端据此分流：
    ``appointment_slot_full``（换时段）、``appointment_slot_conflict``
    （刷新重试）、``appointment_duplicate``（已有预约，跳详情）。

    201 而非 200：这次调用确实创建了一条预约记录。
    """
    appointment = await svc.create_appointment(
        db,
        user_id=current_user.id,
        schedule_id=_parse_uuid(payload.schedule_id, "号源"),
        chief_complaint=payload.chief_complaint,
        triage_result_id=payload.triage_result_id,
        triage_summary=payload.triage_summary,
    )
    return await _to_out(db, appointment)


# ==================== 读取 ====================


@router.get("", response_model=AppointmentListResponse, summary="我的预约列表")
async def list_appointments(
    db: DbSession,
    current_user: CanRead,
    status_filter: Annotated[
        str | None,
        Query(alias="status", description="booked/checked_in/in_progress/completed/cancelled/no_show"),
    ] = None,
    date_from: Annotated[str | None, Query(description="就诊日起 YYYY-MM-DD")] = None,
    date_to: Annotated[str | None, Query(description="就诊日止 YYYY-MM-DD")] = None,
    doctor_id: Annotated[str | None, Query(description="管理员按医生过滤")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> AppointmentListResponse:
    """患者看自己的预约，医生看自己出诊的预约，管理员看全量。

    三类身份共用一个端点而不拆成 /my-appointments 与 /doctor-appointments：
    URL 表达的是「谁的数据」，而这个信息已经在 token 里了；
    拆两个路径只会让前端要先判断角色再决定调哪个接口，
    反而多一处会写错的地方。行级过滤在 service 里按角色强制，
    客户端传什么参数都越不出自己的范围。

    日期字符串交给 Pydantic 解析（见 service 的 date 参数），
    非法格式会得到 422 而不是被静默忽略——静默忽略会让患者以为筛选生效了。
    """
    parsed_from = _parse_date(date_from, "date_from")
    parsed_to = _parse_date(date_to, "date_to")
    items, total = await svc.list_my_appointments(
        db,
        user_id=current_user.id,
        role_name=get_role_name(current_user),
        doctor_id=_parse_optional_uuid(doctor_id),
        status_filter=status_filter,
        date_from=parsed_from,
        date_to=parsed_to,
        page=page,
        page_size=page_size,
    )
    outs = [await _to_out(db, item) for item in items]
    return AppointmentListResponse(
        items=outs,
        total=total,
        page=max(1, page),
        page_size=max(1, min(page_size, 100)),
    )


@router.get("/{appointment_id}", response_model=AppointmentOut, summary="预约详情")
async def get_appointment(
    appointment_id: str,
    db: DbSession,
    current_user: CanRead,
) -> AppointmentOut:
    """取单条预约。越权访问返回 404 而非 403（理由见 service 说明）。"""
    appointment = await svc.get_appointment_detail(
        db,
        _parse_uuid(appointment_id, "预约"),
        requester_user_id=current_user.id,
        requester_role=get_role_name(current_user),
    )
    return await _to_out(db, appointment)


# ==================== 取消 ====================


@router.delete(
    "/{appointment_id}",
    response_model=AppointmentOut,
    summary="取消预约（退号）",
)
async def cancel_appointment(
    appointment_id: str,
    db: DbSession,
    current_user: CanDelete,
) -> AppointmentOut:
    """患者自助退号，同时把占用的号源放回库存。

    用 DELETE 带响应体而不是 204：取消后前端要立刻知道两件事——
    单子确实变成了 cancelled（而不是被软删），以及号源回退后的可迁移状态。
    204 会迫使前端再发一次 GET 才能刷新界面，多一次往返且可能读到
    别人刚改过的状态。knowledge 侧的删除用 204 是因为那行会从列表里消失，
    而这里的单子还要继续展示为「已取消」。

    非 booked 状态一律 409 + ``state_transition_invalid``，
    detail 里带 current_status/target_status/entity，前端能直接翻译成人话。
    """
    appointment = await svc.cancel_appointment(
        db,
        _parse_uuid(appointment_id, "预约"),
        requester_user_id=current_user.id,
    )
    return await _to_out(db, appointment)


# ==================== 签到 ====================


@router.patch(
    "/{appointment_id}/check-in",
    response_model=AppointmentOut,
    status_code=status.HTTP_200_OK,
    summary="到院签到（booked → checked_in）",
)
async def check_in_appointment(
    appointment_id: str,
    db: DbSession,
    _: CanUpdate,
) -> AppointmentOut:
    """把预约标记为已到院。

    **本端点当前由医生或管理员触发**（权限映射 appointment:update，
    见模块 docstring）。阶段 4 的接诊链路要求前置状态是 checked_in，
    因此服务层能力先具备、路由一并挂出便于联调；若日后要让患者自己在
    到院时扫码签到，需要另开一个走 appointment:create 语义之外的端点，
    并把归属校验加进来——届时不能复用本端点的权限位。

    幂等性提示：重复调用第二次会因 booked→checked_in 不再成立而得到 409，
    这是刻意保留的信号（让前台发现「这人已经签过到了」），
    不做成静默成功。
    """
    appointment = await svc.check_in_appointment(db, _parse_uuid(appointment_id, "预约"))
    return await _to_out(db, appointment)


# ==================== 参数解析 ====================


def _parse_date(value: str | None, field_name: str):
    """查询参数里的日期字符串 → ``date``，非法值 422。

    不复用 ``_parse_uuid`` 的「统一 404」策略：UUID 出现在路径里，
    语义是「我要这条资源」，格式错与不存在都是拿不到资源；
    日期出现在查询串里，语义是「请按此筛选」，格式错是请求写错了，
    报 404 会把人引向「没有数据」的错误结论。
    """
    if value is None or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field_name} 日期格式非法，应为 YYYY-MM-DD",
        ) from exc


def _parse_optional_uuid(value: str | None) -> uuid.UUID | None:
    """可选 UUID 查询参数，非法值 404（与路径参数口径一致：都是定位资源）。"""
    if value is None or not value.strip():
        return None
    return _parse_uuid(value, "医生")


__all__ = ["router"]
