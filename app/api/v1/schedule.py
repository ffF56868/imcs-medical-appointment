"""号源（医生排班）查询路由。

只读接口。排班的**写**操作（新建/停诊/改号源数）属于管理员的
``schedule:create/update/delete``，本批次不实现——那是运营后台的活，
与患者挂号链路是两套权限与审计要求，混在一个 router 里会让
「谁能改号源」这件事变得要看函数名才知道。

**权限用 ``schedule:read`` 而非 require_role("patient")**：
按 rbac_matrix，患者、医生都持有 schedule:read，admin 由
require_permission 直接放行；而 lab_tech / cashier / pharmacist 没有。
这个分布正是业务期望——患者查号、医生看同事排班，收费员和药师不需要。
所以这里必须走权限矩阵而不是角色名硬编码。

注意 ``GET /schedules`` 对**患者也开放**：号源列表本身不含任何隐私信息
（医生姓名职称是公开门诊信息），把它做成需要更高权限的接口，
患者端就无法展示「还有号的医生」，整个挂号入口会断掉。
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import ValidationError

from app.api.deps import DbSession, require_permission
from app.models.auth import User
from app.schemas.appointment import (
    DepartmentDoctorsResponse,
    DepartmentDoctorOut,
    ScheduleListResponse,
    ScheduleQuery,
    ScheduleWithDoctorOut,
)
from app.services import appointment_service as svc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/schedules", tags=["号源排班"])

CanReadSchedule = Annotated[User, Depends(require_permission("schedule", "read"))]


def _build_query(
    start_date: str | None,
    end_date: str | None,
    department_code: str | None,
    department_id: str | None,
    doctor_id: str | None,
    title: str | None,
    slot_period: str | None,
    status: str | None,
    keyword: str | None,
    include_expired: bool,
    page: int,
    page_size: int,
) -> ScheduleQuery:
    """把零散的 Query 参数装配成 ScheduleQuery。

    日期以字符串接收后交给 Pydantic 解析（而不是在签名里写 ``date``）：
    这样非法日期格式产生的 422 结构与请求体校验保持一致，
    前端只需一套错误处理逻辑。**但 ``model_validate`` 抛的 ValidationError
    不会被 FastAPI 自动转成 422**——它发生在函数体内，不在 Pydantic 能接管的
    路由签名上。不显式兜住的话，``start_date=not-a-date`` 会冒泡到未处理异常
    兜底变成 500（实测复现），所以这里手动转 422，与请求体校验同构。
    """
    try:
        return ScheduleQuery.model_validate(
            {
                "start_date": start_date,
                "end_date": end_date,
                "department_code": department_code,
                "department_id": department_id,
                "doctor_id": doctor_id,
                "title": title,
                "slot_period": slot_period,
                "status": status,
                "keyword": keyword,
                "include_expired": include_expired,
                "page": page,
                "page_size": page_size,
            }
        )
    except ValidationError as exc:
        # 用数字 422 而非 status.HTTP_422_*：本函数参数名就叫 status
        # （号源状态筛选），已遮蔽模块顶部 from fastapi import status，
        # 写成 status.HTTP_422_UNPROCESSABLE_ENTITY 会抛 AttributeError。
        raise HTTPException(
            status_code=422,
            detail={
                "code": "request_validation",
                "errors": exc.errors(include_url=False),
            },
        ) from exc


@router.get("", response_model=ScheduleListResponse, summary="分页查询可挂号源")
async def list_schedules(
    db: DbSession,
    _: CanReadSchedule,
    start_date: Annotated[str | None, Query(description="起始就诊日 YYYY-MM-DD，缺省今天")] = None,
    end_date: Annotated[str | None, Query(description="截止就诊日 YYYY-MM-DD，缺省窗口 14 天")] = None,
    department_code: Annotated[str | None, Query(max_length=20, description="科室短代码")] = None,
    department_id: Annotated[str | None, Query(description="科室 UUID")] = None,
    doctor_id: Annotated[str | None, Query(description="医生 UUID")] = None,
    title: Annotated[str | None, Query(max_length=50, description="职称精确匹配")] = None,
    slot_period: Annotated[str | None, Query(description="am / pm")] = None,
    status: Annotated[str | None, Query(description="open / full / closed，缺省 open")] = "open",
    keyword: Annotated[str | None, Query(max_length=50, description="医生姓名/职称/专长")] = None,
    include_expired: Annotated[bool, Query(description="是否包含过期排班")] = False,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ScheduleListResponse:
    """按日期区间、科室、医生、职称筛选号源，返回剩余号。

    默认只出 open 且未过期的号源（见 service 说明）。``remaining_slots``
    是查询瞬间的快照值，客户端不能把它当成「一定能挂上」的承诺——
    真正的成败由 POST /appointments 的乐观锁判定。
    """
    query = _build_query(
        start_date,
        end_date,
        department_code,
        department_id,
        doctor_id,
        title,
        slot_period,
        status,
        keyword,
        include_expired,
        page,
        page_size,
    )
    rows, total, (start, end) = await svc.list_available_schedules(db, query)
    return ScheduleListResponse(
        items=[ScheduleWithDoctorOut(**row) for row in rows],
        total=total,
        page=query.page,
        page_size=min(query.page_size, 100),
        start_date=start,
        end_date=end,
    )


@router.get(
    "/doctors-by-department",
    response_model=DepartmentDoctorsResponse,
    summary="按科室列出仍有号源的医生",
)
async def list_doctors_by_department(
    db: DbSession,
    _: CanReadSchedule,
    department_code: Annotated[str | None, Query(max_length=20, description="科室短代码")] = None,
    department_id: Annotated[str | None, Query(description="科室 UUID")] = None,
    start_date: Annotated[str | None, Query(description="起始就诊日，缺省今天")] = None,
    end_date: Annotated[str | None, Query(description="截止就诊日，缺省窗口 14 天")] = None,
    title: Annotated[str | None, Query(max_length=50, description="职称精确匹配")] = None,
    keyword: Annotated[str | None, Query(max_length=50, description="医生姓名/职称/专长")] = None,
) -> DepartmentDoctorsResponse:
    """患者端「心血管科有哪些号」视图：返回该科室未来仍有余号的医生及其可约日期。

    与 GET /schedules 的区别是聚合粒度：这里一行一个医生，回答的是
    「该找谁」；号源列表一行一个半天，回答的是「什么时候还有位置」。
    两个视图的数据来源同一张表，但让前端各自拼 GROUP BY 容易与后端口径漂移，
    所以聚合放在服务端。

    本接口不分页：单科室医生数量级为个位数到十几，全量返回比翻页体验好；
    若日后要支持全院医生浏览，再补 page/page_size。
    """
    query = _build_query(
        start_date,
        end_date,
        department_code,
        department_id,
        None,
        title,
        None,
        "open",
        keyword,
        False,
        1,
        100,
    )
    doctors, (start, end) = await svc.list_doctors_with_open_slots(db, query)
    return DepartmentDoctorsResponse(
        department_code=department_code,
        department_id=department_id,
        start_date=start,
        end_date=end,
        items=[DepartmentDoctorOut(**item) for item in doctors],
        doctor_count=len(doctors),
    )


__all__ = ["router"]
