"""接诊与病历路由（阶段 4 医生侧）。

路由层保持薄：解析请求 → 调 ``consult_service`` → 组装响应。
本模块出现的唯一业务判断是「路径参数是不是合法 UUID」，那属于协议层。

**权限映射与矩阵的对应关系**（rbac_matrix 为准，未做任何改动）：

- ``POST /consult/appointments/{id}/start`` → ``visit:create`` + ``appointment:update``
  （双闸门，理由见该端点 docstring）
- ``PUT /consult/visits/{id}/record`` → ``medical_record:update``（医生持有；
  患者只有 read，因此改写病历天然 403）
- ``POST /consult/visits/{id}/exams`` → ``exam_request:create``（仅医生持有，
  矩阵注释原话「不给 exam_request:create——开单权在医生，检验科只执行」）
- ``PATCH /consult/visits/{id}/close`` → ``visit:update``
- ``GET /consult/visits``、``GET /consult/visits/{id}``、
  ``GET /consult/appointments/pending`` → ``visit:read``
- ``GET /consult/visits/{id}/exams`` → ``exam_request:read``

**为什么"结束就诊""开始接诊"都复用现有动作而不是新增动作**：矩阵的动作清单
里没有 close / start 这类细粒度动作，而两者本质都是「把 visit 这条记录往前推一格」，
与 visit:update / visit:create 是同一个授权位。为此新增一个动作要同时改
RESOURCES/ACTIONS/ROLE_PERMISSIONS/种子四处，收益只是命名更精确，
不值得在本批次动权限矩阵。这与 appointment 侧把 check-in 落在 update 上的取舍同源。

**跨行可见性不在这里控制**：权限矩阵只回答「能不能做这个动作」，
「能看到哪几行」由 service 按角色强制（医生限自己 doctor_id、患者限自己档案），
越权一律 404 而非 403，不泄露「这条病历存在」。收费员对以上全部端点 403——
矩阵里 cashier 不持有 visit / medical_record 的任何动作，
也不持有 exam_request 的 create/read（它只有 read_all）。
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
from app.models.clinical import AppointmentStatus, VisitStatus
from app.schemas.consult import (
    CloseVisitResponse,
    ConsultStartResponse,
    ExamOrderRequest,
    ExamRequestOut,
    MedicalRecordSaveRequest,
    MedicalRecordSaveResponse,
    PendingAppointmentListResponse,
    PendingAppointmentOut,
    VisitListResponse,
    VisitOut,
)
from app.services import consult_service as svc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/consult", tags=["接诊病历"])

# 依赖实例集中定义：require_permission 每次调用都返回一个新闭包，而 FastAPI
# 的「同一请求内相同依赖只解析一次」按依赖对象身份判定。两处各调一次工厂，
# 就会对 permissions 表发两次查询；提成模块级变量后只回表一次。
# （签名里直接写 ``CanReadVisit`` 这个 Annotated 别名即可，不要退化成
# ``Depends(CanReadVisit.depends)``——后者绕过了别名所承载的依赖标识。）
_VisitCreateDep = require_permission("visit", "create")
CanCreateVisit = Annotated[User, Depends(_VisitCreateDep)]
_CanReadVisitDep = require_permission("visit", "read")
CanReadVisit = Annotated[User, Depends(_CanReadVisitDep)]
_CanUpdateVisitDep = require_permission("visit", "update")
CanUpdateVisit = Annotated[User, Depends(_CanUpdateVisitDep)]
CanWriteRecord = Annotated[User, Depends(require_permission("medical_record", "update"))]
CanCreateExam = Annotated[User, Depends(require_permission("exam_request", "create"))]
_CanReadExamDep = require_permission("exam_request", "read")
CanReadExam = Annotated[User, Depends(_CanReadExamDep)]
# 开始接诊会同时推进预约状态，因此除 visit:create 外还要求 appointment:update。
# 只保留 checker 实例本身（挂到路由 dependencies 上），不再包 Annotated 别名——
# 该端点无请求体，把它放进签名参数位会让 FastAPI 误解析出 body。
_AppointmentUpdateDep = require_permission("appointment", "update")


def _parse_uuid(value: str, what: str) -> uuid.UUID:
    """路径参数转 UUID，非法值报 404（与 appointment.py / knowledge.py 同一口径）。

    对客户端而言「ID 格式不对」和「这条记录不存在」是同一种失败——它拿不到资源，
    统一成 404 让前端错误分支只处理一种情况。
    """
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise ResourceNotFoundError(f"{what}不存在") from exc


async def _to_visit_out(db: AsyncSession, visit) -> VisitOut:
    """ORM → 响应模型（序列化逻辑在 service，路由只做装配）。"""
    return VisitOut(**await svc.serialize_visit(db, visit))


# ==================== 开始接诊 ====================


@router.post(
    "/appointments/{appointment_id}/start",
    response_model=ConsultStartResponse,
    status_code=status.HTTP_201_CREATED,
    # 第二个依赖不写进函数签名，而是挂在 dependencies 上：
    # 这个端点**没有请求体**，若把一个返回 ``User`` 的依赖塞进参数位，
    # FastAPI 会把返回类型是 Pydantic 模型的依赖当作 body 来解析——凭空多出一个
    # 必填 body，所有请求都会 422（triage 侧 batch-assess 就因同类写法漏出过 bug）。
    # appointment:update 在这里只是纯闸门，路由代码根本不消费它的返回值，
    # 放 dependencies 既表达准确又能少一个下划线形参。
    dependencies=[Depends(_AppointmentUpdateDep)],
    summary="开始接诊（预约 in_progress + 新建病历）",
)
async def start_consultation(
    appointment_id: str,
    db: DbSession,
    current_user: CanCreateVisit,
) -> ConsultStartResponse:
    """把一张已签到的预约推进为接诊中，并开出对应的病历。

    **双权限闸门 visit:create + appointment:update**：这一次调用同时创建 visit
    和修改 appointment，两个授权位分别覆盖两侧写入。只用 visit:create 的话，
    一个「能建病历但不能动预约」的角色就能把预约状态改掉——权限位与实际写入
    不对齐；反过来只用 appointment:update 则等于让挂号侧的角色凭空获得建病历的权力。
    矩阵里这两个动作恰好都由医生持有，因此正常流程不受影响。
    （appointment:update 挂在路由 ``dependencies`` 而非签名参数位，理由见装饰器注释。）

    前置状态必须是 checked_in：签到端点已在 appointment 路由存在，本批**不重复实现**。
    非 checked_in 会得到 409 + ``state_transition_invalid``，detail 带 current_status。

    同一次预约已有活跃病历时得到 409 + ``appointment_duplicate``（复用现类），
    detail 带回已有 visit_id，前端可直接跳转而不是再查一次。

    201 而非 200：这次调用确实新建了一条 visit 记录。

    响应附带该患者的既往病历摘要：医生首次打开接诊面板时必须能看到
    「这个人以前因什么看过我」，既往诊断直接影响当下的用药与检查决策。
    """
    visit = await svc.start_consultation(
        db, _parse_uuid(appointment_id, "预约"), current_user.id
    )
    priors = await svc.list_prior_visits(
        db, visit.patient_id, limit=svc.PRIOR_VISITS_LIMIT
    )
    # 排除刚建的这一条本身：否则医生会在「既往病历」里看到自己三秒前开的空白病历。
    # 同时只保留已结案（closed）的记录：同一次挂号重开时，上一段还在书写中的
    # 病历属于「未完成的本次诊疗」，混进既往史会让医生误以为那是另一件事。
    history = [
        item
        for item in priors
        if item.id != visit.id and item.status == VisitStatus.CLOSED.value
    ]
    return ConsultStartResponse(
        visit=await _to_visit_out(db, visit),
        appointment_status=AppointmentStatus.IN_PROGRESS.value,
        prior_visits=[await _to_visit_out(db, item) for item in history],
    )


# ==================== 病历书写 ====================


@router.put(
    "/visits/{visit_id}/record",
    response_model=MedicalRecordSaveResponse,
    summary="书写/修改病历（首次写入自动定稿）",
)
async def save_medical_record(
    visit_id: str,
    payload: MedicalRecordSaveRequest,
    db: DbSession,
    current_user: CanWriteRecord,
) -> MedicalRecordSaveResponse:
    """保存病历内容；visit 从 in_progress 首次写入时流转到 documented。

    用 PUT 而不是 PATCH：语义是「把这一次病历的内容提交上去」，
    幂等且可安全重试。网络超时后前端重发同一个请求不会写出两份内容。
    但**不是全量覆盖**：payload 里 None 表示「本次不涉及该栏」，
    service 采用传了才覆盖的部分更新语义（详见 service docstring）。

    documented 之后仍可继续修改，直到 closed 归档为止——AI 辅助开药前医生要能补病历。
    closed 之后拒改，返回 409 + ``state_transition_invalid``。

    空请求体（所有字段都不传）是合法的：它只触发 in_progress→documented 而不改内容，
    对应「医生口头问完就直接定稿」这种轻症场景，不做成 422。
    """
    visit, transitioned = await svc.save_medical_record(
        db, _parse_uuid(visit_id, "就诊记录"), current_user.id, payload
    )
    return MedicalRecordSaveResponse(
        visit=await _to_visit_out(db, visit), transitioned=transitioned
    )


# ==================== 检查开单 ====================


@router.post(
    "/visits/{visit_id}/exams",
    response_model=ExamRequestOut,
    status_code=status.HTTP_201_CREATED,
    summary="开具检查申请单",
)
async def order_exam(
    visit_id: str,
    payload: ExamOrderRequest,
    db: DbSession,
    current_user: CanCreateExam,
) -> ExamRequestOut:
    """为本次就诊开一张检查单，初始态 ordered，费用取演示价目常量。

    仅 in_progress / documented 的 visit 可开单；closed 后需另起一次就诊，
    否则报告挂在已归档病历下会被质控当成漏诊处理。

    ``urgency='stat'`` 时 service 会额外记一条 info 日志作为危急路径标记，
    供并行开发中的检验侧与运维检索——两侧只通过 urgency 列这一个事实通信，
    不建立函数级耦合。

    本端点**不校验 exam_type 是否在价目表内**：临床项目远多于八项演示条目，
    表外项目费用落 0 交由阶段 5 收费侧核价，比在这里拦住医生更符合实际工作流。
    """
    request = await svc.order_exam(
        db, _parse_uuid(visit_id, "就诊记录"), current_user.id, payload
    )
    return ExamRequestOut(**await svc.serialize_exam_request(db, request))


@router.get(
    "/visits/{visit_id}/exams",
    response_model=list[ExamRequestOut],
    summary="本次就诊的检查单列表",
)
async def list_visit_exams(
    visit_id: str,
    db: DbSession,
    current_user: CanReadExam,
) -> list[ExamRequestOut]:
    """列出某次就诊的全部检查单。医生限本人，患者限本人，越权 404。

    患者经 ``exam_request:read`` 也能读到自己的检查单进度（ordered/received/…），
    这是「我的检查做到哪一步了」这一高频咨询的唯一数据来源。
    """
    requests = await svc.list_visit_exam_requests(
        db,
        _parse_uuid(visit_id, "就诊记录"),
        get_role_name(current_user),
        current_user.id,
    )
    return [ExamRequestOut(**await svc.serialize_exam_request(db, item)) for item in requests]


# ==================== 结束就诊 ====================


@router.patch(
    "/visits/{visit_id}/close",
    response_model=CloseVisitResponse,
    summary="结束就诊（病历结案 + 预约完成）",
)
async def close_visit(
    visit_id: str,
    db: DbSession,
    current_user: CanUpdateVisit,
) -> CloseVisitResponse:
    """病历归档，并把关联预约一并标记为已完成。

    权限复用 visit:update（矩阵无 close 动作，理由见模块 docstring）。

    **必须先有 documented 病历**：in_progress 直接结案会得到 409。转换矩阵允许
    in_progress→closed 直跳，本接口刻意不放行——没有诊断结论的结案会留下一条
    无法质控的病历，医疗纠纷时医院拿不出任何诊疗依据。

    跨实体事务：visit.closed 与 appointment.completed 同生同灭，
    任何一步失败整体回滚，不会出现「病历关了但预约还在接诊中」的半态。

    用 PATCH 且返回响应体而不是 204：结案后前端要立刻确认两侧状态都落地了，
    204 会迫使前端再发两次 GET，多两趟往返还可能读到别人刚改过的数据。

    幂等性提示：重复调用第二次会因 closed 是终态而得到 409，这是刻意保留的信号
    （让前台发现「这次已经结过案」），不做成静默成功。
    """
    visit, appointment = await svc.close_visit(
        db, _parse_uuid(visit_id, "就诊记录"), current_user.id
    )
    return CloseVisitResponse(
        visit=await _to_visit_out(db, visit),
        appointment_id=str(appointment.id),
        appointment_status=appointment.status,
    )


# ==================== 医生工作台 ====================


@router.get(
    "/appointments/pending",
    response_model=PendingAppointmentListResponse,
    summary="今日待诊队列（医生工作台）",
)
async def list_pending_appointments(
    db: DbSession,
    current_user: CanReadVisit,
    date_param: Annotated[
        str | None,
        Query(alias="date", description="查询日 YYYY-MM-DD，缺省今天（东八区）"),
    ] = None,
) -> PendingAppointmentListResponse:
    """当前医生在指定日期仍需处理的预约（checked_in / in_progress）。

    **只对医生角色有意义**：路径里的医生身份取自 JWT，患者传自己的 user_id 也
    查不出别人的队列。患者角色走到这里会因为「该医生名下没有预约」而拿到空列表
    ——不报错是因为报错会泄露「这个账号不是医生」这一信息，且患者本来也没有
    查看别人候诊队列的正当需求（矩阵里患者有 visit:read，用于读自己的病历）。

    每行回填活跃 visit 的 id 与状态，前端据此决定按钮文案是「开始接诊」
    还是「继续书写」，不必自己再拉一次病历列表做关联。

    日期非法报 422 而不是 404：查询串里的日期语义是「请按此筛选」，
    格式错是请求写错了；统一成 404 会把人引向「今天没人候诊」的错误结论。
    与 appointment.py 的 ``_parse_date`` 同一取舍。
    """
    day = _parse_date(date_param)
    items, effective_day = await svc.list_pending_appointments(db, current_user.id, day)
    return PendingAppointmentListResponse(
        doctor_id=str(current_user.id),
        date=effective_day,
        items=[PendingAppointmentOut(**item) for item in items],
        total=len(items),
    )


def _parse_date(value: str | None) -> date | None:
    """查询参数里的日期字符串 → ``date``，非法值 422，空值 None（交给 service 取今天）。"""
    if value is None or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="date 日期格式非法，应为 YYYY-MM-DD",
        ) from exc


# ==================== 病历读取 ====================


@router.get("/visits", response_model=VisitListResponse, summary="我的病历列表（分页）")
async def list_visits(
    db: DbSession,
    current_user: CanReadVisit,
    status_filter: Annotated[
        str | None,
        Query(alias="status", description="in_progress/documented/closed"),
    ] = None,
    patient_id: Annotated[str | None, Query(description="医生按患者档案过滤")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> VisitListResponse:
    """医生看自己名下的病历，患者看自己档案下的病历，共用一个端点。

    不拆 /my-visits 与 /doctor-visits：URL 表达的是「谁的数据」，而这个信息
    已经在 token 里；拆两个路径只会让前端先判角色再决定调哪个接口，
    反而多一处会写错的地方。行级过滤在 service 里按角色强制，
    客户端传什么参数都越不出自己的范围。

    ``patient_id`` 只对医生生效（service 内部按角色分流），患者传别人的
    patient_id 会被忽略并按自己的档案过滤——越权读取在数据层就不成立。
    """
    role = get_role_name(current_user)
    if role == "patient":
        # 患者侧的 status 筛选只落地「只看已结案」这一档（对应历史病历页），
        # in_progress / documented 对患者的语义都是「还没结束」，
        # 因此传这两个值时不做过滤而不是把它们当成精确匹配——
        # 精确匹配会让患者以为「书写中」和「接诊中」是两种能查到的东西。
        items, total = await svc.list_patient_visits(
            db,
            user_id=current_user.id,
            include_closed_only=(status_filter == "closed"),
            page=page,
            page_size=page_size,
        )
    else:
        items, total = await svc.list_my_visits(
            db,
            doctor_id=current_user.id,
            status_filter=status_filter,
            patient_id=_parse_uuid(patient_id, "患者") if patient_id else None,
            page=page,
            page_size=page_size,
        )
    outs = [await _to_visit_out(db, item) for item in items]
    return VisitListResponse(
        items=outs,
        total=total,
        page=max(1, page),
        page_size=max(1, min(page_size, 100)),
    )


@router.get("/visits/{visit_id}", response_model=VisitOut, summary="病历详情")
async def get_visit_detail(
    visit_id: str,
    db: DbSession,
    current_user: CanReadVisit,
) -> VisitOut:
    """取单条病历。越权访问返回 404 而非 403（不泄露「这条病历存在」）。

    管理员同样 404：权限矩阵刻意不给 admin 临床记录的读写，
    行级可见范围由 service 决定，不因「超级管理员」这个头衔开后门。
    """
    visit = await svc.get_visit_detail(
        db,
        _parse_uuid(visit_id, "就诊记录"),
        requester_user_id=current_user.id,
        requester_role=get_role_name(current_user),
    )
    return await _to_visit_out(db, visit)


__all__ = ["router"]
