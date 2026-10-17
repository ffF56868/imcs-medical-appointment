"""检验科路由：申请单接收、报告录入、危急值提示与 AI 解读触发/查询。

============================ 权限映射（rbac_matrix 为准，未改动矩阵） ============================

+----------------------------------+-----------------------------------------+-----+
| 端点                             | require_permission                      | 说明|
+----------------------------------+-----------------------------------------+-----+
| GET  /exams/queue                | exam_request:read_all                   | 仅检验科（医生无 read_all）
| POST /exams/requests/{id}/receive| exam_request:update                     | 仅检验科
| POST /exams/requests/{id}/process| exam_request:update                     | 仅检验科
| POST /exams/requests/{id}/reports| exam_report:create                      | 仅检验科
| GET  /exams/reports              | exam_report:read                        | 患者/医生；检验科走 read_all 版
| GET  /exams/reports/all          | exam_report:read_all                    | 仅检验科
| GET  /exams/reports/{id}         | exam_report:read                        | 患者/医生；服务层做行级隔离
| POST /exams/reports/{id}/interpret | exam_interpretation:interpret         | 医生/检验科
+----------------------------------+-----------------------------------------+-----+

``GET /exams/reports/{id}`` 同时给患者和医生用，因此挂的是 ``exam_report:read``
（患者、医生持有）。检验科只有 ``read_all`` 没有 ``read``——这不是矩阵的疏漏，
而是它的读入口本来就是「全院待办」而非「某个患者的报告」，所以给它单独一个
``/reports/all``。若在这里只挂 read，检验科查自己刚录的报告反而会 403。

路由保持薄：解析参数 → 调 ``exam_service`` → 装配响应。唯一的判断是
「UUID 形态是否合法」（协议层校验）与「危急值汇总文本」（纯展示拼装）。
异常判定、状态流转、派发时序全部在服务层。

============================ 异步交互模型（前端必读） ============================

``POST /exams/reports`` **不等 LLM**：报告数据落库后立即返回 201，
响应里 ``report.ai.ai_status`` 通常是 ``pending``（排队中）或 ``processing``
（worker 已接手）。前端拿到 ``report.id`` 后按下面规则轮询本模块的详情接口：

1. ``pending`` / ``processing`` → 继续轮询（建议 2s 起、指数退避到 8s，上限 2 分钟）
2. ``completed`` → 停止轮询，渲染 ``ai.summary`` + ``ai.suggestions``，
   并常驻显示「需医生审核后方可作为诊疗依据」（``ai.need_doctor_review`` 恒 true）
3. ``failed`` → 停止轮询，展示原始指标（数据一定完好）并提供「重新解读」按钮
   → 调 ``POST /exams/reports/{id}/interpret``
4. ``skipped`` → 内容未通过安全审核，同 failed 处理，但文案不能写成「系统故障」

为什么不同步等：一次解读含 RAG 检索 + LLM 结构化生成，实测可达十几秒，
放在请求线程里必然超时；而超时的请求会被用户重发，产生多份解读互相覆盖。
为什么危急值不等 AI：``critical`` / ``critical_message`` 在录入瞬间由规则算出，
前端必须**立刻**弹提示——这是抢救指征，不能排在一次网络调用后面。
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
from app.models.clinical import ExamReport, ExamRequest
from app.schemas.exam import (
    AbnormalItemOut,
    AIInterpretationOut,
    CriticalValueOut,
    DispatchOut,
    ExamQueueResponse,
    ExamReportListResponse,
    ExamReportOut,
    ExamRequestOut,
    InterpretRequest,
    InterpretTriggerResponse,
    ReportSubmitRequest,
    ReportSubmitResponse,
)
from app.services import exam_service as es

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/exams", tags=["检查报告"])

# 集中定义鉴权闸门：同一个动作在不同端点上权限是否一致，只看这一处就能确认。
# 依赖返回值当闸门用，需要 user_id 的端点另外声明 CurrentUser（FastAPI 会复用同一依赖）。
#
# ``require_permission`` 每次调用都返回一个新闭包，而 FastAPI 的「同一请求内相同
# 依赖只解析一次」按依赖对象身份判定，因此工厂结果先提成模块级私有变量再挂进
# Annotated（与 consult.py / triage.py 同口径）。签名里写别名本身没问题——
# 出问题的是本文件曾经把别名当成 ``Depends(...)`` 的**参数**二次包裹（阶段 3 教训），
# 那会让 FastAPI 把一个已含依赖的 Annotated 当作待注入可调用对象解析。
_PermissionDeps: dict[tuple[str, str], Any] = {}


def _gate(resource: str, action: str):
    """按 (资源, 动作) 缓存 require_permission 闭包，保证同请求内只解析一次。"""
    key = (resource, action)
    if key not in _PermissionDeps:
        _PermissionDeps[key] = require_permission(resource, action)
    return _PermissionDeps[key]


def _gate_any(resource: str, *actions: str):
    """组合权限门：持有任一动作即放行，全都没有才 403。

    报告详情必须用它而不是单一 ``_gate(..., "read")``：矩阵里
    患者/医生持 ``exam_report:read``，**检验科只有 ``exam_report:read_all``**
    （它是生产者，反而没有面向协作的单数 read）。详情接口按单数 read
    把关的话，检验科刚录入的报告自己打不开，工作台的重触发入口也就无从进入。
    与 knowledge 的 read/read_all 双门是同一处理。
    """
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


CanReadAnyReport = Annotated[
    User, Depends(_gate_any("exam_report", "read", "read_all"))
]


CanUpdateRequest = Annotated[User, Depends(_gate("exam_request", "update"))]
CanReadAllRequest = Annotated[User, Depends(_gate("exam_request", "read_all"))]
CanCreateReport = Annotated[User, Depends(_gate("exam_report", "create"))]
CanReadReport = Annotated[User, Depends(_gate("exam_report", "read"))]
CanReadAllReport = Annotated[User, Depends(_gate("exam_report", "read_all"))]
CanInterpret = Annotated[User, Depends(_gate("exam_interpretation", "interpret"))]


def _parse_uuid(value: str, what: str) -> uuid.UUID:
    """路径/查询参数转 UUID，非法值 404（与 knowledge/triage/appointment 同一口径）。

    不用 Pydantic UUID 自动校验（那会是 422）：对客户端来说「ID 格式不对」
    和「这条记录不存在」是同一种失败——它拿不到资源。统一 404 让前端只处理一个分支。
    """
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise ResourceNotFoundError(f"{what}不存在") from exc


# ==================== 序列化 ====================


def _abnormal_items_out(structured: Any) -> list[AbnormalItemOut]:
    return [
        AbnormalItemOut(
            name=a.name,
            code=a.code or None,
            value=a.value,
            unit=a.unit or None,
            ref_low=a.ref_low,
            ref_high=a.ref_high,
            direction=a.direction,
            flag=a.to_flag(),
        )
        for a in es.evaluate_abnormalities(structured)
    ]


def _critical_out(structured: Any) -> list[CriticalValueOut]:
    return [
        CriticalValueOut(
            name=c.name,
            code=c.code,
            value=c.value,
            unit=c.unit or None,
            threshold_text=c.threshold_text,
            message=c.to_line(),
        )
        for c in es.detect_critical_values(structured)
    ]


def _critical_message(hits: list[CriticalValueOut]) -> str | None:
    """汇总成一句给前端直接显示的告警文本。"""
    if not hits:
        return None
    names = "、".join(h.name for h in hits)
    return (
        f"⚠ 危急值提醒：{names} 已达危急阈值。请立即通知接诊医生复核，"
        "不要等待 AI 解读结果。"
    )


def to_ai_out(report: ExamReport) -> AIInterpretationOut:
    confidence = report.ai_confidence
    return AIInterpretationOut(
        ai_status=report.ai_status,
        summary=report.ai_summary,
        suggestions=list(report.ai_suggestions or []),
        # severity 不在表结构里（models/clinical.py 没有 ai_severity 列），
        # 它是解读结果的属性而不是独立可查询字段，因此从 suggestions/summary 之外
        # 无处还原。这里如实返回 None，由阶段 5 决定是否新增列或塞进 ai_interpretation。
        severity=None,
        confidence=float(confidence) if confidence is not None else None,
        model=report.ai_model,
        provider=report.ai_provider,
        ai_task_id=str(report.ai_task_id) if report.ai_task_id else None,
        interpreted_at=report.interpreted_at,
        need_doctor_review=True,
    )


def to_report_out(
    report: ExamReport,
    request_info: tuple[str | None, str | None, str | None, str | None] | None = None,
) -> ExamReportOut:
    exam_type, exam_name, urgency, request_status = request_info or (
        None,
        None,
        None,
        None,
    )
    criticals = _critical_out(report.structured_data)
    return ExamReportOut(
        id=str(report.id),
        request_id=str(report.request_id),
        patient_id=str(report.patient_id),
        lab_tech_id=str(report.lab_tech_id),
        exam_type=exam_type,
        exam_name=exam_name,
        urgency=urgency,
        request_status=request_status,
        raw_content=report.raw_content,
        structured_data=report.structured_data,
        abnormal_flags=list(report.abnormal_flags or []),
        abnormal_items=_abnormal_items_out(report.structured_data),
        critical_values=criticals,
        critical=bool(criticals),
        critical_message=_critical_message(criticals),
        conclusion=report.conclusion,
        created_at=report.created_at,
        ai=to_ai_out(report),
    )


def to_dispatch_out(result: es.DispatchResult) -> DispatchOut:
    return DispatchOut(
        dispatched=result.dispatched,
        ai_task_id=str(result.ai_task_id) if result.ai_task_id else None,
        celery_task_id=result.celery_task_id,
        reason=result.reason,
    )


def to_request_out(
    req: ExamRequest,
    report_link: tuple[uuid.UUID, str] | None = None,
) -> ExamRequestOut:
    return ExamRequestOut(
        id=str(req.id),
        visit_id=str(req.visit_id),
        patient_id=str(req.patient_id),
        doctor_id=str(req.doctor_id),
        exam_type=req.exam_type,
        exam_name=req.exam_name,
        body_part=req.body_part,
        clinical_info=req.clinical_info,
        urgency=req.urgency,
        status=req.status,
        created_at=req.created_at,
        report_id=str(report_link[0]) if report_link else None,
        ai_status=report_link[1] if report_link else None,
    )


# ==================== 检验科工作台 ====================


@router.get("/queue", response_model=ExamQueueResponse, summary="检验科待办工作台")
async def exam_queue(
    db: DbSession,
    _: CanReadAllRequest,
    status_filter: Annotated[
        str | None, Query(description="received / processing，留空为两者都要")
    ] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ExamQueueResponse:
    """按紧急度优先、同级先到先做的待办列表。

    STAT 单子会自动插队：检验科的现实是「常规队列按 FIFO 做，
    但临床打电话来说这人快不行了就得先做」，把这条隐性规则显式化在排序里，
    比指望每个人记住要去看颜色标记可靠。
    """
    rows, total = await es.list_queue(
        db, status_filter=status_filter, page=page, page_size=page_size
    )
    links = await es.queue_report_map(db, rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.urgency] = counts.get(r.urgency, 0) + 1
    return ExamQueueResponse(
        items=[to_request_out(r, links.get(r.id)) for r in rows],
        total=total,
        page=max(1, page),
        page_size=min(page_size, 100),
        counts=counts,
    )


# ==================== 申请单状态流转 ====================


@router.get(
    "/requests",
    response_model=ExamQueueResponse,
    summary="待接收申请单（ordered，检验科接单视图）",
)
async def list_ordered_requests(
    db: DbSession,
    _: CanReadAllRequest,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ExamQueueResponse:
    """医生刚开出、还没到检验科手上的单子（status=ordered）。

    为什么单独开这一个端点而不是并进 ``GET /queue``：queue 的语义是
    「已在手上、按 FIFO 推进」，ordered 进去会让「接收」按钮对已接收单
    重复出现，服务端幂等兜得住，但 UI 语义会乱。接单视图与工作台分开，
    两个数字（可接 N 单 / 在办 M 单）也各自可解释。

    返回体复用 ExamQueueResponse：``counts`` 这里只有 ordered 一种来源，
    按 urgency 分布的语义与 queue 完全一致，前端组件可直接共用。
    """
    rows, total = await es.list_ordered_requests(db, page=page, page_size=page_size)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.urgency] = counts.get(r.urgency, 0) + 1
    return ExamQueueResponse(
        items=[to_request_out(r) for r in rows],
        total=total,
        page=max(1, page),
        page_size=min(page_size, 100),
        counts=counts,
    )


@router.post(
    "/requests/{request_id}/receive", response_model=ExamRequestOut, summary="接收检查申请"
)
async def receive_exam_request(
    request_id: str,
    db: DbSession,
    current_user: CanUpdateRequest,
) -> ExamRequestOut:
    """ordered → received。重复接收幂等返回当前状态（不报错）。"""
    req = await es.receive_request(db, _parse_uuid(request_id, "检查申请单"))
    logger.info(
        "申请单接收操作人 user_id=%s request_id=%s", current_user.id, req.id
    )
    return to_request_out(req)


@router.post(
    "/requests/{request_id}/process", response_model=ExamRequestOut, summary="开始执行检查"
)
async def start_exam_processing(
    request_id: str,
    db: DbSession,
    current_user: CanUpdateRequest,
) -> ExamRequestOut:
    """received → processing；转换矩阵亦允许 ordered → processing 直跳。

    非法起点（reported/reviewed）返回 409 且 detail 带 current_status/target_status，
    前端能明确告诉操作人「这份单子已经出过报告了」。
    """
    req = await es.start_processing(db, _parse_uuid(request_id, "检查申请单"))
    logger.info(
        "申请单进入执行操作人 user_id=%s request_id=%s", current_user.id, req.id
    )
    return to_request_out(req)


# ==================== 报告录入 ====================


@router.post(
    "/requests/{request_id}/reports",
    response_model=ReportSubmitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="录入检查报告（自动派发 AI 解读）",
)
async def submit_exam_report(
    request_id: str,
    payload: ReportSubmitRequest,
    db: DbSession,
    current_user: CanCreateReport,
) -> ReportSubmitResponse:
    """检验科出报告：processing → reported，建 ExamReport 并自动派发 AI 解读。

    **立即返回，不等 LLM**（交互模型见模块 docstring）。返回时 AI 多半还在排队，
    前端应接着轮询 ``GET /exams/reports/{id}`` 看 ``ai.ai_status``。

    **危急值在本接口当场给出**：``report.critical`` / ``critical_message`` 由服务端
    规则比对得出，与 AI 无关。命中时 HTTP 仍是 201（录入成功了），
    但响应体带告警文本且服务端已写 WARNING 日志——危急值的传达不能依赖
    一个可能失败的异步任务，也不能因为「AI 也在解读」就往后排。

    **AI 派发失败不回滚报告**：``dispatch.dispatched=false`` 且 ``reason`` 说明原因时，
    报告数据已经安全落库，AITask 标 failed、ai_status 标 failed，
    稍后可用 ``POST /exams/reports/{id}/interpret`` 手动补触发。
    医疗数据录入不能被 broker 故障卡死。

    异常旗标不接受调用方传入，一律由服务端按参考区间算出。
    """
    report, dispatch = await es.submit_report(
        db,
        _parse_uuid(request_id, "检查申请单"),
        lab_tech_id=current_user.id,
        raw_content=payload.raw_content,
        structured_data=payload.structured_data,
        conclusion=payload.conclusion,
        auto_dispatch=payload.auto_interpret,
    )
    info = await es.attach_request_info(db, [report])
    logger.info(
        "报告录入完成 operator=%s report_id=%s critical=%s dispatched=%s",
        current_user.id,
        report.id,
        bool(info and report.id in info and _critical_out(report.structured_data)),
        dispatch.dispatched,
    )
    return ReportSubmitResponse(
        report=to_report_out(report, info.get(report.id)),
        dispatch=to_dispatch_out(dispatch),
    )


# ==================== 报告读取 ====================


@router.get("/reports", response_model=ExamReportListResponse, summary="我的报告列表")
async def list_my_reports(
    db: DbSession,
    current_user: CanReadReport,
    patient_id: Annotated[
        str | None, Query(description="医生可按患者过滤；患者传了也会被忽略")
    ] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ExamReportListResponse:
    """患者只看自己的报告，医生可读全部（矩阵 exam_report:read）。

    行级隔离在服务层强制，越权的 patient_id 不会泄露他人数据。
    """
    rows, total = await es.list_reports(
        db,
        requester_user_id=current_user.id,
        role_name=get_role_name(current_user),
        patient_id=_parse_uuid(patient_id, "患者") if patient_id else None,
        page=page,
        page_size=page_size,
    )
    info = await es.attach_request_info(db, rows)
    return ExamReportListResponse(
        items=[to_report_out(r, info.get(r.id)) for r in rows],
        total=total,
        page=max(1, page),
        page_size=min(page_size, 100),
    )


@router.get(
    "/reports/all", response_model=ExamReportListResponse, summary="报告列表（检验科全域）"
)
async def list_all_reports(
    db: DbSession,
    current_user: CanReadAllReport,
    patient_id: Annotated[str | None, Query(description="按患者过滤")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ExamReportListResponse:
    """检验科的全域报告视图（exam_report:read_all）。

    与 ``/reports`` 分开两个端点而不是在一个端点里判角色：权限矩阵里
    read 与 read_all 是两个独立动作位，混在一个端点上就得靠 role_name 猜，
    而「靠角色名硬编码」正是 rbac_matrix 模块 docstring 要避免的事。
    """
    rows, total = await es.list_reports(
        db,
        requester_user_id=current_user.id,
        role_name=get_role_name(current_user),
        patient_id=_parse_uuid(patient_id, "患者") if patient_id else None,
        page=page,
        page_size=page_size,
    )
    info = await es.attach_request_info(db, rows)
    return ExamReportListResponse(
        items=[to_report_out(r, info.get(r.id)) for r in rows],
        total=total,
        page=max(1, page),
        page_size=min(page_size, 100),
    )


@router.get("/reports/{report_id}", response_model=ExamReportOut, summary="报告详情（含 AI 结果）")
async def exam_report_detail(
    report_id: str,
    db: DbSession,
    current_user: CanReadAnyReport,
) -> ExamReportOut:
    """单份报告全文 + AI 解读块。**前端轮询就看这个接口**。

    患者只能读自己的报告，越权返回 404（不确认他人记录是否存在）。
    ``ai.ai_status`` 的取值与前端该做什么，见模块 docstring 的交互模型四条规则。
    """
    report = await es.get_report_detail(
        db,
        _parse_uuid(report_id, "检查报告"),
        requester_user_id=current_user.id,
        role_name=get_role_name(current_user),
    )
    info = await es.attach_request_info(db, [report])
    return to_report_out(report, info.get(report.id))


# ==================== AI 解读触发 ====================


@router.post(
    "/reports/{report_id}/interpret",
    response_model=InterpretTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="手动触发/重试 AI 报告解读",
)
async def trigger_report_interpretation(
    report_id: str,
    payload: InterpretRequest,
    db: DbSession,
    current_user: CanInterpret,
) -> InterpretTriggerResponse:
    """医生或检验科对一份报告重新发起 AI 解读（失败重试的唯一人工入口）。

    202 而非 200：工作刚开始，结果仍要靠轮询 ``GET /exams/reports/{id}`` 拿。

    允许的起点 ``ai_status``：pending（worker 没接手，可重投）、failed（重试）、
    skipped（审核拦截后人工确认要重来）。
    - ``processing`` → 409 InvalidStateTransitionError：正在跑，重复提交会让两次
      回写交错覆盖 ai_* 字段
    - ``completed`` → 200 语义的幂等拒绝（``dispatched=false`` + reason），
      不烧第二次 LLM 配额，也不覆盖已有解读

    ``reason`` 只进日志不落业务表：重触发是低频动作，字段级留痕够用，
    不值得为此新增一张历史表。
    """
    result = await es.trigger_interpretation(
        db,
        _parse_uuid(report_id, "检查报告"),
        requester_user_id=current_user.id,
    )
    logger.info(
        "手动触发解读 user_id=%s report_id=%s reason=%s dispatched=%s",
        current_user.id,
        report_id,
        payload.reason,
        result.dispatched,
    )
    message = (
        "AI 解读任务已入队，请轮询 GET /exams/reports/{id} 查看 ai_status"
        if result.dispatched
        else (result.reason or "未能派发解读任务")
    )
    return InterpretTriggerResponse(
        report_id=str(report_id),
        dispatch=to_dispatch_out(result),
        message=message,
    )


__all__ = ["router"]
