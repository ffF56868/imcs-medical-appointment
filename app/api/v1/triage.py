"""智能分诊路由。

权限映射（rbac_matrix 为准，未改动矩阵）：
- POST /triage/assess → triage:create（患者持有；admin 依赖工厂放行）
- GET /triage/results、GET /triage/results/{id} → triage:read（患者、医生持有）
- POST /triage/batch-assess → triage:create + 仅 admin 角色（矩阵里没有 batch
  动作，批量派发会成倍消耗 LLM 配额与 token 预算，属管理面操作，
  用 require_role 收窄；患者单次评估走 assess 即可）

路由保持薄：校验参数 → triage_workflow.assess → 装配响应。
分诊后「可挂号源」摘要（next_actions 落地）由本路由调 appointment_service
拼出——两个业务模块的衔接只有这一处函数调用，不共享任何内部状态，
也不产生编译期依赖环（triage → appointment 单向）。
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import DbSession, get_role_name, require_permission, require_role
from app.core.exceptions import PermissionDeniedError, ResourceNotFoundError
from app.models.ai import AITask, TriageResult
from app.models.auth import User
from app.models.patient import PatientProfile
from app.schemas.appointment import ScheduleQuery
from app.schemas.triage import (
    BatchAssessRequest,
    BatchDispatchResponse,
    EvidenceOut,
    SuggestedSlot,
    TriageOut,
    TriageRequest,
    TriageResponse,
)
from app.services import appointment_service
from app.services.triage_workflow import TriageOutcome, assess, get_result, list_results

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/triage", tags=["智能分诊"])

CanCreate = Annotated[User, Depends(require_permission("triage", "create"))]
_CanReadDep = require_permission("triage", "read")
CanRead = Annotated[User, Depends(_CanReadDep)]
AdminUser = Annotated[User, Depends(require_role("admin"))]


def _parse_uuid(value: str, what: str) -> uuid.UUID:
    """路径/查询参数转 UUID，非法值 404（与 appointment.py 同一口径）。"""
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise ResourceNotFoundError(f"{what}不存在") from exc


async def _own_patient_id(db: AsyncSession, user_id: uuid.UUID) -> uuid.UUID | None:
    """当前登录用户的患者档案 id。分诊结果按 patient 归属而非 user 归属，
    与挂号链路保持同一主体（appointments.patient_id 指向档案而非账号）。"""
    row = await db.execute(
        select(PatientProfile.id).where(PatientProfile.user_id == user_id)
    )
    return row.scalar_one_or_none()


async def _suggest_slots(
    db: AsyncSession, outcome: TriageOutcome
) -> list[SuggestedSlot]:
    """分诊结论之后的「下一步」：推荐科室未来 14 天内最早的 3 个可约号源。

    只查摘要不返回分页：患者此刻要的是「马上能挂到谁的号」，
    完整号源浏览走 GET /schedules。查询失败（比如科室 code 为空）
    静默返回空列表——分诊结论本身已经给出，号源是锦上添花，
    不能让附赠功能把主响应炸掉。
    """
    if not outcome.department_code and not outcome.department_id:
        return []
    try:
        query = ScheduleQuery(
            department_id=str(outcome.department_id) if outcome.department_id else None,
            department_code=outcome.department_code,
            page_size=3,
        )
        rows, _total, _span = await appointment_service.list_available_schedules(db, query)
    except Exception:  # noqa: BLE001 - 附加信息失败不影响分诊主体
        logger.warning("分诊后续号源查询失败", exc_info=True)
        return []
    slots: list[SuggestedSlot] = []
    for r in rows[:3]:
        try:
            slots.append(
                SuggestedSlot(
                    schedule_id=str(r["id"]),
                    doctor_name=str(r.get("doctor_name") or ""),
                    doctor_title=r.get("doctor_title"),
                    schedule_date=str(r.get("schedule_date") or ""),
                    time_slot=str(r.get("time_slot") or ""),
                    remaining_slots=int(r.get("remaining_slots") or 0),
                    registration_fee=int(r.get("registration_fee") or 0),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return slots


def _to_response(
    outcome: TriageOutcome,
    suggested_slots: list[SuggestedSlot],
) -> TriageResponse:
    return TriageResponse(
        triage_id=str(outcome.triage_id),
        urgency=outcome.urgency,
        is_emergency=outcome.is_emergency,
        matched_emergency_keywords=outcome.matched_emergency_keywords,
        recommended_department_id=str(outcome.department_id) if outcome.department_id else None,
        recommended_department_name=outcome.department_name,
        recommended_department_code=outcome.department_code,
        confidence=outcome.confidence,
        reason=outcome.reason,
        advice=outcome.advice,
        decision_source=outcome.decision_source,
        degraded=outcome.degraded,
        extracted_symptoms=outcome.extracted,
        evidence=[
            EvidenceOut(
                title=e.title,
                source=e.source,
                source_tier=e.source_tier,
                snippet=e.snippet,
                rrf_score=e.rrf_score,
            )
            for e in outcome.evidence
        ],
        llm_model=outcome.llm_model,
        trace_id=outcome.trace_id,
        suggested_slots=suggested_slots,
    )


# ==================== 同步评估 ====================


@router.post(
    "/assess",
    response_model=TriageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="智能分诊评估（同步）",
)
async def triage_assess(
    payload: TriageRequest,
    db: DbSession,
    current_user: CanCreate,
) -> TriageResponse:
    """一次完整的分诊：急症拦截 → 症状提取 → 科室推荐 → RAG 证据 → 号源摘要。

    **响应 201 而非常见的 200**：本端点每次都新建一条 TriageResult 落库
    （分诊记录必须保留未转化的咨询——models/ai.py 注释理由），
    语义上就是资源创建。

    急症分支**不调用 LLM**：命中关键词直接给出急诊引导，
    模型不可用、预算耗尽、审核拦截都不允许延误急症患者。

    降级透明：degraded=True 时 advice 已带「以现场分诊为准」的免责提示，
    前端应展示提示条而不是静默呈现。
    """
    patient_id = await _own_patient_id(db, current_user.id)
    outcome = await assess(
        db,
        chief_complaint=payload.chief_complaint,
        user_id=current_user.id,
        patient_id=patient_id,
    )
    slots = await _suggest_slots(db, outcome) if payload.include_slots else []
    return _to_response(outcome, slots)


# ==================== 历史记录 ====================


@router.get("/results", response_model=list[TriageOut], summary="分诊历史（分页）")
async def triage_results(
    db: DbSession,
    current_user: CanRead,
    patient_id: Annotated[str | None, Query(description="医生/管理员按患者过滤")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[TriageOut]:
    """患者只看自己的记录，医生/管理员看全量或指定患者。

    行级隔离在服务层强制：患者传别人的 patient_id 会被忽略并按自己的档案过滤
    （见 triage_workflow.list_results），越权读取在数据层就不成立。
    """
    rows, _total = await list_results(
        db,
        user_id=current_user.id,
        role_name=get_role_name(current_user),
        patient_id=_parse_uuid(patient_id, "患者") if patient_id else None,
        page=page,
        page_size=page_size,
    )
    return [
        TriageOut(
            id=str(r.id),
            created_at=r.created_at,
            chief_complaint=r.chief_complaint,
            urgency=r.urgency,
            is_emergency=r.is_emergency,
            recommended_department_name=r.recommended_department_name,
            decision_source=r.decision_source,
            confidence=float(r.confidence) if r.confidence is not None else None,
            trace_id=r.trace_id,
        )
        for r in rows
    ]


@router.get("/results/{triage_id}", response_model=TriageResponse, summary="分诊记录详情")
async def triage_result_detail(
    triage_id: str,
    db: DbSession,
    current_user: CanRead,
) -> TriageResponse:
    """取单条分诊记录。患者只能读自己的（越权返回 404，与 appointment 同一口径）。"""
    record: TriageResult = await get_result(db, _parse_uuid(triage_id, "分诊记录"))
    role = get_role_name(current_user)
    if role == "patient":
        own = await _own_patient_id(db, current_user.id)
        if record.patient_id is None or record.patient_id != own:
            raise ResourceNotFoundError("分诊记录不存在")
    evidence: list[EvidenceOut] = []
    outcome_slots: list[SuggestedSlot] = []
    return TriageResponse(
        triage_id=str(record.id),
        urgency=record.urgency,
        is_emergency=record.is_emergency,
        matched_emergency_keywords=list(record.matched_emergency_keywords or []),
        recommended_department_id=(
            str(record.recommended_department_id) if record.recommended_department_id else None
        ),
        recommended_department_name=record.recommended_department_name,
        recommended_department_code=None,
        confidence=float(record.confidence) if record.confidence is not None else None,
        reason=record.decision_source,
        advice=record.advice,
        decision_source=record.decision_source,
        degraded=False,
        extracted_symptoms=record.extracted_symptoms,
        evidence=evidence,
        llm_model=record.llm_model,
        trace_id=record.trace_id,
        suggested_slots=outcome_slots,
    )


# ==================== 批量评估（异步派发） ====================


@router.post(
    "/batch-assess",
    response_model=BatchDispatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="批量分诊评估（管理员，异步）",
)
async def triage_batch_assess(
    payload: BatchAssessRequest,
    db: DbSession,
    current_user: CanCreate,
    _admin: AdminUser,
) -> BatchDispatchResponse:
    """规则分诊先落库、LLM 补偿异步跑——**先给出能用的结论，再提升质量**。

    为什么不直接同步循环调 LLM：50 条 × 2 节点链路的耗时不可控，
    HTTP 请求会挂几分钟；批量场景的正当需求是「评估历史咨询的科室分布」
    这类离线任务，规则结果立刻可查，AI 结论到货后覆盖（decision_source
    升级为 ai/hybrid，规则结论快照保留在 advice 前缀）。

    双依赖（CanCreate + AdminUser）：批量入口既要 triage:create 权限位，
    又只向 admin 角色开放——矩阵没有 batch 动作，用角色收窄替代，
    不动权限矩阵（矩阵一次改动牵动 RESOURCES/种子/测试三处，收益不成比例）。

    派发失败不回滚规则结果：规则分诊本身对患者/评估者已可用，
    Celery broker 故障时任务标记保持 pending，由后续补偿或人工重派处理。
    """
    from app.tasks.celery_app import dispatch_task

    triage_ids: list[uuid.UUID] = []
    for item in payload.items:
        outcome = await assess(
            db,
            chief_complaint=item.chief_complaint,
            user_id=current_user.id,
            patient_id=_parse_uuid(item.patient_id, "患者") if item.patient_id else None,
            rule_only=True,
        )
        triage_ids.append(outcome.triage_id)

    # total 是补偿进度收敛的分母：run_triage_workflow 每条完成后读 payload 里的
    # done 自增，done>=total 才把父任务标 completed。缺 total 会退化成按 1 计算，
    # 第一条补偿完就把整批标记完成——所以这里必须写真实条数。
    task = AITask(
        task_type="triage",
        ref_id=triage_ids[0] if triage_ids else None,
        status="pending",
        payload={
            "triage_ids": [str(t) for t in triage_ids],
            "source": "batch_assess",
            "total": len(triage_ids),
            "done": 0,
        },
    )
    db.add(task)
    await db.commit()

    dispatched = 0
    for tid in triage_ids:
        # 不显式传 queue：task_routes 已把 run_triage_workflow 固定路由到
        # interpret 队列（worker-online 消费）。在这里再传一个 queue 会覆盖
        # 路由表，破坏「分诊补偿与向量化物理隔离」的队列规划。
        result = dispatch_task(
            "app.tasks.ai_tasks.run_triage_workflow",
            str(tid),
            str(task.id),
        )
        if result is not None:
            dispatched += 1
    logger.info("批量分诊派发：入队 %d/%d 条（ai_task=%s）", dispatched, len(triage_ids), task.id)
    return BatchDispatchResponse(
        ai_task_id=str(task.id),
        queued=dispatched,
        message=(
            "规则分诊结果已可查，AI 结论异步补偿覆盖；"
            f"派发成功 {dispatched}/{len(triage_ids)} 条"
        ),
    )


__all__ = ["router"]
