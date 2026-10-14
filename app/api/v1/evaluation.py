"""评测管理路由：题库 CRUD、同步评测触发、RAGAS 派发与运行查询（阶段 6B）。

============================ 权限映射（rbac_matrix 为准，未改动矩阵） ============================

+------------------------------------------------+--------------------------------+------+
| 端点                                           | require_permission             | 角色 |
+------------------------------------------------+--------------------------------+------+
| GET    /evaluation/cases                       | evaluation:read_all            | ADMIN|
| POST   /evaluation/cases                       | evaluation:create              | ADMIN|
| PUT    /evaluation/cases/{id}                  | evaluation:update              | ADMIN|
| DELETE /evaluation/cases/{id}                  | evaluation:delete              | ADMIN|
| POST   /evaluation/runs                        | evaluation:create              | ADMIN|
| POST   /evaluation/ragas                       | evaluation:create              | ADMIN|
| GET    /evaluation/runs                        | evaluation:read_all            | ADMIN|
| GET    /evaluation/runs/{id}                   | evaluation:read_all            | ADMIN|
| GET    /evaluation/trends                      | evaluation:read_all            | ADMIN|
+------------------------------------------------+--------------------------------+------+

矩阵里 ``evaluation`` 的四个动作（create/read_all/update/delete）全部只给
ADMIN（见 rbac_matrix.py:168-171），本模块没有"任一"型组合门：评测管理是
单一角色的领地，不需要像 prescription/billing 那样为 read 与 read_all 开双路。
**所有资源都是平台级资产**（题库、指标、成本），不属于任何患者或医生，
因此这里没有行级隔离——隔离的对象是"角色"，由 RBAC 单层把关即可。

**为什么触发评测要挂 create 而不是 read_all**：一次 retrieval 评测会打几十次
RRF 检索，一次 RAGAS 更是要为每题生成回答、直接烧 LLM 配额——它是"发起执行"
的动作，与"查看结果"（read_all）不同授权位。

**为什么评测同步接口敢在请求里跑完**：检索层评测每题只有一次本地 RRF 查询
（LIMIT 20 索引查询）、零 LLM，几十题毫秒级；而 RAGAS 每题都要生成回答，
耗时不可控，所以走 Celery（返回 202 + 任务 ID，前端轮询 runs/{id}）。
两类接口的"能不能同步"取决于单题成本量级，不取决于"看起来像不像批处理"。
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, DbSession, get_role_name, require_permission
from app.core.exceptions import ResourceNotFoundError
from app.models.auth import User
from app.models.evaluation import EvalCase
from app.schemas.evaluation import (
    EvalCaseCreateRequest,
    EvalCaseListResponse,
    EvalCaseOut,
    EvalCaseUpdateRequest,
    EvalRunListResponse,
    EvalRunOut,
    EvalTrendResponse,
    RagasDispatchRequest,
    RagasDispatchResponse,
)
from app.services import evaluation_service as svc
from app.services.rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["AI评测"])

_PermissionDeps: dict[tuple[str, str], Any] = {}


def _gate(resource: str, action: str):
    key = (resource, action)
    if key not in _PermissionDeps:
        _PermissionDeps[key] = require_permission(resource, action)
    return _PermissionDeps[key]


CanCreateEval = Annotated[User, Depends(_gate("evaluation", "create"))]
CanReadEval = Annotated[User, Depends(_gate("evaluation", "read_all"))]
CanUpdateEval = Annotated[User, Depends(_gate("evaluation", "update"))]
CanDeleteEval = Annotated[User, Depends(_gate("evaluation", "delete"))]


def _parse_uuid(value: str, what: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise ResourceNotFoundError(f"{what}不存在") from exc


def _case_out(case) -> EvalCaseOut:
    return EvalCaseOut.model_validate(case)


def _run_out(run, *, with_results: bool = False) -> EvalRunOut:
    payload: dict[str, Any] = {
        "id": run.id,
        "kind": run.kind,
        "status": run.status,
        "total_count": run.total_count,
        "completed_count": run.completed_count,
        "top1_accuracy": _f(run.top1_accuracy),
        "recall_at_3": _f(run.recall_at_3),
        "precision_at_3": _f(run.precision_at_3),
        "answer_accuracy": _f(run.answer_accuracy),
        "citation_accuracy": _f(run.citation_accuracy),
        "refusal_accuracy": _f(run.refusal_accuracy),
        "faithfulness": _f(run.faithfulness),
        "answer_relevancy": _f(run.answer_relevancy),
        "context_precision": _f(run.context_precision),
        "context_recall": _f(run.context_recall),
        "metrics": run.metrics,
        "results": run.results if with_results else None,
        "retrieval_config": run.retrieval_config,
        "knowledge_hash": run.knowledge_hash,
        "knowledge_document_count": run.knowledge_document_count,
        "error_message": run.error_message,
        "ai_task_id": run.ai_task_id,
        "triggered_by": run.triggered_by,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
    }
    return EvalRunOut.model_validate(payload)


def _f(value: Any) -> float | None:
    return float(value) if value is not None else None


# ==================== 题库 CRUD ====================


@router.get("/cases", response_model=EvalCaseListResponse, summary="评测题列表")
async def list_cases(
    db: DbSession,
    current_user: CanReadEval,
    category: Annotated[str | None, Query(description="按分类过滤")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 50,
) -> EvalCaseListResponse:
    """列出题库。管理员维护评测集、前端展示覆盖率都看这里。"""
    rows = await svc.list_cases(db, category=category)
    size = max(1, min(page_size, 100))
    current = max(1, page)
    start = (current - 1) * size
    window = rows[start : start + size]
    return EvalCaseListResponse(
        items=[_case_out(c) for c in window],
        total=len(rows),
        page=current,
        page_size=size,
    )


@router.post(
    "/cases",
    response_model=EvalCaseOut,
    status_code=status.HTTP_201_CREATED,
    summary="新增评测题",
)
async def create_case(
    payload: EvalCaseCreateRequest,
    db: DbSession,
    current_user: CanCreateEval,
) -> EvalCaseOut:
    """登记一道评测题。

    **expected_name 必须能对上知识库 title**：命中判定是逐字比对
    （含 alternative_names 并集），对不上的题永远判负，会把指标拖成
    "检索很烂"的假象——而真相只是题库和数据没对齐。
    """
    case = EvalCase(
        question=payload.question,
        expected_name=payload.expected_name,
        expected_type=payload.expected_type,
        category=payload.category or "未分类",
        alternative_names=payload.alternative_names,
        answer_keywords=payload.answer_keywords,
        citation_names=payload.citation_names,
        expected_refusal=payload.expected_refusal,
    )
    db.add(case)
    await db.commit()
    await db.refresh(case)
    logger.info("评测题已创建 id=%s operator=%s", case.id, current_user.id)
    return _case_out(case)


@router.put("/cases/{case_id}", response_model=EvalCaseOut, summary="修改评测题")
async def update_case(
    case_id: str,
    payload: EvalCaseUpdateRequest,
    db: DbSession,
    current_user: CanUpdateEval,
) -> EvalCaseOut:
    """改题。只更新显式传入的字段（None 一律跳过）。"""
    case = await db.get(EvalCase, _parse_uuid(case_id, "评测题"))
    if case is None:
        raise ResourceNotFoundError("评测题不存在")
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(case, key, value)
    await db.commit()
    await db.refresh(case)
    return _case_out(case)


@router.delete(
    "/cases/{case_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除评测题",
)
async def delete_case(
    case_id: str,
    db: DbSession,
    current_user: CanDeleteEval,
) -> None:
    case = await db.get(EvalCase, _parse_uuid(case_id, "评测题"))
    if case is None:
        raise ResourceNotFoundError("评测题不存在")
    await db.delete(case)
    await db.commit()
    logger.info("评测题已删除 id=%s operator=%s", case_id, current_user.id)


# ==================== 同步评测（检索 / 质量） ====================


@router.post(
    "/runs",
    response_model=EvalRunOut,
    status_code=status.HTTP_201_CREATED,
    summary="触发同步评测（retrieval / quality）",
)
async def trigger_evaluation(
    db: DbSession,
    current_user: CanCreateEval,
    kind: Annotated[
        str, Query(description="retrieval（检索三指标）或 quality（回答质量）")
    ],
    category: Annotated[str | None, Query(description="只评某一分类")] = None,
) -> EvalRunOut:
    """在请求内跑完一次确定性评测并返回指标。

    为什么允许同步：这两种评测零 LLM、每题一次 RRF 查询（LIMIT 20 的
    索引查询，毫秒级），题库几十题总耗时以秒计，不值得异步化。
    反过来 RAGAS 每题都要生成回答、耗时不可控，见 /ragas 走 Celery。
    同步/异步的判据是**单题成本量级**，不是"看起来像不像批处理任务"。
    """
    if kind not in ("retrieval", "quality"):
        raise ResourceNotFoundError(
            "kind 仅支持 retrieval / quality；RAGAS 请走 POST /evaluation/ragas"
        )
    retriever = RAGRetriever()
    if kind == "retrieval":
        run = await svc.run_retrieval_evaluation(
            db, retriever=retriever, triggered_by=current_user.id, category=category
        )
    else:
        run = await svc.run_quality_evaluation(
            db, retriever=retriever, triggered_by=current_user.id, category=category
        )
    return _run_out(run, with_results=True)


# ==================== RAGAS（异步） ====================


@router.post(
    "/ragas",
    response_model=RagasDispatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="触发 RAGAS 异步评测",
)
async def trigger_ragas(
    payload: RagasDispatchRequest,
    db: DbSession,
    current_user: CanCreateEval,
) -> RagasDispatchResponse:
    """建运行记录并派发 Celery 任务，立即返回。

    返回 202 + run/ai_task 两个 ID：评测要跑数分钟到数十分钟，
    同步等待只会让 HTTP 超时、而后台其实在跑。前端轮询
    GET /evaluation/runs/{id} 拿终态。
    已有活跃 RAGAS 任务时服务层抛 409（评测烧配额，双跑即双倍成本）。
    """
    run, task = await svc.dispatch_ragas_evaluation(
        db, sample_size=payload.sample_size, triggered_by=current_user.id
    )
    return RagasDispatchResponse(
        eval_run_id=str(run.id),
        ai_task_id=str(task.id),
        status=run.status,
        sample_size=run.total_count,
        message="RAGAS 评测已排队，请轮询运行状态",
    )


# ==================== 运行查询与趋势 ====================


@router.get("/runs", response_model=EvalRunListResponse, summary="评测运行列表")
async def list_runs(
    db: DbSession,
    current_user: CanReadEval,
    kind: Annotated[str | None, Query(description="retrieval/quality/ragas")] = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> EvalRunListResponse:
    """历史运行列表（默认不含逐题明细，控制响应体积）。"""
    rows, total = await svc.list_runs(
        db, kind=kind, status=status_filter, page=page, page_size=page_size
    )
    return EvalRunListResponse(
        items=[_run_out(r) for r in rows],
        total=total,
        page=max(1, page),
        page_size=min(page_size, 100),
    )


@router.get("/runs/{run_id}", response_model=EvalRunOut, summary="评测运行详情（含明细）")
async def get_run(
    run_id: str,
    db: DbSession,
    current_user: CanReadEval,
) -> EvalRunOut:
    """单条运行：指标 + metrics.gate（回归门禁）+ 逐题明细。

    排障路径：先在列表里发现某次 recall_at_3 掉了 → 进详情看 gate 认定
    warning 还是 attention → 顺着 results 定位到具体是哪一类、哪一题失手。
    """
    run = await svc.get_run(db, _parse_uuid(run_id, "评测运行"))
    return _run_out(run, with_results=True)


@router.get("/trends", response_model=EvalTrendResponse, summary="指标回归趋势")
async def get_trend(
    db: DbSession,
    current_user: CanReadEval,
    kind: Annotated[str, Query(description="评测类型")],
    limit: Annotated[int, Query(ge=2, le=100)] = 20,
) -> EvalTrendResponse:
    """同类型历史运行的指标序列 + 基线。

    单点门禁只能发现突变，趋势线才能发现缓慢劣化（知识库一点点膨胀、
    或某次分块参数调整后指标悄悄往下走）。
    """
    data = await svc.run_trend(db, kind, limit=limit)
    return EvalTrendResponse(**data)


__all__ = ["router"]
