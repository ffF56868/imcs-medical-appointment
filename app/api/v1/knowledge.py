"""知识库路由：文档管理、索引重建派发与混合检索。

路由层保持薄：解析请求 → 调 ``knowledge_service`` → 组装响应。
业务规则（content_hash 口径、重复派发判定、状态流转）全部在 service 里，
这里出现的唯一判断是「UUID 形态是否合法」——那属于协议层校验，
非法 ID 应该在进 service 之前就变成 404/422，而不是让 ORM 抛错变 500。

**权限用 require_permission 而非 require_role**：知识库的读权限横跨医生、
药师、检验科三个角色，写权限只在管理员。按角色名硬编码会让「给某个新角色开读」
变成改代码，而权限矩阵里加一条就行。

**rebuild 与 vectorize 的关系**：本批次的 ``POST /rebuild`` 派发的是增量任务
（vectorize_knowledge），不是全量重建。全量重建（rebuild_all_vectors）会清掉整张
向量表并重新调用 embedding，代价是真金白银的 API 配额和数十分钟不可用窗口，
只应在换模型时由运维显式发起，不该做成一个普通按钮。接口名沿用 rebuild
是因为前端管理员的心智模型就是「点一下重建索引」，实际语义由返回的
task_type 字段如实告知。
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import DbSession, require_permission
from app.core.exceptions import ResourceNotFoundError
from app.models.ai import AITask
from app.models.auth import User
from app.models.knowledge import KnowledgeDocument
from app.schemas.knowledge import (
    DocumentCreateRequest,
    DocumentListResponse,
    DocumentOut,
    DocumentSummaryOut,
    DocumentUpdateRequest,
    RebuildAcceptedOut,
    RetrievalDiagnostics,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    TaskOut,
)
from app.services import knowledge_service as ks
from app.services.rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["知识库"])

# 各端点的鉴权依赖：集中定义避免每个函数里重复写工厂调用，
# 也保证「同一个动作在不同端点上权限一致」这件事只需看一处。
# require_permission 的返回值是当前用户对象，这里只把它当闸门用，
# 需要 user_id 的端点另外声明 CurrentUser（同一依赖会被 FastAPI 复用，不多查库）。
CanRead = Annotated[User, Depends(require_permission("knowledge", "read"))]
CanCreate = Annotated[User, Depends(require_permission("knowledge", "create"))]
CanDelete = Annotated[User, Depends(require_permission("knowledge", "delete"))]
CanRebuild = Annotated[User, Depends(require_permission("knowledge", "rebuild"))]


def _parse_uuid(value: str, what: str) -> uuid.UUID:
    """路径参数转 UUID，非法值报 404。

    为什么不用 Pydantic 的 UUID 类型自动校验（那样会是 422）：
    对客户端来说「ID 格式不对」和「这个 ID 不存在」是同一种失败——它拿不到资源。
    统一成 404 可以让前端的错误分支只处理一种情况，不必区分
    「我传了个脏值」和「这条记录被删了」。
    """
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise ResourceNotFoundError(f"{what}不存在") from exc


# ==================== 文档 CRUD ====================


@router.post(
    "/documents",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    summary="新建知识文档",
)
async def create_document(
    payload: DocumentCreateRequest,
    db: DbSession,
    current_user: CanCreate,
) -> DocumentOut:
    """以 JSON 纯文本创建知识文档，创建后进入 pending 等向量化。

    **不在本接口同步建索引**：一次 embedding 可能几十秒，放在请求里必超时；
    且失败无法重试。要立即索引请再调 POST /knowledge/rebuild。

    department_id 目前只做格式校验，不校验科室是否存在：外键会在 flush 时
    拦住真不存在的 ID，但那会变成 500。后续批次应补一次存在性查询给出 422。
    """
    department_id = (
        _parse_uuid(payload.department_id, "关联科室") if payload.department_id else None
    )
    doc = await ks.create_document(
        db,
        title=payload.title,
        content=payload.content,
        doc_type=payload.doc_type,
        source=payload.source,
        source_url=payload.source_url,
        source_tier=payload.source_tier,
        department_id=department_id,
        extra_metadata=payload.extra_metadata,
    )
    return to_document_out(doc)


@router.get("/documents", response_model=DocumentListResponse, summary="分页列出知识文档")
async def list_documents(
    db: DbSession,
    _: CanRead,
    keyword: Annotated[str | None, Query(max_length=200, description="标题/来源模糊匹配")] = None,
    doc_type: Annotated[str | None, Query(description="按资料类型过滤")] = None,
    index_status: Annotated[str | None, Query(description="pending/indexed/stale/failed")] = None,
    include_inactive: Annotated[bool, Query(description="是否包含已下线文档")] = False,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> DocumentListResponse:
    """分页列出文档摘要（不含正文）。

    正文检索请用 POST /knowledge/search：这里的 keyword 只匹配标题与来源，
    对 Text 列做 ILIKE 走不了索引，库灌满后会成为慢查询源头。
    """
    items, total = await ks.list_documents(
        db,
        keyword=keyword,
        doc_type=doc_type,
        index_status=index_status,
        include_inactive=include_inactive,
        page=page,
        page_size=page_size,
    )
    return DocumentListResponse(
        items=[to_summary(doc) for doc in items],
        total=total,
        page=page,
        page_size=min(page_size, 100),
    )


@router.get("/documents/{doc_id}", response_model=DocumentOut, summary="文档详情")
async def get_document(
    doc_id: str,
    db: DbSession,
    _: CanRead,
) -> DocumentOut:
    """取单篇文档全文与索引状态。"""
    doc = await ks.get_document(db, _parse_uuid(doc_id, "知识文档"))
    return to_document_out(doc)


@router.patch(
    "/documents/{doc_id}",
    response_model=DocumentOut,
    summary="更新知识文档",
)
async def update_document(
    doc_id: str,
    payload: DocumentUpdateRequest,
    db: DbSession,
    current_user: CanCreate,
) -> DocumentOut:
    """更新文档；正文或溯源字段变更会把索引状态置为 stale。

    权限复用 knowledge:create 而非 update：当前权限矩阵里只有 admin 同时持有
    create/update/delete，非 admin 角色一律只读，因此两者在本接口的实际效果相同。
    选 create 是因为它与 delete 同属「维护类」动作，语义上比 update 更贴近
    「改一篇资料」。若日后要给某个角色单独开改而不开放新建，
    把这里换成 knowledge:update 即可，无需动权限矩阵。
    """
    changes = payload.model_dump(exclude_unset=True)
    doc = await ks.update_document(db, _parse_uuid(doc_id, "知识文档"), **changes)
    return to_document_out(doc)


@router.delete(
    "/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除知识文档",
)
async def delete_document(
    doc_id: str,
    db: DbSession,
    current_user: CanDelete,
) -> None:
    """删除文档并连带清理切块与向量。

    用 204：资源已不存在，返回任何负载都没有消费方，而且前端此刻正在把这一行
    从列表里移除。
    """
    await ks.delete_document(db, _parse_uuid(doc_id, "知识文档"))


# ==================== 索引任务 ====================


@router.post(
    "/rebuild",
    response_model=RebuildAcceptedOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="触发知识库索引重建",
)
async def rebuild_index(
    db: DbSession,
    current_user: CanRebuild,
) -> RebuildAcceptedOut:
    """派发向量化任务，立即返回 task_id，不等待执行完成。

    **防并发派发**由 service 层的 advisory lock + 同型任务查重两道闸保证：
    单纯「查有没有 running 任务」存在检查—写入竞态窗口，管理员连点两次
    就能排进两个重建，而它们会交错执行「删旧切块再建新切块」，
    留下半套向量。已有任务在跑时本接口返回 409（DuplicateTaskError）。

    202 而非 200：工作刚开始，结果要靠 GET /knowledge/tasks/{id} 轮询。
    """
    task = await ks.start_vectorization(
        db,
        current_user.id,
        task_type=ks.TASK_TYPE_VECTORIZE,
        payload={"trigger": "manual_rebuild", "operator": current_user.username},
    )
    return RebuildAcceptedOut(
        task_id=str(task.id),
        task_type=task.task_type,
        status=task.status,
        celery_task_id=task.celery_task_id,
    )


@router.get("/tasks/{task_id}", response_model=TaskOut, summary="查询知识库任务状态")
async def get_task(task_id: str, db: DbSession, _: CanRead) -> TaskOut:
    """查询异步任务状态与结果统计。

    状态源是 ``ai_tasks`` 表而非 Celery result backend：后者只保留 1 小时，
    而管理员需要在隔天回看「上周那次重建到底成功没有」。
    """
    task = await ks.get_task(db, _parse_uuid(task_id, "任务"))
    return to_task_out(task)


# ==================== 检索 ====================


@router.post("/search", response_model=SearchResponse, summary="混合检索知识切块")
async def search(
    payload: SearchRequest,
    db: DbSession,
    current_user: CanRead,
) -> SearchResponse:
    """向量 + 全文两路召回后经 RRF 融合检索知识切块。

    **SQLite / 非 PG 环境返回零结果而非 500**：两路检索各自在非 PostgreSQL
    后端优雅降级为空列表（见 vector_search / fulltext_search 的降级策略），
    因此这里天然能返回一个空 results。这是刻意保留的行为——测试环境不该因为
    缺 pgvector 就报错，前端也能据此正常渲染「暂无匹配资料」。

    查询向量取自 LLM embed，失败时 retriever 内部已捕获并把向量路降级为空，
    全文路仍可用，所以 embedding 服务抖动不会让整个检索挂掉。
    """
    retriever = RAGRetriever()
    response = await retriever.retrieve(
        db,
        payload.query,
        rewritten_query=payload.rewritten_query,
        trace_id=payload.trace_id,
        user_id=current_user.id,
    )
    return SearchResponse(
        query=payload.query,
        results=[
            SearchResultItem(
                chunk_id=str(r.chunk_id),
                content=r.content,
                rrf_score=round(r.rrf_score, 6),
                title=r.title,
                doc_type=r.doc_type,
                source=r.source,
                source_tier=r.source_tier,
                page_number=r.page_number,
                page_range=r.page_range,
                extra_metadata=r.extra_metadata,
                route_details=r.route_details,
            )
            for r in response.results
        ],
        diagnostics=RetrievalDiagnostics(
            vector_hit_count=response.vector_hit_count,
            keyword_hit_count=response.keyword_hit_count,
            both_hit_count=response.both_hit_count,
            merged_count=response.merged_count,
            is_zero_result=response.is_zero_result,
        ),
    )


# ==================== 序列化 ====================


def to_summary(doc: KnowledgeDocument) -> DocumentSummaryOut:
    """文档 → 列表摘要（不含正文）。"""
    return DocumentSummaryOut(
        id=str(doc.id),
        title=doc.title,
        doc_type=doc.doc_type,
        source=doc.source,
        source_tier=doc.source_tier,
        index_status=doc.index_status,
        chunk_count=doc.chunk_count,
        is_reviewed=doc.is_reviewed,
        is_active=doc.is_active,
        content_length=len(doc.content or ""),
        created_at=doc.created_at,
        indexed_at=doc.indexed_at,
    )


def to_document_out(doc: KnowledgeDocument) -> DocumentOut:
    """文档 → 详情响应。"""
    return DocumentOut(
        **to_summary(doc).model_dump(),
        content=doc.content,
        content_hash=doc.content_hash,
        source_url=doc.source_url,
        department_id=str(doc.department_id) if doc.department_id is not None else None,
        published_at=doc.published_at,
        extra_metadata=doc.extra_metadata,
        updated_at=doc.updated_at,
    )


def to_task_out(task: AITask) -> TaskOut:
    """AITask → 状态响应。"""
    return TaskOut(
        id=str(task.id),
        task_type=task.task_type,
        status=task.status,
        progress=task.progress,
        progress_message=task.progress_message,
        retry_count=task.retry_count,
        error_message=task.error_message,
        result=task.result,
        celery_task_id=task.celery_task_id,
        started_at=task.started_at,
        completed_at=task.completed_at,
        duration_ms=task.duration_ms,
    )


__all__ = ["router"]
