from datetime import UTC, datetime
import os
from threading import Lock

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlmodel import Session, select

from app.access import accessible_documents_statement
from app.cache import invalidate_knowledge_status_cache
from app.database import engine, get_session
from app.knowledge_versions import (
    create_knowledge_snapshot,
    get_current_snapshot_hash,
    restore_snapshot_payload,
)
from app.legacy.models import (
    Condition,
    Drug,
    KnowledgeDocument,
    KnowledgeRebuildJob,
    KnowledgeReviewLog,
    KnowledgeSnapshot,
    User,
)
from app.routers.auth import get_current_user, require_admin
from app.legacy.schemas import (
    KnowledgeRebuildResponse,
    KnowledgeRebuildJobListResponse,
    KnowledgeRebuildJobRead,
    KnowledgeReviewBatchUpdate,
    KnowledgeReviewBatchUpdateResponse,
    KnowledgeReviewLogRead,
    KnowledgeReviewLogResponse,
    KnowledgeRestoreResponse,
    KnowledgeReviewItem,
    KnowledgeReviewResponse,
    KnowledgeSearchItem,
    KnowledgeSearchResponse,
    KnowledgeStatusResponse,
    KnowledgeVersionListResponse,
    KnowledgeVersionRead,
)
from app.source_metadata import get_source_review_reasons, needs_source_review
from app.vector_store import get_knowledge_status, rebuild_vector_store


router = APIRouter(
    prefix="/knowledge",
    tags=["knowledge"],
    dependencies=[Depends(get_current_user)],
)

ACTIVE_REBUILD_STATUSES = {"pending", "running"}
REBUILD_JOB_TIMEOUT_SECONDS = int(
    os.getenv("REBUILD_JOB_TIMEOUT_SECONDS", str(30 * 60))
)
rebuild_start_lock = Lock()
rebuild_execution_lock = Lock()


def find_matching_fields(query: str, fields: dict[str, str]) -> list[str]:
    normalized_query = query.casefold()
    return [
        field_name
        for field_name, value in fields.items()
        if normalized_query in value.casefold()
    ]


def build_excerpt(query: str, text: str, max_length: int = 240) -> str:
    normalized_text = text.casefold()
    match_start = normalized_text.find(query.casefold())
    if match_start == -1 or len(text) <= max_length:
        return text[:max_length]

    start = max(0, match_start - 80)
    end = min(len(text), start + max_length)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def build_review_item(
    record: Condition | Drug | KnowledgeDocument,
    record_type: str,
    title: str,
) -> KnowledgeReviewItem | None:
    review_reasons = get_source_review_reasons(
        record.source,
        record.source_tier,
        record.updated_at,
        record.source_url,
    )
    if not review_reasons:
        return None
    return KnowledgeReviewItem(
        type=record_type,
        record_id=record.id,
        title=title,
        source=record.source,
        source_url=record.source_url,
        source_tier=record.source_tier,
        updated_at=record.updated_at,
        review_reasons=review_reasons,
    )


@router.get("/status", response_model=KnowledgeStatusResponse)
def get_status(session: Session = Depends(get_session)):
    return get_knowledge_status(session)


@router.post("/rebuild", response_model=KnowledgeRebuildResponse)
def rebuild_knowledge(
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    snapshot, snapshot_created = create_knowledge_snapshot(session, "rebuild")
    session.commit()
    document_count, chunk_count = rebuild_vector_store(session)
    return KnowledgeRebuildResponse(
        message="知识库向量重建完成",
        document_count=document_count,
        chunk_count=chunk_count,
        snapshot_id=snapshot.id,
        snapshot_created=snapshot_created,
    )


def get_active_rebuild_job(session: Session) -> KnowledgeRebuildJob | None:
    return session.exec(
        select(KnowledgeRebuildJob)
        .where(KnowledgeRebuildJob.status.in_(ACTIVE_REBUILD_STATUSES))
        .order_by(KnowledgeRebuildJob.created_at.desc())
    ).first()


def recover_stale_rebuild_jobs(session: Session) -> None:
    """Mark jobs interrupted by a crash or an unusually long rebuild as failed."""
    now = datetime.now(UTC)
    stale_jobs = session.exec(
        select(KnowledgeRebuildJob).where(
            KnowledgeRebuildJob.status.in_(ACTIVE_REBUILD_STATUSES)
        )
    ).all()
    changed = False
    timeout_minutes = max(1, REBUILD_JOB_TIMEOUT_SECONDS // 60)
    for job in stale_jobs:
        reference_time = job.started_at or job.created_at
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=UTC)
        if (now - reference_time).total_seconds() <= REBUILD_JOB_TIMEOUT_SECONDS:
            continue
        job.status = "failed"
        job.error_message = (
            f"任务超过 {timeout_minutes} 分钟未完成，可能因服务重启中断，请重试"
        )
        job.completed_at = now
        session.add(job)
        changed = True
    if changed:
        session.commit()


def run_rebuild_job(job_id: int) -> None:
    """Run one rebuild outside the request and persist its final state."""
    with rebuild_execution_lock:
        with Session(engine) as session:
            job = session.get(KnowledgeRebuildJob, job_id)
            if job is None or job.status not in ACTIVE_REBUILD_STATUSES:
                return

            job.status = "running"
            job.started_at = datetime.now(UTC)
            session.add(job)
            session.commit()

            try:
                create_knowledge_snapshot(session, "rebuild")
                session.commit()
                document_count, chunk_count = rebuild_vector_store(session)
                session.refresh(job)
                if job.status != "running":
                    return

                job.status = "completed"
                job.document_count = document_count
                job.chunk_count = chunk_count
                job.completed_at = datetime.now(UTC)
                session.add(job)
                session.commit()
            except Exception as error:
                session.rollback()
                with Session(engine) as failed_session:
                    failed_job = failed_session.get(KnowledgeRebuildJob, job_id)
                    if failed_job is not None:
                        failed_job.status = "failed"
                        failed_job.error_message = str(error)[:2000]
                        failed_job.completed_at = datetime.now(UTC)
                        failed_session.add(failed_job)
                        failed_session.commit()


@router.post(
    "/rebuild/async",
    response_model=KnowledgeRebuildJobRead,
    status_code=202,
)
def start_async_rebuild(
    background_tasks: BackgroundTasks,
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    with rebuild_start_lock:
        recover_stale_rebuild_jobs(session)
        active_job = get_active_rebuild_job(session)
        if active_job is not None:
            raise HTTPException(
                status_code=409,
                detail=f"已有知识库重建任务 #{active_job.id} 正在执行，请先查看它的状态",
            )

        job = KnowledgeRebuildJob()
        session.add(job)
        session.commit()
        session.refresh(job)
        background_tasks.add_task(run_rebuild_job, job.id)
        return job


@router.get(
    "/rebuild/jobs/active",
    response_model=KnowledgeRebuildJobRead | None,
)
def get_active_rebuild_job_status(
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    recover_stale_rebuild_jobs(session)
    job = get_active_rebuild_job(session)
    return job


@router.get(
    "/rebuild/jobs",
    response_model=KnowledgeRebuildJobListResponse,
)
def list_rebuild_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    recover_stale_rebuild_jobs(session)
    jobs = session.exec(
        select(KnowledgeRebuildJob)
        .order_by(KnowledgeRebuildJob.created_at.desc(), KnowledgeRebuildJob.id.desc())
        .limit(limit)
    ).all()
    total_count = len(session.exec(select(KnowledgeRebuildJob)).all())
    return KnowledgeRebuildJobListResponse(total_count=total_count, jobs=jobs)


@router.post(
    "/rebuild/jobs/{job_id}/retry",
    response_model=KnowledgeRebuildJobRead,
    status_code=202,
)
def retry_rebuild_job(
    job_id: int,
    background_tasks: BackgroundTasks,
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    with rebuild_start_lock:
        recover_stale_rebuild_jobs(session)
        source_job = session.get(KnowledgeRebuildJob, job_id)
        if source_job is None:
            raise HTTPException(status_code=404, detail="知识库重建任务不存在")
        if source_job.status != "failed":
            raise HTTPException(
                status_code=409,
                detail="只有失败的知识库重建任务可以重试",
            )
        active_job = get_active_rebuild_job(session)
        if active_job is not None:
            raise HTTPException(
                status_code=409,
                detail=f"已有知识库重建任务 #{active_job.id} 正在执行，请先查看它的状态",
            )

        retry_job = KnowledgeRebuildJob(retry_of_job_id=source_job.id)
        session.add(retry_job)
        session.commit()
        session.refresh(retry_job)
        background_tasks.add_task(run_rebuild_job, retry_job.id)
        return retry_job


@router.get(
    "/rebuild/jobs/{job_id}",
    response_model=KnowledgeRebuildJobRead,
)
def get_rebuild_job_status(
    job_id: int,
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    recover_stale_rebuild_jobs(session)
    job = session.get(KnowledgeRebuildJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="知识库重建任务不存在")
    return job


@router.get("/versions", response_model=KnowledgeVersionListResponse)
def list_knowledge_versions(
    limit: int = Query(default=20, ge=1, le=100),
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    current_hash = get_current_snapshot_hash(session)
    snapshots = session.exec(
        select(KnowledgeSnapshot)
        .order_by(KnowledgeSnapshot.created_at.desc())
        .limit(limit)
    ).all()
    total_count = len(session.exec(select(KnowledgeSnapshot)).all())
    return KnowledgeVersionListResponse(
        total_count=total_count,
        versions=[
            KnowledgeVersionRead(
                id=snapshot.id,
                document_count=snapshot.document_count,
                reason=snapshot.reason,
                created_at=snapshot.created_at,
                is_current=snapshot.content_hash == current_hash,
            )
            for snapshot in snapshots
        ],
    )


@router.post(
    "/versions/{snapshot_id}/restore",
    response_model=KnowledgeRestoreResponse,
)
def restore_knowledge_version(
    snapshot_id: int,
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    snapshot = session.get(KnowledgeSnapshot, snapshot_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="知识库版本不存在")

    backup_snapshot, _ = create_knowledge_snapshot(session, "pre_restore")
    session.commit()
    try:
        restore_snapshot_payload(session, snapshot)
        document_count, chunk_count = rebuild_vector_store(session)
    except ValueError as error:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        session.rollback()
        raise HTTPException(
            status_code=503,
            detail="恢复后重建向量库失败，当前版本已保留，可重试恢复",
        ) from error

    invalidate_knowledge_status_cache()

    return KnowledgeRestoreResponse(
        message="已恢复知识库版本，并完成向量重建",
        restored_version_id=snapshot.id,
        backup_version_id=backup_snapshot.id,
        document_count=document_count,
        chunk_count=chunk_count,
    )


@router.get("/review-queue", response_model=KnowledgeReviewResponse)
def get_review_queue(
    limit: int = Query(default=100, ge=1, le=500),
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    """List sources that still need human verification or refresh."""
    items: list[KnowledgeReviewItem] = []
    record_sets = (
        ("condition", session.exec(select(Condition).order_by(Condition.id)).all()),
        ("drug", session.exec(select(Drug).order_by(Drug.id)).all()),
        (
            "document",
            session.exec(
                select(KnowledgeDocument).order_by(KnowledgeDocument.id)
            ).all(),
        ),
    )
    for record_type, records in record_sets:
        for record in records:
            title = record.name if record_type != "document" else record.title
            item = build_review_item(record, record_type, title)
            if item is not None:
                items.append(item)

    return KnowledgeReviewResponse(
        total_count=len(items),
        results=items[:limit],
    )


@router.post(
    "/review-queue/batch-update",
    response_model=KnowledgeReviewBatchUpdateResponse,
)
def batch_update_review_metadata(
    update_data: KnowledgeReviewBatchUpdate,
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    """Apply one verified source record to several selected knowledge items."""
    model_by_type = {
        "condition": Condition,
        "drug": Drug,
        "document": KnowledgeDocument,
    }
    updated_records: list[tuple[str, Condition | Drug | KnowledgeDocument]] = []
    for target in update_data.targets:
        model = model_by_type[target.type]
        record = session.get(model, target.record_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"待审核资料不存在：{target.type} #{target.record_id}",
            )
        record.source = update_data.source
        record.source_url = update_data.source_url
        record.source_tier = update_data.source_tier
        record.updated_at = datetime.now(UTC)
        session.add(record)
        updated_records.append((target.type, record))

    for record_type, record in updated_records:
        title = record.name if record_type != "document" else record.title
        session.add(
            KnowledgeReviewLog(
                record_type=record_type,
                record_id=record.id,
                record_title=title,
                source=update_data.source,
                source_url=update_data.source_url,
                source_tier=update_data.source_tier,
            )
        )
    session.commit()
    invalidate_knowledge_status_cache()
    items: list[KnowledgeReviewItem] = []
    for record_type, record in updated_records:
        session.refresh(record)
        title = record.name if record_type != "document" else record.title
        items.append(
            KnowledgeReviewItem(
                type=record_type,
                record_id=record.id,
                title=title,
                source=record.source,
                source_url=record.source_url,
                source_tier=record.source_tier,
                updated_at=record.updated_at,
                review_reasons=get_source_review_reasons(
                    record.source,
                    record.source_tier,
                    record.updated_at,
                    record.source_url,
                ),
            )
        )
    return KnowledgeReviewBatchUpdateResponse(
        updated_count=len(items),
        items=items,
    )


@router.get("/review-logs", response_model=KnowledgeReviewLogResponse)
def list_review_logs(
    limit: int = Query(default=20, ge=1, le=100),
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    """Return recent source review actions for traceability."""
    logs = session.exec(
        select(KnowledgeReviewLog)
        .order_by(KnowledgeReviewLog.created_at.desc(), KnowledgeReviewLog.id.desc())
        .limit(limit)
    ).all()
    total_count = len(session.exec(select(KnowledgeReviewLog)).all())
    return KnowledgeReviewLogResponse(
        total_count=total_count,
        logs=[KnowledgeReviewLogRead.model_validate(log) for log in logs],
    )


@router.get("/search", response_model=KnowledgeSearchResponse)
def search_knowledge(
    q: str = Query(min_length=1, max_length=100),
    limit: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    query = q.strip()
    if not query:
        raise HTTPException(status_code=422, detail="搜索关键词不能为空")

    results: list[KnowledgeSearchItem] = []

    for condition in session.exec(select(Condition).order_by(Condition.id)).all():
        fields = {
            "名称": condition.name,
            "症状": condition.symptoms,
            "处理建议": condition.treatment,
        }
        matched_fields = find_matching_fields(query, fields)
        if matched_fields:
            results.append(
                KnowledgeSearchItem(
                    type="condition",
                    record_id=condition.id,
                    title=condition.name,
                    source=condition.source,
                    source_url=condition.source_url,
                    source_tier=condition.source_tier,
                    updated_at=condition.updated_at,
                    needs_review=needs_source_review(
                        condition.source_tier,
                        condition.updated_at,
                        condition.source,
                        condition.source_url,
                    ),
                    matched_fields=matched_fields,
                    excerpt=build_excerpt(query, "\n".join(fields.values())),
                )
            )

    for drug in session.exec(select(Drug).order_by(Drug.id)).all():
        fields = {
            "名称": drug.name,
            "作用": drug.effects,
            "使用说明": drug.instructions,
        }
        matched_fields = find_matching_fields(query, fields)
        if matched_fields:
            results.append(
                KnowledgeSearchItem(
                    type="drug",
                    record_id=drug.id,
                    title=drug.name,
                    source=drug.source,
                    source_url=drug.source_url,
                    source_tier=drug.source_tier,
                    updated_at=drug.updated_at,
                    needs_review=needs_source_review(
                        drug.source_tier,
                        drug.updated_at,
                        drug.source,
                        drug.source_url,
                    ),
                    matched_fields=matched_fields,
                    excerpt=build_excerpt(query, "\n".join(fields.values())),
                )
            )

    for document in session.exec(
        accessible_documents_statement(current_user).order_by(KnowledgeDocument.id)
    ).all():
        fields = {
            "标题": document.title,
            "内容": document.content,
            "来源": document.source,
        }
        matched_fields = find_matching_fields(query, fields)
        if matched_fields:
            results.append(
                KnowledgeSearchItem(
                    type="document",
                    record_id=document.id,
                    title=document.title,
                    source=document.source,
                    source_url=document.source_url,
                    source_tier=document.source_tier,
                    updated_at=document.updated_at,
                    needs_review=needs_source_review(
                        document.source_tier,
                        document.updated_at,
                        document.source,
                        document.source_url,
                    ),
                    matched_fields=matched_fields,
                    excerpt=build_excerpt(query, "\n".join(fields.values())),
                )
            )

    limited_results = results[:limit]
    return KnowledgeSearchResponse(
        query=query,
        total_count=len(results),
        results=limited_results,
    )
