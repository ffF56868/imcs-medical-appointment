import json
import math
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from app.database import get_session
from app.legacy.models import (
    AnswerFeedback,
    ChatMessage,
    RAGEvaluationRun,
    RAGQualityEvaluationRun,
    RAGRequestMetric,
    User,
)
from app.routers.auth import require_admin
from app.legacy.schemas import (
    MonitoringFailureRead,
    MonitoringFeedbackSnapshot,
    MonitoringPathMetric,
    MonitoringQualitySnapshot,
    MonitoringSummary,
)


router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
    dependencies=[Depends(require_admin)],
)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def is_in_window(value: datetime, window_started_at: datetime) -> bool:
    return as_utc(value) >= window_started_at


def parse_response_metadata(message: ChatMessage) -> dict:
    try:
        metadata = json.loads(message.response_metadata_json or "{}")
    except (TypeError, ValueError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def build_legacy_metrics(
    session: Session,
    window_started_at: datetime,
    tracked_message_ids: set[int],
) -> list[RAGRequestMetric]:
    """Expose successful answers created before request metrics were introduced."""
    messages = session.exec(
        select(ChatMessage)
        .where(ChatMessage.role == "assistant")
        .where(ChatMessage.created_at >= window_started_at.replace(tzinfo=None))
    ).all()
    legacy_metrics: list[RAGRequestMetric] = []
    for message in messages:
        if message.id in tracked_message_ids:
            continue
        metadata = parse_response_metadata(message)
        if not metadata or "latency_ms" not in metadata:
            continue
        processing_path = str(metadata.get("processing_path") or "legacy")
        if processing_path == "safety-keyword-guard":
            request_type = "safety_guard"
        elif processing_path == "vector-search-no-match":
            request_type = "no_match"
        else:
            request_type = "rag"
        legacy_metrics.append(
            RAGRequestMetric(
                id=None,
                user_id=message.user_id,
                assistant_message_id=message.id,
                endpoint="ask/stream",
                success=True,
                status_code=200,
                request_type=request_type,
                processing_path=processing_path,
                retrieved_count=int(metadata.get("retrieved_count") or 0),
                latency_ms=int(metadata.get("latency_ms") or 0),
                created_at=message.created_at,
            )
        )
    return legacy_metrics


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def build_path_metrics(metrics: list[RAGRequestMetric]) -> list[MonitoringPathMetric]:
    paths: dict[str, list[RAGRequestMetric]] = {}
    for metric in metrics:
        paths.setdefault(metric.processing_path or "unknown", []).append(metric)

    results: list[MonitoringPathMetric] = []
    for path, path_metrics in sorted(paths.items()):
        retrieval_attempts = [
            item
            for item in path_metrics
            if item.request_type in {"rag", "no_match"}
        ]
        hits = sum(item.retrieved_count > 0 for item in retrieval_attempts)
        results.append(
            MonitoringPathMetric(
                processing_path=path,
                request_count=len(path_metrics),
                success_count=sum(item.success for item in path_metrics),
                failure_count=sum(not item.success for item in path_metrics),
                average_latency_ms=round(
                    sum(item.latency_ms for item in path_metrics) / len(path_metrics),
                    1,
                ),
                retrieval_hit_rate=ratio(hits, len(retrieval_attempts)),
            )
        )
    return results


@router.get("/summary", response_model=MonitoringSummary)
def get_monitoring_summary(
    hours: int = Query(default=24, ge=1, le=24 * 30),
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    now = datetime.now(UTC)
    window_started_at = now - timedelta(hours=hours)
    stored_metrics = session.exec(select(RAGRequestMetric)).all()
    tracked_message_ids = {
        metric.assistant_message_id
        for metric in stored_metrics
        if metric.assistant_message_id is not None
    }
    metrics = [
        metric
        for metric in stored_metrics
        if is_in_window(metric.created_at, window_started_at)
    ]
    metrics.extend(
        build_legacy_metrics(session, window_started_at, tracked_message_ids)
    )
    metrics.sort(key=lambda item: as_utc(item.created_at))

    request_count = len(metrics)
    successful_count = sum(metric.success for metric in metrics)
    failed_count = request_count - successful_count
    retrieval_attempts = [
        metric
        for metric in metrics
        if metric.request_type in {"rag", "no_match"}
    ]
    retrieval_hit_count = sum(metric.retrieved_count > 0 for metric in retrieval_attempts)
    latencies = [metric.latency_ms for metric in metrics]
    latest_retrieval_run = session.exec(
        select(RAGEvaluationRun).order_by(RAGEvaluationRun.created_at.desc())
    ).first()
    latest_quality_run = session.exec(
        select(RAGQualityEvaluationRun)
        .order_by(RAGQualityEvaluationRun.created_at.desc())
    ).first()

    feedback_items = [
        item
        for item in session.exec(select(AnswerFeedback)).all()
        if is_in_window(item.created_at, window_started_at)
    ]
    helpful_count = sum(item.helpful for item in feedback_items)

    recent_failures = [
        MonitoringFailureRead(
            endpoint=metric.endpoint,
            status_code=metric.status_code,
            request_type=metric.request_type,
            processing_path=metric.processing_path,
            error_type=metric.error_type,
            latency_ms=metric.latency_ms,
            created_at=metric.created_at,
        )
        for metric in sorted(
            (item for item in metrics if not item.success),
            key=lambda item: as_utc(item.created_at),
            reverse=True,
        )[:10]
    ]

    return MonitoringSummary(
        window_hours=hours,
        window_started_at=window_started_at,
        generated_at=now,
        request_count=request_count,
        successful_request_count=successful_count,
        failed_request_count=failed_count,
        success_rate=round(successful_count / request_count, 3) if request_count else 0,
        failure_rate=round(failed_count / request_count, 3) if request_count else 0,
        retrieval_attempt_count=len(retrieval_attempts),
        retrieval_hit_count=retrieval_hit_count,
        retrieval_hit_rate=ratio(retrieval_hit_count, len(retrieval_attempts)),
        average_latency_ms=(round(sum(latencies) / len(latencies), 1) if latencies else None),
        p50_latency_ms=percentile(latencies, 0.5),
        p95_latency_ms=percentile(latencies, 0.95),
        quality=MonitoringQualitySnapshot(
            retrieval_pass_rate=latest_retrieval_run.pass_rate
            if latest_retrieval_run
            else None,
            answer_accuracy=latest_quality_run.answer_accuracy
            if latest_quality_run
            else None,
            citation_accuracy=latest_quality_run.citation_accuracy
            if latest_quality_run
            else None,
            refusal_accuracy=latest_quality_run.refusal_accuracy
            if latest_quality_run
            else None,
            evaluated_at=latest_quality_run.created_at if latest_quality_run else None,
        ),
        feedback=MonitoringFeedbackSnapshot(
            total_count=len(feedback_items),
            helpful_count=helpful_count,
            not_helpful_count=len(feedback_items) - helpful_count,
            helpful_rate=ratio(helpful_count, len(feedback_items)),
        ),
        path_metrics=build_path_metrics(metrics),
        recent_failures=recent_failures,
    )
