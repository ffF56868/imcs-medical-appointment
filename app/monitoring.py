from datetime import UTC, datetime

from sqlmodel import Session

from app.legacy.models import RAGRequestMetric


def record_request_metric(
    session: Session,
    *,
    user_id: int | None,
    assistant_message_id: int | None = None,
    endpoint: str,
    success: bool,
    status_code: int,
    request_type: str,
    processing_path: str,
    retrieved_count: int = 0,
    latency_ms: int = 0,
    error_type: str | None = None,
) -> RAGRequestMetric:
    """Stage one request metric; the caller commits it with its own transaction."""
    metric = RAGRequestMetric(
        user_id=user_id,
        assistant_message_id=assistant_message_id,
        endpoint=endpoint,
        success=success,
        status_code=status_code,
        request_type=request_type,
        processing_path=processing_path,
        retrieved_count=max(0, retrieved_count),
        latency_ms=max(0, latency_ms),
        error_type=error_type,
        created_at=datetime.now(UTC),
    )
    session.add(metric)
    return metric
