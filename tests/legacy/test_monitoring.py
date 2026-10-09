from datetime import UTC, datetime, timedelta

from sqlmodel import Session

from app.legacy.models import (
    AnswerFeedback,
    RAGEvaluationRun,
    RAGQualityEvaluationRun,
    RAGRequestMetric,
)


def test_monitoring_summary_combines_runtime_quality_and_feedback(
    client,
    test_engine,
):
    now = datetime.now(UTC)
    with Session(test_engine) as session:
        session.add_all(
            [
                RAGRequestMetric(
                    endpoint="ask/stream",
                    success=True,
                    status_code=200,
                    request_type="rag",
                    processing_path="rag-hybrid-rerank",
                    retrieved_count=2,
                    latency_ms=100,
                    created_at=now - timedelta(minutes=3),
                ),
                RAGRequestMetric(
                    endpoint="ask/stream",
                    success=True,
                    status_code=200,
                    request_type="no_match",
                    processing_path="vector-search-no-match",
                    retrieved_count=0,
                    latency_ms=200,
                    created_at=now - timedelta(minutes=2),
                ),
                RAGRequestMetric(
                    endpoint="ask/stream",
                    success=False,
                    status_code=503,
                    request_type="rag",
                    processing_path="generation-error",
                    retrieved_count=1,
                    latency_ms=300,
                    error_type="generation_error",
                    created_at=now - timedelta(minutes=1),
                ),
                AnswerFeedback(
                    assistant_message_id=101,
                    helpful=True,
                    created_at=now - timedelta(minutes=2),
                ),
                AnswerFeedback(
                    assistant_message_id=102,
                    helpful=False,
                    reason="引用不准确",
                    created_at=now - timedelta(minutes=1),
                ),
                RAGEvaluationRun(
                    total_count=4,
                    passed_count=3,
                    pass_rate=0.75,
                    preset_count=4,
                    custom_count=0,
                    knowledge_document_count=49,
                ),
                RAGQualityEvaluationRun(
                    total_count=4,
                    answer_correct_count=3,
                    answer_accuracy=0.75,
                    citation_correct_count=2,
                    citation_accuracy=0.5,
                    refusal_correct_count=4,
                    refusal_accuracy=1.0,
                    refusal_expected_count=1,
                    refusal_observed_count=1,
                    refusal_rate=0.25,
                    knowledge_document_count=49,
                ),
            ]
        )
        session.commit()

    response = client.get("/monitoring/summary?hours=24")

    assert response.status_code == 200
    data = response.json()
    assert data["request_count"] == 3
    assert data["successful_request_count"] == 2
    assert data["failed_request_count"] == 1
    assert data["failure_rate"] == 0.333
    assert data["retrieval_attempt_count"] == 3
    assert data["retrieval_hit_count"] == 2
    assert data["retrieval_hit_rate"] == 0.667
    assert data["average_latency_ms"] == 200.0
    assert data["p50_latency_ms"] == 200
    assert data["p95_latency_ms"] == 300
    assert data["quality"]["retrieval_pass_rate"] == 0.75
    assert data["quality"]["answer_accuracy"] == 0.75
    assert data["quality"]["citation_accuracy"] == 0.5
    assert data["feedback"]["helpful_rate"] == 0.5
    assert data["recent_failures"][0]["error_type"] == "generation_error"


def test_monitoring_summary_rejects_non_admin(auth_client):
    response = auth_client.get("/monitoring/summary")

    assert response.status_code == 401

