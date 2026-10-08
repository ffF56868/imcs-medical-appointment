from langchain_core.documents import Document
from sqlmodel import Session

from app.legacy.models import Drug
from app.routers import evaluation as evaluation_router
from app.legacy.schemas import RetrievalStrategyResult


class IrrelevantVectorStore:
    def similarity_search_with_relevance_scores(self, _query, k):
        assert k == 8
        return [
            (
                Document(
                    page_content="与测试药物无关的资料。",
                    metadata={
                        "name": "无关资料",
                        "type": "document",
                        "record_id": 99,
                    },
                ),
                0.9,
            )
        ]


def test_retrieval_evaluation_uses_the_live_hybrid_pipeline(
    client,
    test_engine,
    monkeypatch,
):
    case = {
        "case_id": "hybrid-evaluation-case",
        "case_source": "测试题",
        "question": "测试精确药物有什么作用？",
        "expected_name": "测试精确药物",
        "expected_type": "drug",
        "category": "混合检索",
    }
    with Session(test_engine) as session:
        session.add(
            Drug(
                name="测试精确药物",
                effects="用于验证关键词检索可以补充向量漏检。",
                instructions="仅用于自动化测试。",
            )
        )
        session.commit()

    monkeypatch.setattr(evaluation_router, "EVALUATION_CASES", (case,))
    monkeypatch.setattr(
        evaluation_router,
        "get_knowledge_status",
        lambda session: {"is_current": True},
    )
    monkeypatch.setattr(
        evaluation_router,
        "get_vector_store",
        IrrelevantVectorStore,
    )

    response = client.post("/evaluation/run")

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["passed"] is True
    assert result["matched_name"] == "测试精确药物"


def test_retrieval_metrics_report_top1_recall_and_precision():
    metrics = evaluation_router.build_retrieval_metrics(
        [
            RetrievalStrategyResult(
                passed=True,
                expected_rank=1,
                retrieved_count=3,
                relevant_count=1,
            ),
            RetrievalStrategyResult(
                passed=True,
                expected_rank=2,
                retrieved_count=3,
                relevant_count=1,
            ),
            RetrievalStrategyResult(
                passed=False,
                retrieved_count=2,
                relevant_count=0,
            ),
        ]
    )

    assert metrics.top1_correct_count == 1
    assert metrics.top1_accuracy == 1 / 3
    assert metrics.recalled_count == 2
    assert metrics.recall_at_3 == 2 / 3
    assert metrics.relevant_result_count == 2
    assert metrics.retrieved_result_count == 8
    assert metrics.precision_at_3 == 1 / 4
