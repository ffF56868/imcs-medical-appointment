from langchain_core.documents import Document

from app.routers import evaluation as evaluation_router


class FakeQualityVectorStore:
    def __init__(self, matches_by_question):
        self.matches_by_question = matches_by_question
        self.queries = []

    def similarity_search_with_relevance_scores(self, query, k):
        self.queries.append((query, k))
        return self.matches_by_question.get(query, [])[:k]


def test_quality_evaluation_reports_answer_citation_and_refusal_metrics(
    client,
    monkeypatch,
):
    normal_case = {
        "case_id": "quality-normal",
        "case_source": "测试题",
        "question": "布洛芬有什么作用？",
        "expected_name": "布洛芬",
        "expected_type": "drug",
        "category": "用药信息",
        "answer_keywords": ["布洛芬", "发热", "疼痛"],
        "citation_names": ["布洛芬"],
    }
    refusal_case = {
        "case_id": "quality-refusal",
        "case_source": "测试拒答题",
        "question": "知识库没有收录的深海月球花应该使用什么药物？",
        "category": "拒答能力",
        "answer_keywords": [],
        "citation_names": [],
        "expected_refusal": True,
    }
    vector_store = FakeQualityVectorStore(
        {
            normal_case["question"]: [
                (
                    Document(
                        page_content="布洛芬可用于缓解发热和疼痛。",
                        metadata={
                            "name": "布洛芬",
                            "type": "drug",
                            "record_id": 1,
                            "source": "测试资料",
                            "page_number": 1,
                        },
                    ),
                    0.9,
                )
            ],
            refusal_case["question"]: [],
        }
    )
    monkeypatch.setattr(evaluation_router, "EVALUATION_CASES", (normal_case,))
    monkeypatch.setattr(
        evaluation_router,
        "QUALITY_ONLY_CASES",
        (refusal_case,),
    )
    monkeypatch.setattr(
        evaluation_router,
        "get_knowledge_status",
        lambda session: {"is_current": True},
    )
    monkeypatch.setattr(evaluation_router, "get_vector_store", lambda: vector_store)

    response = client.post("/evaluation/quality")

    assert response.status_code == 200
    data = response.json()
    assert data["metrics"] == {
        "total_count": 2,
        "answer_correct_count": 2,
        "answer_accuracy": 1.0,
        "citation_correct_count": 2,
        "citation_accuracy": 1.0,
        "refusal_correct_count": 2,
        "refusal_accuracy": 1.0,
        "refusal_expected_count": 1,
        "refusal_observed_count": 1,
        "refusal_rate": 0.5,
    }
    assert data["results"][0]["answer_match_count"] == 3
    assert data["results"][0]["citation_has_location"] is True
    assert data["results"][1]["refusal_observed"] is True


def test_quality_evaluation_flags_missing_answer_keywords_and_bad_citation(
    client,
    monkeypatch,
):
    case = {
        "case_id": "quality-failure",
        "case_source": "测试题",
        "question": "测试资料是什么？",
        "expected_name": "目标资料",
        "expected_type": "document",
        "category": "质量测试",
        "answer_keywords": ["目标资料", "关键结论"],
        "citation_names": ["目标资料"],
    }
    vector_store = FakeQualityVectorStore(
        {
            case["question"]: [
                (
                    Document(
                        page_content="目标资料只有部分内容。",
                        metadata={
                            "name": "其他资料",
                            "type": "document",
                            "record_id": 9,
                        },
                    ),
                    0.9,
                )
            ]
        }
    )
    monkeypatch.setattr(evaluation_router, "EVALUATION_CASES", (case,))
    monkeypatch.setattr(evaluation_router, "QUALITY_ONLY_CASES", ())
    monkeypatch.setattr(
        evaluation_router,
        "get_knowledge_status",
        lambda session: {"is_current": True},
    )
    monkeypatch.setattr(evaluation_router, "get_vector_store", lambda: vector_store)

    response = client.post("/evaluation/quality")

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["answer_correct"] is False
    assert result["missing_answer_keywords"] == ["关键结论"]
    assert result["citation_correct"] is False
    assert result["missing_citation_names"] == ["目标资料"]
    assert result["citation_has_location"] is False


def test_custom_quality_fields_are_saved_and_returned(client):
    response = client.post(
        "/evaluation/cases",
        json={
            "question": "自定义质量题",
            "expected_name": "自定义资料",
            "expected_type": "document",
            "answer_keywords": ["结论", "注意事项"],
            "citation_names": ["自定义资料"],
            "expected_refusal": True,
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["answer_keywords"] == ["结论", "注意事项"]
    assert data["citation_names"] == ["自定义资料"]
    assert data["expected_refusal"] is True

