from langchain_core.documents import Document
from sqlmodel import Session

from app.hybrid_search import (
    KeywordMatch,
    hybrid_search,
    extract_keyword_terms,
    keyword_search,
    merge_retrieval_matches,
)
from app.legacy.models import Drug, KnowledgeDocument, User


class StrategyVectorStore:
    def __init__(self):
        self.calls = 0

    def similarity_search_with_relevance_scores(self, _query, **_kwargs):
        self.calls += 1
        return [
            (
                Document(
                    page_content="向量结果",
                    metadata={"type": "document", "record_id": 1, "name": "向量资料"},
                ),
                0.8,
            )
        ]


def test_retrieval_strategy_can_disable_vector_and_keyword_stages(test_engine, monkeypatch):
    vector_store = StrategyVectorStore()
    keyword_calls = []
    monkeypatch.setattr(
        "app.hybrid_search.keyword_search",
        lambda *args, **kwargs: keyword_calls.append(True) or [],
    )

    with Session(test_engine) as session:
        assert hybrid_search(
            session, vector_store, "测试", "all", "all", retrieval_strategy="none"
        ) == []
        assert vector_store.calls == 0
        assert keyword_calls == []

        vector_matches = hybrid_search(
            session, vector_store, "测试", "all", "all", retrieval_strategy="vector"
        )
        assert len(vector_matches) == 1
        assert vector_store.calls == 1
        assert keyword_calls == []


def test_hybrid_strategy_skips_reranking(test_engine, monkeypatch):
    vector_store = StrategyVectorStore()
    monkeypatch.setattr(
        "app.hybrid_search.keyword_search",
        lambda *args, **kwargs: [],
    )
    rerank_calls = []
    monkeypatch.setattr(
        "app.hybrid_search.rerank_matches",
        lambda *args, **kwargs: rerank_calls.append(True) or [],
    )

    with Session(test_engine) as session:
        matches = hybrid_search(
            session, vector_store, "测试", "all", "all", retrieval_strategy="hybrid"
        )

    assert len(matches) == 1
    assert rerank_calls == []

    reranked = hybrid_search(
        session,
        vector_store,
        "测试",
        "all",
        "all",
        retrieval_strategy="hybrid-rerank",
    )
    assert reranked == []
    assert rerank_calls == [True]


def test_extract_keyword_terms_keeps_a_chinese_medical_name():
    terms = extract_keyword_terms("布洛芬有什么作用？")

    assert "布洛芬" in terms
    assert "什么" not in terms


def test_extract_keyword_terms_ignores_generic_specialty_overview_title_parts():
    terms = extract_keyword_terms("眼科常见病概览中有哪些就医警示？")

    assert "眼科" in terms
    assert "专科概览" not in terms
    assert "科概览" not in terms
    assert "就医警示" not in terms


def test_extract_keyword_terms_keeps_short_high_signal_clinical_clues():
    terms = extract_keyword_terms("反复出血、瘀斑或贫血相关问题属于哪个专科概览？")

    assert {"出血", "瘀斑", "贫血"}.issubset(terms)


def test_extract_keyword_terms_keeps_specialty_routing_clues():
    terms = extract_keyword_terms("外伤后肿胀、畸形、不能负重应查询哪个专科？")

    assert {"外伤", "肿胀", "畸形", "不能负重"}.issubset(terms)


def test_keyword_search_prioritizes_a_record_covering_multiple_clinical_clues(
    test_engine,
):
    with Session(test_engine) as session:
        session.add_all(
            [
                KnowledgeDocument(
                    title="血液科常见病概览",
                    content="反复出血、皮下瘀斑和贫血需要进行血液科评估。",
                ),
                KnowledgeDocument(
                    title="肿瘤科常见就诊警示概览",
                    content="异常出血需要医学评估。",
                ),
            ]
        )
        session.commit()

        matches = keyword_search(
            session,
            "反复出血、瘀斑或贫血相关问题属于哪个专科概览？",
            knowledge_type="document",
            source_filter="all",
        )

    assert matches[0].document.metadata["name"] == "血液科常见病概览"
    assert matches[0].document.metadata["keyword_priority_match_ratio"] == 1


def test_keyword_search_respects_type_and_private_document_access(test_engine):
    with Session(test_engine) as session:
        session.add(
            Drug(
                name="布洛芬",
                effects="缓解发热和疼痛",
                instructions="请阅读说明书并咨询医生",
            )
        )
        session.add(
            KnowledgeDocument(
                title="用户二的私有资料",
                content="只有用户二可以看到的布洛芬资料",
                owner_user_id=2,
                visibility="private",
            )
        )
        session.commit()

        regular_user = User(id=1, account="reader@example.com", is_admin=False)
        drug_matches = keyword_search(
            session,
            "布洛芬有什么作用",
            knowledge_type="drug",
            source_filter="all",
            current_user=regular_user,
        )
        private_matches = keyword_search(
            session,
            "用户二私有资料",
            knowledge_type="document",
            source_filter="all",
            current_user=regular_user,
        )

    assert [match.document.metadata["name"] for match in drug_matches] == [
        "布洛芬"
    ]
    assert private_matches == []


def test_merge_retrieval_matches_deduplicates_a_record_and_marks_hybrid():
    vector_document = Document(
        page_content="布洛芬的向量切块",
        metadata={"type": "drug", "record_id": 1, "name": "布洛芬"},
    )
    keyword_document = Document(
        page_content="布洛芬的关键词资料",
        metadata={"type": "drug", "record_id": 1, "name": "布洛芬"},
    )
    another_document = Document(
        page_content="感冒资料",
        metadata={"type": "condition", "record_id": 2, "name": "感冒"},
    )

    merged = merge_retrieval_matches(
        [(vector_document, 0.8)],
        [
            KeywordMatch(document=keyword_document, score=1.0),
            KeywordMatch(document=another_document, score=0.8),
        ],
    )

    assert len(merged) == 2
    assert merged[0][0].metadata["name"] == "布洛芬"
    assert merged[0][0].metadata["retrieval_method"] == "hybrid"
    assert merged[0][0].metadata["vector_score"] == 0.8
    assert merged[0][0].metadata["keyword_score"] == 1.0
    assert {item[0].metadata["record_id"] for item in merged} == {1, 2}
