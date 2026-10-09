from types import SimpleNamespace

from langchain_core.documents import Document
from sqlmodel import Session

from app.legacy.models import Drug
from app.routers import ask as ask_router


class HybridFakeVectorStore:
    def __init__(self):
        self.last_filter = None

    def similarity_search_with_relevance_scores(self, _query, k, filter=None):
        self.last_filter = filter
        assert k == 8
        return [
            (
                Document(
                    page_content="布洛芬可用于缓解发热和疼痛。",
                    metadata={
                        "type": "drug",
                        "record_id": 1,
                        "name": "布洛芬",
                    },
                ),
                0.7,
            )
        ]


class HybridFakeChatModel:
    def invoke(self, _messages):
        return SimpleNamespace(content="这是基于混合检索的测试回答。")


def test_rag_combines_mysql_keyword_and_chroma_vector_results(
    client,
    test_engine,
    monkeypatch,
):
    with Session(test_engine) as session:
        session.add(
            Drug(
                name="布洛芬",
                effects="用于缓解发热和疼痛",
                instructions="请按说明书使用",
            )
        )
        session.commit()

    vector_store = HybridFakeVectorStore()
    monkeypatch.setattr(
        ask_router,
        "get_knowledge_status",
        lambda session: {"is_current": True},
    )
    monkeypatch.setattr(ask_router, "get_vector_store", lambda: vector_store)
    monkeypatch.setattr(ask_router, "get_chat_model", HybridFakeChatModel)

    response = client.post(
        "/ask",
        json={
            "conversation_id": "hybrid-retrieval",
            "question": "布洛芬有什么作用？",
            "knowledge_type": "drug",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "hybrid-rerank-retrieval-openai-generation"
    assert data["processing_path"] == "rag-hybrid-rerank"
    assert data["retrieved_count"] == 1
    assert data["references"][0]["name"] == "布洛芬"
    assert data["references"][0]["retrieval_method"] == "hybrid"
    assert data["references"][0]["vector_score"] == 0.7
    assert data["references"][0]["keyword_score"] == 1.0
    assert vector_store.last_filter == {"type": "drug"}
