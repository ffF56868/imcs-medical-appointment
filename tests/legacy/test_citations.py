from langchain_core.documents import Document
from sqlmodel import Session

from app.legacy.models import KnowledgeDocument
from app.routers.ask import build_references
from app.vector_store import build_knowledge_documents


def test_pdf_citation_contains_source_page_and_chunk_location():
    references = build_references(
        [
            (
                Document(
                    page_content="布洛芬可用于缓解发热和疼痛。",
                    metadata={
                        "type": "document",
                        "record_id": 7,
                        "name": "用药指南（第3页）",
                        "source": "上传 PDF：用药指南.pdf",
                        "source_tier": "professional",
                        "page_number": 3,
                        "chunk_index": 0,
                        "chunk_count": 2,
                    },
                ),
                0.82,
            )
        ]
    )

    reference = references[0]
    assert reference["source_kind"] == "PDF 文件"
    assert reference["page_number"] == 3
    assert reference["location"] == "第 3 页 · 切块 1/2"
    assert "用药指南.pdf" in reference["citation"]
    assert "第 3 页" in reference["citation"]


def test_pdf_page_number_survives_cleaning_and_chunking(test_engine):
    with Session(test_engine) as session:
        session.add(
            KnowledgeDocument(
                title="用药指南（第4页）",
                content="布洛芬可用于缓解发热和疼痛。\n请阅读说明书。",
                source="上传 PDF：用药指南.pdf",
                page_number=4,
            )
        )
        session.commit()

        chunks = build_knowledge_documents(session)

    assert len(chunks) == 1
    assert chunks[0].metadata["page_number"] == 4
    assert chunks[0].metadata["name"] == "用药指南（第4页）"


def test_web_citation_uses_web_location_without_a_page_number():
    references = build_references(
        [
            (
                Document(
                    page_content="网页中的健康资料。",
                    metadata={
                        "type": "document",
                        "record_id": 8,
                        "name": "健康网页",
                        "source": "网页导入：https://example.org/health",
                        "source_url": "https://example.org/health",
                    },
                ),
                0.7,
            )
        ]
    )

    reference = references[0]
    assert reference["source_kind"] == "网页"
    assert reference["page_number"] is None
    assert reference["location"] == "网页正文"
