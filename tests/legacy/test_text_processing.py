from langchain_core.documents import Document

from app.text_processing import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKING_STRATEGY,
    TEXT_CLEANING_VERSION,
    clean_text_for_embedding,
    split_document_for_embedding,
)
from app.vector_store import get_knowledge_status


def test_clean_text_for_embedding_removes_formatting_noise_only():
    cleaned = clean_text_for_embedding(
        "\ufeff  感冒  \t可能出现鼻塞。\r\n\r\n\r\n"
        "注意补水。\n注意补水。\n\u200b"
    )

    assert cleaned.text == "感冒 可能出现鼻塞。\n\n注意补水。"
    assert cleaned.removed_blank_line_count == 1
    assert cleaned.removed_duplicate_line_count == 1


def test_split_document_for_embedding_records_chunk_traceability():
    text = "".join(f"第{i}条健康建议：注意休息和补水。" for i in range(1, 120))
    source_document = Document(
        page_content=text,
        metadata={"type": "document", "record_id": 1, "name": "测试资料"},
    )

    chunks = split_document_for_embedding(source_document)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= CHUNK_SIZE for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert all(chunk.metadata["chunk_count"] == len(chunks) for chunk in chunks)
    assert all(
        chunk.metadata["chunking_strategy"] == CHUNKING_STRATEGY for chunk in chunks
    )
    assert all(
        chunk.metadata["text_cleaning_version"] == TEXT_CLEANING_VERSION
        for chunk in chunks
    )
    assert all(chunk.metadata["chunk_overlap"] == CHUNK_OVERLAP for chunk in chunks)
    assert chunks[1].metadata["chunk_start_index"] < chunks[0].metadata[
        "chunk_end_index"
    ]


def test_split_document_does_not_mutate_the_stored_source_document():
    source_document = Document(
        page_content="原文  有空格。\n\n\n原文第二段。",
        metadata={"type": "document", "record_id": 2, "name": "原文资料"},
    )

    chunks = split_document_for_embedding(source_document)

    assert source_document.page_content == "原文  有空格。\n\n\n原文第二段。"
    assert chunks[0].page_content == "原文 有空格。\n\n原文第二段。"


def test_knowledge_status_includes_the_active_text_processing_config(
    test_engine,
):
    from sqlmodel import Session

    with Session(test_engine) as session:
        status = get_knowledge_status(session)

    assert status["text_cleaning_version"] == TEXT_CLEANING_VERSION
    assert status["chunking_strategy"] == CHUNKING_STRATEGY
    assert status["chunk_size"] == CHUNK_SIZE
    assert status["chunk_overlap"] == CHUNK_OVERLAP
