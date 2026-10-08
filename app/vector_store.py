import os
from hashlib import sha256
import json
import socket

from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from sqlmodel import Session, select

from app.legacy.models import Condition, Drug, KnowledgeDocument, KnowledgeIndexState
from app.cache import (
    KNOWLEDGE_STATUS_CACHE_KEY,
    KNOWLEDGE_STATUS_CACHE_TTL_SECONDS,
    get_cached_json,
    invalidate_knowledge_status_cache,
    set_cached_json,
)
from app.source_metadata import get_source_tier_label, needs_source_review
from app.text_processing import (
    split_document_for_embedding,
    get_text_processing_config,
)


COLLECTION_NAME = "medical_health_knowledge"
VECTOR_STORE_TYPE = "milvus"
MIN_RELEVANCE_SCORE = 0.2
RAG_RETRIEVAL_FETCH_COUNT = 8
RAG_RETRIEVAL_RESULT_COUNT = 3


def select_distinct_relevant_matches(
    matches: list[tuple[object, float]],
    min_relevance_score: float = MIN_RELEVANCE_SCORE,
    limit: int = RAG_RETRIEVAL_RESULT_COUNT,
) -> list[tuple[object, float]]:
    """Keep the strongest relevant chunk from each knowledge record."""
    best_matches: dict[tuple[str, str], tuple[object, float]] = {}
    for document, score in matches:
        if score < min_relevance_score:
            continue
        metadata = getattr(document, "metadata", {})
        source_key = (
            str(metadata.get("type", "unknown")),
            str(metadata.get("record_id", metadata.get("name"))),
        )
        previous_match = best_matches.get(source_key)
        if previous_match is None or score > previous_match[1]:
            best_matches[source_key] = (document, score)

    return sorted(
        best_matches.values(),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]


def _quote_milvus_string(value: object) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _milvus_filter_expression(filter_value: object) -> str | None:
    """Convert the existing Chroma-style filter tree to a Milvus expression."""
    if not isinstance(filter_value, dict) or not filter_value:
        return None
    if len(filter_value) == 1 and "$and" in filter_value:
        children = [
            _milvus_filter_expression(child)
            for child in filter_value["$and"]
        ]
        expressions = [child for child in children if child]
        return " and ".join(f"({child})" for child in expressions) or None
    if len(filter_value) == 1 and "$or" in filter_value:
        children = [
            _milvus_filter_expression(child)
            for child in filter_value["$or"]
        ]
        expressions = [child for child in children if child]
        return " or ".join(f"({child})" for child in expressions) or None

    expressions: list[str] = []
    for field, value in filter_value.items():
        if isinstance(value, bool):
            rendered_value = "true" if value else "false"
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            rendered_value = str(value)
        else:
            rendered_value = _quote_milvus_string(value)
        expressions.append(f"{field} == {rendered_value}")
    return " and ".join(f"({expression})" for expression in expressions)


class MilvusVectorStore(Milvus):
    """Milvus adapter that keeps the project's existing vector-store API."""

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: object,
    ):
        # Existing callers use Chroma's ``filter=`` keyword. Milvus calls the
        # equivalent boolean expression ``expr``.
        filter_value = kwargs.pop("filter", None)
        if filter_value is not None and "expr" not in kwargs:
            expression = _milvus_filter_expression(filter_value)
            if expression:
                kwargs["expr"] = expression
        return super().similarity_search_with_relevance_scores(
            query,
            k=k,
            **kwargs,
        )

    def delete_collection(self) -> None:
        """Compatibility name used by the old Chroma rebuild flow."""
        self.drop()


def get_embeddings() -> OpenAIEmbeddings:
    """Create the OpenAI embedding client shared by retrieval and RAGAS."""
    # Embedding requests are made before every vector search.  A transient
    # proxy/API connection failure used to be converted by the ask router into
    # "please rebuild the knowledge base", even when Milvus and the index were
    # healthy.  Keep the retry budget configurable, but make the default large
    # enough for the first request after a container/network wake-up.
    try:
        max_retries = max(0, int(os.getenv("OPENAI_MAX_RETRIES", "5")))
    except ValueError:
        max_retries = 5
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        max_retries=max_retries,
        retry_min_seconds=1,
        retry_max_seconds=8,
    )


def get_vector_store() -> MilvusVectorStore:
    embeddings = get_embeddings()
    milvus_host = os.getenv("MILVUS_HOST", "milvus")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    return MilvusVectorStore(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={
            "uri": f"http://{milvus_host}:{milvus_port}",
        },
        index_params={
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
            "params": {},
        },
        search_params={
            "metric_type": "COSINE",
            "params": {},
        },
        # Explicit metadata fields make filters and citations available after
        # a new API process starts, not only in the process that inserted data.
        enable_dynamic_field=False,
    )


def get_milvus_vector_count() -> int | None:
    """Return the current Milvus row count, or None while Milvus is unavailable."""
    try:
        host = os.getenv("MILVUS_HOST", "milvus")
        port = int(os.getenv("MILVUS_PORT", "19530"))
        # Avoid making the API health endpoint wait on a missing Milvus daemon.
        with socket.create_connection((host, port), timeout=0.5):
            pass
        vector_store = get_vector_store()
        if not vector_store.client.has_collection(COLLECTION_NAME):
            return 0
        stats = vector_store.client.get_collection_stats(COLLECTION_NAME)
        return int(stats.get("row_count", 0))
    except Exception:
        return None


def build_source_metadata(record: object) -> dict[str, str | bool]:
    source = str(getattr(record, "source", "未标注来源") or "未标注来源")
    source_url = str(getattr(record, "source_url", "") or "")
    source_tier = str(getattr(record, "source_tier", "unverified") or "unverified")
    updated_at = getattr(record, "updated_at", None)
    return {
        "source": source,
        "source_url": source_url,
        "source_tier": source_tier,
        "updated_at": updated_at.isoformat() if updated_at else "",
        "needs_review": needs_source_review(
            source_tier,
            updated_at,
            source,
            source_url,
        ),
    }


def build_access_metadata(record: object) -> dict[str, str | int]:
    """Keep database ownership fields available to vector filters."""
    owner_user_id = getattr(record, "owner_user_id", None)
    return {
        "knowledge_base_id": str(
            getattr(record, "knowledge_base_id", "global") or "global"
        ),
        "visibility": str(getattr(record, "visibility", "public") or "public"),
        # 0 represents a public record without an owner.
        "owner_user_id": int(owner_user_id or 0),
    }


def build_knowledge_documents(session: Session) -> list[Document]:
    documents: list[Document] = []

    conditions = session.exec(select(Condition).order_by(Condition.id)).all()
    for condition in conditions:
        documents.append(
            Document(
                page_content=(
                    f"病症名称：{condition.name}\n"
                    f"常见症状：{condition.symptoms}\n"
                    f"处理建议：{condition.treatment}\n"
                    f"资料来源：{condition.source}\n"
                    f"来源链接：{condition.source_url or '未提供'}\n"
                    f"可信度：{get_source_tier_label(condition.source_tier)}\n"
                    f"最后更新：{condition.updated_at.isoformat() if condition.updated_at else '未记录'}"
                ),
                metadata={
                    "type": "condition",
                    "record_id": condition.id,
                    "name": condition.name,
                    "page_number": 0,
                    **build_source_metadata(condition),
                    **build_access_metadata(condition),
                },
            )
        )

    drugs = session.exec(select(Drug).order_by(Drug.id)).all()
    for drug in drugs:
        documents.append(
            Document(
                page_content=(
                    f"药物名称：{drug.name}\n"
                    f"药物作用：{drug.effects}\n"
                    f"使用说明：{drug.instructions}\n"
                    f"资料来源：{drug.source}\n"
                    f"来源链接：{drug.source_url or '未提供'}\n"
                    f"可信度：{get_source_tier_label(drug.source_tier)}\n"
                    f"最后更新：{drug.updated_at.isoformat() if drug.updated_at else '未记录'}"
                ),
                metadata={
                    "type": "drug",
                    "record_id": drug.id,
                    "name": drug.name,
                    "page_number": 0,
                    **build_source_metadata(drug),
                    **build_access_metadata(drug),
                },
            )
        )

    knowledge_documents = session.exec(
        select(KnowledgeDocument).order_by(KnowledgeDocument.id)
    ).all()
    for knowledge_document in knowledge_documents:
        documents.append(
            Document(
                page_content=(
                    f"资料标题：{knowledge_document.title}\n"
                    f"资料来源：{knowledge_document.source}\n"
                    f"来源链接：{knowledge_document.source_url or '未提供'}\n"
                    f"可信度：{get_source_tier_label(knowledge_document.source_tier)}\n"
                    f"最后更新：{knowledge_document.updated_at.isoformat() if knowledge_document.updated_at else '未记录'}\n"
                    f"资料内容：{knowledge_document.content}"
                ),
                metadata={
                    "type": "document",
                    "record_id": knowledge_document.id,
                    "name": knowledge_document.title,
                    "page_number": knowledge_document.page_number or 0,
                    **build_source_metadata(knowledge_document),
                    **build_access_metadata(knowledge_document),
                },
            )
        )

    return [
        chunk
        for document in documents
        for chunk in split_document_for_embedding(document)
    ]


def get_knowledge_fingerprint(documents: list[Document]) -> str:
    content = [
        {
            "page_content": document.page_content,
            "metadata": document.metadata,
        }
        for document in documents
    ]
    serialized = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return sha256(serialized.encode("utf-8")).hexdigest()


def get_knowledge_status(session: Session) -> dict:
    processing_config = get_text_processing_config()
    cached_status = get_cached_json(KNOWLEDGE_STATUS_CACHE_KEY)
    if (
        isinstance(cached_status, dict)
        and "is_current" in cached_status
        and cached_status.get("vector_store_type") == VECTOR_STORE_TYPE
        and all(
            cached_status.get(key) == value
            for key, value in processing_config.items()
        )
    ):
        return cached_status

    documents = build_knowledge_documents(session)
    current_hash = get_knowledge_fingerprint(documents)
    index_state = session.get(KnowledgeIndexState, 1)
    vector_count = get_milvus_vector_count()
    source_document_count = sum(
        len(records)
        for records in (
            session.exec(select(Condition)).all(),
            session.exec(select(Drug)).all(),
            session.exec(select(KnowledgeDocument)).all(),
        )
    )

    status = {
        "is_current": (
            index_state is not None
            and index_state.content_hash == current_hash
            and index_state.vector_store_type == VECTOR_STORE_TYPE
            and (
                vector_count is None
                or vector_count == len(documents)
            )
        ),
        "document_count": source_document_count,
        "chunk_count": len(documents),
        "indexed_document_count": (
            index_state.document_count if index_state is not None else None
        ),
        "vector_store_type": VECTOR_STORE_TYPE,
        "vector_count": vector_count,
        "indexed_at": index_state.indexed_at if index_state is not None else None,
        **processing_config,
    }
    set_cached_json(
        KNOWLEDGE_STATUS_CACHE_KEY,
        status,
        KNOWLEDGE_STATUS_CACHE_TTL_SECONDS,
    )
    return status


def rebuild_vector_store(session: Session) -> tuple[int, int]:
    documents = build_knowledge_documents(session)
    vector_store = get_vector_store()

    vector_store.delete_collection()
    vector_store = get_vector_store()

    if documents:
        ids = [
            "-".join(
                [
                    document.metadata["type"],
                    str(document.metadata["record_id"]),
                    str(document.metadata["chunk_index"]),
                ]
            )
            for document in documents
        ]
        vector_store.add_documents(documents=documents, ids=ids)
        # Milvus reports a row count only after pending inserts are flushed.
        vector_store.client.flush(COLLECTION_NAME)

    index_state = KnowledgeIndexState(
        id=1,
        content_hash=get_knowledge_fingerprint(documents),
        document_count=len(documents),
        vector_store_type=VECTOR_STORE_TYPE,
        vector_count=len(documents),
    )
    session.merge(index_state)
    session.commit()
    invalidate_knowledge_status_cache()

    source_document_count = sum(
        len(records)
        for records in (
            session.exec(select(Condition)).all(),
            session.exec(select(Drug)).all(),
            session.exec(select(KnowledgeDocument)).all(),
        )
    )
    return source_document_count, len(documents)
