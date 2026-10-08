"""Keyword, vector, and hybrid retrieval for the RAG pipeline.

The database search is intentionally small and explainable: MySQL finds
explicit terms with LIKE, while Milvus finds semantically similar text. The
two result sets are merged by the original knowledge record, so one record
cannot occupy several answer slots just because it has several chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from sqlalchemy import or_
from sqlmodel import Session, select

from app.access import accessible_documents_statement
from app.legacy.models import Condition, Drug, KnowledgeDocument, User
from app.reranker import rerank_matches
from app.source_metadata import needs_source_review


KEYWORD_FETCH_COUNT = 8
VECTOR_WEIGHT = 0.65
KEYWORD_WEIGHT = 0.35
RETRIEVAL_STRATEGIES = (
    "none",
    "vector",
    "hybrid",
    "hybrid-rerank",
)
RETRIEVAL_STRATEGY_LABELS = {
    "none": "无检索（直接生成）",
    "vector": "仅向量检索",
    "hybrid": "混合检索（不重排序）",
    "hybrid-rerank": "混合检索 + 重排序",
}
PRIORITY_MEDICAL_TERMS = (
    "对乙酰氨基酚",
    "阿莫西林",
    "布洛芬",
    "反酸",
    "吞咽困难",
    "呼吸困难",
    "意识改变",
    "血小板",
    "白细胞",
    "淋巴结",
    "对青霉素过敏",
    "胸痛",
    "昏迷",
    "外伤",
    "肿胀",
    "畸形",
    "不能负重",
    "肢体麻木",
    "头晕",
    "头痛",
    "言语不清",
    "视物异常",
    "平衡障碍",
    "急症",
    "急诊",
    "贫血",
    "出血",
    "瘀斑",
    "血液",
    "骨髓",
)
PRIORITY_TERM_SCORE_WEIGHT = 0.30

SEARCH_STOP_WORDS = {
    "什么",
    "怎么",
    "如何",
    "可以",
    "是否",
    "有没有",
    "请问",
    "一下",
    "相关",
    "资料",
    "问题",
    "哪些",
    "哪里",
    "这个",
    "那个",
    "以及",
    "还有",
}

# These phrases occur in many document titles. They describe a document's
# format, not the medical topic the user is looking for.
GENERIC_TITLE_PHRASES = (
    "专科概览",
    "常见病概览",
    "健康教育",
    "就医警示",
)


@dataclass
class KeywordMatch:
    document: Document
    score: float


def normalize_retrieval_strategy(value: str | None) -> str:
    """Validate the explicit retrieval mode used by evaluation runs."""
    strategy = str(value or "hybrid-rerank").strip().lower()
    if strategy not in RETRIEVAL_STRATEGIES:
        choices = "、".join(RETRIEVAL_STRATEGIES)
        raise ValueError(f"检索策略必须是：{choices}")
    return strategy


def build_vector_filter(
    knowledge_type: str,
    source_filter: str,
    current_user: User | None = None,
    page_number: int | None = None,
) -> dict | None:
    """Build the vector filter shared by vector and hybrid retrieval."""
    conditions: list[dict] = []
    if knowledge_type != "all":
        conditions.append({"type": knowledge_type})
    if source_filter == "reviewed":
        conditions.append({"needs_review": False})
    if page_number is not None:
        conditions.append({"page_number": page_number})
    if current_user is not None and not current_user.is_admin:
        conditions.append(
            {
                "$or": [
                    {"visibility": "public"},
                    {
                        "$and": [
                            {"visibility": "private"},
                            {"owner_user_id": current_user.id},
                        ]
                    },
                ]
            }
        )
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value.casefold())


def _is_generic_title_fragment(term: str) -> bool:
    return any(term in phrase for phrase in GENERIC_TITLE_PHRASES)


def get_priority_medical_terms(query: str) -> list[str]:
    """Keep short clinical clues when a long Chinese question is not segmented."""
    normalized_query = _normalize_text(query)
    return [
        term
        for term in PRIORITY_MEDICAL_TERMS
        if term in normalized_query
    ]


def extract_keyword_terms(query: str) -> list[str]:
    """Extract useful terms from both spaced and unspaced Chinese questions.

    Chinese questions often have no word separators. Besides normal words,
    short n-grams let a question such as ``布洛芬有什么作用`` find ``布洛芬``
    in MySQL. Generic question words are discarded before the SQL query is
    built, which keeps ranking focused on medical terms.
    """
    normalized_query = _normalize_text(query)
    for phrase in GENERIC_TITLE_PHRASES:
        normalized_query = normalized_query.replace(phrase, "")
    if not normalized_query:
        return []

    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9][a-z0-9_+-]*|[\u4e00-\u9fff]+", normalized_query):
        if token in SEARCH_STOP_WORDS or _is_generic_title_fragment(token):
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if (
                len(token) >= 2
                and token not in SEARCH_STOP_WORDS
                and not _is_generic_title_fragment(token)
            ):
                terms.add(token)
            if len(token) <= 30:
                for size in range(2, min(8, len(token)) + 1):
                    for start in range(0, len(token) - size + 1):
                        ngram = token[start : start + size]
                        if (
                            ngram not in SEARCH_STOP_WORDS
                            and not _is_generic_title_fragment(ngram)
                        ):
                            terms.add(ngram)
        elif len(token) >= 2 or token.isdigit():
            terms.add(token)

    # Longer terms are normally more informative. A bounded list also protects
    # the database query when a user pastes a long paragraph as the question.
    # Keep reviewed medical clues first so unsegmented text does not drop terms
    # such as "贫血" and "瘀斑" before retrieval begins.
    priority_terms = get_priority_medical_terms(query)
    remaining_terms = sorted(
        terms.difference(priority_terms),
        key=lambda term: (-len(term), term),
    )
    return [*priority_terms, *remaining_terms][:60]


def _contains_any(fields: list[object], terms: list[str]):
    clauses = [
        field.contains(term, autoescape=True)
        for field in fields
        for term in terms
    ]
    return or_(*clauses) if clauses else None


def _record_is_allowed(
    record: Condition | Drug | KnowledgeDocument,
    record_type: str,
    source_filter: str,
    current_user: User | None,
) -> bool:
    if source_filter == "reviewed" and needs_source_review(
        record.source_tier,
        record.updated_at,
        record.source,
        record.source_url,
    ):
        return False
    if current_user is not None and not current_user.is_admin:
        owner_user_id = getattr(record, "owner_user_id", None)
        if record.visibility != "public" and owner_user_id != current_user.id:
            return False
    return True


def _build_record_document(
    record: Condition | Drug | KnowledgeDocument,
    record_type: str,
    matched_fields: list[str],
    score: float,
    priority_match_ratio: float = 0.0,
) -> Document:
    if record_type == "condition":
        title = record.name
        page_content = (
            f"病症名称：{record.name}\n"
            f"常见症状：{record.symptoms}\n"
            f"处理建议：{record.treatment}"
        )
    elif record_type == "drug":
        title = record.name
        page_content = (
            f"药物名称：{record.name}\n"
            f"药物作用：{record.effects}\n"
            f"使用说明：{record.instructions}"
        )
    else:
        title = record.title
        page_content = f"资料标题：{record.title}\n资料内容：{record.content}"

    metadata = {
        "type": record_type,
        "record_id": record.id,
        "name": title,
        "source": record.source,
        "source_url": record.source_url or "",
        "source_tier": record.source_tier,
        "updated_at": record.updated_at.isoformat() if record.updated_at else "",
        "needs_review": needs_source_review(
            record.source_tier,
            record.updated_at,
            record.source,
            record.source_url,
        ),
        "knowledge_base_id": record.knowledge_base_id,
        "visibility": record.visibility,
        "owner_user_id": getattr(record, "owner_user_id", 0) or 0,
        "retrieval_method": "keyword",
        "keyword_score": round(score, 4),
        "keyword_fields": "|".join(matched_fields),
        "keyword_priority_match_ratio": round(priority_match_ratio, 4),
    }
    if record_type == "document":
        metadata["page_number"] = record.page_number or 0
    return Document(page_content=page_content, metadata=metadata)


def _score_record(
    fields: dict[str, str],
    terms: list[str],
) -> tuple[float, list[str], float]:
    normalized_fields = {
        field_name: _normalize_text(value)
        for field_name, value in fields.items()
    }
    matched_fields = [
        field_name
        for field_name, value in normalized_fields.items()
        if any(term in value for term in terms)
    ]
    if not matched_fields:
        return 0.0, [], 0.0

    matched_terms = {
        term
        for term in terms
        if any(term in value for value in normalized_fields.values())
    }
    coverage = sum(len(term) for term in matched_terms) / max(
        1,
        sum(len(term) for term in terms),
    )
    title_match = "名称" in matched_fields or "标题" in matched_fields
    title_values = [
        normalized_fields[field_name]
        for field_name in ("名称", "标题")
        if field_name in normalized_fields
    ]
    exact_title_term = any(
        term in title_value
        for term in terms
        if len(term) >= 2
        for title_value in title_values
    )
    # A name/title hit is the strongest exact signal; field coverage adds a
    # smaller bonus without allowing a long pasted question to dominate.
    if title_match and exact_title_term:
        return 1.0, matched_fields, 1.0
    priority_terms = [
        term
        for term in terms
        if term in PRIORITY_MEDICAL_TERMS
    ]
    priority_match_ratio = (
        sum(
            len(term)
            for term in priority_terms
            if any(term in value for value in normalized_fields.values())
        )
        / max(1, sum(len(term) for term in priority_terms))
        if priority_terms
        else 0.0
    )
    score = (
        (0.75 if title_match else 0.45)
        + min(0.25, coverage)
        + PRIORITY_TERM_SCORE_WEIGHT * priority_match_ratio
    )
    return min(1.0, score), matched_fields, priority_match_ratio


def keyword_search(
    session: Session,
    query: str,
    knowledge_type: str,
    source_filter: str,
    current_user: User | None = None,
    limit: int = KEYWORD_FETCH_COUNT,
    page_number: int | None = None,
) -> list[KeywordMatch]:
    """Search MySQL/SQLModel fields for explicit terms and rank the records."""
    terms = extract_keyword_terms(query)
    if not terms:
        return []

    record_sets: list[tuple[str, type, dict[str, object]]] = []
    if knowledge_type in {"all", "condition"}:
        record_sets.append(
            (
                "condition",
                Condition,
                {
                    "名称": Condition.name,
                    "症状": Condition.symptoms,
                    "处理建议": Condition.treatment,
                },
            )
        )
    if knowledge_type in {"all", "drug"}:
        record_sets.append(
            (
                "drug",
                Drug,
                {
                    "名称": Drug.name,
                    "作用": Drug.effects,
                    "使用说明": Drug.instructions,
                },
            )
        )
    if knowledge_type in {"all", "document"}:
        record_sets.append(
            (
                "document",
                KnowledgeDocument,
                {
                    "标题": KnowledgeDocument.title,
                    "内容": KnowledgeDocument.content,
                    "来源": KnowledgeDocument.source,
                },
            )
        )

    matches: list[KeywordMatch] = []
    for record_type, model, fields in record_sets:
        statement = select(model)
        if record_type == "document":
            statement = (
                accessible_documents_statement(current_user)
                if current_user is not None
                else select(KnowledgeDocument)
            )
            # 按页码过滤（仅对 document 类型有效）
            if page_number is not None:
                statement = statement.where(KnowledgeDocument.page_number == page_number)
        condition = _contains_any(list(fields.values()), terms)
        if condition is not None:
            statement = statement.where(condition)
        records = session.exec(statement).all()
        for record in records:
            if not _record_is_allowed(
                record,
                record_type,
                source_filter,
                current_user,
            ):
                continue
            text_fields = {
                field_name: getattr(record, field_name_source.key)
                for field_name, field_name_source in fields.items()
            }
            score, matched_fields, priority_match_ratio = _score_record(
                text_fields,
                terms,
            )
            if score <= 0:
                continue
            matches.append(
                KeywordMatch(
                    document=_build_record_document(
                        record,
                        record_type,
                        matched_fields,
                        score,
                        priority_match_ratio,
                    ),
                    score=score,
                )
            )

    return sorted(
        matches,
        key=lambda match: (
            match.score,
            str(match.document.metadata.get("name", "")),
        ),
        reverse=True,
    )[:limit]


def _record_key(document: Document) -> tuple[str, str]:
    metadata = document.metadata
    return (
        str(metadata.get("type", "unknown")),
        str(metadata.get("record_id", metadata.get("name", ""))),
    )


def _safe_score(score: object) -> float:
    try:
        return max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        return 0.0


def merge_retrieval_matches(
    vector_matches: list[tuple[object, float]],
    keyword_matches: list[KeywordMatch],
) -> list[tuple[Document, float]]:
    """Merge both retrieval paths by ``type + record_id`` without duplicates."""
    grouped: dict[tuple[str, str], dict[str, object]] = {}

    for document, raw_score in vector_matches:
        if not isinstance(document, Document):
            continue
        key = _record_key(document)
        item = grouped.setdefault(key, {})
        vector_score = _safe_score(raw_score)
        if vector_score > float(item.get("vector_score", 0.0)):
            item["vector_score"] = vector_score
            item["vector_document"] = document

    for match in keyword_matches:
        key = _record_key(match.document)
        item = grouped.setdefault(key, {})
        if match.score > float(item.get("keyword_score", 0.0)):
            item["keyword_score"] = match.score
            item["keyword_document"] = match.document
            item["keyword_priority_match_ratio"] = _safe_score(
                match.document.metadata.get("keyword_priority_match_ratio")
            )

    merged: list[tuple[Document, float]] = []
    for item in grouped.values():
        vector_score = float(item.get("vector_score", 0.0))
        keyword_score = float(item.get("keyword_score", 0.0))
        vector_document = item.get("vector_document")
        keyword_document = item.get("keyword_document")
        document = vector_document or keyword_document
        if not isinstance(document, Document):
            continue

        if vector_score and keyword_score:
            score = VECTOR_WEIGHT * vector_score + KEYWORD_WEIGHT * keyword_score
            method = "hybrid"
        elif keyword_score:
            score = keyword_score
            method = "keyword"
        else:
            score = vector_score
            method = "vector"

        metadata = dict(document.metadata)
        metadata.update(
            {
                "retrieval_method": method,
                "vector_score": round(vector_score, 4),
                "keyword_score": round(keyword_score, 4),
                "keyword_priority_match_ratio": round(
                    float(item.get("keyword_priority_match_ratio", 0.0)),
                    4,
                ),
            }
        )
        merged.append(
            (
                Document(page_content=document.page_content, metadata=metadata),
                score,
            )
        )

    return sorted(merged, key=lambda item: item[1], reverse=True)


def hybrid_search(
    session: Session,
    vector_store: object,
    query: str,
    knowledge_type: str,
    source_filter: str,
    current_user: User | None = None,
    vector_fetch_count: int = 8,
    keyword_query: str | None = None,
    retrieval_strategy: str = "hybrid-rerank",
    page_number: int | None = None,
) -> list[tuple[Document, float]]:
    """Run the selected retrieval stages and return one ranked list.

    The function name is retained for compatibility with the existing API.
    ``hybrid-rerank`` is the production default; evaluation can explicitly
    disable either stage without changing normal user conversations.
    """
    strategy = normalize_retrieval_strategy(retrieval_strategy)
    if strategy == "none":
        return []

    vector_options: dict[str, object] = {"k": vector_fetch_count}
    vector_filter = build_vector_filter(
        knowledge_type,
        source_filter,
        current_user,
        page_number,
    )
    if vector_filter is not None:
        vector_options["filter"] = vector_filter
    vector_matches = (
        vector_store.similarity_search_with_relevance_scores(
            query,
            **vector_options,
        )
        if strategy in {"vector", "hybrid", "hybrid-rerank"}
        else []
    )
    keyword_matches = (
        keyword_search(
            session,
            keyword_query or query,
            knowledge_type,
            source_filter,
            current_user,
            page_number=page_number,
        )
        if strategy in {"hybrid", "hybrid-rerank"}
        else []
    )
    merged_matches = merge_retrieval_matches(vector_matches, keyword_matches)
    if strategy == "hybrid-rerank":
        return rerank_matches(keyword_query or query, merged_matches)
    return merged_matches


def get_retrieval_method(matches: list[tuple[object, float]]) -> str:
    methods = {
        str(getattr(document, "metadata", {}).get("retrieval_method", "vector"))
        for document, _ in matches
    }
    if "hybrid" in methods:
        return "hybrid"
    if "keyword" in methods:
        return "keyword"
    return "vector"
