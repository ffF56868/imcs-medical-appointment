import os
import json
import logging
import re
from datetime import datetime
from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sqlmodel import Session, select

from app.database import get_session
from app.legacy.models import ChatMessage, User
from app.routers.auth import get_current_user
from app.legacy.schemas import AskRequest, AskResponse
from app.hybrid_search import (
    build_vector_filter,
    extract_keyword_terms,
    get_retrieval_method,
    hybrid_search,
    normalize_retrieval_strategy,
)
from app.monitoring import record_request_metric
from app.memory import (
    WorkingMemory,
    build_memory_context,
    finalize_conversation_memory,
    remember_explicit_user_facts,
)
from app.intent_router import (
    IntentChannel,
    route_intent,
    build_chat_reply,
    build_followup_context,
)
from app.source_metadata import needs_source_review
from app.vector_store import (
    RAG_RETRIEVAL_FETCH_COUNT,
    get_knowledge_status,
    get_vector_store,
    select_distinct_relevant_matches,
)


router = APIRouter(
    prefix="/ask",
    tags=["ask"],
    dependencies=[Depends(get_current_user)],
)
logger = logging.getLogger(__name__)
URGENT_WARNING_KEYWORDS = (
    "呼吸困难",
    "持续胸痛",
    "加重的胸痛",
    "意识改变",
    "抽搐",
    "单侧肢体无力",
    "言语含糊",
    "严重过敏",
    "头痛突然剧烈",
)

DIRECT_LOOKUP_KEYWORDS = (
    "哪份",
    "哪份资料",
    "什么资料",
    "哪个资料",
    "哪条资料",
    "哪条病症资料",
    "哪篇资料",
    "哪个专科",
    "哪个内科",
    "哪个外科",
    "哪个科",
    "哪种病症",
    "什么病症",
    "哪个概览",
    "查什么",
    "看什么",
    "哪里看",
    "什么说明",
    "使用前",
    "说明书",
)
DRUG_QUESTION_KEYWORDS = (
    "药物",
    "药品",
    "用药",
    "怎么用",
    "如何用",
    "服用",
    "使用说明",
    "注意事项",
    "药效",
    "作用",
)
DOCUMENT_CONTENT_QUESTION_MARKERS = (
    "资料中",
    "概览中",
    "资料对",
    "概览对",
)
SYMPTOM_QUESTION_KEYWORDS = (
    "怎么办",
    "怎么处理",
    "不舒服",
    "症状",
    "疼",
    "痛",
    "发热",
    "发烧",
    "咳嗽",
    "腹泻",
    "头晕",
    "恶心",
    "呕吐",
    "出血",
    "瘀斑",
)


def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("CHAT_MODEL", "gpt-4.1-mini"),
        temperature=0,
    )


def save_message(
    session: Session,
    conversation_id: str,
    role: str,
    content: str,
    response_metadata: dict | None = None,
    user_id: int | None = None,
) -> ChatMessage:
    message = ChatMessage(
        user_id=user_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        response_metadata_json=json.dumps(
            response_metadata or {},
            ensure_ascii=False,
            default=str,
        ),
    )
    session.add(message)
    return message


def get_urgent_warning_response(question: str) -> str | None:
    matched_keywords = [
        keyword
        for keyword in URGENT_WARNING_KEYWORDS
        if keyword in question
    ]
    if not matched_keywords:
        return None

    keywords_text = "、".join(matched_keywords)
    return (
        f"你的描述包含需要及时就医的警示词：{keywords_text}。"
        "这不代表任何具体诊断，但如果这些症状正在发生、持续或加重，"
        "请立即寻求紧急医疗帮助或联系当地急救服务。"
        "不要依赖本系统的在线回答来延误线下评估。\n\n"
        "内容仅供健康信息参考，不代替医生诊断或处方。"
    )


def format_sse_event(event: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {data}\n\n"


def get_stream_text(chunk: object) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )
    return str(content)


def search_knowledge(
    vector_store: object,
    query: str,
    knowledge_type: str,
    source_filter: str,
    current_user: User | None = None,
    session: Session | None = None,
    keyword_query: str | None = None,
    retrieval_strategy: str = "hybrid-rerank",
    page_number: int | None = None,
):
    if session is not None:
        return hybrid_search(
            session,
            vector_store,
            query,
            knowledge_type,
            source_filter,
            current_user,
            RAG_RETRIEVAL_FETCH_COUNT,
            keyword_query=keyword_query,
            retrieval_strategy=retrieval_strategy,
            page_number=page_number,
        )

    strategy = normalize_retrieval_strategy(retrieval_strategy)
    if strategy == "none":
        return []
    search_options = {"k": RAG_RETRIEVAL_FETCH_COUNT}
    vector_filter = build_vector_filter(
        knowledge_type,
        source_filter,
        current_user,
        page_number,
    )
    if vector_filter is not None:
        search_options["filter"] = vector_filter
    return vector_store.similarity_search_with_relevance_scores(query, **search_options)


def build_answer_mode_instruction(question: str) -> str:
    """Choose a concise answer shape before asking the model to generate text.

    A single long medical template made document-location questions sound evasive.
    The rules intentionally remain small and transparent so they can be tuned from
    real evaluation failures without adding another model call.
    """
    normalized_question = question.replace(" ", "")
    requested_page = get_requested_page_number(question)
    if requested_page is not None:
        return (
            f"页码定位模式：只回答用户指定的第 {requested_page} 页资料。"
            "第一句用一句话概括该页的主要内容；随后最多列出 3 个该页已有的重点。"
            "不得补充其他页、其他药物或其他资料的内容。"
        )
    if any(keyword in normalized_question for keyword in DIRECT_LOOKUP_KEYWORDS):
        return (
            "资料定位模式：第一句必须写“推荐资料：《资料标题》”，并直接给出用户要找的"
            "资料标题或专科；其中“资料标题”必须使用本次检索到的第一份资料的真实标题。"
            "随后最多用 1 句话说明该资料为何匹配问题。"
            "除非用户明确要求比较，否则不得列举其他资料、泛化建议或固定免责声明。"
        )
    if is_document_content_question(question):
        return (
            "资料内容模式：第一句必须写“《资料标题》提示：”，并直接回答问题；"
            "其中“资料标题”必须使用本次检索到的真实标题。"
            "整段最多 2 句话，只保留该资料中与问题关键词直接相关的内容。"
            "不要追加泛化建议、重复就医提示或固定免责声明。"
        )
    if any(keyword in normalized_question for keyword in DRUG_QUESTION_KEYWORDS):
        return (
            "药物说明模式：第一句只直接回答用户所问的药物作用或使用前说明。"
            "随后最多补充 2 条与该问题直接相关的资料要点。"
            "不得列举其他药物、泛化的疾病信息或与问题无关的警示。"
            "不要提供资料中没有的剂量、疗程或治疗方案。"
        )
    if any(keyword in normalized_question for keyword in SYMPTOM_QUESTION_KEYWORDS):
        return (
            "症状咨询模式：必须严格按三个部分回答："
            "一、资料内容；二、通用健康建议；三、就医提示。"
            "第一部分只能客观列出资料中的信息，不得把用户症状和疾病建立联系。"
        )
    return (
        "知识问答模式：第一句先直接回答问题，再用最多 2 条资料要点补充。"
        "整段回答最多 3 句话，不得重复相同信息，也不得为展示完整性加入无关背景或安全提示。"
    )


def get_requested_page_number(question: str) -> int | None:
    """Return a requested document page number when the question names one."""
    match = re.search(r"第\s*(\d{1,4})\s*页", question)
    return int(match.group(1)) if match is not None else None


def retrieval_labels(matches: list[tuple[object, float]]) -> tuple[str, str]:
    method = get_retrieval_method(matches)
    rerank_applied = any(
        bool(getattr(document, "metadata", {}).get("rerank_applied"))
        for document, _ in matches
    )
    if rerank_applied and method == "hybrid":
        return (
            "hybrid-rerank-retrieval-openai-generation",
            "rag-hybrid-rerank",
        )
    if rerank_applied and method == "keyword":
        return (
            "keyword-rerank-retrieval-openai-generation",
            "rag-keyword-rerank",
        )
    if rerank_applied:
        return (
            "vector-rerank-retrieval-openai-generation",
            "rag-vector-rerank",
        )
    if method == "hybrid":
        return (
            "hybrid-retrieval-openai-generation",
            "rag-hybrid-retrieval",
        )
    if method == "keyword":
        return (
            "mysql-keyword-retrieval-openai-generation",
            "rag-keyword-retrieval",
        )
    return (
        "milvus-retrieval-openai-generation",
        "rag-vector-retrieval",
    )


def select_title_matched_documents(
    question: str,
    relevant_matches: list[tuple[object, float]],
) -> list[tuple[object, float]]:
    """Select the focused context for explicit document-location questions.

    An exact title match is stronger than retrieval rank. When a user asks for
    one source but does not write its title, the top hybrid result is the best
    available choice. Matching arbitrary two-character title fragments caused
    false positives such as "问题" and "需要", which replaced correct results.
    """
    normalized_question = re.sub(r"\s+", "", question.casefold())
    if is_condition_lookup_question(question):
        condition_matches = [
            match
            for match in relevant_matches
            if str(getattr(match[0], "metadata", {}).get("type", "")) == "condition"
        ]
        if condition_matches:
            return condition_matches[:1]
    if is_specialty_overview_lookup_question(question):
        specialty_matches = [
            match
            for match in relevant_matches
            if is_specialty_overview_title(
                str(getattr(match[0], "metadata", {}).get("name", ""))
            )
        ]
        if specialty_matches:
            return specialty_matches[:1]
    title_matches = [
        match
        for match in relevant_matches
        if _record_title_matches_question(
            str(getattr(match[0], "metadata", {}).get("name", "")),
            normalized_question,
        )
    ]
    if title_matches:
        return title_matches
    if is_explicit_document_lookup_question(question):
        return relevant_matches[:1]
    return relevant_matches


def is_explicit_document_lookup_question(question: str) -> bool:
    """Return whether the user explicitly asks us to locate a source."""
    normalized_question = re.sub(r"\s+", "", question.casefold())
    return (
        get_requested_page_number(question) is not None
        or any(keyword in normalized_question for keyword in DIRECT_LOOKUP_KEYWORDS)
    )


def is_pure_document_lookup_question(question: str) -> bool:
    """Return whether a question only asks us to identify one source."""
    normalized_question = re.sub(r"\s+", "", question.casefold())
    if get_requested_page_number(question) is not None:
        return False
    return any(
        keyword in normalized_question
        for keyword in (
            "哪份",
            "哪个",
            "哪条",
            "哪篇",
            "哪种病症",
            "什么病症",
            "检索哪种",
            "查哪",
            "看哪",
        )
    )


def is_document_content_question(question: str) -> bool:
    """Return whether the user has already named a source and asks its content."""
    normalized_question = re.sub(r"\s+", "", question.casefold())
    return any(marker in normalized_question for marker in DOCUMENT_CONTENT_QUESTION_MARKERS)


def _source_text_lines(document: object) -> list[str]:
    page_content = str(getattr(document, "page_content", ""))
    lines: list[str] = []
    for raw_line in re.split(r"[\n。；]", page_content):
        line = raw_line.strip().lstrip("#-0123456789. ")
        if not line or line.startswith(
            ("资料标题：", "资料来源：", "来源：", "药物名称：", "病症名称：")
        ):
            continue
        for label in ("资料内容：", "常见症状：", "处理建议：", "药物作用：", "使用说明："):
            if line.startswith(label):
                line = line.removeprefix(label).strip()
                break
        lines.append(line)
    return lines


def build_grounded_lookup_reason(question: str, document: object) -> str | None:
    """Extract one source-backed line explaining why a source was selected."""
    terms = [term for term in extract_keyword_terms(question) if len(term) >= 2]
    candidates: list[tuple[int, str]] = []
    for line in _source_text_lines(document):
        normalized_line = re.sub(r"\s+", "", line.casefold())
        matched_terms = {term for term in terms if term in normalized_line}
        if not matched_terms:
            continue
        score = sum(min(len(term), 8) for term in matched_terms)
        candidates.append((score, line))
    if not candidates:
        return None
    _, selected_line = max(candidates, key=lambda item: (item[0], -len(item[1])))
    return selected_line[:180].rstrip("。；，")


def build_lookup_subject(reason: str) -> str:
    """Keep the source-backed clue that should lead a user to this source."""
    subject = reason.split("：", 1)[-1]
    subject = subject.split("，", 1)[0].strip()
    for prefix in ("常见", "可能伴", "可能有", "可有", "出现"):
        if subject.startswith(prefix):
            subject = subject.removeprefix(prefix)
            break
    subject = subject.rstrip("。；，")
    for suffix in ("等表现", "表现", "等症状", "症状"):
        if subject.endswith(suffix):
            return subject[: -len(suffix)].rstrip("、，")
    return subject


def build_document_lookup_answer(
    question: str,
    references: list[dict],
    relevant_documents: list[object],
) -> str:
    """Answer source-selection questions from the selected source, without an LLM."""
    title = str(references[0].get("name", "相关资料"))
    reason = build_grounded_lookup_reason(question, relevant_documents[0])
    if reason:
        subject = build_lookup_subject(reason)
        if subject:
            return (
                f"{subject}等相关表现可查看《{title}》。"
                f"资料中提到：{reason}。"
            )
        return f"资料中提到：{reason}。推荐查看《{title}》。"
    return f"推荐资料：《{title}》。"


def _extract_labeled_document_value(document: object, label: str) -> str:
    page_content = str(getattr(document, "page_content", ""))
    match = re.search(rf"{re.escape(label)}：([^\n]+)", page_content)
    return match.group(1).strip() if match is not None else ""


def build_structured_drug_answer(
    question: str,
    relevant_documents: list[object],
) -> str | None:
    """Answer exact drug questions from the structured source fields directly."""
    normalized_question = re.sub(r"\s+", "", question.casefold())
    drug_document = next(
        (
            document
            for document in relevant_documents
            if str(getattr(document, "metadata", {}).get("type", "")) == "drug"
        ),
        None,
    )
    if drug_document is None:
        return None

    name = str(getattr(drug_document, "metadata", {}).get("name", "该药物"))
    if not (
        any(keyword in normalized_question for keyword in DRUG_QUESTION_KEYWORDS)
        or (
            name.casefold() in normalized_question
            and any(marker in normalized_question for marker in ("资料", "说明", "用前"))
        )
    ):
        return None
    effects = _extract_labeled_document_value(drug_document, "药物作用")
    instructions = _extract_labeled_document_value(drug_document, "使用说明")
    if not effects:
        return None

    answer = f"{name}的作用：{effects}。"
    if instructions:
        answer += f"使用提示：{instructions}。"
    return answer


def build_deterministic_answer(
    question: str,
    references: list[dict],
    relevant_documents: list[object],
) -> str | None:
    """Use source fields directly when generation would add no value."""
    drug_answer = build_structured_drug_answer(question, relevant_documents)
    if drug_answer is not None:
        return drug_answer
    if is_pure_document_lookup_question(question):
        return build_document_lookup_answer(
            question,
            references,
            relevant_documents,
        )
    return None


def is_specialty_overview_lookup_question(question: str) -> bool:
    """Return whether the user asks which medical specialty overview to read."""
    normalized_question = re.sub(r"\s+", "", question.casefold())
    return any(
        keyword in normalized_question
        for keyword in (
            "哪个专科",
            "哪个内科",
            "哪个外科",
            "哪个科",
            "专科概览",
            "外科概览",
        )
    )


def is_condition_lookup_question(question: str) -> bool:
    """Return whether the user asks us to choose a structured condition record."""
    normalized_question = re.sub(r"\s+", "", question.casefold())
    return any(
        keyword in normalized_question
        for keyword in (
            "哪条病症",
            "哪种病症",
            "哪个病症",
            "什么病症",
            "检索哪种",
        )
    )


def is_specialty_overview_title(title: str) -> bool:
    normalized_title = re.sub(r"\s+", "", title.casefold())
    return "科" in normalized_title and "概览" in normalized_title


def _record_title_matches_question(
    title: str,
    normalized_question: str,
) -> bool:
    normalized_title = re.sub(r"\s+", "", title.casefold())
    direct_title = normalized_title.split("：", 1)[0]
    return len(direct_title) >= 2 and direct_title in normalized_question


def select_page_matched_documents(
    question: str,
    relevant_matches: list[tuple[object, float]],
) -> list[tuple[object, float]]:
    """Keep only the requested page when a user explicitly asks about one."""
    requested_page = get_requested_page_number(question)
    if requested_page is None:
        return relevant_matches

    page_matches = [
        match
        for match in relevant_matches
        if str(getattr(match[0], "metadata", {}).get("type", "")) == "document"
        and _document_page_number(match[0]) == requested_page
    ]
    return page_matches or relevant_matches


def _document_page_number(document: object) -> int | None:
    metadata = getattr(document, "metadata", {})
    page_number = parse_metadata_int(metadata.get("page_number"), 1)
    if page_number is not None:
        return page_number
    name = str(metadata.get("name", ""))
    match = re.search(r"第\s*(\d{1,4})\s*页", name)
    return int(match.group(1)) if match is not None else None


def parse_metadata_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_metadata_int(value: object, minimum: int = 0) -> int | None:
    try:
        parsed_value = int(value)
    except (TypeError, ValueError):
        return None
    return parsed_value if parsed_value >= minimum else None


def infer_source_kind(
    knowledge_type: str,
    source: str,
    source_url: str,
) -> str:
    """Return a human-readable source category for a stable citation."""
    if knowledge_type in {"condition", "drug"}:
        return "结构化资料"
    if "PDF" in source.upper():
        return "PDF 文件"
    if "WORD" in source.upper():
        return "Word 文件"
    if "EXCEL" in source.upper():
        return "Excel 文件"
    if "网页" in source or source_url:
        return "网页"
    if "上传文件" in source:
        return "文本文件"
    return "知识文档"


def build_citation_location(
    source_kind: str,
    page_number: int | None,
    chunk_index: int | None,
    chunk_count: int | None,
) -> str:
    location_parts: list[str] = []
    if page_number is not None:
        location_parts.append(f"第 {page_number} 页")
    elif source_kind == "网页":
        location_parts.append("网页正文")
    elif source_kind == "结构化资料":
        location_parts.append("结构化记录")
    else:
        location_parts.append("全文")

    if (
        chunk_index is not None
        and chunk_count is not None
        and chunk_count > 1
    ):
        location_parts.append(f"切块 {chunk_index + 1}/{chunk_count}")
    return " · ".join(location_parts)


def build_references(relevant_matches: list[tuple[object, float]]) -> list[dict]:
    references: list[dict] = []
    for document, score in relevant_matches:
        metadata = document.metadata
        knowledge_type = str(metadata.get("type", "unknown"))
        name = str(metadata.get("name", "未命名资料"))
        source_tier = str(metadata.get("source_tier", "unverified"))
        source = str(metadata.get("source", "未标注来源"))
        source_url = str(metadata.get("source_url", "") or "")
        page_number = parse_metadata_int(metadata.get("page_number"), 1)
        chunk_index = parse_metadata_int(metadata.get("chunk_index"), 0)
        chunk_count = parse_metadata_int(metadata.get("chunk_count"), 1)
        source_kind = infer_source_kind(knowledge_type, source, source_url)
        location = build_citation_location(
            source_kind,
            page_number,
            chunk_index,
            chunk_count,
        )
        citation = f"{name} | {source_kind} | {source} | {location}"
        updated_at = parse_metadata_datetime(metadata.get("updated_at"))
        initial_score = metadata.get("initial_score")
        try:
            initial_score_value = float(initial_score)
        except (TypeError, ValueError):
            initial_score_value = score
        references.append(
            {
                "name": name,
                "type": knowledge_type,
                "record_id": metadata.get("record_id"),
                "source": source,
                "source_url": source_url or None,
                "source_kind": source_kind,
                "location": location,
                "citation": citation,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "source_tier": source_tier,
                "updated_at": updated_at,
                "needs_review": bool(
                    metadata.get(
                        "needs_review",
                        needs_source_review(
                            source_tier,
                            updated_at,
                            source,
                            source_url,
                        ),
                    )
                ),
                "excerpt": document.page_content.replace("\n", " ")[:180],
                # Keep the original field stable; rerank_score is the new
                # second-stage score used for the final ordering.
                "relevance_score": round(initial_score_value, 3),
                "initial_score": metadata.get("initial_score"),
                "rerank_score": metadata.get("rerank_score", round(score, 3)),
                "rerank_text_score": metadata.get("rerank_text_score"),
                "rerank_title_score": metadata.get("rerank_title_score"),
                "retrieval_method": metadata.get("retrieval_method", "vector"),
                "vector_score": metadata.get("vector_score"),
                "keyword_score": metadata.get("keyword_score"),
            }
        )
    return references


def build_source_limitations(references: list[dict]) -> str:
    if any(reference["needs_review"] for reference in references):
        return (
            "部分参考资料的来源待核实、未记录更新时间或已超过一年未更新。"
            "这项状态会展示在引用信息中；只有用户询问资料可靠性、准备据此作重要"
            "医疗决策，或资料不足时，才在回答正文说明这项限制。"
        )
    return "参考资料已标注来源和更新时间。"


def build_response_metadata(
    *,
    source: str,
    references: list[dict],
    processing_path: str,
    retrieval_scope: str,
    source_filter: str,
    retrieved_count: int,
    started_at: float,
) -> dict:
    """Keep the evidence used for an answer available after the page reloads."""
    return {
        "source": source,
        "references": references,
        "processing_path": processing_path,
        "retrieval_scope": retrieval_scope,
        "source_filter": source_filter,
        "retrieved_count": retrieved_count,
        "latency_ms": round((perf_counter() - started_at) * 1000),
    }


def build_safety_references() -> list[dict]:
    return [
        {
            "name": "需要及时就医的警示信号",
            "type": "safety_guard",
            "source": "系统安全规则",
            "source_tier": "professional",
            "updated_at": None,
            "needs_review": False,
            "excerpt": (
                "呼吸困难、持续或加重的胸痛、意识改变、抽搐、"
                "单侧肢体无力、言语含糊或严重过敏表现时，"
                "应立即寻求紧急医疗帮助。"
            ),
            "relevance_score": 1.0,
        }
    ]


def build_rag_answer_prompt() -> ChatPromptTemplate:
    """Return the common prompt used by synchronous, streaming, and RAGAS flows."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是医疗健康知识库助手。只能依据参考资料回答，"
                "不能编造资料中没有的内容。不得根据用户症状判断、推测或声明"
                "用户可能患有任何疾病或感染，也不要使用‘可能是’‘像是’等诊断性表达。"
                "绝不能说用户的症状‘符合’‘相符’‘指向’任何疾病。"
                "不能提供药物剂量或治疗方案。"
                "如果资料不足，要明确说资料不足，并建议咨询医生。"
                "资料可靠性提示：{source_limitations}\n"
                "本次检索到的资料标题：{reference_names}\n"
                "当前回答方式：{answer_mode}\n"
                "只在资料不足、用户描述紧急风险，或当前回答方式明确要求时补充安全提醒；"
                "不要对每次回答机械追加固定免责声明。",
            ),
            (
                "human",
                "历史对话：\n{history}\n\n"
                "参考资料：\n{context}\n\n用户问题：{question}",
            ),
        ]
    )


def run_rag_answer_pipeline(
    session: Session,
    question: str,
    *,
    knowledge_type: str = "all",
    source_filter: str = "all",
    current_user: User | None = None,
    history_text: str = "暂无历史对话",
    retrieval_strategy: str = "hybrid-rerank",
) -> dict:
    """Run the side-effect-free form of the normal first-turn RAG workflow.

    RAGAS calls this function so each sampled case uses the production hybrid
    retrieval, reranking, relevance filtering, prompt, and OpenAI generation
    path without creating user conversation messages.
    """
    started_at = perf_counter()
    retrieval_strategy = normalize_retrieval_strategy(retrieval_strategy)
    urgent_answer = get_urgent_warning_response(question)
    if urgent_answer:
        references = build_safety_references()
        metadata = build_response_metadata(
            source="safety-keyword-guard",
            references=references,
            processing_path="safety-keyword-guard",
            retrieval_scope=knowledge_type,
            source_filter=source_filter,
            retrieved_count=0,
            started_at=started_at,
        )
        return {
            "answer": urgent_answer,
            "contexts": [reference["excerpt"] for reference in references],
            **metadata,
        }

    knowledge_status = get_knowledge_status(session)
    if not knowledge_status["is_current"]:
        raise RuntimeError("知识库已过期，请先执行 POST /knowledge/rebuild")

    try:
        if retrieval_strategy == "none":
            matches = []
        else:
            vector_store = get_vector_store()
            page_number = get_requested_page_number(question)
            matches = search_knowledge(
                vector_store,
                question,
                knowledge_type,
                source_filter,
                current_user,
                session,
                question,
                retrieval_strategy=retrieval_strategy,
                page_number=page_number,
            )
    except Exception as error:
        logger.exception(
            "RAG retrieval failed: strategy=%s question=%s",
            retrieval_strategy,
            question[:200],
        )
        raise RuntimeError(
            "向量检索失败："
            f"{type(error).__name__}: {str(error)[:400]}。"
            "知识库状态正常时无需重建，请检查 Embedding/API 网络配置。"
        ) from error

    relevant_matches = select_distinct_relevant_matches(matches)
    relevant_matches = select_title_matched_documents(question, relevant_matches)
    relevant_matches = select_page_matched_documents(question, relevant_matches)
    relevant_documents = [document for document, _ in relevant_matches]
    if not relevant_documents:
        no_retrieval = retrieval_strategy == "none"
        answer = "知识库中没有找到足够相关的内容。建议换一种更具体的说法。"
        if no_retrieval:
            try:
                response = get_chat_model().invoke(
                    build_rag_answer_prompt().format_messages(
                        context="（本次评测明确关闭检索，没有提供参考资料。）",
                        history=history_text,
                        question=question,
                        source_limitations="本次评测未启用知识库检索。",
                        reference_names="无",
                        answer_mode="无检索基线模式：直接回答问题；如果不确定，明确说明无法确认，不要编造医疗事实。",
                    )
                )
                answer = str(response.content)
            except Exception as error:
                raise RuntimeError("模型暂时不可用，请检查 CHAT_MODEL 和 API 配置") from error
        metadata = build_response_metadata(
            source=(
                "no-retrieval-openai-generation"
                if no_retrieval
                else "milvus-vector-search:no-match"
            ),
            references=[],
            processing_path=(
                "rag-no-retrieval" if no_retrieval else "vector-search-no-match"
            ),
            retrieval_scope=knowledge_type,
            source_filter=source_filter,
            retrieved_count=0,
            started_at=started_at,
        )
        return {
            "answer": answer,
            "contexts": [],
            **metadata,
        }

    context = "\n\n".join(document.page_content for document in relevant_documents)
    references = build_references(relevant_matches)
    direct_answer = build_deterministic_answer(
        question,
        references,
        relevant_documents,
    )
    if direct_answer is not None:
        answer = direct_answer
    else:
        reference_names = "、".join(reference["name"] for reference in references)
        try:
            response = get_chat_model().invoke(
                build_rag_answer_prompt().format_messages(
                    context=context,
                    history=history_text,
                    question=question,
                    source_limitations=build_source_limitations(references),
                    reference_names=reference_names,
                    answer_mode=build_answer_mode_instruction(question),
                )
            )
            answer = str(response.content)
        except Exception as error:
            raise RuntimeError("模型暂时不可用，请检查 CHAT_MODEL 和 API 配置") from error

    retrieval_source, retrieval_path = retrieval_labels(relevant_matches)
    metadata = build_response_metadata(
        source=retrieval_source,
        references=references,
        processing_path=retrieval_path,
        retrieval_scope=knowledge_type,
        source_filter=source_filter,
        retrieved_count=len(relevant_documents),
        started_at=started_at,
    )
    return {
        "answer": answer,
        "contexts": [document.page_content for document in relevant_documents],
        **metadata,
    }


@router.post("", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """同时使用 MySQL 关键词检索和 Milvus 向量检索获取资料。"""
    started_at = perf_counter()
    remember_explicit_user_facts(
        session,
        current_user,
        request.question,
        request.conversation_id,
    )
    history_text, retrieval_memory_context = build_memory_context(
        session,
        current_user,
        request.conversation_id,
    )
    retrieval_query = (
        f"{retrieval_memory_context}\n当前问题：{request.question}"
        if retrieval_memory_context != "暂无历史对话"
        else request.question
    )

    urgent_answer = get_urgent_warning_response(request.question)
    if urgent_answer:
        references = build_safety_references()
        metadata = build_response_metadata(
            source="safety-keyword-guard",
            references=references,
            processing_path="safety-keyword-guard",
            retrieval_scope=request.knowledge_type,
            source_filter=request.source_filter,
            retrieved_count=0,
            started_at=started_at,
        )
        save_message(
            session,
            request.conversation_id,
            "user",
            request.question,
            user_id=current_user.id,
        )
        assistant_message = save_message(
            session,
            request.conversation_id,
            "assistant",
            urgent_answer,
            metadata,
            user_id=current_user.id,
        )
        session.flush()
        finalize_conversation_memory(
            session,
            current_user,
            request.conversation_id,
        )
        record_request_metric(
            session,
            user_id=current_user.id,
            assistant_message_id=assistant_message.id,
            endpoint="ask",
            success=True,
            status_code=200,
            request_type="safety_guard",
            processing_path=metadata["processing_path"],
            retrieved_count=metadata["retrieved_count"],
            latency_ms=metadata["latency_ms"],
        )
        session.commit()
        session.refresh(assistant_message)
        return AskResponse(
            question=request.question,
            answer=urgent_answer,
            source=metadata["source"],
            conversation_id=request.conversation_id,
            assistant_message_id=assistant_message.id,
            references=metadata["references"],
            processing_path=metadata["processing_path"],
            retrieval_scope=metadata["retrieval_scope"],
            source_filter=metadata["source_filter"],
            retrieved_count=metadata["retrieved_count"],
            latency_ms=metadata["latency_ms"],
        )

    knowledge_status = get_knowledge_status(session)
    if not knowledge_status["is_current"]:
        record_request_metric(
            session,
            user_id=current_user.id,
            endpoint="ask",
            success=False,
            status_code=409,
            request_type="error",
            processing_path="knowledge-stale",
            latency_ms=round((perf_counter() - started_at) * 1000),
            error_type="knowledge_stale",
        )
        session.commit()
        raise HTTPException(
            status_code=409,
            detail="知识库已过期，请先执行 POST /knowledge/rebuild",
        )

    try:
        vector_store = get_vector_store()
        page_number = get_requested_page_number(request.question)
        matches = search_knowledge(
            vector_store,
            retrieval_query,
            request.knowledge_type,
            request.source_filter,
            current_user,
            session,
            request.question,
            page_number=page_number,
        )
    except Exception as error:
        record_request_metric(
            session,
            user_id=current_user.id,
            endpoint="ask",
            success=False,
            status_code=503,
            request_type="error",
            processing_path="retrieval-error",
            latency_ms=round((perf_counter() - started_at) * 1000),
            error_type="retrieval_error",
        )
        session.commit()
        raise HTTPException(
            status_code=503,
            detail="知识库暂时不可用，请确认已执行 /knowledge/rebuild",
        ) from error

    relevant_matches = select_distinct_relevant_matches(matches)
    relevant_matches = select_title_matched_documents(
        request.question,
        relevant_matches,
    )
    relevant_matches = select_page_matched_documents(
        request.question,
        relevant_matches,
    )
    relevant_documents = [document for document, _ in relevant_matches]

    if not relevant_documents:
        answer = "知识库中没有找到足够相关的内容。建议换一种更具体的说法。"
        metadata = build_response_metadata(
            source="milvus-vector-search:no-match",
            references=[],
            processing_path="vector-search-no-match",
            retrieval_scope=request.knowledge_type,
            source_filter=request.source_filter,
            retrieved_count=0,
            started_at=started_at,
        )
        save_message(
            session,
            request.conversation_id,
            "user",
            request.question,
            user_id=current_user.id,
        )
        assistant_message = save_message(
            session,
            request.conversation_id,
            "assistant",
            answer,
            metadata,
            user_id=current_user.id,
        )
        session.flush()
        finalize_conversation_memory(
            session,
            current_user,
            request.conversation_id,
        )
        record_request_metric(
            session,
            user_id=current_user.id,
            assistant_message_id=assistant_message.id,
            endpoint="ask",
            success=True,
            status_code=200,
            request_type="no_match",
            processing_path=metadata["processing_path"],
            retrieved_count=metadata["retrieved_count"],
            latency_ms=metadata["latency_ms"],
        )
        session.commit()
        session.refresh(assistant_message)
        return AskResponse(
            question=request.question,
            answer=answer,
            source=metadata["source"],
            conversation_id=request.conversation_id,
            assistant_message_id=assistant_message.id,
            references=metadata["references"],
            processing_path=metadata["processing_path"],
            retrieval_scope=metadata["retrieval_scope"],
            source_filter=metadata["source_filter"],
            retrieved_count=metadata["retrieved_count"],
            latency_ms=metadata["latency_ms"],
        )

    context = "\n\n".join(
        document.page_content
        for document in relevant_documents
    )
    references = build_references(relevant_matches)
    direct_answer = build_deterministic_answer(
        request.question,
        references,
        relevant_documents,
    )
    if direct_answer is not None:
        answer = direct_answer
    else:
        reference_names = "、".join(reference["name"] for reference in references)
        prompt = build_rag_answer_prompt()
        try:
            response = get_chat_model().invoke(
                prompt.format_messages(
                    context=context,
                    history=history_text,
                    question=request.question,
                    source_limitations=build_source_limitations(references),
                    reference_names=reference_names,
                    answer_mode=build_answer_mode_instruction(request.question),
                )
            )
            answer = str(response.content)
        except Exception as error:
            record_request_metric(
                session,
                user_id=current_user.id,
                endpoint="ask",
                success=False,
                status_code=503,
                request_type="rag",
                processing_path="generation-error",
                retrieved_count=len(relevant_documents),
                latency_ms=round((perf_counter() - started_at) * 1000),
                error_type="generation_error",
            )
            session.commit()
            raise HTTPException(
                status_code=503,
                detail="模型暂时不可用，请检查 CHAT_MODEL 和 API 配置",
            ) from error

    retrieval_source, retrieval_path = retrieval_labels(relevant_matches)
    metadata = build_response_metadata(
        source=retrieval_source,
        references=references,
        processing_path=retrieval_path,
        retrieval_scope=request.knowledge_type,
        source_filter=request.source_filter,
        retrieved_count=len(relevant_documents),
        started_at=started_at,
    )
    save_message(
        session,
        request.conversation_id,
        "user",
        request.question,
        user_id=current_user.id,
    )
    assistant_message = save_message(
        session,
        request.conversation_id,
        "assistant",
        answer,
        metadata,
        user_id=current_user.id,
    )
    session.flush()
    finalize_conversation_memory(
        session,
        current_user,
        request.conversation_id,
    )
    record_request_metric(
        session,
        user_id=current_user.id,
        assistant_message_id=assistant_message.id,
        endpoint="ask",
        success=True,
        status_code=200,
        request_type="rag",
        processing_path=metadata["processing_path"],
        retrieved_count=metadata["retrieved_count"],
        latency_ms=metadata["latency_ms"],
    )
    session.commit()
    session.refresh(assistant_message)

    return AskResponse(
        question=request.question,
        answer=answer,
        source=metadata["source"],
        conversation_id=request.conversation_id,
        assistant_message_id=assistant_message.id,
        references=metadata["references"],
        processing_path=metadata["processing_path"],
        retrieval_scope=metadata["retrieval_scope"],
        source_filter=metadata["source_filter"],
        retrieved_count=metadata["retrieved_count"],
        latency_ms=metadata["latency_ms"],
    )


@router.post("/stream")
def stream_answer(
    request: AskRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """Use SSE to return answer chunks while preserving the normal RAG rules."""
    started_at = perf_counter()
    remember_explicit_user_facts(
        session,
        current_user,
        request.question,
        request.conversation_id,
    )
    history_text, retrieval_memory_context = build_memory_context(
        session,
        current_user,
        request.conversation_id,
    )
    retrieval_query = (
        f"{retrieval_memory_context}\n当前问题：{request.question}"
        if retrieval_memory_context != "暂无历史对话"
        else request.question
    )

    urgent_answer = get_urgent_warning_response(request.question)
    if urgent_answer:
        def urgent_event_stream():
            metadata = build_response_metadata(
                source="safety-keyword-guard",
                references=build_safety_references(),
                processing_path="safety-keyword-guard",
                retrieval_scope=request.knowledge_type,
                source_filter=request.source_filter,
                retrieved_count=0,
                started_at=started_at,
            )
            save_message(
                session,
                request.conversation_id,
                "user",
                request.question,
                user_id=current_user.id,
            )
            assistant_message = save_message(
                session,
                request.conversation_id,
                "assistant",
                urgent_answer,
                metadata,
                user_id=current_user.id,
            )
            session.flush()
            finalize_conversation_memory(
                session,
                current_user,
                request.conversation_id,
            )
            record_request_metric(
                session,
                user_id=current_user.id,
                assistant_message_id=assistant_message.id,
                endpoint="ask/stream",
                success=True,
                status_code=200,
                request_type="safety_guard",
                processing_path=metadata["processing_path"],
                retrieved_count=metadata["retrieved_count"],
                latency_ms=metadata["latency_ms"],
            )
            session.commit()
            session.refresh(assistant_message)
            yield format_sse_event(
                "metadata",
                metadata,
            )
            yield format_sse_event("token", {"text": urgent_answer})
            yield format_sse_event(
                "done",
                {"assistant_message_id": assistant_message.id, **metadata},
            )

        return StreamingResponse(
            urgent_event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    knowledge_status = get_knowledge_status(session)
    if not knowledge_status["is_current"]:
        record_request_metric(
            session,
            user_id=current_user.id,
            endpoint="ask/stream",
            success=False,
            status_code=409,
            request_type="error",
            processing_path="knowledge-stale",
            latency_ms=round((perf_counter() - started_at) * 1000),
            error_type="knowledge_stale",
        )
        session.commit()
        raise HTTPException(
            status_code=409,
            detail="知识库已过期，请先执行 POST /knowledge/rebuild",
        )

    try:
        vector_store = get_vector_store()
        page_number = get_requested_page_number(request.question)
        matches = search_knowledge(
            vector_store,
            retrieval_query,
            request.knowledge_type,
            request.source_filter,
            current_user,
            session,
            request.question,
            page_number=page_number,
        )
    except Exception as error:
        record_request_metric(
            session,
            user_id=current_user.id,
            endpoint="ask/stream",
            success=False,
            status_code=503,
            request_type="error",
            processing_path="retrieval-error",
            latency_ms=round((perf_counter() - started_at) * 1000),
            error_type="retrieval_error",
        )
        session.commit()
        raise HTTPException(
            status_code=503,
            detail="知识库暂时不可用，请确认已执行 /knowledge/rebuild",
        ) from error

    relevant_matches = select_distinct_relevant_matches(matches)
    relevant_matches = select_title_matched_documents(
        request.question,
        relevant_matches,
    )
    relevant_matches = select_page_matched_documents(
        request.question,
        relevant_matches,
    )
    relevant_documents = [document for document, _ in relevant_matches]

    if not relevant_documents:
        no_match_answer = "知识库中没有找到足够相关的内容。建议换一种更具体的说法。"

        def no_match_event_stream():
            metadata = build_response_metadata(
                source="milvus-vector-search:no-match",
                references=[],
                processing_path="vector-search-no-match",
                retrieval_scope=request.knowledge_type,
                source_filter=request.source_filter,
                retrieved_count=0,
                started_at=started_at,
            )
            save_message(
                session,
                request.conversation_id,
                "user",
                request.question,
                user_id=current_user.id,
            )
            assistant_message = save_message(
                session,
                request.conversation_id,
                "assistant",
                no_match_answer,
                metadata,
                user_id=current_user.id,
            )
            session.flush()
            finalize_conversation_memory(
                session,
                current_user,
                request.conversation_id,
            )
            record_request_metric(
                session,
                user_id=current_user.id,
                assistant_message_id=assistant_message.id,
                endpoint="ask/stream",
                success=True,
                status_code=200,
                request_type="no_match",
                processing_path=metadata["processing_path"],
                retrieved_count=metadata["retrieved_count"],
                latency_ms=metadata["latency_ms"],
            )
            session.commit()
            session.refresh(assistant_message)
            yield format_sse_event(
                "metadata",
                metadata,
            )
            yield format_sse_event("token", {"text": no_match_answer})
            yield format_sse_event(
                "done",
                {"assistant_message_id": assistant_message.id, **metadata},
            )

        return StreamingResponse(
            no_match_event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    context = "\n\n".join(document.page_content for document in relevant_documents)
    references = build_references(relevant_matches)
    retrieval_source, retrieval_path = retrieval_labels(relevant_matches)
    direct_answer = build_deterministic_answer(
        request.question,
        references,
        relevant_documents,
    )
    messages = None
    if direct_answer is None:
        reference_names = "、".join(reference["name"] for reference in references)
        messages = build_rag_answer_prompt().format_messages(
            context=context,
            history=history_text,
            question=request.question,
            source_limitations=build_source_limitations(references),
            reference_names=reference_names,
            answer_mode=build_answer_mode_instruction(request.question),
        )

    def answer_event_stream():
        initial_metadata = build_response_metadata(
            source=retrieval_source,
            references=references,
            processing_path=retrieval_path,
            retrieval_scope=request.knowledge_type,
            source_filter=request.source_filter,
            retrieved_count=len(relevant_documents),
            started_at=started_at,
        )
        yield format_sse_event(
            "metadata",
            initial_metadata,
        )
        answer_parts: list[str] = []
        if direct_answer is not None:
            answer_parts.append(direct_answer)
            yield format_sse_event("token", {"text": direct_answer})
        else:
            try:
                for chunk in get_chat_model().stream(messages):
                    text = get_stream_text(chunk)
                    if not text:
                        continue
                    answer_parts.append(text)
                    yield format_sse_event("token", {"text": text})
            except Exception:
                record_request_metric(
                    session,
                    user_id=current_user.id,
                    endpoint="ask/stream",
                    success=False,
                    status_code=503,
                    request_type="rag",
                    processing_path="generation-error",
                    retrieved_count=len(relevant_documents),
                    latency_ms=round((perf_counter() - started_at) * 1000),
                    error_type="generation_error",
                )
                session.commit()
                yield format_sse_event(
                    "error",
                    {"detail": "模型暂时不可用，请检查 CHAT_MODEL 和 API 配置"},
                )
                return

        answer = "".join(answer_parts).strip()
        if not answer:
            record_request_metric(
                session,
                user_id=current_user.id,
                endpoint="ask/stream",
                success=False,
                status_code=502,
                request_type="rag",
                processing_path="generation-empty",
                retrieved_count=len(relevant_documents),
                latency_ms=round((perf_counter() - started_at) * 1000),
                error_type="empty_generation",
            )
            session.commit()
            yield format_sse_event("error", {"detail": "模型没有返回可用内容"})
            return
        metadata = build_response_metadata(
            source=retrieval_source,
            references=references,
            processing_path=retrieval_path,
            retrieval_scope=request.knowledge_type,
            source_filter=request.source_filter,
            retrieved_count=len(relevant_documents),
            started_at=started_at,
        )
        save_message(
            session,
            request.conversation_id,
            "user",
            request.question,
            user_id=current_user.id,
        )
        assistant_message = save_message(
            session,
            request.conversation_id,
            "assistant",
            answer,
            metadata,
            user_id=current_user.id,
        )
        session.flush()
        finalize_conversation_memory(
            session,
            current_user,
            request.conversation_id,
        )
        record_request_metric(
            session,
            user_id=current_user.id,
            assistant_message_id=assistant_message.id,
            endpoint="ask/stream",
            success=True,
            status_code=200,
            request_type="rag",
            processing_path=metadata["processing_path"],
            retrieved_count=metadata["retrieved_count"],
            latency_ms=metadata["latency_ms"],
        )
        session.commit()
        session.refresh(assistant_message)
        yield format_sse_event(
            "done",
            {"assistant_message_id": assistant_message.id, **metadata},
        )

    return StreamingResponse(
        answer_event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
