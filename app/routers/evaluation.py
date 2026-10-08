import json
import logging
import math
import os
from datetime import UTC, datetime
from random import SystemRandom
from threading import Lock

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlmodel import Session, select

from app.database import engine, get_session
from app.hybrid_search import hybrid_search, normalize_retrieval_strategy
from app.legacy.models import (
    Condition,
    Drug,
    KnowledgeDocument,
    KnowledgeIndexState,
    RAGASAutoEvaluationRun,
    RAGEvaluationCase,
    RAGEvaluationRun,
    RAGQualityEvaluationRun,
)
from app.ragas_compat import enable_ragas_langchain_compatibility
from app.ragas_metrics import build_medical_answer_relevancy_metric
from app.routers.ask import get_chat_model, run_rag_answer_pipeline
from app.routers.auth import require_admin
from app.legacy.schemas import (
    RAGASAutoEvaluationRunListResponse,
    RAGASAutoEvaluationRunRead,
    RAGASMetricScores,
    RAGASQuestionResult,
    RAGEvaluationCaseCreate,
    RAGEvaluationCaseListResponse,
    RAGEvaluationCaseRead,
    RAGEvaluationCaseResult,
    RAGEvaluationCategoryMetric,
    RAGEvaluationHistoryRead,
    RAGEvaluationHistoryResponse,
    RAGEvaluationQualityGate,
    RAGEvaluationResponse,
    RAGQualityCaseResult,
    RAGQualityResponse,
    RetrievalDiagnosticCandidate,
    RetrievalDiagnosticCaseResult,
    RetrievalDiagnosticResponse,
    RetrievalComparisonCaseResult,
    RetrievalComparisonResponse,
    RetrievalEvaluationMetrics,
    RetrievalStrategyResult,
    RetrievalStrategySummary,
)
from app.vector_store import (
    MIN_RELEVANCE_SCORE,
    RAG_RETRIEVAL_FETCH_COUNT,
    RAG_RETRIEVAL_RESULT_COUNT,
    get_embeddings,
    get_knowledge_status,
    get_vector_store,
    select_distinct_relevant_matches,
)


router = APIRouter(
    prefix="/evaluation",
    tags=["evaluation"],
    dependencies=[Depends(require_admin)],
)
BASELINE_RETRIEVAL_COUNT = 3
DEFAULT_RETRIEVAL_STRATEGY = "hybrid-rerank"

# All evaluation cases are user-managed records stored in MySQL. Keeping the
# defaults empty ensures every metric reflects the current 100-question set.
EVALUATION_CASES: tuple[dict, ...] = ()
QUALITY_ONLY_CASES: tuple[dict, ...] = ()
QUALITY_CASE_CONFIG: dict[str, dict] = {}
RAGAS_DEFAULT_SAMPLE_SIZE = 10
RAGAS_MAX_SAMPLE_SIZE = 100
RAGAS_REFERENCE_MAX_CHARS = 6000
RAGAS_JOB_TIMEOUT_SECONDS = int(
    os.getenv("RAGAS_JOB_TIMEOUT_SECONDS", str(60 * 60))
)
ACTIVE_RAGAS_STATUSES = {"pending", "running"}
ragas_start_lock = Lock()
ragas_execution_lock = Lock()
logger = logging.getLogger(__name__)


def get_alternative_names(case: dict) -> list[str]:
    raw_names = case.get("alternative_names", [])
    if not isinstance(raw_names, list):
        return []
    return [name for name in raw_names if isinstance(name, str)]


def get_case_category(case: dict) -> str:
    category = case.get("category")
    return category.strip() if isinstance(category, str) and category.strip() else "未分类"


def normalize_evaluation_case(case: dict) -> dict:
    normalized_case = dict(case)
    normalized_case["category"] = get_case_category(case)
    normalized_case["alternative_names"] = get_alternative_names(case)
    configured_quality = QUALITY_CASE_CONFIG.get(case.get("case_id"), {})
    answer_keywords = case.get(
        "answer_keywords",
        configured_quality.get("answer_keywords", []),
    )
    citation_names = case.get(
        "citation_names",
        configured_quality.get("citation_names", []),
    )
    normalized_case["answer_keywords"] = (
        answer_keywords if isinstance(answer_keywords, list) else []
    )
    normalized_case["citation_names"] = (
        citation_names if isinstance(citation_names, list) else []
    )
    normalized_case["expected_refusal"] = bool(case.get("expected_refusal", False))
    return normalized_case


def get_acceptable_names(case: dict) -> set[str]:
    return {str(case["expected_name"]), *get_alternative_names(case)}


def validate_retrieval_strategy(value: str) -> str:
    try:
        return normalize_retrieval_strategy(value)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def build_strategy_result(
    matches: list[tuple[object, float]],
    case: dict,
) -> RetrievalStrategyResult:
    top_name = None
    top_type = None
    top_score = None
    expected_rank = None
    matched_name = None
    relevant_count = 0
    retrieved_count = 0
    acceptable_names = get_acceptable_names(case)

    for rank, (document, score) in enumerate(matches, start=1):
        if rank == 1:
            top_name = document.metadata.get("name")
            top_type = document.metadata.get("type")
            top_score = round(score, 3)
        if score < MIN_RELEVANCE_SCORE:
            continue
        retrieved_count += 1
        is_expected_result = (
            document.metadata.get("name") in acceptable_names
            and document.metadata.get("type") == case["expected_type"]
        )
        if is_expected_result:
            relevant_count += 1
            if expected_rank is None:
                expected_rank = rank
                matched_name = str(document.metadata.get("name"))

    return RetrievalStrategyResult(
        passed=expected_rank is not None,
        expected_rank=expected_rank,
        matched_name=matched_name,
        top_name=top_name,
        top_type=top_type,
        top_score=top_score,
        retrieved_count=retrieved_count,
        relevant_count=relevant_count,
    )


def evaluate_baseline_case(
    session: Session,
    vector_store: object,
    case: dict,
    retrieval_strategy: str = "vector",
) -> RetrievalStrategyResult:
    matches = get_strategy_matches(
        session,
        vector_store,
        case,
        retrieval_strategy,
        BASELINE_RETRIEVAL_COUNT,
    )
    return build_strategy_result(matches, case)


def evaluate_current_case(
    session: Session,
    vector_store: object,
    case: dict,
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
) -> RetrievalStrategyResult:
    return build_strategy_result(
        get_current_matches(session, vector_store, case, retrieval_strategy),
        case,
    )


def get_strategy_matches(
    session: Session,
    vector_store: object,
    case: dict,
    retrieval_strategy: str,
    fetch_count: int,
) -> list[tuple[object, float]]:
    """Run exactly the selected retrieval stages for an evaluation case."""
    return hybrid_search(
        session,
        vector_store,
        case["question"],
        knowledge_type="all",
        source_filter="all",
        vector_fetch_count=fetch_count,
        retrieval_strategy=retrieval_strategy,
    )


def get_current_matches(
    session: Session,
    vector_store: object,
    case: dict,
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
) -> list[tuple[object, float]]:
    # This must follow the same retrieval path as /ask. Otherwise the
    # evaluation page would report vector-only results for a hybrid RAG API.
    matches = get_strategy_matches(
        session,
        vector_store,
        case,
        retrieval_strategy,
        RAG_RETRIEVAL_FETCH_COUNT,
    )
    return select_distinct_relevant_matches(
        matches,
        limit=RAG_RETRIEVAL_RESULT_COUNT,
    )


def evaluate_case(
    session: Session,
    vector_store: object,
    case: dict,
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
) -> RAGEvaluationCaseResult:
    result = evaluate_current_case(
        session,
        vector_store,
        case,
        retrieval_strategy,
    )
    return RAGEvaluationCaseResult(**case, **result.model_dump())


def build_retrieval_metrics(
    results: list[RetrievalStrategyResult | RAGEvaluationCaseResult],
) -> RetrievalEvaluationMetrics:
    total_count = len(results)
    top1_correct_count = sum(result.expected_rank == 1 for result in results)
    recalled_count = sum(result.passed for result in results)
    relevant_result_count = sum(result.relevant_count for result in results)
    retrieved_result_count = sum(result.retrieved_count for result in results)
    return RetrievalEvaluationMetrics(
        top1_correct_count=top1_correct_count,
        top1_accuracy=top1_correct_count / total_count if total_count else 0,
        recalled_count=recalled_count,
        recall_at_3=recalled_count / total_count if total_count else 0,
        relevant_result_count=relevant_result_count,
        retrieved_result_count=retrieved_result_count,
        precision_at_3=(
            relevant_result_count / retrieved_result_count
            if retrieved_result_count
            else 0
        ),
    )


def compare_case(
    session: Session,
    vector_store: object,
    case: dict,
    baseline_strategy: str = "vector",
    current_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
) -> RetrievalComparisonCaseResult:
    baseline = evaluate_baseline_case(
        session,
        vector_store,
        case,
        baseline_strategy,
    )
    current = evaluate_current_case(
        session,
        vector_store,
        case,
        current_strategy,
    )
    baseline_rank = baseline.expected_rank or (RAG_RETRIEVAL_RESULT_COUNT + 1)
    current_rank = current.expected_rank or (RAG_RETRIEVAL_RESULT_COUNT + 1)
    if current_rank < baseline_rank:
        change = "improved"
    elif current_rank > baseline_rank:
        change = "regressed"
    else:
        change = "unchanged"
    return RetrievalComparisonCaseResult(
        **case,
        baseline=baseline,
        current=current,
        change=change,
    )


def get_match_name(document: object) -> str:
    return str(getattr(document, "metadata", {}).get("name", "未命名资料"))


def get_match_type(document: object) -> str:
    return str(getattr(document, "metadata", {}).get("type", "unknown"))


def build_diagnostic_case(
    session: Session,
    vector_store: object,
    case: dict,
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
) -> RetrievalDiagnosticCaseResult:
    raw_matches = get_strategy_matches(
        session,
        vector_store,
        case,
        retrieval_strategy,
        RAG_RETRIEVAL_FETCH_COUNT,
    )
    selected_matches = select_distinct_relevant_matches(
        raw_matches,
        limit=RAG_RETRIEVAL_RESULT_COUNT,
    )
    strategy_result = build_strategy_result(selected_matches, case)
    candidates = [
        RetrievalDiagnosticCandidate(
            rank=rank,
            name=get_match_name(document),
            type=get_match_type(document),
            relevance_score=round(score, 3),
        )
        for rank, (document, score) in enumerate(selected_matches, start=1)
    ]
    acceptable_names = get_acceptable_names(case)
    expected_raw_match = next(
        (
            (rank, score, get_match_name(document))
            for rank, (document, score) in enumerate(raw_matches, start=1)
            if get_match_name(document) in acceptable_names
            and get_match_type(document) == case["expected_type"]
        ),
        None,
    )

    if strategy_result.expected_rank == 1:
        diagnostic_level = "healthy"
        diagnostic = (
            f"可接受资料“{strategy_result.matched_name}”位于首位，"
            "当前检索表现正常。"
        )
        suggested_action = "暂不需要为这道题调整资料或检索参数。"
    elif strategy_result.expected_rank is not None:
        diagnostic_level = "attention"
        diagnostic = (
            f"可接受资料“{strategy_result.matched_name}”命中第 "
            f"{strategy_result.expected_rank} 条，"
            "能被找到，但前面还有更相近的资料。"
        )
        suggested_action = (
            "可在目标资料标题或正文中补充用户会使用的关键词，"
            "再运行评测确认排名变化。"
        )
    elif expected_raw_match is None:
        diagnostic_level = "failed"
        diagnostic = "目标资料没有进入前 8 个向量候选，关键词检索也没有补充命中。"
        suggested_action = (
            "检查目标资料是否缺少问题中的核心词，或补充更直接对应的资料内容。"
        )
    elif expected_raw_match[1] < MIN_RELEVANCE_SCORE:
        diagnostic_level = "failed"
        diagnostic = (
            f"可接受资料“{expected_raw_match[2]}”进入候选的第 "
            f"{expected_raw_match[0]} 条，"
            f"但相关度 {expected_raw_match[1]:.3f} 低于阈值 {MIN_RELEVANCE_SCORE}。"
        )
        suggested_action = (
            "在资料标题和正文中补充问题常用的表达，再重建知识库并运行评测。"
        )
    else:
        diagnostic_level = "failed"
        diagnostic = (
            f"可接受资料“{expected_raw_match[2]}”进入候选的第 "
            f"{expected_raw_match[0]} 条，"
            "但被更高相关度的不同资料挤出了最终前 3 条。"
        )
        suggested_action = (
            "补充目标资料的关键词或内容；若业务确实需要更多来源，可再评估是否调整最终保留数量。"
        )

    return RetrievalDiagnosticCaseResult(
        **case,
        passed=strategy_result.passed,
        expected_rank=strategy_result.expected_rank,
        matched_name=strategy_result.matched_name,
        diagnostic_level=diagnostic_level,
        diagnostic=diagnostic,
        suggested_action=suggested_action,
        candidates=candidates,
    )


def get_record_alternative_names(record: RAGEvaluationCase) -> list[str]:
    try:
        parsed_names = json.loads(record.alternative_names_json)
    except json.JSONDecodeError:
        return []
    return parsed_names if isinstance(parsed_names, list) else []


def get_record_terms(record: RAGEvaluationCase, field_name: str) -> list[str]:
    try:
        parsed_terms = json.loads(getattr(record, field_name))
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed_terms if isinstance(parsed_terms, list) else []


def get_run_result_snapshot(run: RAGEvaluationRun) -> list[dict]:
    try:
        snapshot = json.loads(run.results_json)
    except (json.JSONDecodeError, TypeError):
        return []
    return snapshot if isinstance(snapshot, list) else []


def get_latest_result_snapshot(
    session: Session,
) -> tuple[RAGEvaluationRun | None, list[dict]]:
    runs = session.exec(
        select(RAGEvaluationRun).order_by(RAGEvaluationRun.created_at.desc())
    ).all()
    for run in runs:
        snapshot = get_run_result_snapshot(run)
        if snapshot:
            return run, snapshot
    return None, []


def build_result_snapshot(results: list[RAGEvaluationCaseResult]) -> list[dict]:
    return [
        {
            "case_id": result.case_id,
            "question": result.question,
            "category": result.category,
            "passed": result.passed,
            "expected_rank": result.expected_rank,
        }
        for result in results
    ]


def build_category_metrics(
    results: list[RAGEvaluationCaseResult],
) -> list[RAGEvaluationCategoryMetric]:
    totals: dict[str, int] = {}
    passed_counts: dict[str, int] = {}
    for result in results:
        category = result.category
        totals[category] = totals.get(category, 0) + 1
        passed_counts[category] = passed_counts.get(category, 0) + int(result.passed)

    return [
        RAGEvaluationCategoryMetric(
            category=category,
            total_count=total,
            passed_count=passed_counts[category],
            pass_rate=passed_counts[category] / total,
        )
        for category, total in totals.items()
    ]


def build_quality_gate(
    results: list[RAGEvaluationCaseResult],
    previous_run: RAGEvaluationRun | None,
    previous_snapshot: list[dict],
) -> RAGEvaluationQualityGate:
    if previous_run is None:
        return RAGEvaluationQualityGate(
            status="baseline",
            message="已保存本次评测明细，下一次运行时将自动检查检索能力是否回退。",
        )

    previous_results = {
        item.get("case_id"): item
        for item in previous_snapshot
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }
    regressed_questions: list[str] = []
    improved_questions: list[str] = []
    new_questions: list[str] = []
    rank_regressed_questions: list[str] = []
    rank_improved_questions: list[str] = []
    for result in results:
        previous_result = previous_results.get(result.case_id)
        if previous_result is None:
            new_questions.append(result.question)
            continue
        previous_passed = previous_result.get("passed")
        if previous_passed is True and not result.passed:
            regressed_questions.append(result.question)
        elif previous_passed is False and result.passed:
            improved_questions.append(result.question)
        else:
            previous_rank = previous_result.get("expected_rank")
            current_rank = result.expected_rank
            if (
                previous_passed is True
                and result.passed
                and isinstance(previous_rank, int)
                and current_rank is not None
            ):
                if current_rank > previous_rank:
                    rank_regressed_questions.append(result.question)
                elif current_rank < previous_rank:
                    rank_improved_questions.append(result.question)

    if regressed_questions:
        return RAGEvaluationQualityGate(
            status="warning",
            compared_history_id=previous_run.id,
            message=(
                f"发现 {len(regressed_questions)} 道题从通过变为未通过，"
                "建议检查本次知识库或检索改动。"
            ),
            regressed_questions=regressed_questions,
            improved_questions=improved_questions,
            new_questions=new_questions,
            rank_regressed_questions=rank_regressed_questions,
            rank_improved_questions=rank_improved_questions,
        )
    if rank_regressed_questions:
        return RAGEvaluationQualityGate(
            status="attention",
            compared_history_id=previous_run.id,
            message=(
                f"有 {len(rank_regressed_questions)} 道题仍然通过，"
                "但目标资料的排名下降了。"
            ),
            rank_regressed_questions=rank_regressed_questions,
            rank_improved_questions=rank_improved_questions,
            new_questions=new_questions,
        )
    if improved_questions or rank_improved_questions:
        message = (
            f"本次有 {len(improved_questions)} 道题从未通过变为通过，"
            "且没有发现回退。"
            if improved_questions
            else (
                f"本次有 {len(rank_improved_questions)} 道题的目标资料排名提升，"
                "且没有发现回退。"
            )
        )
        return RAGEvaluationQualityGate(
            status="improved",
            compared_history_id=previous_run.id,
            message=message,
            improved_questions=improved_questions,
            new_questions=new_questions,
            rank_improved_questions=rank_improved_questions,
        )
    if new_questions:
        return RAGEvaluationQualityGate(
            status="expanded",
            compared_history_id=previous_run.id,
            message=(
                f"本次新增 {len(new_questions)} 道评测题；"
                "已有题没有发现通过状态回退。"
            ),
            new_questions=new_questions,
        )
    return RAGEvaluationQualityGate(
        status="stable",
        compared_history_id=previous_run.id,
        message="与上一次保存的评测明细相比，没有发现通过状态回退。",
    )


def serialize_evaluation_case(record: RAGEvaluationCase) -> RAGEvaluationCaseRead:
    return RAGEvaluationCaseRead(
        id=record.id,
        question=record.question,
        expected_name=record.expected_name,
        expected_type=record.expected_type,
        category=record.category,
        alternative_names=get_record_alternative_names(record),
        answer_keywords=get_record_terms(record, "answer_keywords_json"),
        citation_names=get_record_terms(record, "citation_names_json"),
        expected_refusal=record.expected_refusal,
        created_at=record.created_at,
    )


def build_custom_case(record: RAGEvaluationCase) -> dict:
    return {
        "case_id": f"custom-{record.id}",
        "case_source": "自定义题",
        "question": record.question,
        "expected_name": record.expected_name,
        "expected_type": record.expected_type,
        "category": record.category,
        "alternative_names": get_record_alternative_names(record),
        "answer_keywords": get_record_terms(record, "answer_keywords_json"),
        "citation_names": get_record_terms(record, "citation_names_json"),
        "expected_refusal": record.expected_refusal,
    }


def get_evaluation_cases(session: Session) -> tuple[list[dict], int]:
    custom_cases = session.exec(
        select(RAGEvaluationCase).order_by(RAGEvaluationCase.created_at)
    ).all()
    cases = [
        *(normalize_evaluation_case(case) for case in EVALUATION_CASES),
        *(
            normalize_evaluation_case(build_custom_case(case))
            for case in custom_cases
            if not case.expected_refusal
        ),
    ]
    return cases, sum(not case.expected_refusal for case in custom_cases)


def get_quality_evaluation_cases(session: Session) -> tuple[list[dict], int]:
    custom_cases = session.exec(
        select(RAGEvaluationCase).order_by(RAGEvaluationCase.created_at)
    ).all()
    cases = [
        *(normalize_evaluation_case(case) for case in EVALUATION_CASES),
        *(normalize_evaluation_case(case) for case in QUALITY_ONLY_CASES),
        *(normalize_evaluation_case(build_custom_case(case)) for case in custom_cases),
    ]
    return cases, len(custom_cases)


@router.get("/cases", response_model=RAGEvaluationCaseListResponse)
def list_evaluation_cases(session: Session = Depends(get_session)):
    cases = session.exec(
        select(RAGEvaluationCase).order_by(RAGEvaluationCase.created_at.desc())
    ).all()
    return RAGEvaluationCaseListResponse(
        total_count=len(cases),
        cases=[serialize_evaluation_case(case) for case in cases],
    )


@router.post(
    "/cases",
    response_model=RAGEvaluationCaseRead,
    status_code=status.HTTP_201_CREATED,
)
def create_evaluation_case(
    payload: RAGEvaluationCaseCreate,
    session: Session = Depends(get_session),
):
    payload_data = payload.model_dump()
    alternative_names = payload_data.pop("alternative_names")
    answer_keywords = payload_data.pop("answer_keywords")
    citation_names = payload_data.pop("citation_names")
    case = RAGEvaluationCase(
        **payload_data,
        alternative_names_json=json.dumps(alternative_names, ensure_ascii=False),
        answer_keywords_json=json.dumps(answer_keywords, ensure_ascii=False),
        citation_names_json=json.dumps(citation_names, ensure_ascii=False),
    )
    session.add(case)
    session.commit()
    session.refresh(case)
    return serialize_evaluation_case(case)


@router.delete("/cases/{case_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_evaluation_case(case_id: int, session: Session = Depends(get_session)):
    case = session.get(RAGEvaluationCase, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="自定义评测题不存在")
    session.delete(case)
    session.commit()


@router.get("/history", response_model=RAGEvaluationHistoryResponse)
def list_evaluation_history(
    limit: int = 20,
    session: Session = Depends(get_session),
):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="历史记录数量必须在 1 到 100 之间")
    runs = session.exec(
        select(RAGEvaluationRun)
        .order_by(RAGEvaluationRun.created_at.desc())
        .limit(limit)
    ).all()
    return RAGEvaluationHistoryResponse(
        total_count=len(session.exec(select(RAGEvaluationRun)).all()),
        runs=[RAGEvaluationHistoryRead.model_validate(run) for run in runs],
    )


@router.post("/run", response_model=RAGEvaluationResponse)
def run_rag_evaluation(
    retrieval_strategy: str = Query(
        default=DEFAULT_RETRIEVAL_STRATEGY,
        description="评测使用的检索策略",
    ),
    session: Session = Depends(get_session),
):
    retrieval_strategy = validate_retrieval_strategy(retrieval_strategy)
    knowledge_status = get_knowledge_status(session)
    if not knowledge_status["is_current"]:
        raise HTTPException(
            status_code=409,
            detail="知识库已过期，请先重建知识库后再运行评测",
        )

    try:
        vector_store = (
            None
            if retrieval_strategy == "none"
            else get_vector_store()
        )
        cases, custom_count = get_evaluation_cases(session)
        results = [
            evaluate_case(session, vector_store, case, retrieval_strategy)
            for case in cases
        ]
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="RAG 评测暂时不可用，请检查知识库和 Embedding 配置",
        ) from error

    passed_count = sum(result.passed for result in results)
    total_count = len(results)
    pass_rate = passed_count / total_count if total_count else 0
    knowledge_index = session.get(KnowledgeIndexState, 1)
    previous_run, previous_snapshot = get_latest_result_snapshot(session)
    quality_gate = build_quality_gate(results, previous_run, previous_snapshot)
    run = RAGEvaluationRun(
        total_count=total_count,
        passed_count=passed_count,
        pass_rate=pass_rate,
        preset_count=len(EVALUATION_CASES),
        custom_count=custom_count,
        retrieval_strategy=retrieval_strategy,
        knowledge_document_count=knowledge_status.get("document_count", 0),
        knowledge_hash=knowledge_index.content_hash if knowledge_index else None,
        results_json=json.dumps(build_result_snapshot(results), ensure_ascii=False),
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return RAGEvaluationResponse(
        history_id=run.id,
        total_count=total_count,
        passed_count=passed_count,
        pass_rate=pass_rate,
        preset_count=len(EVALUATION_CASES),
        custom_count=custom_count,
        retrieval_strategy=retrieval_strategy,
        retrieval_metrics=build_retrieval_metrics(results),
        category_metrics=build_category_metrics(results),
        quality_gate=quality_gate,
        results=results,
    )


def get_document_evidence(document: object) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    return " ".join(
        str(value)
        for value in (
            metadata.get("name", ""),
            getattr(document, "page_content", ""),
        )
        if value
    )


def has_citation_location(document: object) -> bool:
    metadata = getattr(document, "metadata", {}) or {}
    return any(
        metadata.get(key) not in (None, "")
        for key in (
            "source",
            "source_url",
            "location",
            "page_number",
            "chunk_index",
        )
    )


def get_quality_answer_keywords(case: dict) -> list[str]:
    configured_keywords = case.get("answer_keywords", [])
    if isinstance(configured_keywords, list) and configured_keywords:
        return [str(keyword) for keyword in configured_keywords if str(keyword).strip()]
    if case.get("expected_refusal"):
        return []
    expected_name = case.get("expected_name")
    return [str(expected_name)] if expected_name else []


def get_quality_citation_names(case: dict) -> list[str]:
    configured_names = case.get("citation_names", [])
    if isinstance(configured_names, list) and configured_names:
        return [str(name) for name in configured_names if str(name).strip()]
    if case.get("expected_refusal"):
        return []
    expected_name = case.get("expected_name")
    return [str(expected_name)] if expected_name else []


def evaluate_quality_case(
    session: Session,
    vector_store: object,
    case: dict,
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
) -> RAGQualityCaseResult:
    selected_matches = get_current_matches(
        session,
        vector_store,
        case,
        retrieval_strategy,
    )
    evidence = " ".join(
        get_document_evidence(document) for document, _ in selected_matches
    ).casefold()
    answer_keywords = get_quality_answer_keywords(case)
    missing_answer_keywords = [
        keyword for keyword in answer_keywords if keyword.casefold() not in evidence
    ]
    refusal_observed = not selected_matches
    expected_refusal = bool(case.get("expected_refusal", False))
    answer_correct = (
        refusal_observed if expected_refusal else not missing_answer_keywords
    )

    expected_citation_names = get_quality_citation_names(case)
    cited_names: list[str] = []
    for document, _ in selected_matches:
        name = get_match_name(document)
        if name not in cited_names:
            cited_names.append(name)
    missing_citation_names = [
        name for name in expected_citation_names if name not in cited_names
    ]
    citation_has_location = (
        expected_refusal
        and refusal_observed
        or bool(selected_matches)
        and all(has_citation_location(document) for document, _ in selected_matches)
    )
    citation_correct = (
        refusal_observed
        if expected_refusal
        else bool(expected_citation_names)
        and not missing_citation_names
        and citation_has_location
    )
    refusal_correct = expected_refusal == refusal_observed

    if expected_refusal:
        if refusal_observed:
            diagnostic = "正确拒答：没有检索到达到相关度阈值的资料。"
        else:
            diagnostic = "拒答失败：检索到了资料，系统可能会继续生成回答。"
    else:
        diagnostic_parts = []
        diagnostic_parts.append(
            "答案证据完整" if answer_correct else "答案证据不完整"
        )
        diagnostic_parts.append(
            "引用正确且有位置" if citation_correct else "引用资料或位置不完整"
        )
        diagnostic = "；".join(diagnostic_parts) + "。"

    return RAGQualityCaseResult(
        case_id=str(case.get("case_id", "unknown")),
        case_source=str(case.get("case_source", "未标注来源")),
        question=str(case.get("question", "")),
        category=get_case_category(case),
        expected_refusal=expected_refusal,
        answer_correct=answer_correct,
        citation_correct=citation_correct,
        refusal_observed=refusal_observed,
        refusal_correct=refusal_correct,
        answer_keyword_count=len(answer_keywords),
        answer_match_count=len(answer_keywords) - len(missing_answer_keywords),
        missing_answer_keywords=missing_answer_keywords,
        expected_citation_names=expected_citation_names,
        cited_names=cited_names,
        missing_citation_names=missing_citation_names,
        citation_has_location=citation_has_location,
        retrieved_count=len(selected_matches),
        diagnostic=diagnostic,
    )


@router.post("/quality", response_model=RAGQualityResponse)
def run_rag_quality_evaluation(
    retrieval_strategy: str = Query(
        default=DEFAULT_RETRIEVAL_STRATEGY,
        description="质量评测使用的检索策略",
    ),
    session: Session = Depends(get_session),
):
    """Evaluate evidence support, citation quality, and refusal behavior deterministically."""
    retrieval_strategy = validate_retrieval_strategy(retrieval_strategy)
    knowledge_status = get_knowledge_status(session)
    if not knowledge_status["is_current"]:
        raise HTTPException(
            status_code=409,
            detail="知识库已过期，请先重建知识库后再运行质量评测",
        )

    try:
        vector_store = (
            None
            if retrieval_strategy == "none"
            else get_vector_store()
        )
        cases, custom_count = get_quality_evaluation_cases(session)
        results = [
            evaluate_quality_case(
                session,
                vector_store,
                case,
                retrieval_strategy,
            )
            for case in cases
        ]
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="回答质量评测暂时不可用，请检查知识库和 Embedding 配置",
        ) from error

    total_count = len(results)
    answer_correct_count = sum(result.answer_correct for result in results)
    citation_correct_count = sum(result.citation_correct for result in results)
    refusal_correct_count = sum(result.refusal_correct for result in results)
    expected_refusal_count = sum(result.expected_refusal for result in results)
    refusal_observed_count = sum(result.refusal_observed for result in results)

    def rate(count: int) -> float:
        return count / total_count if total_count else 0

    quality_metrics = {
        "total_count": total_count,
        "answer_correct_count": answer_correct_count,
        "answer_accuracy": rate(answer_correct_count),
        "citation_correct_count": citation_correct_count,
        "citation_accuracy": rate(citation_correct_count),
        "refusal_correct_count": refusal_correct_count,
        "refusal_accuracy": rate(refusal_correct_count),
        "refusal_expected_count": expected_refusal_count,
        "refusal_observed_count": refusal_observed_count,
        "refusal_rate": rate(refusal_observed_count),
    }
    knowledge_index = session.get(KnowledgeIndexState, 1)
    quality_run = RAGQualityEvaluationRun(
        **quality_metrics,
        retrieval_strategy=retrieval_strategy,
        knowledge_document_count=knowledge_status.get("document_count", 0),
        knowledge_hash=knowledge_index.content_hash if knowledge_index else None,
        results_json=json.dumps(
            [result.model_dump() for result in results],
            ensure_ascii=False,
        ),
    )
    session.add(quality_run)
    session.commit()
    session.refresh(quality_run)
    return RAGQualityResponse(
        history_id=quality_run.id,
        metrics=quality_metrics,
        preset_count=len(EVALUATION_CASES) + len(QUALITY_ONLY_CASES),
        custom_count=custom_count,
        retrieval_strategy=retrieval_strategy,
        results=results,
    )


@router.post("/compare", response_model=RetrievalComparisonResponse)
def compare_retrieval_strategies(
    baseline_strategy: str = Query(
        default="vector",
        description="对比基线策略",
    ),
    current_strategy: str = Query(
        default=DEFAULT_RETRIEVAL_STRATEGY,
        description="对比当前策略",
    ),
    session: Session = Depends(get_session),
):
    """Compare two explicitly selected retrieval strategies."""
    baseline_strategy = validate_retrieval_strategy(baseline_strategy)
    current_strategy = validate_retrieval_strategy(current_strategy)
    knowledge_status = get_knowledge_status(session)
    if not knowledge_status["is_current"]:
        raise HTTPException(
            status_code=409,
            detail="知识库已过期，请先重建知识库后再运行检索对比",
        )

    try:
        vector_store = (
            None
            if baseline_strategy == "none" and current_strategy == "none"
            else get_vector_store()
        )
        cases, custom_count = get_evaluation_cases(session)
        results = [
            compare_case(
                session,
                vector_store,
                case,
                baseline_strategy,
                current_strategy,
            )
            for case in cases
        ]
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="检索对比暂时不可用，请检查知识库和 Embedding 配置",
        ) from error

    total_count = len(results)
    baseline_passed_count = sum(result.baseline.passed for result in results)
    current_passed_count = sum(result.current.passed for result in results)
    baseline_pass_rate = baseline_passed_count / total_count if total_count else 0
    current_pass_rate = current_passed_count / total_count if total_count else 0
    baseline_metrics = build_retrieval_metrics([result.baseline for result in results])
    current_metrics = build_retrieval_metrics([result.current for result in results])
    return RetrievalComparisonResponse(
        total_count=total_count,
        preset_count=len(EVALUATION_CASES),
        custom_count=custom_count,
        baseline_strategy=baseline_strategy,
        current_strategy=current_strategy,
        baseline=RetrievalStrategySummary(
            passed_count=baseline_passed_count,
            pass_rate=baseline_pass_rate,
            metrics=baseline_metrics,
        ),
        current=RetrievalStrategySummary(
            passed_count=current_passed_count,
            pass_rate=current_pass_rate,
            metrics=current_metrics,
        ),
        pass_rate_delta=current_pass_rate - baseline_pass_rate,
        top1_accuracy_delta=(
            current_metrics.top1_accuracy - baseline_metrics.top1_accuracy
        ),
        recall_at_3_delta=current_metrics.recall_at_3 - baseline_metrics.recall_at_3,
        precision_at_3_delta=(
            current_metrics.precision_at_3 - baseline_metrics.precision_at_3
        ),
        improved_count=sum(result.change == "improved" for result in results),
        regressed_count=sum(result.change == "regressed" for result in results),
        unchanged_count=sum(result.change == "unchanged" for result in results),
        results=results,
    )


@router.post("/diagnose", response_model=RetrievalDiagnosticResponse)
def diagnose_retrieval(
    retrieval_strategy: str = Query(
        default=DEFAULT_RETRIEVAL_STRATEGY,
        description="诊断使用的检索策略",
    ),
    session: Session = Depends(get_session),
):
    """Explain why each evaluation target did or did not reach the live top three."""
    retrieval_strategy = validate_retrieval_strategy(retrieval_strategy)
    knowledge_status = get_knowledge_status(session)
    if not knowledge_status["is_current"]:
        raise HTTPException(
            status_code=409,
            detail="知识库已过期，请先重建知识库后再运行检索诊断",
        )

    try:
        vector_store = (
            None if retrieval_strategy == "none" else get_vector_store()
        )
        cases, _ = get_evaluation_cases(session)
        results = [
            build_diagnostic_case(session, vector_store, case, retrieval_strategy)
            for case in cases
        ]
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="检索诊断暂时不可用，请检查知识库和 Embedding 配置",
        ) from error

    return RetrievalDiagnosticResponse(
        total_count=len(results),
        healthy_count=sum(result.diagnostic_level == "healthy" for result in results),
        attention_count=sum(
            result.diagnostic_level == "attention" for result in results
        ),
        failed_count=sum(result.diagnostic_level == "failed" for result in results),
        results=results,
    )


def get_ragas_active_run(session: Session) -> RAGASAutoEvaluationRun | None:
    return session.exec(
        select(RAGASAutoEvaluationRun)
        .where(RAGASAutoEvaluationRun.status.in_(ACTIVE_RAGAS_STATUSES))
        .order_by(RAGASAutoEvaluationRun.created_at.desc())
    ).first()


def recover_stale_ragas_runs(session: Session) -> None:
    """Mark tasks interrupted by an API restart or an excessive runtime."""
    now = datetime.now(UTC)
    changed = False
    timeout_minutes = max(1, RAGAS_JOB_TIMEOUT_SECONDS // 60)
    active_runs = session.exec(
        select(RAGASAutoEvaluationRun).where(
            RAGASAutoEvaluationRun.status.in_(ACTIVE_RAGAS_STATUSES)
        )
    ).all()
    for run in active_runs:
        reference_time = run.started_at or run.created_at
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=UTC)
        if (now - reference_time).total_seconds() <= RAGAS_JOB_TIMEOUT_SECONDS:
            continue
        run.status = "failed"
        run.error_message = (
            f"RAGAS 评测超过 {timeout_minutes} 分钟未完成，可能因服务重启中断，请重试"
        )
        run.completed_at = now
        session.add(run)
        changed = True
    if changed:
        session.commit()


def normalize_ragas_score(value: object) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return round(score, 4) if math.isfinite(score) else None


def get_ragas_results(run: RAGASAutoEvaluationRun) -> list[RAGASQuestionResult]:
    try:
        raw_results = json.loads(run.results_json)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(raw_results, list):
        return []
    results: list[RAGASQuestionResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        try:
            results.append(RAGASQuestionResult.model_validate(item))
        except ValueError:
            continue
    return results


def get_ragas_metrics(run: RAGASAutoEvaluationRun) -> RAGASMetricScores | None:
    values = {
        "faithfulness": run.faithfulness,
        "answer_relevancy": run.answer_relevancy,
        "context_precision": run.context_precision,
        "context_recall": run.context_recall,
    }
    if not any(value is not None for value in values.values()):
        return None
    return RAGASMetricScores(**values)


def serialize_ragas_run(
    run: RAGASAutoEvaluationRun,
) -> RAGASAutoEvaluationRunRead:
    return RAGASAutoEvaluationRunRead(
        id=run.id,
        status=run.status,
        sample_size=run.sample_size,
        retrieval_strategy=run.retrieval_strategy,
        total_count=run.total_count,
        completed_count=run.completed_count,
        metrics=get_ragas_metrics(run),
        knowledge_document_count=run.knowledge_document_count,
        knowledge_hash=run.knowledge_hash,
        error_message=run.error_message,
        created_at=run.created_at,
        started_at=run.started_at,
        completed_at=run.completed_at,
        results=get_ragas_results(run),
    )


def get_case_reference_answer(session: Session, case: dict) -> tuple[str, bool]:
    """Use the expected source record as RAGAS's answer reference."""
    names = [str(case.get("expected_name", "")), *get_alternative_names(case)]
    names = [name for name in names if name]
    expected_type = case.get("expected_type")
    if expected_type == "condition":
        records = session.exec(
            select(Condition).where(Condition.name.in_(names))
        ).all()
        by_name = {record.name: record for record in records}
        for name in names:
            record = by_name.get(name)
            if record is not None:
                return (
                    f"病症名称：{record.name}\n"
                    f"常见症状：{record.symptoms}\n"
                    f"处理建议：{record.treatment}"[:RAGAS_REFERENCE_MAX_CHARS],
                    True,
                )
    elif expected_type == "drug":
        records = session.exec(select(Drug).where(Drug.name.in_(names))).all()
        by_name = {record.name: record for record in records}
        for name in names:
            record = by_name.get(name)
            if record is not None:
                return (
                    f"药物名称：{record.name}\n"
                    f"药物作用：{record.effects}\n"
                    f"使用说明：{record.instructions}"[:RAGAS_REFERENCE_MAX_CHARS],
                    True,
                )
    elif expected_type == "document":
        records = session.exec(
            select(KnowledgeDocument).where(KnowledgeDocument.title.in_(names))
        ).all()
        by_title = {record.title: record for record in records}
        for name in names:
            record = by_title.get(name)
            if record is not None:
                return (
                    f"资料标题：{record.title}\n资料内容：{record.content}"[
                        :RAGAS_REFERENCE_MAX_CHARS
                    ],
                    True,
                )

    # Preserve the sampled case when its target was removed. The result marks
    # this degraded reference explicitly instead of silently excluding a row.
    return str(case.get("expected_name", "未提供参考资料")), False


def calculate_ragas_averages(
    results: list[dict],
) -> dict[str, float | None]:
    metric_names = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )
    averages: dict[str, float | None] = {}
    for name in metric_names:
        values = [
            score
            for result in results
            if (score := normalize_ragas_score(result.get(name))) is not None
        ]
        averages[name] = round(sum(values) / len(values), 4) if values else None
    return averages


def execute_ragas_evaluation(
    session: Session,
    run: RAGASAutoEvaluationRun,
) -> list[dict]:
    """Generate live answers, then score them through the four RAGAS metrics."""
    enable_ragas_langchain_compatibility()
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    knowledge_status = get_knowledge_status(session)
    if not knowledge_status["is_current"]:
        raise RuntimeError("知识库已过期，请先重建知识库后再运行 RAGAS 评测")

    cases, _ = get_evaluation_cases(session)
    if not cases:
        raise RuntimeError("没有可运行的评测题，请先添加自定义评测题")
    sampled_cases = SystemRandom().sample(
        cases,
        k=min(run.sample_size, len(cases)),
    )
    knowledge_index = session.get(KnowledgeIndexState, 1)
    run.total_count = len(sampled_cases)
    run.knowledge_document_count = knowledge_status.get("document_count", 0)
    run.knowledge_hash = knowledge_index.content_hash if knowledge_index else None
    session.add(run)
    session.commit()

    question_results: list[dict] = []
    samples: list[SingleTurnSample] = []
    for case in sampled_cases:
        pipeline = run_rag_answer_pipeline(
            session,
            case["question"],
            retrieval_strategy=run.retrieval_strategy,
        )
        reference, reference_available = get_case_reference_answer(session, case)
        references = pipeline.get("references", [])
        context_titles = [
            str(item.get("name", "未命名资料"))
            for item in references
            if isinstance(item, dict)
        ]
        question_results.append(
            {
                "case_id": str(case.get("case_id", "unknown")),
                "case_source": str(case.get("case_source", "未标注来源")),
                "question": str(case["question"]),
                "expected_name": str(case.get("expected_name", "")),
                "expected_type": str(case.get("expected_type", "")),
                "category": get_case_category(case),
                "answer": str(pipeline["answer"]),
                "processing_path": str(pipeline["processing_path"]),
                "retrieved_count": int(pipeline["retrieved_count"]),
                "context_titles": context_titles,
                "retrieved_contexts": list(pipeline["contexts"]),
                "reference_available": reference_available,
            }
        )
        samples.append(
            SingleTurnSample(
                user_input=str(case["question"]),
                response=str(pipeline["answer"]),
                retrieved_contexts=list(pipeline["contexts"]),
                reference=reference,
            )
        )

    run.completed_count = len(question_results)
    session.add(run)
    session.commit()

    ragas_result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(),
            build_medical_answer_relevancy_metric(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=get_chat_model(),
        embeddings=get_embeddings(),
        raise_exceptions=False,
        show_progress=False,
        experiment_name=f"medical-ragas-run-{run.id}",
    )
    for result, scores in zip(question_results, ragas_result.scores, strict=True):
        for name in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ):
            result[name] = normalize_ragas_score(scores.get(name))
    return question_results


def run_ragas_evaluation_job(run_id: int) -> None:
    """Execute one persisted RAGAS task outside the request lifecycle."""
    with ragas_execution_lock:
        with Session(engine) as session:
            run = session.get(RAGASAutoEvaluationRun, run_id)
            if run is None or run.status not in ACTIVE_RAGAS_STATUSES:
                return
            run.status = "running"
            run.started_at = datetime.now(UTC)
            session.add(run)
            session.commit()
            try:
                results = execute_ragas_evaluation(session, run)
                averages = calculate_ragas_averages(results)
                session.refresh(run)
                run.status = "completed"
                run.completed_count = len(results)
                run.faithfulness = averages["faithfulness"]
                run.answer_relevancy = averages["answer_relevancy"]
                run.context_precision = averages["context_precision"]
                run.context_recall = averages["context_recall"]
                run.results_json = json.dumps(
                    results,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                run.completed_at = datetime.now(UTC)
                session.add(run)
                session.commit()
            except Exception as error:
                session.rollback()
                logger.exception("RAGAS evaluation failed: run_id=%s", run_id)
                with Session(engine) as failed_session:
                    failed_run = failed_session.get(RAGASAutoEvaluationRun, run_id)
                    if failed_run is not None:
                        failed_run.status = "failed"
                        failed_run.error_message = str(error)[:2000]
                        failed_run.completed_at = datetime.now(UTC)
                        failed_session.add(failed_run)
                        failed_session.commit()


@router.post(
    "/ragas/tasks",
    response_model=RAGASAutoEvaluationRunRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_ragas_evaluation(
    background_tasks: BackgroundTasks,
    sample_size: int = Query(
        default=RAGAS_DEFAULT_SAMPLE_SIZE,
        ge=1,
        le=RAGAS_MAX_SAMPLE_SIZE,
    ),
    retrieval_strategy: str = Query(
        default=DEFAULT_RETRIEVAL_STRATEGY,
        description="RAGAS 生成回答使用的检索策略",
    ),
    session: Session = Depends(get_session),
):
    """Start a background RAGAS assessment; ten sampled cases are the default."""
    retrieval_strategy = validate_retrieval_strategy(retrieval_strategy)
    with ragas_start_lock:
        recover_stale_ragas_runs(session)
        active_run = get_ragas_active_run(session)
        if active_run is not None:
            raise HTTPException(
                status_code=409,
                detail=f"RAGAS 评测任务 #{active_run.id} 正在执行，请先查看它的状态",
            )
        knowledge_status = get_knowledge_status(session)
        if not knowledge_status["is_current"]:
            raise HTTPException(
                status_code=409,
                detail="知识库已过期，请先重建知识库后再运行 RAGAS 评测",
            )
        cases, _ = get_evaluation_cases(session)
        if not cases:
            raise HTTPException(status_code=422, detail="没有可运行的评测题")
        run = RAGASAutoEvaluationRun(
            sample_size=sample_size,
            retrieval_strategy=retrieval_strategy,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        background_tasks.add_task(run_ragas_evaluation_job, run.id)
        return serialize_ragas_run(run)


@router.get(
    "/ragas/tasks/active",
    response_model=RAGASAutoEvaluationRunRead | None,
)
def get_active_ragas_evaluation(
    session: Session = Depends(get_session),
):
    recover_stale_ragas_runs(session)
    run = get_ragas_active_run(session)
    return serialize_ragas_run(run) if run is not None else None


@router.get(
    "/ragas/tasks",
    response_model=RAGASAutoEvaluationRunListResponse,
)
def list_ragas_evaluations(
    limit: int = Query(default=10, ge=1, le=100),
    session: Session = Depends(get_session),
):
    recover_stale_ragas_runs(session)
    runs = session.exec(
        select(RAGASAutoEvaluationRun)
        .order_by(
            RAGASAutoEvaluationRun.created_at.desc(),
            RAGASAutoEvaluationRun.id.desc(),
        )
        .limit(limit)
    ).all()
    return RAGASAutoEvaluationRunListResponse(
        total_count=len(session.exec(select(RAGASAutoEvaluationRun)).all()),
        runs=[serialize_ragas_run(run) for run in runs],
    )


@router.get(
    "/ragas/tasks/{run_id}",
    response_model=RAGASAutoEvaluationRunRead,
)
def get_ragas_evaluation(
    run_id: int,
    session: Session = Depends(get_session),
):
    recover_stale_ragas_runs(session)
    run = session.get(RAGASAutoEvaluationRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="RAGAS 评测任务不存在")
    return serialize_ragas_run(run)
