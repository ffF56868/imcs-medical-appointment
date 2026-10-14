"""AI 评测服务：检索质量、回答质量、回归门禁（阶段 6B）。

三层评测中的前两层（检索 / 回答质量）是**纯确定性判定、零 LLM 调用**，
可在一次请求内同步跑完（题库几十题、每题一次 RRF 检索）；
第三层 RAGAS 要生成回答并跑 ragas.evaluate，是重活，走 Celery
（见 tasks/ai_tasks.run_ragas_evaluation）。

算法移植自旧系统 ``app/routers/evaluation.py``，但适配 IMCS 的
异步检索内核（``RAGRetriever.retrieve`` 返回 ``RetrievalResponse``），
并修正三处旧实现的问题：

1. **资料级去重**：一篇文档会被切成多块，一次检索可能命中同一 ``title``
   的多个 chunk。旧系统靠 ``select_distinct_relevant_matches`` 按 name 去重，
   IMCS 侧在 ``_distinct_by_title`` 里做同样的事——否则 Precision@3 会把
   「同一份资料排了三次」算成三条有效结果，指标虚高。
2. **命中口径对齐 doc_type**：``expected_type`` 直接与 ``RetrievalResult.doc_type``
   比对（取值域 condition/guideline），别名集合取并集，避免同资料换叫法判负。
3. **零结果分母**：Recall/Precision 的分母为 0 时返回 None 而不是 0.0——
   「一题都没检到」与「检到了但都不对」是两种完全不同的失败，
   用 0.0 冒充前者会掩盖知识库根本没内容这种更严重的情况。

拒答题（expected_refusal）按旧口径「检索为空即正确拒答」。注意 IMCS 全文检索
对任意 query 都可能返回低相关命中，因此真实环境下拒答判定存在偏差，
本批次不承诺拒答精确率（旧 100 题也没有拒答题）；判定逻辑由打桩单测覆盖。
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    DuplicateTaskError,
    ResourceNotFoundError,
    TaskQueueUnavailableError,
)
from app.models.ai import AITask, AITaskStatus
from app.models.evaluation import EvalCase, EvalKind, EvalRun, EvalRunStatus
from app.models.knowledge import KnowledgeDocument

logger = logging.getLogger(__name__)

# AITask.task_type 取值。models/ai.py 注释登记的正是 ragas_evaluate，
# 派发侧与任务侧必须共用同一字符串，写错一个字符监控就统计不到。
TASK_TYPE_RAGAS = "ragas_evaluate"
RAGAS_TASK_NAME = "app.tasks.ai_tasks.run_ragas_evaluation"
# 派发防重：pending/running/retrying 都算占位中（与 knowledge_service 同口径）。
# 只拦 running 的话，队列积压期间的重复点击会排进第二个评测任务——
# 评测是烧 LLM 配额的动作，双跑等于双倍成本。
ACTIVE_AI_STATUSES = frozenset(
    {AITaskStatus.PENDING.value, AITaskStatus.RUNNING.value, AITaskStatus.RETRYING.value}
)

# Precision/Recall 的截断窗口（旧口径 @3；同时用于 Top1 判定）
EVAL_TOP_K = 3
# 与旧系统 get_acceptable_names 同源：expected_name ∪ alternative_names
METRIC_KEYS = ("top1_accuracy", "recall_at_3", "precision_at_3")
QUALITY_KEYS = ("answer_accuracy", "citation_accuracy", "refusal_accuracy")
# RAGAS 四指标列名。取值与 ragas_runner.RAGAS_METRIC_NAMES 一致——这里不
# import 那个模块，是为了让趋势查询不在 Web 进程里牵进评测运行器的依赖；
# 名字对应 EvalRun 上固定的四个列，漂移会被单测的 getattr 断言抓到。
RAGAS_METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# 回归门禁阈值：与上一次同类运行相比，指标下降超过这些值就升级关注级别。
GATE_ATTENTION_DROP = 0.10   # 绝对下降 ≥ 10 个百分点 → attention（红）
GATE_WARNING_DROP = 0.05     # ≥ 5 个百分点 → warning（黄）


# ==================== 纯判定函数（无 IO，便于单测） ====================


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """兼容 ORM 对象、dataclass 与 dict 三种取法（打桩对象形态不一）。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def acceptable_names(case: Any) -> set[str]:
    """一道题可接受的资料名集合：期望名 + 所有别名。"""
    expected = str(_get(case, "expected_name", "") or "")
    names = {expected} if expected else set()
    for alt in _get(case, "alternative_names", []) or []:
        text = str(alt).strip()
        if text:
            names.add(text)
    return names


def distinct_by_title(results: list[Any]) -> list[Any]:
    """按 title 去重、保留首次出现的顺序（= 该资料的最高排名位置）。"""
    seen: set[str] = set()
    out: list[Any] = []
    for item in results:
        title = str(_get(item, "title", "") or "")
        if title and title in seen:
            continue
        if title:
            seen.add(title)
        out.append(item)
    return out


def eval_case_to_dict(case: EvalCase) -> dict[str, Any]:
    """ORM 题库行 → 判定函数消费的普通字典。"""
    return {
        "id": str(case.id),
        "question": case.question,
        "expected_name": case.expected_name,
        "expected_type": case.expected_type,
        "category": case.category,
        "alternative_names": list(case.alternative_names or []),
        "answer_keywords": list(case.answer_keywords or []),
        "citation_names": list(case.citation_names or []),
        "expected_refusal": bool(case.expected_refusal),
    }


def build_retrieval_result(
    results: list[Any], case: dict[str, Any]
) -> dict[str, Any]:
    """单题检索判定：期望排名、是否命中、窗口内相关/检回数。

    口径与旧 ``build_strategy_result`` 一致，但去重与类型匹配显式化：
    - 期望命中 = title ∈ acceptable_names 且 doc_type == expected_type
    - expected_rank 是去重后序列里第一次命中的 1-based 位置
    - passed = 命中且落在前 EVAL_TOP_K 窗口内（这就是 Recall@3 的「3」）
    """
    ranked = distinct_by_title(results)
    names = acceptable_names(case)
    expected_type = case["expected_type"]
    expected_rank: int | None = None
    matched_name: str | None = None
    relevant_count = 0
    window = ranked[:EVAL_TOP_K]

    for rank, item in enumerate(ranked, start=1):
        title = str(_get(item, "title", "") or "")
        doc_type = str(_get(item, "doc_type", "") or "")
        is_expected = title in names and doc_type == expected_type
        if not is_expected:
            continue
        relevant_count += 1
        if expected_rank is None:
            expected_rank = rank
            matched_name = title

    passed = expected_rank is not None and expected_rank <= EVAL_TOP_K
    top = ranked[0] if ranked else None
    return {
        "question": case["question"],
        "category": case["category"],
        "expected_name": case["expected_name"],
        "expected_type": expected_type,
        "expected_refusal": case["expected_refusal"],
        "passed": passed,
        "expected_rank": expected_rank,
        "matched_name": matched_name,
        "top_name": str(_get(top, "title", "")) if top is not None else None,
        "top_score": (
            round(float(_get(top, "rrf_score", 0.0) or 0.0), 6) if top is not None else None
        ),
        "retrieved_count": len(window),
        "relevant_count": min(relevant_count, EVAL_TOP_K),
    }


def build_quality_result(results: list[Any], case: dict[str, Any]) -> dict[str, Any]:
    """单题回答质量判定（证据覆盖 / 引用 / 拒答），零 LLM。

    与旧 ``evaluate_quality_case`` 同源：答案正确性看「关键词是否都在
    检索到的证据正文里」，是证据覆盖而非对生成文本打分；引用正确性看
    「期望引用的资料是否都被检到且带出处位置」；拒答题看「检索为空」。
    """
    ranked = distinct_by_title(results)
    window = ranked[:EVAL_TOP_K]
    evidence_parts: list[str] = []
    for item in window:
        content = str(_get(item, "content", "") or "")
        title = str(_get(item, "title", "") or "")
        evidence_parts.append(f"{title} {content}")
    evidence = " ".join(evidence_parts).casefold()

    answer_keywords = _effective(case, "answer_keywords")
    missing_keywords = [
        kw for kw in answer_keywords if str(kw).casefold() not in evidence
    ]
    refusal_observed = len(window) == 0
    expected_refusal = case["expected_refusal"]
    answer_correct = (
        refusal_observed if expected_refusal else not missing_keywords
    )

    citation_names = _effective(case, "citation_names")
    cited_titles = []
    for item in window:
        title = str(_get(item, "title", "") or "")
        if title and title not in cited_titles:
            cited_titles.append(title)
    missing_citations = [n for n in citation_names if n not in cited_titles]
    citation_has_location = all(
        _get(item, "source") not in (None, "") for item in window
    ) if window else expected_refusal
    citation_correct = (
        refusal_observed
        if expected_refusal
        else bool(citation_names) and not missing_citations and citation_has_location
    )
    refusal_correct = expected_refusal == refusal_observed

    return {
        "question": case["question"],
        "category": case["category"],
        "expected_refusal": expected_refusal,
        "answer_correct": answer_correct,
        "citation_correct": citation_correct,
        "refusal_observed": refusal_observed,
        "refusal_correct": refusal_correct,
        "answer_keyword_count": len(answer_keywords),
        "missing_answer_keywords": missing_keywords,
        "expected_citation_names": citation_names,
        "cited_titles": cited_titles,
        "missing_citation_names": missing_citations,
        "retrieved_count": len(window),
    }


def _effective(case: dict[str, Any], field_name: str) -> list[str]:
    """质量层关键词/引用名缺省回退到 expected_name（旧口径）。"""
    values = case.get(field_name) or []
    if isinstance(values, list) and values:
        return [str(v) for v in values if str(v).strip()]
    if case.get("expected_refusal"):
        return []
    expected = case.get("expected_name")
    return [str(expected)] if expected else []


def aggregate_retrieval(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总检索三指标（整体 + 按 category 下钻）。"""
    total = len(results)
    top1 = sum(1 for r in results if r["expected_rank"] == 1)
    recalled = sum(1 for r in results if r["passed"])
    relevant = sum(r["relevant_count"] for r in results)
    retrieved = sum(r["retrieved_count"] for r in results)
    overall = {
        "total_count": total,
        "top1_correct_count": top1,
        "top1_accuracy": round(top1 / total, 4) if total else None,
        "recalled_count": recalled,
        "recall_at_3": round(recalled / total, 4) if total else None,
        "relevant_result_count": relevant,
        "retrieved_result_count": retrieved,
        "precision_at_3": round(relevant / retrieved, 4) if retrieved else None,
    }
    overall["by_category"] = _aggregate_by_category(results, _retrieval_of)
    return overall


def aggregate_quality(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总回答质量三指标（整体 + 按 category 下钻）。"""
    total = len(results)
    answer = sum(1 for r in results if r["answer_correct"])
    citation = sum(1 for r in results if r["citation_correct"])
    refusal = sum(1 for r in results if r["refusal_correct"])
    return {
        "total_count": total,
        "answer_correct_count": answer,
        "answer_accuracy": round(answer / total, 4) if total else None,
        "citation_correct_count": citation,
        "citation_accuracy": round(citation / total, 4) if total else None,
        "refusal_correct_count": refusal,
        "refusal_accuracy": round(refusal / total, 4) if total else None,
        "by_category": _aggregate_by_category(results, _quality_of),
    }


def _retrieval_of(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "top1_accuracy": r["expected_rank"] == 1,
        "recall_at_3": r["passed"],
        # 单题不带 precision 分子分母之外的信息，这里用 passed 近似该类召回，
        # 精度整体值由 aggregate 顶层给；分类层只给 top1/recall 两项。
    }


def _quality_of(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer_accuracy": r["answer_correct"],
        "citation_accuracy": r["citation_correct"],
        "refusal_accuracy": r["refusal_correct"],
    }


def _aggregate_by_category(
    results: list[dict[str, Any]], projector
) -> dict[str, dict[str, float]]:
    """按 category 分组算比率（布尔命中率的均值）。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        grouped.setdefault(str(r.get("category") or "未分类"), []).append(r)
    out: dict[str, dict[str, float]] = {}
    for category, items in grouped.items():
        flags = [projector(item) for item in items]
        metric: dict[str, float] = {}
        for key in flags[0].keys():
            hits = sum(1 for f in flags if f.get(key))
            metric[key] = round(hits / len(items), 4)
        out[category] = metric
    return out


def apply_gate(
    current: dict[str, Any], previous: dict[str, Any] | None
) -> dict[str, Any]:
    """回归门禁：与上一次运行比整体指标，输出每指标的变化档位。

    档位：improved（升）/ unchanged（±5pp 内）/ warning（降 5-10pp）/
    attention（降 ≥10pp）。previous 缺失或该指标为 None → baseline（无对比基准）。
    知识库变了（hash 不同）时不给 warning/attention，只报 changed——
    指标下滑可能只是「删了一批资料」，不是检索退化。
    """
    gate: dict[str, Any] = {}
    if previous is None:
        return {"status": "baseline", "metrics": {}}
    if (
        current.get("knowledge_hash")
        and previous.get("knowledge_hash")
        and current["knowledge_hash"] != previous["knowledge_hash"]
    ):
        return {"status": "changed", "reason": "知识库指纹变化，指标不可直接比较"}
    for key in (*METRIC_KEYS, *QUALITY_KEYS):
        cur, prev = current.get(key), previous.get(key)
        if cur is None or prev is None:
            continue
        delta = round(cur - prev, 4)
        if delta >= GATE_WARNING_DROP:
            level = "improved"
        elif delta <= -GATE_ATTENTION_DROP:
            level = "attention"
        elif delta <= -GATE_WARNING_DROP:
            level = "warning"
        else:
            level = "unchanged"
        gate[key] = {"current": cur, "previous": prev, "delta": delta, "level": level}
    return {"status": "compared", "metrics": gate}


# ==================== 运行入口（读写数据库） ====================


async def list_cases(
    db: AsyncSession, *, category: str | None = None, kind_hint: str | None = None
) -> list[EvalCase]:
    """取题库。kind_hint=quality 时保留全部（含拒答题）；retrieval 也全取，
    拒答题对检索层同样有意义（应检不到）。"""
    stmt = select(EvalCase).order_by(EvalCase.category, EvalCase.expected_name)
    if category:
        stmt = stmt.where(EvalCase.category == category)
    return list((await db.execute(stmt)).scalars().all())


async def _knowledge_fingerprint(db: AsyncSession) -> tuple[str | None, int]:
    """知识库文档数与聚合内容哈希（按 id 序拼接 content_hash）。"""
    rows = (
        await db.execute(
            select(KnowledgeDocument.content_hash).order_by(KnowledgeDocument.id)
        )
    ).scalars().all()
    count = len(rows)
    if not count:
        return None, 0
    import hashlib

    digest = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()
    return digest, count


async def run_retrieval_evaluation(
    db: AsyncSession,
    *,
    retriever: Any,
    triggered_by: uuid.UUID | None = None,
    category: str | None = None,
) -> EvalRun:
    """跑一次检索质量评测，落一条 kind=retrieval 的 EvalRun 并返回。"""
    # 排除拒答题：它们的正确行为是「检索为空」，在检索层永远算 miss，
    # 混进来会把 recall_at_3 拖成假性偏低——旧 /run 同口径（拒答题归质量层评）。
    cases = [
        c for c in eval_case_to_dict_list(await list_cases(db, category=category))
        if not c["expected_refusal"]
    ]
    run = await _begin_run(db, EvalKind.RETRIEVAL, len(cases), triggered_by)
    try:
        results: list[dict[str, Any]] = []
        for case in cases:
            resp = await retriever.retrieve(
                db, case["question"], trace_id=run.id.hex
            )
            results.append(build_retrieval_result(list(_get(resp, "results", []) or []), case))
        metrics = aggregate_retrieval(results)
        await _finish_run(db, run, metrics, results)
    except Exception as exc:  # noqa: BLE001
        await _fail_run(db, run, exc)
        raise
    return run


async def run_quality_evaluation(
    db: AsyncSession,
    *,
    retriever: Any,
    triggered_by: uuid.UUID | None = None,
    category: str | None = None,
) -> EvalRun:
    """跑一次回答质量评测（证据覆盖/引用/拒答），落 kind=quality 的 EvalRun。"""
    cases = eval_case_to_dict_list(await list_cases(db, category=category))
    run = await _begin_run(db, EvalKind.QUALITY, len(cases), triggered_by)
    try:
        results: list[dict[str, Any]] = []
        for case in cases:
            resp = await retriever.retrieve(db, case["question"], trace_id=run.id.hex)
            results.append(build_quality_result(list(_get(resp, "results", []) or []), case))
        metrics = aggregate_quality(results)
        await _finish_run(db, run, metrics, results)
    except Exception as exc:  # noqa: BLE001
        await _fail_run(db, run, exc)
        raise
    return run


def eval_case_to_dict_list(rows: list[EvalCase]) -> list[dict[str, Any]]:
    return [eval_case_to_dict(row) for row in rows]


async def _begin_run(
    db: AsyncSession, kind: EvalKind, total: int, triggered_by: uuid.UUID | None
) -> EvalRun:
    digest, count = await _knowledge_fingerprint(db)
    run = EvalRun(
        kind=kind.value,
        status=EvalRunStatus.RUNNING.value,
        total_count=total,
        completed_count=0,
        knowledge_hash=digest,
        knowledge_document_count=count,
        triggered_by=triggered_by,
        started_at=datetime.now(UTC),
        retrieval_config={
            "rag_result_count": settings.rag_result_count,
            "rag_rrf_k": settings.rag_rrf_k,
            "eval_top_k": EVAL_TOP_K,
        },
    )
    db.add(run)
    await db.flush()
    return run


async def _finish_run(
    db: AsyncSession, run: EvalRun, metrics: dict[str, Any], results: list[dict[str, Any]]
) -> None:
    """落指标、写逐题明细、跑回归门禁。"""
    prev = await _previous_completed(db, run.kind)
    current_flat = {
        **{k: metrics.get(k) for k in METRIC_KEYS},
        **{k: metrics.get(k) for k in QUALITY_KEYS},
        "knowledge_hash": run.knowledge_hash,
    }
    run.completed_count = len(results)
    run.top1_accuracy = metrics.get("top1_accuracy")
    run.recall_at_3 = metrics.get("recall_at_3")
    run.precision_at_3 = metrics.get("precision_at_3")
    run.answer_accuracy = metrics.get("answer_accuracy")
    run.citation_accuracy = metrics.get("citation_accuracy")
    run.refusal_accuracy = metrics.get("refusal_accuracy")
    run.metrics = {**metrics, "gate": apply_gate(current_flat, prev)}
    run.results = results
    run.status = EvalRunStatus.COMPLETED.value
    run.completed_at = datetime.now(UTC)
    await db.commit()


async def _fail_run(db: AsyncSession, run: EvalRun, exc: Exception) -> None:
    await db.rollback()
    run.status = EvalRunStatus.FAILED.value
    run.error_message = f"{type(exc).__name__}: {exc}"[:2000]
    run.completed_at = datetime.now(UTC)
    db.add(run)
    await db.commit()


async def _previous_completed(
    db: AsyncSession, kind: str
) -> dict[str, Any] | None:
    """取同 kind 的上一条 completed 运行的扁平指标，用于回归对比。"""
    stmt = (
        select(EvalRun)
        .where(EvalRun.kind == kind, EvalRun.status == EvalRunStatus.COMPLETED.value)
        .order_by(EvalRun.created_at.desc())
        .limit(1)
    )
    prev = (await db.execute(stmt)).scalars().first()
    if prev is None:
        return None
    return {
        **{k: getattr(prev, k) for k in METRIC_KEYS},
        **{k: getattr(prev, k) for k in QUALITY_KEYS},
        "knowledge_hash": prev.knowledge_hash,
    }


# ==================== 运行查询 ====================


async def list_runs(
    db: AsyncSession,
    *,
    kind: str | None = None,
    status: str | None = None,
    page: int = 1,
    page_size: int = 20,
    include_results: bool = False,
) -> tuple[list[EvalRun], int]:
    """评测运行列表（默认剔除逐题明细，详情接口才给全量）。"""
    conditions = []
    if kind:
        conditions.append(EvalRun.kind == kind)
    if status:
        conditions.append(EvalRun.status == status)

    count_stmt = select(EvalRun.id)
    if conditions:
        count_stmt = count_stmt.where(*conditions)
    total = len((await db.execute(count_stmt)).all())

    size = max(1, min(int(page_size or 20), 100))
    current = max(1, int(page or 1))
    stmt = (
        select(EvalRun)
        .where(*conditions)
        .order_by(EvalRun.created_at.desc(), EvalRun.id.desc())
        .offset((current - 1) * size)
        .limit(size)
    )
    runs = list((await db.execute(stmt)).scalars().all())
    if not include_results:
        for run in runs:
            # **先 expunge 再清空**：直接改受管对象的 results 会被标记为脏——
            # 调用方在同一会话里后续 commit 时，会把库里的逐题明细真的抹成
            # NULL（数据丢失级）；同会话再 get_run 也会经身份映射读到被清空的
            # 对象。脱离会话后改内存属性才真正做到「瘦身不入库」。
            db.expunge(run)
            run.results = None
    return runs, total


async def get_run(db: AsyncSession, run_id: uuid.UUID) -> EvalRun:
    run = await db.get(EvalRun, run_id)
    if run is None:
        raise ResourceNotFoundError("评测运行不存在")
    return run


async def run_trend(db: AsyncSession, kind: str, limit: int = 20) -> dict[str, Any]:
    """同 kind 的历史运行序列 + 基线（最早一次 completed）。

    门禁要看的是「这条线往哪走」，单点比较只能发现突变、发现不了缓慢劣化；
    趋势接口把每次运行的整体指标按时间给出，配合 metrics.gate 就能定位
    是从哪一次运行、哪一类题目开始掉的。
    """
    size = max(2, min(int(limit or 20), 100))
    stmt = (
        select(EvalRun)
        .where(EvalRun.kind == kind, EvalRun.status == EvalRunStatus.COMPLETED.value)
        .order_by(EvalRun.created_at.asc())
    )
    runs = list((await db.execute(stmt)).scalars().all())
    keys = (*METRIC_KEYS, *QUALITY_KEYS, *RAGAS_METRIC_NAMES)
    points = []
    for run in runs[-size:]:
        point = {
            "run_id": str(run.id),
            "created_at": run.created_at,
            "total_count": run.total_count,
            "knowledge_hash": run.knowledge_hash,
        }
        for key in keys:
            value = getattr(run, key, None)
            if value is not None:
                point[key] = float(value)
        points.append(point)
    baseline = points[0] if points else None
    return {"kind": kind, "points": points, "baseline": baseline}


# ==================== RAGAS 派发（异步） ====================


async def dispatch_ragas_evaluation(
    db: AsyncSession,
    *,
    sample_size: int = 10,
    triggered_by: uuid.UUID | None = None,
) -> tuple[EvalRun, AITask]:
    """建 EvalRun(pending) + AITask(pending) 并派发 RAGAS 评测。

    **先落库再派发**（与解读/向量化同口径）：派发成功但库里没记录，
    前端查不到进度、僵尸清理也无从下手；反过来只多一条 pending 记录，
    可由清理任务收口，损失更小。

    重复派发防护查 AITask 占位状态：评测是全局动作（不像解读是 per-report），
    同型任务有活跃记录就 409。注意 EvalRun 侧不查重——一个 pending 的
    AITask 对应的 EvalRun 必然也在跑，查一边就够，查两边会出现
    「run 已建、task 没建」的半截状态。

    EvalRun.total_count 先存 sample_size（期望抽样数），worker 认领后
    改写为实际抽到的题数——题库可能比请求的样本量还小。
    """
    from app.services.ragas_runner import RAGAS_MAX_SAMPLE_SIZE, create_ragas_run

    size = max(1, min(int(sample_size), RAGAS_MAX_SAMPLE_SIZE))

    active = (
        await db.execute(
            select(AITask.id).where(
                AITask.task_type == TASK_TYPE_RAGAS,
                AITask.status.in_(ACTIVE_AI_STATUSES),
            ).limit(1)
        )
    ).scalars().first()
    if active is not None:
        raise DuplicateTaskError(
            "已有 RAGAS 评测在执行，请等待其完成后再触发",
            task_type=TASK_TYPE_RAGAS,
            task_id=str(active),
        )

    task = AITask(
        task_type=TASK_TYPE_RAGAS,
        status=AITaskStatus.PENDING.value,
        payload={"sample_size": size, "triggered_by": str(triggered_by) if triggered_by else None},
        progress=0,
        progress_message="已排队，等待评测 worker 接手",
        idempotency_key=f"{TASK_TYPE_RAGAS}:{uuid.uuid4().hex}",
    )
    db.add(task)
    # 主键在 flush 时才求值，必须先 flush 才能把 task.id 关联给 EvalRun
    # （阶段 4 同款坑：不 flush 就写关联，落库是 NULL）。
    await db.flush()

    run = create_ragas_run(
        db, sample_size=size, triggered_by=triggered_by, ai_task_id=task.id
    )
    await db.commit()

    from app.tasks.celery_app import QUEUE_EVALUATE, dispatch_task

    async_result = dispatch_task(
        RAGAS_TASK_NAME, str(run.id), queue=QUEUE_EVALUATE, task_id=str(task.id)
    )
    if async_result is None:
        # broker 不可用：两张表一起收口 failed，否则 run 永远 pending、
        # 前端一直转圈，而 AITask 会被僵尸清理误判成排队超时。
        task.status = AITaskStatus.FAILED.value
        task.error_message = "任务队列不可用，未能派发"
        task.completed_at = datetime.now(UTC)
        run.status = EvalRunStatus.FAILED.value
        run.error_message = "任务队列不可用，未能派发"
        await db.commit()
        raise TaskQueueUnavailableError("评测任务队列暂时不可用，请稍后重试")

    logger.info("RAGAS 评测已派发 run_id=%s task_id=%s size=%d", run.id, task.id, size)
    return run, task


__all__ = [
    "EVAL_TOP_K",
    "RAGAS_TASK_NAME",
    "TASK_TYPE_RAGAS",
    "acceptable_names",
    "aggregate_quality",
    "aggregate_retrieval",
    "apply_gate",
    "build_quality_result",
    "build_retrieval_result",
    "dispatch_ragas_evaluation",
    "distinct_by_title",
    "eval_case_to_dict",
    "get_run",
    "list_cases",
    "list_runs",
    "run_quality_evaluation",
    "run_retrieval_evaluation",
    "run_trend",
]
