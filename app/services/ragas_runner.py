"""RAGAS 评测运行器（阶段 6B）：旧 BackgroundTasks 链路的 Celery 化替代。

与旧实现（``app/routers/evaluation.py`` 的 ``execute_ragas_evaluation``）的
三点差异：

1. **LLM/Embedding 接入方式**。旧版经 langchain-openai 的 ``get_chat_model``
   绑 OpenAI；IMCS 的主供应商是 DashScope 的 OpenAI 兼容端点。
   ragas 0.2.15 的 ``llm_factory`` / ``embedding_factory`` 只认
   model 名不认自定义 base_url，因此走官方支持的包装路径：
   ``LangchainLLMWrapper(ChatOpenAI(base_url=兼容端点))``，
   嵌入同理。供应商、密钥、端点全部读 IMCS 配置，与生产 LLM 服务同源。
2. **Vertex 垫片必须先执行**。ragas 0.2.15 导入期硬依赖
   ``langchain_community.chat_models.vertexai``（LangChain 1.x 已删除该模块），
   ``app/ragas_compat`` 的 shim 必须在任何 ragas 导入之前调用。
   shim 是幂等的，但「先调 shim 再 import ragas」的顺序约束是硬性的——
   模块级不做导入，全部延迟到运行函数内部，防止未来有人
   ``from ... import`` 本模块时把 ragas 拖进 API 进程的导入链。
3. **执行体在 Celery worker**。旧版用进程内 ``threading.Lock`` + BackgroundTasks
   防并发，两个 API 副本各自有锁、重启即丢任务。这里一次运行 = 一条
   ``eval_runs(kind=ragas)`` 记录 + 一个 Celery 任务（acks_late、独立
   evaluate 队列、并发 1），状态机由数据库承载；进程崩溃后
   monitoring 侧的僵尸清理会把停在 running 的运行标 failed，不再永久占位。

四指标：faithfulness / answer_relevancy（医疗中文口径，
见 ``app/ragas_metrics``，其 noncommittal 定义校准过「仅供参考」类
医疗免责语，避免正确答案被误判回避）/ context_precision / context_recall。
逐题得分与均值都落 ``eval_runs``，整体值写指标列、明细进 ``results``。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from random import SystemRandom
from typing import Any

from sqlalchemy import select

from app.core.config import settings
from app.core.database import get_sync_db
from app.models.evaluation import EvalCase, EvalKind, EvalRun, EvalRunStatus
from app.models.knowledge import KnowledgeDocument

logger = logging.getLogger(__name__)

RAGAS_DEFAULT_SAMPLE_SIZE = 10
RAGAS_MAX_SAMPLE_SIZE = 100
RAGAS_REFERENCE_MAX_CHARS = 6000
RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def normalize_ragas_score(value: object) -> float | None:
    """ragas 分数归一：非数值/非有限值 → None（而不是 0 或 NaN）。"""
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return round(score, 4) if math.isfinite(score) else None


def calculate_ragas_averages(results: list[dict[str, Any]]) -> dict[str, float | None]:
    """逐题得分求均值。某指标全为 None 时均值也是 None——
    「评不了」与「评了是 0 分」必须可区分。"""
    averages: dict[str, float | None] = {}
    for name in RAGAS_METRIC_NAMES:
        values = [
            score
            for r in results
            if (score := normalize_ragas_score(r.get(name))) is not None
        ]
        averages[name] = round(sum(values) / len(values), 4) if values else None
    return averages


def build_ragas_clients() -> tuple[Any, Any]:
    """构造 (llm, embeddings) 供 ragas.evaluate 使用。

    必须在 ``enable_ragas_langchain_compatibility()`` 之后调用（本函数内部
    先调 shim，保证任何调用顺序下都安全）。供应商选择与生产一致：
    llm_primary_provider=dashscope 时用百炼兼容端点，否则 OpenAI。
    """
    from app.ragas_compat import enable_ragas_langchain_compatibility

    enable_ragas_langchain_compatibility()

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    # **模型名必须走生产的供应商场景映射，而不是直接读 settings.llm_model_chat**。
    # llm_model_chat 是 OpenAI 语义的名字（gpt-4.1-mini）；DashScope 适配器
    # 有自己的 model_map（全场景 qwen-plus）。上一轮直接拿 gpt-4.1-mini 打到
    # 百炼兼容端点，HTTP 404 model_not_found，四指标全批打分失败。
    from app.services.llm.provider import get_adapter, get_configured_providers
    from app.services.llm.types import LLMScenario

    providers = get_configured_providers()
    if not providers:
        raise RuntimeError("RAGAS 评测需要可用的 LLM 供应商密钥")
    adapter = get_adapter(providers[0])
    api_key = adapter.api_key
    base_url = adapter.base_url
    chat_model = adapter.resolve_model(LLMScenario.CHAT, use_fallback=False)
    embed_model = getattr(adapter, "embedding_model", None) or settings.embedding_model

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=chat_model,
            api_key=api_key,
            base_url=base_url,
            temperature=settings.llm_temperature,
            timeout=int(settings.llm_read_timeout),
            max_retries=settings.llm_max_retries,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=embed_model,
            api_key=api_key,
            base_url=base_url,
            # **必须关闭 len-safe 预处理**：langchain-openai 默认先把文本
            # 切成 token id 列表再发（给 OpenAI 原生端点算长度用的），
            # 百炼兼容端点只认字符串，token id 数组直接 400
            # "contents is neither str nor list of str"——answer_relevancy
            # 的整批 embedding 调用就是这么全挂的（worker 实测复现）。
            check_embedding_ctx_length=False,
        )
    )
    return llm, embeddings


def reference_for_case(session: Any, case: dict[str, Any]) -> tuple[str, bool]:
    """参考资料 = 期望命中的知识文档正文（旧版按三表反查，IMCS 一表收敛）。

    IMCS 的知识库只有一张 KnowledgeDocument（旧系统拆病症/药物/文档三表），
    expected_name 直接按 title 查；查不到时返回占位并标 available=False，
    该题在结果里显式标降级——悄悄剔除会让样本量说谎。
    """
    names = [case["expected_name"], *case.get("alternative_names", [])]
    names = [n for n in names if n]
    if not names:
        return "未提供参考资料", False
    rows = session.execute(
        select(KnowledgeDocument.title, KnowledgeDocument.content).where(
            KnowledgeDocument.title.in_(names)
        )
    ).all()
    by_title = {row[0]: row[1] for row in rows}
    for name in names:
        content = by_title.get(name)
        if content:
            return f"资料标题：{name}\n资料内容：{content}"[:RAGAS_REFERENCE_MAX_CHARS], True
    return str(case["expected_name"]), False


async def _collect_answers(
    questions: list[str], trace_prefix: str
) -> list[Any]:
    """在独立事件循环里跑完全部评测问答（worker 线程内无既有循环）。"""
    from app.core.database import async_session_factory
    from app.services.eval_answer import answer_for_evaluation

    outs = []
    async with async_session_factory() as session:
        for idx, question in enumerate(questions):
            outs.append(
                await answer_for_evaluation(
                    session, question, trace_id=f"{trace_prefix}-{idx}"
                )
            )
    return outs


def execute_ragas_evaluation(run_id: str) -> dict[str, Any]:
    """执行一次完整的 RAGAS 评测（同步入口，Celery 任务体调用）。

    流程：CAS 认领 running → 抽样题库 → 逐题生成回答 → ragas 打分 →
    均值与明细落库。返回结构化摘要；异常由调用方（任务体）收口。
    """
    # **shim 必须先于任何 ragas 顶层导入**：ragas 0.2.15 的 ragas.llms.base
    # 在模块级 `from langchain_community.chat_models.vertexai import ChatVertexAI`，
    # 该模块在新版 langchain-community 里已删除——上一轮把 shim 放在
    # build_ragas_llm_pair 里，结果第 178 行的 `from ragas import evaluate`
    # 先执行，worker 直接 ModuleNotFoundError（阶段4踩过同款：收口链路都对，
    # 就差这一行导入顺序）。
    from app.ragas_compat import enable_ragas_langchain_compatibility

    enable_ragas_langchain_compatibility()
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness
    from app.ragas_metrics import build_medical_answer_relevancy_metric

    with get_sync_db() as session:
        run = session.get(EvalRun, uuid.UUID(run_id))
        if run is None or run.kind != EvalKind.RAGAS.value:
            raise ValueError(f"评测运行不存在或类型不符：{run_id}")
        if run.status != EvalRunStatus.PENDING.value:
            # CAS：非 pending 直接退出——重复投递/重试竞态下只有一个赢家
            logger.info("RAGAS 运行状态为 %s，跳过 run_id=%s", run.status, run_id)
            return {"ok": True, "skipped": True, "run_id": run_id, "status": run.status}

        cases = [
            {
                "id": str(c.id),
                "question": c.question,
                "expected_name": c.expected_name,
                "expected_type": c.expected_type,
                "category": c.category,
                "alternative_names": list(c.alternative_names or []),
                "expected_refusal": bool(c.expected_refusal),
            }
            for c in session.execute(
                select(EvalCase).order_by(EvalCase.created_at)
            ).scalars().all()
        ]
        if not cases:
            raise RuntimeError("没有可运行的评测题，请先写入题库种子")

        sample_size = min(int(run.total_count or RAGAS_DEFAULT_SAMPLE_SIZE), len(cases))
        # SystemRandom：抽样影响成本与结果分布，不能用可预测的随机源
        sampled = SystemRandom().sample(cases, k=sample_size)

        # 知识库指纹（与 evaluation_service 同口径：按 id 序拼接 content_hash）
        hashes = [
            row[0]
            for row in session.execute(
                select(KnowledgeDocument.content_hash).order_by(KnowledgeDocument.id)
            ).all()
        ]
        run.total_count = len(sampled)
        run.knowledge_document_count = len(hashes)
        run.knowledge_hash = (
            hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest() if hashes else None
        )
        run.status = EvalRunStatus.RUNNING.value
        from datetime import UTC, datetime

        run.started_at = datetime.now(UTC)
        session.add(run)
        session.commit()
        session.refresh(run)

    trace_prefix = f"ragas-{run_id[:8]}"
    # 用 celery_app.run_async 而不是裸 asyncio.run：它会在同一活循环内
    # aclose_adapter_clients 收尾 LLM 客户端缓存。评测问答走 llm_service
    # （AsyncOpenAI 客户端绑定首次使用的循环），裸 asyncio.run 关循环后
    # 缓存里留着死客户端，GC 触发 aclose 被调度到已关闭循环上——阶段 4
    # 实测刷出成片 Event loop is closed。评测跑整轮问答，正是最踩这个的形态。
    from app.tasks.celery_app import run_async

    answers = run_async(_collect_answers([c["question"] for c in sampled], trace_prefix))

    references: list[tuple[str, bool]] = []
    with get_sync_db() as session:
        for case in sampled:
            references.append(reference_for_case(session, case))

    samples: list[SingleTurnSample] = []
    question_results: list[dict[str, Any]] = []
    for case, ans, (reference, ref_available) in zip(sampled, answers, references):
        samples.append(
            SingleTurnSample(
                user_input=case["question"],
                response=ans.answer,
                retrieved_contexts=list(ans.contexts),
                reference=reference,
            )
        )
        question_results.append(
            {
                "case_id": case["id"],
                "question": case["question"],
                "category": case["category"],
                "expected_name": case["expected_name"],
                "answer": ans.answer,
                "retrieved_count": ans.retrieved_count,
                "context_titles": ans.context_titles,
                "answer_degraded": ans.degraded,
                "reference_available": ref_available,
            }
        )

    llm, embeddings = build_ragas_clients()
    ragas_result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(),
            build_medical_answer_relevancy_metric(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=False,
    )

    # raise_exceptions=False 时失败题的指标可能整批缺失，按长度取 min 防御，
    # 而不是 strict zip 让最后一次 DB 写入前炸掉（前面已花的 token 不该白烧）。
    for result, scores in zip(question_results, ragas_result.scores):
        for name in RAGAS_METRIC_NAMES:
            result[name] = normalize_ragas_score(scores.get(name))

    averages = calculate_ragas_averages(question_results)

    with get_sync_db() as session:
        run = session.get(EvalRun, uuid.UUID(run_id))
        run.completed_count = len(question_results)
        for name in RAGAS_METRIC_NAMES:
            setattr(run, name, averages[name])
        # 按 category 下钻（与检索层同一可解释性原则：均值会掩盖单类塌陷）
        by_category: dict[str, dict[str, float]] = {}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for r in question_results:
            grouped.setdefault(str(r.get("category") or "未分类"), []).append(r)
        for category, items in grouped.items():
            metric = {}
            for name in RAGAS_METRIC_NAMES:
                values = [
                    v for v in (r.get(name) for r in items) if v is not None
                ]
                if values:
                    metric[name] = round(sum(values) / len(values), 4)
            by_category[category] = metric
        run.metrics = {
            **{k: averages[k] for k in RAGAS_METRIC_NAMES},
            "by_category": by_category,
            "degraded_answer_count": sum(1 for a in answers if a.degraded),
            "reference_missing_count": sum(1 for _, ok in references if not ok),
        }
        run.results = question_results
        run.status = EvalRunStatus.COMPLETED.value
        run.completed_at = datetime.now(UTC)
        session.add(run)
        session.commit()

    logger.info("RAGAS 评测完成 run_id=%s 均分=%s", run_id, averages)
    return {"ok": True, "run_id": run_id, "averages": averages}


def create_ragas_run(
    db: Any,
    *,
    sample_size: int = RAGAS_DEFAULT_SAMPLE_SIZE,
    triggered_by: uuid.UUID | None = None,
    ai_task_id: uuid.UUID | None = None,
) -> EvalRun:
    """登记一条 pending 的 RAGAS 运行（API 侧调用，随后派发 Celery 任务）。"""
    size = max(1, min(int(sample_size), RAGAS_MAX_SAMPLE_SIZE))
    run = EvalRun(
        kind=EvalKind.RAGAS.value,
        status=EvalRunStatus.PENDING.value,
        total_count=size,
        triggered_by=triggered_by,
        ai_task_id=ai_task_id,
        retrieval_config={
            "rag_result_count": settings.rag_result_count,
            "rag_rrf_k": settings.rag_rrf_k,
            "sample_size": size,
        },
    )
    db.add(run)
    return run


__all__ = [
    "RAGAS_DEFAULT_SAMPLE_SIZE",
    "RAGAS_MAX_SAMPLE_SIZE",
    "calculate_ragas_averages",
    "create_ragas_run",
    "execute_ragas_evaluation",
    "normalize_ragas_score",
    "reference_for_case",
]
