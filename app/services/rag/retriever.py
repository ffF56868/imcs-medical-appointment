"""RAG 检索编排层。

顺序执行向量与全文两路检索 → RRF 融合 → 截断到 ``settings.rag_result_count``
→ 经 ``observability.log_retrieval`` 写 ``RAGRetrievalLog`` → 返回带引用信息的结构化结果。

**日志写入失败不得影响检索结果返回**：可观测性是辅助，不能反噬主路径。
落库统一收敛到 ``app/services/observability.py``，那里所有函数内部吞异常；
本模块再包一层 try/except，连字段组装阶段的意外也不能冒泡。

**两路顺序执行且互不依赖**：任一路失败都会降级为空结果（由各路模块内部处理），
不影响另一路。两路都是 LIMIT 20 的索引查询，故不追求并发——详见
``retrieve`` 内第 2 步的说明，共用 AsyncSession 并发会触发
InterfaceError，而派生独立异步引擎的收益不足以抵消其构造风险。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.knowledge import KnowledgeChunk
from app.services.llm.service import get_llm_service
from app.services.observability import log_retrieval, project_candidates
from app.services.rag.fulltext_search import FulltextHit, fulltext_search
from app.services.rag.rrf import RRFInput, reciprocal_rank_fusion
from app.services.rag.vector_search import VectorHit, vector_search

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalResult:
    """单条检索结果，携带引用信息供前端展示出处。"""

    chunk_id: uuid.UUID
    content: str
    rrf_score: float
    title: str
    doc_type: str
    source: str
    source_tier: str
    page_number: int | None = None
    page_range: str | None = None
    extra_metadata: dict[str, Any] | None = None
    # 各路贡献明细，用于调试与可观测性
    route_details: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResponse:
    """检索响应，包含结果列表与诊断信息。"""

    results: list[RetrievalResult]
    vector_hit_count: int
    keyword_hit_count: int
    both_hit_count: int
    merged_count: int
    is_zero_result: bool


class RAGRetriever:
    """RAG 检索器。无状态设计，可安全共享实例。"""

    async def retrieve(
        self,
        session: AsyncSession,
        query_text: str,
        *,
        rewritten_query: str | None = None,
        trace_id: str | None = None,
        user_id: uuid.UUID | None = None,
    ) -> RetrievalResponse:
        """执行完整的混合检索流程。

        Args:
            session: 异步数据库会话。
            query_text: 用户原始查询文本。
            rewritten_query: 改写后的查询（多轮对话指代消解后）。
                若提供，全文检索使用改写后的查询，向量检索也使用改写后的查询。
            trace_id: 链路追踪 ID，贯穿日志。
            user_id: 当前用户 ID，用于日志关联。

        Returns:
            包含检索结果与诊断信息的响应对象。
        """
        effective_query = rewritten_query or query_text
        overall_start = time.perf_counter()

        # ---------- 1. 获取查询向量 ----------
        query_embedding: list[float] = []
        try:
            llm = get_llm_service()
            vectors, _usage = await llm.embed(
                [effective_query],
                trace_id=trace_id,
                user_id=user_id,
            )
            if vectors:
                query_embedding = vectors[0]
        except Exception:
            logger.warning("获取查询向量失败，向量检索将降级为空结果", exc_info=True)

        # ---------- 2. 顺序执行两路检索 ----------
        # **为什么串行而非 asyncio.gather**：两路共用同一个 AsyncSession，
        # 而 AsyncSession 不支持并发操作——在 PostgreSQL 上两路都真正发 SQL 时
        # 会抛 InterfaceError: another operation is in progress。该缺陷在
        # SQLite 测试环境**测不出来**（非 PG 后端两路都直接降级为空、不发 SQL），
        # 只有上真实 PG 才暴露，因此必须在容器里集成验证。
        #
        # 理论上可各开独立会话并发，但那需要从 session 派生 AsyncEngine：
        # session.bind 拿到的是同步 Engine，直接喂给 AsyncSession() 会因缺少
        # .sync_engine 属性而抛错，而 retrieve() 没有单元测试覆盖，
        # 这类构造风险不该埋在这里。两路都是 LIMIT 20 的索引查询，
        # 串行与并发的耗时差仅几毫秒——用确定正确换这点延迟是划算的。
        # 若日后压测证明这两次 DB 往返是瓶颈，再用正确的 AsyncEngine 句柄改回并发。
        vector_start = time.perf_counter()
        vector_hits = await vector_search(session, query_embedding)
        vector_latency_ms = int((time.perf_counter() - vector_start) * 1000)

        ft_start = time.perf_counter()
        ft_hits = await fulltext_search(session, effective_query)
        keyword_latency_ms = int((time.perf_counter() - ft_start) * 1000)

        # ---------- 3. RRF 融合 ----------
        rrf_start = time.perf_counter()

        routes: list[RRFInput] = []
        if vector_hits:
            routes.append(
                RRFInput(
                    source="vector",
                    hits=[
                        {"chunk_id": h.chunk_id, "score": h.similarity}
                        for h in vector_hits
                    ],
                )
            )
        if ft_hits:
            routes.append(
                RRFInput(
                    source="fulltext",
                    hits=[
                        {"chunk_id": h.chunk_id, "score": h.score}
                        for h in ft_hits
                    ],
                )
            )

        rrf_results = reciprocal_rank_fusion(routes)
        rrf_latency_ms = int((time.perf_counter() - rrf_start) * 1000)

        # 截断到配置数量
        final_results = rrf_results[: settings.rag_result_count]

        # ---------- 4. 计算交叉命中数 ----------
        vector_chunk_ids = {h.chunk_id for h in vector_hits}
        ft_chunk_ids = {h.chunk_id for h in ft_hits}
        both_hit_count = len(vector_chunk_ids & ft_chunk_ids)

        # ---------- 5. 批量加载切块详情 ----------
        chunk_map = await self._load_chunks(session, [r.chunk_id for r in final_results])

        # ---------- 6. 组装结果 ----------
        retrieval_results: list[RetrievalResult] = []
        for rrf_item in final_results:
            chunk = chunk_map.get(rrf_item.chunk_id)
            if chunk is None:
                continue
            retrieval_results.append(
                RetrievalResult(
                    chunk_id=rrf_item.chunk_id,
                    content=chunk.content,
                    rrf_score=rrf_item.rrf_score,
                    title=chunk.title,
                    doc_type=chunk.doc_type,
                    source=chunk.source,
                    source_tier=chunk.source_tier,
                    page_number=chunk.page_number,
                    page_range=chunk.page_range,
                    extra_metadata=chunk.extra_metadata,
                    route_details={
                        src: {"rank": info["rank"], "score": info["score"]}
                        for src, info in rrf_item.sources.items()
                    },
                )
            )

        response = RetrievalResponse(
            results=retrieval_results,
            vector_hit_count=len(vector_hits),
            keyword_hit_count=len(ft_hits),
            both_hit_count=both_hit_count,
            merged_count=len(rrf_results),
            is_zero_result=len(retrieval_results) == 0,
        )

        # ---------- 6.5 组装日志字段（纯内存，不碰 DB）----------
        top_score = final_results[0].rrf_score if final_results else None
        candidates = self._project_candidates(retrieval_results)

        # overall_start 是端到端计时，但 rag_retrieval_logs 没有总耗时列，
        # 各路耗时已分别落库；这里保留计算是为了口径显式可见，
        # 后续批次接入监控聚合时可直接消费。
        total_latency_ms = int((time.perf_counter() - overall_start) * 1000)

        # ---------- 7. 写检索日志（失败不影响结果返回）----------
        # 落库统一走 observability.log_retrieval：它开独立短事务而不是复用这里的
        # session。原因是本方法会被 SSE 流式响应调用，客户端中途断开时请求
        # session 已进入回收，复用它只会让日志在最需要留痕的那一次写不进去；
        # 且日志与业务无原子性要求，跟着主事务回滚反而有害。
        # 外层再包一层 try/except：连「组装日志字段」这一步都不能冒泡。
        try:
            await log_retrieval(
                trace_id=trace_id,
                user_id=user_id,
                query_text=query_text,
                rewritten_query=rewritten_query,
                vector_hit_count=response.vector_hit_count,
                keyword_hit_count=response.keyword_hit_count,
                both_hit_count=response.both_hit_count,
                merged_count=response.merged_count,
                final_count=len(response.results),
                is_zero_result=response.is_zero_result,
                top_score=top_score,
                # 向量/全文/RRF 三路耗时是本轮 perf_counter 真实采集；
                # rerank 通道尚未接入本批次，恒为占位 0 / None（见交付报告缺口）。
                vector_latency_ms=vector_latency_ms,
                keyword_latency_ms=keyword_latency_ms,
                rrf_latency_ms=rrf_latency_ms,
                rerank_latency_ms=0,
                rerank_channel=None,
                rrf_k=settings.rag_rrf_k,
                candidates=candidates,
            )
        except Exception:
            logger.warning("写入 RAG 检索日志失败", exc_info=True)

        return response

    async def _load_chunks(
        self,
        session: AsyncSession,
        chunk_ids: list[uuid.UUID],
    ) -> dict[uuid.UUID, KnowledgeChunk]:
        """批量加载切块详情。空列表时直接返回空字典。"""
        if not chunk_ids:
            return {}

        stmt = select(KnowledgeChunk).where(KnowledgeChunk.id.in_(chunk_ids))
        result = await session.execute(stmt)
        chunks = result.scalars().all()
        return {c.id: c for c in chunks}

    @staticmethod
    def _project_candidates(
        results: list[RetrievalResult],
    ) -> list[dict[str, Any]]:
        """把最终结果投影成 candidates 明细（不含正文）。

        实现委托给 ``observability.project_candidates``，保证「谁写日志都是同一份
        投影口径」。**刻意不塞 chunk.content**：正文已在 knowledge_chunks 里，
        复制进高频写入的日志表会让它迅速膨胀；日志要回答的是「哪些 chunk 以第几名
        进了融合、各路给了多少分」，chunk_id / title / source / rank / rrf_score
        足够回放。
        """
        return project_candidates(results)
