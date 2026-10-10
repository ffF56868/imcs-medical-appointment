"""PostgreSQL 全文检索（异步）。

使用 ``tsv @@ to_tsquery('simple', :q)`` 匹配 + ``ts_rank_cd`` 排序。

**对称性要求（关键）**：写入侧用 ``tokenize_for_tsv`` 分词后以空格连接存入
``KnowledgeChunk.tsv``（配 ``simple`` 配置避免 PG 二次分词），查询侧
**必须**用同一个 ``tokenize_for_tsv`` 对查询文本分词，再拼成 ``to_tsquery``
表达式。若任一侧改用不同分词方式或 PG 配置，token 对齐失败，
会**静默返回零结果**且不报错——这是 RAG 最常见的隐性故障。

**为什么用 OR（``|``）而不是 AND（``&``）连接查询词**：中文自然语言查询
分词后常有 5-10 个 token，AND 语义要求文档**同时包含全部** token 才命中。
患者提问如「高血压患者饮食要注意什么」，几乎不可能有任何文档同时含齐
所有词，结果是全文检索一路**常态性零命中**，RRF 融合退化成纯向量检索，
「向量 + 全文 + RRF」的混合检索名存实亡。改用 OR 后，命中任一 token 即入选，
由 ``ts_rank_cd`` 按命中词的数量与权重排序——命中越多分越高，天然形成
「相关度优先、宽松兜底」的效果，这才是全文检索在混合检索中该扮演的角色。

**降级策略**：非 PostgreSQL 后端（tsv 退化为 TEXT）时优雅返回空列表。
全文检索的真实 SQL 正确性由容器内集成测试验证（SQLite 下两路均降级为空，
单测覆盖不到 SQL 本身）。
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.rag.text_processing import tokenize_for_tsv

logger = logging.getLogger(__name__)

# tsquery 语法字符：在 to_tsquery 里是操作符或定界符，出现在 token 中会破坏
# 查询语法。中文 token 一般不含，但中英混合查询（英文所有格 's、Q&A 等）可能触发。
_TSQUERY_SPECIAL_RE = re.compile(r"[&|!():*'<>\\]")


@dataclass(slots=True)
class FulltextHit:
    """全文检索的单条命中结果。"""

    chunk_id: uuid.UUID
    score: float  # ts_rank_cd 返回值，无固定范围，仅用于路内排序


async def fulltext_search(
    session: AsyncSession,
    query_text: str,
    *,
    limit: int | None = None,
) -> list[FulltextHit]:
    """执行 PostgreSQL 全文检索。

    Args:
        session: 异步数据库会话。
        query_text: 用户原始查询文本。函数内部会用 ``tokenize_for_tsv``
            分词并拼成 ``to_tsquery`` 表达式。
        limit: 返回条数上限，默认取 ``settings.rag_keyword_fetch_count``。

    Returns:
        按 ts_rank_cd 降序排列的命中列表。
        非 PG 后端、空查询或全停用词时返回空列表。
    """
    if limit is None:
        limit = settings.rag_keyword_fetch_count

    # 检查当前会话是否连接 PostgreSQL
    dialect_name = session.bind.dialect.name if session.bind else ""
    if dialect_name != "postgresql":
        logger.debug("非 PostgreSQL 后端（%s），全文检索降级为空结果", dialect_name)
        return []

    # 分词并构造 to_tsquery 表达式
    tokenized = tokenize_for_tsv(query_text)
    if not tokenized:
        # 空查询或全部为停用词/标点
        return []

    # 过滤掉含 tsquery 语法字符的 token，避免破坏查询语法（中英混合查询时可能出现）。
    # 用 OR（|）连接：命中任一 token 即入选，由 ts_rank_cd 按命中词数与权重排序，
    # 命中越多分越高。若用 AND（&），中文长查询几乎必然零命中——详见模块 docstring。
    tokens = [t for t in tokenized.split() if not _TSQUERY_SPECIAL_RE.search(t)]
    if not tokens:
        return []
    # 例如 "布洛芬 退烧 儿童" → "布洛芬 | 退烧 | 儿童"
    tsquery_expr = " | ".join(tokens)

    try:
        # 使用原生 SQL：to_tsquery + cross join 无法用 ORM select 干净表达，
        # 且 ts_rank_cd 的参数需要直接引用 tsquery 别名
        raw_sql = text("""
            SELECT
                kc.id AS chunk_id,
                ts_rank_cd(kc.tsv, query) AS rank_score
            FROM knowledge_chunks kc,
                 to_tsquery('simple', :tsquery) AS query
            WHERE kc.tsv @@ query
            ORDER BY rank_score DESC
            LIMIT :limit
        """)

        # SAVEPOINT 隔离，理由同 vector_search：防止一路 SQL 失败污染整个事务，
        # 使另一路检索连带 InFailedSqlTransaction。
        async with session.begin_nested():
            result = await session.execute(
                raw_sql,
                {"tsquery": tsquery_expr, "limit": limit},
            )
            rows = result.all()

        hits: list[FulltextHit] = []
        for row in rows:
            hits.append(
                FulltextHit(
                    chunk_id=row.chunk_id,
                    score=float(row.rank_score),
                )
            )
        return hits

    except Exception:
        # tsv 列为空、表不存在、PG 扩展问题等运行时错误
        # 不应阻断主检索路径，降级为空结果并记录
        logger.warning("全文检索执行失败，降级为空结果", exc_info=True)
        return []
