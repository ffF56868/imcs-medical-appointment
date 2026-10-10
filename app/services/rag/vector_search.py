"""pgvector 余弦距离检索（异步）。

使用 pgvector 的 ``<=>`` 运算符计算余弦距离，按距离升序取 top-k。
相似度 = 1 - cosine_distance，作为该路的原始分。

**降级策略**：非 PostgreSQL 后端（如 SQLite 测试环境）或 pgvector
不可用时，优雅返回空列表而非抛异常。向量检索的真实 SQL 正确性
由容器内集成测试验证，单元测试只覆盖降级行为。

**min_score 语义**：``settings.rag_min_score`` 作用在向量相似度上，
作为融合前的入选门槛。RRF 分数与余弦相似度量纲不同、不可比，
因此 min_score 不应作用在融合后——否则会因阈值错位而误杀有效结果。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.knowledge import PGVECTOR_AVAILABLE

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VectorHit:
    """向量检索的单条命中结果。"""

    chunk_id: uuid.UUID
    similarity: float  # 1 - cosine_distance，范围 [0, 1]


async def vector_search(
    session: AsyncSession,
    query_embedding: list[float],
    *,
    limit: int | None = None,
    min_score: float | None = None,
) -> list[VectorHit]:
    """执行 pgvector 余弦距离检索。

    Args:
        session: 异步数据库会话。
        query_embedding: 查询向量，维度须与 ``settings.embedding_dim`` 一致。
        limit: 返回条数上限，默认取 ``settings.rag_vector_fetch_count``。
        min_score: 相似度门槛，低于此值的命中被过滤。
            默认取 ``settings.rag_min_score``。设为 0 表示不过滤。

    Returns:
        按相似度降序排列的命中列表。非 PG 后端或 pgvector 不可用时返回空列表。
    """
    if limit is None:
        limit = settings.rag_vector_fetch_count
    if min_score is None:
        min_score = settings.rag_min_score

    # 非 PG 或 pgvector 未安装时优雅降级
    if not PGVECTOR_AVAILABLE:
        logger.debug("pgvector 不可用，向量检索降级为空结果")
        return []

    # 检查当前会话是否连接 PostgreSQL
    dialect_name = session.bind.dialect.name if session.bind else ""
    if dialect_name != "postgresql":
        logger.debug("非 PostgreSQL 后端（%s），向量检索降级为空结果", dialect_name)
        return []

    if not query_embedding:
        return []

    try:
        # pgvector <=> 是余弦距离运算符，距离越小越相似。
        #
        # **必须显式转 vector 类型**：直接把 Python list 作为参数绑定时，psycopg
        # 会把它适配成 PG 数组 `double precision[]`，而 pgvector 的距离运算符
        # 不接受数组类型——实测报错 `operator does not exist: vector <=> double
        # precision[]`。SQLite 单测两路都降级为空，触达不到这条 SQL，所以这个
        # bug 只在真实 PG 上暴露。解法是把向量转成 pgvector 的文本字面量
        # `[x,y,...]` 再 `::vector` 显式转换；文本字面量由我们构造、只含数字与
        # 分隔符，不存在注入面。
        query_vec_literal = "[" + ",".join(repr(float(x)) for x in query_embedding) + "]"

        # 为什么用整条 raw SQL，而不是 ORM select() 混 text()/literal_column()：
        # 这条路上连环踩过三个坑，全部实测复现——
        #   1) text("expr AS similarity") 混入 select()：SQL 别名不注册为结果列
        #      key，row.similarity 抛 AttributeError；
        #   2) literal_column("... :query_vec ...").label()：literal_column 把
        #      字符串当原始 SQL 片段**原样输出**，不做绑定参数解析，生成的 SQL
        #      里裸 :query_vec 直达 psycopg，报 syntax error at or near ":"；
        #   3) select() 里各 text() 片段间的同名 bindparam 是否会被合并成同一个
        #      参数，行为依赖版本细节，不值得赌。
        # 整条 text() 则全部交给 SQLAlchemy 的 text 解析器：`:name` 统一编译为
        # 命名参数，列标签由数据库游标提供，结果行按名访问最稳。
        # min_score 的条件串在 Python 侧拼接，但拼接内容只有固定的 TRUE / 阈值
        # 比较式两种，参数本身走绑定，无注入面。
        where_clause = (
            "WHERE 1 - (embedding <=> CAST(:query_vec AS vector)) >= :min_score"
            if min_score > 0
            else ""
        )
        sql = text(
            f"""
            SELECT chunk_id,
                   1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
            FROM vector_embeddings
            {where_clause}
            ORDER BY embedding <=> CAST(:query_vec AS vector)
            LIMIT :limit
            """
        )

        # SAVEPOINT 隔离：在 PostgreSQL 上，任一 SQL 失败会让整个事务进入
        # aborted 状态，后续所有查询都报 InFailedSqlTransaction。用 begin_nested
        # 把本次查询包进 SAVEPOINT，失败时只回滚到保存点，外层事务与另一路
        # 检索不受影响——否则「降级为空、不影响另一路」的承诺在真实 PG 上不成立
        # （实测中向量路的类型错误正是这样连累了全文路）。
        async with session.begin_nested():
            result = await session.execute(
                sql,
                {"query_vec": query_vec_literal, "min_score": min_score, "limit": limit},
            )
            rows = result.mappings().all()

        hits: list[VectorHit] = []
        for row in rows:
            hits.append(
                VectorHit(
                    chunk_id=row["chunk_id"],
                    similarity=float(row["similarity"]),
                )
            )
        return hits

    except Exception:
        # pgvector 扩展未在数据库中启用、表不存在等运行时错误
        # 不应阻断主检索路径，降级为空结果并记录
        logger.warning("向量检索执行失败，降级为空结果", exc_info=True)
        return []
