"""Reciprocal Rank Fusion（RRF）融合。

纯函数模块，不碰数据库、不调 LLM。

公式：``score(d) = Σ_r 1 / (k + rank_r(d))``

- k 取 ``settings.rag_rrf_k``（默认 60，论文推荐值）
- rank 从 **1** 开始
- 文档在某路缺席则该路不贡献分数（不补零、不惩罚）
- 同分时用 chunk_id 作次级排序键，保证确定性（分页与测试不抖动）
- 支持任意路数，便于将来加重排序模型作为第三路

输入输出数据结构供下游对接：

输入::

    [
        {
            "source": "vector",          # 来源标记
            "hits": [                     # 有序候选列表（按原始分降序）
                {"chunk_id": uuid, "score": 0.92},
                ...
            ]
        },
        {
            "source": "fulltext",
            "hits": [...]
        },
    ]

输出::

    [
        {
            "chunk_id": uuid,
            "rrf_score": 0.03278,         # RRF 融合分
            "sources": {                   # 各路贡献明细
                "vector": {"rank": 1, "score": 0.92},
                "fulltext": {"rank": 3, "score": 0.45},
            }
        },
        ...
    ]
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings


@dataclass(slots=True)
class RRFInput:
    """单路检索结果。"""

    source: str  # 来源标记，如 "vector" / "fulltext"
    hits: list[dict[str, Any]]  # [{"chunk_id": uuid, "score": float}, ...]


@dataclass(slots=True)
class RRFResult:
    """RRF 融合后的单条结果。"""

    chunk_id: uuid.UUID
    rrf_score: float
    sources: dict[str, dict[str, Any]]  # {"vector": {"rank": 1, "score": 0.92}}


def reciprocal_rank_fusion(
    routes: list[RRFInput],
    *,
    k: int | None = None,
) -> list[RRFResult]:
    """执行 RRF 融合。

    Args:
        routes: 多路有序候选列表。每路包含来源标记与按原始分降序排列的命中。
        k: RRF 平滑常数，默认取 ``settings.rag_rrf_k``。
            较大的 k 使排名差异对分数的影响更平缓。

    Returns:
        按 rrf_score 降序排列的融合结果。同分时按 chunk_id 升序保证确定性。
        空输入返回空列表。
    """
    if k is None:
        k = settings.rag_rrf_k

    if not routes:
        return []

    # 累加每个 chunk 在各路的 RRF 贡献
    scores: dict[uuid.UUID, float] = {}
    details: dict[uuid.UUID, dict[str, dict[str, Any]]] = {}

    for route in routes:
        for rank_zero_based, hit in enumerate(route.hits):
            chunk_id = hit["chunk_id"]
            raw_score = hit.get("score", 0.0)
            rank = rank_zero_based + 1  # rank 从 1 开始

            contribution = 1.0 / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution

            if chunk_id not in details:
                details[chunk_id] = {}
            details[chunk_id][route.source] = {
                "rank": rank,
                "score": raw_score,
            }

    # 构造结果并按 (rrf_score DESC, chunk_id ASC) 排序
    results: list[RRFResult] = []
    for chunk_id, rrf_score in scores.items():
        results.append(
            RRFResult(
                chunk_id=chunk_id,
                rrf_score=rrf_score,
                sources=details[chunk_id],
            )
        )

    results.sort(key=lambda r: (-r.rrf_score, r.chunk_id))
    return results
