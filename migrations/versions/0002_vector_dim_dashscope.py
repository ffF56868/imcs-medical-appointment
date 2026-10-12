"""向量列维度 1536 → 1024（切换 embedding 供应商至 DashScope text-embedding-v3）。

背景：主供应商从 OpenAI（text-embedding-3-small，1536 维）切换为百炼
（text-embedding-v3，输出 1024 维）。不同维度的向量不可能共存于同一列，
且 ivfflat 索引与列类型绑定，改类型前必须先 DROP 索引。

执行顺序即依赖顺序：删索引 → 改列类型 → 重建索引。
前置条件（本次已确认）：vector_embeddings 为空表——旧供应商向量尚未落库过，
无数据迁移负担。若在有旧向量的库上重放本迁移，1536→1024 的截断会直接报错，
此时应先 TRUNCATE vector_embeddings 并全量重嵌（rebuild_all_vectors 任务）。
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ivfflat 索引按 vector(1536) 建立，ALTER TYPE 前必须移除
    op.execute("DROP INDEX IF EXISTS idx_vector_embedding_ivfflat")
    op.alter_column(
        "vector_embeddings",
        "embedding",
        existing_type=Vector(dim=1536),
        type_=Vector(dim=1024),
        existing_nullable=False,
    )
    # 维度随 settings.embedding_dim 走（迁移里必须写死字面值：Alembic 重放
    # 历史时不能依赖当前环境变量，否则环境与历史错位）。1024 维同样适用
    # cosine 距离算子，lists=100 对万级切块仍是合理档位。
    op.execute(
        "CREATE INDEX idx_vector_embedding_ivfflat "
        "ON vector_embeddings USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )


def downgrade() -> None:
    # 回退到 OpenAI 1536 维。同样要求回退前 vector_embeddings 为空或已重嵌，
    # 1024→1536 补零没有语义，数据库层也不允许隐式变宽。
    op.execute("DROP INDEX IF EXISTS idx_vector_embedding_ivfflat")
    op.alter_column(
        "vector_embeddings",
        "embedding",
        existing_type=Vector(dim=1024),
        type_=Vector(dim=1536),
        existing_nullable=False,
    )
    op.execute(
        "CREATE INDEX idx_vector_embedding_ivfflat "
        "ON vector_embeddings USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )
