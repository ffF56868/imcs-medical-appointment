"""知识库模型：文档、切块、向量。

三张表分离的设计理由：

- ``knowledge_documents``：原始资料与 sha256 内容指纹，用于判断哪些文档需要重新索引
- ``knowledge_chunks``：切块文本 + tsvector 全文索引，是检索返回的引用单位
- ``vector_embeddings``：向量与模型版本**单独成表**

向量单独成表而非并入切块表，是为了换 embedding 模型时只重建向量表，
不必重写切块文本。``model_version`` 字段记录每条向量由哪个模型生成，
换模型后能识别出哪些向量已过期、需要重算。

向量索引用 HNSW 而非 IVFFlat：HNSW 无需预先训练聚类中心、召回率更高、
支持增量插入。IVFFlat 需要先灌数据再建索引，且新增数据落在错误聚类时召回会退化。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from app.core.config import settings

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from app.models.base import Base, TimestampMixin, UUIDPrimaryKey
from app.models.types import JSONBText, TSVectorText

try:  # pgvector 只在 PostgreSQL 环境安装
    from pgvector.sqlalchemy import Vector

    PGVECTOR_AVAILABLE = True
except ImportError:  # pragma: no cover - 测试环境降级路径
    Vector = None  # type: ignore[assignment]
    PGVECTOR_AVAILABLE = False


class VectorType(TypeDecorator):
    """可移植向量列类型。

    PostgreSQL 上用 pgvector 原生 ``vector(n)`` 类型，享受 HNSW 索引与
    ``<=>`` 余弦距离运算符；SQLite 测试环境退化为 JSON 文本，
    此时向量检索由应用层余弦相似度兜底（测试只验证业务逻辑，不验证检索性能）。
    """

    impl = Text
    cache_ok = True

    def __init__(self, dim: int) -> None:
        self.dim = dim
        super().__init__()

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql" and PGVECTOR_AVAILABLE:
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql" and PGVECTOR_AVAILABLE:
            return value
        return json.dumps(list(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql" and PGVECTOR_AVAILABLE:
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (ValueError, TypeError):
                return None
        return value


class KnowledgeDocument(UUIDPrimaryKey, TimestampMixin, Base):
    """知识文档：临床指南、药品说明书、检验参考值、病症与药物资料。"""

    __tablename__ = "knowledge_documents"
    __table_args__ = (
        Index("idx_knowledge_doc_type", "doc_type"),
        Index("idx_knowledge_doc_content_hash", "content_hash"),
    )

    title: Mapped[str] = mapped_column(String(300), nullable=False)
    # guideline（临床指南）/ drug_manual（药品说明书）/ lab_reference（检验参考值）
    # / condition（病症资料）/ drug（药物资料）/ textbook（教材）
    doc_type: Mapped[str] = mapped_column(String(30), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # sha256 内容指纹：增量向量化时比对它判断文档是否变更
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(200), default="未标注来源", nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(2000))
    # authoritative（权威机构）/ professional（专业机构）/ general（一般资料）/ unverified（待核实）
    source_tier: Mapped[str] = mapped_column(String(20), default="unverified", nullable=False)
    # 来源发布时间，用于判断资料时效性
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # 关联的科室，用于分诊时的科室定向召回
    department_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("departments.id"),
    )
    extra_metadata: Mapped[dict | None] = mapped_column(JSONBText())
    # 是否已通过人工审核，未审核资料默认不参与检索
    is_reviewed: Mapped[bool] = mapped_column(default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    # 索引状态：pending / indexed / stale / failed
    index_status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    chunks: Mapped[list["KnowledgeChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class KnowledgeChunk(UUIDPrimaryKey, Base):
    """知识切块。检索返回的引用单位，承载溯源信息。"""

    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_doc_index"),
        Index("idx_chunk_document", "document_id"),
        # 切块内容 sha256，增量向量化时精确到切块级别判断是否需重算向量
        Index("idx_chunk_content_hash", "content_hash"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # 溯源信息，沿用旧项目已验证的引用设计
    page_number: Mapped[int | None] = mapped_column(Integer, comment="PDF 页码")
    page_range: Mapped[str | None] = mapped_column(String(20), comment="页码区间，如 12-15")
    chunk_start_index: Mapped[int | None] = mapped_column(Integer)
    chunk_end_index: Mapped[int | None] = mapped_column(Integer)
    # 分块策略版本，参数调整后用于识别旧切块
    chunking_strategy: Mapped[str] = mapped_column(String(40), default="recursive-chinese-v1")
    # 反规范化字段：检索时无需 JOIN 文档表即可过滤与展示引用来源。
    # 冗余存储换取查询性能，写入时由向量化任务保证一致性。
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(30), nullable=False)
    source: Mapped[str] = mapped_column(String(200), nullable=False)
    source_tier: Mapped[str] = mapped_column(String(20), nullable=False)
    extra_metadata: Mapped[dict | None] = mapped_column(JSONBText())

    # 全文检索列。由应用层 jieba 分词后写入，配 simple 配置避免二次分词。
    # 不用 pg_jieba / zhparser 扩展：那需要定制 PostgreSQL 镜像并在容器内编译，
    # 而应用层分词还能挂医学自定义词典（保证「对乙酰氨基酚」不被切碎）。
    # SQLite 测试环境无 tsvector，自动退化为 TEXT，仅存分词结果。
    tsv = mapped_column(TSVectorText(), nullable=True)

    document: Mapped["KnowledgeDocument"] = relationship(back_populates="chunks")


class VectorEmbedding(Base):
    """向量嵌入表。与切块一对一，单独成表便于换模型时只重建向量。"""

    __tablename__ = "vector_embeddings"
    __table_args__ = (
        Index("idx_vector_chunk", "chunk_id"),
        Index("idx_vector_model_version", "model_version"),
    )

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("knowledge_chunks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    # 维度跟随 settings.embedding_dim：OpenAI text-embedding-3-small 是 1536，
    # DashScope text-embedding-v3 是 1024。换 embedding 模型必须同步改
    # EMBEDDING_DIM 并跑 0002 式的全列重建（不同维度连类型都不兼容）。
    embedding = mapped_column(VectorType(settings.embedding_dim), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
