"""RAG 检索内核。

实现 pgvector 向量检索 + PostgreSQL 全文检索 + RRF 融合的混合检索管线。

模块划分：
- ``text_processing``  文本清洗、中文递归分块、jieba 分词 → tsvector 输入
- ``vector_search``    pgvector 余弦距离检索（异步）
- ``fulltext_search``  PostgreSQL tsvector 全文检索（异步）
- ``rrf``              Reciprocal Rank Fusion 融合（纯函数）
- ``retriever``        编排层：并发两路检索 → RRF 融合 → 写日志 → 返回结果
"""

from app.services.rag.retriever import RAGRetriever

__all__ = ["RAGRetriever"]
