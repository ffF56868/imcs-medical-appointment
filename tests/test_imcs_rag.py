"""RAG 检索内核单元测试（纯内存 / SQLite，不依赖 PostgreSQL）。

覆盖范围：
- RRF 融合：单路、双路、多路、缺席文档、同分确定性、k 值影响、空输入
- 文本清洗与分块：中文长文本切分、重叠、空文本、超长单段
- tokenize_for_tsv：中文分词结果、标点过滤、空串
- 降级行为：向量/全文检索在非 PG 后端返回空而非抛异常

向量与全文检索的真实 SQL 正确性由主代理在容器里集成验证，
这里只验证纯函数逻辑与降级路径。
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.rag.rrf import RRFInput, reciprocal_rank_fusion
from app.services.rag.text_processing import clean_text, split_text, tokenize_for_tsv


# ==================== RRF 融合测试 ====================


class TestRRF:
    """Reciprocal Rank Fusion 单元测试。"""

    def _make_input(
        self, source: str, chunk_ids: list[uuid.UUID], scores: list[float]
    ) -> RRFInput:
        return RRFInput(
            source=source,
            hits=[
                {"chunk_id": cid, "score": s}
                for cid, s in zip(chunk_ids, scores)
            ],
        )

    def test_empty_input(self) -> None:
        """空路由列表返回空结果。"""
        assert reciprocal_rank_fusion([]) == []

    def test_empty_hits(self) -> None:
        """路由存在但命中为空时返回空结果。"""
        route = RRFInput(source="vector", hits=[])
        assert reciprocal_rank_fusion([route]) == []

    def test_single_route(self) -> None:
        """单路输入时 RRF 分数 = 1/(k+rank)。"""
        ids = [uuid.uuid4() for _ in range(3)]
        route = self._make_input("vector", ids, [0.9, 0.8, 0.7])
        results = reciprocal_rank_fusion([route], k=60)

        assert len(results) == 3
        # rank 1: 1/61 ≈ 0.01639
        assert abs(results[0].rrf_score - 1.0 / 61) < 1e-9
        assert results[0].chunk_id == ids[0]
        # rank 2: 1/62
        assert abs(results[1].rrf_score - 1.0 / 62) < 1e-9
        # rank 3: 1/63
        assert abs(results[2].rrf_score - 1.0 / 63) < 1e-9

    def test_dual_route_overlap(self) -> None:
        """双路有交叉文档时，交叉文档分数高于仅出现在一路的文档。"""
        shared_id = uuid.uuid4()
        only_vector_id = uuid.uuid4()
        only_ft_id = uuid.uuid4()

        vector_route = self._make_input(
            "vector", [shared_id, only_vector_id], [0.95, 0.8]
        )
        ft_route = self._make_input(
            "fulltext", [shared_id, only_ft_id], [0.5, 0.3]
        )

        results = reciprocal_rank_fusion([vector_route, ft_route], k=60)

        assert len(results) == 3
        # shared_id 在两路都出现，RRF 分数应最高
        assert results[0].chunk_id == shared_id
        expected_score = 1.0 / 61 + 1.0 / 61  # 两路都是 rank 1
        assert abs(results[0].rrf_score - expected_score) < 1e-9
        # 验证来源明细
        assert "vector" in results[0].sources
        assert "fulltext" in results[0].sources

    def test_absent_document_no_penalty(self) -> None:
        """文档只在一路出现时，缺席路不贡献分数也不惩罚。"""
        id_a = uuid.uuid4()
        id_b = uuid.uuid4()

        route_a = self._make_input("vector", [id_a], [0.9])
        route_b = self._make_input("fulltext", [id_b], [0.8])

        results = reciprocal_rank_fusion([route_a, route_b], k=60)

        assert len(results) == 2
        # 两个文档各只出现在一路，rank 都是 1，分数相同
        assert abs(results[0].rrf_score - results[1].rrf_score) < 1e-9

    def test_deterministic_tiebreak(self) -> None:
        """同分时按 chunk_id 升序排列，保证确定性。"""
        id_low = uuid.UUID("00000000-0000-0000-0000-000000000001")
        id_high = uuid.UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")

        # 两路各一个文档，rank 相同 → RRF 分数相同
        route_a = self._make_input("a", [id_high], [0.9])
        route_b = self._make_input("b", [id_low], [0.8])

        results = reciprocal_rank_fusion([route_a, route_b], k=60)

        # 同分时 chunk_id 小的排前面
        assert results[0].chunk_id == id_low
        assert results[1].chunk_id == id_high

    def test_k_value_impact(self) -> None:
        """不同 k 值改变排名差异对分数的影响程度。"""
        ids = [uuid.uuid4() for _ in range(2)]
        route = self._make_input("vector", ids, [0.9, 0.8])

        results_small_k = reciprocal_rank_fusion([route], k=1)
        results_large_k = reciprocal_rank_fusion([route], k=100)

        # k=1: rank1=1/2=0.5, rank2=1/3≈0.333, 差距=0.167
        gap_small = results_small_k[0].rrf_score - results_small_k[1].rrf_score
        # k=100: rank1=1/101≈0.0099, rank2=1/102≈0.0098, 差距≈0.0001
        gap_large = results_large_k[0].rrf_score - results_large_k[1].rrf_score

        # k 越大，排名差异的影响越小
        assert gap_small > gap_large

    def test_multi_route(self) -> None:
        """三路融合正常工作。"""
        shared_id = uuid.uuid4()
        v_route = self._make_input("vector", [shared_id], [0.9])
        ft_route = self._make_input("fulltext", [shared_id], [0.5])
        rerank_route = self._make_input("rerank", [shared_id], [0.7])

        results = reciprocal_rank_fusion([v_route, ft_route, rerank_route], k=60)

        assert len(results) == 1
        expected = 1.0 / 61 + 1.0 / 61 + 1.0 / 61
        assert abs(results[0].rrf_score - expected) < 1e-9
        assert len(results[0].sources) == 3

    def test_source_details_preserved(self) -> None:
        """融合结果保留各路原始排名与分数。"""
        cid = uuid.uuid4()
        v_route = self._make_input("vector", [cid], [0.92])
        ft_route = self._make_input("fulltext", [cid], [0.45])

        results = reciprocal_rank_fusion([v_route, ft_route], k=60)

        assert results[0].sources["vector"]["rank"] == 1
        assert abs(results[0].sources["vector"]["score"] - 0.92) < 1e-9
        assert results[0].sources["fulltext"]["rank"] == 1
        assert abs(results[0].sources["fulltext"]["score"] - 0.45) < 1e-9


# ==================== 文本清洗与分块测试 ====================


class TestTextProcessing:
    """文本清洗与中文递归分块测试。"""

    def test_clean_text_removes_control_chars(self) -> None:
        """控制字符被移除。"""
        result = clean_text("hello\x00world\u200b!")
        assert "\x00" not in result
        assert "\u200b" not in result
        assert "helloworld!" in result

    def test_clean_text_normalizes_whitespace(self) -> None:
        """连续空白压缩为单个空格。"""
        result = clean_text("hello   world\t\tfoo")
        assert result == "hello world foo"

    def test_clean_text_preserves_paragraph_breaks(self) -> None:
        """段落间空行保留一个，多余去除。"""
        result = clean_text("第一段\n\n\n\n第二段")
        assert result == "第一段\n\n第二段"

    def test_clean_text_removes_duplicate_lines(self) -> None:
        """连续重复行去重。"""
        result = clean_text("重复内容\n重复内容\n重复内容\n新内容")
        assert result == "重复内容\n新内容"

    def test_clean_text_empty(self) -> None:
        """空文本返回空串。"""
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_clean_text_fullwidth_space(self) -> None:
        """全角空格转半角。"""
        result = clean_text("hello\u3000world")
        assert result == "hello world"

    def test_split_text_chinese(self) -> None:
        """中文长文本能被切分。"""
        # 生成超过 chunk_size(700) 的中文文本
        text = "这是一段测试文本。" * 100
        chunks = split_text(text)
        assert len(chunks) > 1
        # 每个 chunk 有必要的元数据
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i
            assert "content" in chunk
            assert "chunk_start_index" in chunk
            assert "chunk_end_index" in chunk

    def test_split_text_overlap(self) -> None:
        """相邻切块有重叠内容（overlap > 0）。"""
        text = "句子一。句子二。句子三。句子四。句子五。" * 50
        chunks = split_text(text)
        if len(chunks) >= 2:
            # 验证重叠：后一块的起始位置 < 前一块的结束位置
            assert chunks[1]["chunk_start_index"] < chunks[0]["chunk_end_index"]

    def test_split_text_empty(self) -> None:
        """空文本返回空列表。"""
        assert split_text("") == []
        assert split_text("   \n\n  ") == []

    def test_split_text_short(self) -> None:
        """短文本不分块，返回单块。"""
        chunks = split_text("短文本")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "短文本"


# ==================== tokenize_for_tsv 测试 ====================


class TestTokenizeForTsv:
    """jieba 分词 → tsvector 输入串测试。"""

    def test_chinese_segmentation(self) -> None:
        """中文文本被正确分词。"""
        result = tokenize_for_tsv("布洛芬有什么作用")
        assert "布洛芬" in result
        # 结果应是空格分隔的词串
        words = result.split()
        assert len(words) >= 2

    def test_punctuation_filtered(self) -> None:
        """单字符标点被过滤。"""
        result = tokenize_for_tsv("你好，世界！")
        words = result.split()
        # 逗号和感叹号不应出现在结果中
        assert "，" not in words
        assert "！" not in words

    def test_empty_string(self) -> None:
        """空串返回空串。"""
        assert tokenize_for_tsv("") == ""
        assert tokenize_for_tsv("   ") == ""

    def test_medical_term_preserved(self) -> None:
        """医学术语不被过度切分。"""
        result = tokenize_for_tsv("对乙酰氨基酚片")
        # 精确模式下「对乙酰氨基酚」应保持完整或合理切分
        assert "对乙酰氨基酚" in result or "乙酰氨基酚" in result

    def test_output_is_space_separated(self) -> None:
        """输出格式是空格分隔的词串。"""
        result = tokenize_for_tsv("头痛发热咳嗽")
        if result:
            # 不应包含换行或其他分隔符
            assert "\n" not in result
            assert "\t" not in result
            # 每个 token 非空
            for word in result.split():
                assert word.strip() != ""


# ==================== 降级行为测试 ====================


class TestDegradation:
    """向量与全文检索在非 PG 后端的降级行为。"""

    @pytest.mark.asyncio
    async def test_vector_search_non_pg_returns_empty(self) -> None:
        """非 PG 后端向量检索返回空列表而非抛异常。"""
        from app.services.rag.vector_search import vector_search

        # 模拟非 PG 会话
        mock_session = AsyncMock()
        mock_session.bind.dialect.name = "sqlite"

        result = await vector_search(mock_session, [0.1] * 1536)
        assert result == []

    @pytest.mark.asyncio
    async def test_fulltext_search_non_pg_returns_empty(self) -> None:
        """非 PG 后端全文检索返回空列表而非抛异常。"""
        from app.services.rag.fulltext_search import fulltext_search

        mock_session = AsyncMock()
        mock_session.bind.dialect.name = "sqlite"

        result = await fulltext_search(mock_session, "布洛芬")
        assert result == []

    @pytest.mark.asyncio
    async def test_vector_search_no_pgvector_returns_empty(self) -> None:
        """pgvector 不可用时向量检索返回空列表。"""
        from app.services.rag import vector_search as vs_module

        original = vs_module.PGVECTOR_AVAILABLE
        try:
            vs_module.PGVECTOR_AVAILABLE = False
            mock_session = AsyncMock()
            mock_session.bind.dialect.name = "postgresql"

            result = await vs_module.vector_search(mock_session, [0.1] * 1536)
            assert result == []
        finally:
            vs_module.PGVECTOR_AVAILABLE = original

    @pytest.mark.asyncio
    async def test_fulltext_search_empty_query_returns_empty(self) -> None:
        """空查询返回空列表而非报错。"""
        from app.services.rag.fulltext_search import fulltext_search

        mock_session = AsyncMock()
        mock_session.bind.dialect.name = "postgresql"

        result = await fulltext_search(mock_session, "")
        assert result == []

    @pytest.mark.asyncio
    async def test_vector_search_empty_embedding_returns_empty(self) -> None:
        """空向量返回空列表。"""
        from app.services.rag.vector_search import vector_search

        mock_session = AsyncMock()
        mock_session.bind.dialect.name = "postgresql"

        result = await vector_search(mock_session, [])
        assert result == []
