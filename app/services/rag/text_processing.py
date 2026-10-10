"""文本清洗、中文递归分块与 jieba 分词。

纯函数模块，不碰数据库、不调 LLM。所有参数从 ``settings`` 读取，
避免硬编码导致调参与配置不一致。

**为什么应用层用 jieba 分词而不是 PG 扩展**：
pg_jieba / zhparser 需要定制 PostgreSQL 镜像并在容器内编译，
Windows/ARM 下构建慢且易失败；应用层分词还能挂医学自定义词典，
保证「对乙酰氨基酚」这类药名不被切碎。写入侧用 jieba 精确模式切词后
以空格连接存入 tsvector，查询侧必须用同一个 ``tokenize_for_tsv`` 分词，
再用 ``simple`` 配置检索——否则 PG 会对已分好词的中文再切一次，
导致静默零命中。这是 RAG 最常见的隐性故障。
"""

from __future__ import annotations

import re
import unicodedata

import jieba
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


# ==================== 文本清洗 ====================

# 需要移除的控制字符（不含换行与制表符，它们在分块时作为段落边界保留）
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufeff\u200b]"
)

# 连续空白压缩为单个空格（换行单独处理，保留段落结构）
_MULTI_SPACE_RE = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    """清洗文本：去控制字符、统一换行、压缩空白、去首尾空白。

    参考旧 ``app/text_processing.py`` 的清洗策略但独立实现，
    不 import legacy 模块。保留了核心清洗规则（BOM、零宽空格、
    全角空格、CRLF 统一），去掉了与旧数据模型耦合的统计字段。
    """
    if not text:
        return ""

    # 统一换行符
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # 全角空格 → 半角
    normalized = normalized.replace("\u3000", " ").replace("\u00a0", " ")
    # 去控制字符
    normalized = _CONTROL_CHAR_RE.sub("", normalized)

    lines: list[str] = []
    prev_line: str | None = None
    blank_kept = False

    for raw_line in normalized.split("\n"):
        line = _MULTI_SPACE_RE.sub(" ", raw_line).strip()
        if not line:
            # 保留一个空行作为段落分隔，多余空行丢弃
            if lines and not blank_kept:
                lines.append("")
                blank_kept = True
            continue
        # 去重连续相同行（复制粘贴产生的重复段落）
        if line == prev_line:
            continue
        lines.append(line)
        prev_line = line
        blank_kept = False

    return "\n".join(lines).strip()


# ==================== 中文递归分块 ====================


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """构造中文友好的递归分块器。

    分隔符优先级：段落 > 换行 > 句号 > 感叹号 > 问号 > 分号 > 逗号 > 字符级。
    这样每个 chunk 尽量在语义完整处断开，而不是机械地按字数截断。
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""],
        add_start_index=True,
    )


def split_text(text: str) -> list[dict[str, object]]:
    """将清洗后的文本切成带元数据的块列表。

    返回格式::

        [
            {
                "content": "切块文本",
                "chunk_index": 0,
                "chunk_start_index": 0,
                "chunk_end_index": 680,
            },
            ...
        ]

    空文本或清洗后为空时返回空列表。
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    splitter = _build_splitter()
    # RecursiveCharacterTextSplitter 接受纯字符串，返回带 start_index 的 Document
    from langchain_core.documents import Document

    doc = Document(page_content=cleaned)
    chunks = splitter.split_documents([doc])

    result: list[dict[str, object]] = []
    for idx, chunk in enumerate(chunks):
        start = int(chunk.metadata.get("start_index", 0))
        result.append(
            {
                "content": chunk.page_content,
                "chunk_index": idx,
                "chunk_start_index": start,
                "chunk_end_index": start + len(chunk.page_content),
            }
        )
    return result


# ==================== jieba 分词 → tsvector 输入 ====================

# 过滤掉的单字符：标点、空白、纯数字单字等无检索价值的 token
_SINGLE_CHAR_FILTER_RE = re.compile(
    r"^[\s\d\W]$"
)


def tokenize_for_tsv(text: str) -> str:
    """jieba 精确模式分词 → 空格连接的 tsvector 输入串。

    **对称性要求**：写入侧（KnowledgeChunk.tsv）与查询侧（fulltext_search）
    必须使用同一个函数分词。写入时用 ``to_tsvector('simple', tokenize_for_tsv(text))``
    存储，查询时用 ``to_tsquery('simple', tokenize_for_tsv(query))`` 检索。
    若任一侧改用其他分词方式或 PG 配置（如 ``chinese`` / ``english``），
    会导致 token 对齐失败，**静默返回零结果**且不报错。

    选择精确模式而非搜索引擎模式：精确模式不会把「对乙酰氨基酚」拆成
    「对乙酰」+「氨基酚」，医学名词完整性更好。搜索引擎模式虽然召回更高，
    但在 tsvector + simple 配置下已经通过 RRF 融合补偿了召回率，
    不需要靠过度切词来提升。

    Args:
        text: 待分词的原始文本。

    Returns:
        空格分隔的词串，可直接传入 ``to_tsvector('simple', ...)`` 或
        拼成 ``to_tsquery('simple', ...)`` 表达式。空文本返回空串。
    """
    if not text or not text.strip():
        return ""

    words = jieba.lcut(text)
    filtered: list[str] = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        # 过滤纯空白、单字符标点、纯数字单字
        if len(w) == 1 and _SINGLE_CHAR_FILTER_RE.match(w):
            continue
        filtered.append(w)

    return " ".join(filtered)
