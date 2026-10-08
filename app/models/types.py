"""跨数据库可移植的列类型。

生产用 PostgreSQL，单元测试用 SQLite（无需起数据库容器即可验证业务逻辑）。
PostgreSQL 专有的 ``ARRAY`` / ``JSONB`` 在 SQLite 上不存在，因此这里用
``with_variant`` 声明：PostgreSQL 走原生类型（享受 GIN 索引与数组运算符），
其他后端退化为 JSON 文本并在 Python 侧序列化。

这样模型定义只有一份，测试与生产共用。
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import JSON, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.types import TypeDecorator


class JSONBText(TypeDecorator):
    """PostgreSQL 用 JSONB，其他后端用 TEXT + JSON 序列化。

    JSONB 相比 JSON 的优势：二进制存储、可建 GIN 索引、支持 ``@>`` 包含查询。
    结构化检查报告数据与收费明细都要按内部字段过滤，必须用 JSONB。
    """

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: Any, dialect) -> Any:
        if value is None or dialect.name == "postgresql":
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    def process_result_value(self, value: Any, dialect) -> Any:
        if value is None or dialect.name == "postgresql":
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (ValueError, TypeError):
                return None
        return value


class TextArray(TypeDecorator):
    """PostgreSQL 用 TEXT[]，其他后端用 JSON 数组。

    过敏史、慢性病、异常项标记这类字段用数组比逗号分隔字符串更安全：
    不会出现值里含分隔符导致的解析歧义。
    """

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(ARRAY(Text()))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: Any, dialect) -> Any:
        if value is None or dialect.name == "postgresql":
            return value
        return json.dumps(list(value), ensure_ascii=False)

    def process_result_value(self, value: Any, dialect) -> Any:
        if value is None or dialect.name == "postgresql":
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                return []
            return list(parsed) if isinstance(parsed, list) else []
        return value


class TSVectorText(TypeDecorator):
    """PostgreSQL 用 ``tsvector``，其他后端退化为 ``TEXT``。

    分词在应用层用 jieba 完成，写入时存成空格分隔的词串，
    再用 ``to_tsvector('simple', ...)`` 建索引——用 ``simple`` 配置是为了
    避免 PostgreSQL 对已经分好词的中文再切一次。

    选应用层分词而不是 pg_jieba / zhparser 扩展：装扩展要定制 PostgreSQL
    镜像并在容器内编译，Windows/ARM 下构建慢且易失败；应用层分词还能挂
    医学自定义词典，保证「对乙酰氨基酚」这类药名不被切碎。

    SQLite 无 tsvector 类型，退化为 TEXT 后仅存分词结果、不参与检索，
    单元测试通过应用层余弦相似度兜底验证业务逻辑。
    """

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import TSVECTOR

            return dialect.type_descriptor(TSVECTOR())
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: Any, dialect) -> Any:
        # 应用层始终传「已分好词的空格分隔字符串」，两个后端都能直接存
        return value

    def process_result_value(self, value: Any, dialect) -> Any:
        return value


__all__ = ["JSONBText", "TextArray", "TSVectorText", "JSON"]
