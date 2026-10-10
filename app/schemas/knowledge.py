"""知识库相关的 Pydantic v2 协议模型。

与 ``schemas/auth.py`` 同一原则：**枚举取值复用 service 层常量**，
不在 schema 里另写一份字面量列表。doc_type / source_tier 的合法值同时被
``knowledge_service`` 的写入逻辑和向量化任务引用，两处各写一遍必然漂移；
一旦某侧新增取值而另一侧没跟上，新类型的文档会在创建时被 422 拒掉，
或者落库后向量化不认——后者更糟，因为不报错。

响应模型一律显式列字段而非 ``from_attributes`` 全量导出：
``KnowledgeDocument.content`` 是完整原文，列表接口把它带上会让响应体
随文档数量线性膨胀（一篇指南几万字），前端列表页也根本不消费它。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.knowledge_service import DOC_TYPES, SOURCE_TIERS

# **为什么手写字面量而不是 ``Literal[DOC_TYPES]``**：Python 的 typing 会把
# ``Literal[某个 tuple]`` 展开成「一个字面量元素」而不是逐项展开，
# ``Literal[SOURCE_TIERS]`` 实际等价于只接受那一个 tuple 值（还因不可哈希而报错）。
# 因此这里显式列出取值。同步保障有两层：
# 1. 文件末尾的 ``_assert_literals_match_constants`` 在 import 时比对集合，
#    任一侧漂移立即抛错——这类断言比注释可靠，注释会过时，断言不会。
# 2. tests/test_imcs_knowledge.py 有用例直接构造非法取值断言被 422 拒掉。
DocTypeLiteral = Literal[
    "guideline",
    "drug_manual",
    "lab_reference",
    "condition",
    "drug",
    "textbook",
]
SourceTierLiteral = Literal["authoritative", "professional", "general", "unverified"]


# ==================== 请求 ====================


class DocumentCreateRequest(BaseModel):
    """新建知识文档。

    只接受纯文本 content：PDF / docx 的解析在后续批次，届时由上传接口
    先转成字符串再走本 schema，向量化链路本身不变。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=300, description="文档标题")
    content: str = Field(min_length=1, description="正文纯文本（Markdown 或纯文本）")
    doc_type: DocTypeLiteral = Field(description="资料类型")
    source: str = Field(
        default="未标注来源",
        max_length=200,
        description="来源机构/出版物，检索引用会展示它",
    )
    source_url: str | None = Field(default=None, max_length=2000, description="来源链接")
    source_tier: SourceTierLiteral = Field(
        default="unverified",
        description="来源可信分级，影响检索结果的权重展示",
    )
    department_id: str | None = Field(default=None, description="关联科室 UUID，可选")
    extra_metadata: dict[str, Any] | None = Field(default=None, description="扩展信息")

    @field_validator("department_id")
    @classmethod
    def _validate_department_id(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        import uuid as _uuid

        try:
            _uuid.UUID(value)
        except (ValueError, TypeError) as exc:
            raise ValueError("department_id 必须是合法的 UUID 字符串") from exc
        return value


class DocumentUpdateRequest(BaseModel):
    """更新文档，所有字段可选（PATCH 语义）。

    不提供 id / index_status / chunk_count：索引状态是向量化任务的产出，
    让客户端直接改等于允许伪造「已索引」，检索会返回根本不存在的切块。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str | None = Field(default=None, min_length=1, max_length=300)
    content: str | None = Field(default=None, min_length=1)
    doc_type: DocTypeLiteral | None = None
    source: str | None = Field(default=None, max_length=200)
    source_url: str | None = Field(default=None, max_length=2000)
    source_tier: SourceTierLiteral | None = None
    is_reviewed: bool | None = None
    is_active: bool | None = None


class SearchRequest(BaseModel):
    """混合检索请求。

    ``top_k`` 不开放给客户端：结果条数由 ``settings.rag_result_count`` 统一控制，
    放开后调用方可以用一个超大 top_k 把整库切块拖出来，既绕过 min_score
    又把 LIMIT 变成摆设。要调这个参数应该改配置并观察检索质量。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000, description="检索问题")
    rewritten_query: str | None = Field(
        default=None,
        max_length=2000,
        description="改写后的检索问题（多轮对话指代消解后），可选",
    )
    trace_id: str | None = Field(default=None, max_length=36, description="链路追踪 ID")


# ==================== 响应 ====================


class DocumentSummaryOut(BaseModel):
    """列表用摘要，不含正文。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    doc_type: str
    source: str
    source_tier: str
    index_status: str
    chunk_count: int
    is_reviewed: bool
    is_active: bool
    content_length: int = Field(description="正文字符数，用于判断是否需要重建而不必拉全文")
    created_at: datetime | None = None
    indexed_at: datetime | None = None


class DocumentOut(DocumentSummaryOut):
    """详情响应：摘要 + 正文与溯源字段。"""

    content: str
    content_hash: str
    source_url: str | None = None
    department_id: str | None = None
    published_at: datetime | None = None
    extra_metadata: dict[str, Any] | None = None
    updated_at: datetime | None = None


class DocumentListResponse(BaseModel):
    """分页列表响应。

    page/page_size 回显而非让前端自己记：翻页控件据此计算当前页高亮，
    且当服务端把越界的 page_size 夹到 100 时，回显值能让前端知道实际生效值。
    """

    items: list[DocumentSummaryOut]
    total: int
    page: int
    page_size: int


class TaskOut(BaseModel):
    """异步任务状态。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    task_type: str
    status: str
    progress: int
    progress_message: str | None = None
    retry_count: int
    error_message: str | None = None
    result: dict[str, Any] | None = None
    celery_task_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None


class RebuildAcceptedOut(BaseModel):
    """派发重建任务的响应。

    用 202 而非 200：请求被接受但工作尚未开始，真正的完成要靠
    GET /knowledge/tasks/{id} 轮询。返回体里直接带 task_id，
    省掉客户端「派发后再查一次列表找最新任务」的往返。
    """

    task_id: str
    task_type: str
    status: str
    celery_task_id: str | None = None
    message: str = "任务已排队，可通过 /api/v1/knowledge/tasks/{task_id} 查询进度"


class SearchResultItem(BaseModel):
    """单条检索命中。"""

    chunk_id: str
    content: str
    rrf_score: float
    title: str
    doc_type: str
    source: str
    source_tier: str
    page_number: int | None = None
    page_range: str | None = None
    extra_metadata: dict[str, Any] | None = None
    route_details: dict[str, Any] = Field(
        default_factory=dict, description="各路召回的 rank 与原始分，用于调试归因"
    )


class RetrievalDiagnostics(BaseModel):
    """召回诊断计数。

    单独成组而不是拍平进响应根：这三个数字是给工程看的（判断哪路召回起作用、
    是否该重新审视分词或模型），前端业务代码只消费 results。分组后
    「只要结果」的调用方可以整体忽略这一块。
    """

    vector_hit_count: int
    keyword_hit_count: int
    both_hit_count: int
    merged_count: int
    is_zero_result: bool


class SearchResponse(BaseModel):
    """混合检索响应。"""

    query: str
    results: list[SearchResultItem]
    diagnostics: RetrievalDiagnostics


# ==================== 取值一致性自检 ====================


def _assert_literals_match_constants() -> None:
    """协议层字面量与 service 常量必须完全一致，不一致就在 import 期炸掉。

    **为什么需要这个断言**：doc_type / source_tier 的合法值在本文件以 Literal
    显式列出（Python 不会把 ``Literal[某个 tuple]`` 逐项展开，只能手写），
    而 knowledge_service 里另有一份 tuple 常量供 worker 侧使用。两份声明一旦漂移，
    后果是静默的：新类型在 API 侧被 422 拒掉，或绕过校验落库后向量化不认。
    放在 import 期而非测试期，是为了让「启动即失败」替代「线上静默放行」。
    """
    literal_doc_types = set(get_args(DocTypeLiteral))
    if literal_doc_types != set(DOC_TYPES):
        raise RuntimeError(
            "schemas.knowledge 的 doc_type 取值与 knowledge_service.DOC_TYPES 不一致："
            f"{sorted(literal_doc_types ^ set(DOC_TYPES))}"
        )

    literal_tiers = set(get_args(SourceTierLiteral))
    if literal_tiers != set(SOURCE_TIERS):
        raise RuntimeError(
            "schemas.knowledge 的 source_tier 取值与 knowledge_service.SOURCE_TIERS 不一致："
            f"{sorted(literal_tiers ^ set(SOURCE_TIERS))}"
        )


_assert_literals_match_constants()


__all__ = [
    "DOC_TYPES",
    "SOURCE_TIERS",
    "DocumentCreateRequest",
    "DocumentListResponse",
    "DocumentOut",
    "DocumentSummaryOut",
    "DocumentUpdateRequest",
    "RebuildAcceptedOut",
    "RetrievalDiagnostics",
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
    "TaskOut",
]
