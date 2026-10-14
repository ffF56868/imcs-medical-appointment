"""评测接口的请求/响应模型（阶段 6B）。

口径与前序批次一致：UUID 一律对外为字符串、比率字段是 0-1 的小数
（前端负责百分比格式化）、时间戳带时区。

``EvalRunOut`` 把 ``metrics`` / ``results`` 原样透出而不逐字段建模：
逐题明细的结构随评测类型不同（retrieval 有 expected_rank，ragas 有
四指标得分），为每种运行建一套 schema 会让新增评测类型必须改两层。
整体指标（top1_accuracy 等）是跨类型稳定的，才建模成显式字段。
"""

from __future__ import annotations

import uuid as uuid_lib
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EvalKindLiteral = Literal["retrieval", "quality", "ragas"]


def _stringify(value: Any) -> Any:
    """把 UUID 归一成字符串。

    SQLite 与 PG 驱动返回的 id 类型不同（前者可能给 str、后者给 UUID 对象），
    协议层必须统一成 str，否则同一份响应在不同环境下形态不一致。
    """
    if isinstance(value, uuid_lib.UUID):
        return str(value)
    return value


class EvalCaseCreateRequest(BaseModel):
    """新增一道评测题。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=2, max_length=1000)
    expected_name: str = Field(
        min_length=1, max_length=300,
        description="期望命中的资料标题，必须与知识库 title 逐字一致",
    )
    expected_type: str = Field(
        min_length=1, max_length=30,
        description="取值域 = KnowledgeChunk.doc_type：condition / guideline",
    )
    category: str = Field(default="未分类", max_length=50)
    alternative_names: list[str] = Field(default_factory=list, max_length=20)
    answer_keywords: list[str] = Field(default_factory=list, max_length=20)
    citation_names: list[str] = Field(default_factory=list, max_length=20)
    expected_refusal: bool = Field(
        default=False, description="知识库不覆盖的题设 True：检索为空才算正确"
    )


class EvalCaseUpdateRequest(BaseModel):
    """修改评测题（全字段可选，只更新传入的）。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str | None = Field(default=None, min_length=2, max_length=1000)
    expected_name: str | None = Field(default=None, min_length=1, max_length=300)
    expected_type: str | None = Field(default=None, min_length=1, max_length=30)
    category: str | None = Field(default=None, max_length=50)
    alternative_names: list[str] | None = Field(default=None, max_length=20)
    answer_keywords: list[str] | None = Field(default=None, max_length=20)
    citation_names: list[str] | None = Field(default=None, max_length=20)
    expected_refusal: bool | None = None


class EvalCaseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    question: str
    expected_name: str
    expected_type: str
    category: str
    alternative_names: list[str] = Field(default_factory=list)
    answer_keywords: list[str] = Field(default_factory=list)
    citation_names: list[str] = Field(default_factory=list)
    expected_refusal: bool
    created_at: datetime

    @field_validator("id", mode="before")
    @classmethod
    def _normalize_id(cls, value: Any) -> Any:
        return _stringify(value)


class EvalCaseListResponse(BaseModel):
    items: list[EvalCaseOut]
    total: int
    page: int
    page_size: int


class EvalRunOut(BaseModel):
    """一次评测运行的状态与产出。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: str
    status: str
    total_count: int
    completed_count: int

    # 跨类型稳定的整体指标（不适用的为 None）
    top1_accuracy: float | None = None
    recall_at_3: float | None = None
    precision_at_3: float | None = None
    answer_accuracy: float | None = None
    citation_accuracy: float | None = None
    refusal_accuracy: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    metrics: dict[str, Any] | None = None
    # 逐题明细在列表接口里剔除（体积大），detail 才给全量
    results: list[dict[str, Any]] | None = None

    retrieval_config: dict[str, Any] | None = None
    knowledge_hash: str | None = None
    knowledge_document_count: int = 0
    error_message: str | None = None
    ai_task_id: str | None = None
    triggered_by: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @field_validator("id", "ai_task_id", "triggered_by", mode="before")
    @classmethod
    def _normalize_ids(cls, value: Any) -> Any:
        return _stringify(value)


class EvalRunListResponse(BaseModel):
    items: list[EvalRunOut]
    total: int
    page: int
    page_size: int


class RagasDispatchRequest(BaseModel):
    """触发一次 RAGAS 异步评测。"""

    sample_size: int = Field(
        default=10, ge=1, le=100,
        description="抽样题数。RAGAS 要逐题生成回答并打分，成本与题数线性相关",
    )


class RagasDispatchResponse(BaseModel):
    eval_run_id: str
    ai_task_id: str | None = None
    status: str
    sample_size: int
    message: str


class EvalTrendResponse(BaseModel):
    """回归趋势：同 kind 的历史运行序列，供门禁曲线展示。"""

    kind: str
    points: list[dict[str, Any]]
    baseline: dict[str, Any] | None = None


__all__ = [
    "EvalCaseCreateRequest",
    "EvalCaseListResponse",
    "EvalCaseOut",
    "EvalCaseUpdateRequest",
    "EvalKindLiteral",
    "EvalRunListResponse",
    "EvalRunOut",
    "EvalTrendResponse",
    "RagasDispatchRequest",
    "RagasDispatchResponse",
]
