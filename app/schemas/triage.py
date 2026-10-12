"""分诊接口的请求/响应模型。

字段口径对齐 TriageResult 表：响应把落库字段尽量回显，
前端不需要再查一次详情就能渲染完整分诊卡。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TriageRequest(BaseModel):
    """分诊评估请求。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    chief_complaint: str = Field(
        min_length=2,
        max_length=500,
        description="患者主诉原文，口语描述即可（分诊会先做急症拦截再做结构化提取）",
    )
    include_slots: bool = Field(
        default=True,
        description="是否在响应里附带推荐科室的可挂号源摘要（next_actions）。"
        "关闭可省一次号源查询，用于只要分诊结论的批量场景",
    )


class EvidenceOut(BaseModel):
    """一条知识库证据（推荐依据展示）。"""

    title: str
    source: str
    source_tier: str = Field(description="来源可信度分级：authoritative/textbook/popular 等")
    snippet: str
    rrf_score: float


class SuggestedSlot(BaseModel):
    """分诊后给出的可挂号源摘要（next_actions 的组成部分）。

    只带患者做决策需要的最小字段；完整号源列表走 GET /schedules。
    """

    schedule_id: str
    doctor_name: str
    doctor_title: str | None = None
    schedule_date: str
    time_slot: str
    remaining_slots: int
    registration_fee: int = Field(description="单位分")


class TriageResponse(BaseModel):
    """同步分诊响应。"""

    triage_id: str
    urgency: str = Field(description="routine / urgent / stat")
    is_emergency: bool
    matched_emergency_keywords: list[str] = Field(default_factory=list)
    recommended_department_id: str | None = None
    recommended_department_name: str | None = None
    recommended_department_code: str | None = None
    confidence: float | None = None
    reason: str
    advice: str | None = None
    decision_source: str = Field(
        description="ai / rule / hybrid / emergency——结论由哪条路径给出，"
        "用于评估系统对 LLM 的依赖度（models/ai.py 注释原话）"
    )
    degraded: bool = Field(description="LLM 链路是否处于降级模式")
    extracted_symptoms: dict[str, Any] | None = None
    evidence: list[EvidenceOut] = Field(default_factory=list)
    llm_model: str | None = None
    trace_id: str | None = None
    suggested_slots: list[SuggestedSlot] = Field(default_factory=list)


class TriageListResponse(BaseModel):
    items: list[TriageResponse]
    total: int
    page: int
    page_size: int


class BatchAssessItem(BaseModel):
    """批量分诊的单条输入（管理员/医生做批量评估场景）。"""

    chief_complaint: str = Field(min_length=2, max_length=500)
    patient_id: str | None = Field(default=None, description="可选：关联到某患者档案")


class BatchAssessRequest(BaseModel):
    items: list[BatchAssessItem] = Field(min_length=1, max_length=50)


class BatchDispatchResponse(BaseModel):
    ai_task_id: str
    queued: int
    message: str


class TriageOut(BaseModel):
    """历史列表用的精简输出（完整字段在详情端点）。"""

    id: str
    created_at: datetime
    chief_complaint: str
    urgency: str
    is_emergency: bool
    recommended_department_name: str | None = None
    decision_source: str
    confidence: float | None = None
    trace_id: str | None = None


__all__ = [
    "TriageRequest",
    "TriageResponse",
    "TriageListResponse",
    "TriageOut",
    "EvidenceOut",
    "SuggestedSlot",
    "BatchAssessItem",
    "BatchAssessRequest",
    "BatchDispatchResponse",
]
