"""检验科接口的请求/响应模型。

字段口径对齐 ``ExamRequest`` / ``ExamReport`` 表；``structured_data`` 的内部契约
在 ``exam_service.STRUCTURED_DATA_CONTRACT`` 里定义，本模块只保证「是个 JSON 对象」
并把 items 的形态收窄到可解析——**不在 Pydantic 层做数值校验**，理由见
``ReportSubmitRequest.structured_data`` 的说明。

响应侧刻意把 AI 相关字段单独分组（``AIInterpretationOut``）而不是拍平：
前端轮询只需要看 ``ai_status`` 一个字段决定要不要继续轮询，
分组后 ``report.ai`` 整体可以是 null（未解读），语义比八个 null 字段清楚。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ==================== 请求 ====================


class ReportSubmitRequest(BaseModel):
    """检验科录入报告。"""
    model_config = ConfigDict(str_strip_whitespace=True)

    raw_content: str = Field(
        min_length=1,
        max_length=20000,
        description=(
            "原始报告文本（仪器导出原文或人工录入全文）。必填：结构化数据是"
            "对它的加工，不能替代原件——医疗记录必须保留可回溯的原始形态"
        ),
    )
    structured_data: dict[str, Any] | None = Field(
        default=None,
        description=(
            '结构化指标，契约：{"items":[{"name":"血钾","code":"K","value":2.8,'
            '"unit":"mmol/L","ref_low":3.5,"ref_high":5.3}]}。'
            "value/ref_low/ref_high 允许数字或数字字符串；无法解析成数的项"
            "（如定性「尿蛋白 +/-」）会被保留但不参与异常判定"
        ),
    )
    conclusion: str | None = Field(
        default=None, max_length=2000, description="检验科结论（人工填写）"
    )
    auto_interpret: bool = Field(
        default=True,
        description="是否在录入成功后自动派发 AI 解读任务。关闭用于批量补录历史报告",
    )

    @field_validator("structured_data")
    @classmethod
    def _check_shape(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """只校验骨架，不校验数值。

        为什么不在这里做严格校验：LIS 导出的字段名与取值形态五花八门
        （"2.8" / 2.8 / Decimal / "＜0.5"），在协议层用强类型卡住会让
        「格式不完全符合但内容可读」的报告直接 422 拒收，检验科就得手工改数据
        ——那是引入人为错误，不是防错。真正的判定在服务层的容错解析里做
        （``exam_service._to_float``），解析不出数的项跳过判定但保留原值。

        唯一硬拦的是「items 存在却不是列表」这种结构性错误：它一定来自调用方 bug，
        放过去会在服务层静默变成「全部正常、零旗标」，
        而漏报一个危急值的代价不可接受。
        """
        if v is None:
            return v
        if "items" in v and not isinstance(v["items"], list):
            raise ValueError("structured_data.items 必须是数组")
        for idx, item in enumerate(v.get("items") or []):
            if not isinstance(item, dict):
                raise ValueError(f"structured_data.items[{idx}] 必须是对象")
        return v


class InterpretRequest(BaseModel):
    """手动触发/重试 AI 解读。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    reason: str | None = Field(
        default=None,
        max_length=200,
        description="重触发原因（留痕用，写入日志便于事后复盘）",
    )


# ==================== 响应 ====================


class AbnormalItemOut(BaseModel):
    """一条异常指标的明细（由服务端规则算出，非 AI 产出）。"""

    name: str
    code: str | None = None
    value: float
    unit: str | None = None
    ref_low: float | None = None
    ref_high: float | None = None
    direction: str = Field(description="high / low")
    flag: str = Field(description='人读旗标，如 "血钾 2.8 ↓(参考 3.5-5.3)"')


class CriticalValueOut(BaseModel):
    """一条命中的危急值。"""

    name: str
    code: str
    value: float
    unit: str | None = None
    threshold_text: str
    message: str = Field(description="一行可直接展示的告警文本")


class AIInterpretationOut(BaseModel):
    """AI 解读结果块。"""

    ai_status: str = Field(
        description=(
            "pending / processing / completed / failed / skipped。"
            "前端据此决定是否继续轮询：pending 与 processing 都要继续轮，"
            "completed 展示结果，failed/skipped 展示提示条并提供重新解读按钮"
        )
    )
    summary: str | None = None
    suggestions: list[str] = Field(default_factory=list)
    severity: str | None = Field(default=None, description="low / medium / high")
    confidence: float | None = None
    model: str | None = None
    provider: str | None = None
    ai_task_id: str | None = None
    interpreted_at: datetime | None = None
    need_doctor_review: bool = Field(
        default=True,
        description="恒为 true：AI 永不直接下诊断结论，报告须经医生审核才生效",
    )


class ExamReportOut(BaseModel):
    """报告详情（含 AI 解读块）。"""

    id: str
    request_id: str
    patient_id: str
    lab_tech_id: str
    exam_type: str | None = None
    exam_name: str | None = None
    urgency: str | None = None
    request_status: str | None = None
    raw_content: str
    structured_data: dict[str, Any] | None = None
    abnormal_flags: list[str] = Field(default_factory=list)
    abnormal_items: list[AbnormalItemOut] = Field(default_factory=list)
    critical_values: list[CriticalValueOut] = Field(default_factory=list)
    critical: bool = Field(
        description="是否命中危急值。true 时前端必须弹醒目提示，不等 AI 结果"
    )
    critical_message: str | None = Field(
        default=None, description="危急值汇总提示文本（critical=true 时非空）"
    )
    conclusion: str | None = None
    created_at: datetime | None = None
    ai: AIInterpretationOut


class ExamReportListResponse(BaseModel):
    items: list[ExamReportOut]
    total: int
    page: int
    page_size: int


class ExamRequestOut(BaseModel):
    """检查申请单（工作台行 / 详情通用）。"""

    id: str
    visit_id: str
    patient_id: str
    doctor_id: str
    exam_type: str
    exam_name: str
    body_part: str | None = None
    clinical_info: str | None = None
    urgency: str = Field(description="routine / urgent / stat")
    status: str = Field(description="ordered/received/processing/reported/reviewed")
    created_at: datetime | None = None
    report_id: str | None = Field(
        default=None, description="已出报告时带回报告 id，未出为 null"
    )
    ai_status: str | None = Field(
        default=None, description="有报告时回显解读状态，便于工作台一屏看进度"
    )


class ExamQueueResponse(BaseModel):
    """检验科工作台响应。"""

    items: list[ExamRequestOut]
    total: int
    page: int
    page_size: int
    counts: dict[str, int] = Field(
        default_factory=dict,
        description="按紧急度统计的待办数量，供工作台顶部标签显示",
    )


class DispatchOut(BaseModel):
    """AI 解读派发结果。"""

    dispatched: bool
    ai_task_id: str | None = None
    celery_task_id: str | None = None
    reason: str | None = Field(
        default=None,
        description="未派发成功的原因。dispatched=false 时必有值，前端须如实展示",
    )


class ReportSubmitResponse(BaseModel):
    """录入报告的响应（201）。

    **交互模型**：本接口不等 LLM，返回时 ai.ai_status 通常是 pending/processing。
    前端拿到 report.id 后轮询 GET /exams/reports/{id}，直到 ai.ai_status 落到终态。
    危急值提示（critical/critical_message）在录入瞬间就已确定，不依赖 AI。
    """

    report: ExamReportOut
    dispatch: DispatchOut


class InterpretTriggerResponse(BaseModel):
    """手动触发解读的响应（202）。"""

    report_id: str
    dispatch: DispatchOut
    message: str


__all__ = [
    "AbnormalItemOut",
    "AIInterpretationOut",
    "CriticalValueOut",
    "DispatchOut",
    "ExamQueueResponse",
    "ExamReportListResponse",
    "ExamReportOut",
    "ExamRequestOut",
    "InterpretRequest",
    "InterpretTriggerResponse",
    "ReportSubmitRequest",
    "ReportSubmitResponse",
]
