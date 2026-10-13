"""接诊、病历与检查开单相关的 Pydantic v2 协议模型。

覆盖阶段 4 医生侧的四个动作：开始接诊（写请求为空，身份全在 URL 与 token 里）、
书写/修改病历、开检查申请单、结束就诊；以及医生工作台与患者视图的响应契约。

约定与前序批次（``schemas/appointment.py``、``schemas/knowledge.py``）保持一致：

1. **响应模型显式列字段**，不用 ``from_attributes`` 全量导出 ORM。
   ``Visit.doctor`` 指向 ``User``，而 User 带 ``password_hash``；
   病历详情接口一旦全量展开关系，就等于把口令哈希挂到医生工作站页面上。
2. **金额保持整数「分」**：``fee_amount`` 与 ``registration_fee`` 同源，
   schema 层不做除以 100 的换算，精度决策留在模型层，格式化交给前端展示。
3. **UUID 对外一律字符串**：SQLite 测试环境与 PG 驱动返回类型不同，
   在边界统一成 str 让响应契约与后端方言无关。
4. **枚举取值复用模型层常量**（见文件末尾的 import 期断言）：
   urgency / exam_type 的合法集合同时是 service 的写入依据和前端的下拉数据源，
   两处各写一遍必然漂移，而漂移的后果是非法值静默落库。

**为什么请求体里没有 doctor_id / patient_id**：接诊医生的身份来自 JWT，
患者身份由预约单反推（``appointments.patient_id`` 指向 patient_profiles.id）。
这两个字段一旦可由客户端指定，就等于任何人都能替别的医生写病历、
或把病历挂到别人患者身上——归属校验会失去意义。
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.clinical import ExamRequestStatus, Urgency, VisitStatus

# ==================== 取值常量 ====================

# 紧急度三档，唯一真相来源是 clinical.Urgency 枚举。
URGENCY_VALUES: tuple[str, ...] = tuple(item.value for item in Urgency)

# visit 状态四态（含终态 closed），唯一真相来源是 clinical.VisitStatus。
VISIT_STATUS_VALUES: tuple[str, ...] = tuple(item.value for item in VisitStatus)

# 本批医生侧只写入 ordered；后续状态由检验科侧推进，不作为请求入参。
EXAM_INITIAL_STATUS: str = ExamRequestStatus.ORDERED.value

# icd_codes 条数上限。多诊断确实存在（合并症），但几十条只会是把自由文本
# 塞进编码字段的误用；限死上限能让「编码数组」保持可枚举、可统计。
MAX_ICD_CODES = 20


def _as_uuid(value: object, field_name: str) -> uuid.UUID:
    """把字符串解析成 UUID，失败则抛 ValueError（协议层 → 422）。

    放在 schema 层而不是 service 层：ID 形态非法属于请求本身有问题，
    应该在进入业务逻辑之前拒掉，不给 service 制造「拿到 None 还是脏值」
    的分支负担。**注意与路径参数的口径差异**：``/consult/visits/{visit_id}``
    里的非法 UUID 走路由层的 ``_parse_uuid`` 报 404（语义是定位资源），
    而这里出现在请求体里的 UUID 语义是「你请求写错了」，因此是 422。
    """
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field_name} 必须是合法的 UUID 字符串") from exc


def _normalize_icd_codes(value: list[str] | None) -> list[str] | None:
    """ICD 编码数组去空白、去重、保序。

    去重必须在服务端做：同一编码重复两次会让「按 ICD 统计诊断分布」的
    报表直接翻倍，而这类统计错误不会报错，只会在结论里长期存在。
    保序（而非排序）是因为医生书写的顺序就是主次诊断顺序，
    第一条通常被视为主要诊断，这个信息不能丢。
    """
    if value is None:
        return None
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in value:
        code = raw.strip().upper()
        if not code or code in seen:
            continue
        seen.add(code)
        cleaned.append(code)
    return cleaned


# ==================== 请求模型 ====================


class MedicalRecordSaveRequest(BaseModel):
    """书写/修改病历。

    全部字段可选的原因：**病历是渐进成文的**。医生先记现病史、查完体再补
    体格检查、最后定诊断，一次 HTTP 请求写完五项是不现实的假设。
    service 采用「传了才覆盖」的部分更新语义（不传即保持原值），
    因此这里不能用 required 字段逼医生把没内容的栏目编造出来。

    刻意**不包含 chief_complaint**：主诉由挂号时患者填写并带入病历
    （``start_consultation`` 从 appointment 复制），医生改写主诉会让
    「患者自己说的」和「医生记录的」两种口径混在一个字段里，
    失去可追溯性。要修正主诉应另开修订流程。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    present_illness: str | None = Field(
        default=None, max_length=5000, description="现病史：起病情况、发展演变、诊疗经过"
    )
    past_history: str | None = Field(
        default=None, max_length=3000, description="既往史：慢性病、手术、过敏"
    )
    physical_exam: str | None = Field(
        default=None, max_length=3000, description="体格检查所见"
    )
    diagnosis: str | None = Field(
        default=None, max_length=1000, description="诊断结论文本"
    )
    icd_codes: list[str] | None = Field(
        default=None, max_length=MAX_ICD_CODES, description="ICD-10 编码数组，支持多诊断"
    )
    treatment_advice: str | None = Field(
        default=None, max_length=5000, description="治疗建议 / 医嘱"
    )

    @field_validator("icd_codes")
    @classmethod
    def _validate_icd_codes(cls, value: list[str] | None) -> list[str] | None:
        return _normalize_icd_codes(value)


class ExamOrderRequest(BaseModel):
    """开检查申请单。

    ``exam_name`` 允许自定义而不强制取自价目表：临床项目远多于演示价目里的
    八项，遇到表外项目时费用给 0 并留待阶段 5 收费侧核价，
    比在这里就把医生挡住更符合实际工作流。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    exam_type: str = Field(
        min_length=1,
        max_length=50,
        description="检查项目类型代码，如 blood_routine / ecg / ultrasound_abdomen",
    )
    exam_name: str = Field(min_length=1, max_length=100, description="检查项目名称（展示用）")
    body_part: str | None = Field(default=None, max_length=100, description="检查部位，如 胸部")
    clinical_info: str | None = Field(
        default=None, max_length=2000, description="供检验科参考的临床信息"
    )
    urgency: str = Field(
        default=Urgency.ROUTINE.value,
        description=f"紧急度，取值 {'/'.join(URGENCY_VALUES)}；stat 会额外记录危急日志",
    )

    @field_validator("urgency")
    @classmethod
    def _validate_urgency(cls, value: str) -> str:
        allowed = set(URGENCY_VALUES)
        if value not in allowed:
            raise ValueError(f"urgency 非法，允许值：{'、'.join(sorted(allowed))}")
        return value


# ==================== 响应模型 ====================


class VisitOut(BaseModel):
    """就诊/病历详情。

    同时服务医生端与患者端：两个角色读的是同一条记录，字段口径必须一致，
    拆成 VisitDoctorOut / VisitPatientOut 会出现「患者看不到医生看得到」
    这种无法解释的差异，而且两套字段日后必然漂移。行级隔离在 service 层
    决定「能不能看到这条」，而不是靠少返几个字段来实现。
    """

    id: str
    appointment_id: str
    patient_id: str = Field(description="患者档案 UUID（patient_profiles.id），非账号 id")
    doctor_id: str
    doctor_name: str
    chief_complaint: str
    present_illness: str | None = None
    past_history: str | None = None
    physical_exam: str | None = None
    diagnosis: str | None = None
    icd_codes: list[str] = Field(default_factory=list)
    treatment_advice: str | None = None
    status: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    exam_request_count: int = Field(
        default=0, description="本次就诊已开的检查单数量，便于医生判断是否还需补检"
    )
    allowed_transitions: list[str] = Field(
        default_factory=list, description="visit 当前状态可迁移到的目标状态"
    )


class VisitListResponse(BaseModel):
    """就诊分页响应。page/page_size 回显实际生效值。"""

    items: list[VisitOut]
    total: int
    page: int
    page_size: int


class PendingAppointmentOut(BaseModel):
    """医生工作台「待接诊」列表里的一行。

    返回预约而不是 visit：此刻 visit 可能还不存在（医生还没点开始接诊），
    而工作台要展示的正是「谁在等我」。``visit_id`` 在有活跃病历时回填，
    前端据此决定按钮文案是「开始接诊」还是「继续书写」。
    """

    appointment_id: str
    patient_id: str
    time_slot: str
    appointment_date: date
    status: str
    chief_complaint: str | None = None
    triage_summary: str | None = None
    checked_in_at: datetime | None = None
    visit_id: str | None = None
    visit_status: str | None = None


class PendingAppointmentListResponse(BaseModel):
    """医生今日待诊队列。"""

    doctor_id: str
    date: date
    items: list[PendingAppointmentOut]
    total: int


class ExamRequestOut(BaseModel):
    """检查申请单。

    ``is_charged`` / ``fee_amount`` 在本批只是默认值（未收费 + 演示价目），
    照实返回而不是隐藏：医生开完单需要立刻知道这张单子会产多少钱的检查费，
    这是问诊谈话里要交代的内容，不是收费侧的内部状态。
    """

    id: str
    visit_id: str
    patient_id: str
    doctor_id: str
    exam_type: str
    exam_name: str
    body_part: str | None = None
    clinical_info: str | None = None
    urgency: str
    status: str
    is_charged: bool
    fee_amount: int = Field(description="检查费，单位分；取自模块内演示价目常量")
    created_at: datetime | None = None
    allowed_transitions: list[str] = Field(
        default_factory=list, description="exam_request 当前状态可迁移到的目标状态"
    )


class ConsultStartResponse(BaseModel):
    """开始接诊的响应：病历 + 该患者的历史病历摘要。

    带上历史病历不是为了省一次请求，而是医疗必需——医生首次打开接诊面板时
    必须能看到「这个人以前因什么看过我」，既往诊断直接影响当下的用药与检查决策。
    若前端自己去调列表接口，它得先猜过滤条件，更容易出错。
    """

    visit: VisitOut
    appointment_status: str
    prior_visits: list[VisitOut] = Field(
        default_factory=list, description="同一患者此前已结束的病历（最多 5 条，倒序）"
    )


class MedicalRecordSaveResponse(BaseModel):
    """病历保存后的回执。

    ``transitioned`` 明确告知这次写入是否触发了 in_progress→documented。
    前端据此提示「病历已定稿」还是「仍在书写中」，不必自己比对状态字符串
    ——状态由服务端算出，让客户端重新推断等于把规则复制两份。
    """

    visit: VisitOut
    transitioned: bool = Field(description="本次写入是否发生了状态流转")


class CloseVisitResponse(BaseModel):
    """结束就诊的回执。

    同时返回 visit 与 appointment 两个状态：这是一次跨实体事务，
    客户端需要确认「病历结案」和「预约完成」都真的落地了，
    只返其中一个会让另一半变成不可验证的黑箱。
    """

    visit: VisitOut
    appointment_id: str
    appointment_status: str


# ==================== 取值一致性守卫 ====================


def _assert_literals_match_constants() -> None:
    """import 期比对 schema 侧常量与模型枚举，任一侧漂移立即抛错。

    注释会过时，断言不会。历史上 knowledge 侧就靠同类断言拦住过一次
    doc_type 新增取值忘了同步 Literal 的事故。
    """
    assert set(URGENCY_VALUES) == {item.value for item in Urgency}
    assert set(VISIT_STATUS_VALUES) == {item.value for item in VisitStatus}
    assert EXAM_INITIAL_STATUS == ExamRequestStatus.ORDERED.value


_assert_literals_match_constants()


__all__ = [
    "EXAM_INITIAL_STATUS",
    "MAX_ICD_CODES",
    "URGENCY_VALUES",
    "VISIT_STATUS_VALUES",
    "CloseVisitResponse",
    "ConsultStartResponse",
    "ExamOrderRequest",
    "ExamRequestOut",
    "MedicalRecordSaveRequest",
    "MedicalRecordSaveResponse",
    "PendingAppointmentListResponse",
    "PendingAppointmentOut",
    "VisitListResponse",
    "VisitOut",
]
