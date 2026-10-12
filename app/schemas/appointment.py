"""号源（排班）与预约相关的 Pydantic v2 协议模型。

两组 schema 放同一个模块的原因：它们描述的是同一条业务链的两端——
患者先查号源、再对着号源下单。字段口径（金额单位「分」、UUID 一律字符串、
日期用 ``date``）必须完全一致，拆成两个文件最容易出现的事故是
「改了排班的输出格式忘了改预约的」。

约定与 ``schemas/knowledge.py`` 保持一致：

1. **响应模型显式列字段**，不用 ``from_attributes`` 全量导出 ORM。
   ``DoctorSchedule.doctor`` 是指向 ``User`` 的关系，而 User 带
   ``password_hash``；全量导出等于把口令哈希挂到公开号源列表上。
2. **金额保持整数「分」**：``registration_fee`` 在模型层的注释就写明了
   「避免浮点误差」，schema 若在这里除以 100 变成 float，就把精度决策
   推给了二进制浮点。格式化（¥35.00）留给前端展示层。
3. **UUID 对外一律字符串**：SQLite 测试环境与 PG 的驱动返回类型不同，
   在边界统一成 str 可以让响应契约与后端方言无关。
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ==================== 取值常量 ====================

# 时段单元，与 seed/schedules.py 的 SCHEDULE_TEMPLATES.slot_period 一致。
SlotPeriodLiteral = Literal["am", "pm"]

# 排班状态，与 DoctorSchedule.status 注释（open / full / closed）一致。
ScheduleStatusLiteral = Literal["open", "full", "closed"]

# 预约来源。models/clinical.py 注释只定义了这两种取值：
# self = 患者自主挂号，triage_referral = 智能分诊引导挂号。
AppointmentSourceLiteral = Literal["self", "triage_referral"]

# 默认号源查询窗口天数，与种子数据生成的 14 天排班对齐。
DEFAULT_SCHEDULE_WINDOW_DAYS = 14


def _as_uuid(value: object, field_name: str) -> uuid.UUID:
    """把字符串解析成 UUID，失败则抛 ValueError（协议层 → 422）。

    放在 schema 层而不是 service 层：ID 形态非法属于请求本身有问题，
    应该在进入业务逻辑之前就被拒掉，不给 service 制造「拿到 None 还是脏值」
    的分支负担。
    """
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field_name} 必须是合法的 UUID 字符串") from exc


# ==================== 号源查询 ====================


class ScheduleQuery(BaseModel):
    """号源查询条件。

    作为 service 侧的参数载体：路由把若干 Query 参数装配成它再传下去，
    这样 service 函数签名不会随着筛选条件增加而无限膨胀，
    也便于后续「保存常用筛选条件」这类需求直接复用同一结构。

    日期区间默认「东八区今天起 14 天」：种子排班正好覆盖未来 14 个工作日，
    不传参数时给患者的就该是「能挂到的号」，而不是把整个表倒出来。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    start_date: date | None = Field(default=None, description="起始就诊日（含），缺省为今天")
    end_date: date | None = Field(default=None, description="截止就诊日（含），缺省为起始日起 14 天")
    department_code: str | None = Field(
        default=None, max_length=20, description="科室短代码，如 CARDIO"
    )
    department_id: str | None = Field(default=None, description="科室 UUID，与 code 二选一")
    doctor_id: str | None = Field(default=None, description="指定医生 UUID")
    title: str | None = Field(default=None, max_length=50, description="职称精确匹配，如 主任医师")
    slot_period: SlotPeriodLiteral | None = Field(default=None, description="am / pm")
    status: ScheduleStatusLiteral | None = Field(
        default="open", description="排班状态，缺省只出可约的 open"
    )
    keyword: str | None = Field(
        default=None, max_length=50, description="医生姓名 / 职称 / 专长模糊匹配"
    )
    include_expired: bool = Field(
        default=False,
        description="是否包含已过期排班；演示历史号源时才打开",
    )
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)

    @field_validator("department_id", "doctor_id")
    @classmethod
    def _validate_optional_uuids(cls, value: str | None, info) -> str | None:
        if value is None or not value.strip():
            return None
        _as_uuid(value, info.field_name)
        return value

    def resolved_window(self, today: date) -> tuple[date, date]:
        """解析出实际生效的日期区间（含端点）。

        ``today`` 由调用方传入而不是在这里调 ``date.today()``：
        「今天」必须按 Asia/Shanghai 口径算（见 seed/schedules.py 的时区注释），
        而容器默认 UTC。把取时间的责任留给 service，schema 只做纯计算，
        这样本方法可以脱离时钟做单元测试。

        区间倒置在这里抛 ValueError 而不是静默交换：start=明天 end=今天
        几乎肯定是前端日期控件传反了，静默交换会让调用方以为筛选生效了
        却拿到一个非预期的窗口；显式失败才能定位到请求本身。
        """
        start = self.start_date or today
        end = self.end_date or (start + timedelta(days=DEFAULT_SCHEDULE_WINDOW_DAYS - 1))
        if end < start:
            raise ValueError("end_date 不能早于 start_date")
        return start, end


# ==================== 号源响应 ====================


class ScheduleWithDoctorOut(BaseModel):
    """一条号源 + 医生与科室展示信息。

    医生信息反序列化进同一对象而不是嵌套 user 对象：患者端的号源列表就是
    「某医生某天的一个号」，前端渲染一行需要的字段全在这一层，
    嵌套只会让模板写成 ``item.doctor.real_name`` 这种无谓的层级。
    """

    id: str
    doctor_id: str
    doctor_name: str = Field(description="医生真实姓名（users.real_name）")
    doctor_title: str | None = Field(default=None, description="职称，未设置时为空")
    doctor_specialty: str | None = Field(default=None, description="专长描述")
    department_id: str
    department_name: str
    department_code: str
    schedule_date: date
    time_slot: str = Field(description='时段文本，如 "08:00-12:00"')
    slot_period: str
    total_slots: int
    booked_slots: int
    remaining_slots: int = Field(description="剩余号 = max(0, total - booked)")
    status: str
    registration_fee: int = Field(description="挂号费，单位分；格式化由前端负责")
    version: int = Field(description="乐观锁版本号，客户端回传可用于抢占校验")


class ScheduleListResponse(BaseModel):
    """号源分页响应。page/page_size 回显实际生效值（与 knowledge 风格一致）。"""

    items: list[ScheduleWithDoctorOut]
    total: int
    page: int
    page_size: int
    start_date: date
    end_date: date


class DepartmentDoctorOut(BaseModel):
    """「某科室有哪些号」视图里的一个医生。"""

    doctor_id: str
    doctor_name: str
    doctor_title: str | None = None
    doctor_specialty: str | None = None
    department_id: str
    department_name: str
    department_code: str
    available_dates: list[date] = Field(
        description="该医生在查询窗口内仍有余号的就诊日，升序去重"
    )
    schedule_count: int = Field(description="可约号源条数（status=open 且仍有余号）")
    total_remaining: int = Field(description="上述号源的剩余号之和")
    min_registration_fee: int | None = Field(
        default=None, description="该医生可约号源中的最低挂号费（分），供前端展示「起」"
    )


class DepartmentDoctorsResponse(BaseModel):
    """按科室聚合的可约医生列表。"""

    department_code: str | None = None
    department_id: str | None = None
    start_date: date
    end_date: date
    items: list[DepartmentDoctorOut]
    doctor_count: int


# ==================== 预约请求 ====================

class AppointmentCreate(BaseModel):
    """创建预约请求。

    ``triage_result_id`` 是分诊转预约的衔接字段：appointments 表已有该列，
    这里只做 UUID 形态校验，**不查其存在性也不导入任何 triage 模块**——
    跨模块的引用完整性交给外键约束与分诊侧自己保证（详见 service docstring）。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    schedule_id: str = Field(min_length=1, description="要抢的号源 UUID")
    chief_complaint: str | None = Field(
        default=None, max_length=1000, description="患者主诉，就诊时带入病历"
    )
    triage_result_id: str | None = Field(
        default=None, description="来源分诊结果 UUID（可选）"
    )
    triage_summary: str | None = Field(
        default=None, max_length=2000, description="分诊摘要，用于追溯 AI 建议依据"
    )

    @field_validator("schedule_id")
    @classmethod
    def _validate_schedule_id(cls, value: str) -> str:
        _as_uuid(value, "schedule_id")
        return value

    @field_validator("triage_result_id")
    @classmethod
    def _validate_triage_result_id(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        _as_uuid(value, "triage_result_id")
        return value


# ==================== 预约响应 ====================


class AppointmentOut(BaseModel):
    """预约单详情。

    只暴露预约自身字段 + 冗余的医生/科室名（列表页要用），
    不展开排班对象：号源余量是易变数据，塞进预约详情会让客户端误以为
    它能反映当前库存。
    """

    id: str
    patient_id: str
    doctor_id: str
    doctor_name: str
    department_id: str
    department_name: str
    schedule_id: str
    appointment_date: date
    time_slot: str
    status: str
    source: str
    chief_complaint: str | None = None
    triage_summary: str | None = None
    triage_result_id: str | None = None
    registration_fee: int = Field(description="挂号费快照，单位分")
    checked_in_at: datetime | None = None
    cancelled_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    allowed_transitions: list[str] = Field(
        default_factory=list,
        description="当前状态可迁移到的目标状态，前端据此禁用按钮",
    )


class AppointmentListResponse(BaseModel):
    """预约分页响应。"""

    items: list[AppointmentOut]
    total: int
    page: int
    page_size: int


__all__ = [
    "AppointmentCreate",
    "AppointmentListResponse",
    "AppointmentOut",
    "AppointmentSourceLiteral",
    "DEFAULT_SCHEDULE_WINDOW_DAYS",
    "DepartmentDoctorOut",
    "DepartmentDoctorsResponse",
    "ScheduleListResponse",
    "ScheduleQuery",
    "ScheduleStatusLiteral",
    "ScheduleWithDoctorOut",
    "SlotPeriodLiteral",
]
