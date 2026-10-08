"""医生排班种子数据。

为所有医生角色用户生成未来 14 天工作日的门诊排班，使挂号模块在演示时
有可查号源——没有排班的医生无法被预约，整个挂号流程就无从演示。

**幂等策略**：按 (doctor_id, schedule_date, time_slot) 预查询已存在组合，
只插入缺失的记录。不使用 ``ON CONFLICT DO NOTHING``，因为需要在内存中
精确计数新建条数并记录日志；预查询方式与 users.py / departments.py 风格一致。

**为什么提前生成排班**：真实医院排班由医务科按周/月编排，本系统简化为
种子脚本一次性生成两周数据，保证演示与面试时有完整的号源可供挂号、
退号、并发扣减等业务闭环跑通。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.auth import DoctorSchedule, Role, RoleName, User

logger = logging.getLogger(__name__)

# 本地时区：挂号业务面向国内患者，schedule_date 是 Date 类型（无时区信息），
# 按 Asia/Shanghai 计算「今天」更符合直觉——避免 UTC 导致北京时间凌晨
# 算成前一天，演示时出现「今天的号挂不了」的困惑。
_LOCAL_TZ = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True, slots=True)
class ScheduleTemplate:
    """一个排班模板，描述某个时段的基础配置。

    time_slot 使用整段文本（如 "08:00-12:00"）而非半小时粒度，原因见
    DoctorSchedule.time_slot 模型注释：「便于按半天/整段排班」。门诊实际
    也是按上午/下午两个半天单元管理号源，半小时粒度只在具体预约时才拆分。
    """

    slot_period: str
    time_slot: str
    # 不同职称的号源上限
    total_slots_chief: int       # 主任医师
    total_slots_deputy: int      # 副主任医师
    total_slots_attending: int   # 主治医师
    total_slots_default: int     # 其他/未设职称
    # 挂号费（单位分），按职称区分
    fee_chief: int
    fee_deputy: int
    fee_attending: int
    fee_default: int


SCHEDULE_TEMPLATES: tuple[ScheduleTemplate, ...] = (
    ScheduleTemplate(
        slot_period="am",
        time_slot="08:00-12:00",
        total_slots_chief=20,
        total_slots_deputy=25,
        total_slots_attending=30,
        total_slots_default=30,
        fee_chief=5000,      # 50 元
        fee_deputy=3500,     # 35 元
        fee_attending=2500,  # 25 元
        fee_default=2500,
    ),
    ScheduleTemplate(
        slot_period="pm",
        time_slot="14:00-17:30",
        total_slots_chief=15,
        total_slots_deputy=20,
        total_slots_attending=25,
        total_slots_default=25,
        fee_chief=5000,
        fee_deputy=3500,
        fee_attending=2500,
        fee_default=2500,
    ),
)

# 职称 -> (total_slots 取值函数, fee 取值函数) 的分发映射。
# 用 lambda 而非 if-elif 链，便于后续扩展职称档位。
_TITLE_DISPATCH: dict[str | None, tuple[str, str]] = {
    "主任医师": ("chief", "chief"),
    "副主任医师": ("deputy", "deputy"),
    "主治医师": ("attending", "attending"),
}


def _resolve_slots_and_fee(
    title: str | None,
    template: ScheduleTemplate,
) -> tuple[int, int]:
    """根据职称从模板中取出对应的号源数和挂号费。"""
    suffix_pair = _TITLE_DISPATCH.get(title)
    if suffix_pair is None:
        return template.total_slots_default, template.fee_default
    slot_suffix, fee_suffix = suffix_pair
    total_slots = getattr(template, f"total_slots_{slot_suffix}")
    fee = getattr(template, f"fee_{fee_suffix}")
    return total_slots, fee


def _generate_workdays(start: date, days: int) -> list[date]:
    """从 start 起生成 days 个自然日中的工作日（周一到周五）。"""
    workdays: list[date] = []
    for i in range(days):
        d = start + timedelta(days=i)
        if d.weekday() < 5:  # 0=Mon .. 4=Fri
            workdays.append(d)
    return workdays


async def seed_schedules(db: AsyncSession) -> int:
    """为所有医生生成未来 14 天工作日的排班，返回新建条数。

    依赖 ``seed_users`` 与 ``seed_departments`` 先执行：医生用户及其
    department_id 必须已存在。若没有找到任何医生用户，安全返回 0。

    幂等实现：预查询已存在的 (doctor_id, schedule_date, time_slot) 组合，
    只插入缺失的记录，重复运行不会报错也不会产生重复数据。
    """
    # 1. 查出所有医生角色用户
    # 通过 role name 反查 role_id，避免硬编码 ID
    doctor_role_id: int | None = (
        await db.execute(
            select(Role.id).where(Role.name == RoleName.DOCTOR.value)
        )
    ).scalar_one_or_none()

    if doctor_role_id is None:
        logger.warning("未找到医生角色，跳过排班种子数据")
        return 0

    doctors = (
        await db.execute(
            select(User).where(
                User.role_id == doctor_role_id,
                User.is_active.is_(True),
            )
        )
    ).scalars().all()

    if not doctors:
        logger.warning("没有找到活跃的医生用户，跳过排班种子数据")
        return 0

    # 过滤掉没有科室的医生（不应出现在排班里）
    valid_doctors = [d for d in doctors if d.department_id is not None]
    if len(valid_doctors) < len(doctors):
        skipped = len(doctors) - len(valid_doctors)
        logger.warning(
            "%d 个医生缺少 department_id，已跳过排班生成", skipped
        )

    if not valid_doctors:
        logger.warning("所有医生均缺少 department_id，跳过排班种子数据")
        return 0

    # 2. 生成工作日列表
    # 必须用 _LOCAL_TZ 而非 date.today()：容器默认时区是 UTC，
    # date.today() 在北京时间 00:00-08:00 之间会返回「昨天」，
    # 导致当天号源整体缺失一天。schedule_date 是 Date 类型（无时区语义），
    # 因此这里先取东八区的当前时刻，再截断为日期。
    today = datetime.now(_LOCAL_TZ).date()
    workdays = _generate_workdays(today, 14)

    if not workdays:
        logger.info("未来 14 天无工作日，跳过排班种子数据")
        return 0

    # 3. 预查询已存在的排班组合
    rows = (
        await db.execute(
            select(
                DoctorSchedule.doctor_id,
                DoctorSchedule.schedule_date,
                DoctorSchedule.time_slot,
            )
        )
    ).all()
    existing_keys = {(row[0], row[1], row[2]) for row in rows}

    # 4. 逐医生、逐日、逐时段生成
    created = 0
    for doctor in valid_doctors:
        for day in workdays:
            for tmpl in SCHEDULE_TEMPLATES:
                key = (doctor.id, day, tmpl.time_slot)
                if key in existing_keys:
                    continue

                total_slots, fee = _resolve_slots_and_fee(doctor.title, tmpl)

                db.add(
                    DoctorSchedule(
                        doctor_id=doctor.id,
                        department_id=doctor.department_id,
                        schedule_date=day,
                        time_slot=tmpl.time_slot,
                        slot_period=tmpl.slot_period,
                        total_slots=total_slots,
                        booked_slots=0,
                        version=1,
                        status="open",
                        registration_fee=fee,
                    )
                )
                existing_keys.add(key)
                created += 1

    await db.flush()
    logger.info(
        "排班种子数据：新建 %d 条（%d 位医生 × %d 个工作日 × %d 时段）",
        created,
        len(valid_doctors),
        len(workdays),
        len(SCHEDULE_TEMPLATES),
    )
    return created


__all__ = ["seed_schedules"]
