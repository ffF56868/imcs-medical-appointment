"""号源查询与预约（挂号）业务逻辑。

本模块是 IMCS 里第一处真正面对**并发写冲突**的业务代码，因此实现纪律
比其它 service 更严：所有关于「还剩几个号」的判断都必须由数据库在
一条 UPDATE 里完成，Python 侧不做任何「先读后改」。

============================ 乐观锁为什么绕不开两次往返 ============================

``DoctorSchedule`` 的类 docstring 已经定死了方案::

    UPDATE doctor_schedules
       SET booked_slots = booked_slots + 1, version = version + 1
     WHERE id = :id AND version = :version AND booked_slots < total_slots

如果改成「先 SELECT 判断 remaining > 0，再 UPDATE SET booked_slots = :新值」，
两条语句之间必然存在一个窗口：另一个请求在同一时刻读到同样的余量、
写入同样的新值，两个患者各占一号但 ``booked_slots`` 只加了一次——
超卖。这不是概率问题，只要并发数 ≥2 就一定会撞到。
把「读—判—写」压进一条带条件的 UPDATE，数据库行锁保证三件事原子完成，
受影响行数就是「我到底抢到了没有」的唯一可信答案。

============================ 号源扣减与预约记录同生同灭 ============================

扣号成功但插预约失败 → 患者没挂上号却白扣一个号，号源永久少一格；
插预约成功但扣号失败 → 同一时段被挂进超过 total_slots 的人，
到院后没人能看上病。两种都是真实的患者纠纷与投诉来源，
所以两步必须在**同一个事务**里提交，任何一步失败整体回滚
（见 ``create_appointment`` 的 try/except 结构）。

============================ 与分诊侧的边界 ============================

本模块**不 import 任何 triage 相关文件**。分诊转预约只靠
``AppointmentCreate.triage_result_id`` 字段透传：appointments 表该列是
无外键约束的裸 UUID（见 models/clinical.py），因此这里只做 UUID 形态校验，
不做存在性查询。查不到对应分诊记录不会报错——这是刻意取舍，
详见 ``_resolve_triage_link`` 的说明。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    DuplicateAppointmentError,
    InvalidStateTransitionError,
    ResourceNotFoundError,
    SlotConflictError,
    SlotFullError,
)
from app.models.auth import Department, DoctorSchedule, RoleName, User
from app.models.base import utcnow
from app.models.clinical import Appointment, AppointmentStatus
from app.models.patient import PatientProfile
from app.schemas.appointment import ScheduleQuery
from app.services.state_machine import allowed_targets, assert_transition, validate_status_value

logger = logging.getLogger(__name__)

# 预约业务面向国内患者，「今天」必须按东八区算，理由与 seed/schedules.py 完全一致：
# 容器默认 UTC，date.today() 在北京时间 00:00–08:00 之间会返回「昨天」，
# 于是当天号源被整体判定为过期，演示时表现为「今天的号挂不了」。
_LOCAL_TZ = ZoneInfo("Asia/Shanghai")

# 乐观锁重试参数。三次退避（50/100/200ms）覆盖的是「同一号源被连点」这种
# 毫秒级竞争；再多次只是把一个已经排队的请求拖得更久，患者端体验反而变差，
# 所以耗尽后直接报 SlotConflictError 让前端刷新重选。
SLOT_MAX_RETRIES = 3
SLOT_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (0.05, 0.10, 0.20)

# 有效预约状态：占用号源、参与重复挂号防呆的状态。
# cancelled / no_show 不占号，因此既不计入防呆也不受部分唯一索引约束。
EFFECTIVE_APPOINTMENT_STATUSES: tuple[str, ...] = (
    AppointmentStatus.BOOKED.value,
    AppointmentStatus.CHECKED_IN.value,
    AppointmentStatus.IN_PROGRESS.value,
    AppointmentStatus.COMPLETED.value,
)

# 「已取消」这一动作的目标状态，集中一处便于路由层复用。
CANCEL_TARGET = AppointmentStatus.CANCELLED.value

# 医生角色的固定主键。Role.id 由 ROLE_DEFINITIONS 以 autoincrement=False 写死，
# 是种子数据与权限矩阵共同承认的稳定标识（tests/imcs_sqlite.ROLE_IDS 同源）。
# 刻意不做「按角色名 JOIN roles 表」的子查询：那会给每个号源查询多一次关联，
# 而这里的用途只是从 users 里筛出医生，常量比较更直白也更快。
DOCTOR_ROLE_ID = 2


def _stringify_uuid_fields(row: dict[str, object], keys: tuple[str, ...]) -> None:
    """把行内指定键的 uuid.UUID 原地转成 str（None 保持 None）。

    ORM 读出的 PGUUID 列是 ``uuid.UUID``，而响应 schema 对应字段声明为 ``str``。
    Pydantic v2 严格模式不会把 UUID 隐式转字符串，会抛 string_type 校验错误
    （实测 PG 与 SQLite 同现，不是测试环境专属）。转换统一放在 service 出口，
    路由层才能保持 ``ScheduleWithDoctorOut(**row)`` 这种薄装配写法。
    """
    for key in keys:
        value = row.get(key)
        if value is not None and not isinstance(value, str):
            row[key] = str(value)


def today_local() -> date:
    """东八区口径的今天。单独成函数便于测试冻结时间。"""
    return datetime.now(_LOCAL_TZ).date()


# ==================== 号源查询 ====================


async def list_available_schedules(
    db: AsyncSession,
    query: ScheduleQuery,
) -> tuple[list[dict[str, object]], int, tuple[date, date]]:
    """分页查询号源，返回 (当前页行, 总数, 实际生效日期区间)。

    筛选条件全部下推到 SQL：号源表随排班周次线性增长，
    若在 Python 侧过滤 keyword（医生姓名/专长），分页 total 就会与实际
    可约数量不一致，前端会出现「共 3 页但第 2 页是空的」。

    默认只出 ``status='open'`` 且 ``schedule_date >= 今天`` 的行：
    full/closed 的号源对患者的意义与「不存在」相同，展示出来只会制造
    点了才知道挂不上的无效点击。
    """
    start, end = query.resolved_window(today_local())

    conditions = [
        DoctorSchedule.schedule_date >= start,
        DoctorSchedule.schedule_date <= end,
    ]
    if not query.include_expired:
        conditions.append(DoctorSchedule.schedule_date >= today_local())
    if query.status is not None:
        conditions.append(DoctorSchedule.status == query.status)
    if query.slot_period:
        conditions.append(DoctorSchedule.slot_period == query.slot_period)
    if query.doctor_id:
        conditions.append(DoctorSchedule.doctor_id == uuid.UUID(query.doctor_id))
    if query.title:
        conditions.append(User.title == query.title)
    if query.department_code:
        conditions.append(Department.code == query.department_code)
    if query.department_id:
        conditions.append(DoctorSchedule.department_id == uuid.UUID(query.department_id))
    if query.keyword:
        pattern = f"%{query.keyword}%"
        # 三个 OR 分支都是短列（String(50)/String(200)），前置通配走不了索引
        # 但结果集已被日期区间收敛到两周内的号源，规模可控。
        conditions.append(
            or_(
                User.real_name.ilike(pattern),
                User.title.ilike(pattern),
                User.specialty.ilike(pattern),
            )
        )

    base = (
        select(
            DoctorSchedule.id.label("id"),
            DoctorSchedule.schedule_date.label("schedule_date"),
            DoctorSchedule.time_slot.label("time_slot"),
            DoctorSchedule.slot_period.label("slot_period"),
            DoctorSchedule.total_slots.label("total_slots"),
            DoctorSchedule.booked_slots.label("booked_slots"),
            DoctorSchedule.status.label("status"),
            DoctorSchedule.version.label("version"),
            DoctorSchedule.registration_fee.label("registration_fee"),
            DoctorSchedule.doctor_id.label("doctor_id"),
            DoctorSchedule.department_id.label("department_id"),
            User.real_name.label("doctor_name"),
            User.title.label("doctor_title"),
            User.specialty.label("doctor_specialty"),
            Department.name.label("department_name"),
            Department.code.label("department_code"),
        )
        .join(User, User.id == DoctorSchedule.doctor_id)
        .join(Department, Department.id == DoctorSchedule.department_id)
        .where(*conditions)
    )

    count_stmt = select(func.count()).select_from(base.subquery())
    total = int((await db.execute(count_stmt)).scalar_one())

    page = max(1, query.page)
    page_size = max(1, min(query.page_size, 100))
    # id 收尾排序是必需的：前三列（日期/时段/姓名）在「同科室多位同名医生同半天」
    # 时仍可能完全相同，没有唯一列兜底的话 OFFSET/LIMIT 的分页顺序不稳定，
    # 同一请求两次可能返回不同行，患者翻页会看到重复或漏项。
    page_stmt = (
        base.order_by(
            DoctorSchedule.schedule_date.asc(),
            DoctorSchedule.slot_period.asc(),
            User.real_name.asc(),
            DoctorSchedule.id.asc(),
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows = [dict(row._mapping) for row in (await db.execute(page_stmt)).all()]
    for row in rows:
        row["remaining_slots"] = max(
            0, int(row["total_slots"]) - int(row["booked_slots"])
        )
        # UUID → str：ORM 列读出来是 uuid.UUID，而响应 schema 字段是 str，
        # Pydantic v2 不会把 UUID 隐式转成 str（会抛 string_type 校验错，
        # 在 PostgreSQL 上同样发生，不是 SQLite 专属）。序列化统一在 service
        # 出口完成，路由只做 ScheduleWithDoctorOut(**row) 的薄装配。
        _stringify_uuid_fields(row, ("id", "doctor_id", "department_id"))
    return rows, total, (start, end)


async def list_doctors_with_open_slots(
    db: AsyncSession,
    query: ScheduleQuery,
) -> tuple[list[dict[str, object]], tuple[date, date]]:
    """按科室聚合「未来还有号的医生」，对应患者端「心血管科有哪些号」。

    **用 EXISTS 而不是 JOIN 号源表**：一个医生两周有十几个半天号，
    JOIN 后行数放大十几倍，聚合前还得 DISTINCT 医生 ID；EXISTS 让数据库
    自己短路（找到一条满足条件的号源即返回），并且天然不受分页影响。

    余号统计放在拿到医生列表之后逐医生补一次小聚合，而不是塞进主查询：
    主查询一旦带上 GROUP BY 就要重新对齐每一列的聚合语义，可读性和
    正确性都会下降。医生数量有限（单科室个位数），多几次往返成本可接受。

    **不用 ``array_agg``**：该函数在 SQLite 上不存在（3.38 之前根本没有），
    而本项目的单元测试跑在 SQLite 上；PG 侧能用的聚合函数并不等于跨方言可用。
    取原始行在 Python 侧去重排序，两个后端行为完全一致。
    """
    start, end = query.resolved_window(today_local())
    today = today_local()
    effective_start = start if start >= today else today

    open_slot = (
        select(DoctorSchedule.id)
        .where(
            DoctorSchedule.doctor_id == User.id,
            DoctorSchedule.schedule_date >= effective_start,
            DoctorSchedule.schedule_date <= end,
            DoctorSchedule.status == "open",
            DoctorSchedule.booked_slots < DoctorSchedule.total_slots,
        )
        .correlate(User)
        .exists()
    )

    conditions = [
        User.role_id == DOCTOR_ROLE_ID,
        User.is_active.is_(True),
        User.department_id.is_not(None),
        open_slot,
    ]
    if query.department_code:
        conditions.append(Department.code == query.department_code)
    if query.department_id:
        conditions.append(User.department_id == uuid.UUID(query.department_id))
    if query.title:
        conditions.append(User.title == query.title)
    if query.keyword:
        pattern = f"%{query.keyword}%"
        conditions.append(
            or_(
                User.real_name.ilike(pattern),
                User.title.ilike(pattern),
                User.specialty.ilike(pattern),
            )
        )

    stmt = (
        select(
            User.id.label("doctor_id"),
            User.real_name.label("doctor_name"),
            User.title.label("doctor_title"),
            User.specialty.label("doctor_specialty"),
            User.department_id.label("department_id"),
            Department.name.label("department_name"),
            Department.code.label("department_code"),
        )
        .join(Department, Department.id == User.department_id)
        .where(and_(*conditions))
        .order_by(Department.code.asc(), User.title.asc(), User.real_name.asc())
    )
    doctors = [dict(row._mapping) for row in (await db.execute(stmt)).all()]

    for item in doctors:
        detail_rows = (
            await db.execute(
                select(
                    DoctorSchedule.schedule_date,
                    DoctorSchedule.total_slots - DoctorSchedule.booked_slots,
                    DoctorSchedule.registration_fee,
                ).where(
                    DoctorSchedule.doctor_id == item["doctor_id"],
                    DoctorSchedule.schedule_date >= effective_start,
                    DoctorSchedule.schedule_date <= end,
                    DoctorSchedule.status == "open",
                    DoctorSchedule.booked_slots < DoctorSchedule.total_slots,
                )
            )
        ).all()
        dates = sorted({row[0] for row in detail_rows})
        fees = [int(row[2]) for row in detail_rows]
        item["available_dates"] = dates
        item["schedule_count"] = len(detail_rows)
        item["total_remaining"] = sum(max(0, int(r[1])) for r in detail_rows)
        item["min_registration_fee"] = min(fees) if fees else None
        # 与 list_available_schedules 同一出口约定：detail 查询已用完 UUID 对象，
        # 转字符串放在最后，响应模型字段是 str。
        _stringify_uuid_fields(item, ("doctor_id", "department_id"))
    return doctors, (effective_start, end)


# ==================== 预约创建 ====================


async def _resolve_patient_profile(
    db: AsyncSession, user_id: uuid.UUID
) -> PatientProfile:
    """把 CurrentUser 映射到 PatientProfile。

    appointments.patient_id 的外键指向 patient_profiles.id 而非 users.id
    （见 models/clinical.py），所以挂号必须以档案为主体。
    注册流程已同步建档（auth_service.register 的注释说明了为什么不能推迟），
    查不到档案通常意味着这个账号不是走自助注册的——例如管理员手工建的患者号，
    此时必须明确报错，不能让外键在 flush 时报错变成 500。
    """
    profile = (
        await db.execute(
            select(PatientProfile).where(PatientProfile.user_id == user_id)
        )
    ).scalar_one_or_none()
    if profile is None:
        raise ResourceNotFoundError(
            "当前账号尚未建立患者健康档案，无法挂号；请先完善实名档案信息"
        )
    return profile


async def _load_bookable_schedule(
    db: AsyncSession, schedule_id: uuid.UUID
) -> DoctorSchedule:
    """取号源并做「能不能约」的前置判定。

    过期与关闭状态的检查放在这里（而不是等 UPDATE 影响行数为 0 再猜原因）：
    乐观锁那条 UPDATE 只能告诉我们「没抢到」，抢不到的原因可能是满号、
    可能是排班被关闭、也可能是号已经过期。原因不同，给患者的引导完全不同
    （满号→换时段；过期→明天再来；关闭→联系医院），所以能提前确定的
    一定要在这里确定。

    「不存在」与「已过期」都报 ResourceNotFoundError：过期排班对患者的可观测
    结果与它不存在时完全一样（挂不上），统一成 404 可以让前端只处理一种分支。
    closed 却报 SlotFullError 而非 404：停诊是真实存在的排班状态，
    用 409 + 「已停诊」能让患者明白该换时段而不是怀疑链接坏了。
    """
    schedule = await db.get(DoctorSchedule, schedule_id)
    if schedule is None:
        raise ResourceNotFoundError("号源不存在或已下架")
    if schedule.status == "closed":
        raise SlotFullError("该医生此时间段门诊已停诊，请选择其他时段")
    if schedule.schedule_date < today_local():
        raise ResourceNotFoundError("该排班日期已过期，无法挂号")
    return schedule


async def _assert_not_duplicate(
    db: AsyncSession, patient_id: uuid.UUID, schedule: DoctorSchedule
) -> None:
    """重复挂号防呆（第一层：给出精确提示）。

    判据列组合取自 ``uq_appt_patient_slot_active`` 部分唯一索引的实际定义：
    ``(patient_id, schedule_id) WHERE status NOT IN ('cancelled','no_show')``。
    也就是说约束的粒度是「同一患者 + 同一号源」，而**不是**任务描述里提到的
    「schedule_date + slot_period」——后者跨医生：同一上午挂心内科和呼吸内科
    两个号在本系统里是合法的（现实中确实有人这么干），若按 date+period 防呆
    会把合法行为拦掉，还会与数据库约束口径不一致导致防呆形同虚设。

    状态过滤用 EFFECTIVE_APPOINTMENT_STATUSES 白名单而非直接复刻索引谓词里的
    黑名单：白名单等价于「占用号源的状态」，将来新增一个不占号的状态时，
    黑名单写法会把它误判为有效预约，白名单则默认放行，方向上更安全。
    """
    existing = (
        await db.execute(
            select(Appointment.id).where(
                Appointment.patient_id == patient_id,
                Appointment.schedule_id == schedule.id,
                Appointment.status.in_(EFFECTIVE_APPOINTMENT_STATUSES),
            )
        )
    ).first()
    if existing is not None:
        raise DuplicateAppointmentError(
            "你在该医生的同一时段已有预约，无需重复挂号",
            detail={"appointment_conflict_id": str(existing[0])},
        )


async def _consume_slot(
    db: AsyncSession, schedule_id: uuid.UUID
) -> DoctorSchedule:
    """乐观锁扣号：一条 UPDATE 完成扣减与版本推进，失败则有限重试。

    **为什么禁止先 SELECT 再改**：见模块 docstring。补充一层——即便在
    SELECT 上加 FOR UPDATE，也只是把冲突从「静默超卖」换成「请求排队等待」，
    高并发抢同一个热门专家号时会形成锁等待链甚至死锁；而号源竞争的典型形态是
    「一百个人抢二十个号」，绝大多数请求注定失败，让它们立刻拿到失败答案
    比让它们排队更符合业务预期。

    rowcount==0 后必须重读一行才能区分三种情况：
    - version 变了但仍有余号 → 纯竞争，重试可能成功
    - booked_slots >= total_slots 或 status != open → 确定性失败，重试无意义
    - 行不见了 → 排班被删，报 404

    **重读前只「清除未命中的 ORM 对象」，绝不 rollback**：
    ``db.get()`` 会把这行缓存进 identity map。UPDATE 没打中时数据库里的值
    根本没变，ORM 缓存却是过期的——不清除，下一轮循环仍拿旧 version 去撞，
    三次退避全部白等（这是本函数唯一需要处理的缓存陈旧问题）。
    之所以不能用 rollback 达成同样的刷新效果：调用方 ``create_appointment``
    把扣号放在插入之前，若这里回滚，事务里已经写过的任何语句都会被抹掉；
    而「撤销扣号」的正确时机是 INSERT 撞唯一约束之后由那个 except 分支处理。
    一个吞事务的 rollback 藏在重试循环里，是最难在 review 中发现的副作用。
    """
    for attempt in range(SLOT_MAX_RETRIES + 1):
        current = await db.get(DoctorSchedule, schedule_id)
        if current is None:
            raise ResourceNotFoundError("号源不存在或已下架")
        expected_version = current.version

        result = await db.execute(
            update(DoctorSchedule)
            # 三个条件缺一不可：id 定位行、version 检测竞争、booked<total 兜住满号。
            # 少了 booked<total，并发下两个请求都可能把 booked 推到 total 之上，
            # 而 ck_booked_within_total 约束会让其中一个收到 IntegrityError（500）。
            .where(
                DoctorSchedule.id == schedule_id,
                DoctorSchedule.version == expected_version,
                DoctorSchedule.booked_slots < DoctorSchedule.total_slots,
            )
            # 用 SQL 表达式而非 Python 算术：SET booked_slots = booked_slots + 1
            # 由数据库基于它自己看到的当前值计算，客户端传入的旧值不参与运算，
            # 这正是乐观锁能安全重试的原因。
            .values(
                booked_slots=DoctorSchedule.booked_slots + 1,
                version=DoctorSchedule.version + 1,
            )
        )

        if result.rowcount == 1:
            # autoflush=False 的会话里，core UPDATE 不会自动同步 ORM 属性，
            # 必须显式 refresh 才能读到数据库算出的新 booked/version，
            # 否则调用方拿到的仍是扣号前的数字（满号联动会因此漏判）。
            await db.refresh(current)
            return current

        # rowcount == 0：丢弃过期缓存后重读，判断到底是竞争还是真满。
        db.expunge(current)
        reloaded = await db.get(DoctorSchedule, schedule_id)
        if reloaded is None:
            raise ResourceNotFoundError("号源不存在或已下架")
        if reloaded.status != "open":
            raise SlotFullError("该时段号源已停止预约，请选择其他时段")
        if reloaded.booked_slots >= reloaded.total_slots:
            raise SlotFullError()
        if reloaded.version == expected_version:
            # 版本没变、也没满、也没关闭，却更新失败——不是竞争，重试也不会成功。
            raise SlotConflictError(
                "号源更新未生效，请刷新后重试",
                detail={
                    "schedule_id": str(schedule_id),
                    "version": reloaded.version,
                },
            )
        if attempt >= SLOT_MAX_RETRIES:
            break
        logger.info(
            "号源 %s 乐观锁竞争，第 %d 次重试", schedule_id, attempt + 1
        )
        await asyncio.sleep(SLOT_RETRY_BACKOFF_SECONDS[attempt])

    raise SlotConflictError(
        detail={"schedule_id": str(schedule_id), "retry_count": SLOT_MAX_RETRIES}
    )


def _resolve_triage_link(
    triage_result_id: str | None, triage_summary: str | None
) -> tuple[uuid.UUID | None, str | None]:
    """处理分诊转预约的衔接字段。

    **只做 UUID 形态校验，不查存在性**：appointments.triage_result_id 是无外键
    约束的裸 UUID 列（models/clinical.py），要校验存在性就必须 import triage 侧
    模型或 schema——那会让预约模块与并行开发中的分诊模块产生编译期耦合，
    两侧任何一方改名都会让对方在收集阶段就 ImportError。

    代价是可能落库一个指向不存在记录的 UUID。这个代价可以接受：
    该字段的用途是「评估分诊到挂号的转化率与准确率」（模型注释原话），
    属于分析口径，不是交易凭据；分析侧做 JOIN 时自然会把悬空引用过滤掉。
    若日后要收紧，正确做法是给该列加外键迁移，而不是在 service 里手写查询。
    """
    if triage_result_id is None or not triage_result_id.strip():
        return None, triage_summary
    return uuid.UUID(triage_result_id), triage_summary


async def create_appointment(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    schedule_id: uuid.UUID,
    chief_complaint: str | None = None,
    triage_result_id: str | None = None,
    triage_summary: str | None = None,
) -> Appointment:
    """患者自主挂号（分诊转挂号同样走本函数，靠 triage_* 字段区分来源）。

    执行顺序是刻意的：**先查重、再扣号、最后插记录**。
    查重放在扣号之前，是因为重复请求根本不该消耗一个号——
    反过来做的话，每次误点都会先扣掉一个号再因唯一约束回滚，
    虽然数据最终一致，但期间版本号被推高，会给真正的竞争者制造无谓的重试。

    全程单事务：``_consume_slot`` 的 UPDATE 与 Appointment 的 INSERT
    要么一起提交，要么一起回滚（模块 docstring 说明了半途失败的后果）。

    **扣号必须在插入之前**：会话工厂是 ``autoflush=False``，但乐观锁那条
    ``update()`` 本身会开启并下发到数据库。若把 add + flush 放在它前面，
    INSERT 就先进了同一个事务——之后抢不到号时我们面对的是
    「一条已插入的预约 + 一次空 UPDATE」，回滚范围与错误归因都会变模糊；
    更糟的是 autoflush 语义一旦在别处被改成 True，重复预约的 IntegrityError
    就会从「兜并发窗口」变成「抢先于扣号发生」，两条失败路径彻底混在一起。
    """
    profile = await _resolve_patient_profile(db, user_id)
    schedule = await _load_bookable_schedule(db, schedule_id)
    await _assert_not_duplicate(db, profile.id, schedule)

    link_id, summary = _resolve_triage_link(triage_result_id, triage_summary)
    source = "triage_referral" if link_id is not None else "self"

    # 校验初始状态值本身合法。「新建即 booked」不是一次状态迁移（没有 from 状态），
    # 所以不能用 assert_transition；但状态值写错会让后续所有流转判断失效，
    # validate_status_value 正是为此准备的入口——它同时兜住了「有人改了枚举值」
    # 与「本模块传了别实体的状态串」这两类静默错误。
    initial_status = validate_status_value("appointment", AppointmentStatus.BOOKED.value)

    consumed = await _consume_slot(db, schedule_id)

    appointment = Appointment(
        patient_id=profile.id,
        doctor_id=consumed.doctor_id,
        department_id=consumed.department_id,
        schedule_id=consumed.id,
        appointment_date=consumed.schedule_date,
        time_slot=consumed.time_slot,
        status=initial_status,
        source=source,
        triage_summary=summary,
        triage_result_id=link_id,
        chief_complaint=chief_complaint,
        # 费用从号源快照复制到预约：挂号费后续可能调价，
        # 但患者已成交的那一单金额必须锁定当时看到的数字。
        registration_fee=consumed.registration_fee,
    )
    db.add(appointment)

    try:
        # 先 flush 再标满：flush 才能拿到 appointment.id（UUID 主键是 flush 时
        # 才生成的默认值），也让部分唯一索引 uq_appt_patient_slot_active 在这一刻
        # 真正生效——并发窗口的兜底全靠它。
        await db.flush()
        await _sync_schedule_full_flag(db, consumed)
        await db.commit()
    except IntegrityError as exc:
        # 第二层兜底：查重通过但在扣号与插入之间，另一个并发请求插进了同一
        # (patient_id, schedule_id) 有效预约。PG 的部分唯一索引会拦住 INSERT，
        # 而我们的事务里已经扣了一个号——commit 失败后 SQLAlchemy 把会话置为
        # needs_rollback，rollback 会把那次 UPDATE 一并撤销，号源自动回到原位。
        await db.rollback()
        logger.info(
            "并发重复挂号被唯一索引拦截 patient=%s schedule=%s：%s",
            profile.id,
            schedule_id,
            exc.orig,
        )
        raise DuplicateAppointmentError(
            "你在该医生的同一时段已有预约，请勿重复提交"
        ) from exc

    await db.refresh(appointment)
    logger.info(
        "挂号成功 appointment=%s patient=%s schedule=%s booked=%s/%s",
        appointment.id,
        profile.id,
        schedule_id,
        consumed.booked_slots,
        consumed.total_slots,
    )
    return appointment


async def _sync_schedule_full_flag(
    db: AsyncSession, schedule: DoctorSchedule
) -> None:
    """满号联动：booked 追平 total 时把 status 从 open 推到 full。

    为什么要维护这个冗余状态而不是每次都算 ``booked < total``：
    号源查询接口按 status 过滤（并走 idx_schedule_date_status 复合索引），
    满了却不标 full，患者会在列表里看到一排点进去就报满的号。

    这条 UPDATE 不带 version 条件，因为它发生在扣号成功之后的同一事务内：
    本事务刚刚把 version 推进过，此刻不与其他人竞争同一个「是否刚满」的判断，
    且它是幂等的派生标记（full 再写 full 无害）。

    WHERE 里重复一遍 ``status='open'`` 的判定而不是只按 id：Python 侧那个
    if 读的是本事务 refresh 后的快照，数据库里别人可能已经把它改成 closed。
    带上条件后最坏情况是空操作，不会把停诊的排班复活成 full。
    """
    if schedule.booked_slots >= schedule.total_slots and schedule.status == "open":
        await db.execute(
            update(DoctorSchedule)
            .where(DoctorSchedule.id == schedule.id, DoctorSchedule.status == "open")
            .values(status="full")
        )


# ==================== 取消 / 签到 ====================


async def _release_slot(
    db: AsyncSession, schedule_id: uuid.UUID, *, reason: str
) -> None:
    """回退一个号源（取消预约用），同样走乐观锁。

    为什么回退也要乐观锁：退号与别人的抢号是同一段临界区。
    若写成无条件 ``booked_slots = booked_slots - 1``，它与「另一请求刚把
    booked 推到 total」交错时，会把已满的号源减到 total-1 却仍标着 full，
    于是列表里永远看不见这个号——凭空损失一个可售号源。
    带上 version 条件后，交错会被检测成竞争并重试。

    ``booked_slots > 0`` 是回退方向特有的守卫（扣号方向对应的是 ``< total``）：
    它挡住「重复退号把计数压成负数」。ck_booked_slots_non_negative 约束虽然
    也能拦，但那会变成 500；放在 WHERE 里则退化成一次可解释的空操作。

    重试前同样只 expunge 不 rollback（理由见 ``_consume_slot``）。
    本函数由 ``cancel_appointment`` 在 flush 之后调用，若这里回滚整事务，
    刚写好的 status='cancelled' 会被一起抹掉——那反而是我们想要的最终效果
    （号没退成就别标取消），但绝不能靠「顺手回滚整个事务」去实现它：
    失败路径应当显式，见调用方 except 分支里的注释。
    """
    for attempt in range(SLOT_MAX_RETRIES + 1):
        current = await db.get(DoctorSchedule, schedule_id)
        if current is None:
            # 排班行已被删除：预约记录还在（外键 ondelete 用的是 RESTRICT 语义的
            # 普通 FK），此处不再回退号源，让取消本身照常成功——
            # 患者已经拿不到服务了，不该再被一个库存记账问题卡住。
            logger.warning("退号时排班已不存在 schedule=%s reason=%s", schedule_id, reason)
            return

        result = await db.execute(
            update(DoctorSchedule)
            .where(
                DoctorSchedule.id == schedule_id,
                DoctorSchedule.version == current.version,
                DoctorSchedule.booked_slots > 0,
            )
            .values(
                booked_slots=DoctorSchedule.booked_slots - 1,
                version=DoctorSchedule.version + 1,
            )
        )
        if result.rowcount == 1:
            await db.refresh(current)
            # 曾满后释放出余号 → 回到 open，否则 full 状态会让它在号源列表里隐身。
            if current.status == "full" and current.booked_slots < current.total_slots:
                await db.execute(
                    update(DoctorSchedule)
                    .where(DoctorSchedule.id == schedule_id)
                    .values(status="open")
                )
            return

        db.expunge(current)
        reloaded = await db.get(DoctorSchedule, schedule_id)
        if reloaded is None or reloaded.booked_slots <= 0:
            logger.warning(
                "退号未生效（号源已不存在或计数已归零）schedule=%s reason=%s",
                schedule_id,
                reason,
            )
            return
        if reloaded.version == current.version:
            # 版本没变、计数也还大于 0，却没更新成功：不是竞争，重试也不会成功。
            raise SlotConflictError(
                "号源回退未生效，请稍后重试", detail={"schedule_id": str(schedule_id)}
            )
        if attempt >= SLOT_MAX_RETRIES:
            break
        await asyncio.sleep(SLOT_RETRY_BACKOFF_SECONDS[attempt])

    raise SlotConflictError(
        "号源回退失败，请稍后重试", detail={"schedule_id": str(schedule_id)}
    )


async def cancel_appointment(
    db: AsyncSession, appointment_id: uuid.UUID, *, requester_user_id: uuid.UUID
) -> Appointment:
    """取消预约（患者退号）。

    归属校验用「取到记录后比对 patient_id，不匹配就报 404」而不是 403：
    见 ``get_appointment_detail`` 的说明，两者在本项目统一取 404 口径。

    状态机：booked → cancelled 合法；checked_in → cancelled 在转换矩阵里也是
    允许的（models/clinical.py 如此定义），但已签到意味着患者已到院、
    叫号系统已经生成排队号，由患者自助撤销会造成前台与医生端数据打架。
    因此本函数在状态机之外**额外收紧**：只有 booked 允许走这条自助取消路径，
    checked_in 及以后的状态一律拒绝，需要撤销应由人工在阶段 4 的接诊接口处理。

    收紧用「白名单 + 仍调 assert_transition」而不是直接改转换矩阵：
    矩阵是全局唯一真相来源，为了一个自助端点的体验去放宽它，
    等于同时给未来所有人工退号入口开了口子。业务策略留在调用点，
    合法性校验照旧走矩阵，两层各管各的。

    **事务边界**：先 flush 把取消状态写进事务，再退号，最后一次性 commit。
    ``_release_slot`` 的重试路径不会回滚整事务（它只 expunge 过期缓存），
    所以这条顺序是安全的：任何一步抛错都由下面的 except 统一 rollback，
    不会出现「单子标了已取消但号没退」或反之的半态。

    为什么这里不像 create_appointment 那样在意「UPDATE 与 INSERT 谁先」：
    取消只改一行状态、不新增行，不存在唯一约束与扣号交错的问题。
    """
    appointment = await _get_appointment_for_requester(db, appointment_id, requester_user_id)

    if appointment.status != AppointmentStatus.BOOKED.value:
        raise InvalidStateTransitionError(
            f"当前状态「{appointment.status}」不支持自助取消，请联系导诊台处理",
            current=appointment.status,
            target=CANCEL_TARGET,
            entity="appointment",
        )

    # assert_transition 仍要调用：上面的收紧是业务策略，真正的合法性真相来源
    # 始终是转换矩阵。两层都过才写库，避免策略调整时漏掉矩阵校验。
    appointment.status = assert_transition(
        "appointment", appointment.status, CANCEL_TARGET
    )
    appointment.cancelled_at = utcnow()

    try:
        # 先 flush 再退号：两步在同一次 commit 里落地。
        await db.flush()
        await _release_slot(db, appointment.schedule_id, reason="patient_cancel")
        await db.commit()
    except Exception:
        # 兜住退号之外的失败（连接中断、约束冲突等）。
        # _release_slot 重试耗尽时抛 SlotConflictError 但并未回滚，
        # 所以这个 rollback 是**必需**的：它把已 flush 的 cancelled 状态一起撤销，
        # 保证「号没退成就别标取消」，患者下次仍能看到 booked 并重试。
        await db.rollback()
        raise

    await db.refresh(appointment)
    logger.info("预约已取消 appointment=%s", appointment.id)
    return appointment


async def check_in_appointment(
    db: AsyncSession, appointment_id: uuid.UUID
) -> Appointment:
    """到院签到：booked → checked_in。

    **本批只在 service 层实现，路由在阶段 4 接诊链路里挂出**。
    现在就把迁移逻辑写完的理由有两个：一是阶段 4 的「开始接诊」要求前置状态
    必须是 checked_in，服务层先具备能力才能让接诊功能直接复用；
    二是签到会写入 checked_in_at 这个时间戳，时间戳的取值口径（UTC）
    需要在本批就与其它字段统一，留到阶段 4 再补容易写出 naive datetime。

    签到不改动号源：号已经在挂号时被占用，患者到院只是把「占号」兑现成
    「就诊」，库存数字不变。爽约（no_show）才涉及是否释放号源，属阶段 4 议题。
    """
    appointment = await db.get(Appointment, appointment_id)
    if appointment is None:
        raise ResourceNotFoundError("预约不存在")

    appointment.status = assert_transition(
        "appointment",
        appointment.status,
        AppointmentStatus.CHECKED_IN.value,
    )
    appointment.checked_in_at = utcnow()
    await db.commit()
    await db.refresh(appointment)
    logger.info("预约已签到 appointment=%s", appointment.id)
    return appointment


# ==================== 读取 ====================


async def _get_appointment_for_requester(
    db: AsyncSession, appointment_id: uuid.UUID, requester_user_id: uuid.UUID
) -> Appointment:
    """取预约并做行级归属校验（患者只能碰自己的）。

    越权访问返回 404 而不是 403：403 会告诉攻击者「这个 ID 确实存在，
    只是不属于你」，等于用一个免费接口枚举系统中的预约单；404 与
    「记录不存在」不可区分，不泄露任何信息。这也与 knowledge 侧
    「非法/不存在的 ID 统一 404」的错误分支收敛原则一致。

    医生/管理员的可见范围由调用方决定：本函数只负责患者本人的严格隔离。
    """
    appointment = await db.get(Appointment, appointment_id)
    if appointment is None:
        raise ResourceNotFoundError("预约不存在")

    profile = (
        await db.execute(
            select(PatientProfile.id).where(PatientProfile.user_id == requester_user_id)
        )
    ).scalar_one_or_none()
    if profile is None or appointment.patient_id != profile:
        raise ResourceNotFoundError("预约不存在")
    return appointment


async def get_appointment_detail(
    db: AsyncSession,
    appointment_id: uuid.UUID,
    *,
    requester_user_id: uuid.UUID,
    requester_role: str,
) -> Appointment:
    """按角色取预约详情：患者限本人，医生限本人出诊，管理员放行。

    医生只能看自己出诊的预约——这是 rbac_matrix 里「行级数据隔离在 service 层
    实现」这条设计的落地：appointment:read 只控动作，不给具体行。
    管理员刻意**不放行临床内容读取之外的写操作**，此处只读，符合矩阵意图。
    """
    appointment = await db.get(Appointment, appointment_id)
    if appointment is None:
        raise ResourceNotFoundError("预约不存在")

    if requester_role == RoleName.ADMIN.value:
        return appointment

    if requester_role == RoleName.DOCTOR.value:
        if appointment.doctor_id != requester_user_id:
            raise ResourceNotFoundError("预约不存在")
        return appointment

    return await _get_appointment_for_requester(
        db, appointment_id, requester_user_id
    )


async def list_my_appointments(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    role_name: str,
    doctor_id: uuid.UUID | None = None,
    status_filter: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[Appointment], int]:
    """分页列出「我的预约」。

    患者视角按 patient_id 过滤，医生视角按 doctor_id 过滤（今日待诊列表），
    两种身份共用一套筛选参数，避免出现两个几乎相同的端点。
    """
    conditions = []
    if role_name == RoleName.PATIENT.value:
        profile = await _resolve_patient_profile(db, user_id)
        conditions.append(Appointment.patient_id == profile.id)
    elif role_name == RoleName.DOCTOR.value:
        conditions.append(Appointment.doctor_id == user_id)
    elif doctor_id is not None:
        # 管理员显式指定医生时才按其过滤，否则不加条件（全量视图）。
        conditions.append(Appointment.doctor_id == doctor_id)

    if status_filter:
        conditions.append(Appointment.status == status_filter)
    if date_from:
        conditions.append(Appointment.appointment_date >= date_from)
    if date_to:
        conditions.append(Appointment.appointment_date <= date_to)

    stmt = select(Appointment)
    if conditions:
        stmt = stmt.where(*conditions)

    total = int(
        (
            await db.execute(
                select(func.count()).select_from(stmt.subquery())
            )
        ).scalar_one()
    )

    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    items = list(
        (
            await db.execute(
                stmt.order_by(Appointment.appointment_date.desc(), Appointment.id)
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
        )
        .scalars()
        .all()
    )
    return items, total


# ==================== 序列化辅助 ====================


def _user_name_stmt(user_id: uuid.UUID):
    """按 ID 取用户名的语句（供序列化用）。"""
    return select(User.real_name).where(User.id == user_id)


async def serialize_appointment(
    db: AsyncSession, appointment: Appointment
) -> dict[str, object]:
    """预约 ORM → 可直接喂给 ``AppointmentOut`` 的字典。

    **不读 appointment.doctor / .schedule 关系**：两者都是 ``lazy="selectin"``，
    selectin 只在查询时预取；对一条已经通过 ``db.get()`` 主键取回的对象，
    首次访问关系属性会触发隐式懒加载，而 AsyncSession 里的同步 IO 会抛
    MissingGreenlet，把一次正常的详情读取变成 500（knowledge_service 里
    已因同类时序踩过这个坑）。因此这里显式发单列 SELECT。

    单列 SELECT 而非 JOIN 整表：详情接口一次只序列化一条，列表接口的条数
    由 page_size（上限 100）控制，多两次轻量往返比维护一份复合查询更划算。
    """
    doctor_name = (
        await db.execute(_user_name_stmt(appointment.doctor_id))
    ).scalar_one_or_none()

    department_name_row = (
        await db.execute(
            select(Department.name).where(Department.id == appointment.department_id)
        )
    ).scalar_one_or_none()

    return {
        "id": str(appointment.id),
        "patient_id": str(appointment.patient_id),
        "doctor_id": str(appointment.doctor_id),
        "doctor_name": doctor_name or "",
        "department_id": str(appointment.department_id),
        "department_name": department_name_row or "",
        "schedule_id": str(appointment.schedule_id),
        "appointment_date": appointment.appointment_date,
        "time_slot": appointment.time_slot,
        "status": appointment.status,
        "source": appointment.source,
        "chief_complaint": appointment.chief_complaint,
        "triage_summary": appointment.triage_summary,
        "triage_result_id": (
            str(appointment.triage_result_id)
            if appointment.triage_result_id is not None
            else None
        ),
        "registration_fee": appointment.registration_fee,
        "checked_in_at": appointment.checked_in_at,
        "cancelled_at": appointment.cancelled_at,
        "created_at": appointment.created_at,
        "updated_at": appointment.updated_at,
        "allowed_transitions": allowed_targets("appointment", appointment.status),
    }


__all__ = [
    "CANCEL_TARGET",
    "EFFECTIVE_APPOINTMENT_STATUSES",
    "SLOT_MAX_RETRIES",
    "SLOT_RETRY_BACKOFF_SECONDS",
    "cancel_appointment",
    "check_in_appointment",
    "create_appointment",
    "list_available_schedules",
    "list_doctors_with_open_slots",
    "list_my_appointments",
    "serialize_appointment",
    "today_local",
]
