"""号源查询与预约（挂号）服务/接口单元测试（文件型 SQLite，不依赖 PostgreSQL）。

**能跑的部分**：号源筛选与分页、可约医生聚合、乐观锁扣号的四种结局
（成功 / 竞争后重试成功 / 真满号 / 重试耗尽）、重复预约双层防呆、
取消与号源回退、状态机收紧、行级归属隔离、过期排班拒单、权限矩阵边界、
路由是否出现在 openapi。

**不能在这里跑的部分及原因**：

1. **真正的并发抢号**。SQLite 单连接串行执行，两个请求不会交错，
   ``UPDATE ... WHERE version=:v`` 的 rowcount 语义虽然与 PG 一致可用，
   但「同时到达」这件事本身造不出来。因此本文件用**逻辑等价**的方式覆盖：
   手工把库里的 version 推高（模拟别人抢先提交），再让被测函数拿着旧版本号去撞，
   从而确定性地走进 rowcount==0 的重读分支。这验证的是 service 的判定与重试代码，
   不声称验证了跨进程互斥——那属于容器集成冒烟。
2. **部分唯一索引**。``uq_appt_patient_slot_active`` 带 ``postgresql_where``，
   SQLite 不渲染该谓词（``table.create`` 只建表，根本不建这个非主键索引），
   所以数据库层的 IntegrityError 兜底在测试里无法触发；
   同名错误码由 service 的第一层查重给出，纯约束兜底由 PG 侧保证。
3. **ILIKE 大小写折叠**。SQLite 只对 ASCII 生效，故 keyword 用例只用中文检索，
   不断言英文大小写行为。
4. **FOR UPDATE / advisory lock**。本实现刻意不用行级锁（见 DoctorSchedule docstring），
   因此没有需要打桩的数据库专有特性。

Redis 在本链路里完全不参与（挂号不碰缓存），所以不需要 auth 测试那套双补丁。
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.exceptions import (
    DuplicateAppointmentError,
    InvalidStateTransitionError,
    ResourceNotFoundError,
    SlotConflictError,
    SlotFullError,
)
from app.imcs_main import app
from app.models.auth import DoctorSchedule, User
from app.models.base import Base, utcnow
from app.models.clinical import AppointmentStatus
from app.schemas.appointment import AppointmentCreate, ScheduleQuery
from app.services import appointment_service as svc
from tests.imcs_sqlite import (
    build_session_factory,
    error_code_of,
    seed_role_permissions,
    seed_roles,
)

# ==================== 建表 ====================

APPOINTMENT_TABLES: tuple[str, ...] = (
    "roles",
    "permissions",
    "role_permissions",
    "departments",
    "users",
    "patient_profiles",
    "doctor_schedules",
    "appointments",
)

ROLE_IDS: dict[str, int] = {
    "patient": 1,
    "doctor": 2,
    "lab_tech": 3,
    "cashier": 4,
    "pharmacist": 5,
    "admin": 6,
}


@pytest.fixture()
def engine(tmp_path):
    """文件型 SQLite：同步建表 + 异步被测会话共用同一份数据（沿用 knowledge 测试策略）。"""
    import tests.imcs_sqlite  # noqa: F401  - 注册 PGUUID 的 SQLite DDL 编译器

    db_path = tmp_path / "imcs_appointment.db"
    test_engine = sa.create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    tables = [Base.metadata.tables[name] for name in APPOINTMENT_TABLES]
    with test_engine.begin() as conn:
        for table in tables:
            table.create(conn, checkfirst=True)
        seed_roles(conn)
        # require_permission 要 JOIN permissions，空表会让非 admin 一律 403。
        seed_role_permissions(conn)
        # appointments.uq_appt_patient_slot_active 带 postgresql_where，
        # SQLite 建表时忽略谓词、退化成「(patient_id, schedule_id) 全量唯一」，
        # 退号后重挂会被误拦（生产 PG 上不存在此问题）。按 SQLite 语法重建为
        # 真正的部分唯一索引，让退号重挂用例测到与 PG 一致的约束语义。
        conn.execute(
            sa.text("DROP INDEX IF EXISTS uq_appt_patient_slot_active")
        )
        conn.execute(
            sa.text(
                "CREATE UNIQUE INDEX uq_appt_patient_slot_active "
                "ON appointments (patient_id, schedule_id) "
                "WHERE status NOT IN ('cancelled', 'no_show')"
            )
        )
    yield test_engine
    test_engine.dispose()


@pytest.fixture()
async def session_factory(engine):
    factory = build_session_factory(engine)
    yield factory
    await factory.kw["bind"].dispose()


@pytest.fixture()
async def db(session_factory) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


# ==================== 时间口径 ====================

_LOCAL_TZ = ZoneInfo("Asia/Shanghai")


def local_today() -> date:
    """东八区今天，与被测代码的 today_local() 同源，用于构造相对日期断言。"""
    return datetime.now(_LOCAL_TZ).date()


def next_workday(from_day: date) -> date:
    """从 from_day 次日起第一个工作日（周一到周五）。

    种子排班只覆盖工作日，测试若硬编码「明天」碰上周末就会拿不到号源，
    表现为随运行日期漂移的偶发失败。所有需要「一个真实存在的就诊日」的
    用例都走这个函数。
    """
    day = from_day + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


# ==================== 数据助手 ====================


def _insert(engine, table: str, values: dict[str, Any]) -> None:  # noqa: ANN001
    """直插一行（绕过 ORM，便于构造各种身份与状态的前提数据）。"""
    cols = list(values)
    quoted = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    with engine.begin() as conn:
        conn.execute(
            sa.text(f"INSERT INTO {table} ({quoted}) VALUES ({placeholders})"),
            values,
        )


def make_department(engine, code: str, name: str) -> uuid.UUID:  # noqa: ANN001
    did = uuid.uuid4()
    _insert(
        engine,
        "departments",
        {"id": did.hex, "name": name, "code": code, "description": None, "parent_id": None},
    )
    return did


def make_user(  # noqa: ANN001
    engine,
    *,
    role_name: str,
    username: str,
    real_name: str | None = None,
    department_id: uuid.UUID | None = None,
    title: str | None = None,
    specialty: str | None = None,
) -> uuid.UUID:
    uid = uuid.uuid4()
    _insert(
        engine,
        "users",
        {
            "id": uid.hex,
            "username": username,
            "password_hash": "x",
            "real_name": real_name or username,
            "phone": None,
            "email": None,
            "role_id": ROLE_IDS[role_name],
            "department_id": department_id.hex if department_id else None,
            "title": title,
            "specialty": specialty,
            "is_active": 1,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return uid


def make_profile(engine, user_id: uuid.UUID, *, gender: str = "M") -> uuid.UUID:  # noqa: ANN001
    pid = uuid.uuid4()
    _insert(
        engine,
        "patient_profiles",
        {
            "id": pid.hex,
            "user_id": user_id.hex,
            "gender": gender,
            "birth_date": date(1990, 1, 1),
            "medical_notes": None,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return pid


def make_schedule(  # noqa: ANN001
    engine,
    *,
    doctor_id: uuid.UUID,
    department_id: uuid.UUID,
    schedule_date: date,
    time_slot: str = "08:00-12:00",
    slot_period: str = "am",
    total_slots: int = 20,
    booked_slots: int = 0,
    version: int = 1,
    status: str = "open",
    registration_fee: int = 5000,
) -> uuid.UUID:
    sid = uuid.uuid4()
    _insert(
        engine,
        "doctor_schedules",
        {
            "id": sid.hex,
            "doctor_id": doctor_id.hex,
            "department_id": department_id.hex,
            "schedule_date": schedule_date.isoformat(),
            "time_slot": time_slot,
            "slot_period": slot_period,
            "total_slots": total_slots,
            "booked_slots": booked_slots,
            "version": version,
            "status": status,
            "registration_fee": registration_fee,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return sid


def read_schedule(engine, schedule_id: uuid.UUID) -> dict[str, Any]:  # noqa: ANN001
    """直读库里的号源行，断言「数据库真实发生了什么」而非 ORM 内存态。"""
    with engine.begin() as conn:
        row = conn.execute(
            sa.text(
                "SELECT booked_slots, total_slots, version, status "
                "FROM doctor_schedules WHERE id = :id"
            ),
            {"id": schedule_id.hex},
        ).fetchone()
    assert row is not None
    return dict(row._mapping)


def count_appointments(engine, schedule_id: uuid.UUID) -> int:  # noqa: ANN001
    with engine.begin() as conn:
        return int(
            conn.execute(
                sa.text("SELECT COUNT(*) FROM appointments WHERE schedule_id = :sid"),
                {"sid": schedule_id.hex},
            ).scalar_one()
        )


def bump_version_only(engine, schedule_id: uuid.UUID) -> None:  # noqa: ANN001
    """只推进 version 而不改 booked/status —— 模拟「别人刚提交过一次」。

    这是本文件制造乐观锁竞争的**唯一打桩手段**：不改数据语义，只让版本号
    与被测代码即将使用的期望值不一致，从而确定性地让那条 UPDATE 打空。
    """
    with engine.begin() as conn:
        conn.execute(
            sa.text("UPDATE doctor_schedules SET version = version + 1 WHERE id = :id"),
            {"id": schedule_id.hex},
        )


def insert_raw_appointment(  # noqa: ANN001
    engine,
    *,
    patient_id: uuid.UUID,
    doctor_id: uuid.UUID,
    department_id: uuid.UUID,
    schedule_id: uuid.UUID,
    appointment_date: date,
    time_slot: str = "08:00-12:00",
    status: str = AppointmentStatus.BOOKED.value,
    registration_fee: int = 5000,
) -> uuid.UUID:
    aid = uuid.uuid4()
    _insert(
        engine,
        "appointments",
        {
            "id": aid.hex,
            "patient_id": patient_id.hex,
            "doctor_id": doctor_id.hex,
            "department_id": department_id.hex,
            "schedule_id": schedule_id.hex,
            "appointment_date": appointment_date.isoformat(),
            "time_slot": time_slot,
            "status": status,
            "source": "self",
            "registration_fee": registration_fee,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return aid


@pytest.fixture()
def world(engine) -> dict[str, Any]:
    """一套最小可用的科室 / 医生 / 患者 / 排班数据。

    三个医生的职称与专长各不相同且分属两个科室，是为了让
    「按科室聚合」「keyword 搜专长」「按职称过滤」三类用例互不干扰。
    排班刻意铺齐五种形态：有余号 open、满号 full、停诊 closed、已过期、
    以及另一科室的一个 open，覆盖号源筛选的全部分支。
    """
    cardio = make_department(engine, "CARDIO", "心血管内科")
    neuro = make_department(engine, "NEURO", "神经内科")

    doctor_chief = make_user(
        engine,
        role_name="doctor",
        username="doctor01",
        real_name="张三",
        department_id=cardio,
        title="主任医师",
        specialty="高血压、冠心病、心律失常",
    )
    doctor_deputy = make_user(
        engine,
        role_name="doctor",
        username="doctor02",
        real_name="李四",
        department_id=cardio,
        title="副主任医师",
        specialty="呼吸道感染、胃肠疾病",
    )
    doctor_attending = make_user(
        engine,
        role_name="doctor",
        username="doctor03",
        real_name="王五",
        department_id=neuro,
        title="主治医师",
        specialty="头痛、头晕、脑血管病",
    )

    patient01 = make_user(engine, role_name="patient", username="patient01", real_name="患者一")
    patient02 = make_user(engine, role_name="patient", username="patient02", real_name="患者二")
    profile01 = make_profile(engine, patient01)
    profile02 = make_profile(engine, patient02, gender="F")

    # 有医生账号但没有患者档案：用于验证「查不到档案要明确报错而不是撞外键」
    doctor_no_profile = make_user(engine, role_name="doctor", username="doctor_x", real_name="赵六")

    today = local_today()
    target_day = next_workday(today)

    return {
        "engine": engine,
        "today": today,
        "target_day": target_day,
        "yesterday": today - timedelta(days=1),
        "far_day": next_workday(today) + timedelta(days=40),
        "cardio": cardio,
        "neuro": neuro,
        "doctor_chief": doctor_chief,
        "doctor_deputy": doctor_deputy,
        "doctor_attending": doctor_attending,
        "doctor_no_profile": doctor_no_profile,
        "patient01": patient01,
        "patient02": patient02,
        "profile01": profile01,
        "profile02": profile02,
        "sched_am": make_schedule(
            engine,
            doctor_id=doctor_chief,
            department_id=cardio,
            schedule_date=target_day,
            time_slot="08:00-12:00",
            slot_period="am",
            total_slots=20,
            registration_fee=5000,
        ),
        "sched_pm": make_schedule(
            engine,
            doctor_id=doctor_chief,
            department_id=cardio,
            schedule_date=target_day,
            time_slot="14:00-17:30",
            slot_period="pm",
            total_slots=15,
            registration_fee=5000,
        ),
        "sched_neuro": make_schedule(
            engine,
            doctor_id=doctor_attending,
            department_id=neuro,
            schedule_date=target_day,
            time_slot="08:00-12:00",
            slot_period="am",
            total_slots=30,
            registration_fee=2500,
        ),
        "sched_full": make_schedule(
            engine,
            doctor_id=doctor_deputy,
            department_id=cardio,
            schedule_date=target_day,
            time_slot="08:00-12:00",
            slot_period="am",
            total_slots=25,
            booked_slots=25,
            status="full",
            registration_fee=3500,
        ),
        "sched_closed": make_schedule(
            engine,
            doctor_id=doctor_deputy,
            department_id=cardio,
            schedule_date=target_day,
            time_slot="14:00-17:30",
            slot_period="pm",
            total_slots=20,
            status="closed",
            registration_fee=3500,
        ),
        "sched_expired": make_schedule(
            engine,
            doctor_id=doctor_deputy,
            department_id=cardio,
            schedule_date=today - timedelta(days=1),
            time_slot="08:00-12:00",
            slot_period="am",
            total_slots=25,
            registration_fee=3500,
        ),
    }


def as_user(user_id: uuid.UUID, username: str, role_name: str) -> User:
    """构造与 deps.get_current_user 同形态的轻量 User（含 _role_name）。"""
    user = User(
        id=user_id,
        username=username,
        real_name=username,
        password_hash="",
        role_id=ROLE_IDS[role_name],
        is_active=True,
    )
    user._role_name = role_name  # type: ignore[attr-defined]
    return user


# ==================== schemas 校验（纯内存）====================


class TestSchemas:
    def test_schedule_query_defaults(self) -> None:
        q = ScheduleQuery()
        assert q.page == 1
        assert q.page_size == 20
        assert q.status == "open", "缺省应只出可约号源"
        assert q.include_expired is False

    def test_resolved_window_default_14_days(self) -> None:
        anchor = date(2026, 3, 1)
        start, end = ScheduleQuery().resolved_window(anchor)
        assert start == anchor
        assert end == anchor + timedelta(days=13), "默认窗口应为含端点共 14 天"

    def test_resolved_window_explicit_dates(self) -> None:
        q = ScheduleQuery(start_date=date(2026, 3, 2), end_date=date(2026, 3, 5))
        assert q.resolved_window(date(2026, 1, 1)) == (date(2026, 3, 2), date(2026, 3, 5))

    def test_resolved_window_rejects_inverted_range(self) -> None:
        q = ScheduleQuery(start_date=date(2026, 3, 5), end_date=date(2026, 3, 2))
        with pytest.raises(ValueError, match="end_date"):
            q.resolved_window(date(2026, 1, 1))

    def test_create_request_accepts_uuid_triage_id(self) -> None:
        tid = str(uuid.uuid4())
        req = AppointmentCreate(schedule_id=str(uuid.uuid4()), triage_result_id=tid)
        assert req.triage_result_id == tid

    def test_create_request_rejects_bad_schedule_id(self) -> None:
        with pytest.raises(ValueError):
            AppointmentCreate(schedule_id="not-a-uuid")

    def test_create_request_rejects_bad_triage_id(self) -> None:
        with pytest.raises(ValueError):
            AppointmentCreate(schedule_id=str(uuid.uuid4()), triage_result_id="oops")

    def test_create_request_normalizes_blank_triage_id(self) -> None:
        req = AppointmentCreate(schedule_id=str(uuid.uuid4()), triage_result_id="   ")
        assert req.triage_result_id is None, "空白串应归一化为 None 而非留给 service 判断"

    def test_page_size_upper_bound(self) -> None:
        with pytest.raises(ValueError):
            ScheduleQuery(page_size=101)


# ==================== 号源查询 ====================


class TestScheduleQueryService:
    async def test_filters_by_department_and_open_only(self, db: AsyncSession, world) -> None:
        rows, total, _ = await svc.list_available_schedules(
            db, ScheduleQuery(department_code="CARDIO")
        )
        assert total == 2, "心内只有 am/pm 两条 open，full/closed/过期都不该出现"
        assert {r["department_code"] for r in rows} == {"CARDIO"}
        assert all(r["status"] == "open" for r in rows)

    async def test_remaining_slots_computed(self, db: AsyncSession, world) -> None:
        rows, _, _ = await svc.list_available_schedules(
            db, ScheduleQuery(doctor_id=str(world["doctor_chief"]))
        )
        am = next(r for r in rows if r["slot_period"] == "am")
        assert am["remaining_slots"] == 20
        assert am["total_slots"] - am["booked_slots"] == am["remaining_slots"]

    async def test_excludes_expired_by_default(self, db: AsyncSession, world) -> None:
        _, total, _ = await svc.list_available_schedules(
            db, ScheduleQuery(department_code="CARDIO")
        )
        assert total == 2
        _, total_inc, _ = await svc.list_available_schedules(
            db, ScheduleQuery(status=None, include_expired=True)
        )
        assert total_inc == 5, "打开 include_expired 后 full/closed/昨天的号都应出现"

    async def test_keyword_matches_name_title_specialty(self, db: AsyncSession, world) -> None:
        by_name, n1, _ = await svc.list_available_schedules(db, ScheduleQuery(keyword="张三"))
        assert n1 == 2 and all(r["doctor_name"] == "张三" for r in by_name)

        _, n2, _ = await svc.list_available_schedules(db, ScheduleQuery(keyword="脑血管"))
        assert n2 == 1, "专长关键词应命中王五的一个半天"

        _, n3, _ = await svc.list_available_schedules(db, ScheduleQuery(keyword="主治医师"))
        assert n3 == 1, "职称关键词同样参与匹配"

        _, n4, _ = await svc.list_available_schedules(db, ScheduleQuery(keyword="不存在的人"))
        assert n4 == 0

    async def test_title_filter_exact_match(self, db: AsyncSession, world) -> None:
        _, total, _ = await svc.list_available_schedules(
            db, ScheduleQuery(title="主任医师", department_code="CARDIO")
        )
        assert total == 2

    async def test_slot_period_filter(self, db: AsyncSession, world) -> None:
        rows, total, _ = await svc.list_available_schedules(
            db, ScheduleQuery(slot_period="pm", department_code="CARDIO")
        )
        assert total == 1 and rows[0]["slot_period"] == "pm"

    async def test_pagination_slices_and_reports_total(self, db: AsyncSession, world) -> None:
        p1, total, _ = await svc.list_available_schedules(
            db, ScheduleQuery(status=None, page=1, page_size=2)
        )
        p2, _, _ = await svc.list_available_schedules(
            db, ScheduleQuery(status=None, page=2, page_size=2)
        )
        assert total == 5
        ids1 = {str(r["id"]) for r in p1}
        ids2 = {str(r["id"]) for r in p2}
        assert len(ids1) == 2 and len(ids2) == 2
        assert ids1 & ids2 == set(), "翻页不得重叠"
        assert len(ids1 | ids2) == 4

    async def test_out_of_range_page_returns_empty_not_error(
        self, db: AsyncSession, world
    ) -> None:
        rows, total, _ = await svc.list_available_schedules(
            db, ScheduleQuery(status=None, page=9, page_size=10)
        )
        assert rows == [] and total == 5, "越界页返回空列表，总数仍如实报告"

    async def test_date_window_excludes_far_future(self, db: AsyncSession, engine, world) -> None:
        make_schedule(
            engine,
            doctor_id=world["doctor_attending"],
            department_id=world["neuro"],
            schedule_date=world["far_day"],
        )
        _, total, (start, end) = await svc.list_available_schedules(
            db, ScheduleQuery(department_code="NEURO")
        )
        assert total == 1, "40 天外的排班落在默认 14 天窗口之外"
        assert start <= world["target_day"] <= end

    async def test_doctors_by_department_aggregates(self, db: AsyncSession, world) -> None:
        doctors, (start, end) = await svc.list_doctors_with_open_slots(
            db, ScheduleQuery(department_code="CARDIO")
        )
        assert start <= end
        assert len(doctors) == 1, "李四的两个半天分别是 full/closed，不构成可约医生"
        item = doctors[0]
        assert item["doctor_name"] == "张三"
        assert item["schedule_count"] == 2
        assert item["total_remaining"] == 35, "20 + 15"
        assert item["available_dates"] == [world["target_day"]]
        assert item["min_registration_fee"] == 5000

    async def test_doctors_without_department_filter(self, db: AsyncSession, world) -> None:
        doctors, _ = await svc.list_doctors_with_open_slots(db, ScheduleQuery())
        assert {d["doctor_name"] for d in doctors} == {"张三", "王五"}

    async def test_doctors_by_department_keyword(self, db: AsyncSession, world) -> None:
        doctors, _ = await svc.list_doctors_with_open_slots(db, ScheduleQuery(keyword="头晕"))
        assert [d["doctor_name"] for d in doctors] == ["王五"]

    async def test_doctors_excludes_non_doctor_roles(self, db: AsyncSession, engine, world) -> None:
        """角色过滤必须生效：给患者也塞一条排班行，他不应出现在可约医生里。"""
        make_schedule(
            engine,
            doctor_id=world["patient01"],
            department_id=world["cardio"],
            schedule_date=world["target_day"],
            time_slot="15:00-16:00",
        )
        doctors, _ = await svc.list_doctors_with_open_slots(
            db, ScheduleQuery(department_code="CARDIO")
        )
        assert [d["doctor_name"] for d in doctors] == ["张三"]

    async def test_doctors_excludes_fully_booked_doctor(self, db: AsyncSession, engine, world) -> None:
        """医生只剩满号 → 整体消失，而不是显示一个「有号但挂不上」的条目。"""
        make_schedule(
            engine,
            doctor_id=world["doctor_attending"],
            department_id=world["neuro"],
            schedule_date=world["target_day"],
            time_slot="15:00-16:00",
            total_slots=5,
            booked_slots=5,
            status="full",
        )
        # 先确认原本可约
        before, _ = await svc.list_doctors_with_open_slots(db, ScheduleQuery(department_code="NEURO"))
        assert len(before) == 1
        # 把唯一的 open 号源也订满
        with engine.begin() as conn:
            conn.execute(
                sa.text(
                    "UPDATE doctor_schedules SET booked_slots = total_slots, status='full' "
                    "WHERE doctor_id = :d"
                ),
                {"d": world["doctor_attending"].hex},
            )
        after, _ = await svc.list_doctors_with_open_slots(db, ScheduleQuery(department_code="NEURO"))
        assert after == []


# ==================== 创建预约（正常路径）====================


class TestCreateAppointment:
    async def test_success_deducts_slot_and_creates_record(
        self, db: AsyncSession, engine, world
    ) -> None:
        appt = await svc.create_appointment(
            db,
            user_id=world["patient01"],
            schedule_id=world["sched_am"],
            chief_complaint="近一周活动后气短",
        )
        assert appt.status == AppointmentStatus.BOOKED.value
        assert appt.source == "self"
        assert appt.patient_id == world["profile01"], "预约挂在档案上而非 user"
        assert appt.doctor_id == world["doctor_chief"]
        assert appt.department_id == world["cardio"]
        assert appt.schedule_id == world["sched_am"]
        assert appt.appointment_date == world["target_day"]
        assert appt.time_slot == "08:00-12:00"
        assert appt.registration_fee == 5000, "费用应取号源快照"
        assert appt.chief_complaint == "近一周活动后气短"

        raw = read_schedule(engine, world["sched_am"])
        assert raw["booked_slots"] == 1, "号源应真实减一"
        assert raw["version"] == 2, "乐观锁版本应推进"

    async def test_triage_referral_sets_source_and_link_id(self, db: AsyncSession, world) -> None:
        tid = uuid.uuid4()
        appt = await svc.create_appointment(
            db,
            user_id=world["patient01"],
            schedule_id=world["sched_am"],
            triage_result_id=str(tid),
            triage_summary="AI 建议心血管内科，紧急度 routine",
        )
        assert appt.source == "triage_referral"
        assert appt.triage_result_id == tid
        assert appt.triage_summary == "AI 建议心血管内科，紧急度 routine"

    async def test_dangling_triage_id_is_accepted(self, db: AsyncSession, world) -> None:
        """悬空 UUID 也要收下：本模块不 import triage，无法也不该校验存在性。"""
        appt = await svc.create_appointment(
            db,
            user_id=world["patient01"],
            schedule_id=world["sched_pm"],
            triage_result_id=str(uuid.uuid4()),
        )
        assert appt.triage_result_id is not None

    async def test_two_patients_can_book_same_schedule(self, db: AsyncSession, engine, world) -> None:
        a1 = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        a2 = await svc.create_appointment(
            db, user_id=world["patient02"], schedule_id=world["sched_am"]
        )
        assert a1.id != a2.id
        assert read_schedule(engine, world["sched_am"])["booked_slots"] == 2

    async def test_same_patient_can_book_two_departures_same_morning(
        self, db: AsyncSession, engine, world
    ) -> None:
        """防呆粒度是 (patient, schedule)，不是 (patient, date+period)。

        同一上午挂两个不同科室是合法行为，必须放行——这条断言直接钉住
        所选判据口径，防止日后有人「顺手」改成按日期拦截。
        """
        await svc.create_appointment(db, user_id=world["patient01"], schedule_id=world["sched_am"])
        second = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_neuro"]
        )
        assert second.department_id == world["neuro"]
        assert read_schedule(engine, world["sched_neuro"])["booked_slots"] == 1

    async def test_missing_profile_raises_resource_not_found(self, db: AsyncSession, world) -> None:
        with pytest.raises(ResourceNotFoundError) as exc:
            await svc.create_appointment(
                db, user_id=world["doctor_no_profile"], schedule_id=world["sched_am"]
            )
        assert "档案" in exc.value.message
        assert exc.value.code == "resource_not_found"


# ==================== 重复预约防呆 ====================


class TestDuplicateGuard:
    async def test_second_booking_same_schedule_rejected(
        self, db: AsyncSession, engine, world
    ) -> None:
        await svc.create_appointment(db, user_id=world["patient01"], schedule_id=world["sched_am"])
        with pytest.raises(DuplicateAppointmentError) as exc:
            await svc.create_appointment(
                db, user_id=world["patient01"], schedule_id=world["sched_am"]
            )
        assert exc.value.code == "appointment_duplicate"
        assert "appointment_conflict_id" in exc.value.detail

        raw = read_schedule(engine, world["sched_am"])
        assert raw["booked_slots"] == 1, "重复请求不得二次扣号"
        assert raw["version"] == 2
        assert count_appointments(engine, world["sched_am"]) == 1

    async def test_cancelled_then_rebook_allowed(self, db: AsyncSession, engine, world) -> None:
        """取消后同一号源可以再挂——这正是部分唯一索引排除 cancelled 的意义。"""
        appt = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        await svc.cancel_appointment(db, appt.id, requester_user_id=world["patient01"])
        again = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        assert again.status == AppointmentStatus.BOOKED.value
        assert read_schedule(engine, world["sched_am"])["booked_slots"] == 1

    async def test_preexisting_effective_row_triggers_guard(self, db: AsyncSession, world) -> None:
        """第一层查重要能识别「库里已有但本会话没见过」的有效预约。"""
        insert_raw_appointment(
            world["engine"],
            patient_id=world["profile01"],
            doctor_id=world["doctor_chief"],
            department_id=world["cardio"],
            schedule_id=world["sched_am"],
            appointment_date=world["target_day"],
        )
        with pytest.raises(DuplicateAppointmentError):
            await svc.create_appointment(db, user_id=world["patient01"], schedule_id=world["sched_am"])

    @pytest.mark.parametrize(
        "released_status",
        [AppointmentStatus.CANCELLED.value, AppointmentStatus.NO_SHOW.value],
    )
    async def test_released_statuses_do_not_block_rebooking(
        self, db: AsyncSession, world, released_status: str
    ) -> None:
        """cancelled / no_show 不占号：与索引谓词 ``NOT IN (...)`` 严格对齐。"""
        insert_raw_appointment(
            world["engine"],
            patient_id=world["profile01"],
            doctor_id=world["doctor_chief"],
            department_id=world["cardio"],
            schedule_id=world["sched_am"],
            appointment_date=world["target_day"],
            status=released_status,
        )
        appt = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        assert appt.status == AppointmentStatus.BOOKED.value


# ==================== 满号 / 关闭 / 过期 ====================


class TestSlotAvailability:
    async def test_full_schedule_raises_slot_full_without_side_effects(
        self, db: AsyncSession, engine, world
    ) -> None:
        with pytest.raises(SlotFullError) as exc:
            await svc.create_appointment(
                db, user_id=world["patient01"], schedule_id=world["sched_full"]
            )
        assert exc.value.code == "appointment_slot_full"
        raw = read_schedule(engine, world["sched_full"])
        assert raw["booked_slots"] == 25 and raw["version"] == 1
        assert count_appointments(engine, world["sched_full"]) == 0

    async def test_last_slot_marks_schedule_full(self, db: AsyncSession, engine, world) -> None:
        """抢到最后一个号 → status 联动为 full（否则列表会显示挂不上的号）。"""
        sid = make_schedule(
            engine,
            doctor_id=world["doctor_attending"],
            department_id=world["neuro"],
            schedule_date=world["target_day"],
            time_slot="09:00-10:00",
            total_slots=1,
            registration_fee=2500,
        )
        await svc.create_appointment(db, user_id=world["patient01"], schedule_id=sid)
        raw = read_schedule(engine, sid)
        assert raw["booked_slots"] == 1
        assert raw["status"] == "full"

    async def test_closed_schedule_rejected(self, db: AsyncSession, engine, world) -> None:
        with pytest.raises(SlotFullError):
            await svc.create_appointment(
                db, user_id=world["patient01"], schedule_id=world["sched_closed"]
            )
        assert count_appointments(engine, world["sched_closed"]) == 0

    async def test_expired_schedule_rejected_as_404(self, db: AsyncSession, engine, world) -> None:
        """过期排班对患者的可观测结果与「不存在」相同，统一报 404。"""
        with pytest.raises(ResourceNotFoundError):
            await svc.create_appointment(
                db, user_id=world["patient01"], schedule_id=world["sched_expired"]
            )
        assert read_schedule(engine, world["sched_expired"])["booked_slots"] == 0

    async def test_missing_schedule_rejected(self, db: AsyncSession, world) -> None:
        with pytest.raises(ResourceNotFoundError):
            await svc.create_appointment(
                db, user_id=world["patient01"], schedule_id=uuid.uuid4()
            )


# ==================== 乐观锁并发（逻辑等价）====================


class TestOptimisticLocking:
    async def test_version_mismatch_then_retry_succeeds(
        self, db: AsyncSession, engine, world, monkeypatch
    ) -> None:
        """竞争一次后重试成功：库里 version 已被别人推高，service 应重读并重试。"""
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(svc.asyncio, "sleep", fake_sleep)
        original_get = db.get
        bumped = {"done": False}

        async def racing_get_once(entity, pk, *args, **kwargs):  # noqa: ANN001
            obj = await original_get(entity, pk, *args, **kwargs)
            # 竞争方必须用同一会话写入：SQLite 单写者，第二条连接在事务中途
            # UPDATE 会撞 database is locked（PG 上才是正常并发）。会话内推高
            # version 后，被测 UPDATE 的 WHERE version 匹配失败，竞争语义等价。
            if entity is DoctorSchedule and not bumped["done"]:
                bumped["done"] = True
                await db.execute(
                    sa.text(
                        "UPDATE doctor_schedules SET version = version + 1 "
                        "WHERE id = :id"
                    ),
                    {"id": pk.hex},
                )
            return obj

        monkeypatch.setattr(db, "get", racing_get_once)

        appt = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        assert appt.status == AppointmentStatus.BOOKED.value
        raw = read_schedule(engine, world["sched_am"])
        assert raw["booked_slots"] == 1, "重试成功后只扣一次号"
        assert raw["version"] == 3, "外部推高一次 + 本方成功一次"
        assert sleeps == [svc.SLOT_RETRY_BACKOFF_SECONDS[0]], "首次退避即成功，不应继续等"

    async def test_retry_exhausted_raises_slot_conflict(
        self, db: AsyncSession, engine, world, monkeypatch
    ) -> None:
        """持续竞争：每次重读都发现 version 变了 → 耗尽重试后报 SlotConflictError。

        打桩方式是把 ``db.get`` 包一层：每次读取号源后立刻在库里推高 version，
        于是被测代码永远追不上版本号，等价于「对手提交速度极快」。
        """
        async def fake_sleep(seconds: float) -> None:
            del seconds

        monkeypatch.setattr(svc.asyncio, "sleep", fake_sleep)
        original_get = db.get

        async def racing_get(entity, pk, *args, **kwargs):  # noqa: ANN001
            obj = await original_get(entity, pk, *args, **kwargs)
            if entity is DoctorSchedule:
                # 同会话内推高（SQLite 单写者，跨连接会撞写锁），等价于
                # 「对手提交极快，每次重读都被抢先」
                await db.execute(
                    sa.text(
                        "UPDATE doctor_schedules SET version = version + 1 "
                        "WHERE id = :id"
                    ),
                    {"id": pk.hex},
                )
            return obj

        monkeypatch.setattr(db, "get", racing_get)

        with pytest.raises(SlotConflictError) as exc:
            await svc.create_appointment(
                db, user_id=world["patient01"], schedule_id=world["sched_am"]
            )
        assert exc.value.code == "appointment_slot_conflict"
        assert count_appointments(engine, world["sched_am"]) == 0, "抢号失败不得留下预约"
        assert read_schedule(engine, world["sched_am"])["booked_slots"] == 0, (
            "抢号失败不得扣掉自己的号（外部推高的 version 不算扣号）"
        )

    async def test_real_full_via_update_condition(
        self, db: AsyncSession, engine, world, monkeypatch
    ) -> None:
        """绕过前置判定、直接让 UPDATE 因 booked>=total 打空 → 报满号且不扣号。

        做法：把号源日期改成今天（能通过过期检查），status 保持 open，
        但 booked 已等于 total —— 前置检查只看 closed 与日期，不看余量，
        因此必然走到那条 UPDATE 上，由 ``booked_slots < total_slots`` 条件拦下。
        """
        sid = make_schedule(
            engine,
            doctor_id=world["doctor_chief"],
            department_id=world["cardio"],
            schedule_date=world["today"],
            time_slot="16:00-17:00",
            total_slots=3,
            booked_slots=3,
            status="open",
        )

        async def fake_sleep(seconds: float) -> None:
            del seconds

        monkeypatch.setattr(svc.asyncio, "sleep", fake_sleep)

        with pytest.raises(SlotFullError):
            await svc.create_appointment(db, user_id=world["patient01"], schedule_id=sid)
        raw = read_schedule(engine, sid)
        assert raw["booked_slots"] == 3 and raw["version"] == 1
        assert count_appointments(engine, sid) == 0

    async def test_backoff_sequence_is_bounded(self) -> None:
        """重试参数契约：≤3 次、指数递增、总等待不超过半秒。"""
        assert svc.SLOT_MAX_RETRIES == 3
        assert len(svc.SLOT_RETRY_BACKOFF_SECONDS) == svc.SLOT_MAX_RETRIES
        assert svc.SLOT_RETRY_BACKOFF_SECONDS == (0.05, 0.10, 0.20)
        assert sum(svc.SLOT_RETRY_BACKOFF_SECONDS) < 0.5

    async def test_sequential_bookings_never_oversell(
        self, db: AsyncSession, engine, world
    ) -> None:
        """total=2 的号源被两人各占一格后，第三人必被挡住。

        SQLite 串行，这里验证的是「booked<total 条件确实挡住了超卖」，
        而不是并行时序——后者由那条 WHERE 子句本身在 PG 上保证。
        """
        sid = make_schedule(
            engine,
            doctor_id=world["doctor_chief"],
            department_id=world["cardio"],
            schedule_date=world["target_day"],
            time_slot="10:00-11:00",
            total_slots=2,
        )
        await svc.create_appointment(db, user_id=world["patient01"], schedule_id=sid)
        await svc.create_appointment(db, user_id=world["patient02"], schedule_id=sid)
        assert read_schedule(engine, sid)["booked_slots"] == 2

        third = make_user(engine, role_name="patient", username="patient03", real_name="患者三")
        make_profile(engine, third)
        with pytest.raises(SlotFullError):
            await svc.create_appointment(db, user_id=third, schedule_id=sid)
        assert read_schedule(engine, sid)["booked_slots"] == 2, "第三次尝试不得改动计数"
        assert count_appointments(engine, sid) == 2


# ==================== 取消与号源回退 ====================


class TestCancel:
    async def test_cancel_restores_slot(self, db: AsyncSession, engine, world) -> None:
        appt = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        assert read_schedule(engine, world["sched_am"])["booked_slots"] == 1

        cancelled = await svc.cancel_appointment(
            db, appt.id, requester_user_id=world["patient01"]
        )
        assert cancelled.status == AppointmentStatus.CANCELLED.value
        assert cancelled.cancelled_at is not None
        raw = read_schedule(engine, world["sched_am"])
        assert raw["booked_slots"] == 0, "号源必须归还"
        assert raw["version"] == 3, "扣号一次 + 退号一次，各推进一个版本"

    async def test_cancel_reopens_full_schedule(self, db: AsyncSession, engine, world) -> None:
        """曾满后释放出余号 → status 回到 open，否则这个号在列表里永久隐身。"""
        sid = make_schedule(
            engine,
            doctor_id=world["doctor_attending"],
            department_id=world["neuro"],
            schedule_date=world["target_day"],
            time_slot="11:00-12:00",
            total_slots=1,
        )
        appt = await svc.create_appointment(db, user_id=world["patient01"], schedule_id=sid)
        assert read_schedule(engine, sid)["status"] == "full"

        await svc.cancel_appointment(db, appt.id, requester_user_id=world["patient01"])
        raw = read_schedule(engine, sid)
        assert raw["booked_slots"] == 0
        assert raw["status"] == "open"

    async def test_cancel_checked_in_rejected_not_500(self, db: AsyncSession, engine, world) -> None:
        """checked_in 之后不允许自助取消，且必须是带 detail 的 409 而非 500。"""
        appt = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        await svc.check_in_appointment(db, appt.id)

        with pytest.raises(InvalidStateTransitionError) as exc:
            await svc.cancel_appointment(db, appt.id, requester_user_id=world["patient01"])
        err = exc.value
        assert err.code == "state_transition_invalid"
        assert err.status_code == 409
        assert err.detail["current_status"] == AppointmentStatus.CHECKED_IN.value
        assert err.detail["target_status"] == AppointmentStatus.CANCELLED.value
        assert err.detail["entity"] == "appointment"
        assert read_schedule(engine, world["sched_am"])["booked_slots"] == 1, (
            "取消被拒时不得把号退掉"
        )

    async def test_double_cancel_rejected(self, db: AsyncSession, engine, world) -> None:
        appt = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        await svc.cancel_appointment(db, appt.id, requester_user_id=world["patient01"])
        with pytest.raises(InvalidStateTransitionError):
            await svc.cancel_appointment(db, appt.id, requester_user_id=world["patient01"])
        assert read_schedule(engine, world["sched_am"])["booked_slots"] == 0, (
            "重复取消不得二次退号"
        )

    async def test_cancel_completed_rejected(self, db: AsyncSession, engine, world) -> None:
        """completed 是终态：连人工都不能取消，更不该让自助入口放行。"""
        appt = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        # 前置：把单子推到 completed 终态。create_appointment 已 commit，
        # 被测会话此刻不持行锁，第二条连接可以安全写入。
        with engine.begin() as conn:
            conn.execute(
                sa.text("UPDATE appointments SET status='completed' WHERE id=:i"),
                {"i": appt.id.hex},
            )
        # 外部连接改了库里的行，被测会话的 identity map 里还是旧的 booked 对象。
        # 用 await db.refresh 而不是 expire_all：后者把对象整体过期后，
        # 测试里下一次访问 appt.id 会在异步会话中触发同步 lazy load，
        # 抛 MissingGreenlet——refresh 是 await 的，读回新状态且不留过期对象。
        await db.refresh(appt)
        assert appt.status == AppointmentStatus.COMPLETED.value
        with pytest.raises(InvalidStateTransitionError):
            await svc.cancel_appointment(db, appt.id, requester_user_id=world["patient01"])


# ==================== 签到 ====================


class TestCheckIn:
    async def test_check_in_transitions_and_stamps(self, db: AsyncSession, world) -> None:
        appt = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        checked = await svc.check_in_appointment(db, appt.id)
        assert checked.status == AppointmentStatus.CHECKED_IN.value
        assert checked.checked_in_at is not None
        # 「全项目禁止 naive datetime」的口径由 PG 集成测试验证：
        # SQLite 的 DATETIME 列不存时区偏移，utcnow() 写入后读回必然 naive，
        # 在这里断言 tz 感知测的是 SQLite 列类型而不是业务代码。

    async def test_check_in_does_not_touch_slots(self, db: AsyncSession, engine, world) -> None:
        appt = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        before = read_schedule(engine, world["sched_am"])
        await svc.check_in_appointment(db, appt.id)
        assert read_schedule(engine, world["sched_am"]) == before, "到院签到不改变库存"

    async def test_double_check_in_rejected(self, db: AsyncSession, world) -> None:
        appt = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        await svc.check_in_appointment(db, appt.id)
        with pytest.raises(InvalidStateTransitionError):
            await svc.check_in_appointment(db, appt.id)

    async def test_check_in_missing_appointment(self, db: AsyncSession) -> None:
        with pytest.raises(ResourceNotFoundError):
            await svc.check_in_appointment(db, uuid.uuid4())


# ==================== 读取与行级隔离 ====================


class TestReadIsolation:
    async def test_list_scopes_to_patient(self, db: AsyncSession, world) -> None:
        await svc.create_appointment(db, user_id=world["patient01"], schedule_id=world["sched_am"])
        items, total = await svc.list_my_appointments(
            db, user_id=world["patient02"], role_name="patient"
        )
        assert total == 0 and items == [], "patient02 不该看到 patient01 的单子"

        items, total = await svc.list_my_appointments(
            db, user_id=world["patient01"], role_name="patient"
        )
        assert total == 1 and items[0].patient_id == world["profile01"]

    async def test_doctor_list_scopes_to_own_schedule(self, db: AsyncSession, world) -> None:
        await svc.create_appointment(db, user_id=world["patient01"], schedule_id=world["sched_am"])
        _, mine = await svc.list_my_appointments(
            db, user_id=world["doctor_chief"], role_name="doctor"
        )
        assert mine == 1
        _, others = await svc.list_my_appointments(
            db, user_id=world["doctor_attending"], role_name="doctor"
        )
        assert others == 0

    async def test_detail_rejects_other_patient(self, db: AsyncSession, world) -> None:
        appt = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        with pytest.raises(ResourceNotFoundError):
            await svc.get_appointment_detail(
                db, appt.id, requester_user_id=world["patient02"], requester_role="patient"
            )

    async def test_detail_allows_own_doctor_and_admin(self, db: AsyncSession, world) -> None:
        appt = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        by_doctor = await svc.get_appointment_detail(
            db, appt.id, requester_user_id=world["doctor_chief"], requester_role="doctor"
        )
        assert by_doctor.id == appt.id
        by_admin = await svc.get_appointment_detail(
            db, appt.id, requester_user_id=world["doctor_no_profile"], requester_role="admin"
        )
        assert by_admin.id == appt.id

    async def test_detail_rejects_other_doctor(self, db: AsyncSession, world) -> None:
        appt = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        with pytest.raises(ResourceNotFoundError):
            await svc.get_appointment_detail(
                db, appt.id, requester_user_id=world["doctor_attending"], requester_role="doctor"
            )

    async def test_cancel_by_other_patient_rejected(self, db: AsyncSession, engine, world) -> None:
        appt = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        with pytest.raises(ResourceNotFoundError):
            await svc.cancel_appointment(db, appt.id, requester_user_id=world["patient02"])
        assert read_schedule(engine, world["sched_am"])["booked_slots"] == 1

    async def test_serialize_reports_names_and_transitions(self, db: AsyncSession, world) -> None:
        appt = await svc.create_appointment(
            db, user_id=world["patient01"], schedule_id=world["sched_am"]
        )
        data = await svc.serialize_appointment(db, appt)
        assert data["doctor_name"] == "张三"
        assert data["department_name"] == "心血管内科"
        assert set(data["allowed_transitions"]) == {
            AppointmentStatus.CANCELLED.value,
            AppointmentStatus.CHECKED_IN.value,
            AppointmentStatus.NO_SHOW.value,
        }


# ==================== HTTP 端点 ====================

PATIENT1 = "patient1"
PATIENT2 = "patient2"
DOCTOR = "doctor"
CASHIER = "cashier"
ADMIN = "admin"

_WORLD_KEYS = {
    PATIENT1: ("patient01", "patient"),
    PATIENT2: ("patient02", "patient"),
    DOCTOR: ("doctor_chief", "doctor"),
    CASHIER: ("cashier01", "cashier"),
    ADMIN: ("admin01", "admin"),
}


def _resolve_identity(world, tag: str) -> tuple:
    """把用例声明的身份标签换成 (user_id, username, role_name, role_id)。

    cashier/admin 在 world 里没有预置账号，按需补建：权限矩阵断言需要
    「真实存在于 users 表的角色」，否则 403 可能来自别的原因而不是缺授权。
    """
    key, role_name = _WORLD_KEYS[tag]
    if key in world:
        return (world[key], key, role_name, ROLE_IDS[role_name])
    uid = make_user(world["engine"], role_name=role_name, username=key, real_name=key)
    world[key] = uid
    return (uid, key, role_name, ROLE_IDS[role_name])


@pytest.fixture()
async def client(engine, session_factory, world, request):
    """ASGI 客户端：覆盖 get_db 与 get_current_user（沿用 knowledge 测试策略）。"""
    import app.imcs_main as imcs_main_module

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    async def fake_check_redis() -> bool:
        return True

    app.dependency_overrides[get_db] = override_get_db
    tag = getattr(request, "param", None)
    if tag is not None:
        user_id, username, role_name, _role_id = _resolve_identity(world, tag)

        async def override_current_user():
            return as_user(user_id, username, role_name)

        app.dependency_overrides[get_current_user] = override_current_user

    originals = (imcs_main_module.async_session_factory, imcs_main_module.check_redis)
    imcs_main_module.async_session_factory = session_factory
    imcs_main_module.check_redis = fake_check_redis

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client

    app.dependency_overrides.clear()
    imcs_main_module.async_session_factory, imcs_main_module.check_redis = originals


# router 由主代理挂载到 api_router；未挂载时端点用例会以 404 失败，
# 因此先探测一次并给出明确 skip 理由，避免误判成业务缺陷。
def _routes_mounted() -> bool:
    return any("/appointments" in p for p in app.openapi()["paths"])


needs_routes = pytest.mark.skipif(
    not _routes_mounted(),
    reason="schedules / appointments router 尚未由主代理挂载到 api_router",
)


@needs_routes
class TestScheduleEndpoints:
    @pytest.mark.parametrize("client", [PATIENT1], indirect=True)
    async def test_list_schedules_ok_for_patient(self, client: AsyncClient, world) -> None:
        resp = await client.get("/api/v1/schedules", params={"department_code": "CARDIO"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total"] == 2
        assert body["items"][0]["remaining_slots"] > 0
        assert body["items"][0]["registration_fee"] == 5000, "金额保持分为整数"

    @pytest.mark.parametrize("client", [PATIENT1], indirect=True)
    async def test_invalid_date_format_is_422(self, client: AsyncClient, world) -> None:
        resp = await client.get("/api/v1/schedules", params={"start_date": "not-a-date"})
        assert resp.status_code == 422

    @pytest.mark.parametrize("client", [PATIENT1], indirect=True)
    async def test_doctors_by_department(self, client: AsyncClient, world) -> None:
        resp = await client.get(
            "/api/v1/schedules/doctors-by-department", params={"department_code": "CARDIO"}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["doctor_count"] == 1
        assert body["items"][0]["doctor_name"] == "张三"

    @pytest.mark.parametrize("client", [CASHIER], indirect=True)
    async def test_cashier_forbidden(self, client: AsyncClient, world) -> None:
        resp = await client.get("/api/v1/schedules")
        assert resp.status_code == 403, "收费员矩阵里没有 schedule:read"


@needs_routes
class TestAppointmentEndpoints:
    @pytest.mark.parametrize("client", [PATIENT1], indirect=True)
    async def test_create_read_cancel_flow(self, client: AsyncClient, engine, world) -> None:
        created = await client.post(
            "/api/v1/appointments",
            json={"schedule_id": str(world["sched_am"]), "chief_complaint": "胸闷"},
        )
        assert created.status_code == 201, created.text
        payload = created.json()
        assert payload["status"] == "booked"
        assert payload["doctor_name"] == "张三"
        assert payload["registration_fee"] == 5000
        appt_id = payload["id"]

        mine = await client.get("/api/v1/appointments")
        assert mine.status_code == 200
        assert mine.json()["total"] == 1

        detail = await client.get(f"/api/v1/appointments/{appt_id}")
        assert detail.status_code == 200
        assert detail.json()["chief_complaint"] == "胸闷"

        cancelled = await client.delete(f"/api/v1/appointments/{appt_id}")
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["status"] == "cancelled"
        assert read_schedule(engine, world["sched_am"])["booked_slots"] == 0

    @pytest.mark.parametrize("client", [PATIENT1], indirect=True)
    async def test_duplicate_returns_typed_409(self, client: AsyncClient, world) -> None:
        first = await client.post(
            "/api/v1/appointments", json={"schedule_id": str(world["sched_am"])}
        )
        assert first.status_code == 201
        second = await client.post(
            "/api/v1/appointments", json={"schedule_id": str(world["sched_am"])}
        )
        assert second.status_code == 409
        assert error_code_of(second) == "appointment_duplicate"

    @pytest.mark.parametrize("client", [PATIENT1], indirect=True)
    async def test_full_slot_returns_typed_409(self, client: AsyncClient, world) -> None:
        resp = await client.post(
            "/api/v1/appointments", json={"schedule_id": str(world["sched_full"])}
        )
        assert resp.status_code == 409
        assert error_code_of(resp) == "appointment_slot_full"

    @pytest.mark.parametrize("client", [PATIENT1], indirect=True)
    async def test_bad_schedule_id_is_422(self, client: AsyncClient, world) -> None:
        """请求体字段格式错误 → 422，与路径参数（定位资源、统一 404）口径不同。

        schedule_id 出现在 POST body 里，AppointmentCreate 的 field_validator
        先于任何业务逻辑把它拦下：对客户端这是「你请求写错了」，不是「资源找不到」，
        所以是 422。路径参数（如 /appointments/{id}）语义才是定位资源，才统一 404。
        """
        resp = await client.post("/api/v1/appointments", json={"schedule_id": "garbage"})
        assert resp.status_code == 422

    @pytest.mark.parametrize("client", [PATIENT2], indirect=True)
    async def test_cannot_read_or_cancel_others_appointment(
        self, client: AsyncClient, engine, world
    ) -> None:
        """行级隔离在 HTTP 层的落点是 404（不是 403）：不泄露「这张单子存在」。"""
        aid = insert_raw_appointment(
            engine,
            patient_id=world["profile01"],
            doctor_id=world["doctor_chief"],
            department_id=world["cardio"],
            schedule_id=world["sched_am"],
            appointment_date=world["target_day"],
        )
        detail = await client.get(f"/api/v1/appointments/{aid}")
        assert detail.status_code == 404
        assert error_code_of(detail) == "resource_not_found"

        cancel = await client.delete(f"/api/v1/appointments/{aid}")
        assert cancel.status_code == 404
        assert read_schedule(engine, world["sched_am"])["booked_slots"] == 0, (
            "越权取消不得动到别人的号源"
        )

    @pytest.mark.parametrize("client", [CASHIER], indirect=True)
    async def test_cashier_blocked_on_all_four_endpoints(self, client: AsyncClient, world) -> None:
        random_id = str(uuid.uuid4())
        cases = (
            ("get", "/api/v1/appointments", None),
            ("post", "/api/v1/appointments", {"schedule_id": str(world["sched_am"])}),
            ("delete", f"/api/v1/appointments/{random_id}", None),
            ("patch", f"/api/v1/appointments/{random_id}/check-in", None),
        )
        for method, path, body in cases:
            # httpx 的 get()/delete() 不接受 json 参数（只有写方法才有请求体），
            # 统一 getattr(client, method)(path, json=body) 会对前两种抛
            # TypeError。带体与否按方法区分，断言目标不变。
            sender = getattr(client, method)
            resp = await sender(path, json=body) if body is not None else await sender(path)
            assert resp.status_code == 403, f"{method.upper()} {path} 应对收费员拒绝"

    @pytest.mark.parametrize("client", [DOCTOR], indirect=True)
    async def test_doctor_can_read_but_not_create(self, client: AsyncClient, world) -> None:
        listing = await client.get("/api/v1/appointments")
        assert listing.status_code == 200, "医生有 appointment:read"
        creating = await client.post(
            "/api/v1/appointments", json={"schedule_id": str(world["sched_am"])}
        )
        assert creating.status_code == 403, "医生无 appointment:create"

    @pytest.mark.parametrize("client", [DOCTOR], indirect=True)
    async def test_doctor_can_check_in(self, client: AsyncClient, engine, world) -> None:
        """签到落在 appointment:update（医生持有），这是本批的权限映射选择。"""
        aid = insert_raw_appointment(
            engine,
            patient_id=world["profile01"],
            doctor_id=world["doctor_chief"],
            department_id=world["cardio"],
            schedule_id=world["sched_am"],
            appointment_date=world["target_day"],
        )
        resp = await client.patch(f"/api/v1/appointments/{aid}/check-in")
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "checked_in"

    @pytest.mark.parametrize("client", [PATIENT1], indirect=True)
    async def test_patient_cannot_check_in(self, client: AsyncClient, engine, world) -> None:
        aid = insert_raw_appointment(
            engine,
            patient_id=world["profile01"],
            doctor_id=world["doctor_chief"],
            department_id=world["cardio"],
            schedule_id=world["sched_am"],
            appointment_date=world["target_day"],
        )
        resp = await client.patch(f"/api/v1/appointments/{aid}/check-in")
        assert resp.status_code == 403, "患者矩阵里没有 appointment:update"

    @pytest.mark.parametrize("client", [ADMIN], indirect=True)
    async def test_admin_sees_all(self, client: AsyncClient, engine, world) -> None:
        insert_raw_appointment(
            engine,
            patient_id=world["profile02"],
            doctor_id=world["doctor_attending"],
            department_id=world["neuro"],
            schedule_id=world["sched_neuro"],
            appointment_date=world["target_day"],
        )
        resp = await client.get("/api/v1/appointments")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    @pytest.mark.parametrize("client", [PATIENT1], indirect=True)
    async def test_status_filter_param(self, client: AsyncClient, world) -> None:
        await client.post("/api/v1/appointments", json={"schedule_id": str(world["sched_am"])})
        hit = await client.get("/api/v1/appointments", params={"status": "booked"})
        miss = await client.get("/api/v1/appointments", params={"status": "completed"})
        assert hit.json()["total"] == 1
        assert miss.json()["total"] == 0

    @pytest.mark.parametrize("client", [PATIENT1], indirect=True)
    async def test_bad_date_filter_is_422(self, client: AsyncClient, world) -> None:
        resp = await client.get("/api/v1/appointments", params={"date_from": "2026-13-45"})
        assert resp.status_code == 422


# ==================== 路由注册 ====================


def _probe_app():
    """把两个 router 单独装进一个临时 FastAPI 实例求 openapi。

    不依赖主代理是否已 include_router：挂载是编排责任，
    而「router 自身定义了什么」是本批次的交付内容，必须独立可验证。
    """
    from fastapi import FastAPI

    from app.api.v1.appointment import router as appt_router
    from app.api.v1.schedule import router as sched_router

    probe = FastAPI()
    probe.include_router(appt_router, prefix="/api/v1")
    probe.include_router(sched_router, prefix="/api/v1")
    return probe


class TestRouterRegistration:
    def test_routers_expose_expected_prefix(self) -> None:
        from app.api.v1.appointment import router as appt_router
        from app.api.v1.schedule import router as sched_router

        assert sched_router.prefix == "/schedules"
        assert appt_router.prefix == "/appointments"

    def test_openapi_paths_present(self) -> None:
        paths = set(_probe_app().openapi()["paths"])
        for expected in (
            "/api/v1/schedules",
            "/api/v1/schedules/doctors-by-department",
            "/api/v1/appointments",
            "/api/v1/appointments/{appointment_id}",
            "/api/v1/appointments/{appointment_id}/check-in",
        ):
            assert expected in paths, f"缺少端点 {expected}"

    def test_methods_match_design(self) -> None:
        spec = _probe_app().openapi()["paths"]
        assert set(spec["/api/v1/appointments"]) == {"get", "post"}
        assert set(spec["/api/v1/appointments/{appointment_id}"]) == {"get", "delete"}
        assert set(spec["/api/v1/appointments/{appointment_id}/check-in"]) == {"patch"}
        assert set(spec["/api/v1/schedules"]) == {"get"}
        assert set(spec["/api/v1/schedules/doctors-by-department"]) == {"get"}


# ==================== 与其它批次的一致性 ====================


class TestCrossModuleConsistency:
    def test_effective_statuses_match_index_semantics(self) -> None:
        """防呆白名单必须等于「占用号源」的状态集合，且与索引排除项互补。"""
        assert set(svc.EFFECTIVE_APPOINTMENT_STATUSES) == {
            AppointmentStatus.BOOKED.value,
            AppointmentStatus.CHECKED_IN.value,
            AppointmentStatus.IN_PROGRESS.value,
            AppointmentStatus.COMPLETED.value,
        }
        released = {s.value for s in AppointmentStatus} - set(svc.EFFECTIVE_APPOINTMENT_STATUSES)
        assert released == {"cancelled", "no_show"}, (
            "索引谓词排除的就是这两个状态，两侧必须互补"
        )

    def test_no_legacy_or_triage_imports(self) -> None:
        """硬性约束：本批不得 import legacy 模块或 triage 侧文件。"""
        import inspect

        from app.api.v1 import appointment as appt_api
        from app.api.v1 import schedule as sched_api

        sources = [
            inspect.getsource(svc),
            inspect.getsource(appt_api),
            inspect.getsource(sched_api),
        ]
        joined = "\n".join(sources)
        for forbidden in (
            "app.routers",
            "app.database",
            "app.llm_factory",
            "milvus",
            "sqlmodel",
            "triage_workflow",
            "schemas.triage",
            "api.v1.triage",
            "models.ai",
        ):
            assert forbidden not in joined, f"出现禁用引用：{forbidden}"

    def test_service_functions_are_async(self) -> None:
        for name in (
            "create_appointment",
            "cancel_appointment",
            "check_in_appointment",
            "list_available_schedules",
            "list_doctors_with_open_slots",
            "list_my_appointments",
            "serialize_appointment",
        ):
            assert asyncio.iscoroutinefunction(getattr(svc, name)), name
