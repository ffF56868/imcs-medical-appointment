"""接诊、病历、检查开单与结案（阶段 4 医生侧）单元测试（文件型 SQLite，不打网络）。

覆盖 service 的状态流转与行级隔离、路由的权限映射与响应装配、
以及 router 自身的 openapi 契约。

**为什么不依赖主代理是否已 include_router**：挂载是编排责任，而「本批次交付了
什么」必须独立可验证，因此 openapi 断言走 ``_probe_app()`` 把 consult router
单独装进临时 FastAPI 实例。HTTP 端点用例同理——未挂载时按 appointment 测试的做法
整类 skip，而不是让它以 404 失败被误判成业务缺陷。

**为什么开始接诊的用例大多先直插一条 checked_in 预约**：签到属于挂号侧能力
（appointment_service.check_in_appointment），本批不该依赖它的正确性来验证自己；
直插能精确构造「前置状态已满足」这一前提，失败归因才唯一。反过来，
「非 checked_in 开始接诊 → 409」那条用例刻意用 booked 状态，
验证的是状态机闸门本身而不是数据准备。

SQLite 上的已知差异（不影响本文件结论）：
1. ``visits.appointment_id`` 的 unique 约束在 SQLite 正常生效，
   但本批的「同一预约单活跃 visit」判据走的是 service 显式查询而非约束兜底，
   因此测的是判据逻辑而非数据库特性。
2. DateTime 列不存时区偏移，``utcnow()`` 写回必然 naive，故不断言 tz 感知
   （口径与 appointment 测试一致，由 PG 集成测试负责）。
3. ``icd_codes`` 是 TextArray，SQLite 下按 JSON 数组存取——本批全部经 ORM 写入，
   不涉及手工 json.dumps；直插历史病历时必须 json.dumps，否则读回是 []（首轮踩过）。
"""

from __future__ import annotations

import asyncio
import inspect
import io
import uuid
from collections.abc import AsyncIterator
from datetime import date, datetime, timedelta
import tokenize
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
)
from app.imcs_main import app
from app.models.auth import User
from app.models.base import Base, utcnow
from app.models.clinical import (
    AppointmentStatus,
    ExamRequestStatus,
    Urgency,
    VisitStatus,
)
from app.schemas.consult import (
    ExamOrderRequest,
    MedicalRecordSaveRequest,
    VisitOut,
)
from app.services import consult_service as svc
from tests.imcs_sqlite import (
    build_session_factory,
    error_code_of,
    seed_role_permissions,
    seed_roles,
)

# ==================== 建表 ====================

CONSULT_TABLES: tuple[str, ...] = (
    "roles",
    "permissions",
    "role_permissions",
    "departments",
    "users",
    "patient_profiles",
    "doctor_schedules",
    "appointments",
    "visits",
    "exam_requests",
)

ROLE_IDS: dict[str, int] = {
    "patient": 1,
    "doctor": 2,
    "lab_tech": 3,
    "cashier": 4,
    "pharmacist": 5,
    "admin": 6,
}

_LOCAL_TZ = ZoneInfo("Asia/Shanghai")


@pytest.fixture()
def engine(tmp_path):
    """文件型 SQLite：同步建表 + 异步被测会话共用同一份数据（沿用 knowledge 测试策略）。"""
    import tests.imcs_sqlite  # noqa: F401  - 注册 PGUUID 的 SQLite DDL 编译器

    db_path = tmp_path / "imcs_consult.db"
    test_engine = sa.create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    tables = [Base.metadata.tables[name] for name in CONSULT_TABLES]
    with test_engine.begin() as conn:
        for table in tables:
            table.create(conn, checkfirst=True)
        seed_roles(conn)
        # require_permission 要 JOIN permissions，空表会让非 admin 一律 403。
        seed_role_permissions(conn)
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


# ==================== 数据助手 ====================


def _insert(engine, table: str, values: dict[str, Any]) -> None:  # noqa: ANN001
    cols = list(values)
    quoted = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    with engine.begin() as conn:
        conn.execute(
            sa.text(f"INSERT INTO {table} ({quoted}) VALUES ({placeholders})"),
            values,
        )


def local_today() -> date:
    return datetime.now(_LOCAL_TZ).date()


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
            "title": None,
            "specialty": None,
            "is_active": 1,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return uid


def make_profile(engine, user_id: uuid.UUID) -> uuid.UUID:  # noqa: ANN001
    pid = uuid.uuid4()
    _insert(
        engine,
        "patient_profiles",
        {
            "id": pid.hex,
            "user_id": user_id.hex,
            "gender": "M",
            "birth_date": date(1990, 1, 1),
            "medical_notes": None,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return pid


def make_appointment(  # noqa: ANN001
    engine,
    *,
    patient_id: uuid.UUID,
    doctor_id: uuid.UUID,
    department_id: uuid.UUID,
    status: str = AppointmentStatus.CHECKED_IN.value,
    appointment_date: date | None = None,
    time_slot: str = "08:00-12:00",
    chief_complaint: str | None = "近一周活动后胸闷气短",
) -> uuid.UUID:
    aid = uuid.uuid4()
    day = appointment_date or local_today()
    _insert(
        engine,
        "appointments",
        {
            "id": aid.hex,
            "patient_id": patient_id.hex,
            "doctor_id": doctor_id.hex,
            "department_id": department_id.hex,
            # schedule_id 在本批只被读取、从不参与写入判断，且该列无外键约束
            # （见 models/clinical.py 里 triage_result_id 同类处理），
            # 因此用一个随机 UUID 占位，避免为了测试再建 doctor_schedules 表。
            "schedule_id": uuid.uuid4().hex,
            "appointment_date": day.isoformat(),
            "time_slot": time_slot,
            "status": status,
            "source": "self",
            "chief_complaint": chief_complaint,
            "registration_fee": 5000,
            "checked_in_at": utcnow() if status != AppointmentStatus.BOOKED.value else None,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return aid


def insert_raw_visit(  # noqa: ANN001
    engine,
    *,
    appointment_id: uuid.UUID,
    patient_id: uuid.UUID,
    doctor_id: uuid.UUID,
    status: str = VisitStatus.IN_PROGRESS.value,
    diagnosis: str | None = None,
    icd_codes: list[str] | None = None,
    chief_complaint: str = "直插构造的主诉",
) -> uuid.UUID:
    vid = uuid.uuid4()
    _insert(
        engine,
        "visits",
        {
            "id": vid.hex,
            "appointment_id": appointment_id.hex,
            "patient_id": patient_id.hex,
            "doctor_id": doctor_id.hex,
            "chief_complaint": chief_complaint,
            "diagnosis": diagnosis,
            # TextArray 在 SQLite 走 JSON 数组，必须 json.dumps，
            # 逗号串会让 process_result_value 解析失败静默返回 []。
            "icd_codes": _json_array(icd_codes),
            "status": status,
            "started_at": utcnow(),
            "ended_at": utcnow() if status == VisitStatus.CLOSED.value else None,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return vid


def _json_array(values: list[str] | None) -> str | None:
    import json

    if values is None:
        return None
    return json.dumps(values, ensure_ascii=False)


def read_row(engine, table: str, row_id: uuid.UUID) -> dict[str, Any]:  # noqa: ANN001    """直读库里的行，断言「数据库真实发生了什么」而非 ORM 内存态。"""
    with engine.begin() as conn:
        row = conn.execute(
            sa.text(f"SELECT * FROM {table} WHERE id = :id"),  # noqa: S608 - 表名来自常量白名单
            {"id": row_id.hex},
        ).fetchone()
    assert row is not None, f"{table} 行不存在"
    return dict(row._mapping)


# ==================== world fixture ====================


# 非法前置状态各自的时段串：四种都要造一张独立预约，串必须互不相同，
# 否则「按 time_slot 升序」的队列断言会失去确定性。
_SLOT_BY_STATUS: dict[str, str] = {
    AppointmentStatus.BOOKED.value: "09:00-10:00",
    AppointmentStatus.COMPLETED.value: "10:00-11:00",
    AppointmentStatus.CANCELLED.value: "11:00-12:00",
    AppointmentStatus.NO_SHOW.value: "13:00-14:00",
}


@pytest.fixture()
def world(engine) -> dict[str, Any]:
    """两位医生 + 两位患者 + 若干预约的最小可用数据集。

    doctor_a / doctor_b 分属两个科室，是为了让「医生 A 看不到医生 B 的病历」
    这条隔离断言有真实的跨医生数据可撞；patient02 与 doctor_a 之间刻意**不预置**
    任何预约，用于构造「患者请求别人的 visit」。
    """
    cardio = make_department(engine, "CARDIO", "心血管内科")
    neuro = make_department(engine, "NEURO", "神经内科")

    doctor_a = make_user(engine, role_name="doctor", username="doctor_a", real_name="张三",
                         department_id=cardio)
    doctor_b = make_user(engine, role_name="doctor", username="doctor_b", real_name="李四",
                         department_id=neuro)
    patient01 = make_user(engine, role_name="patient", username="patient_01", real_name="患者一")
    patient02 = make_user(engine, role_name="patient", username="patient_02", real_name="患者二")
    profile01 = make_profile(engine, patient01)
    profile02 = make_profile(engine, patient02)

    appt_ci = make_appointment(
        engine, patient_id=profile01, doctor_id=doctor_a, department_id=cardio
    )
    appt_booked = make_appointment(
        engine,
        patient_id=profile01,
        doctor_id=doctor_a,
        department_id=cardio,
        status=AppointmentStatus.BOOKED.value,
        time_slot="14:00-17:30",
    )
    # doctor_b 名下的一张 checked_in 预约：用于跨医生越权开始接诊的断言
    appt_b = make_appointment(
        engine, patient_id=profile02, doctor_id=doctor_b, department_id=neuro,
        time_slot="09:00-10:00",
    )
    return {
        "engine": engine,
        "today": local_today(),
        "cardio": cardio,
        "neuro": neuro,
        "doctor_a": doctor_a,
        "doctor_b": doctor_b,
        "patient01": patient01,
        "patient02": patient02,
        "profile01": profile01,
        "profile02": profile02,
        "appt_ci": appt_ci,
        "appt_booked": appt_booked,
        "appt_b": appt_b,
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


# ==================== 1. 开始接诊 ====================


class TestStartConsultation:
    async def test_success_transitions_both_entities(self, db: AsyncSession, world) -> None:
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])

        assert visit.status == VisitStatus.IN_PROGRESS.value
        assert visit.patient_id == world["profile01"], "患者身份从预约带出，指向档案而非账号"
        assert visit.doctor_id == world["doctor_a"]
        assert visit.chief_complaint == "近一周活动后胸闷气短", "主诉应自预约带入病历"
        assert visit.started_at is not None

        appt_row = read_row(world["engine"], "appointments", world["appt_ci"])
        assert appt_row["status"] == AppointmentStatus.IN_PROGRESS.value

    async def test_missing_chief_complaint_falls_back_to_placeholder(
        self, db: AsyncSession, engine, world
    ) -> None:
        """预约没填主诉时给可读占位符，而不是撞 nullable=False 变成 500。"""
        aid = make_appointment(
            engine,
            patient_id=world["profile01"],
            doctor_id=world["doctor_a"],
            department_id=world["cardio"],
            time_slot="10:00-11:00",
            chief_complaint=None,
        )
        visit = await svc.start_consultation(db, aid, world["doctor_a"])
        assert visit.chief_complaint
        assert "主诉" in visit.chief_complaint

    async def test_duplicate_start_rejected(self, db: AsyncSession, engine, world) -> None:
        """同一预约已有活跃 visit → 409，且库里不得出现第二条病历。"""
        first = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        with pytest.raises(DuplicateAppointmentError) as exc:
            await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        assert exc.value.code == "appointment_duplicate"
        assert exc.value.detail["visit_id"] == str(first.id)

        with engine.begin() as conn:
            n = int(
                conn.execute(
                    sa.text("SELECT COUNT(*) FROM visits WHERE appointment_id=:a"),
                    {"a": world["appt_ci"].hex},
                ).scalar_one()
            )
        assert n == 1

    async def test_reopen_on_completed_appointment_rejected(
        self, db: AsyncSession, engine, world
    ) -> None:
        """上次 visit 结案后，预约已 completed——它不能再被原样重开。

        判据要分清：活跃 visit 判据看 status!=closed（见上条用例），
        但**能否开始接诊的前置是预约状态**。close_visit 会把预约推进到
        completed，而 completed 在 APPOINTMENT_TRANSITIONS 里是终态，
        start_consultation 里的 checked_in→in_progress 自然被拦（409）。
        真实业务的「带检查结果回来复诊」走新挂号新预约，而不是复用旧单——
        否则一次预约会挂出两段互不相干的病历，费用与随访都对不上账。
        """
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        await svc.save_medical_record(
            db, visit.id, world["doctor_a"], MedicalRecordSaveRequest(diagnosis="稳定性冠心病")
        )
        await svc.close_visit(db, visit.id, world["doctor_a"])
        assert read_row(engine, "appointments", world["appt_ci"])["status"] == (
            AppointmentStatus.COMPLETED.value
        )

        with pytest.raises(InvalidStateTransitionError) as exc:
            await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        assert exc.value.detail["entity"] == "appointment"
        assert exc.value.detail["current_status"] == AppointmentStatus.COMPLETED.value

    @pytest.mark.parametrize(
        "bad_status",
        [
            AppointmentStatus.BOOKED.value,
            AppointmentStatus.COMPLETED.value,
            AppointmentStatus.CANCELLED.value,
            AppointmentStatus.NO_SHOW.value,
        ],
    )
    async def test_non_checked_in_rejected_with_409_detail(
        self, db: AsyncSession, engine, world, bad_status: str
    ) -> None:
        """非法前置由状态机自然拦下：409 + current/target/entity 三件套。"""
        aid = make_appointment(
            engine,
            patient_id=world["profile01"],
            doctor_id=world["doctor_a"],
            department_id=world["cardio"],
            status=bad_status,
            # 每个状态配一个独立时段：appointments 上没有 (patient, date, slot)
            # 唯一约束，但复用同一串会让「按 time_slot 排序」的其它用例失去确定性。
            time_slot=_SLOT_BY_STATUS[bad_status],
        )
        with pytest.raises(InvalidStateTransitionError) as exc:
            await svc.start_consultation(db, aid, world["doctor_a"])
        err = exc.value
        assert err.status_code == 409
        assert err.detail["current_status"] == bad_status
        assert err.detail["target_status"] == AppointmentStatus.IN_PROGRESS.value
        assert err.detail["entity"] == "appointment"
        assert read_row(engine, "appointments", aid)["status"] == bad_status, "被拒时不得改状态"

    async def test_other_doctor_cannot_start(self, db: AsyncSession, engine, world) -> None:
        """医生 B 试图接诊医生 A 的预约 → 404（不泄露这张单子存在）。"""
        with pytest.raises(ResourceNotFoundError):
            await svc.start_consultation(db, world["appt_ci"], world["doctor_b"])
        assert read_row(engine, "appointments", world["appt_ci"])["status"] == (
            AppointmentStatus.CHECKED_IN.value
        )

    async def test_missing_appointment_404(self, db: AsyncSession, world) -> None:
        with pytest.raises(ResourceNotFoundError):
            await svc.start_consultation(db, uuid.uuid4(), world["doctor_a"])


# ==================== 2. 病历书写 ====================


class TestMedicalRecord:
    async def test_first_save_transitions_to_documented(self, db: AsyncSession, world) -> None:
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        saved, transitioned = await svc.save_medical_record(
            db,
            visit.id,
            world["doctor_a"],
            MedicalRecordSaveRequest(
                present_illness="活动后胸闷一个月，休息可缓解",
                physical_exam="BP 130/85mmHg，心律齐",
                diagnosis="稳定性冠心病",
                icd_codes=["I25.1", "I10"],
                treatment_advice="低盐饮食，门诊随访",
            ),
        )
        assert transitioned is True
        assert saved.status == VisitStatus.DOCUMENTED.value
        assert saved.icd_codes == ["I25.1", "I10"]
        assert read_row(world["engine"], "visits", visit.id)["status"] == "documented"

    async def test_second_save_does_not_transition_again(self, db: AsyncSession, world) -> None:
        """documented 后仍可回改（AI 开药前要能补病历），且不再触发流转。"""
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        await svc.save_medical_record(
            db, visit.id, world["doctor_a"], MedicalRecordSaveRequest(diagnosis="冠心病")
        )
        second, transitioned = await svc.save_medical_record(
            db, visit.id, world["doctor_a"], MedicalRecordSaveRequest(diagnosis="稳定性冠心病")
        )
        assert transitioned is False
        assert second.status == VisitStatus.DOCUMENTED.value
        assert second.diagnosis == "稳定性冠心病"

    async def test_partial_update_preserves_untouched_fields(
        self, db: AsyncSession, world
    ) -> None:
        """传了才覆盖：只改诊断不应抹掉既往史。"""
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        await svc.save_medical_record(
            db,
            visit.id,
            world["doctor_a"],
            MedicalRecordSaveRequest(past_history="青霉素过敏", diagnosis="冠心病"),
        )
        updated, _ = await svc.save_medical_record(
            db, visit.id, world["doctor_a"], MedicalRecordSaveRequest(diagnosis="心绞痛")
        )
        assert updated.past_history == "青霉素过敏"
        assert updated.diagnosis == "心绞痛"

    async def test_empty_payload_only_finalizes(self, db: AsyncSession, world) -> None:
        """空请求体合法：只完成 in_progress→documented，不改任何内容。"""
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        saved, transitioned = await svc.save_medical_record(
            db, visit.id, world["doctor_a"], MedicalRecordSaveRequest()
        )
        assert transitioned is True
        assert saved.status == VisitStatus.DOCUMENTED.value
        assert saved.diagnosis is None

    async def test_closed_visit_rejects_edit(self, db: AsyncSession, engine, world) -> None:
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        await svc.save_medical_record(
            db, visit.id, world["doctor_a"], MedicalRecordSaveRequest(diagnosis="冠心病")
        )
        await svc.close_visit(db, visit.id, world["doctor_a"])

        with pytest.raises(InvalidStateTransitionError) as exc:
            await svc.save_medical_record(
                db, visit.id, world["doctor_a"], MedicalRecordSaveRequest(diagnosis="改写尝试")
            )
        assert exc.value.status_code == 409
        assert exc.value.detail["current_status"] == VisitStatus.CLOSED.value
        assert read_row(engine, "visits", visit.id)["diagnosis"] == "冠心病", "归档后不得被改写"

    async def test_other_doctor_cannot_edit(self, db: AsyncSession, world) -> None:
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        with pytest.raises(ResourceNotFoundError):
            await svc.save_medical_record(
                db, visit.id, world["doctor_b"], MedicalRecordSaveRequest(diagnosis="越权")
            )

    async def test_icd_codes_normalized(self) -> None:
        """schema 层去空白/去重/大写归一：重复编码会让诊断分布报表翻倍。"""
        req = MedicalRecordSaveRequest(icd_codes=[" i25.1 ", "I25.1", "", "E11.9"])
        assert req.icd_codes == ["I25.1", "E11.9"]


# ==================== 3. 检查开单 ====================


class TestOrderExam:
    @pytest.mark.parametrize(
        ("exam_type", "expected_fee"),
        [
            ("blood_routine", 2500),
            ("urine_routine", 1500),
            ("liver_function", 6000),
            ("renal_function", 6000),
            ("blood_glucose", 500),
            ("blood_lipid", 800),
            ("ecg", 3000),
            ("ultrasound_abdomen", 12000),
        ],
    )
    async def test_price_table_hits(
        self, db: AsyncSession, world, exam_type: str, expected_fee: int
    ) -> None:
        """八项演示价目逐项核对（单位分）。"""
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        request = await svc.order_exam(
            db,
            visit.id,
            world["doctor_a"],
            ExamOrderRequest(exam_type=exam_type, exam_name="演示项目"),
        )
        assert request.fee_amount == expected_fee
        assert request.status == ExamRequestStatus.ORDERED.value
        assert request.is_charged is False

    async def test_unknown_exam_type_defaults_to_zero(
        self, db: AsyncSession, world, caplog
    ) -> None:
        """表外项目费用落 0 并打日志，交阶段 5 收费侧核价，而不是拦住医生。"""
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        request = await svc.order_exam(
            db,
            visit.id,
            world["doctor_a"],
            ExamOrderRequest(exam_type="pet_ct", exam_name="PET-CT"),
        )
        assert request.fee_amount == svc.DEFAULT_EXAM_FEE_CENTS
        assert request.status == ExamRequestStatus.ORDERED.value

    async def test_ids_inherited_from_visit(self, db: AsyncSession, world) -> None:
        """patient_id / doctor_id 从 visit 带出，不由客户端指定。"""
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        request = await svc.order_exam(
            db,
            visit.id,
            world["doctor_a"],
            ExamOrderRequest(
                exam_type="ecg", exam_name="心电图", body_part="胸部", urgency=Urgency.STAT.value
            ),
        )
        assert request.visit_id == visit.id
        assert request.patient_id == visit.patient_id
        assert request.doctor_id == world["doctor_a"]
        assert request.body_part == "胸部"
        assert request.urgency == Urgency.STAT.value

    async def test_stat_urgency_logs_critical_path(
        self, db: AsyncSession, world, caplog: pytest.LogCaptureFixture
    ) -> None:
        """stat 单必须留危急日志——这是给检验侧与运维检索的锚点。"""
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        with caplog.at_level("INFO", logger="app.services.consult_service"):
            request = await svc.order_exam(
                db,
                visit.id,
                world["doctor_a"],
                ExamOrderRequest(exam_type="ecg", exam_name="心电图", urgency="stat"),
            )
        critical = [
            record.message
            for record in caplog.records
            if "危急检查单" in record.message
        ]
        assert len(critical) == 1, "stat 单应恰好留下一条危急路径记录"
        assert str(request.id) in critical[0]
        assert "心电图" in critical[0]

    async def test_routine_urgency_logs_nothing_critical(
        self, db: AsyncSession, world, caplog: pytest.LogCaptureFixture
    ) -> None:
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        with caplog.at_level("INFO", logger="app.services.consult_service"):
            await svc.order_exam(
                db,
                visit.id,
                world["doctor_a"],
                ExamOrderRequest(exam_type="ecg", exam_name="心电图"),
            )
        assert not any("危急检查单" in record.message for record in caplog.records)

    async def test_ordering_on_closed_visit_rejected(self, db: AsyncSession, engine, world) -> None:
        """结案后不得开单：报告挂在归档病历下会被质控当成漏诊。"""
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        await svc.save_medical_record(
            db, visit.id, world["doctor_a"], MedicalRecordSaveRequest(diagnosis="冠心病")
        )
        await svc.close_visit(db, visit.id, world["doctor_a"])

        with pytest.raises(InvalidStateTransitionError) as exc:
            await svc.order_exam(
                db,
                visit.id,
                world["doctor_a"],
                ExamOrderRequest(exam_type="ecg", exam_name="心电图"),
            )
        assert exc.value.detail["current_status"] == VisitStatus.CLOSED.value
        with engine.begin() as conn:
            n = int(
                conn.execute(
                    sa.text("SELECT COUNT(*) FROM exam_requests WHERE visit_id=:v"),
                    {"v": visit.id.hex},
                ).scalar_one()
            )
        assert n == 0

    async def test_documented_visit_can_still_order(self, db: AsyncSession, world) -> None:
        """documented 仍可补检——定稿不等于禁止加做检查。"""
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        await svc.save_medical_record(
            db, visit.id, world["doctor_a"], MedicalRecordSaveRequest(diagnosis="冠心病")
        )
        request = await svc.order_exam(
            db,
            visit.id,
            world["doctor_a"],
            ExamOrderRequest(exam_type="liver_function", exam_name="肝功能全套"),
        )
        assert request.status == ExamRequestStatus.ORDERED.value

    async def test_other_doctor_cannot_order(self, db: AsyncSession, world) -> None:
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        with pytest.raises(ResourceNotFoundError):
            await svc.order_exam(
                db, visit.id, world["doctor_b"], ExamOrderRequest(exam_type="ecg", exam_name="心电图")
            )

    def test_price_table_units_are_cents(self) -> None:
        """价目契约：整数分、与注释里的元价一致、不含浮点。"""
        assert svc.EXAM_FEE_TABLE_CENTS["blood_routine"] == 2500
        assert all(isinstance(v, int) and v > 0 for v in svc.EXAM_FEE_TABLE_CENTS.values())
        assert set(svc.EXAM_FEE_TABLE_CENTS) == set(svc.EXAM_FEE_TABLE_NAMES)


# ==================== 4. 结束就诊 ====================


class TestCloseVisit:
    async def test_close_marks_both_entities(self, db: AsyncSession, engine, world) -> None:
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        await svc.save_medical_record(
            db, visit.id, world["doctor_a"], MedicalRecordSaveRequest(diagnosis="冠心病")
        )
        closed, appointment = await svc.close_visit(db, visit.id, world["doctor_a"])

        assert closed.status == VisitStatus.CLOSED.value
        assert closed.ended_at is not None
        assert appointment.status == AppointmentStatus.COMPLETED.value
        assert read_row(engine, "appointments", world["appt_ci"])["status"] == "completed"

    async def test_close_without_record_rejected(self, db: AsyncSession, engine, world) -> None:
        """强制先写病历：矩阵允许 in_progress→closed 直跳，本模块刻意不放行。"""
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        with pytest.raises(InvalidStateTransitionError) as exc:
            await svc.close_visit(db, visit.id, world["doctor_a"])
        err = exc.value
        assert err.detail["current_status"] == VisitStatus.IN_PROGRESS.value
        assert err.detail["target_status"] == VisitStatus.CLOSED.value
        assert err.detail["entity"] == "visit"
        assert "病历" in err.message
        # 两侧都不得被改动（同生同灭的反面检查：拒绝时也不能只动一半）
        assert read_row(engine, "visits", visit.id)["status"] == VisitStatus.IN_PROGRESS.value
        assert read_row(engine, "appointments", world["appt_ci"])["status"] == (
            AppointmentStatus.IN_PROGRESS.value
        )

    async def test_double_close_rejected(self, db: AsyncSession, world) -> None:
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        await svc.save_medical_record(
            db, visit.id, world["doctor_a"], MedicalRecordSaveRequest(diagnosis="冠心病")
        )
        await svc.close_visit(db, visit.id, world["doctor_a"])
        with pytest.raises(InvalidStateTransitionError):
            await svc.close_visit(db, visit.id, world["doctor_a"])

    async def test_other_doctor_cannot_close(self, db: AsyncSession, engine, world) -> None:
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        await svc.save_medical_record(
            db, visit.id, world["doctor_a"], MedicalRecordSaveRequest(diagnosis="冠心病")
        )
        with pytest.raises(ResourceNotFoundError):
            await svc.close_visit(db, visit.id, world["doctor_b"])
        assert read_row(engine, "visits", visit.id)["status"] == VisitStatus.DOCUMENTED.value

    async def test_matrix_still_allows_direct_jump(self) -> None:
        """记录既有事实：收紧发生在调用点而不是全局矩阵。

        若有人误以为「矩阵本身要求先 documented」，就会在别处（如人工归档脚本）
        重复加一层判定；这条断言钉住两层各管各的设计。
        """
        from app.services.state_machine import can_transition

        assert can_transition("visit", VisitStatus.IN_PROGRESS.value, VisitStatus.CLOSED.value)
        assert not can_transition("visit", VisitStatus.CLOSED.value, "documented")


# ==================== 5. 待诊队列与读取隔离 ====================


class TestPendingQueueAndReads:
    async def test_pending_filters_by_doctor_and_day(self, db: AsyncSession, engine, world) -> None:
        """医生 A 的队列不含医生 B 的预约，也不含昨天的预约。"""
        yesterday = world["today"] - timedelta(days=1)
        make_appointment(
            engine,
            patient_id=world["profile01"],
            doctor_id=world["doctor_a"],
            department_id=world["cardio"],
            appointment_date=yesterday,
            time_slot="11:00-12:00",
        )
        items, effective = await svc.list_pending_appointments(db, world["doctor_a"])
        assert effective == world["today"]
        assert [item["appointment_id"] for item in items] == [str(world["appt_ci"])]

        items_b, _ = await svc.list_pending_appointments(db, world["doctor_b"])
        assert [item["appointment_id"] for item in items_b] == [str(world["appt_b"])]

    async def test_pending_backfills_active_visit(self, db: AsyncSession, world) -> None:
        """已开始接诊的行要带回 visit_id/status，前端据此切换按钮文案。"""
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        items, _ = await svc.list_pending_appointments(db, world["doctor_a"])
        row = next(item for item in items if item["appointment_id"] == str(world["appt_ci"]))
        assert row["visit_id"] == str(visit.id)
        assert row["visit_status"] == VisitStatus.IN_PROGRESS.value

    async def test_pending_excludes_completed_and_cancelled(
        self, db: AsyncSession, engine, world
    ) -> None:
        for idx, st in enumerate((AppointmentStatus.COMPLETED.value, AppointmentStatus.CANCELLED.value)):
            make_appointment(
                engine,
                patient_id=world["profile01"],
                doctor_id=world["doctor_a"],
                department_id=world["cardio"],
                status=st,
                time_slot=f"1{idx + 4}:00-1{idx + 6}:00",
            )
        items, _ = await svc.list_pending_appointments(db, world["doctor_a"])
        assert len(items) == 1

    async def test_explicit_date_query(self, db: AsyncSession, engine, world) -> None:
        day = world["today"] + timedelta(days=1)
        make_appointment(
            engine,
            patient_id=world["profile02"],
            doctor_id=world["doctor_a"],
            department_id=world["cardio"],
            appointment_date=day,
            time_slot="08:30-09:30",
        )
        items, effective = await svc.list_pending_appointments(db, world["doctor_a"], day)
        assert effective == day
        assert len(items) == 1

    async def test_doctor_list_scopes_to_own_visits(self, db: AsyncSession, world) -> None:
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        mine, total = await svc.list_my_visits(db, doctor_id=world["doctor_a"])
        assert total == 1 and mine[0].id == visit.id
        _, others = await svc.list_my_visits(db, doctor_id=world["doctor_b"])
        assert others == 0, "医生 B 一条都不该看到"

    async def test_doctor_list_filters_by_status(self, db: AsyncSession, world) -> None:
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        _, n1 = await svc.list_my_visits(db, doctor_id=world["doctor_a"], status_filter="in_progress")
        _, n2 = await svc.list_my_visits(db, doctor_id=world["doctor_a"], status_filter="closed")
        assert n1 == 1 and n2 == 0
        await svc.save_medical_record(
            db, visit.id, world["doctor_a"], MedicalRecordSaveRequest(diagnosis="冠心病")
        )
        _, n3 = await svc.list_my_visits(db, doctor_id=world["doctor_a"], status_filter="documented")
        assert n3 == 1

    async def test_patient_view_sees_only_own(self, db: AsyncSession, world) -> None:
        await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        items, total = await svc.list_patient_visits(db, user_id=world["patient01"])
        assert total == 1 and items[0].patient_id == world["profile01"]
        _, other_total = await svc.list_patient_visits(db, user_id=world["patient02"])
        assert other_total == 0

    async def test_patient_without_profile_404(self, db: AsyncSession, world) -> None:
        """没建档的账号应给出可读 404，而不是让外键炸成 500。"""
        with pytest.raises(ResourceNotFoundError):
            await svc.list_patient_visits(db, user_id=world["doctor_a"])

    async def test_detail_isolation(self, db: AsyncSession, world) -> None:
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])

        by_owner = await svc.get_visit_detail(
            db, visit.id, requester_user_id=world["doctor_a"], requester_role="doctor"
        )
        assert by_owner.id == visit.id

        with pytest.raises(ResourceNotFoundError):
            await svc.get_visit_detail(
                db, visit.id, requester_user_id=world["doctor_b"], requester_role="doctor"
            )
        with pytest.raises(ResourceNotFoundError):
            await svc.get_visit_detail(
                db, visit.id, requester_user_id=world["patient02"], requester_role="patient"
            )
        by_patient = await svc.get_visit_detail(
            db, visit.id, requester_user_id=world["patient01"], requester_role="patient"
        )
        assert by_patient.id == visit.id

    async def test_admin_not_exempt_from_row_isolation(self, db: AsyncSession, world) -> None:
        """矩阵刻意不给 admin 临床读写，service 也不该开后门放行。"""
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        with pytest.raises(ResourceNotFoundError):
            await svc.get_visit_detail(
                db, visit.id, requester_user_id=uuid.uuid4(), requester_role="admin"
            )

    async def test_serialize_visit_shape(self, db: AsyncSession, world) -> None:
        visit = await svc.start_consultation(db, world["appt_ci"], world["doctor_a"])
        await svc.order_exam(
            db, visit.id, world["doctor_a"], ExamOrderRequest(exam_type="ecg", exam_name="心电图")
        )
        data = await svc.serialize_visit(db, visit)
        out = VisitOut(**data)
        assert out.doctor_name == "张三"
        assert out.exam_request_count == 1
        assert out.allowed_transitions == [VisitStatus.DOCUMENTED.value]
        assert isinstance(out.id, str) and isinstance(out.patient_id, str)


# ==================== HTTP 端点 ====================

DOCTOR_A = "doctor_a"
DOCTOR_B = "doctor_b"
PATIENT1 = "patient1"
PATIENT2 = "patient2"
CASHIER = "cashier"
LAB_TECH = "lab_tech"
ADMIN = "admin"

_WORLD_KEYS: dict[str, tuple[str, str]] = {
    DOCTOR_A: ("doctor_a", "doctor"),
    DOCTOR_B: ("doctor_b", "doctor"),
    PATIENT1: ("patient01", "patient"),
    PATIENT2: ("patient02", "patient"),
    CASHIER: ("cashier01", "cashier"),
    LAB_TECH: ("labtech01", "lab_tech"),
    ADMIN: ("admin01", "admin"),
}


def _resolve_identity(world, tag: str) -> tuple:
    """把身份标签换成 (user_id, username, role_name)。

    cashier/lab_tech/admin 在 world 里没有预置账号，按需补建：权限矩阵断言需要
    「真实存在于 users 表的角色」，否则 403 可能来自别的原因而不是缺授权。
    """
    key, role_name = _WORLD_KEYS[tag]
    if key in world:
        return (world[key], key, role_name)
    uid = make_user(world["engine"], role_name=role_name, username=key, real_name=key)
    world[key] = uid
    return (uid, key, role_name)


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
        user_id, username, role_name = _resolve_identity(world, tag)

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


def _consult_mounted() -> bool:
    return any("/consult" in p for p in app.openapi()["paths"])


needs_routes = pytest.mark.skipif(
    not _consult_mounted(), reason="consult router 尚未由主代理挂载到 api_router"
)


@needs_routes
class TestConsultEndpoints:
    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_start_then_record_then_exam_then_close(
        self, client: AsyncClient, engine, world
    ) -> None:
        """完整闭环走一遍 HTTP：每步的状态都能从响应里读到。"""
        started = await client.post(f"/api/v1/consult/appointments/{world['appt_ci']}/start")
        assert started.status_code == 201, started.text
        body = started.json()
        visit_id = body["visit"]["id"]
        assert body["visit"]["status"] == "in_progress"
        assert body["appointment_status"] == "in_progress"
        assert body["prior_visits"] == []

        recorded = await client.put(
            f"/api/v1/consult/visits/{visit_id}/record",
            json={
                "present_illness": "活动后胸闷一月",
                "diagnosis": "稳定性冠心病",
                "icd_codes": ["I25.1"],
                "treatment_advice": "规律服药，两周复诊",
            },
        )
        assert recorded.status_code == 200, recorded.text
        assert recorded.json()["transitioned"] is True
        assert recorded.json()["visit"]["status"] == "documented"

        ordered = await client.post(
            f"/api/v1/consult/visits/{visit_id}/exams",
            json={"exam_type": "blood_routine", "exam_name": "血常规"},
        )
        assert ordered.status_code == 201, ordered.text
        exam = ordered.json()
        assert exam["fee_amount"] == 2500
        assert exam["status"] == "ordered"
        assert exam["is_charged"] is False

        exams = await client.get(f"/api/v1/consult/visits/{visit_id}/exams")
        assert exams.status_code == 200
        assert len(exams.json()) == 1

        closed = await client.patch(f"/api/v1/consult/visits/{visit_id}/close")
        assert closed.status_code == 200, closed.text
        assert closed.json()["visit"]["status"] == "closed"
        assert closed.json()["appointment_status"] == "completed"
        assert read_row(engine, "appointments", world["appt_ci"])["status"] == "completed"

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_duplicate_start_returns_typed_409(
        self, client: AsyncClient, world
    ) -> None:
        first = await client.post(f"/api/v1/consult/appointments/{world['appt_ci']}/start")
        assert first.status_code == 201
        second = await client.post(f"/api/v1/consult/appointments/{world['appt_ci']}/start")
        assert second.status_code == 409
        assert error_code_of(second) == "appointment_duplicate"

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_start_non_checked_in_is_409(self, client: AsyncClient, world) -> None:
        resp = await client.post(f"/api/v1/consult/appointments/{world['appt_booked']}/start")
        assert resp.status_code == 409
        assert error_code_of(resp) == "state_transition_invalid"

    @pytest.mark.parametrize("client", [DOCTOR_B], indirect=True)
    async def test_other_doctor_start_is_404(self, client: AsyncClient, world) -> None:
        """越权开始接诊 → 404，不用错误信息反推这张单子属于谁。"""
        resp = await client.post(f"/api/v1/consult/appointments/{world['appt_ci']}/start")
        assert resp.status_code == 404
        assert error_code_of(resp) == "resource_not_found"

    @pytest.mark.parametrize("client", [DOCTOR_B], indirect=True)
    async def test_other_doctor_cannot_read_or_edit_visit(
        self, client: AsyncClient, engine, world
    ) -> None:
        """B 能读写自己的病历，但对 A 名下的病历三个入口全部 404。"""
        own = await client.post(f"/api/v1/consult/appointments/{world['appt_b']}/start")
        assert own.status_code == 201
        own_id = own.json()["visit"]["id"]
        assert (await client.get(f"/api/v1/consult/visits/{own_id}")).status_code == 200

        # 直插一段属于医生 A 的病历（A 已接诊但尚未定稿）
        with engine.begin() as conn:
            conn.execute(
                sa.text("UPDATE appointments SET status='in_progress' WHERE id=:i"),
                {"i": world["appt_ci"].hex},
            )
        a_visit = insert_raw_visit(
            engine,
            appointment_id=world["appt_ci"],
            patient_id=world["profile01"],
            doctor_id=world["doctor_a"],
            diagnosis="冠心病",
        )

        got = await client.get(f"/api/v1/consult/visits/{a_visit}")
        assert got.status_code == 404
        edited = await client.put(
            f"/api/v1/consult/visits/{a_visit}/record", json={"diagnosis": "越权改写"}
        )
        assert edited.status_code == 404
        ordered = await client.post(
            f"/api/v1/consult/visits/{a_visit}/exams",
            json={"exam_type": "ecg", "exam_name": "心电图"},
        )
        assert ordered.status_code == 404
        closed = await client.patch(f"/api/v1/consult/visits/{a_visit}/close")
        assert closed.status_code == 404
        assert read_row(engine, "visits", a_visit)["diagnosis"] == "冠心病"

    @pytest.mark.parametrize("client", [PATIENT2], indirect=True)
    async def test_patient_cross_visit_is_404(self, client: AsyncClient, world) -> None:
        """患者请求别人的 visit → 404（不是 403）。"""
        visit_id = insert_raw_visit(
            world["engine"],
            appointment_id=world["appt_ci"],
            patient_id=world["profile01"],
            doctor_id=world["doctor_a"],
            status=VisitStatus.CLOSED.value,
            diagnosis="冠心病",
        )
        resp = await client.get(f"/api/v1/consult/visits/{visit_id}")
        assert resp.status_code == 404
        assert error_code_of(resp) == "resource_not_found"

    @pytest.mark.parametrize("client", [PATIENT1], indirect=True)
    async def test_patient_can_read_own_but_not_write(self, client: AsyncClient, world) -> None:
        """患者持有 visit:read / medical_record:read，但没有 update → 改写 403。"""
        visit_id = insert_raw_visit(
            world["engine"],
            appointment_id=world["appt_ci"],
            patient_id=world["profile01"],
            doctor_id=world["doctor_a"],
            diagnosis="冠心病",
        )
        mine = await client.get("/api/v1/consult/visits")
        assert mine.status_code == 200
        assert mine.json()["total"] == 1

        detail = await client.get(f"/api/v1/consult/visits/{visit_id}")
        assert detail.status_code == 200
        assert detail.json()["diagnosis"] == "冠心病"

        edit = await client.put(
            f"/api/v1/consult/visits/{visit_id}/record", json={"diagnosis": "自行改诊断"}
        )
        assert edit.status_code == 403, "患者矩阵里没有 medical_record:update"

        order = await client.post(
            f"/api/v1/consult/visits/{visit_id}/exams",
            json={"exam_type": "ecg", "exam_name": "心电图"},
        )
        assert order.status_code == 403, "患者矩阵里没有 exam_request:create"

        closing = await client.patch(f"/api/v1/consult/visits/{visit_id}/close")
        assert closing.status_code == 403, "患者矩阵里没有 visit:update"

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_close_before_documented_is_409(self, client: AsyncClient, world) -> None:
        started = await client.post(f"/api/v1/consult/appointments/{world['appt_ci']}/start")
        visit_id = started.json()["visit"]["id"]
        resp = await client.patch(f"/api/v1/consult/visits/{visit_id}/close")
        assert resp.status_code == 409
        assert error_code_of(resp) == "state_transition_invalid"

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_pending_queue_endpoint(self, client: AsyncClient, world) -> None:
        resp = await client.get("/api/v1/consult/appointments/pending")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["doctor_id"] == str(world["doctor_a"])
        assert body["total"] == 1
        assert body["items"][0]["appointment_id"] == str(world["appt_ci"])
        assert body["items"][0]["visit_id"] is None

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_pending_bad_date_is_422(self, client: AsyncClient, world) -> None:
        resp = await client.get("/api/v1/consult/appointments/pending", params={"date": "x"})
        assert resp.status_code == 422

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_bad_visit_id_in_path_is_404(self, client: AsyncClient, world) -> None:
        """路径参数口径：非法 UUID 与不存在统一 404（区别于请求体字段的 422）。"""
        resp = await client.get("/api/v1/consult/visits/not-a-uuid")
        assert resp.status_code == 404
        put = await client.put("/api/v1/consult/visits/garbage/record", json={"diagnosis": "x"})
        assert put.status_code == 404

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_bad_urgency_in_body_is_422(self, client: AsyncClient, world) -> None:
        started = await client.post(f"/api/v1/consult/appointments/{world['appt_ci']}/start")
        visit_id = started.json()["visit"]["id"]
        resp = await client.post(
            f"/api/v1/consult/visits/{visit_id}/exams",
            json={"exam_type": "ecg", "exam_name": "心电图", "urgency": "asap"},
        )
        assert resp.status_code == 422, "urgency 取值校验属协议层，应为 422 而非 404"


@needs_routes
class TestCashierForbidden:
    """收费员对全部接诊端点 403：矩阵里 cashier 不持有 visit / medical_record 任何动作，
    也没有 exam_request:create / read（它只有 read_all）。"""

    @pytest.mark.parametrize("client", [CASHIER], indirect=True)
    async def test_all_endpoints_forbidden(self, client: AsyncClient, world) -> None:
        rid = str(uuid.uuid4())
        cases = (
            ("post", f"/api/v1/consult/appointments/{rid}/start", None),
            ("put", f"/api/v1/consult/visits/{rid}/record", {"diagnosis": "x"}),
            ("post", f"/api/v1/consult/visits/{rid}/exams", {"exam_type": "ecg", "exam_name": "n"}),
            ("patch", f"/api/v1/consult/visits/{rid}/close", None),
            ("get", "/api/v1/consult/appointments/pending", None),
            ("get", "/api/v1/consult/visits", None),
            ("get", f"/api/v1/consult/visits/{rid}", None),
            ("get", f"/api/v1/consult/visits/{rid}/exams", None),
        )
        for method, path, body in cases:
            sender = getattr(client, method)
            resp = await sender(path, json=body) if body is not None else await sender(path)
            assert resp.status_code == 403, f"{method.upper()} {path} 应对收费员拒绝"


@needs_routes
class TestLabTechBoundary:
    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_lab_tech_cannot_open_or_read_visit(self, client: AsyncClient, world) -> None:
        """检验科有 exam_request:read_all/update，但没有 create，也没有 visit:read。

        开单权在医生（矩阵注释原话），因此这两个入口都必须 403。
        """
        rid = str(uuid.uuid4())
        order = await client.post(
            f"/api/v1/consult/visits/{rid}/exams",
            json={"exam_type": "ecg", "exam_name": "心电图"},
        )
        assert order.status_code == 403
        listing = await client.get("/api/v1/consult/visits")
        assert listing.status_code == 403


# ==================== 路由注册（静态契约，不依赖挂载）====================


def _probe_app():
    """把 consult router 单独装进临时 FastAPI 实例求 openapi。

    挂载是主代理的编排责任；「router 自身定义了什么」是本批次交付内容，
    必须独立可验证，否则未挂载时端点用例只会以 404 失败并被误判成业务缺陷。
    """
    from fastapi import FastAPI

    from app.api.v1.consult import router as consult_router

    probe = FastAPI()
    probe.include_router(consult_router, prefix="/api/v1")
    return probe


class TestRouterRegistration:
    def test_router_prefix_and_tags(self) -> None:
        from app.api.v1.consult import router as consult_router

        assert consult_router.prefix == "/consult"
        assert consult_router.tags == ["接诊病历"]

    def test_openapi_paths_present(self) -> None:
        paths = set(_probe_app().openapi()["paths"])
        for expected in (
            "/api/v1/consult/appointments/{appointment_id}/start",
            "/api/v1/consult/appointments/pending",
            "/api/v1/consult/visits",
            "/api/v1/consult/visits/{visit_id}",
            "/api/v1/consult/visits/{visit_id}/record",
            "/api/v1/consult/visits/{visit_id}/exams",
            "/api/v1/consult/visits/{visit_id}/close",
        ):
            assert expected in paths, f"缺少端点 {expected}"

    def test_methods_match_design(self) -> None:
        spec = _probe_app().openapi()["paths"]
        assert set(spec["/api/v1/consult/appointments/{appointment_id}/start"]) == {"post"}
        assert set(spec["/api/v1/consult/appointments/pending"]) == {"get"}
        assert set(spec["/api/v1/consult/visits"]) == {"get"}
        assert set(spec["/api/v1/consult/visits/{visit_id}"]) == {"get"}
        assert set(spec["/api/v1/consult/visits/{visit_id}/record"]) == {"put"}
        assert set(spec["/api/v1/consult/visits/{visit_id}/exams"]) == {"get", "post"}
        assert set(spec["/api/v1/consult/visits/{visit_id}/close"]) == {"patch"}

    def test_route_conflict_between_static_and_param(self) -> None:
        """``/appointments/pending`` 与 ``/appointments/{id}/start`` 不得互相遮蔽。

        两者尾段形态不同（pending 是静态段，{appointment_id} 是参数段），
        但 FastAPI 按注册顺序匹配，若将来出现 ``/appointments/{id}`` 这类
        裸参数路径就会抢走 pending。这里显式钉住当前不冲突的事实。
        """
        spec = _probe_app().openapi()["paths"]
        assert "/api/v1/consult/appointments/pending" in spec
        assert "/api/v1/consult/appointments/{appointment_id}" not in spec


# ==================== 硬性约束回归 ====================


def _code_only(source: str) -> str:
    """从源码里只留下「真正的代码」，剔除注释与 docstring。

    禁用引用检查针对的是 import 与函数调用，而模块 docstring 里必然会写出
    「不 import 某某侧文件」这类说明——若拿整份源码做子串匹配，解释边界的那段话
    本身就会命中关键词，测试会因为写得清楚而失败。用 tokenize 精确剥离
    注释与字符串 token（不执行任何被测代码），断言才只反映真实的依赖关系。

    实现要点：``TokenInfo.line`` 是该 token 所在的**物理行原文**，因此遇到
    ``NEWLINE`` / ``NL`` 时取它即为整条逻辑行的收尾文本；STRING token 整体跳过，
    这样既去掉 docstring，也去掉 ``from app.services import exam_service``
    这类把模块名写进字符串的真实依赖形态。不执行任何被测代码。
    """
    lines: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.STRING:
            continue
        if token.type in (tokenize.NEWLINE, tokenize.NL):
            text = token.line.strip()
            if text:
                lines.append(text)
    return "\n".join(lines)


class TestCrossModuleConsistency:
    def test_no_legacy_or_lab_side_imports(self) -> None:
        """本批不得 import legacy 模块，也不得耦合并行同事的检验侧文件。"""
        from app.api.v1 import consult as consult_api
        from app.schemas import consult as consult_schema

        joined = "\n".join(
            [
                _code_only(inspect.getsource(svc)),
                _code_only(inspect.getsource(consult_api)),
                _code_only(inspect.getsource(consult_schema)),
            ]
        )
        for forbidden in (
            "app.routers",
            "app.database",
            "app.llm_factory",
            "app.legacy",
            "sqlmodel",
            "milvus",
            # 检验侧（同事负责）的任何模块名与文件名
            "exam_service",
            "report_service",
            "api.v1.exam",
            "schemas.exam",
            "services.exam",
        ):
            assert forbidden not in joined, f"出现禁用引用：{forbidden}"

    def test_service_functions_are_async(self) -> None:
        for name in (
            "start_consultation",
            "save_medical_record",
            "order_exam",
            "close_visit",
            "list_pending_appointments",
            "get_visit_detail",
            "list_my_visits",
            "list_patient_visits",
            "serialize_visit",
        ):
            assert asyncio.iscoroutinefunction(getattr(svc, name)), name

    def test_openable_statuses_exclude_closed(self) -> None:
        assert svc.OPENABLE_VISIT_STATUSES == (
            VisitStatus.IN_PROGRESS.value,
            VisitStatus.DOCUMENTED.value,
        )
        assert VisitStatus.CLOSED.value not in svc.OPENABLE_VISIT_STATUSES

    def test_active_statuses_are_complement_of_terminal(self) -> None:
        """活跃判据 = 全部 visit 状态 - 终态，两处必须互补，否则重开口径会漂移。"""
        from app.services.state_machine import TERMINAL_STATES

        all_statuses = {item.value for item in VisitStatus}
        assert set(svc.ACTIVE_VISIT_STATUSES) == all_statuses - set(
            TERMINAL_STATES["visit"]
        )

    def test_pending_statuses_cover_in_progress(self) -> None:
        """做到一半的人仍需在等待区可见，否则医生刷新页面就找不到他。"""
        assert set(svc.PENDING_APPOINTMENT_STATUSES) == {
            AppointmentStatus.CHECKED_IN.value,
            AppointmentStatus.IN_PROGRESS.value,
        }

    def test_visit_out_has_no_password_leak(self) -> None:
        """响应模型不得包含任何用户敏感字段（序列化手写 dict 的第二道保险）。"""
        fields = set(VisitOut.model_fields)
        for forbidden in ("password_hash", "phone", "email", "id_card_no", "doctor"):
            assert forbidden not in fields
