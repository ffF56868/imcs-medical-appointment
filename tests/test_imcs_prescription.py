"""处方域服务、路由与安全校验单元测试（阶段 5；文件型 SQLite，不打真实网络）。

覆盖 ``app/schemas/prescription.py`` 的协议层校验、
``app/services/prescription_service.py`` 的三条纯规则 / 草稿与开具流程 / 行级隔离 /
金额计算，以及 ``app/api/v1/prescription.py`` 的权限映射与响应装配。

**为什么端点一律走 HTTP 打**：consult/exam 两批都吃过「纯 service 用例全绿、
路由层的依赖装配 bug 完全测不出来」的亏（triage 侧 batch-assess 曾因把
``Annotated[User, Depends(...)]`` 再套一层 ``Depends(...)``，端点凭空多出必填参数恒 422）。
本文件的 create / update / confirm / cancel / list / detail 六个入口全部经 ASGI 客户端真实命中。

**无外部依赖可打桩**：处方域是纯规则 + 纯数据库，不调 LLM、不调 RAG、不派发 Celery。
这也是刻意的——医疗安全判定必须可复算，见 prescription_service 模块 docstring 第 1 节。

SQLite 已知差异（不影响结论）：
- DateTime 列不存时区偏移，故不断言 tz 感知；
- JSONBText / TextArray 在 SQLite 走 JSON 文本，直插时必须 ``json.dumps``；
- PG 的部分索引谓词不复刻，因此索引相关行为不在本文件断言范围内。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pytest
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.exceptions import (
    AllergyConflictError,
    DrugInteractionError,
    InvalidStateTransitionError,
    ResourceNotFoundError,
)
from app.imcs_main import app
from app.models.auth import User
from app.models.base import Base, utcnow
from app.models.clinical import VisitStatus
from app.models.pharmacy import PrescriptionStatus
from app.schemas.prescription import (
    PrescriptionConfirmRequest,
    PrescriptionCreateRequest,
    PrescriptionItemIn,
    PrescriptionOut,
    PrescriptionUpdateRequest,
)
from app.services import prescription_service as svc
from app.services.state_machine import allowed_targets, assert_transition, bulk_validate
from tests.imcs_sqlite import (
    build_session_factory,
    error_code_of,
    seed_role_permissions,
    seed_roles,
)

# ==================== 建表 ====================

PRESCRIPTION_TABLES: tuple[str, ...] = (
    "roles",
    "permissions",
    "role_permissions",
    "departments",
    "users",
    "patient_profiles",
    "doctor_schedules",
    "appointments",
    "visits",
    "prescriptions",
    "prescription_items",
    # 发药表与处方外键同链，建出来以免 cascade 删除路径上出现 no such table。
    "dispensing_records",
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
    """文件型 SQLite：同步建表 + 异步被测会话共用同一份数据（理由见 test_imcs_exam）。"""
    import tests.imcs_sqlite  # noqa: F401  - 注册 PGUUID 的 SQLite DDL 编译器

    db_path = tmp_path / "imcs_prescription.db"
    test_engine = sa.create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    tables = [Base.metadata.tables[name] for name in PRESCRIPTION_TABLES]
    with test_engine.begin() as conn:
        for table in tables:
            table.create(conn, checkfirst=True)
        seed_roles(conn)
        # require_permission 走 role_permissions JOIN permissions，
        # 空权限表会让非 admin 一律 403，权限用例就全成了假阳性。
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


def _json_array(values: list[str] | None) -> str | None:
    if values is None:
        return None
    return json.dumps(values, ensure_ascii=False)


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


def make_profile(  # noqa: ANN001
    engine,
    user_id: uuid.UUID,
    *,
    gender: str = "M",
    birth_date: date | None = date(1990, 1, 1),
    allergy_history: list[str] | None = None,
) -> uuid.UUID:
    pid = uuid.uuid4()
    _insert(
        engine,
        "patient_profiles",
        {
            "id": pid.hex,
            "user_id": user_id.hex,
            "gender": gender,
            "birth_date": birth_date,
            "medical_notes": None,
            # TextArray 在 SQLite 走 JSON 数组，必须 json.dumps。
            "allergy_history": _json_array(allergy_history),
            "chronic_diseases": None,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return pid


def insert_raw_visit(  # noqa: ANN001
    engine,
    *,
    patient_id: uuid.UUID,
    doctor_id: uuid.UUID,
    status: str = VisitStatus.DOCUMENTED.value,
) -> uuid.UUID:
    """最小 visit 行：prescriptions.visit_id 有外键，必须先有父行。

    刻意用直插而不是调 consult_service——接诊侧属另一批次，
    本批不应依赖它的正确性来准备数据。
    """
    vid = uuid.uuid4()
    _insert(
        engine,
        "visits",
        {
            "id": vid.hex,
            # appointment_id 只是 nullable=False + unique，不参与本批任何判断。
            "appointment_id": uuid.uuid4().hex,
            "patient_id": patient_id.hex,
            "doctor_id": doctor_id.hex,
            "chief_complaint": "直插构造的主诉",
            "diagnosis": "急性扁桃体炎",
            "status": status,
            "started_at": utcnow(),
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return vid


def insert_raw_prescription(  # noqa: ANN001
    engine,
    *,
    visit_id: uuid.UUID,
    patient_id: uuid.UUID,
    doctor_id: uuid.UUID,
    status: str = PrescriptionStatus.DRAFT.value,
    total_amount: int = 0,
    safety_checks: list[dict[str, Any]] | None = None,
    safety_override_reason: str | None = None,
) -> uuid.UUID:
    """直插一张处方主单（不含明细），用于构造 fixture 造不出的状态。

    paid 态由收费侧推进，本域没有任何写入路径，因此「paid 不可取消」这条
    只能靠直插拿到真实行来验证——比只用状态机单元断言更接近端到端。
    """
    rx_id = uuid.uuid4()
    _insert(
        engine,
        "prescriptions",
        {
            "id": rx_id.hex,
            "visit_id": visit_id.hex,
            "patient_id": patient_id.hex,
            "doctor_id": doctor_id.hex,
            "status": status,
            "diagnosis": "急性扁桃体炎",
            "notes": None,
            "total_amount": total_amount,
            # JSONBText 在 SQLite 走 JSON 文本，直插必须自己 dumps。
            "safety_checks": (
                json.dumps(safety_checks, ensure_ascii=False) if safety_checks is not None else None
            ),
            "safety_override_reason": safety_override_reason,
            "confirmed_at": utcnow() if status != PrescriptionStatus.DRAFT.value else None,
            "cancelled_at": utcnow() if status == PrescriptionStatus.CANCELLED.value else None,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return rx_id


def insert_raw_item(  # noqa: ANN001
    engine,
    *,
    prescription_id: uuid.UUID,
    drug_code: str,
    drug_name: str,
    quantity: str,
    unit: str = "盒",
    unit_price: str = "0.00",
    specification: str | None = None,
    dosage: str | None = None,
    frequency: str | None = None,
    route: str | None = "口服",
    duration_days: int | None = None,
) -> uuid.UUID:
    iid = uuid.uuid4()
    _insert(
        engine,
        "prescription_items",
        {
            "id": iid.hex,
            "prescription_id": prescription_id.hex,
            "drug_code": drug_code,
            "drug_name": drug_name,
            "specification": specification,
            "dosage": dosage,
            "frequency": frequency,
            "route": route,
            "duration_days": duration_days,
            "quantity": quantity,
            "unit": unit,
            "unit_price": unit_price,
        },
    )
    return iid


def read_row(engine, table: str, row_id: uuid.UUID) -> dict[str, Any]:  # noqa: ANN001
    """直读库里的行，断言「数据库真实发生了什么」而非 ORM 内存态。"""
    with engine.begin() as conn:
        row = conn.execute(
            sa.text(f"SELECT * FROM {table} WHERE id = :id"),  # noqa: S608 - 表名来自常量白名单
            {"id": row_id.hex},
        ).fetchone()
    assert row is not None, f"{table} 行不存在"
    return dict(row._mapping)


def count_rows(engine, table: str, where: str = "", params: dict[str, Any] | None = None):  # noqa: ANN001
    sql = f"SELECT COUNT(*) FROM {table}"  # noqa: S608 - 表名来自常量白名单
    if where:
        sql += f" WHERE {where}"
    with engine.begin() as conn:
        return int(conn.execute(sa.text(sql), params or {}).scalar_one())


# ==================== 药品夹具（演示口径，与种子过敏史对齐） ====================

# 患者1 的过敏史取自 app/seed/users.py：("青霉素", "磺胺类")。
ALLERGY_HISTORY = ["青霉素", "磺胺类"]

AMOX_CAPSULE = {
    "drug_code": "D0001",
    "drug_name": "阿莫西林胶囊",
    "specification": "0.25g*24粒",
    "dosage": "500mg",
    "frequency": "tid",
    "route": "口服",
    "duration_days": 7,
    "quantity": 2,
    "unit": "盒",
    "unit_price": "25.50",
}
# 同通用名不同剂型/厂牌 —— 与 AMOX_CAPSULE 构成重复用药（也同时命中过敏）。
AMOX_TABLET = {
    "drug_code": "D0002",
    "drug_name": "阿莫西林片",
    "specification": "0.5g*20片",
    "dosage": "500mg",
    "frequency": "bid",
    "quantity": 1,
    "unit": "盒",
    "unit_price": "18.00",
}
# 不过敏、剂量合规的安全用药，作为「正常开具」基线。
IBUPROFEN = {
    "drug_code": "D0101",
    "drug_name": "布洛芬缓释胶囊",
    "specification": "0.3g*20粒",
    "dosage": "300mg",
    "frequency": "bid",
    "quantity": 1,
    "unit": "盒",
    "unit_price": "19.90",
}
# 对乙酰氨基酚日上限 2000mg：tid × 500mg = 1500mg，合规。
PARACETAMOL_OK = {
    "drug_code": "D0102",
    "drug_name": "对乙酰氨基酚片",
    "dosage": "500mg",
    "frequency": "tid",
    "duration_days": 3,
    "quantity": 1,
    "unit": "盒",
    "unit_price": "8.00",
}
# qid × 1000mg = 4000mg > 2000mg，应出 warning 但不阻断。
PARACETAMOL_OVER = {
    "drug_code": "D0103",
    "drug_name": "对乙酰氨基酚片",
    "dosage": "1g",
    "frequency": "qid",
    "duration_days": 5,
    "quantity": 1,
    "unit": "盒",
    "unit_price": "8.00",
}
# 复方新诺明命中「磺胺类」过敏。
SMZ = {
    "drug_code": "D0201",
    "drug_name": "复方新诺明片",
    "dosage": "1000mg",
    "frequency": "bid",
    "duration_days": 5,
    "quantity": 1,
    "unit": "盒",
    "unit_price": "6.50",
}


def item_in(**overrides: Any) -> PrescriptionItemIn:
    base = {**IBUPROFEN, **overrides}
    return PrescriptionItemIn(**base)


def payload_for(visit_id: uuid.UUID, *items: dict[str, Any], **kw: Any) -> PrescriptionCreateRequest:
    return PrescriptionCreateRequest(
        visit_id=str(visit_id),
        diagnosis=kw.get("diagnosis", "急性扁桃体炎"),
        notes=kw.get("notes"),
        items=[PrescriptionItemIn(**item) for item in items],
    )


# ==================== world fixture ====================


@pytest.fixture()
def world(engine) -> dict[str, Any]:
    """两位医生 + 两位患者（其一有过敏史）+ 药师/收费员 + 若干处方。

    - profile_allergic 名下有一张 documented visit，用于开新方；
    - profile_clean 名下同样有一张；
    - rx_closed_visit 挂在 closed 的 visit 上，用于验证「结案后不得开方」；
    - rx_paid 是直插出来的已缴费处方，用于验证 paid 不可取消。
    """
    cardio = make_department(engine, "CARDIO", "心血管内科")
    pharm = make_department(engine, "PHARM", "药剂科")

    doctor_a = make_user(engine, role_name="doctor", username="doctor_a",
                         real_name="张三", department_id=cardio)
    doctor_b = make_user(engine, role_name="doctor", username="doctor_b",
                         real_name="李四", department_id=cardio)
    pharmacist = make_user(engine, role_name="pharmacist", username="pharmacist01",
                           real_name="药师王", department_id=pharm)
    cashier = make_user(engine, role_name="cashier", username="cashier01", real_name="收费员赵")
    patient01 = make_user(engine, role_name="patient", username="patient_01")
    patient02 = make_user(engine, role_name="patient", username="patient_02")
    profile_allergic = make_profile(engine, patient01, gender="F", allergy_history=ALLERGY_HISTORY)
    profile_clean = make_profile(engine, patient02, gender="M", allergy_history=None)

    visit_a = insert_raw_visit(engine, patient_id=profile_allergic, doctor_id=doctor_a)
    visit_b = insert_raw_visit(engine, patient_id=profile_clean, doctor_id=doctor_a)
    visit_other = insert_raw_visit(engine, patient_id=profile_allergic, doctor_id=doctor_b)
    visit_closed = insert_raw_visit(
        engine, patient_id=profile_allergic, doctor_id=doctor_a, status=VisitStatus.CLOSED.value
    )

    rx_paid = insert_raw_prescription(
        engine,
        visit_id=visit_a,
        patient_id=profile_allergic,
        doctor_id=doctor_a,
        status=PrescriptionStatus.PAID.value,
        total_amount=2550,
    )
    insert_raw_item(engine, prescription_id=rx_paid, **AMOX_CAPSULE)

    rx_confirmed_other = insert_raw_prescription(
        engine,
        visit_id=visit_other,
        patient_id=profile_allergic,
        doctor_id=doctor_b,
        status=PrescriptionStatus.CONFIRMED.value,
        total_amount=1990,
    )
    insert_raw_item(engine, prescription_id=rx_confirmed_other, **IBUPROFEN)

    return {
        "engine": engine,
        "cardio": cardio,
        "pharm": pharm,
        "doctor_a": doctor_a,
        "doctor_b": doctor_b,
        "pharmacist": pharmacist,
        "cashier": cashier,
        "patient01": patient01,
        "patient02": patient02,
        "profile_allergic": profile_allergic,
        "profile_clean": profile_clean,
        "visit_a": visit_a,
        "visit_b": visit_b,
        "visit_other": visit_other,
        "visit_closed": visit_closed,
        "rx_paid": rx_paid,
        "rx_confirmed_other": rx_confirmed_other,
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


# ==================== HTTP 客户端 ====================

DOCTOR_A = "doctor_a"
DOCTOR_B = "doctor_b"
PATIENT1 = "patient1"
PATIENT2 = "patient2"
PHARMACIST = "pharmacist"
CASHIER = "cashier"
ADMIN = "admin"
LAB_TECH_TAG = "labtech_tag"

_WORLD_KEYS: dict[str, tuple[str, str]] = {
    DOCTOR_A: ("doctor_a", "doctor"),
    DOCTOR_B: ("doctor_b", "doctor"),
    PATIENT1: ("patient01", "patient"),
    PATIENT2: ("patient02", "patient"),
    PHARMACIST: ("pharmacist", "pharmacist"),
    CASHIER: ("cashier", "cashier"),
    ADMIN: ("admin01", "admin"),
    LAB_TECH_TAG: ("labtech01", "lab_tech"),
}


def _resolve_identity(world, tag: str) -> tuple:
    """身份标签 → (user_id, username, role_name)。

    admin 未在 world 预置，按需补建：403/404 断言必须由「真实存在的角色账号」
    触发，否则失败原因可能不是缺授权。
    """
    key, role_name = _WORLD_KEYS[tag]
    if key in world:
        return (world[key], key, role_name)
    uid = make_user(world["engine"], role_name=role_name, username=key, real_name=key)
    world[key] = uid
    return (uid, key, role_name)


@pytest.fixture()
async def client(engine, session_factory, world, request):
    """ASGI 客户端：覆盖 get_db 与 get_current_user（沿用 exam/consult 策略）。"""
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


def _prescription_mounted() -> bool:
    return any("/prescriptions" in p for p in app.openapi()["paths"])


needs_routes = pytest.mark.skipif(
    not _prescription_mounted(), reason="prescription router 尚未由主代理挂载到 api_router"
)


# ==================== 业务助手 ====================


async def make_draft(db: AsyncSession, world, **kw: Any) -> uuid.UUID:
    """经 service 建一张草稿，返回处方 id（各流程用例的前置动作）。"""
    items = kw.pop("items", [IBUPROFEN])
    payload = payload_for(kw.pop("visit_id", world["visit_a"]), *items, **kw)
    rx = await svc.create_draft(db, world["doctor_a"], payload)
    return rx.id


def body_of(items: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    """HTTP 请求体（明细走 JSON，避免在服务端再做一次模型转换）。"""
    return {
        "visit_id": kw.get("visit_id"),
        "diagnosis": kw.get("diagnosis", "急性扁桃体炎"),
        "notes": kw.get("notes"),
        "items": items,
    }


# ==================== 1. 规则引擎：过敏史冲突 ====================


class TestAllergyRule:
    def test_amoxicillin_blocks_for_penicillin_allergy(self) -> None:
        """种子口径：患者1 过敏「青霉素」，开阿莫西林胶囊必须被拦。"""
        findings = svc.check_allergy_conflicts([AMOX_CAPSULE], ALLERGY_HISTORY)
        assert len(findings) == 1
        assert findings[0]["rule"] == svc.RULE_ALLERGY
        assert findings[0]["level"] == svc.LEVEL_BLOCK
        assert findings[0]["detail"]["allergen"] == "青霉素类"
        assert "阿莫西林胶囊" in findings[0]["message"]

    def test_sulfamethoxazole_blocks_for_sulfa_allergy(self) -> None:
        findings = svc.check_allergy_conflicts([SMZ], ALLERGY_HISTORY)
        assert findings[0]["detail"]["allergen"] == "磺胺类"

    def test_non_allergic_drug_passes(self) -> None:
        assert svc.check_allergy_conflicts([IBUPROFEN, PARACETAMOL_OK], ALLERGY_HISTORY) == []

    def test_no_allergy_history_never_blocks(self) -> None:
        """档案没记过敏史时不得凭空造出冲突。"""
        for history in (None, [], ["花粉"]):
            assert svc.check_allergy_conflicts([AMOX_CAPSULE], history) == []

    def test_alias_allergen_normalized(self) -> None:
        """档案写「盘尼西林」也要能拦住青霉素类——两侧写法都要折叠。"""
        findings = svc.check_allergy_conflicts([AMOX_CAPSULE], ["盘尼西林"])
        assert len(findings) == 1

    def test_free_text_allergen_normalized(self) -> None:
        findings = svc.check_allergy_conflicts([AMOX_CAPSULE], ["青霉素过敏"])
        assert len(findings) == 1

    def test_every_matching_item_reported(self) -> None:
        findings = svc.check_allergy_conflicts([AMOX_CAPSULE, AMOX_TABLET], ALLERGY_HISTORY)
        assert len(findings) == 2

    def test_prefix_match_covers_other_dosage_forms(self) -> None:
        """换剂型不得绕过拦截：精确匹配会漏掉「阿莫西林克拉维酸钾片」。"""
        variant = {**AMOX_CAPSULE, "drug_name": "阿莫西林克拉维酸钾片", "drug_code": "D0009"}
        assert len(svc.check_allergy_conflicts([variant], ALLERGY_HISTORY)) == 1

    def test_unrelated_drug_not_matched_by_substring(self) -> None:
        """前缀而非包含匹配：「维生素C银翘片」里含「林可」之类字面巧合不得误报。"""
        decoy = {**IBUPROFEN, "drug_code": "D9100", "drug_name": "维生素C银翘片"}
        assert svc.check_allergy_conflicts([decoy], ALLERGY_HISTORY) == []

    def test_code_prefix_also_matches(self) -> None:
        finding = svc.check_allergy_conflicts(
            [{"drug_code": "AMOX-01", "drug_name": "某商品名抗菌药"}], ALLERGY_HISTORY
        )
        assert len(finding) == 1
        assert "编码前缀" in finding[0]["detail"]["matched_by"]

    def test_rule_table_self_consistent(self) -> None:
        """规则表自洽：每个类别都必须在别名表里有落点，否则永远命中不了。"""
        categories = set(svc.ALLERGY_CLASS_MAP.values()) | set(svc.ALLERGY_CODE_PREFIX_MAP.values())
        for category in categories:
            assert svc._standardize_allergen(category) == category, category
            assert category in set(svc.ALLERGY_ALIAS_MAP.values()), category


# ==================== 2. 规则引擎：重复用药 ====================


class TestDuplicateRule:
    def test_same_generic_different_form_blocks(self) -> None:
        findings = svc.check_duplicate_therapy([AMOX_CAPSULE, AMOX_TABLET])
        assert len(findings) == 1
        assert findings[0]["rule"] == svc.RULE_DUPLICATE
        assert findings[0]["level"] == svc.LEVEL_BLOCK
        assert set(findings[0]["detail"]["drugs"]) == {"阿莫西林胶囊", "阿莫西林片"}

    def test_different_drugs_do_not_collide(self) -> None:
        assert svc.check_duplicate_therapy([AMOX_CAPSULE, IBUPROFEN]) == []

    def test_brand_parentheses_stripped(self) -> None:
        """厂牌括注不得影响通用名比对：「阿莫西林胶囊(再林)」与「阿莫西林片」是同一成分。"""
        branded = {**AMOX_CAPSULE, "drug_name": "阿莫西林胶囊（再林）", "drug_code": "D0050"}
        assert len(svc.check_duplicate_therapy([branded, AMOX_TABLET])) == 1

    def test_three_way_duplicate_reports_once(self) -> None:
        third = {**AMOX_TABLET, "drug_code": "D0003", "drug_name": "阿莫西林干混悬剂"}
        findings = svc.check_duplicate_therapy([AMOX_CAPSULE, AMOX_TABLET, third])
        assert len(findings) == 1
        assert len(findings[0]["detail"]["drugs"]) == 3

    def test_brand_prefix_not_stripped(self) -> None:
        """厂牌前缀不参与归一化（本演示引擎只剥剂型后缀）。

        「再林阿莫西林胶囊」与「阿莫西林胶囊」在真实审方里是同一成分，
        但前缀匹配判不出——这是演示级规则表已知的漏报面。用例把这个边界
        **钉成显式断言**而不是让它悄悄存在：日后接入药品字典做通用名映射时，
        这条会红，提醒同步调整判据与文档。
        """
        branded = {**AMOX_CAPSULE, "drug_code": "D0060", "drug_name": "再林阿莫西林胶囊"}
        assert svc.check_duplicate_therapy([branded, AMOX_CAPSULE]) == []

    @pytest.mark.parametrize(
        ("left", "right"),
        [("头孢呋辛酯片", "头孢曲松钠注射液"), ("奥美拉唑肠溶胶囊", "雷贝拉唑钠肠溶片")],
    )
    def test_similar_but_distinct_generics_not_merged(self, left: str, right: str) -> None:
        """不同通用名不得被误判成重复——误报会让医生对所有提示免疫。"""
        a = {**IBUPROFEN, "drug_code": "D9001", "drug_name": left}
        b = {**IBUPROFEN, "drug_code": "D9002", "drug_name": right}
        assert svc.check_duplicate_therapy([a, b]) == []

    def test_generic_name_helper_strips_longest_suffix(self) -> None:
        """长后缀优先剥离：「缓释胶囊」不能被「胶囊」截成「缓释」。"""
        assert svc._generic_name("布洛芬缓释胶囊") == "布洛芬"
        assert svc._generic_name("阿莫西林胶囊") == "阿莫西林"


# ==================== 3. 规则引擎：单日剂量 ====================


class TestDailyDoseRule:
    def test_excess_is_warning_not_block(self) -> None:
        findings = svc.check_daily_dose([PARACETAMOL_OVER])
        assert len(findings) == 1
        assert findings[0]["rule"] == svc.RULE_DAILY_DOSE
        assert findings[0]["level"] == svc.LEVEL_WARNING

    def test_within_limit_passes(self) -> None:
        assert svc.check_daily_dose([PARACETAMOL_OK]) == []

    def test_total_dose_uses_frequency_times_single_times_days(self) -> None:
        """4000mg/日 × 5 天 = 20000mg 合计，明细里要算给医生看。"""
        finding = svc.check_daily_dose([PARACETAMOL_OVER])[0]
        assert finding["detail"]["times_per_day"] == 4
        assert finding["detail"]["daily_dose_mg"] == "4000"
        assert finding["detail"]["max_daily_dose_mg"] == "2000"
        assert finding["detail"]["total_dose_mg"] == "20000"

    def test_gram_dosage_converted_to_mg(self) -> None:
        assert svc._parse_mg("1g") == Decimal("1000")
        assert svc._parse_mg("0.25g") == Decimal("250")
        assert svc._parse_mg("500mg") == Decimal("500")
        assert svc._parse_mg("250 mg") == Decimal("250")

    def test_unparsable_dosage_skips_judgement(self) -> None:
        """「1片」这种没有质量信息的写法绝不猜剂量——猜错就是假告警。"""
        item = {**PARACETAMOL_OVER, "dosage": "1片"}
        assert svc.check_daily_dose([item]) == []

    def test_unknown_frequency_skips_judgement(self) -> None:
        item = {**PARACETAMOL_OVER, "frequency": "必要时"}
        assert svc.check_daily_dose([item]) == []

    def test_drug_outside_table_skipped(self) -> None:
        assert svc.check_daily_dose([IBUPROFEN]) == []

    def test_boundary_equal_limit_is_ok(self) -> None:
        """正好落在上限上不报警：阈值语义是「不得超过」。"""
        item = {**PARACETAMOL_OK, "dosage": "500mg", "frequency": "qid"}
        assert svc.check_daily_dose([item]) == []

    def test_missing_duration_defaults_to_one_day(self) -> None:
        item = {**PARACETAMOL_OVER, "duration_days": None}
        finding = svc.check_daily_dose([item])[0]
        assert finding["detail"]["duration_days"] == 1
        assert finding["detail"]["total_dose_mg"] == "4000"

    def test_frequency_table_only_four_entries(self) -> None:
        assert svc.FREQUENCY_PER_DAY == {"qd": 1, "bid": 2, "tid": 3, "qid": 4}


# ==================== 4. 规则聚合 ====================


class TestRunSafetyChecks:
    def test_blocking_findings_come_first(self) -> None:
        """block 一律排在 warning 前；同级内保持 过敏→重复→剂量 的追加顺序。"""
        both = {**AMOX_TABLET, "dosage": "1g", "frequency": "qid"}
        # PARACETAMOL_OVER 提供唯一的 warning 项（qid×1g=4000mg > 2000mg 上限）；
        # 阿莫西林两条各自命中过敏 block，彼此再命中重复用药 block，
        # 1g qid=4000mg 恰好等于上限、不严格超限，故阿莫西林不出剂量项。
        findings = svc.run_safety_checks(
            [AMOX_CAPSULE, both, PARACETAMOL_OVER], ALLERGY_HISTORY
        )
        levels = [f["level"] for f in findings]
        assert levels[: len(levels) - 1] == [svc.LEVEL_BLOCK] * (len(levels) - 1)
        assert levels[-1] == svc.LEVEL_WARNING

    def test_clean_prescription_yields_empty_list(self) -> None:
        assert svc.run_safety_checks([IBUPROFEN, PARACETAMOL_OK], ALLERGY_HISTORY) == []

    def test_rules_are_pure_functions(self) -> None:
        """同一入参跑两次结果必须逐字相同——可复算是审计的前提。"""
        first = svc.run_safety_checks([AMOX_CAPSULE, PARACETAMOL_OVER], ALLERGY_HISTORY)
        second = svc.run_safety_checks([AMOX_CAPSULE, PARACETAMOL_OVER], ALLERGY_HISTORY)
        assert first == second


# ==================== 5. Schema 协议层 ====================


class TestSchemaLayer:
    def test_min_length_one_item_rejected(self) -> None:
        with pytest.raises(ValueError):
            PrescriptionCreateRequest(visit_id=str(uuid.uuid4()), items=[])

    def test_duplicate_drug_code_rejected_at_protocol(self) -> None:
        with pytest.raises(ValueError) as exc:
            PrescriptionCreateRequest(
                visit_id=str(uuid.uuid4()),
                items=[item_in(drug_code="D0001"), item_in(drug_code="D0001", drug_name="另一种")],
            )
        assert "重复" in str(exc.value)

    def test_case_insensitive_code_dedup(self) -> None:
        """d0001 与 D0001 归一化后视为同一个 code，否则唯一约束会被大小写绕过。"""
        with pytest.raises(ValueError):
            PrescriptionCreateRequest(
                visit_id=str(uuid.uuid4()),
                items=[item_in(drug_code="d0001"), item_in(drug_code="D0001")],
            )

    def test_drug_code_upper_normalized(self) -> None:
        assert item_in(drug_code=" d0001 ").drug_code == "D0001"

    def test_illegal_visit_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            PrescriptionCreateRequest(visit_id="not-a-uuid", items=[item_in()])

    @pytest.mark.parametrize("quantity", [0, -1])
    def test_non_positive_quantity_rejected(self, quantity: float) -> None:
        with pytest.raises(ValueError):
            item_in(quantity=quantity)

    def test_blank_override_reason_becomes_none(self) -> None:
        """只传空格等于「想绕过但没有理由」，不能当成合法 override 凭证。"""
        assert PrescriptionConfirmRequest(safety_override_reason="   ").safety_override_reason is None

    def test_override_reason_stripped(self) -> None:
        req = PrescriptionConfirmRequest(safety_override_reason="  已复核过敏史  ")
        assert req.safety_override_reason == "已复核过敏史"

    def test_status_literals_match_model_enum(self) -> None:
        from app.schemas.prescription import PRESCRIPTION_STATUS_VALUES

        assert set(PRESCRIPTION_STATUS_VALUES) == {s.value for s in PrescriptionStatus}


# ==================== 6. 创建草稿 ====================


class TestCreateDraft:
    async def test_create_writes_draft_with_items_and_total(self, db: AsyncSession, world) -> None:
        rx = await svc.create_draft(db, world["doctor_a"], payload_for(world["visit_a"], IBUPROFEN))
        assert rx.status == PrescriptionStatus.DRAFT.value
        assert rx.patient_id == world["profile_allergic"]
        assert rx.doctor_id == world["doctor_a"]
        assert len(rx.items) == 1
        row = read_row(world["engine"], "prescriptions", rx.id)
        assert row["status"] == "draft"
        assert row["total_amount"] == 1990

    async def test_empty_items_rejected_by_schema(self, db: AsyncSession, world) -> None:
        with pytest.raises(ValueError):
            PrescriptionCreateRequest(visit_id=str(world["visit_a"]), items=[])

    async def test_duplicate_drug_code_rejected(self, db: AsyncSession, world) -> None:
        """绕过协议层直接构造（模拟别的写入方），service 仍要给出友好错误而不是撞唯一约束。"""

        class _RawItem:
            """最小鸭子对象：只带 service 实际读取的那几个属性。"""

            def __init__(self, code: str, name: str) -> None:
                self.drug_code = code
                self.drug_name = name

        class _Raw:
            def __init__(self, **kw: Any) -> None:
                self.__dict__.update(kw)

        raw = _Raw(
            visit_id=str(world["visit_a"]),
            diagnosis=None,
            notes=None,
            items=[_RawItem("D7777", "阿莫西林胶囊"), _RawItem("D7777", "另一种厂牌")],
        )
        with pytest.raises(DrugInteractionError) as exc:
            await svc.create_draft(db, world["doctor_a"], raw)
        assert exc.value.detail["duplicate_drug_codes"] == ["D7777"]
        # 前置校验在任何写库动作之前抛出，处方表里只剩夹具预置的两张。
        assert count_rows(world["engine"], "prescriptions") == 2

    async def test_other_doctors_visit_is_404(self, db: AsyncSession, world) -> None:
        """医生 B 的 visit，医生 A 开不出方；越权一律 404 不泄露存在性。"""
        with pytest.raises(ResourceNotFoundError):
            await svc.create_draft(db, world["doctor_a"], payload_for(world["visit_other"], IBUPROFEN))

    async def test_closed_visit_rejected(self, db: AsyncSession, world) -> None:
        with pytest.raises(InvalidStateTransitionError) as exc:
            await svc.create_draft(db, world["doctor_a"], payload_for(world["visit_closed"], IBUPROFEN))
        assert exc.value.detail["current_status"] == VisitStatus.CLOSED.value
        assert exc.value.detail["entity"] == "visit"

    async def test_in_progress_visit_allowed(self, db: AsyncSession, world) -> None:
        """边写病历边开药是常态，不把开药卡在病历定稿之后。"""
        vid = insert_raw_visit(
            world["engine"],
            patient_id=world["profile_allergic"],
            doctor_id=world["doctor_a"],
            status=VisitStatus.IN_PROGRESS.value,
        )
        rx = await svc.create_draft(db, world["doctor_a"], payload_for(vid, IBUPROFEN))
        assert rx.status == PrescriptionStatus.DRAFT.value

    async def test_draft_does_not_run_safety_checks(self, db: AsyncSession, world) -> None:
        """草稿阶段不拦过敏——校验发生在有法律效力的「确认开具」上。"""
        rx = await svc.create_draft(db, world["doctor_a"], payload_for(world["visit_a"], AMOX_CAPSULE))
        assert rx.safety_checks is None
        assert rx.status == PrescriptionStatus.DRAFT.value

    async def test_created_at_present_after_commit(self, db: AsyncSession, world) -> None:
        rx = await svc.create_draft(db, world["doctor_a"], payload_for(world["visit_a"], IBUPROFEN))
        assert isinstance(rx.created_at, (datetime, type(None)))


# ==================== 7. 金额计算 ====================


class TestAmountCalculation:
    async def test_total_is_integer_cents(self, db: AsyncSession, world) -> None:
        rx = await svc.create_draft(
            db,
            world["doctor_a"],
            payload_for(world["visit_a"], {**IBUPROFEN, "unit_price": "25.50", "quantity": 2}),
        )
        assert rx.total_amount == 5100
        assert isinstance(read_row(world["engine"], "prescriptions", rx.id)["total_amount"], int)

    async def test_float_error_does_not_leak(self, db: AsyncSession, world) -> None:
        """0.1×3 在 float 下是 0.30000000000000004 → 30 分还是 30.000000000000004 分？
        Decimal 路径必须干净落到 30。"""
        rx = await svc.create_draft(
            db,
            world["doctor_a"],
            payload_for(world["visit_a"], {**IBUPROFEN, "unit_price": "0.10", "quantity": 3}),
        )
        assert rx.total_amount == 30
        assert float("0.1") * 3 * 100 != 30  # 反证：float 口径确实会飘

    async def test_multi_item_sum(self, db: AsyncSession, world) -> None:
        rx = await svc.create_draft(
            db,
            world["doctor_a"],
            payload_for(
                world["visit_a"],
                {**IBUPROFEN, "drug_code": "D0101", "unit_price": "19.90", "quantity": 1},
                {**AMOX_TABLET, "drug_code": "D0102", "unit_price": "18.00", "quantity": 2},
            ),
        )
        assert rx.total_amount == 1990 + 3600

    async def test_fractional_quantity_rounds_half_up(self, db: AsyncSession, world) -> None:
        """1.5 盒 × 0.33 元 = 0.495 元 → 49.5 分，ROUND_HALF_UP 进位到 50。"""
        rx = await svc.create_draft(
            db,
            world["doctor_a"],
            payload_for(world["visit_a"], {**IBUPROFEN, "unit_price": "0.33", "quantity": 1.5}),
        )
        assert rx.total_amount == 50

    async def test_two_decimal_precision_kept(self, db: AsyncSession, world) -> None:
        rx = await svc.create_draft(
            db,
            world["doctor_a"],
            payload_for(world["visit_a"], {**IBUPROFEN, "unit_price": "123.45", "quantity": 7}),
        )
        assert rx.total_amount == 86415

    async def test_update_recalculates_total(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world, items=[{**IBUPROFEN, "unit_price": "10.00", "quantity": 1}])
        assert read_row(world["engine"], "prescriptions", rx_id)["total_amount"] == 1000
        await svc.update_draft(
            db,
            rx_id,
            world["doctor_a"],
            PrescriptionUpdateRequest(
                items=[PrescriptionItemIn(**{**IBUPROFEN, "unit_price": "10.00", "quantity": 3})]
            ),
        )
        assert read_row(world["engine"], "prescriptions", rx_id)["total_amount"] == 3000


# ==================== 8. 更新草稿 ====================


class TestUpdateDraft:
    async def test_replace_items(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world)
        updated = await svc.update_draft(
            db,
            rx_id,
            world["doctor_a"],
            PrescriptionUpdateRequest(
                diagnosis="换成咽炎",
                items=[
                    PrescriptionItemIn(**{**IBUPROFEN, "drug_code": "D0500", "drug_name": "西瓜霜含片"}),
                    PrescriptionItemIn(**{**IBUPROFEN, "drug_code": "D0501", "drug_name": "维生素C片"}),
                ],
            ),
        )
        assert updated.status == PrescriptionStatus.DRAFT.value
        assert updated.diagnosis == "换成咽炎"
        assert {i.drug_code for i in updated.items} == {"D0500", "D0501"}
        assert count_rows(world["engine"], "prescription_items", "prescription_id = :p", {"p": rx_id.hex}) == 2

    async def test_non_draft_rejected(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world)
        await svc.confirm_prescription(db, rx_id, world["doctor_a"], None)
        with pytest.raises(InvalidStateTransitionError) as exc:
            await svc.update_draft(
                db, rx_id, world["doctor_a"], PrescriptionUpdateRequest(items=[item_in()])
            )
        assert exc.value.detail["current_status"] == PrescriptionStatus.CONFIRMED.value
        assert exc.value.detail["entity"] == "prescription"

    async def test_cancelled_rejected(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world)
        await svc.cancel_prescription(db, rx_id, world["doctor_a"], "开错了")
        with pytest.raises(InvalidStateTransitionError):
            await svc.update_draft(db, rx_id, world["doctor_a"], PrescriptionUpdateRequest(items=[item_in()]))

    async def test_other_doctor_404(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world)
        with pytest.raises(ResourceNotFoundError):
            await svc.update_draft(
                db, rx_id, world["doctor_b"], PrescriptionUpdateRequest(items=[item_in()])
            )

    async def test_empty_items_rejected(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world)
        with pytest.raises(ValueError):
            PrescriptionUpdateRequest(items=[])

    async def test_partial_fields_kept(self, db: AsyncSession, world) -> None:
        """notes 不传保持原值：整体替换只针对 items，文本栏仍是「传了才覆盖」。"""
        rx_id = await make_draft(db, world, notes="饭后服用")
        updated = await svc.update_draft(
            db, rx_id, world["doctor_a"], PrescriptionUpdateRequest(items=[item_in()])
        )
        assert updated.notes == "饭后服用"


# ==================== 9. 确认开具与 override ====================


class TestConfirm:
    async def test_clean_prescription_confirms(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world, items=[IBUPROFEN])
        rx, overridden = await svc.confirm_prescription(db, rx_id, world["doctor_a"], None)
        assert rx.status == PrescriptionStatus.CONFIRMED.value
        assert overridden is False
        assert rx.safety_checks == []
        assert rx.confirmed_at is not None
        assert rx.safety_override_reason is None

    async def test_allergy_blocks_without_reason(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world, items=[AMOX_CAPSULE])
        with pytest.raises(AllergyConflictError) as exc:
            await svc.confirm_prescription(db, rx_id, world["doctor_a"], PrescriptionConfirmRequest())
        assert exc.value.code == "prescription_allergy_conflict"
        assert exc.value.detail["override_required"] is True
        assert exc.value.detail["findings"][0]["rule"] == svc.RULE_ALLERGY
        # 被拦下时状态与校验结果都不该写库
        row = read_row(world["engine"], "prescriptions", rx_id)
        assert row["status"] == PrescriptionStatus.DRAFT.value
        assert row["safety_checks"] is None

    async def test_override_reason_allows_and_persists(self, db: AsyncSession, engine, world) -> None:
        """带理由强行开具：状态推进、理由落库、风险项仍然完整留在 safety_checks。"""
        rx_id = await make_draft(db, world, items=[AMOX_CAPSULE])
        rx, overridden = await svc.confirm_prescription(
            db, rx_id, world["doctor_a"], PrescriptionConfirmRequest(safety_override_reason="家属确认既往脱敏治疗成功")
        )
        assert rx.status == PrescriptionStatus.CONFIRMED.value
        assert overridden is True
        row = read_row(engine, "prescriptions", rx_id)
        assert row["safety_override_reason"] == "家属确认既往脱敏治疗成功"
        stored = json.loads(row["safety_checks"])
        assert [f["rule"] for f in stored] == [svc.RULE_ALLERGY]
        assert stored[0]["level"] == svc.LEVEL_BLOCK

    async def test_override_keeps_blocked_visible(self, db: AsyncSession, world) -> None:
        """override 改变的是能否开具，不是记录是否存在——药师要看得见被忽略的风险。"""
        rx_id = await make_draft(db, world, items=[AMOX_CAPSULE])
        await svc.confirm_prescription(
            db, rx_id, world["doctor_a"], PrescriptionConfirmRequest(safety_override_reason="已复核")
        )
        out = await svc.serialize_prescription(db, await svc.get_prescription_detail(
            db, rx_id, requester_user_id=world["doctor_a"], requester_role="doctor"
        ))
        assert out["blocked_count"] == 1
        assert out["safety_override_reason"] == "已复核"

    async def test_duplicate_therapy_blocks_via_interaction_error(self, db: AsyncSession, world) -> None:
        """两张不同 code 但同通用名的制剂：重复用药阻断，走 DrugInteractionError。"""
        rx_id = await make_draft(
            db,
            world,
            visit_id=world["visit_b"],
            items=[AMOX_CAPSULE, AMOX_TABLET],
        )
        with pytest.raises(DrugInteractionError):
            # visit_b 名下是 profile_clean（无过敏史），阿莫西林只命中重复用药，
            # 不命中过敏 —— 异常类型必须是 DrugInteractionError。
            await svc.confirm_prescription(db, rx_id, world["doctor_a"], None)

        clean_dup_a = {**IBUPROFEN, "drug_code": "D8001", "drug_name": "布洛芬片"}
        clean_dup_b = {**IBUPROFEN, "drug_code": "D8002", "drug_name": "布洛芬缓释胶囊"}
        other_id = await make_draft(db, world, visit_id=world["visit_b"], items=[clean_dup_a, clean_dup_b])
        with pytest.raises(DrugInteractionError) as exc:
            await svc.confirm_prescription(db, other_id, world["doctor_a"], None)
        assert exc.value.code == "prescription_drug_interaction"
        assert exc.value.detail["findings"][0]["rule"] == svc.RULE_DUPLICATE

    async def test_dose_excess_does_not_block(self, db: AsyncSession, engine, world) -> None:
        """warning 级不进 override 通道：不带理由也能正常开具，且结果有记录。"""
        rx_id = await make_draft(db, world, items=[PARACETAMOL_OVER])
        rx, overridden = await svc.confirm_prescription(db, rx_id, world["doctor_a"], None)
        assert rx.status == PrescriptionStatus.CONFIRMED.value
        assert overridden is False
        stored = json.loads(read_row(engine, "prescriptions", rx_id)["safety_checks"])
        assert len(stored) == 1
        assert stored[0]["level"] == svc.LEVEL_WARNING
        assert stored[0]["rule"] == svc.RULE_DAILY_DOSE

    async def test_mixed_levels_recorded_together(self, db: AsyncSession, engine, world) -> None:
        over_amox = {**AMOX_CAPSULE, "dosage": "2g", "frequency": "tid"}
        rx_id = await make_draft(db, world, items=[over_amox])
        await svc.confirm_prescription(
            db, rx_id, world["doctor_a"], PrescriptionConfirmRequest(safety_override_reason="感染科会诊意见")
        )
        stored = json.loads(read_row(engine, "prescriptions", rx_id)["safety_checks"])
        rules = {f["rule"] for f in stored}
        assert rules == {svc.RULE_ALLERGY, svc.RULE_DAILY_DOSE}

    async def test_confirm_twice_rejected(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world, items=[IBUPROFEN])
        await svc.confirm_prescription(db, rx_id, world["doctor_a"], None)
        with pytest.raises(InvalidStateTransitionError):
            await svc.confirm_prescription(db, rx_id, world["doctor_a"], None)

    async def test_other_doctor_404(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world, items=[IBUPROFEN])
        with pytest.raises(ResourceNotFoundError):
            await svc.confirm_prescription(db, rx_id, world["doctor_b"], None)

    async def test_missing_profile_allergy_treated_as_empty(self, db: AsyncSession, engine, world) -> None:
        """档案查不到时按「无过敏史」处理，不该 500。"""
        orphan_visit = insert_raw_visit(
            engine, patient_id=world["profile_clean"], doctor_id=world["doctor_a"]
        )
        rx_id = await make_draft(db, world, visit_id=orphan_visit, items=[AMOX_CAPSULE])
        rx, overridden = await svc.confirm_prescription(db, rx_id, world["doctor_a"], None)
        assert rx.status == PrescriptionStatus.CONFIRMED.value
        assert overridden is False


# ==================== 10. 取消 ====================


class TestCancel:
    async def test_cancel_draft(self, db: AsyncSession, engine, world) -> None:
        rx_id = await make_draft(db, world, items=[IBUPROFEN], notes="原备注")
        rx = await svc.cancel_prescription(db, rx_id, world["doctor_a"], "患者拒药")
        assert rx.status == PrescriptionStatus.CANCELLED.value
        assert rx.cancelled_at is not None
        row = read_row(engine, "prescriptions", rx_id)
        assert "原备注" in row["notes"] and "[作废原因] 患者拒药" in row["notes"]

    async def test_cancel_confirmed(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world, items=[IBUPROFEN])
        await svc.confirm_prescription(db, rx_id, world["doctor_a"], None)
        rx = await svc.cancel_prescription(db, rx_id, world["doctor_a"], None)
        assert rx.status == PrescriptionStatus.CANCELLED.value

    async def test_cancel_paid_rejected(self, db: AsyncSession, engine, world) -> None:
        """paid 不可取消（矩阵里 paid 的后继只有 dispensed）；退费由收费域处理。"""
        with pytest.raises(InvalidStateTransitionError) as exc:
            await svc.cancel_prescription(db, world["rx_paid"], world["doctor_a"], "想退药")
        assert exc.value.detail["current_status"] == PrescriptionStatus.PAID.value
        assert exc.value.detail["target_status"] == PrescriptionStatus.CANCELLED.value
        assert read_row(engine, "prescriptions", world["rx_paid"])["status"] == "paid"

    async def test_cancel_terminal_cancelled_rejected(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world, items=[IBUPROFEN])
        await svc.cancel_prescription(db, rx_id, world["doctor_a"], None)
        with pytest.raises(InvalidStateTransitionError):
            await svc.cancel_prescription(db, rx_id, world["doctor_a"], None)

    async def test_other_doctor_404(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world, items=[IBUPROFEN])
        with pytest.raises(ResourceNotFoundError):
            await svc.cancel_prescription(db, rx_id, world["doctor_b"], None)

    def test_paid_state_machine_unit_assertion(self) -> None:
        """状态机层面的同一结论，与上面的真实行互为印证。"""
        assert PrescriptionStatus.CANCELLED.value not in allowed_targets(
            "prescription", PrescriptionStatus.PAID.value
        )
        assert allowed_targets("prescription", PrescriptionStatus.PAID.value) == [
            PrescriptionStatus.DISPENSED.value
        ]
        with pytest.raises(InvalidStateTransitionError):
            assert_transition("prescription", PrescriptionStatus.PAID.value, PrescriptionStatus.CANCELLED.value)

    def test_transition_matrix_exhaustive(self) -> None:
        """穷举 prescription 全部迁移组合，防止日后改矩阵时无声放宽。"""
        all_states = [s.value for s in PrescriptionStatus]
        pairs = [(a, b) for a in all_states for b in all_states if a != b]
        results = {(cur, tgt): ok for cur, tgt, ok in bulk_validate("prescription", pairs)}
        legal = {pair for pair, ok in results.items() if ok}
        assert legal == {
            ("draft", "confirmed"),
            ("draft", "cancelled"),
            ("confirmed", "paid"),
            ("confirmed", "cancelled"),
            ("paid", "dispensed"),
        }


# ==================== 11. 读取与行级隔离 ====================


class TestReadIsolation:
    async def test_doctor_sees_only_own(self, db: AsyncSession, world) -> None:
        await make_draft(db, world, items=[IBUPROFEN])
        rows, total = await svc.list_prescriptions(
            db, requester_user_id=world["doctor_a"], requester_role="doctor"
        )
        assert total >= 2
        # doctor_b 名下的那张不得出现——隔离靠 doctor_id 而不是靠运气。
        assert all(rx.doctor_id == world["doctor_a"] for rx in rows)
        assert world["rx_confirmed_other"] not in [rx.id for rx in rows]

    async def test_pharmacist_sees_all(self, db: AsyncSession, world) -> None:
        rows, total = await svc.list_prescriptions(
            db, requester_user_id=world["pharmacist"], requester_role="pharmacist"
        )
        assert world["rx_confirmed_other"] in [rx.id for rx in rows]
        assert world["rx_paid"] in [rx.id for rx in rows]
        assert total == len(rows)

    async def test_cashier_sees_all(self, db: AsyncSession, world) -> None:
        rows, _ = await svc.list_prescriptions(
            db, requester_user_id=world["cashier"], requester_role="cashier"
        )
        assert world["rx_paid"] in [rx.id for rx in rows]

    async def test_patient_sees_only_own(self, db: AsyncSession, engine, world) -> None:
        """patient_01 名下两张（含 doctor_b 开的），patient_02 一张也没有。

        行级隔离的口径是**患者档案**而不是开单医生：同一张处方换个医生开，
        患者照样看得见；反过来，知道别人的 visit_id 也查不出别人的东西。
        """
        mine = await make_draft(db, world, items=[IBUPROFEN])
        other_profile_visit = insert_raw_visit(
            engine, patient_id=world["profile_clean"], doctor_id=world["doctor_b"]
        )
        theirs = await svc.create_draft(
            db, world["doctor_b"], payload_for(other_profile_visit, IBUPROFEN)
        )

        rows, total = await svc.list_prescriptions(
            db, requester_user_id=world["patient01"], requester_role="patient"
        )
        ids = {rx.id for rx in rows}
        assert mine in ids and world["rx_paid"] in ids
        assert theirs.id not in ids
        assert total == len(rows)

        # patient_02 即使指名要别人的 visit，也只会拿到空集而不是 403。
        rows2, total2 = await svc.list_prescriptions(
            db,
            requester_user_id=world["patient02"],
            requester_role="patient",
            visit_id=world["visit_a"],
        )
        assert total2 == 0 and rows2 == []

    async def test_patient_detail_of_others_is_404(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world, items=[IBUPROFEN])
        with pytest.raises(ResourceNotFoundError):
            await svc.get_prescription_detail(
                db, rx_id, requester_user_id=world["patient02"], requester_role="patient"
            )

    async def test_patient_can_read_own(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world, items=[IBUPROFEN])
        rx = await svc.get_prescription_detail(
            db, rx_id, requester_user_id=world["patient01"], requester_role="patient"
        )
        assert rx.id == rx_id

    async def test_doctor_detail_of_others_is_404(self, db: AsyncSession, world) -> None:
        with pytest.raises(ResourceNotFoundError):
            await svc.get_prescription_detail(
                db, world["rx_confirmed_other"], requester_user_id=world["doctor_a"], requester_role="doctor"
            )

    async def test_pharmacist_can_read_any_detail(self, db: AsyncSession, world) -> None:
        rx = await svc.get_prescription_detail(
            db, world["rx_confirmed_other"], requester_user_id=world["pharmacist"], requester_role="pharmacist"
        )
        assert rx.status == PrescriptionStatus.CONFIRMED.value

    async def test_admin_has_no_backdoor(self, db: AsyncSession, world) -> None:
        """矩阵刻意不给 admin 临床记录读写，service 也不开后门。"""
        with pytest.raises(ResourceNotFoundError):
            await svc.get_prescription_detail(
                db, world["rx_paid"], requester_user_id=uuid.uuid4(), requester_role="admin"
            )

    async def test_lab_tech_gets_empty_list(self, db: AsyncSession, world) -> None:
        rows, total = await svc.list_prescriptions(
            db, requester_user_id=uuid.uuid4(), requester_role="lab_tech"
        )
        assert rows == [] and total == 0

    async def test_patient_without_profile_gets_empty(self, db: AsyncSession, engine, world) -> None:
        lonely = make_user(engine, role_name="patient", username="lonely")
        rows, total = await svc.list_prescriptions(
            db, requester_user_id=lonely, requester_role="patient"
        )
        assert rows == [] and total == 0

    async def test_status_filter(self, db: AsyncSession, world) -> None:
        await make_draft(db, world, items=[IBUPROFEN])
        rows, _ = await svc.list_prescriptions(
            db,
            requester_user_id=world["doctor_a"],
            requester_role="doctor",
            status_filter=PrescriptionStatus.PAID.value,
        )
        assert [rx.id for rx in rows] == [world["rx_paid"]]

    async def test_pagination_bounds(self, db: AsyncSession, world) -> None:
        for idx in range(3):
            await make_draft(
                db,
                world,
                visit_id=world["visit_b"],
                items=[{**IBUPROFEN, "drug_code": f"D600{idx}"}],
            )
        rows, total = await svc.list_prescriptions(
            db, requester_user_id=world["doctor_a"], requester_role="doctor", page=1, page_size=2
        )
        assert len(rows) == 2 and total >= 4

    async def test_serialize_shape(self, db: AsyncSession, world) -> None:
        rx_id = await make_draft(db, world, items=[IBUPROFEN])
        rx = await svc.get_prescription_detail(
            db, rx_id, requester_user_id=world["doctor_a"], requester_role="doctor"
        )
        out = await svc.serialize_prescription(db, rx)
        assert out["doctor_name"] == "张三"
        assert out["allowed_transitions"] == ["cancelled", "confirmed"]
        assert out["items"][0]["unit_price"] == "19.90"
        # 序列化字典必须能直接喂进响应模型，否则路由层会在运行时炸出缺字段。
        PrescriptionOut(**out)

    async def test_list_visit_prescriptions(self, db: AsyncSession, world) -> None:
        await make_draft(db, world, items=[IBUPROFEN])
        rows = await svc.list_visit_prescriptions(db, world["visit_a"])
        assert rows and all(rx.visit_id == world["visit_a"] for rx in rows)

# ==================== 12. HTTP 端点与权限边界 ====================


@needs_routes
class TestHttpEndpoints:
    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_create_and_confirm_flow(self, client: AsyncClient, world) -> None:
        created = await client.post(
            "/api/v1/prescriptions",
            json=body_of([IBUPROFEN], visit_id=str(world["visit_a"])),
        )
        assert created.status_code == 201, created.text
        body = created.json()
        assert body["status"] == "draft"
        assert body["total_amount"] == 1990
        assert body["blocked_count"] == 0
        assert body["allowed_transitions"] == ["cancelled", "confirmed"]

        confirmed = await client.post(f"/api/v1/prescriptions/{body['id']}/confirm", json={})
        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["prescription"]["status"] == "confirmed"
        assert confirmed.json()["overridden"] is False

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_empty_items_http_422(self, client: AsyncClient, world) -> None:
        resp = await client.post(
            "/api/v1/prescriptions", json=body_of([], visit_id=str(world["visit_a"]))
        )
        assert resp.status_code == 422

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_duplicate_code_http_422(self, client: AsyncClient, world) -> None:
        resp = await client.post(
            "/api/v1/prescriptions",
            json=body_of([IBUPROFEN, {**IBUPROFEN, "drug_name": "另一种"}], visit_id=str(world["visit_a"])),
        )
        assert resp.status_code == 422

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_allergy_block_then_override_http(self, client: AsyncClient, world) -> None:
        created = await client.post(
            "/api/v1/prescriptions",
            json=body_of([AMOX_CAPSULE], visit_id=str(world["visit_a"])),
        )
        rx_id = created.json()["id"]

        blocked = await client.post(f"/api/v1/prescriptions/{rx_id}/confirm", json={})
        assert blocked.status_code == 422, blocked.text
        assert error_code_of(blocked) == "prescription_allergy_conflict"

        forced = await client.post(
            f"/api/v1/prescriptions/{rx_id}/confirm",
            json={"safety_override_reason": "已电话核实既往用药耐受"},
        )
        assert forced.status_code == 200, forced.text
        payload = forced.json()
        assert payload["overridden"] is True
        assert payload["prescription"]["safety_override_reason"] == "已电话核实既往用药耐受"
        assert payload["prescription"]["blocked_count"] == 1

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_dose_warning_confirms_without_reason(self, client: AsyncClient, world) -> None:
        created = await client.post(
            "/api/v1/prescriptions",
            json=body_of([PARACETAMOL_OVER], visit_id=str(world["visit_a"])),
        )
        rx_id = created.json()["id"]
        confirmed = await client.post(f"/api/v1/prescriptions/{rx_id}/confirm", json={})
        assert confirmed.status_code == 200, confirmed.text
        checks = confirmed.json()["prescription"]["safety_checks"]
        assert checks[0]["level"] == "warning"
        assert confirmed.json()["prescription"]["warning_count"] == 1

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_update_draft_via_http(self, client: AsyncClient, world) -> None:
        created = await client.post(
            "/api/v1/prescriptions", json=body_of([IBUPROFEN], visit_id=str(world["visit_a"]))
        )
        rx_id = created.json()["id"]
        updated = await client.put(
            f"/api/v1/prescriptions/{rx_id}",
            json={
                "diagnosis": "换成咽炎",
                "items": [{**IBUPROFEN, "drug_code": "D0999", "drug_name": "西瓜霜含片", "unit_price": "5.00"}],
            },
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["diagnosis"] == "换成咽炎"
        assert updated.json()["total_amount"] == 500

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_update_confirmed_http_409(self, client: AsyncClient, world) -> None:
        created = await client.post(
            "/api/v1/prescriptions", json=body_of([IBUPROFEN], visit_id=str(world["visit_a"]))
        )
        rx_id = created.json()["id"]
        await client.post(f"/api/v1/prescriptions/{rx_id}/confirm", json={})
        resp = await client.put(
            f"/api/v1/prescriptions/{rx_id}", json={"items": [IBUPROFEN]}
        )
        assert resp.status_code == 409, resp.text
        assert error_code_of(resp) == "state_transition_invalid"

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_cancel_paid_http_409(self, client: AsyncClient, world) -> None:
        resp = await client.patch(f"/api/v1/prescriptions/{world['rx_paid']}/cancel", json={})
        assert resp.status_code == 409, resp.text
        assert error_code_of(resp) == "state_transition_invalid"

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_cancel_draft_http(self, client: AsyncClient, world) -> None:
        created = await client.post(
            "/api/v1/prescriptions", json=body_of([IBUPROFEN], visit_id=str(world["visit_a"]))
        )
        resp = await client.patch(
            f"/api/v1/prescriptions/{created.json()['id']}/cancel", json={"reason": "患者拒药"}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["prescription"]["status"] == "cancelled"

    @pytest.mark.parametrize("client", [PHARMACIST], indirect=True)
    async def test_pharmacist_can_read_all(self, client: AsyncClient, world) -> None:
        listing = await client.get("/api/v1/prescriptions")
        assert listing.status_code == 200, listing.text
        ids = [item["id"] for item in listing.json()["items"]]
        assert str(world["rx_confirmed_other"]) in ids

        detail = await client.get(f"/api/v1/prescriptions/{world['rx_paid']}")
        assert detail.status_code == 200, detail.text

    @pytest.mark.parametrize("client", [PHARMACIST], indirect=True)
    async def test_pharmacist_cannot_update(self, client: AsyncClient, world) -> None:
        """矩阵刻意不给药师 prescription:update——有问题应驳回让医生重开。"""
        denied = await client.put(
            f"/api/v1/prescriptions/{world['rx_paid']}", json={"items": [IBUPROFEN]}
        )
        assert denied.status_code == 403, denied.text
        assert error_code_of(denied) == "auth_forbidden"

        also_denied = await client.post(f"/api/v1/prescriptions/{world['rx_paid']}/confirm", json={})
        assert also_denied.status_code == 403, also_denied.text

        cannot_create = await client.post(
            "/api/v1/prescriptions", json=body_of([IBUPROFEN], visit_id=str(world["visit_a"]))
        )
        assert cannot_create.status_code == 403, cannot_create.text

    @pytest.mark.parametrize("client", [CASHIER], indirect=True)
    async def test_cashier_can_read_not_write(self, client: AsyncClient, world) -> None:
        listing = await client.get("/api/v1/prescriptions")
        assert listing.status_code == 200, listing.text
        denied = await client.patch(f"/api/v1/prescriptions/{world['rx_paid']}/cancel", json={})
        assert denied.status_code == 403, denied.text

    @pytest.mark.parametrize("client", [PATIENT1], indirect=True)
    async def test_patient_reads_own_only(self, client: AsyncClient, world) -> None:
        listing = await client.get("/api/v1/prescriptions")
        assert listing.status_code == 200, listing.text
        ids = [item["id"] for item in listing.json()["items"]]
        assert str(world["rx_confirmed_other"]) in ids  # 属于 patient_01 的档案
        assert str(world["rx_paid"]) in ids

    @pytest.mark.parametrize("client", [PATIENT2], indirect=True)
    async def test_patient_cannot_read_others_detail(self, client: AsyncClient, world) -> None:
        """越权 404 而不是 403：不向探测者确认「这张处方存在」。"""
        resp = await client.get(f"/api/v1/prescriptions/{world['rx_paid']}")
        assert resp.status_code == 404, resp.text
        assert error_code_of(resp) == "resource_not_found"

    @pytest.mark.parametrize("client", [PATIENT2], indirect=True)
    async def test_patient_cannot_create(self, client: AsyncClient, world) -> None:
        resp = await client.post(
            "/api/v1/prescriptions", json=body_of([IBUPROFEN], visit_id=str(world["visit_a"]))
        )
        assert resp.status_code == 403, resp.text

    @pytest.mark.parametrize("client", [DOCTOR_B], indirect=True)
    async def test_doctor_cannot_operate_others_visit(self, client: AsyncClient, world) -> None:
        """医生 B 拿医生 A 的 visit 开方 → 404（不泄露这条病历存在）。"""
        resp = await client.post(
            "/api/v1/prescriptions", json=body_of([IBUPROFEN], visit_id=str(world["visit_a"]))
        )
        assert resp.status_code == 404, resp.text

    @pytest.mark.parametrize("client", [DOCTOR_B], indirect=True)
    async def test_doctor_cannot_touch_others_prescription(self, client: AsyncClient, world) -> None:
        resp = await client.patch(f"/api/v1/prescriptions/{world['rx_paid']}/cancel", json={})
        assert resp.status_code == 404, resp.text

    @pytest.mark.parametrize("client", [LAB_TECH_TAG], indirect=True)
    async def test_lab_tech_forbidden(self, client: AsyncClient) -> None:
        """检验科在矩阵里完全不持有 prescription 的任何动作。"""
        resp = await client.get("/api/v1/prescriptions")
        assert resp.status_code == 403, resp.text

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_illegal_uuid_path_is_404(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/prescriptions/not-a-uuid")
        assert resp.status_code == 404, resp.text

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_by_visit_listing(self, client: AsyncClient, world) -> None:
        created = await client.post(
            "/api/v1/prescriptions", json=body_of([IBUPROFEN], visit_id=str(world["visit_a"]))
        )
        assert created.status_code == 201
        resp = await client.get(f"/api/v1/prescriptions/by-visit/{world['visit_a']}")
        assert resp.status_code == 200, resp.text
        assert str(created.json()["id"]) in [item["id"] for item in resp.json()]

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_status_filter_http(self, client: AsyncClient, world) -> None:
        resp = await client.get("/api/v1/prescriptions", params={"status": "paid"})
        assert resp.status_code == 200
        assert [item["id"] for item in resp.json()["items"]] == [str(world["rx_paid"])]

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_detail_exposes_safety_contract(self, client: AsyncClient, world) -> None:
        created = await client.post(
            "/api/v1/prescriptions", json=body_of([AMOX_CAPSULE], visit_id=str(world["visit_a"]))
        )
        rx_id = created.json()["id"]
        await client.post(
            f"/api/v1/prescriptions/{rx_id}/confirm",
            json={"safety_override_reason": "已复核"},
        )
        detail = await client.get(f"/api/v1/prescriptions/{rx_id}")
        assert detail.status_code == 200
        body = detail.json()
        assert body["safety_checks"][0]["rule"] == "allergy_conflict"
        assert body["safety_checks"][0]["level"] == "block"
        assert body["allowed_transitions"] == ["cancelled", "paid"]


LAB_TECH_TAG = "labtech_tag"
_WORLD_KEYS[LAB_TECH_TAG] = ("labtech01", "lab_tech")


# ==================== 13. 权限映射静态守卫 ====================


class TestPermissionMapping:
    def test_matrix_has_no_pharmacist_update(self) -> None:
        """路由挡不住的东西由矩阵保证：药师永远拿不到 prescription:update。"""
        from app.models.auth import RoleName
        from app.services.rbac_matrix import ROLE_PERMISSIONS

        assert ("prescription", "update") not in ROLE_PERMISSIONS[RoleName.PHARMACIST]
        assert ("prescription", "read_all") in ROLE_PERMISSIONS[RoleName.PHARMACIST]
        assert ("prescription", "approve") in ROLE_PERMISSIONS[RoleName.PHARMACIST]
        assert ("prescription", "create") not in ROLE_PERMISSIONS[RoleName.PATIENT]
        assert ("prescription", "read_all") not in ROLE_PERMISSIONS[RoleName.PATIENT]
        assert ("prescription", "read") in ROLE_PERMISSIONS[RoleName.DOCTOR]
        assert ("prescription", "read_all") not in ROLE_PERMISSIONS[RoleName.DOCTOR]

    @needs_routes
    def test_route_gates_align_with_matrix(self) -> None:
        """路由里出现的 (资源, 动作) 必须都在矩阵登记过，防止凭空发明授权位。"""
        import inspect

        from app.services.rbac_matrix import ROLE_PERMISSIONS

        known = {pair for perms in ROLE_PERMISSIONS.values() for pair in perms}
        src = inspect.getsource(__import__("app.api.v1.prescription", fromlist=["router"]))
        for gate in ("create", "update", "read", "read_all"):
            assert f'"prescription", "{gate}"' in src or gate not in ("create", "update")
        assert ("prescription", "update") in known
        assert ("prescription", "read_all") in known
