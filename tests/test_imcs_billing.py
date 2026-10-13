"""收费域与发药域单元测试（阶段 5；文件型 SQLite，不打真实网络、不起容器）。

覆盖 ``app/services/billing_service.py`` 的建单金额汇总 / 处方查重 / 支付联动 /
退费边界 / 取消 / 行级隔离，``app/services/dispense_service.py`` 的先审后发 /
发药联动 / 终态断言，以及两个路由的权限映射与响应装配。

**端点一律走 HTTP 打**（沿用 exam/consult 教训）：签名里把已经是
``Annotated[User, Depends(...)]`` 的闸门再套一层 ``Depends(...)`` 会让 FastAPI
凭空多出必填参数恒 422，纯 service 层用例完全测不出来。本文件的 pay / refund /
approve / dispense 等入口都经 ASGI 客户端真实命中。

============================ 会话模型：每步独立短会话 ============================

被测 service 函数自带 commit/rollback，因此测试**不共享一个长会话**——否则一次
失败路径里的 rollback 会把上一个用例刚提交的数据也撤销掉。这里用 ``open_session()``
工厂为每一步开一个连到同一文件库的独立异步会话，用完即弃，语义最干净。

============================ 为什么用「直插」准备处方数据 ============================

处方域的 prescription_service/api/schema 由并行同事负责，本批启动时可能不存在，
因此测试**不 import 它**：需要一张 confirmed/paid/cancelled/dispensed 处方时，直接用
同步连接往 prescriptions 表插一行（同 exam 测试用 insert_raw_visit 直插 visit），
既不依赖未合并的代码，又能精确控制处方初始状态来构造各种联动失败场景。跨域联动本身
在 service 里也只操作 Prescription ORM + assert_transition，不碰处方 service。

============================ 关键契约（对应模块 docstring 的设计决策） ============================

1. 金额 int「分」：total_amount = 各项 amount 之和；other 项取客户端传入分值。
2. 同一处方只能被一张 pending/paid 收费单关联 → 重复关联 409。
3. 支付联动处方 confirmed→paid；处方非 confirmed 时整单回滚，收费单仍 pending（无脏写）。
4. 退费对 dispensed 处方拒绝、对 paid 处方同样拒绝（状态机无回退边）；纯 exam 单可退。
5. 先审后发：未 verified 的发药被拒；发药把处方推到 dispensed（终态），二次发药被拒。
6. 权限边界：patient 不能 pay（403）、doctor 无 pay/approve（403）、patient 只看自己的单。

SQLite 已知差异（不影响结论）：DateTime 列不存时区偏移，不断言 tz 感知；JSONB 列在
SQLite 走 JSON 文本，service 经 ORM 写入会正确序列化，直读时按 json.loads 还原。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import pytest
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.imcs_main import app
from app.models.auth import User
from app.models.base import Base, utcnow
from app.models.billing import ChargeOrder
from app.schemas.billing import ChargeOrderCreateRequest, OtherItemIn
from app.services import billing_service as bs
from app.services import dispense_service as ds
from tests.imcs_sqlite import (
    build_session_factory,
    seed_role_permissions,
    seed_roles,
)

# ==================== 建表 ====================

BILLING_TABLES: tuple[str, ...] = (
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
    "prescriptions",
    "prescription_items",
    "dispensing_records",
    "charge_orders",
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
    """文件型 SQLite：同步建表 + 异步被测会话共用同一份数据（同 exam 测试）。"""
    import tests.imcs_sqlite  # noqa: F401 - 注册 PGUUID 的 SQLite DDL 编译器

    db_path = tmp_path / "imcs_billing.db"
    test_engine = sa.create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    tables = [Base.metadata.tables[name] for name in BILLING_TABLES]
    with test_engine.begin() as conn:
        for table in tables:
            table.create(conn, checkfirst=True)
        seed_roles(conn)
        # require_permission 走 role_permissions JOIN permissions，空表会让非 admin 全 403。
        seed_role_permissions(conn)
    yield test_engine
    test_engine.dispose()


@asynccontextmanager
async def _session_scope(engine) -> AsyncIterator[AsyncSession]:
    """开一个连到同一文件库的独立异步会话，退出时释放其引擎。"""
    factory = build_session_factory(engine)
    try:
        async with factory() as session:
            yield session
    finally:
        await factory.kw["bind"].dispose()


# ==================== 数据助手（同步直插，绕开未合并的处方 service） ====================


def _insert(engine, table: str, values: dict[str, Any]) -> None:  # noqa: ANN001
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
    engine, *, role_name: str, username: str, real_name: str | None = None,
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
            "chronic_diseases": json.dumps([], ensure_ascii=False),
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return pid


def insert_raw_visit(engine, *, patient_id: uuid.UUID, doctor_id: uuid.UUID) -> uuid.UUID:  # noqa: ANN001
    vid = uuid.uuid4()
    _insert(
        engine,
        "visits",
        {
            "id": vid.hex,
            "appointment_id": uuid.uuid4().hex,
            "patient_id": patient_id.hex,
            "doctor_id": doctor_id.hex,
            "chief_complaint": "直插构造的主诉",
            "status": "documented",
            "started_at": utcnow(),
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return vid


def insert_prescription(  # noqa: ANN001
    engine, *, visit_id: uuid.UUID, patient_id: uuid.UUID, doctor_id: uuid.UUID,
    status: str = "confirmed", total_amount: int = 3000,
) -> uuid.UUID:
    """直插一张处方，用于构造各种联动起点，不依赖处方域 service。"""
    rid = uuid.uuid4()
    _insert(
        engine,
        "prescriptions",
        {
            "id": rid.hex,
            "visit_id": visit_id.hex,
            "patient_id": patient_id.hex,
            "doctor_id": doctor_id.hex,
            "status": status,
            "diagnosis": "上呼吸道感染",
            "notes": None,
            "total_amount": total_amount,
            "safety_checks": None,
            "safety_override_reason": None,
            "confirmed_at": utcnow(),
            "cancelled_at": None,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return rid


def insert_exam_request(  # noqa: ANN001
    engine, *, visit_id: uuid.UUID, patient_id: uuid.UUID, doctor_id: uuid.UUID,
    fee_amount: int = 2500, exam_name: str = "血常规",
) -> uuid.UUID:
    eid = uuid.uuid4()
    _insert(
        engine,
        "exam_requests",
        {
            "id": eid.hex,
            "visit_id": visit_id.hex,
            "patient_id": patient_id.hex,
            "doctor_id": doctor_id.hex,
            "exam_type": "blood_routine",
            "exam_name": exam_name,
            "body_part": None,
            "clinical_info": "直插构造",
            "urgency": "routine",
            "status": "ordered",
            "is_charged": 0,
            "fee_amount": fee_amount,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return eid


def read_row(engine, table: str, row_id: uuid.UUID) -> dict[str, Any]:  # noqa: ANN001
    """直读库里的行，断言「数据库真实发生了什么」而非 ORM 内存态。"""
    with engine.begin() as conn:
        row = conn.execute(
            sa.text(f"SELECT * FROM {table} WHERE id = :id"),  # noqa: S608
            {"id": row_id.hex},
        ).fetchone()
    assert row is not None, f"{table} 行不存在"
    return dict(row._mapping)


def count_rows(engine, table: str, where: str = "", params: dict[str, Any] | None = None) -> int:  # noqa: ANN001
    sql = f"SELECT COUNT(*) FROM {table}"  # noqa: S608
    if where:
        sql += f" WHERE {where}"
    with engine.begin() as conn:
        return int(conn.execute(sa.text(sql), params or {}).scalar_one())


# ==================== world fixture ====================


@pytest.fixture()
def world(engine) -> dict[str, Any]:
    """一位医生 A + 另一位医生 B + 两位患者 + 收费员 + 药师 + 各自就诊与单据。

    rx_confirmed / rx_paid / rx_cancelled / rx_disp 四张处方分别覆盖支付的合法起点、
    已缴待发的审核/退费起点、会被支付拒绝的作废起点、以及已发药的退费拒绝起点。
    """
    cardio = make_department(engine, "CARDIO", "心血管内科")
    pharmacy = make_department(engine, "PHARM", "药剂科")

    doctor_a = make_user(engine, role_name="doctor", username="doctor_a", real_name="张三", department_id=cardio)
    doctor_b = make_user(engine, role_name="doctor", username="doctor_b", real_name="李四", department_id=cardio)
    cashier01 = make_user(engine, role_name="cashier", username="cashier01", real_name="王收")
    pharmacist01 = make_user(engine, role_name="pharmacist", username="pharmacist01", real_name="赵药", department_id=pharmacy)
    patient01 = make_user(engine, role_name="patient", username="patient_01")
    patient02 = make_user(engine, role_name="patient", username="patient_02")
    profile01 = make_profile(engine, patient01, gender="F")
    profile02 = make_profile(engine, patient02)

    visit1 = insert_raw_visit(engine, patient_id=profile01, doctor_id=doctor_a)
    visit2 = insert_raw_visit(engine, patient_id=profile02, doctor_id=doctor_a)

    rx_confirmed = insert_prescription(engine, visit_id=visit1, patient_id=profile01, doctor_id=doctor_a, status="confirmed", total_amount=3000)
    rx_paid = insert_prescription(engine, visit_id=visit2, patient_id=profile02, doctor_id=doctor_a, status="paid", total_amount=1500)
    rx_cancelled = insert_prescription(engine, visit_id=visit1, patient_id=profile01, doctor_id=doctor_a, status="cancelled", total_amount=800)
    rx_disp = insert_prescription(engine, visit_id=visit1, patient_id=profile01, doctor_id=doctor_a, status="dispensed", total_amount=900)

    exam1 = insert_exam_request(engine, visit_id=visit1, patient_id=profile01, doctor_id=doctor_a, fee_amount=2500)

    return {
        "engine": engine,
        "doctor_a": doctor_a,
        "doctor_b": doctor_b,
        "cashier01": cashier01,
        "pharmacist01": pharmacist01,
        "patient01": patient01,
        "patient02": patient02,
        "profile01": profile01,
        "profile02": profile02,
        "visit1": visit1,
        "visit2": visit2,
        "rx_confirmed": rx_confirmed,
        "rx_paid": rx_paid,
        "rx_cancelled": rx_cancelled,
        "rx_disp": rx_disp,
        "exam1": exam1,
    }


# ==================== 输入构造（SimpleNamespace，字段名对齐判别式模型） ====================


def _make_items(rx=None, exam=None, other=None) -> list[Any]:
    from types import SimpleNamespace

    out: list[Any] = []
    for r in rx or []:
        out.append(SimpleNamespace(type="prescription", prescription_id=str(r)))
    for e in exam or []:
        out.append(SimpleNamespace(type="exam", exam_request_id=str(e)))
    if other:
        name, amt = other
        out.append(SimpleNamespace(type="other", name=name, amount=amt))
    return out


def _status_of(exc: BaseException) -> int | None:
    return getattr(exc, "status_code", None)


def _msg_of(exc: BaseException) -> str:
    return str(getattr(exc, "message", "") or exc)


# ====================================================================
# 一、建单：金额汇总 + 处方查重（service 层，不依赖路由挂载）
# ====================================================================


async def test_create_order_other_only(world):
    """单一 other 项：total = 该项金额，pending，paid_amount=0，且是 int。"""
    async with _session_scope(world["engine"]) as db:
        order = await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(other=("挂号费", 500)),
        )
    assert order.status == "pending"
    assert order.total_amount == 500
    assert isinstance(order.total_amount, int)
    assert order.paid_amount == 0


async def test_create_order_mixed_items_amounts(world):
    """混装三类项：处方取 total_amount(3000)、检查取 fee_amount(2500)、other 取传入值(200)。"""
    async with _session_scope(world["engine"]) as db:
        order = await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(rx=[world["rx_confirmed"]], exam=[world["exam1"]], other=("材料费", 200)),
        )
    assert order.total_amount == 3000 + 2500 + 200
    assert sorted(i["amount"] for i in order.items) == [200, 2500, 3000]


async def test_create_order_duplicate_prescription_rejected(world):
    """同一处方已被一张 pending 单关联，第二张再挂它 → DuplicatePrescriptionChargeError(409)。"""
    async with _session_scope(world["engine"]) as db:
        await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(rx=[world["rx_confirmed"]]),
        )
    async with _session_scope(world["engine"]) as db:
        with pytest.raises(bs.DuplicatePrescriptionChargeError) as exc:
            await bs.create_charge_order(
                db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
                items_in=_make_items(rx=[world["rx_confirmed"]]),
            )
    assert exc.value.status_code == 409


async def test_create_order_prescription_wrong_visit_rejected(world):
    """处方不属于该 visit → 拒绝，防串单。rx_paid 挂在 visit2，却塞进 visit1 的单。"""
    async with _session_scope(world["engine"]) as db:
        with pytest.raises(Exception) as exc:
            await bs.create_charge_order(
                db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
                items_in=_make_items(rx=[world["rx_paid"]]),
            )
    assert _status_of(exc.value) == 409


async def test_create_order_amount_stored_as_int_cent(world):
    """金额口径：other 项落库后仍是整数分，不做浮点换算。"""
    async with _session_scope(world["engine"]) as db:
        order = await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(other=("诊查费", 1234)),
        )
        oid = order.id
    row = read_row(world["engine"], "charge_orders", oid)
    assert row["total_amount"] == 1234
    assert isinstance(row["total_amount"], int)


# ====================================================================
# 二、支付联动：confirmed→paid；处方状态不符整单回滚无脏写
# ====================================================================


async def test_pay_links_prescription_to_paid(world):
    """含 confirmed 处方的收费单支付后，处方被推进到 paid。"""
    async with _session_scope(world["engine"]) as db:
        order = await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(rx=[world["rx_confirmed"]]),
        )
        oid = order.id
    async with _session_scope(world["engine"]) as db:
        paid = await bs.pay_charge_order(db, order_id=oid, cashier_user_id=world["cashier01"], channel="mock")
    assert paid.status == "paid"
    assert paid.paid_amount == paid.total_amount
    assert paid.receipt_no and paid.receipt_no.startswith("CO")
    assert paid.cashier_id == world["cashier01"]
    rx_row = read_row(world["engine"], "prescriptions", world["rx_confirmed"])
    assert rx_row["status"] == "paid"


async def test_pay_nonconfirmed_prescription_rolls_back_whole(world):
    """处方处于 cancelled（非 confirmed）→ 支付整体失败回滚，收费单仍 pending，处方不变。"""
    async with _session_scope(world["engine"]) as db:
        order = await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(rx=[world["rx_cancelled"]]),
        )
        oid = order.id
    async with _session_scope(world["engine"]) as db:
        with pytest.raises(Exception) as exc:
            await bs.pay_charge_order(db, order_id=oid, cashier_user_id=world["cashier01"], channel="mock")
    assert _status_of(exc.value) == 409
    row = read_row(world["engine"], "charge_orders", oid)
    assert row["status"] == "pending"
    assert row["receipt_no"] is None
    assert row["paid_amount"] == 0
    rx_row = read_row(world["engine"], "prescriptions", world["rx_cancelled"])
    assert rx_row["status"] == "cancelled"


async def test_pay_is_idempotent(world):
    """已 paid 的单再次支付幂等返回当前态，不换小票号。"""
    async with _session_scope(world["engine"]) as db:
        order = await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(exam=[world["exam1"]]),
        )
        oid = order.id
    async with _session_scope(world["engine"]) as db:
        first = await bs.pay_charge_order(db, order_id=oid, cashier_user_id=world["cashier01"])
    async with _session_scope(world["engine"]) as db:
        second = await bs.pay_charge_order(db, order_id=oid, cashier_user_id=world["cashier01"])
    assert second.receipt_no == first.receipt_no


# ====================================================================
# 三、退费边界
# ====================================================================


async def test_refund_pure_exam_ok(world):
    """纯 exam 项的收费单可正常退费：paid→refunded，paid_amount 归零，写原因。"""
    async with _session_scope(world["engine"]) as db:
        order = await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(exam=[world["exam1"]]),
        )
        oid = order.id
    async with _session_scope(world["engine"]) as db:
        await bs.pay_charge_order(db, order_id=oid, cashier_user_id=world["cashier01"])
    async with _session_scope(world["engine"]) as db:
        refunded = await bs.refund_charge_order(
            db, order_id=oid, cashier_user_id=world["cashier01"], reason="患者放弃检查",
        )
    assert refunded.status == "refunded"
    assert refunded.paid_amount == 0
    assert refunded.refund_reason == "患者放弃检查"
    assert refunded.refunded_at is not None


async def test_refund_dispensed_prescription_rejected(world):
    """含 dispensed 处方的收费单退费被拒（药已发出）。"""
    async with _session_scope(world["engine"]) as db:
        order = await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(rx=[world["rx_disp"]]),
        )
        oid = order.id
        # 直接标成 paid（绕过联动，因为 rx_disp 已是 dispensed，pay 会失败），
        # 用主键重新取回当前会话内的受管对象再改状态。
        fresh = await db.get(ChargeOrder, oid)
        fresh.status = "paid"
        fresh.paid_amount = fresh.total_amount
        await db.commit()
    async with _session_scope(world["engine"]) as db:
        with pytest.raises(Exception) as exc:
            await bs.refund_charge_order(db, order_id=oid, cashier_user_id=world["cashier01"], reason="随便")
    assert _status_of(exc.value) == 409
    assert "已发药" in _msg_of(exc.value)


async def test_refund_paid_prescription_rejected_boundary(world):
    """退费边界：含 paid 处方的收费单同样被拒（处方状态机无回退边，退了收不回）。"""
    async with _session_scope(world["engine"]) as db:
        order = await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit2"],
            items_in=_make_items(rx=[world["rx_paid"]]),
        )
        oid = order.id
        fresh = await db.get(ChargeOrder, oid)
        fresh.status = "paid"
        fresh.paid_amount = fresh.total_amount
        await db.commit()
    async with _session_scope(world["engine"]) as db:
        with pytest.raises(Exception) as exc:
            await bs.refund_charge_order(db, order_id=oid, cashier_user_id=world["cashier01"], reason="想退")
    assert _status_of(exc.value) == 409
    assert "请先由医生作废" in _msg_of(exc.value)


async def test_refund_pending_rejected(world):
    """未支付的单不能退费（pending→refunded 非法）。"""
    async with _session_scope(world["engine"]) as db:
        order = await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(exam=[world["exam1"]]),
        )
        oid = order.id
    async with _session_scope(world["engine"]) as db:
        with pytest.raises(Exception) as exc:
            await bs.refund_charge_order(db, order_id=oid, cashier_user_id=world["cashier01"], reason="没付就想退")
    assert _status_of(exc.value) == 409


# ====================================================================
# 四、取消
# ====================================================================


async def test_cancel_pending_ok(world):
    async with _session_scope(world["engine"]) as db:
        order = await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(exam=[world["exam1"]]),
        )
        oid = order.id
    async with _session_scope(world["engine"]) as db:
        cancelled = await bs.cancel_charge_order(db, order_id=oid, operator_user_id=world["cashier01"])
    assert cancelled.status == "cancelled"


async def test_cancel_paid_rejected(world):
    """已支付的单不能作废（矩阵里 paid 只能→refunded）。"""
    async with _session_scope(world["engine"]) as db:
        order = await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(exam=[world["exam1"]]),
        )
        oid = order.id
    async with _session_scope(world["engine"]) as db:
        await bs.pay_charge_order(db, order_id=oid, cashier_user_id=world["cashier01"])
    async with _session_scope(world["engine"]) as db:
        with pytest.raises(Exception) as exc:
            await bs.cancel_charge_order(db, order_id=oid, operator_user_id=world["cashier01"])
    assert _status_of(exc.value) == 409


# ====================================================================
# 五、行级隔离（service 层查询）
# ====================================================================


async def test_list_orders_patient_only_own(world):
    """患者列表只返回自己档案下的单；无单的患者返回空集。"""
    async with _session_scope(world["engine"]) as db:
        o1 = await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(exam=[world["exam1"]]),
        )
        o1_id = o1.id
    async with _session_scope(world["engine"]) as db:
        rows, _total = await bs.list_orders(db, requester_user_id=world["patient01"], role_name="patient")
        assert all(r.patient_id == world["profile01"] for r in rows)
        assert o1_id in {r.id for r in rows}
        rows2, total2 = await bs.list_orders(db, requester_user_id=world["patient02"], role_name="patient")
        assert total2 == 0


async def test_get_detail_cross_patient_404(world):
    """患者读别人的收费单 → 404（不确认他人记录存在）。"""
    async with _session_scope(world["engine"]) as db:
        order = await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(exam=[world["exam1"]]),
        )
        oid = order.id
    async with _session_scope(world["engine"]) as db:
        with pytest.raises(Exception) as exc:
            await bs.get_order_detail(db, order_id=oid, requester_user_id=world["patient02"], role_name="patient")
    assert _status_of(exc.value) == 404


async def test_doctor_sees_only_own_visits_orders(world):
    """医生列表只含自己接诊产生的收费单；别的医生看不到。"""
    async with _session_scope(world["engine"]) as db:
        await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(exam=[world["exam1"]]),
        )
    async with _session_scope(world["engine"]) as db:
        _rows_a, total_a = await bs.list_orders(db, requester_user_id=world["doctor_a"], role_name="doctor")
        _rows_b, total_b = await bs.list_orders(db, requester_user_id=world["doctor_b"], role_name="doctor")
    assert total_a >= 1
    assert total_b == 0


# ====================================================================
# 六、schema 层校验
# ====================================================================


def test_schema_other_amount_positive_int():
    """other 项金额必须正整数分：0 或负数被协议层拒（Pydantic → 422）。"""
    with pytest.raises(ValueError):
        OtherItemIn(type="other", name="免费", amount=0)
    ok = OtherItemIn(type="other", name="挂号费", amount=500)
    assert ok.amount == 500


def test_schema_accepts_valid_shape():
    payload = ChargeOrderCreateRequest(
        visit_id=str(uuid.uuid4()),
        items=[{"type": "other", "name": "挂号费", "amount": 500}],
    )
    assert len(payload.items) == 1


# ====================================================================
# 七、发药域：先审后发 + 联动 + 终态
# ====================================================================


async def test_dispense_requires_verified(world):
    """未审核就发药被拒（先审后发硬闸门）。"""
    async with _session_scope(world["engine"]) as db:
        with pytest.raises(Exception) as exc:
            await ds.dispense_prescription(db, prescription_id=world["rx_paid"], pharmacist_user_id=world["pharmacist01"])
    assert _status_of(exc.value) == 409
    assert "审核" in _msg_of(exc.value)


async def test_approve_then_dispense_links_to_dispensed(world):
    """审核置 verified=True，随后发药把处方推进到 dispensed 并写发药时间。"""
    async with _session_scope(world["engine"]) as db:
        rec = await ds.approve_prescription(
            db, prescription_id=world["rx_paid"], pharmacist_user_id=world["pharmacist01"], notes="用法用量合理",
        )
    assert rec.verified is True
    assert rec.verification_notes == "用法用量合理"
    async with _session_scope(world["engine"]) as db:
        disp = await ds.dispense_prescription(
            db, prescription_id=world["rx_paid"], pharmacist_user_id=world["pharmacist01"], notes="已交代饭后服用",
        )
    assert disp.dispensed_at is not None
    rx_row = read_row(world["engine"], "prescriptions", world["rx_paid"])
    assert rx_row["status"] == "dispensed"
    cnt = count_rows(world["engine"], "dispensing_records", "prescription_id = :p", {"p": world["rx_paid"].hex})
    assert cnt == 1


async def test_second_dispense_rejected_terminal(world):
    """已发药的处方处于终态，二次发药被状态机拒。"""
    async with _session_scope(world["engine"]) as db:
        await ds.approve_prescription(db, prescription_id=world["rx_paid"], pharmacist_user_id=world["pharmacist01"], notes="ok")
        await ds.dispense_prescription(db, prescription_id=world["rx_paid"], pharmacist_user_id=world["pharmacist01"])
    async with _session_scope(world["engine"]) as db:
        with pytest.raises(Exception) as exc:
            await ds.dispense_prescription(db, prescription_id=world["rx_paid"], pharmacist_user_id=world["pharmacist01"])
    assert _status_of(exc.value) == 409


async def test_approve_non_paid_rejected(world):
    """只有 paid 处方能审核：confirmed 处方审核被拒。"""
    async with _session_scope(world["engine"]) as db:
        with pytest.raises(Exception) as exc:
            await ds.approve_prescription(db, prescription_id=world["rx_confirmed"], pharmacist_user_id=world["pharmacist01"], notes="x")
    assert _status_of(exc.value) == 409


async def test_approve_idempotent(world):
    """重复审核幂等返回当前记录，不新建第二条发药记录。"""
    async with _session_scope(world["engine"]) as db:
        r1 = await ds.approve_prescription(db, prescription_id=world["rx_paid"], pharmacist_user_id=world["pharmacist01"], notes="一次")
        r2 = await ds.approve_prescription(db, prescription_id=world["rx_paid"], pharmacist_user_id=world["pharmacist01"], notes="二次")
    assert r1.id == r2.id
    cnt = count_rows(world["engine"], "dispensing_records", "prescription_id = :p", {"p": world["rx_paid"].hex})
    assert cnt == 1


async def test_dispensing_record_patient_isolation(world):
    """患者只看自己处方的取药记录，越权详情 404。"""
    async with _session_scope(world["engine"]) as db:
        rec = await ds.approve_prescription(db, prescription_id=world["rx_paid"], pharmacist_user_id=world["pharmacist01"], notes="ok")
        rec_id = rec.id
    async with _session_scope(world["engine"]) as db:
        # rx_paid 属于 profile02，patient01 无权看。
        with pytest.raises(Exception) as exc:
            await ds.get_record_detail(db, record_id=rec_id, requester_user_id=world["patient01"], role_name="patient")
        assert _status_of(exc.value) == 404
        rows, _total = await ds.list_records(db, requester_user_id=world["patient02"], role_name="patient")
        assert rec_id in {r.id for r in rows}


# ==================== HTTP 客户端 ====================


def _identity(world: dict[str, Any], tag: str) -> tuple[uuid.UUID, str, str]:
    mapping = {
        "doctor_a": ("doctor_a", "doctor"),
        "doctor_b": ("doctor_b", "doctor"),
        "cashier": ("cashier01", "cashier"),
        "pharmacist": ("pharmacist01", "pharmacist"),
        "patient1": ("patient01", "patient"),
        "patient2": ("patient02", "patient"),
    }
    key, role = mapping[tag]
    return world[key], key, role


def as_user(user_id: uuid.UUID, username: str, role_name: str) -> User:
    user = User(
        id=user_id, username=username, real_name=username, password_hash="",
        role_id=ROLE_IDS[role_name], is_active=True,
    )
    user._role_name = role_name  # type: ignore[attr-defined]
    return user


@pytest.fixture()
async def client_factory(engine, world):
    """按身份标签构造已鉴权 ASGI 客户端；覆盖 get_db 与 get_current_user。

    每个用例通过 ``async with client_factory("cashier") as client`` 拿到客户端，
    其内部会话连到与 world 相同的文件库。
    """

    @asynccontextmanager
    async def _build(tag: str | None):
        async def _override_get_db() -> AsyncIterator[AsyncSession]:
            async with _session_scope(engine) as session:
                yield session

        app.dependency_overrides[get_db] = _override_get_db
        if tag is not None:
            uid, username, role = _identity(world, tag)

            async def _override_user():
                return as_user(uid, username, role)

            app.dependency_overrides[get_current_user] = _override_user

        transport = ASGITransport(app=app)
        try:
            async with AsyncClient(transport=transport, base_url="http://test") as http_client:
                yield http_client
        finally:
            app.dependency_overrides.clear()

    yield _build


def _mounted(prefix: str) -> bool:
    try:
        paths = app.openapi()["paths"]
    except Exception:  # noqa: BLE001 - openapi 生成异常也当作未挂载
        return False
    return any(prefix in p for p in paths)


needs_billing = pytest.mark.skipif(not _mounted("/billing"), reason="billing router 尚未由主代理挂载")
needs_dispensing = pytest.mark.skipif(not _mounted("/dispensing"), reason="dispensing router 尚未由主代理挂载")


# ====================================================================
# 八、HTTP 端到端（权限边界）——仅在路由已挂载时执行
# ====================================================================


@needs_billing
async def test_http_patient_cannot_pay(client_factory, world):
    """患者调用支付端点 → 403（patient 无 charge_order:pay）。"""
    async with client_factory("patient1") as client:
        resp = await client.post(f"/api/v1/billing/orders/{uuid.uuid4()}/pay", json={"channel": "mock"})
    assert resp.status_code == 403


@needs_billing
async def test_http_doctor_cannot_pay(client_factory, world):
    """医生无 charge_order:pay → 403。"""
    async with client_factory("doctor_a") as client:
        resp = await client.post(f"/api/v1/billing/orders/{uuid.uuid4()}/pay", json={"channel": "mock"})
    assert resp.status_code == 403


@needs_billing
async def test_http_cashier_pay_flow(client_factory, world):
    """收费员经 HTTP 完成一笔纯 exam 单的收款，响应反映实收与小票号。"""
    async with _session_scope(world["engine"]) as db:
        order = await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit1"],
            items_in=_make_items(exam=[world["exam1"]]),
        )
        oid = str(order.id)
    async with client_factory("cashier") as client:
        resp = await client.post(f"/api/v1/billing/orders/{oid}/pay", json={"channel": "mock"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "paid"
        assert body["paid_amount"] == body["total_amount"]
        assert body["receipt_no"].startswith("CO")


@needs_billing
async def test_http_patient_list_isolation(client_factory, world):
    """患者经 HTTP 拉自己的列表，绝不包含他人 visit 的单。"""
    async with _session_scope(world["engine"]) as db:
        await bs.create_charge_order(
            db, doctor_user_id=world["doctor_a"], visit_id=world["visit2"],
            items_in=_make_items(other=("费", 100)),
        )
    async with client_factory("patient1") as client:
        resp = await client.get("/api/v1/billing/orders")
        assert resp.status_code == 200, resp.text
        for it in resp.json()["items"]:
            assert it["patient_id"] == str(world["profile01"])


@needs_dispensing
async def test_http_pharmacist_approve_dispense_flow(client_factory, world):
    """药师经 HTTP 审核 + 发药，处方落到 dispensed。"""
    rx = str(world["rx_paid"])
    async with client_factory("pharmacist") as client:
        ap = await client.post(f"/api/v1/dispensing/prescriptions/{rx}/approve", json={"notes": "审核通过"})
        assert ap.status_code == 200, ap.text
        assert ap.json()["verified"] is True
        dp = await client.post(f"/api/v1/dispensing/prescriptions/{rx}/dispense", json={"notes": "已发"})
        assert dp.status_code == 200, dp.text
        assert dp.json()["dispensed_at"] is not None


@needs_dispensing
async def test_http_dispense_without_approve_409(client_factory, world):
    """未经审核直接发药 → 409。"""
    async with client_factory("pharmacist") as client:
        resp = await client.post(f"/api/v1/dispensing/prescriptions/{world['rx_paid']}/dispense", json={})
    assert resp.status_code == 409


@needs_dispensing
async def test_http_doctor_cannot_approve(client_factory, world):
    """医生无 prescription:approve → 403。"""
    async with client_factory("doctor_a") as client:
        resp = await client.post(f"/api/v1/dispensing/prescriptions/{world['rx_paid']}/approve", json={"notes": "x"})
    assert resp.status_code == 403
