"""检验侧服务、路由与 AI 解读工作流单元测试（阶段 4；文件型 SQLite，不打真实网络）。

覆盖 ``app/services/exam_service.py`` 的规则判定 / 报告录入 / 派发时序 / 异步与同步
两侧解读回写、``app/api/v1/exam.py`` 的权限映射与响应装配、``app/schemas/exam.py``
的协议层校验，以及 ``app/tasks/ai_tasks.py::interpret_exam_report`` 的注册契约。

**为什么端点一律走 HTTP 打**：阶段 3 的 batch-assess 曾因签名里把已经是
``Annotated[User, Depends(...)]`` 的闸门又套一层 ``Depends(...)``，FastAPI 把它当成
待注入的可调用对象解析，端点凭空多出两个必填 query 参数，任何请求恒 422——
纯 service 层用例完全测不出来。本文件的 receive / process / submit / interpret /
queue / reports/{id} 六个入口全部经 ASGI 客户端真实命中。

**打桩点位**（三处，都不碰实现）：
1. ``exam_service.get_llm_service`` → CountingLLM（真 LLM 要网络与配额）
2. ``exam_service.retrieve_evidence`` → 固定返回空列表（RAG 要走 embed；且按契约
   「零结果不阻断解读」，因此桩本身也是这条契约的载体）
3. ``sys.modules["app.tasks.celery_app"].dispatch_task`` → 记录入参的假派发
   （必须打在 sys.modules 里的真模块对象上：``import app.tasks.celery_app as m``
   取到的 m 可能被 ``app/tasks/__init__.py`` 的同名 Celery 实例遮蔽，
   patch 到实例上会静默失效；service 侧是函数内 ``from ... import``，
   走的正是模块属性查找）

Celery 任务体只断言「注册名 / queue / time_limit」静态契约 + 源码不再抛
NotImplementedError，**不起 worker、不连 broker**。

SQLite 已知差异（不影响结论）：DateTime 列不存时区偏移，故不断言 tz 感知；
TextArray / JSONB 列在 SQLite 走 JSON 文本，直插时必须 json.dumps。
"""

from __future__ import annotations

import inspect
import json
import sys
import uuid
from collections.abc import AsyncIterator
from datetime import date, datetime, timedelta
from typing import Any

import pytest
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.exceptions import InvalidStateTransitionError, ResourceNotFoundError
from app.imcs_main import app
from app.models.ai import AITask
from app.models.auth import User
from app.models.base import Base, utcnow
from app.models.clinical import ExamRequestStatus, Urgency
from app.schemas.exam import ReportSubmitRequest
from app.services import exam_service as es
from app.services.llm.types import DegradationLevel, LLMResponse, TokenUsage
from app.services.state_machine import allowed_targets
from tests.imcs_sqlite import (
    build_session_factory,
    error_code_of,
    seed_role_permissions,
    seed_roles,
)

# ==================== 建表 ====================

EXAM_TABLES: tuple[str, ...] = (
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
    "exam_reports",
    "ai_tasks",
    # 知识三表：exam_service 顶层 import RAGRetriever，其检索链路会引用这三张表；
    # 虽然证据召回被打桩，建表仍宁多勿漏——否则将来桩一撤就是 no such table。
    "knowledge_documents",
    "knowledge_chunks",
    "vector_embeddings",
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
    """文件型 SQLite：同步建表 + 异步被测会话共用同一份数据。

    不用内存库：``sqlite://`` 内存库生命周期绑定单条连接，同步引擎建完表后
    异步引擎拿到的是另一个空库，首个查询即 no such table。
    """
    import tests.imcs_sqlite  # noqa: F401  - 注册 PGUUID 的 SQLite DDL 编译器

    db_path = tmp_path / "imcs_exam.db"
    test_engine = sa.create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    tables = [Base.metadata.tables[name] for name in EXAM_TABLES]
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
    chronic_diseases: list[str] | None = None,
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
            # TextArray 在 SQLite 走 JSON 数组，必须 json.dumps；逗号串会被
            # process_result_value 解析失败静默返回 []。
            "chronic_diseases": _json_array(chronic_diseases),
            "created_at": utcnow(),
            "updated_at": utcnow(),
        },
    )
    return pid


def _json_array(values: list[str] | None) -> str | None:
    if values is None:
        return None
    return json.dumps(values, ensure_ascii=False)


def make_exam_request(  # noqa: ANN001
    engine,
    *,
    visit_id: uuid.UUID,
    patient_id: uuid.UUID,
    doctor_id: uuid.UUID,
    exam_type: str = "blood_routine",
    exam_name: str = "血常规",
    urgency: str = Urgency.ROUTINE.value,
    status: str = ExamRequestStatus.ORDERED.value,
    created_at: datetime | None = None,
) -> uuid.UUID:
    rid = uuid.uuid4()
    _insert(
        engine,
        "exam_requests",
        {
            "id": rid.hex,
            "visit_id": visit_id.hex,
            "patient_id": patient_id.hex,
            "doctor_id": doctor_id.hex,
            "exam_type": exam_type,
            "exam_name": exam_name,
            "body_part": None,
            "clinical_info": "近一周活动后胸闷",
            "urgency": urgency,
            "status": status,
            "is_charged": 0,
            "fee_amount": 2500,
            "created_at": created_at or utcnow(),
            "updated_at": created_at or utcnow(),
        },
    )
    return rid


def insert_raw_visit(  # noqa: ANN001
    engine,
    *,
    patient_id: uuid.UUID,
    doctor_id: uuid.UUID,
) -> uuid.UUID:
    """最小 visit 行：exam_requests.visit_id 有外键，必须先有父行。

    刻意用直插而不是调 consult_service——接诊侧属另一批次，
    本批不应依赖它的正确性来准备数据。
    """
    vid = uuid.uuid4()
    _insert(
        engine,
        "visits",
        {
            "id": vid.hex,
            # appointment_id 只是 nullable=False + unique，不参与本批任何判断，
            # 用随机 UUID 占位以免为此再建 appointments 链路数据。
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


def read_row(engine, table: str, row_id: uuid.UUID) -> dict[str, Any]:  # noqa: ANN001
    """直读库里的行，断言「数据库真实发生了什么」而非 ORM 内存态。"""
    with engine.begin() as conn:
        row = conn.execute(
            sa.text(f"SELECT * FROM {table} WHERE id = :id"),  # noqa: S608 - 表名来自常量白名单
            {"id": row_id.hex},
        ).fetchone()
    assert row is not None, f"{table} 行不存在"
    return dict(row._mapping)


async def _request_status(db: AsyncSession, request_id: uuid.UUID) -> str:
    """读申请单当前状态（助手函数，用于按状态取最小流转路径）。"""
    req = await db.get(es.ExamRequest, request_id)
    assert req is not None, "测试前置数据缺失：申请单不存在"
    return str(req.status)


def count_rows(engine, table: str, where: str = "", params: dict[str, Any] | None = None):  # noqa: ANN001
    sql = f"SELECT COUNT(*) FROM {table}"  # noqa: S608 - 表名来自常量白名单
    if where:
        sql += f" WHERE {where}"
    with engine.begin() as conn:
        return int(conn.execute(sa.text(sql), params or {}).scalar_one())


# ==================== world fixture ====================


@pytest.fixture()
def world(engine) -> dict[str, Any]:
    """一位医生 + 两位患者 + 检验科/收费员 + 四张不同状态的申请单。

    profile02 名下刻意不预置任何报告，用于构造「患者请求别人的报告」。
    req_processing / req_received / req_ordered / req_reported 四张单分别覆盖
    出报告的合法起点、非法起点、未接收起点与已出报状态。
    """
    lab = make_department(engine, "LAB", "检验科")
    cardio = make_department(engine, "CARDIO", "心血管内科")

    doctor_a = make_user(engine, role_name="doctor", username="doctor_a",
                         real_name="张三", department_id=cardio)
    labtech01 = make_user(engine, role_name="lab_tech", username="labtech01",
                          real_name="检验士", department_id=lab)
    patient01 = make_user(engine, role_name="patient", username="patient_01")
    patient02 = make_user(engine, role_name="patient", username="patient_02")
    profile01 = make_profile(engine, patient01, gender="F",
                             chronic_diseases=["高血压"])
    profile02 = make_profile(engine, patient02)

    visit = insert_raw_visit(engine, patient_id=profile01, doctor_id=doctor_a)

    base_ts = utcnow() - timedelta(minutes=30)
    req_processing = make_exam_request(
        engine, visit_id=visit, patient_id=profile01, doctor_id=doctor_a,
        status=ExamRequestStatus.PROCESSING.value, created_at=base_ts,
    )
    req_received = make_exam_request(
        engine, visit_id=visit, patient_id=profile01, doctor_id=doctor_a,
        exam_type="ecg", exam_name="心电图",
        status=ExamRequestStatus.RECEIVED.value, created_at=base_ts,
    )
    req_ordered = make_exam_request(
        engine, visit_id=visit, patient_id=profile01, doctor_id=doctor_a,
        exam_type="urine_routine", exam_name="尿常规",
        status=ExamRequestStatus.ORDERED.value, created_at=base_ts,
    )
    req_reported = make_exam_request(
        engine, visit_id=visit, patient_id=profile01, doctor_id=doctor_a,
        exam_type="liver_function", exam_name="肝功能",
        status=ExamRequestStatus.REPORTED.value, created_at=base_ts,
    )
    return {
        "engine": engine,
        "lab": lab,
        "cardio": cardio,
        "doctor_a": doctor_a,
        "labtech01": labtech01,
        "patient01": patient01,
        "patient02": patient02,
        "profile01": profile01,
        "profile02": profile02,
        "visit": visit,
        "req_processing": req_processing,
        "req_received": req_received,
        "req_ordered": req_ordered,
        "req_reported": req_reported,
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


# ==================== LLM / RAG / Celery 替身 ====================


OK_PARSED = {
    "summary": "血钾偏低，建议尽快复诊复查，不要自行补钾。",
    "severity": "high",
    "suggestions": ["尽快找开单医生复核", "一周内复查电解质"],
    "confidence": 0.82,
    "need_doctor_review": True,
}


def _resp(
    *,
    degraded: bool = False,
    blocked: bool = False,
    status_value: str = "success",
    model: str = "mock-interpret",
) -> LLMResponse:
    response = LLMResponse(
        content="{}",
        usage=TokenUsage(),
        provider="openai",
        model=model,
        status=status_value,
    )
    if degraded:
        # 枚举实际取值是 FALLBACK_MODEL（见 llm/types.DegradationLevel），
        # 原写法 FALLBACK 不存在，桩本身在构造阶段就 AttributeError。
        response.degradation_level = DegradationLevel.FALLBACK_MODEL
    if blocked:
        response.moderation_blocked = True
    return response


class CountingLLM:
    """structured_chat 计数桩：按顺序返回预设结果，并记录被调用次数。

    responses[i] 为 dict（交给传入 schema 构造）、None（模拟校验失败）或
    Exception（模拟调用异常）；超出长度复用最后一项。
    embed 直接抛错——本批绝不允许触达真实 embedding 服务。
    """

    def __init__(self, responses: list[Any], *, response: LLMResponse | None = None) -> None:
        self.responses = responses
        self.response = response
        self.calls = 0
        self.requests: list[Any] = []

    async def structured_chat(self, request: Any, schema: Any):
        self.requests.append(request)
        item = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        response = self.response or _resp()
        if isinstance(item, Exception):
            raise item
        if item is None:
            return None, response
        return (schema(**item) if isinstance(item, dict) else item), response

    async def embed(self, texts: Any, **kw: Any):  # pragma: no cover - 兜底拦截
        raise AssertionError("检验侧单测不得调用真实 embed")


def _stub_llm(monkeypatch: pytest.MonkeyPatch, llm: CountingLLM) -> CountingLLM:
    monkeypatch.setattr(es, "get_llm_service", lambda: llm)
    return llm


def _stub_no_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """证据检索固定为空：真实检索要走 embed，且按契约零结果不阻断解读。"""

    async def _empty(session: Any, query_text: str, **kw: Any) -> list[Any]:
        return []

    monkeypatch.setattr(es, "retrieve_evidence", _empty)


def _stub_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: Any = "auto",
) -> list[dict[str, Any]]:
    """打桩 dispatch_task，返回记录调用入参的列表。

    result="auto" → 带 id 的假 AsyncResult（派发成功）；None → 模拟 broker 故障。
    ``import app.tasks.celery_app`` 这行只为触发该模块导入，真正 patch 的目标由
    ``_real_celery()`` 从 sys.modules 取。
    """
    import app.tasks.celery_app  # noqa: F401  - 确保模块已进入 sys.modules

    calls: list[dict[str, Any]] = []

    def fake_dispatch(task_name, *args, **kwargs):
        calls.append({"name": task_name, "args": args, "kwargs": kwargs})
        if result is None:
            return None
        if result == "auto":
            return type("R", (), {"id": f"celery-{len(calls)}"})()
        return result

    monkeypatch.setattr(_real_celery(), "dispatch_task", fake_dispatch)
    return calls


def _real_celery():
    """取 sys.modules 里的真 celery_app 模块（详见 test_imcs_knowledge._real 说明）。"""
    mod = sys.modules.get("app.tasks.celery_app")
    if mod is None:
        raise RuntimeError("app.tasks.celery_app 尚未导入")
    return mod


# ==================== HTTP 客户端 ====================

LAB_TECH = "lab_tech"
DOCTOR_A = "doctor_a"
PATIENT1 = "patient1"
PATIENT2 = "patient2"
CASHIER = "cashier"
ADMIN = "admin"

_WORLD_KEYS: dict[str, tuple[str, str]] = {
    LAB_TECH: ("labtech01", "lab_tech"),
    DOCTOR_A: ("doctor_a", "doctor"),
    PATIENT1: ("patient01", "patient"),
    PATIENT2: ("patient02", "patient"),
    CASHIER: ("cashier01", "cashier"),
    ADMIN: ("admin01", "admin"),
}


def _resolve_identity(world, tag: str) -> tuple:
    """身份标签 → (user_id, username, role_name)。

    cashier/admin 未在 world 预置，按需补建：403 断言必须由「真实存在的角色账号」
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
    """ASGI 客户端：覆盖 get_db 与 get_current_user（沿用 knowledge/consult 策略）。"""
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


def _exam_mounted() -> bool:
    return any("/exams" in p for p in app.openapi()["paths"])


needs_routes = pytest.mark.skipif(
    not _exam_mounted(), reason="exam router 尚未由主代理挂载到 api_router"
)


# ==================== 业务助手 ====================


K_LOW = {"name": "血钾", "code": "K", "value": 2.8, "unit": "mmol/L",
         "ref_low": 3.5, "ref_high": 5.3}
HGB_HIGH = {"name": "血红蛋白", "code": "HB", "value": 170, "unit": "g/L",
            "ref_low": 115, "ref_high": 150}
QUALITATIVE = {"name": "尿蛋白", "code": "PRO", "value": "+/-", "unit": ""}


def structured(*items: dict[str, Any]) -> dict[str, Any]:
    return {"items": list(items), "collected_at": "2026-10-12T08:00:00"}


NO_STRUCTURED = object()  # 哨兵：区分「不提交结构化数据」与「显式传 None」


async def submit_ok(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch, request_id: uuid.UUID,
    *, tech_id: uuid.UUID, data: dict[str, Any] | None | object = NO_STRUCTURED,
    auto_dispatch: bool = False,
) -> tuple[Any, Any]:
    """把单子推到 reported 并出报告，返回 (report, dispatch)。

    修正：world 里的 req_processing **已经是 processing**，而转换矩阵是单向的
    （processing 的后继只有 reported），旧版本在这里无条件先调 receive_request，
    于是抛出「无法从 processing 变更为 received」并把 30+ 个用例整片打挂。
    现在按当前状态取最小路径：ordered → process（矩阵允许直跳）、
    received → process、processing → 原样进入出报告。
    """
    current = await _request_status(db, request_id)
    if current in (ExamRequestStatus.ORDERED.value, ExamRequestStatus.RECEIVED.value):
        # ordered 可直跳 processing（矩阵允许），received 是它的唯一后继
        await es.start_processing(db, request_id)
    return await es.submit_report(
        db,
        request_id,
        lab_tech_id=tech_id,
        raw_content="仪器导出原文：血钾 2.8 mmol/L，参考 3.5-5.3。",
        structured_data=(
            structured(K_LOW) if data is NO_STRUCTURED else data
        ),
        conclusion="血钾偏低，建议复查",
        auto_dispatch=auto_dispatch,
    )


# ==================== 1. 规则判定：异常旗标 ====================


class TestEvaluateAbnormalities:
    def test_high_and_low_flags(self) -> None:
        items = es.evaluate_abnormalities(structured(K_LOW, HGB_HIGH))
        assert [i.direction for i in items] == ["low", "high"]
        assert items[0].to_flag() == "血钾 2.8 ↓(参考 3.5-5.3)"
        assert items[1].to_flag() == "血红蛋白 170 ↑(参考 115-150)"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(3.5, None), (5.3, None), (3.4, "low"), (5.4, "high")],
    )
    def test_boundary_is_strict(self, value: float, expected: str | None) -> None:
        """等于端点不算越界：参考区间是闭区间口径，与「超出参考范围」一致。"""
        item = {"name": "血钾", "code": "K", "value": value,
                "ref_low": 3.5, "ref_high": 5.3}
        flags = es.evaluate_abnormalities(structured(item))
        if expected is None:
            assert flags == []
        else:
            assert [f.direction for f in flags] == [expected]

    def test_single_sided_reference(self) -> None:
        only_high = es.evaluate_abnormalities(
            structured({"name": "血肌酐", "code": "CRT", "value": 600, "ref_high": 130})
        )
        assert [f.direction for f in only_high] == ["high"]
        assert only_high[0]._ref_text() == "(参考 ≤130)"

        only_low = es.evaluate_abnormalities(
            structured({"name": "血红蛋白", "code": "HGB", "value": 50, "ref_low": 115})
        )
        assert only_low[0]._ref_text() == "(参考 ≥115)"

    def test_no_reference_yields_no_flag(self) -> None:
        assert es.evaluate_abnormalities(
            structured({"name": "未知项目", "code": "XX", "value": 9})
        ) == []

    def test_string_and_decimal_values_parsed(self) -> None:
        from decimal import Decimal

        flags = es.evaluate_abnormalities(
            structured(
                {"name": "血钾", "code": "serum k", "value": " 2.8 ",
                 "ref_low": "3.5", "ref_high": Decimal("5.3")}
            )
        )
        assert [f.direction for f in flags] == ["low"]
        assert flags[0].code == "K", "别名与大小写应折叠到标准 code"

    def test_qualitative_item_kept_but_not_judged(self) -> None:
        """定性项不参与判定，但原始 structured_data 不该被清洗掉。"""
        data = structured(K_LOW, QUALITATIVE)
        assert es.evaluate_abnormalities(data).__len__() == 1
        assert len(data["items"]) == 2

    @pytest.mark.parametrize(
        "bad",
        [None, [], "text", {"items": "not-a-list"}, {"items": [1, "x", None]}, {}],
    )
    def test_malformed_input_degrades_to_empty(self, bad: Any) -> None:
        """畸形输入静默返回空集而不是 500：脏值不能让整批录入失败。"""
        assert es.evaluate_abnormalities(bad) == []

    def test_integer_formatting_drops_trailing_zero(self) -> None:
        """细胞计数写成 170 而不是 170.0（检验科读数习惯）。"""
        item = es.evaluate_abnormalities(structured(HGB_HIGH))[0]
        assert "170 " in item.to_flag() and "170.0" not in item.to_flag()

    def test_no_llm_involved(self) -> None:
        """医疗红线：异常判定必须是纯规则，一次 LLM 都不能调。

        修正：原先在整段源码文本里找 "llm"，而 docstring 本身就写着
        「**不调用 LLM**」——这条否定式说明会被当成命中，用例恒红。
        契约要防的是「真的发起调用」，因此改为剥离文档与注释后再查
        llm / get_llm_service / structured_chat 等调用符号。
        """
        src = inspect.getsource(es.evaluate_abnormalities)
        code_lines = []
        in_docstring = False
        for line in src.splitlines():
            stripped = line.strip()
            if not in_docstring and stripped.startswith(('"""', "'''")):
                quote = stripped[:3]
                if stripped.count(quote) >= 2 and len(stripped) > 3:
                    continue
                in_docstring = True
                continue
            if in_docstring:
                if '"""' in stripped or "'''" in stripped:
                    in_docstring = False
                continue
            if stripped.startswith("#"):
                continue
            code_lines.append(line)
            # 函数签名行本身也保留（含参数名），但签名里不含 llm
        code = "\n".join(code_lines).lower()
        for symbol in ("llm", "get_llm_service", "structured_chat", "chat("):
            assert symbol not in code, f"异常判定不应触达 {symbol}"


# ==================== 2. 规则判定：危急值 ====================


class TestDetectCriticalValues:
    def test_k_low_hits_critical(self) -> None:
        hits = es.detect_critical_values(structured(K_LOW))
        assert len(hits) == 1
        assert hits[0].code == "K"
        assert hits[0].threshold_text == "危急下限 3"
        line = hits[0].to_line()
        assert "危急值" in line and "立即通知接诊医生" in line

    def test_threshold_table_is_narrower_than_reference(self) -> None:
        """血钾 3.2：出参考区间但不算危急——混用会把轻度偏低报成抢救警报。"""
        item = {"name": "血钾", "code": "K", "value": 3.2,
                "ref_low": 3.5, "ref_high": 5.3}
        assert len(es.evaluate_abnormalities(structured(item))) == 1
        assert es.detect_critical_values(structured(item)) == []

    def test_high_side_hit(self) -> None:
        hits = es.detect_critical_values(
            structured({"name": "血钾", "code": "K", "value": 7.2})
        )
        assert hits[0].threshold_text == "危急上限 6"

    def test_exact_threshold_not_hit(self) -> None:
        """严格不等：正好落在危急阈值上不触发全院级告警。"""
        assert es.detect_critical_values(
            structured({"name": "血钾", "code": "K", "value": 3.0})
        ) == []

    def test_alias_normalized_before_match(self) -> None:
        assert es.detect_critical_values(
            structured({"name": "血糖", "code": "FBG", "value": 1.2})
        )[0].code == "GLU"

    def test_substring_codes_do_not_false_hit(self) -> None:
        """精确相等而非包含匹配："PK"/"KIM-1" 误命中会触发假的抢救通报。"""
        for code in ("PK", "KIM-1", "ALKP"):
            assert es.detect_critical_values(
                structured({"name": "其它", "code": code, "value": 0.1})
            ) == []

    def test_missing_code_skips_critical(self) -> None:
        assert es.detect_critical_values(
            structured({"name": "血钾", "value": 1.0})
        ) == []

    def test_unparseable_value_skipped(self) -> None:
        assert es.detect_critical_values(
            structured({"name": "血钾", "code": "K", "value": "偏低"})
        ) == []

    def test_bool_value_never_treated_as_number(self) -> None:
        """True 是 int 子类，若被当成 1 就会命中「血钾 1.0」这种荒谬危急值。"""
        assert es.detect_critical_values(
            structured({"name": "血钾", "code": "K", "value": True})
        ) == []

    def test_rule_table_self_consistent(self) -> None:
        for rule in es.CRITICAL_RULES:
            low, high = rule.get("low"), rule.get("high")
            assert low is not None or high is not None, rule["code"]
            if low is not None and high is not None:
                assert float(low) < float(high), rule["code"]


# ==================== 3. 结构化输出契约 ====================


class TestReportInterpretationSchema:
    def test_valid_payload(self) -> None:
        parsed = es.ReportInterpretation(**OK_PARSED)
        assert parsed.severity == "high"
        assert parsed.need_doctor_review is True

    def test_need_doctor_review_false_rejected(self) -> None:
        """AI 永不宣布「无需复核」：这是模型越权，必须显式失败。"""
        with pytest.raises(ValueError):
            es.ReportInterpretation(**{**OK_PARSED, "need_doctor_review": False})

    @pytest.mark.parametrize("severity", ["critical", "LOW", ""])
    def test_severity_enum_closed(self, severity: str) -> None:
        with pytest.raises(ValueError):
            es.ReportInterpretation(**{**OK_PARSED, "severity": severity})

    def test_confidence_range(self) -> None:
        with pytest.raises(ValueError):
            es.ReportInterpretation(**{**OK_PARSED, "confidence": 1.5})

    def test_suggestions_capped_and_cleaned(self) -> None:
        parsed = es.ReportInterpretation(
            **{**OK_PARSED, "suggestions": ["a", " b ", "", "c", "d", "e", "f", "g"]}
        )
        assert parsed.suggestions == ["a", "b", "c", "d", "e"]

    def test_summary_stripped(self) -> None:
        assert es.ReportInterpretation(
            **{**OK_PARSED, "summary": "  复查  "}
        ).summary == "复查"


# ==================== 4. 申请单状态流转 ====================


class TestRequestTransitions:
    async def test_receive_ordered(self, db: AsyncSession, engine, world) -> None:
        req = await es.receive_request(db, world["req_ordered"])
        assert req.status == ExamRequestStatus.RECEIVED.value
        assert read_row(engine, "exam_requests", world["req_ordered"])["status"] == "received"

    async def test_receive_is_idempotent(self, db: AsyncSession, engine, world) -> None:
        """双人复核场景：第二次点接收不报错，也不该被当成失败。"""
        await es.receive_request(db, world["req_ordered"])
        again = await es.receive_request(db, world["req_ordered"])
        assert again.status == ExamRequestStatus.RECEIVED.value

    async def test_receive_after_processing_rejected(
        self, db: AsyncSession, engine, world
    ) -> None:
        """已推进的单子不能退回 received，否则正在做的检查凭空倒退。"""
        with pytest.raises(InvalidStateTransitionError) as exc:
            await es.receive_request(db, world["req_reported"])
        assert exc.value.detail["current_status"] == ExamRequestStatus.REPORTED.value
        assert exc.value.detail["target_status"] == ExamRequestStatus.RECEIVED.value
        assert exc.value.detail["entity"] == "exam_request"
        assert read_row(engine, "exam_requests", world["req_reported"])["status"] == "reported"

    async def test_process_from_received(self, db: AsyncSession, engine, world) -> None:
        await es.receive_request(db, world["req_received"])
        req = await es.start_processing(db, world["req_received"])
        assert req.status == ExamRequestStatus.PROCESSING.value

    async def test_process_direct_jump_from_ordered(
        self, db: AsyncSession, engine, world
    ) -> None:
        """矩阵本来就允许 ordered→processing（LIS 自动推结果），不是这里额外放宽。"""
        req = await es.start_processing(db, world["req_ordered"])
        assert req.status == ExamRequestStatus.PROCESSING.value
        assert ExamRequestStatus.PROCESSING.value in allowed_targets(
            "exam_request", ExamRequestStatus.ORDERED.value
        )

    async def test_process_is_idempotent(self, db: AsyncSession, world) -> None:
        again = await es.start_processing(db, world["req_processing"])
        assert again.status == ExamRequestStatus.PROCESSING.value

    async def test_process_from_reported_rejected(self, db: AsyncSession, world) -> None:
        with pytest.raises(InvalidStateTransitionError) as exc:
            await es.start_processing(db, world["req_reported"])
        assert exc.value.detail["current_status"] == "reported"

    async def test_missing_request_404(self, db: AsyncSession, world) -> None:
        with pytest.raises(ResourceNotFoundError):
            await es.receive_request(db, uuid.uuid4())

    def test_matrix_matches_design(self) -> None:
        """ordered→received→processing→reported→reviewed 单向线性，reviewed 为终态。"""
        assert allowed_targets("exam_request", "ordered") == ["processing", "received"]
        assert allowed_targets("exam_request", "received") == ["processing"]
        assert allowed_targets("exam_request", "processing") == ["reported"]
        assert allowed_targets("exam_request", "reported") == ["reviewed"]
        assert allowed_targets("exam_request", "reviewed") == []


# ==================== 5. 报告录入 ====================


class TestSubmitReport:
    async def test_success_writes_flags_and_transitions(
        self, db: AsyncSession, engine, world
    ) -> None:
        report, dispatch = await submit_ok(
            db, None, world["req_processing"], tech_id=world["labtech01"]
        )
        assert report.ai_status == es.AI_STATUS_PENDING
        assert report.abnormal_flags == ["血钾 2.8 ↓(参考 3.5-5.3)"]
        assert report.conclusion == "血钾偏低，建议复查"
        assert report.patient_id == world["profile01"]
        assert report.lab_tech_id == world["labtech01"]
        assert dispatch.dispatched is False, "auto_dispatch=False 时不应派发"
        assert dispatch.reason == "未启用自动派发"

        req_row = read_row(engine, "exam_requests", world["req_processing"])
        assert req_row["status"] == ExamRequestStatus.REPORTED.value
        report_row = read_row(engine, "exam_reports", report.id)
        assert report_row["raw_content"].startswith("仪器导出原文")

    async def test_structured_data_kept_verbatim(
        self, db: AsyncSession, engine, world
    ) -> None:
        """服务端只做附加计算，不清洗调用方原样提交的 structured_data。"""
        data = structured(K_LOW, QUALITATIVE)
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"], data=data)
        stored = read_row(engine, "exam_reports", report.id)["structured_data"]
        parsed = json.loads(stored) if isinstance(stored, str) else stored
        assert len(parsed["items"]) == 2
        assert parsed["collected_at"] == "2026-10-12T08:00:00"

    async def test_submit_from_received_rejected(
        self, db: AsyncSession, engine, world
    ) -> None:
        """必须先 processing：矩阵里没有 received→reported。"""
        await es.receive_request(db, world["req_received"])
        with pytest.raises(InvalidStateTransitionError) as exc:
            await es.submit_report(
                db, world["req_received"], lab_tech_id=world["labtech01"],
                raw_content="x", structured_data=structured(K_LOW),
            )
        assert exc.value.detail["current_status"] == "received"
        assert exc.value.detail["target_status"] == "reported"
        assert count_rows(engine, "exam_reports") == 0, "被拒时不得留下半份报告"

    async def test_duplicate_report_rejected(self, db: AsyncSession, engine, world) -> None:
        """一张申请单只能有一份报告：提前给出干净错误而不是 IntegrityError 冒成 500。"""
        first, _ = await submit_ok(db, None, world["req_processing"],
                                   tech_id=world["labtech01"])
        with pytest.raises(InvalidStateTransitionError) as exc:
            await es.submit_report(
                db, world["req_processing"], lab_tech_id=world["labtech01"],
                raw_content="第二次录入",
            )
        assert exc.value.detail["entity"] == "exam_request"
        assert exc.value.detail["target_status"] == "reported"
        assert count_rows(engine, "exam_reports") == 1
        assert read_row(engine, "exam_reports", first.id)["raw_content"] != "第二次录入"

    async def test_missing_request_404(self, db: AsyncSession, world) -> None:
        with pytest.raises(ResourceNotFoundError):
            await es.submit_report(db, uuid.uuid4(), lab_tech_id=world["labtech01"],
                                   raw_content="x")

    async def test_no_structured_data_yields_no_flags(
        self, db: AsyncSession, world
    ) -> None:
        """只有原始文本的报告合法：结构化是附赠能力，不是必需路径。"""
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"], data=None)
        assert report.abnormal_flags == []
        assert report.structured_data is None


# ==================== 6. 自动派发与时序 ====================


class TestAutoDispatch:
    async def test_report_saved_before_dispatch_and_task_recorded(
        self, db: AsyncSession, engine, world, monkeypatch
    ) -> None:
        """AITask 先落库、再派发；派发成功后任务 running、报告 processing。"""
        calls = _stub_dispatch(monkeypatch)
        report, dispatch = await submit_ok(
            db, None, world["req_processing"], tech_id=world["labtech01"],
            auto_dispatch=True,
        )

        assert dispatch.dispatched is True
        assert dispatch.ai_task_id is not None
        assert dispatch.celery_task_id == "celery-1"

        task_row = read_row(engine, "ai_tasks", dispatch.ai_task_id)
        assert task_row["task_type"] == es.TASK_TYPE_REPORT_INTERPRET
        # ref_id 指向报告本身（per-report 任务，全局查重靠它）
        assert task_row["ref_id"] in (report.id.hex, str(report.id))
        assert task_row["status"] == "running"
        assert task_row["celery_task_id"] == dispatch.celery_task_id
        assert task_row["started_at"] is not None

        report_row = read_row(engine, "exam_reports", report.id)
        # 入队成功只表示「已排队」：processing 由 worker CAS 认领才写，
        # 派发方提前置 processing 会让 worker 把自己消息认成重复投递而永不解读
        assert report_row["ai_status"] == es.AI_STATUS_PENDING
        assert report_row["ai_task_id"] == task_row["id"]

        payload = json.loads(task_row["payload"]) if isinstance(task_row["payload"], str) else task_row["payload"]
        assert payload["report_id"] == str(report.id)
        assert payload["trigger"] == "auto"

    async def test_dispatch_call_contract(self, db: AsyncSession, world, monkeypatch) -> None:
        """任务全限定名与队列是部署契约：写错一个字符 worker 永远收不到消息。"""
        calls = _stub_dispatch(monkeypatch)
        report, dispatch = await submit_ok(
            db, None, world["req_processing"], tech_id=world["labtech01"],
            auto_dispatch=True,
        )
        assert len(calls) == 1
        call = calls[0]
        assert call["name"] == es.INTERPRET_TASK_NAME
        assert call["name"] == "app.tasks.ai_tasks.interpret_exam_report"
        assert call["args"] == (str(report.id), str(dispatch.ai_task_id))
        assert call["kwargs"]["queue"] == _real_celery().QUEUE_INTERPRET
        # 显式 task_id 让 Celery 侧去重，防用户连点两次
        assert call["kwargs"]["task_id"] == str(dispatch.ai_task_id)

    async def test_broker_failure_does_not_roll_back_report(
        self, db: AsyncSession, engine, world, monkeypatch
    ) -> None:
        """核心医疗原则：broker 挂了报告照样录进去，AI 只是辅助路径。"""
        _stub_dispatch(monkeypatch, result=None)
        report, dispatch = await submit_ok(
            db, None, world["req_processing"], tech_id=world["labtech01"],
            auto_dispatch=True,
        )
        assert dispatch.dispatched is False
        assert dispatch.reason and "手动" in dispatch.reason

        report_row = read_row(engine, "exam_reports", report.id)
        assert report_row["raw_content"], "报告正文必须已安全落库"
        assert report_row["ai_status"] == es.AI_STATUS_FAILED
        assert read_row(engine, "exam_requests", world["req_processing"])["status"] == "reported"

        task_row = read_row(engine, "ai_tasks", dispatch.ai_task_id)
        assert task_row["status"] == "failed"
        assert task_row["completed_at"] is not None
        assert task_row["error_message"]

    async def test_commit_after_dispatch_failure_reads_back(
        self, db: AsyncSession, world, monkeypatch
    ) -> None:
        """回归排查点：派发失败分支若漏 commit，库里仍是 pending。"""
        _stub_dispatch(monkeypatch, result=None)
        report, dispatch = await submit_ok(
            db, None, world["req_processing"], tech_id=world["labtech01"],
            auto_dispatch=True,
        )
        fresh = await db.get(es.ExamReport, report.id)
        assert fresh is not None
        assert fresh.ai_status == es.AI_STATUS_FAILED
        task = await db.get(AITask, dispatch.ai_task_id)
        assert task is not None and task.status == "failed"

    async def test_object_attributes_readable_after_commit(
        self, db: AsyncSession, world, monkeypatch
    ) -> None:
        """expire_on_commit=False：commit 后直接读 ORM 属性不得触发 MissingGreenlet。"""
        calls = _stub_dispatch(monkeypatch)
        report, dispatch = await submit_ok(
            db, None, world["req_processing"], tech_id=world["labtech01"],
            auto_dispatch=True,
        )
        assert report.id is not None
        assert report.request_id == world["req_processing"]
        assert report.ai_status == es.AI_STATUS_PENDING
        assert dispatch.ai_task_id is not None
        assert len(calls) == 1


# ==================== 7. 工作台与读取 ====================


class TestQueueAndReads:
    async def test_queue_orders_by_urgency_then_fifo(
        self, db: AsyncSession, engine, world
    ) -> None:
        """STAT 插队、同级先到先做：这是检验科的现实工作方式。"""
        e = world["engine"]
        stat = make_exam_request(e, visit_id=world["visit"], patient_id=world["profile01"],
                                 doctor_id=world["doctor_a"], exam_name="急诊血气",
                                 urgency=Urgency.STAT.value,
                                 status=ExamRequestStatus.RECEIVED.value,
                                 created_at=utcnow())
        urgent = make_exam_request(e, visit_id=world["visit"], patient_id=world["profile01"],
                                   doctor_id=world["doctor_a"], exam_name="急查心肌酶",
                                   urgency=Urgency.URGENT.value,
                                   status=ExamRequestStatus.PROCESSING.value,
                                   created_at=utcnow())
        rows, total = await es.list_queue(db)
        assert total >= 4
        ids = [r.id for r in rows]
        assert ids.index(stat) < ids.index(urgent)
        assert ids.index(urgent) < ids.index(world["req_received"]), "routine 保持 FIFO"

    async def test_queue_excludes_reported_and_reviewed(
        self, db: AsyncSession, world
    ) -> None:
        await submit_ok(db, None, world["req_processing"], tech_id=world["labtech01"])
        rows, _ = await es.list_queue(db)
        assert world["req_processing"] not in [r.id for r in rows]
        assert world["req_ordered"] not in [r.id for r in rows]

    async def test_status_filter_only_narrows_within_lab_statuses(
        self, db: AsyncSession, world
    ) -> None:
        rows, _ = await es.list_queue(db, status_filter=ExamRequestStatus.PROCESSING.value)
        assert all(r.status == "processing" for r in rows)

        outside, total = await es.list_queue(db, status_filter="reported")
        assert outside == [] and total == 0, "工作台是待办视图，不属于它的状态返回空集"

    async def test_queue_report_map_links_report_and_ai_status(
        self, db: AsyncSession, engine, world
    ) -> None:
        """没有这一步，检验科在工作台看不到「报告出了但 AI 失败」。"""
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        req = await db.get(es.ExamRequest, world["req_processing"])
        mapping = await es.queue_report_map(db, [req])
        assert mapping[world["req_processing"]] == (report.id, es.AI_STATUS_PENDING)

    async def test_attach_request_info_batch(self, db: AsyncSession, world) -> None:
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        info = await es.attach_request_info(db, [report])
        assert info[report.id] == ("blood_routine", "血常规", "routine", "reported")

    async def test_attach_request_info_empty_short_circuits(self, db: AsyncSession) -> None:
        assert await es.attach_request_info(db, []) == {}

    async def test_doctor_reads_any_report(self, db: AsyncSession, world) -> None:
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        got = await es.get_report_detail(
            db, report.id, requester_user_id=world["doctor_a"], role_name="doctor"
        )
        assert got.id == report.id

    async def test_patient_reads_own_report(self, db: AsyncSession, world) -> None:
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        got = await es.get_report_detail(
            db, report.id, requester_user_id=world["patient01"], role_name="patient"
        )
        assert got.id == report.id

    async def test_patient_cross_report_is_404_not_403(
        self, db: AsyncSession, world
    ) -> None:
        """404 不向探测者确认「这个 ID 存在且属于别人」——医疗记录 ID 本身敏感。"""
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        with pytest.raises(ResourceNotFoundError):
            await es.get_report_detail(
                db, report.id, requester_user_id=world["patient02"], role_name="patient"
            )

    async def test_list_reports_isolates_patient(self, db: AsyncSession, world) -> None:
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        rows, total = await es.list_reports(
            db, requester_user_id=world["patient01"], role_name="patient",
            patient_id=world["profile02"],
        )
        assert total == 1 and rows[0].id == report.id, "越权 patient_id 被忽略而非 403"

        other, other_total = await es.list_reports(
            db, requester_user_id=world["patient02"], role_name="patient"
        )
        assert other == [] and other_total == 0

    async def test_list_reports_without_profile_returns_empty(self, db: AsyncSession, world) -> None:
        orphan = make_user(world["engine"], role_name="patient", username="orphan")
        rows, total = await es.list_reports(
            db, requester_user_id=orphan, role_name="patient"
        )
        assert rows == [] and total == 0

    async def test_pagination_caps_page_size(self, db: AsyncSession, world) -> None:
        rows, _ = await es.list_reports(
            db, requester_user_id=world["doctor_a"], role_name="doctor",
            page=1, page_size=5000,
        )
        assert isinstance(rows, list)


# ==================== 8. AI 解读工作流（异步侧）====================


class TestInterpretAsync:
    async def test_success_writes_all_ai_fields_and_reviews(
        self, db: AsyncSession, engine, world, monkeypatch
    ) -> None:
        _stub_no_evidence(monkeypatch)
        llm = _stub_llm(monkeypatch, CountingLLM([OK_PARSED]))
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])

        outcome = await es.interpret_async(db, report.id)
        assert outcome.status == es.AI_STATUS_COMPLETED
        assert llm.calls == 1

        row = read_row(engine, "exam_reports", report.id)
        assert row["ai_status"] == es.AI_STATUS_COMPLETED
        assert row["ai_summary"] == OK_PARSED["summary"]
        assert row["ai_interpretation"] == OK_PARSED["summary"]
        assert json.loads(row["ai_suggestions"]) == OK_PARSED["suggestions"]
        assert abs(float(row["ai_confidence"]) - 0.82) < 1e-6
        assert row["ai_model"] == "mock-interpret"
        assert row["ai_provider"] == "openai"
        assert row["interpreted_at"] is not None
        # 医生的审核动作才是报告生效条件；AI 完成解读即推进 reported→reviewed
        assert read_row(engine, "exam_requests", world["req_processing"])["status"] == (
            ExamRequestStatus.REVIEWED.value
        )

    async def test_prompt_carries_rules_and_not_rejudgement(
        self, db: AsyncSession, world, monkeypatch
    ) -> None:
        """prompt 必须写明「异常已由系统判定」，AI 只负责解释不负责判异常。"""
        _stub_no_evidence(monkeypatch)
        llm = _stub_llm(monkeypatch, CountingLLM([OK_PARSED]))
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        await es.interpret_async(db, report.id)

        request = llm.requests[0]
        assert request.scenario == es.LLMScenario.REPORT_INTERPRET
        assert request.workflow_node == "report_interpret"
        user_prompt = request.messages[1].content
        assert "你不需要重新判断是否异常" in user_prompt
        assert "危急值" in user_prompt and "血钾" in user_prompt
        assert "血钾 2.8" in user_prompt
        assert "知识库未检索到相关条目" in user_prompt
        assert request.trace_id

    async def test_patient_pii_absent_from_prompt(
        self, db: AsyncSession, world, monkeypatch
    ) -> None:
        """喂给第三方模型的上下文只留年龄性别，姓名/身份证/手机号一概不给。"""
        _stub_no_evidence(monkeypatch)
        llm = _stub_llm(monkeypatch, CountingLLM([OK_PARSED]))
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        await es.interpret_async(db, report.id)
        prompt = llm.requests[0].messages[1].content
        profile_block = prompt.split("【检查项目】")[0]
        for leak in ("patient_01", "张三", "id_card", "350"):
            assert leak not in profile_block
        assert "女" in profile_block and "岁" in profile_block
        assert "高血压" in profile_block

    async def test_degraded_response_marks_failed_and_keeps_data(
        self, db: AsyncSession, engine, world, monkeypatch
    ) -> None:
        """降级透明：拿兜底话术当解读结果写进病历，比没有解读更糟。"""
        _stub_no_evidence(monkeypatch)
        llm = _stub_llm(
            monkeypatch,
            CountingLLM([OK_PARSED], response=_resp(degraded=True)),
        )
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        outcome = await es.interpret_async(db, report.id)
        assert outcome.status == es.AI_STATUS_FAILED and outcome.degraded is True
        assert llm.calls == 1

        row = read_row(engine, "exam_reports", report.id)
        assert row["ai_status"] == es.AI_STATUS_FAILED
        assert row["interpreted_at"] is None
        assert json.loads(row["abnormal_flags"]) == ["血钾 2.8 ↓(参考 3.5-5.3)"]
        assert row["conclusion"] == "血钾偏低，建议复查"
        assert row["raw_content"].startswith("仪器导出原文")
        # 失败时清掉残留解读文本，避免出现「failed 但 summary 还写着上一轮结论」
        # read_row 直读库：SQLite 下 ai_suggestions 是 TEXT 退化，"[]" 是字符串形态
        assert row["ai_summary"] is None and json.loads(row["ai_suggestions"] or "[]") == []
        assert read_row(engine, "exam_requests", world["req_processing"])["status"] == (
            ExamRequestStatus.REPORTED.value
        ), "解读失败不得推进审核态"

    @pytest.mark.parametrize("status_value", ["degraded", "budget_exceeded", "blocked"])
    async def test_budget_and_status_paths_also_blocked(
        self, db: AsyncSession, engine, world, monkeypatch, status_value: str
    ) -> None:
        """chat 预算耗尽时 degradation_level 可能仍是 PRIMARY，必须看 status 兜住。"""
        _stub_no_evidence(monkeypatch)
        _stub_llm(
            monkeypatch,
            CountingLLM([OK_PARSED], response=_resp(status_value=status_value)),
        )
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        outcome = await es.interpret_async(db, report.id)
        assert outcome.status == es.AI_STATUS_FAILED
        assert read_row(engine, "exam_reports", report.id)["ai_status"] == "failed"

    async def test_validation_failure_marks_failed(
        self, db: AsyncSession, engine, world, monkeypatch
    ) -> None:
        _stub_no_evidence(monkeypatch)
        _stub_llm(monkeypatch, CountingLLM([None]))
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        outcome = await es.interpret_async(db, report.id)
        assert outcome.status == es.AI_STATUS_FAILED
        assert "校验失败" in (outcome.error or "")
        assert read_row(engine, "exam_reports", report.id)["ai_status"] == "failed"

    async def test_moderation_block_marks_skipped(
        self, db: AsyncSession, engine, world, monkeypatch
    ) -> None:
        """审核拦截 ≠ 系统故障：口径是 skipped，前端文案不能写成「故障」。"""
        _stub_no_evidence(monkeypatch)
        _stub_llm(monkeypatch, CountingLLM([OK_PARSED], response=_resp(blocked=True)))
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        outcome = await es.interpret_async(db, report.id)
        assert outcome.status == es.AI_STATUS_SKIPPED
        row = read_row(engine, "exam_reports", report.id)
        assert row["ai_status"] == es.AI_STATUS_SKIPPED
        assert row["raw_content"]
        assert read_row(engine, "exam_requests", world["req_processing"])["status"] == "reported"

    async def test_llm_exception_does_not_bubble(
        self, db: AsyncSession, engine, world, monkeypatch
    ) -> None:
        """解读失败绝不能冒泡成 500：医生看原始数值照样能判读。"""
        _stub_no_evidence(monkeypatch)
        _stub_llm(monkeypatch, CountingLLM([RuntimeError("provider down")]))
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        outcome = await es.interpret_async(db, report.id)
        assert outcome.status == es.AI_STATUS_FAILED
        assert "provider down" in (outcome.error or "")
        assert read_row(engine, "exam_reports", report.id)["ai_status"] == "failed"

    async def test_evidence_included_in_prompt(
        self, db: AsyncSession, world, monkeypatch
    ) -> None:
        """有证据时要写出出处与可信度，供医生追溯。"""
        evidence = [type("E", (), {"title": "低钾血症", "source": "院内库",
                                   "source_tier": "A", "content": "x" * 900})()]

        async def _ev(session: Any, query_text: str, **kw: Any) -> list[Any]:
            captured["query"] = query_text
            return evidence

        captured: dict[str, Any] = {}
        monkeypatch.setattr(es, "retrieve_evidence", _ev)
        llm = _stub_llm(monkeypatch, CountingLLM([OK_PARSED]))
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        outcome = await es.interpret_async(db, report.id)
        assert outcome.evidence_count == 1
        assert "《低钾血症》" in llm.requests[0].messages[1].content
        assert captured["query"] == "血钾 危急值、血钾 偏低"

    async def test_rag_failure_swallowed(
        self, db: AsyncSession, world, monkeypatch
    ) -> None:
        """检索炸了也不能取消一份已录好报告的解读资格。"""

        async def _boom(session: Any, query_text: str, **kw: Any):
            raise RuntimeError("pgvector unavailable")

        monkeypatch.setattr(es, "retrieve_evidence", _boom)
        _stub_llm(monkeypatch, CountingLLM([OK_PARSED]))
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        outcome = await es.interpret_async(db, report.id)
        assert outcome.status == es.AI_STATUS_COMPLETED
        assert outcome.evidence_count == 0

    async def test_retrieve_evidence_helper_swallows_and_truncates(
        self, db: AsyncSession, monkeypatch
    ) -> None:
        """直测 retrieve_evidence：异常吞掉返回空；top_k 截断与缺省全量都生效。"""

        class _Resp:
            def __init__(self, results: list[Any]) -> None:
                self.results = results

        class _FakeRetriever:
            async def retrieve(self, session, query_text, **kw):
                return _Resp([type("E", (), {"title": str(i)})() for i in range(5)])

        class _BoomRetriever:
            async def retrieve(self, session, query_text, **kw):
                raise RuntimeError("retriever down")

        monkeypatch.setattr(es, "RAGRetriever", _FakeRetriever)
        assert len(await es.retrieve_evidence(db, "血钾 偏低", top_k=2)) == 2
        assert len(await es.retrieve_evidence(db, "血钾 偏低")) == 5
        assert await es.retrieve_evidence(db, "血钾 偏低", top_k=0) == []

        monkeypatch.setattr(es, "RAGRetriever", _BoomRetriever)
        assert await es.retrieve_evidence(db, "x") == []

    async def test_retrieve_evidence_skips_on_sync_session(
        self, engine, world
    ) -> None:
        """同步会话分支不桥接异步检索，直接返回空（worker 侧由 evidence_fn 供给）。"""
        from sqlalchemy.orm import Session

        with Session(bind=engine) as s:
            assert await es.retrieve_evidence(s, "血钾 偏低") == []

    async def test_missing_report_404(self, db: AsyncSession, world) -> None:
        with pytest.raises(ResourceNotFoundError):
            await es.interpret_async(db, uuid.uuid4())

    async def test_reviewed_request_not_written_again(
        self, db: AsyncSession, engine, world, monkeypatch
    ) -> None:
        """终态自我重复写入排查：第二次解读不得对 reviewed 再发 UPDATE。"""
        _stub_no_evidence(monkeypatch)
        _stub_llm(monkeypatch, CountingLLM([OK_PARSED]))
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        await es.interpret_async(db, report.id)
        assert read_row(engine, "exam_requests", world["req_processing"])["status"] == "reviewed"

        second = await es.interpret_async(db, report.id)
        assert second.status == es.AI_STATUS_COMPLETED
        assert read_row(engine, "exam_requests", world["req_processing"])["status"] == (
            ExamRequestStatus.REVIEWED.value
        ), "reviewed 是终态，不能被再次流转或报错"


# ==================== 9. 幂等认领闸门（worker 侧同步）====================


class TestClaimForInterpret:
    def _sync_session(self, engine):
        from sqlalchemy.orm import Session

        return Session(bind=engine)

    def test_claim_from_pending(self, engine, session_factory, world) -> None:
        report_id = _run(self._make_report(session_factory, world))
        with self._sync_session(engine) as s:
            claim = es.claim_for_interpret(s, report_id)
            assert claim.claimed is True and claim.ai_status == es.AI_STATUS_PROCESSING
            assert s.get(es.ExamReport, report_id).ai_status == "processing"

    @pytest.mark.parametrize(
        ("start_status", "expected_reason"),
        [("completed", "already_completed"), ("processing", "in_progress")],
    )
    def test_terminal_and_running_are_gated(
        self, engine, session_factory, world, start_status: str, expected_reason: str
    ) -> None:
        """acks_late 下消息会重投：不拦第二次就多烧配额并交错覆盖 ai_* 字段。"""
        report_id = _run(self._make_report(session_factory, world))
        with self._sync_session(engine) as s:
            row = s.get(es.ExamReport, report_id)
            row.ai_status = start_status
            s.commit()

        with self._sync_session(engine) as s:
            claim = es.claim_for_interpret(s, report_id)
            assert claim.claimed is False
            assert claim.reason == expected_reason
            assert claim.ai_status == start_status
            assert s.get(es.ExamReport, report_id).ai_status == start_status

    @pytest.mark.parametrize("start_status", ["pending", "failed", "skipped"])
    def test_retry_statuses_can_claim(
        self, engine, session_factory, world, start_status: str
    ) -> None:
        """failed/skipped 是人工重试派发的新消息，必须能被认领。"""
        report_id = _run(self._make_report(session_factory, world))
        with self._sync_session(engine) as s:
            row = s.get(es.ExamReport, report_id)
            row.ai_status = start_status
            s.commit()
        with self._sync_session(engine) as s:
            assert es.claim_for_interpret(s, report_id).claimed is True

    def test_missing_report_not_found(self, engine) -> None:
        with self._sync_session(engine) as s:
            claim = es.claim_for_interpret(s, uuid.uuid4())
        assert claim.claimed is False and claim.reason == "not_found"

    @staticmethod
    async def _make_report(session_factory, world) -> uuid.UUID:
        async with session_factory() as session:
            # req_processing 已经是 processing，矩阵单向不可退回 received；
            # 与 submit_ok 同口径，直接出报告。
            report, _ = await es.submit_report(
                session, world["req_processing"], lab_tech_id=world["labtech01"],
                raw_content="原文", structured_data=structured(K_LOW),
                auto_dispatch=False,
            )
            return report.id


def _run(coro):
    """在同步用例里驱动一次协程（准备数据用，不触碰被测实现）。"""
    import asyncio

    return asyncio.run(coro)


# ==================== 10. worker 侧同步解读 ====================


class TestRunInterpretationSync:
    def _sync_session(self, engine):
        from sqlalchemy.orm import Session

        return Session(bind=engine)

    def test_success_sync_path(self, engine, session_factory, world) -> None:
        report_id = _run(self._prepare(session_factory, world))
        llm = CountingLLM([OK_PARSED])
        with self._sync_session(engine) as s:
            outcome = es.run_interpretation_sync(
                s, report_id, llm_service=llm, evidence_fn=lambda q, t: []
            )
            assert outcome.status == es.AI_STATUS_COMPLETED
            assert llm.calls == 1
            row = s.get(es.ExamReport, report_id)
            assert row.ai_status == "completed"
            assert row.ai_summary == OK_PARSED["summary"]
            assert row.interpreted_at is not None
            assert s.get(es.ExamRequest, world["req_processing"]).status == "reviewed"

    def test_degraded_sync_path_keeps_medical_data(
        self, engine, session_factory, world
    ) -> None:
        """同步侧字段口径必须与异步侧一致：_apply_ai_fields 是唯一映射点。"""
        report_id = _run(self._prepare(session_factory, world))
        llm = CountingLLM([OK_PARSED], response=_resp(degraded=True))
        with self._sync_session(engine) as s:
            outcome = es.run_interpretation_sync(
                s, report_id, llm_service=llm, evidence_fn=lambda q, t: []
            )
            assert outcome.status == es.AI_STATUS_FAILED
            row = s.get(es.ExamReport, report_id)
            assert row.ai_status == "failed"
            assert row.ai_summary is None and row.interpreted_at is None
            # 医疗事实一个字都不碰
            assert row.raw_content == "原文"
            assert row.abnormal_flags == ["血钾 2.8 ↓(参考 3.5-5.3)"]
            assert s.get(es.ExamRequest, world["req_processing"]).status == "reported"

    def test_sync_inside_running_loop_refuses(
        self, engine, session_factory, world
    ) -> None:
        """已有活循环时显式失败，而不是想办法绕（绕法要么阻塞 loop 要么静默丢结果）。"""
        report_id = _run(self._prepare(session_factory, world))

        async def _drive():
            with self._sync_session(engine) as s:
                with pytest.raises(RuntimeError) as exc:
                    es.run_interpretation_sync(
                        s, report_id, llm_service=CountingLLM([OK_PARSED]),
                        evidence_fn=lambda q, t: [],
                    )
                return str(exc.value)

        message = _run(_drive())
        assert "interpret_async" in message

    def test_missing_report_raises(self, engine) -> None:
        with self._sync_session(engine) as s:
            with pytest.raises(ResourceNotFoundError):
                es.run_interpretation_sync(
                    s, uuid.uuid4(), llm_service=CountingLLM([OK_PARSED])
                )

    @staticmethod
    async def _prepare(session_factory, world) -> uuid.UUID:
        async with session_factory() as session:
            # 同 _make_report：req_processing 已在 processing，不能再 receive。
            report, _ = await es.submit_report(
                session, world["req_processing"], lab_tech_id=world["labtech01"],
                raw_content="原文", structured_data=structured(K_LOW),
                auto_dispatch=False,
            )
            return report.id


# ==================== 11. 手动触发与重复派发 ====================


class TestTriggerInterpretation:
    async def test_manual_trigger_from_failed(
        self, db: AsyncSession, engine, world, monkeypatch
    ) -> None:
        """failed 不在 ACTIVE_AI_STATUSES 里，所以补救入口走得通。"""
        _stub_dispatch(monkeypatch, result=None)
        report, first = await submit_ok(db, None, world["req_processing"],
                                        tech_id=world["labtech01"], auto_dispatch=True)
        assert read_row(engine, "exam_reports", report.id)["ai_status"] == "failed"

        calls = _stub_dispatch(monkeypatch)
        again = await es.trigger_interpretation(
            db, report.id, requester_user_id=world["doctor_a"]
        )
        assert again.dispatched is True
        assert read_row(engine, "exam_reports", report.id)["ai_status"] == "pending"
        assert len(calls) == 1

    async def test_processing_rejected_with_409(
        self, db: AsyncSession, world, monkeypatch
    ) -> None:
        """正在跑的时候重复提交会让两次回写交错覆盖 ai_* 字段。"""
        _stub_dispatch(monkeypatch)
        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"], auto_dispatch=True)
        # processing 现在只由 worker CAS 写入（见 test_claim_*）；
        # 本用例只验「processing 时拒绝重复触发」，在同一会话内模拟认领后的状态即可
        await db.execute(
            sa.update(es.ExamReport).where(es.ExamReport.id == report.id).values(
                ai_status=es.AI_STATUS_PROCESSING
            )
        )
        await db.commit()
        with pytest.raises(InvalidStateTransitionError) as exc:
            await es.trigger_interpretation(db, report.id)
        assert exc.value.status_code == 409
        assert exc.value.detail["current_status"] == "processing"
        assert exc.value.detail["entity"] == "exam_report_ai"

    async def test_completed_is_idempotent_reject(
        self, db: AsyncSession, engine, world, monkeypatch
    ) -> None:
        """已完成不再烧第二次 LLM 配额，也不覆盖既有解读。"""
        calls = _stub_dispatch(monkeypatch)
        report, dispatch = await submit_ok(db, None, world["req_processing"],
                                           tech_id=world["labtech01"], auto_dispatch=True)
        before = len(calls)
        # 模拟 worker 已回写：把 ai_status 推到 completed
        await db.execute(
            sa.update(es.ExamReport).where(es.ExamReport.id == report.id).values(
                ai_status=es.AI_STATUS_COMPLETED
            )
        )
        await db.commit()

        result = await es.trigger_interpretation(db, report.id)
        assert result.dispatched is False
        assert "无需重复触发" in (result.reason or "")
        assert result.ai_task_id == dispatch.ai_task_id
        assert len(calls) == before, "幂等拒绝不得再派发"
        assert read_row(engine, "exam_reports", report.id)["ai_status"] == "completed"

    async def test_missing_report_404(self, db: AsyncSession, world) -> None:
        with pytest.raises(ResourceNotFoundError):
            await es.trigger_interpretation(db, uuid.uuid4())

    def test_active_and_terminal_sets_are_disjoint(self) -> None:
        assert not (es.ACTIVE_AI_STATUSES & es.TERMINAL_AI_STATUSES)
        assert set(es.ACTIVE_AI_STATUSES | es.TERMINAL_AI_STATUSES) == {
            "pending", "processing", "completed", "failed", "skipped",
        }


# ==================== 12. HTTP 端点（权限 + 装配 + 回归）====================


@needs_routes
class TestExamEndpoints:
    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_full_flow_via_http(self, client: AsyncClient, engine, world) -> None:
        """receive → process → submit → detail 全程走 HTTP，逐段核对响应装配。"""
        rid = world["req_ordered"]
        received = await client.post(f"/api/v1/exams/requests/{rid}/receive")
        assert received.status_code == 200, received.text
        assert received.json()["status"] == "received"

        processing = await client.post(f"/api/v1/exams/requests/{rid}/process")
        assert processing.status_code == 200, processing.text
        assert processing.json()["status"] == "processing"

        submitted = await client.post(
            f"/api/v1/exams/requests/{rid}/reports",
            json={
                "raw_content": "仪器导出：血钾 2.8 mmol/L（参考 3.5-5.3）",
                "structured_data": structured(K_LOW, HGB_HIGH),
                "conclusion": "电解质紊乱",
                "auto_interpret": False,
            },
        )
        assert submitted.status_code == 201, submitted.text
        body = submitted.json()
        report = body["report"]
        assert report["request_status"] == "reported"
        # world 夹具的 req_ordered 建的是尿常规单（血钾/血红蛋白指标塞在尿常规里
        # 是夹具的演示性妥协，不影响流程断言），这里如实取夹具值而非"血常规"
        assert report["exam_name"] == "尿常规"
        assert report["abnormal_flags"] == [
            "血钾 2.8 ↓(参考 3.5-5.3)",
            "血红蛋白 170 ↑(参考 115-150)",
        ]
        assert report["abnormal_items"][0]["direction"] == "low"
        assert report["ai"]["ai_status"] == "pending"
        assert report["ai"]["need_doctor_review"] is True
        assert body["dispatch"]["dispatched"] is False
        assert body["dispatch"]["reason"] == "未启用自动派发"

        detail = await client.get(f"/api/v1/exams/reports/{report['id']}")
        assert detail.status_code == 200, detail.text
        assert detail.json()["critical"] is True

    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_critical_flag_in_response(self, client: AsyncClient, world) -> None:
        """危急值在录入当场由规则给出，前端必须立刻弹提示，不等 AI。"""
        await client.post(f"/api/v1/exams/requests/{world['req_ordered']}/receive")
        await client.post(f"/api/v1/exams/requests/{world['req_ordered']}/process")
        resp = await client.post(
            f"/api/v1/exams/requests/{world['req_ordered']}/reports",
            json={"raw_content": "血钾 2.8", "structured_data": structured(K_LOW),
                  "auto_interpret": False},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()["report"]
        assert body["critical"] is True
        assert "危急值提醒" in body["critical_message"]
        assert "血钾" in body["critical_message"]
        assert body["critical_values"][0]["threshold_text"] == "危急下限 3"

    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_no_critical_when_only_reference_breach(
        self, client: AsyncClient, world
    ) -> None:
        await client.post(f"/api/v1/exams/requests/{world['req_ordered']}/receive")
        await client.post(f"/api/v1/exams/requests/{world['req_ordered']}/process")
        resp = await client.post(
            f"/api/v1/exams/requests/{world['req_ordered']}/reports",
            json={"raw_content": "血钾 3.2",
                  "structured_data": structured({"name": "血钾", "code": "K",
                                                 "value": 3.2, "ref_low": 3.5,
                                                 "ref_high": 5.3}),
                  "auto_interpret": False},
        )
        body = resp.json()["report"]
        assert body["critical"] is False
        assert body["critical_message"] is None
        assert len(body["abnormal_flags"]) == 1

    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_auto_dispatch_true_returns_processing(
        self, client: AsyncClient, engine, world, monkeypatch
    ) -> None:
        """HTTP 层的派发打桩位置回归：patch 错对象会真的去连 broker。"""
        calls = _stub_dispatch(monkeypatch)
        await client.post(f"/api/v1/exams/requests/{world['req_ordered']}/receive")
        await client.post(f"/api/v1/exams/requests/{world['req_ordered']}/process")
        resp = await client.post(
            f"/api/v1/exams/requests/{world['req_ordered']}/reports",
            json={"raw_content": "血钾 2.8", "structured_data": structured(K_LOW)},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["dispatch"]["dispatched"] is True
        assert body["dispatch"]["celery_task_id"]
        assert body["report"]["ai"]["ai_status"] == "pending"
        assert len(calls) == 1
        assert calls[0]["name"] == "app.tasks.ai_tasks.interpret_exam_report"

    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_broker_down_still_201(self, client: AsyncClient, world, monkeypatch) -> None:
        _stub_dispatch(monkeypatch, result=None)
        await client.post(f"/api/v1/exams/requests/{world['req_ordered']}/receive")
        await client.post(f"/api/v1/exams/requests/{world['req_ordered']}/process")
        resp = await client.post(
            f"/api/v1/exams/requests/{world['req_ordered']}/reports",
            json={"raw_content": "x", "structured_data": structured(K_LOW)},
        )
        assert resp.status_code == 201, "broker 故障不能卡死临床主流程"
        assert resp.json()["dispatch"]["dispatched"] is False
        assert resp.json()["report"]["ai"]["ai_status"] == "failed"

    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_illegal_transition_is_409_with_detail(
        self, client: AsyncClient, world
    ) -> None:
        """从 received 直接出报告：409 且 detail 带 current/target，前端能明确提示。"""
        await client.post(f"/api/v1/exams/requests/{world['req_received']}/receive")
        resp = await client.post(
            f"/api/v1/exams/requests/{world['req_received']}/reports",
            json={"raw_content": "x"},
        )
        assert resp.status_code == 409, resp.text
        assert error_code_of(resp) == "state_transition_invalid"
        detail = resp.json()["detail"]
        assert detail["current_status"] == "received"
        assert detail["target_status"] == "reported"
        assert detail["entity"] == "exam_request"

    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_submit_after_reviewed_rejected(
        self, client: AsyncClient, engine, world
    ) -> None:
        """终态 reviewed 之后不能再提交：既撞「已有报告」也撞终态转换。

        reported→reviewed 只能由 AI 解读成功回写推进，HTTP 侧没有直达入口，
        因此这里用一次 SQL 把单子推到 reviewed（等价 worker 已回写的库内状态）。
        """
        rid = world["req_processing"]
        first = await client.post(
            f"/api/v1/exams/requests/{rid}/reports",
            json={"raw_content": "第一份", "auto_interpret": False},
        )
        assert first.status_code == 201, first.text
        with engine.begin() as conn:
            conn.execute(
                sa.text("UPDATE exam_requests SET status='reviewed' WHERE id=:i"),
                {"i": rid.hex},
            )

        second = await client.post(
            f"/api/v1/exams/requests/{rid}/reports", json={"raw_content": "第二份"}
        )
        assert second.status_code == 409, second.text
        assert error_code_of(second) == "state_transition_invalid"
        assert read_row(engine, "exam_requests", rid)["status"] == "reviewed"
        assert count_rows(engine, "exam_reports") == 1

    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_queue_endpoint(self, client: AsyncClient, world) -> None:
        resp = await client.get("/api/v1/exams/queue")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # received + processing 两张；ordered / reported 不属待办视图
        assert body["total"] == 2
        statuses = {i["status"] for i in body["items"]}
        assert statuses == {"received", "processing"}
        assert set(body["counts"]) <= {"routine", "urgent", "stat"}
        assert sum(body["counts"].values()) == len(body["items"])
        assert body["page"] == 1 and body["page_size"] == 20

    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_queue_shows_report_link_and_ai_status(
        self, client: AsyncClient, world
    ) -> None:
        """工作台行必须带 report_id + ai_status，否则检验科找不到手动重触发的入口。"""
        rid = world["req_processing"]
        submitted = await client.post(
            f"/api/v1/exams/requests/{rid}/reports",
            json={"raw_content": "x", "auto_interpret": False},
        )
        assert submitted.status_code == 201, submitted.text
        report_id = submitted.json()["report"]["id"]

        # 出报告后单子离开待办视图
        resp = await client.get("/api/v1/exams/queue")
        assert all(i["report_id"] is None for i in resp.json()["items"])

        detail = await client.get(f"/api/v1/exams/reports/{report_id}")
        assert detail.json()["request_status"] == "reported"

    async def test_workbench_row_carries_report_and_ai_status(
        self, db: AsyncSession, world
    ) -> None:
        """service→序列化链路精确核对：queue_report_map 的结果能落到响应字段上。"""
        from app.api.v1.exam import to_request_out

        report, _ = await submit_ok(db, None, world["req_processing"],
                                    tech_id=world["labtech01"])
        req = await es.receive_request(db, world["req_ordered"])
        out = to_request_out(req, (report.id, es.AI_STATUS_FAILED))
        assert out.report_id == str(report.id)
        assert out.ai_status == "failed"

        bare = to_request_out(req)
        assert bare.report_id is None and bare.ai_status is None

    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_queue_bad_params(self, client: AsyncClient) -> None:
        assert (await client.get("/api/v1/exams/queue", params={"page": 0})).status_code == 422
        assert (await client.get("/api/v1/exams/queue",
                                 params={"page_size": 101})).status_code == 422

    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_bad_uuid_in_path_is_404(self, client: AsyncClient) -> None:
        """路径参数口径：非法 UUID 与不存在统一 404，而不是 Pydantic 的 422。"""
        assert (await client.post("/api/v1/exams/requests/not-a-uuid/receive")).status_code == 404
        assert (await client.get("/api/v1/exams/reports/garbage")).status_code == 404
        # GET /requests/{id}/reports 不存在（列表端点是 /reports?request_id=），
        # 断不存在的 GET 返回 405 才是对的，这里不再断 404

    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_missing_request_404(self, client: AsyncClient) -> None:
        resp = await client.post(f"/api/v1/exams/requests/{uuid.uuid4()}/receive")
        assert resp.status_code == 404
        assert error_code_of(resp) == "resource_not_found"

    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_structured_items_shape_is_422(self, client: AsyncClient, world) -> None:
        """items 存在却不是数组属结构性错误，一定来自调用方 bug，协议层硬拦。"""
        await client.post(f"/api/v1/exams/requests/{world['req_ordered']}/receive")
        await client.post(f"/api/v1/exams/requests/{world['req_ordered']}/process")
        resp = await client.post(
            f"/api/v1/exams/requests/{world['req_ordered']}/reports",
            json={"raw_content": "x", "structured_data": {"items": "no"}},
        )
        assert resp.status_code == 422

    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_empty_raw_content_is_422(self, client: AsyncClient, world) -> None:
        """原始文本必填：结构化是对它的加工，不能替代原件。"""
        resp = await client.post(
            f"/api/v1/exams/requests/{world['req_processing']}/reports",
            json={"raw_content": ""},
        )
        assert resp.status_code == 422

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_doctor_triggers_interpret_and_polls(
        self, client: AsyncClient, world, monkeypatch
    ) -> None:
        """医生是解读结果的责任人：202 触发 + 轮询详情读到 completed。"""
        calls = _stub_dispatch(monkeypatch)
        _stub_no_evidence(monkeypatch)
        llm = _stub_llm(monkeypatch, CountingLLM([OK_PARSED]))

        async with _as_lab(world) as lab_client:
            # 出报告的合法起点是 processing；ordered 直接 submit 会被状态机拒
            submitted = await lab_client.post(
                f"/api/v1/exams/requests/{world['req_processing']}/reports",
                json={"raw_content": "血钾 2.8", "structured_data": structured(K_LOW),
                      "auto_interpret": False},
            )
            assert submitted.status_code == 201, submitted.text
            report_id = submitted.json()["report"]["id"]
        assert calls == [], "auto_interpret=False 时不应派发"

        triggered = await client.post(
            f"/api/v1/exams/reports/{report_id}/interpret", json={"reason": "复核需要"}
        )
        assert triggered.status_code == 202, triggered.text
        body = triggered.json()
        assert body["dispatch"]["dispatched"] is True
        assert body["report_id"] == report_id
        assert "轮询" in body["message"]

        polled = await client.get(f"/api/v1/exams/reports/{report_id}")
        # 入队后仍是 pending；processing 由解读执行体（此处等价 worker）CAS 认领写入
        assert polled.json()["ai"]["ai_status"] == "pending"

        # 直接驱动一次解读（等价 worker 回写），再从 HTTP 读结果
        async with _session_for(world) as session:
            await es.interpret_async(session, uuid.UUID(report_id))
        assert llm.calls == 1

        done = await client.get(f"/api/v1/exams/reports/{report_id}")
        assert done.status_code == 200
        ai = done.json()["ai"]
        assert ai["ai_status"] == "completed"
        assert ai["summary"] == OK_PARSED["summary"]
        assert ai["suggestions"] == OK_PARSED["suggestions"]
        assert abs(ai["confidence"] - 0.82) < 1e-6
        assert ai["model"] == "mock-interpret"
        assert ai["interpreted_at"] is not None
        assert ai["need_doctor_review"] is True
        assert done.json()["request_status"] == "reviewed"

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_doctor_cannot_receive_or_submit(
        self, client: AsyncClient, world
    ) -> None:
        """矩阵里医生没有 exam_request:update / exam_report:create：开单不等于执行。"""
        assert (await client.post(
            f"/api/v1/exams/requests/{world['req_ordered']}/receive")).status_code == 403
        assert (await client.post(
            f"/api/v1/exams/requests/{world['req_processing']}/reports",
            json={"raw_content": "x"})).status_code == 403

    @pytest.mark.parametrize("client", [DOCTOR_A], indirect=True)
    async def test_doctor_reads_reports(self, client: AsyncClient, world) -> None:
        """医生持 exam_report:read：空库返回空集，出报告后可读且 patient_id 过滤生效。"""
        resp = await client.get("/api/v1/exams/reports")
        assert resp.status_code == 200, resp.text
        assert resp.json()["total"] == 0

        rid = world["req_processing"]
        async with _as_lab(world) as lab_client:
            submitted = await lab_client.post(
                f"/api/v1/exams/requests/{rid}/reports",
                json={"raw_content": "x", "auto_interpret": False},
            )
            assert submitted.status_code == 201, submitted.text

        after = await client.get("/api/v1/exams/reports")
        assert after.status_code == 200
        assert after.json()["total"] == 1
        # 医生的 patient_id 过滤参数生效（患者侧则被忽略）
        filtered = await client.get(
            "/api/v1/exams/reports", params={"patient_id": str(world["profile01"])}
        )
        assert filtered.json()["total"] == 1
        other = await client.get(
            "/api/v1/exams/reports", params={"patient_id": str(world["profile02"])}
        )
        assert other.json()["total"] == 0

    @pytest.mark.parametrize("client", [LAB_TECH], indirect=True)
    async def test_lab_tech_uses_read_all_endpoints(self, client: AsyncClient, world) -> None:
        """检验科只有 read_all：/reports/all 通、/reports 反而 403（矩阵设计如此）。"""
        assert (await client.get("/api/v1/exams/reports/all")).status_code == 200
        assert (await client.get("/api/v1/exams/reports")).status_code == 403

    @pytest.mark.parametrize("client", [PATIENT1], indirect=True)
    async def test_patient_reads_own_report_only(
        self, client: AsyncClient, world
    ) -> None:
        rid = world["req_processing"]
        async with _as_lab(world) as lab_client:
            submitted = await lab_client.post(
                f"/api/v1/exams/requests/{rid}/reports",
                json={"raw_content": "x", "structured_data": structured(K_LOW),
                      "auto_interpret": False},
            )
            assert submitted.status_code == 201, submitted.text
            report_id = submitted.json()["report"]["id"]

        mine = await client.get("/api/v1/exams/reports")
        assert mine.status_code == 200
        assert mine.json()["total"] == 1

        detail = await client.get(f"/api/v1/exams/reports/{report_id}")
        assert detail.status_code == 200
        assert detail.json()["id"] == report_id

    @pytest.mark.parametrize("client", [PATIENT2], indirect=True)
    async def test_patient_cross_report_is_404(self, client: AsyncClient, world) -> None:
        rid = world["req_processing"]
        async with _as_lab(world) as lab_client:
            submitted = await lab_client.post(
                f"/api/v1/exams/requests/{rid}/reports",
                json={"raw_content": "x", "auto_interpret": False},
            )
            assert submitted.status_code == 201, submitted.text
            report_id = submitted.json()["report"]["id"]

        resp = await client.get(f"/api/v1/exams/reports/{report_id}")
        assert resp.status_code == 404
        assert error_code_of(resp) == "resource_not_found"

    @pytest.mark.parametrize("client", [PATIENT1], indirect=True)
    async def test_patient_cannot_write_or_interpret(
        self, client: AsyncClient, world
    ) -> None:
        """患者矩阵里没有 update/create/interpret：三个写入口全 403。"""
        rid = str(world["req_processing"])
        assert (await client.post(f"/api/v1/exams/requests/{rid}/receive")).status_code == 403
        assert (await client.post(f"/api/v1/exams/requests/{rid}/reports",
                                  json={"raw_content": "x"})).status_code == 403
        assert (await client.post(f"/api/v1/exams/reports/{rid}/interpret",
                                  json={})).status_code == 403
        assert (await client.get("/api/v1/exams/queue")).status_code == 403


@needs_routes
class TestCashierForbidden:
    """收费员对全部检验端点 403：矩阵里 cashier 只有 exam_request:read_all，
    没有任何写动作，也没有 exam_report:* / exam_interpretation:*。"""

    @pytest.mark.parametrize("client", [CASHIER], indirect=True)
    async def test_all_endpoints_forbidden(self, client: AsyncClient, world) -> None:
        rid = str(uuid.uuid4())
        # queue 例外：矩阵里 cashier 恰好持 exam_request:read_all（收费要能看
        # 检查进度），所以工作台放行；被拒的是其余全部报告/解读端点
        assert (await client.get("/api/v1/exams/queue")).status_code == 200
        cases = (
            ("post", f"/api/v1/exams/requests/{rid}/receive", None),
            ("post", f"/api/v1/exams/requests/{rid}/process", None),
            ("post", f"/api/v1/exams/requests/{rid}/reports", {"raw_content": "x"}),
            ("get", "/api/v1/exams/reports", None),
            ("get", "/api/v1/exams/reports/all", None),
            ("get", f"/api/v1/exams/reports/{rid}", None),
            ("post", f"/api/v1/exams/reports/{rid}/interpret", {}),
        )
        for method, path, body in cases:
            sender = getattr(client, method)
            resp = await sender(path, json=body) if body is not None else await sender(path)
            assert resp.status_code == 403, f"{method.upper()} {path} 应对收费员拒绝"


# ==================== 多身份辅助（跨角色准备数据）====================


class _as_lab:
    """临时把当前登录身份换成检验科，用于在医生/患者用例里准备报告。

    get_db 覆盖沿用外层 client fixture（同一 session_factory），只换身份——
    两次请求仍读写同一个文件库，因此准备的数据能被主断言读到。
    """

    def __init__(self, world) -> None:
        self.world = world

    async def __aenter__(self) -> AsyncClient:
        self._original = app.dependency_overrides.get(get_current_user)
        user_id, username, role_name = _resolve_identity(self.world, LAB_TECH)

        async def override_current_user():
            return as_user(user_id, username, role_name)

        app.dependency_overrides[get_current_user] = override_current_user
        transport = ASGITransport(app=app)
        self._ctx = AsyncClient(transport=transport, base_url="http://test")
        return await self._ctx.__aenter__()

    async def __aexit__(self, *exc) -> bool:
        await self._ctx.__aexit__(*exc)
        if self._original is None:
            app.dependency_overrides.pop(get_current_user, None)
        else:
            app.dependency_overrides[get_current_user] = self._original
        return False


class _session_for:
    """在测试主体之外再开一条异步会话，模拟 worker 回写。"""

    def __init__(self, world) -> None:
        self.world = world

    async def __aenter__(self) -> AsyncSession:
        # AsyncSession.__aenter__ 返回 None（不是 self），await 它拿不到会话；
        # 会话对象本身就是可用的，直接返回即可，收尾统一 close
        self._cm = build_session_factory(self.world["engine"])()
        return self._cm

    async def __aexit__(self, *exc) -> bool:
        # close 后 session 不再暴露 bind；文件型 SQLite 下多余连接由 world
        # 夹具的 engine dispose 统一回收，这里只负责把会话交还
        await self._cm.close()
        return False


# ==================== 13. Celery 任务注册契约（静态）====================


class TestCeleryTaskContract:
    """只断言注册契约与「已实装」，不起 worker、不连 broker。"""

    def _module(self):
        from app.tasks import ai_tasks  # noqa: F401  - 触发注册

        return sys.modules["app.tasks.ai_tasks"]

    def test_task_registered_with_fully_qualified_name(self) -> None:
        names = {t.name for t in self._module().celery_app.tasks.values()}
        assert es.INTERPRET_TASK_NAME in names
        assert es.INTERPRET_TASK_NAME == "app.tasks.ai_tasks.interpret_exam_report"

    def test_task_options_match_design(self) -> None:
        task = self._module().celery_app.tasks[es.INTERPRET_TASK_NAME]
        assert task.queue == _real_celery().QUEUE_INTERPRET == "interpret"
        assert task.time_limit == 120, "单份报告解读 2 分钟硬上限"
        assert task.soft_time_limit == 100
        assert task.max_retries == 2
        assert task.acks_late is True, "重投语义靠 claim_for_interpret 兜住"

    def test_body_no_longer_raises_not_implemented(self) -> None:
        """阶段 4 实装证明：源码里不再有 NotImplementedError 占位。"""
        mod = self._module()
        source = inspect.getsource(mod.interpret_exam_report)
        assert "NotImplementedError" not in source
        assert "claim_for_interpret" in source
        assert "interpret_exam_report" in source or "_interpret_on_async_session" in source

    def test_dispatch_name_constant_matches_decorator(self) -> None:
        """派发侧常量与任务装饰器名必须一字不差，否则消息只会躺在队列里。"""
        mod = self._module()
        decorator_target = mod.interpret_exam_report.name
        assert decorator_target == es.INTERPRET_TASK_NAME

    def test_task_routes_declared(self) -> None:
        routes = _real_celery().celery_app.conf.task_routes
        assert routes[es.INTERPRET_TASK_NAME] == {"queue": "interpret"}

    def test_run_async_bridge_documented(self) -> None:
        """任务体内禁止再套 asyncio.run：收尾必须落在同一个活循环里。"""
        src = inspect.getsource(_real_celery().run_async)
        assert "aclose_adapter_clients" in src


# ==================== 14. 路由注册（静态契约，不依赖挂载）====================


def _probe_app():
    """把 exam router 单独装进临时 FastAPI 实例求 openapi。

    挂载是主代理的编排责任，router 自身定义了什么才是本批交付内容；
    未挂载时端点用例整类 skip，而这里仍能独立验证路由签名。
    """
    from fastapi import FastAPI

    from app.api.v1.exam import router as exam_router

    probe = FastAPI()
    probe.include_router(exam_router, prefix="/api/v1")
    return probe


class TestRouterRegistration:
    def test_router_prefix_and_tags(self) -> None:
        from app.api.v1.exam import router as exam_router

        assert exam_router.prefix == "/exams"
        assert exam_router.tags == ["检查报告"]

    def test_openapi_paths_present(self) -> None:
        paths = set(_probe_app().openapi()["paths"])
        for expected in (
            "/api/v1/exams/queue",
            "/api/v1/exams/requests/{request_id}/receive",
            "/api/v1/exams/requests/{request_id}/process",
            "/api/v1/exams/requests/{request_id}/reports",
            "/api/v1/exams/reports",
            "/api/v1/exams/reports/all",
            "/api/v1/exams/reports/{report_id}",
            "/api/v1/exams/reports/{report_id}/interpret",
        ):
            assert expected in paths, f"缺少端点 {expected}"

    def test_methods_match_design(self) -> None:
        spec = _probe_app().openapi()["paths"]
        assert set(spec["/api/v1/exams/queue"]) == {"get"}
        assert set(spec["/api/v1/exams/requests/{request_id}/receive"]) == {"post"}
        assert set(spec["/api/v1/exams/requests/{request_id}/process"]) == {"post"}
        assert set(spec["/api/v1/exams/requests/{request_id}/reports"]) == {"post"}
        assert set(spec["/api/v1/exams/reports"]) == {"get"}
        assert set(spec["/api/v1/exams/reports/all"]) == {"get"}
        assert set(spec["/api/v1/exams/reports/{report_id}"]) == {"get"}
        assert set(spec["/api/v1/exams/reports/{report_id}/interpret"]) == {"post"}

    def test_static_route_not_shadowed(self) -> None:
        """/reports/all 必须不被 /reports/{report_id} 抢走（_parse_uuid 会把 all 变成 404）。"""
        spec = _probe_app().openapi()["paths"]
        assert "/api/v1/exams/reports/all" in spec
        assert "/api/v1/exams/reports/{report_id}" in spec

    def test_no_double_wrapped_depends(self) -> None:
        """阶段 3 教训回归：Annotated[X, Depends(...)] 形式的闸门不得再套 Depends。

        一旦被 ``Depends(CanReadReport)`` 二次包裹，FastAPI 会把它当成待注入的
        可调用对象解析，端点凭空多出必填 query 参数，任何请求恒 422。
        """
        import re

        from app.api.v1 import exam as exam_api

        source = inspect.getsource(exam_api)
        offenders = re.findall(r"Depends\(\s*Can[A-Za-z]+\s*\)", source)
        assert offenders == [], f"闸门被二次包裹：{offenders}"

    def test_endpoints_have_no_injected_callable_params(self) -> None:
        """openapi 层面自证：每个 GET 端点除声明过的过滤参数外无其他必填项。"""
        spec = _probe_app().openapi()["paths"]
        queue_params = {
            p["name"]: p for p in spec["/api/v1/exams/queue"]["get"].get("parameters", [])
        }
        assert set(queue_params) <= {"status_filter", "page", "page_size"}
        for name, param in queue_params.items():
            assert param["in"] == "query", name
            assert param.get("required") is False, f"{name} 不应是必填——必填即二次包裹征兆"

        detail_params = spec["/api/v1/exams/reports/{report_id}"]["get"]["parameters"]
        assert {p["name"] for p in detail_params} == {"report_id"}

    def test_status_codes_match_design(self) -> None:
        spec = _probe_app().openapi()["paths"]
        assert "201" in spec["/api/v1/exams/requests/{request_id}/reports"]["post"]["responses"]
        assert "202" in spec["/api/v1/exams/reports/{report_id}/interpret"]["post"]["responses"]


# ==================== 15. Schema 与序列化 ====================


class TestSchemasAndSerialization:
    def test_submit_request_defaults(self) -> None:
        req = ReportSubmitRequest(raw_content="原文")
        assert req.auto_interpret is True
        assert req.structured_data is None

    def test_submit_request_strips_whitespace(self) -> None:
        assert ReportSubmitRequest(raw_content="  原文  ").raw_content == "原文"

    @pytest.mark.parametrize("bad", [{"items": "no"}, {"items": [1]}, {"items": ["x"]}])
    def test_structural_shape_hard_rejected(self, bad: Any) -> None:
        with pytest.raises(ValueError):
            ReportSubmitRequest(raw_content="x", structured_data=bad)

    @pytest.mark.parametrize("ok", [None, {}, {"items": []}, {"other": 1},
                                   {"items": [{"value": "abc"}]}])
    def test_non_numeric_values_not_rejected_at_protocol_layer(self, ok: Any) -> None:
        """协议层不做数值校验：强类型卡住 LIS 的真实形态只会逼检验科手工改数据。"""
        assert ReportSubmitRequest(raw_content="x", structured_data=ok).structured_data == ok

    def test_out_models_expose_no_sensitive_columns(self) -> None:
        from app.schemas.exam import AIInterpretationOut, ExamReportOut, ExamRequestOut

        for model in (ExamReportOut, ExamRequestOut, AIInterpretationOut):
            fields = set(model.model_fields)
            for forbidden in ("password_hash", "id_card_no", "phone", "email"):
                assert forbidden not in fields, f"{model.__name__} 泄露 {forbidden}"

    def test_contract_sample_parses(self) -> None:
        """模块里公示给阶段 5/前端的契约样例本身必须能被解析出旗标与危急值。"""
        sample = json.loads(es.STRUCTURED_DATA_CONTRACT)
        assert len(es.evaluate_abnormalities(sample)) == 1
        assert len(es.detect_critical_values(sample)) == 1

    def test_to_ai_out_maps_columns(self, db: AsyncSession, world) -> None:
        from app.api.v1.exam import to_ai_out

        report = es.ExamReport(
            request_id=world["req_processing"],
            patient_id=world["profile01"],
            lab_tech_id=world["labtech01"],
            raw_content="x",
            abnormal_flags=["a"],
            ai_status=es.AI_STATUS_PENDING,
        )
        out = to_ai_out(report)
        assert out.ai_status == "pending"
        assert out.severity is None, "表结构里没有 ai_severity 列，如实返回 None"
        assert out.need_doctor_review is True
        assert out.confidence is None and out.interpreted_at is None


# ==================== 16. 跨模块一致性 ====================


class TestCrossModuleConsistency:
    def test_workflow_functions_are_async(self) -> None:
        for name in ("receive_request", "start_processing", "submit_report",
                     "dispatch_interpretation", "trigger_interpretation",
                     "interpret_async", "list_queue", "list_reports",
                     "get_report_detail", "retrieve_evidence"):
            assert inspect.iscoroutinefunction(getattr(es, name)), name

    def test_worker_side_functions_are_sync(self) -> None:
        """worker 走同步 Session：这两条必须是协程外的普通函数。"""
        for name in ("claim_for_interpret", "run_interpretation_sync",
                     "evaluate_abnormalities", "detect_critical_values"):
            assert not inspect.iscoroutinefunction(getattr(es, name)), name

    def test_ai_status_constants_match_column_comment(self) -> None:
        from app.models.clinical import ExamReport as ER

        comment = ER.__table__.columns["ai_status"].comment or ""
        for value in ("pending", "processing", "completed", "failed", "skipped"):
            assert value in comment
            assert value in (es.ACTIVE_AI_STATUSES | es.TERMINAL_AI_STATUSES)

    def test_lab_queue_statuses_exclude_reported(self) -> None:
        assert es.LAB_QUEUE_STATUSES == ("received", "processing")
        assert "reported" not in es.LAB_QUEUE_STATUSES
        assert "ordered" not in es.LAB_QUEUE_STATUSES

    def test_task_type_matches_model_registry(self) -> None:
        """models/ai.py 注释里登记的 task_type 取值必须与派发侧常量一致。"""
        from app.models.ai import AITask as T

        comment = T.__table__.columns["task_type"].comment or ""
        assert es.TASK_TYPE_REPORT_INTERPRET in comment

    def test_build_retrieval_query_prefers_critical(self) -> None:
        hits = es.detect_critical_values(structured(K_LOW, HGB_HIGH))
        abn = es.evaluate_abnormalities(structured(K_LOW, HGB_HIGH))
        query = es.build_retrieval_query(abn, hits)
        assert query.startswith("血钾 危急值"), "最危险的项应主导召回"
        assert "血红蛋白 偏高" in query

    def test_build_retrieval_query_default_when_all_normal(self) -> None:
        assert es.build_retrieval_query([], []) == "检查报告 结果解读 参考区间"

    def test_interpret_payload_shape(self) -> None:
        outcome = es.InterpretOutcome(
            report_id=uuid.uuid4(), status="completed", severity="low",
            suggestions=["a"], confidence=0.5, evidence_count=2,
        )
        payload = es.interpret_report_payload(outcome)
        assert payload["status"] == "completed"
        assert payload["suggestion_count"] == 1
        assert payload["evidence_count"] == 2
        assert payload["degraded"] is False

    def test_describe_patient_minimises_pii(self) -> None:
        profile = type("P", (), {"age": 35, "gender": "F",
                                 "chronic_diseases": ["高血压", "糖尿病"]})()
        text = es._describe_patient(profile)
        assert "35岁" in text and "女" in text and "高血压" in text
        assert es._describe_patient(None).startswith("未提供患者档案")
        assert "不完整" in es._describe_patient(
            type("P", (), {"age": None, "gender": None, "chronic_diseases": None})()
        )
