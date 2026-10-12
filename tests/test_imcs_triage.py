"""智能分诊工作流单元测试（文件型 SQLite，不打真实 LLM/RAG/网络）。

覆盖 triage_workflow.assess 的四个分支 + triage 路由的鉴权与响应装配。
LLM 用 monkeypatch 替换 get_llm_service().structured_chat，RAG 证据检索与
号源摘要各自打桩——真实模型调用与号源联动由容器集成实测覆盖。

**为什么急症分支要断言「LLM 一次没被调用」**：急症拦截是医疗安全红线，
必须在模型完全不可用时仍然给出急诊引导。测试用「计数桩」记录调用次数
而不是用「被调就抛」的炸弹桩——后者看似更严格，但 _extract_symptoms
的 except Exception 会吞掉炸弹抛出的 AssertionError，让断言以误导性的方式
通过或失败（实测踩过）；计数桩直接验证「calls == 0」，语义清晰且不受
兜底逻辑影响。

departments 的 symptom_keywords 列是 TextArray：SQLite 下按 JSON 数组存，
必须用 json.dumps 写入，逗号串会让 process_result_value 解析失败静默返回 []，
规则匹配随之全体失效（首轮实测正是这个原因导致 8 条失败）。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.imcs_main import app
from app.models.auth import User
from app.models.base import Base
from app.services import triage_workflow as tw
from app.services.llm.types import (
    DegradationLevel,
    LLMResponse,
    TokenUsage,
)
from tests.imcs_sqlite import build_session_factory, seed_role_permissions, seed_roles

ROLE_USER_IDS: dict[str, int] = {
    "patient": 1,
    "doctor": 2,
    "lab_tech": 3,
    "cashier": 4,
    "pharmacist": 5,
    "admin": 6,
}

TRIAGE_TABLES: tuple[str, ...] = (
    "roles",
    "permissions",
    "role_permissions",
    "departments",
    "users",
    "patient_profiles",
    "triage_results",
    "ai_tasks",
)

# 口径取自 app/seed/departments.py：EMERGENCY + 若干带症状关键词的分诊目标
_DEPTS = [
    ("EMERGENCY", "急诊科", ["胸痛", "昏迷", "外伤"]),
    ("CARDIO", "心血管内科", ["胸痛", "心悸", "胸闷", "血压高"]),
    ("RESP", "呼吸内科", ["咳嗽", "咳痰", "气促", "呼吸困难"]),
    ("GASTRO", "消化内科", ["腹痛", "腹泻", "恶心", "反酸"]),
    ("ENDO", "内分泌科", ["多饮", "多尿", "血糖高", "消瘦"]),
    ("NEURO", "神经内科", ["头痛", "头晕", "失眠"]),
]


def _import_sqlite_patch() -> None:
    import tests.imcs_sqlite  # noqa: F401  - 注册 PGUUID 的 SQLite DDL 编译器


@pytest.fixture()
def engine(tmp_path):
    _import_sqlite_patch()
    db_path = tmp_path / "imcs_triage.db"
    test_engine = sa.create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    tables = [Base.metadata.tables[name] for name in TRIAGE_TABLES]
    with test_engine.begin() as conn:
        for table in tables:
            table.create(conn, checkfirst=True)
        seed_roles(conn)
        seed_role_permissions(conn)
        for code, name, kws in _DEPTS:
            conn.execute(
                sa.text(
                    "INSERT INTO departments (id, name, code, symptom_keywords)"
                    " VALUES (:id, :name, :code, :kw)"
                ),
                {
                    "id": uuid.uuid4().hex,
                    "name": name,
                    "code": code,
                    # TextArray 在 SQLite 走 JSON 数组，必须 json.dumps
                    "kw": json.dumps(kws, ensure_ascii=False),
                },
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


# ==================== LLM 替身 ====================


def _resp(model: str = "mock", degraded: bool = False) -> LLMResponse:
    response = LLMResponse(
        content="{}",
        usage=TokenUsage(),
        provider="openai",
        model=model,
    )
    if degraded:
        response.degradation_level = DegradationLevel.FALLBACK
    return response


class CountingLLM:
    """按顺序返回预设结构化结果，并记录被调用次数。

    responses[i] 是 dict（用对应 schema 构造）或 None（模拟校验失败）；
    超出长度时复用最后一项，方便 extract / department 两次调用共用配置。
    """

    def __init__(self, responses: list[Any], *, degraded: bool = False) -> None:
        self.responses = responses
        self.degraded = degraded
        self.calls = 0

    async def structured_chat(self, request: Any, schema: Any):
        item = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        response = _resp(degraded=self.degraded)
        if item is None:
            return None, response
        return (schema(**item) if isinstance(item, dict) else item), response

    async def embed(self, texts: Any, **kw: Any):  # pragma: no cover - 兜底拦截
        raise AssertionError("分诊单测不得调用真实 embed")


EXTRACT_CHEST = {"symptoms": ["胸闷", "心悸"], "severity": 3, "accompanying": []}
DEPT_CARDIO = {"department_name": "心血管内科", "reason": "胸闷心悸提示心源性"}


def _stub_no_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """证据检索固定为空：知识库尚未向量化，且真实检索要走 embed。"""

    async def _empty(db_: AsyncSession, query_text_: str, **kw_: Any) -> list[Any]:
        return []

    monkeypatch.setattr(tw, "_gather_evidence", _empty)


# ==================== 1. 急症拦截 ====================


async def test_emergency_short_circuits_before_llm(db, monkeypatch):
    counting = CountingLLM([EXTRACT_CHEST])
    monkeypatch.setattr(tw, "get_llm_service", lambda: counting)
    _stub_no_evidence(monkeypatch)

    outcome = await tw.assess(db, chief_complaint="突然胸痛倒地，意识不清叫不醒")
    assert outcome.is_emergency is True
    assert outcome.decision_source == "emergency"
    assert outcome.urgency == "stat"
    assert outcome.department_code == "EMERGENCY"
    assert counting.calls == 0, "急症分支绝不能调用 LLM"
    assert "120" in (outcome.advice or "")


async def test_emergency_persists_and_skips_llm(db, monkeypatch):
    counting = CountingLLM([EXTRACT_CHEST])
    monkeypatch.setattr(tw, "get_llm_service", lambda: counting)
    _stub_no_evidence(monkeypatch)

    outcome = await tw.assess(db, chief_complaint="孩子高热惊厥抽搐")
    record = await tw.get_result(db, outcome.triage_id)
    assert record.is_emergency is True
    assert record.decision_source == "emergency"
    assert record.recommended_department_name == "急诊科"
    # 命中顺序跟随词表定义顺序（"抽搐"在前），断言用集合不锁顺序
    assert set(record.matched_emergency_keywords or []) == {"高热惊厥", "抽搐"}
    assert counting.calls == 0


async def test_non_emergency_does_call_llm(db, monkeypatch):
    """反证：非急症主诉确实会走 LLM 提取，急症短路是精确命中而非一律拦截。"""
    counting = CountingLLM([EXTRACT_CHEST, DEPT_CARDIO])
    monkeypatch.setattr(tw, "get_llm_service", lambda: counting)
    _stub_no_evidence(monkeypatch)

    await tw.assess(db, chief_complaint="最近总咳嗽有痰")
    assert counting.calls >= 1


# ==================== 2. AI 正常路径 ====================


async def test_ai_path_hybrid_agreement(db, monkeypatch):
    """模型与规则同时指向心血管内科 → hybrid，置信度最高。"""
    counting = CountingLLM([EXTRACT_CHEST, DEPT_CARDIO])
    monkeypatch.setattr(tw, "get_llm_service", lambda: counting)
    _stub_no_evidence(monkeypatch)

    outcome = await tw.assess(db, chief_complaint="活动后心前区闷堵伴心慌一个月")
    assert outcome.decision_source == "hybrid"
    assert outcome.department_name == "心血管内科"
    assert outcome.confidence is not None and outcome.confidence >= 0.7
    assert counting.calls == 2


async def test_ai_illegal_department_falls_back_to_rule(db, monkeypatch):
    """模型幻觉出不存在的科室 → 合法性校验拦下，落回规则结果。"""
    counting = CountingLLM(
        [
            {"symptoms": ["腹痛"], "severity": 2, "accompanying": []},
            {"department_name": "量子医学部", "reason": "（幻觉）"},
        ]
    )
    monkeypatch.setattr(tw, "get_llm_service", lambda: counting)
    _stub_no_evidence(monkeypatch)

    outcome = await tw.assess(db, chief_complaint="吃完东西肚子疼")
    assert outcome.department_name != "量子医学部"
    assert outcome.decision_source == "rule"
    assert outcome.department_name == "消化内科"


# ==================== 3. 降级与坏输出 ====================


async def test_degraded_llm_uses_rule(db, monkeypatch):
    """LLM 降级时仍出结论：走规则命中呼吸内科，advice 带降级提示。"""
    counting = CountingLLM(
        [{"symptoms": ["咳嗽"], "severity": 2, "accompanying": []}], degraded=True
    )
    monkeypatch.setattr(tw, "get_llm_service", lambda: counting)
    _stub_no_evidence(monkeypatch)

    outcome = await tw.assess(db, chief_complaint="咳嗽咳痰三天")
    assert outcome.decision_source == "rule"
    assert outcome.degraded is True
    assert outcome.department_name == "呼吸内科"
    assert "降级" in (outcome.advice or "")


async def test_structured_invalid_json_no_500(db, monkeypatch):
    """结构化输出校验失败（parsed=None）→ 兜底规则，绝不抛异常中断流程。"""
    counting = CountingLLM([None])
    monkeypatch.setattr(tw, "get_llm_service", lambda: counting)
    _stub_no_evidence(monkeypatch)

    outcome = await tw.assess(db, chief_complaint="恶心反酸烧心")
    assert outcome.decision_source == "rule"
    assert outcome.department_name == "消化内科"


async def test_rule_only_skips_llm_and_embed(db, monkeypatch):
    """批量先行的规则路径：整条 LLM 链路（含 embed）零调用。"""
    counting = CountingLLM([EXTRACT_CHEST])
    monkeypatch.setattr(tw, "get_llm_service", lambda: counting)
    _stub_no_evidence(monkeypatch)

    outcome = await tw.assess(db, chief_complaint="多饮多尿消瘦血糖高", rule_only=True)
    assert outcome.decision_source == "rule"
    assert outcome.department_name == "内分泌科"
    assert counting.calls == 0


# ==================== 路由 ====================


def _make_user(engine, role_name: str, username: str) -> uuid.UUID:
    uid = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO users (id, username, password_hash, real_name, role_id, is_active)"
                " VALUES (:id, :username, :pw, :name, :role_id, 1)"
            ),
            {"id": uid.hex, "username": username, "pw": "x",
             "name": username, "role_id": ROLE_USER_IDS[role_name]},
        )
    return uid


def as_user(user_id: uuid.UUID, role_name: str) -> User:
    user = User(
        id=user_id,
        username=role_name,
        real_name=role_name,
        password_hash="",
        role_id=ROLE_USER_IDS[role_name],
        is_active=True,
    )
    user._role_name = role_name  # type: ignore[attr-defined]
    return user


async def _make_client(
    engine,
    session_factory,
    monkeypatch,
    *,
    role_name: str,
    username: str,
    with_profile: bool,
    llm: CountingLLM,
):
    """装配指定身份的 ASGI 客户端：覆盖 get_db / get_current_user / 证据桩。"""
    import app.imcs_main as imcs_main_module
    from app.api.v1 import triage as triage_router_mod

    uid = _make_user(engine, role_name, username)
    if with_profile:
        with engine.begin() as conn:
            conn.execute(
                sa.text("INSERT INTO patient_profiles (id, user_id) VALUES (:id, :uid)"),
                {"id": uuid.uuid4().hex, "uid": uid.hex},
            )

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    async def override_current_user() -> User:
        return as_user(uid, role_name)

    async def fake_check_redis() -> bool:
        return True

    async def _no_slots(db_: AsyncSession, outcome_: Any) -> list[Any]:
        return []

    monkeypatch.setattr(tw, "get_llm_service", lambda: llm)
    monkeypatch.setattr(triage_router_mod, "_suggest_slots", _no_slots)
    _stub_no_evidence(monkeypatch)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_current_user
    originals = (imcs_main_module.async_session_factory, imcs_main_module.check_redis)
    imcs_main_module.async_session_factory = session_factory
    imcs_main_module.check_redis = fake_check_redis

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
    (
        imcs_main_module.async_session_factory,
        imcs_main_module.check_redis,
    ) = originals


PATIENT = ("patient", "p_a", True)
CASHIER = ("cashier", "c_wu", False)


@pytest.fixture()
async def patient_client(engine, session_factory, monkeypatch):
    async for c in _make_client(
        engine, session_factory, monkeypatch,
        role_name="patient", username="p_a", with_profile=True,
        llm=CountingLLM([EXTRACT_CHEST, DEPT_CARDIO]),
    ):
        yield c


@pytest.fixture()
async def cashier_client(engine, session_factory, monkeypatch):
    async for c in _make_client(
        engine, session_factory, monkeypatch,
        role_name="cashier", username="c_wu", with_profile=False,
        llm=CountingLLM([EXTRACT_CHEST]),
    ):
        yield c


async def test_assess_endpoint_201(patient_client):
    resp = await patient_client.post(
        "/api/v1/triage/assess", json={"chief_complaint": "活动后心前区闷堵伴心慌"}
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["triage_id"]
    assert body["decision_source"] in ("ai", "hybrid", "rule")
    assert body["recommended_department_name"]


async def test_assess_emergency_endpoint_ignores_llm(patient_client, monkeypatch):
    """急症请求即使换成计数桩也应 201 且不调 LLM。"""
    counting = CountingLLM([EXTRACT_CHEST])
    monkeypatch.setattr(tw, "get_llm_service", lambda: counting)
    resp = await patient_client.post(
        "/api/v1/triage/assess", json={"chief_complaint": "胸痛呼吸困难叫不醒"}
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["is_emergency"] is True
    assert counting.calls == 0


async def test_results_list_for_patient(patient_client):
    await patient_client.post(
        "/api/v1/triage/assess", json={"chief_complaint": "活动后心前区闷堵"}
    )
    resp = await patient_client.get("/api/v1/triage/results")
    assert resp.status_code == 200, resp.text
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["recommended_department_name"]


async def test_chief_complaint_too_short_422(patient_client):
    resp = await patient_client.post("/api/v1/triage/assess", json={"chief_complaint": "疼"})
    assert resp.status_code == 422


async def test_cashier_cannot_create_triage(cashier_client):
    """收费员矩阵里没有 triage 授权 → 403。"""
    resp = await cashier_client.post(
        "/api/v1/triage/assess", json={"chief_complaint": "心前区闷堵不适"}
    )
    assert resp.status_code == 403, resp.text


async def test_cashier_cannot_read_triage(cashier_client):
    resp = await cashier_client.get("/api/v1/triage/results")
    assert resp.status_code == 403


# ==================== 路由挂载与任务注册（静态契约）====================


@pytest.fixture()
async def admin_client(engine, session_factory, monkeypatch):
    async for c in _make_client(
        engine, session_factory, monkeypatch,
        role_name="admin", username="root_adm", with_profile=False,
        llm=CountingLLM([EXTRACT_CHEST]),
    ):
        yield c


async def test_batch_assess_202_not_422(admin_client, monkeypatch):
    """批量分诊端点的回归测试。

    曾经的真实 bug：签名里把已经是 ``Annotated[User, Depends(...)]`` 的
    ``CanCreate`` 又套了一层 ``Depends(CanCreate)``，FastAPI 把它当成一个
    待注入的可调用对象去解析，端点凭空多出两个必填 query 参数 args/kwargs，
    任何请求都 422。单测从没通过 HTTP 打过这个端点（只断言过路由存在），
    所以缺陷一路漏到容器 e2e 才暴露——本用例把 HTTP 路径补上。
    """
    import sys

    celery_mod = sys.modules["app.tasks.celery_app"]

    class _R:
        id = "celery-batch-1"

    monkeypatch.setattr(celery_mod, "dispatch_task", lambda *a, **k: _R())

    resp = await admin_client.post("/api/v1/triage/batch-assess", json={"items": [
        {"chief_complaint": "肚子痛拉肚子三天"},
        {"chief_complaint": "反酸烧心饭后加重"},
    ]})
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["ai_task_id"]
    # 规则分诊同步落库，两条都该有结果（rule_only 分支不调 LLM）
    listing = await admin_client.get("/api/v1/triage/results")
    assert len(listing.json()) >= 2


def test_triage_routes_registered():
    paths = app.openapi()["paths"]
    assert "/api/v1/triage/assess" in paths
    assert "/api/v1/triage/results" in paths
    assert "/api/v1/triage/batch-assess" in paths


def test_triage_tasks_registered_with_full_names():
    """任务注册名是部署契约，写错 worker 就接不到活。

    celery 的 tasks 键是**全限定名**（app.tasks.ai_tasks.xxx），
    断言里必须带前缀——只写方法名是上一版用例的失误。
    """
    from app.tasks import ai_tasks  # noqa: F401  - 触发注册

    import sys

    mod = sys.modules["app.tasks.ai_tasks"]
    names = {t.name for t in mod.celery_app.tasks.values()}
    assert "app.tasks.ai_tasks.run_triage_workflow" in names
    # 阶段 4 / 6 的任务仍应完成册（实装与否不由本文件负责）
    assert "app.tasks.ai_tasks.interpret_exam_report" in names
    assert "app.tasks.ai_tasks.run_ragas_evaluation" in names
