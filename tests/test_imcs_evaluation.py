"""AI 评测域单元测试（阶段 6 批次 B）。

覆盖：
1. 纯判定函数：别名并集、按 title 去重、检索命中与排名、质量断言（拒答/覆盖）。
2. 聚合与门禁：三指标、零分母 None、按类下钻、baseline/changed/四档回归。
3. ragas 辅助：分数归一（NaN/inf/None）、均值跳过无效值。
4. 种子：幂等（二次运行 created=0）。
5. run_retrieval_evaluation（假检索器）：落库、拒答题被排除、指标正确。
6. dispatch_ragas_evaluation（打桩派发）：正常派发两表落库、活跃任务 409、
   broker 不可用时两表收口 failed 且抛 503。
7. API：admin CRUD 与触发、doctor/patient 403、RAGAS 派发 202、非法 kind 404。

SQLite 注意：eval_runs 主键 UUID 由应用侧生成，BigInteger 补丁不涉及本域；
知识指纹查 knowledge_documents（空表 → hash None），门禁 changed 分支直接
用 dict 单测覆盖，不依赖真实知识库内容。
"""

from __future__ import annotations

import math
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

import app.api.v1.evaluation as eval_api
import app.services.evaluation_service as es
from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.exceptions import DuplicateTaskError, TaskQueueUnavailableError
from app.imcs_main import app
from app.models.ai import AITask, AITaskStatus
from app.models.auth import User
from app.models.base import Base
from app.models.evaluation import EvalCase, EvalKind, EvalRun, EvalRunStatus
from app.seed.eval_cases import EVAL_CASES, seed_eval_cases
from app.services.ragas_runner import calculate_ragas_averages, normalize_ragas_score
from tests.imcs_sqlite import (
    ROLE_IDS,
    build_session_factory,
    error_code_of,
    seed_role_permissions,
    seed_roles,
)

EVAL_TABLES = (
    "roles", "permissions", "role_permissions", "departments", "users",
    "patient_profiles", "knowledge_documents", "eval_cases", "eval_runs",
    "ai_tasks",
)


@pytest.fixture()
def engine(tmp_path):
    import tests.imcs_sqlite  # noqa: F401

    db_path = tmp_path / "imcs_eval.db"
    test_engine = sa.create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    tables = [Base.metadata.tables[name] for name in EVAL_TABLES]
    with test_engine.begin() as conn:
        for table in tables:
            table.create(conn, checkfirst=True)
        seed_roles(conn)
        seed_role_permissions(conn)
    yield test_engine
    test_engine.dispose()


@asynccontextmanager
async def open_session(engine):  # noqa: ANN001
    factory = build_session_factory(engine)
    try:
        async with factory() as session:
            yield session
    finally:
        await factory.kw["bind"].dispose()


def hit(title: str, doc_type: str = "condition", score: float = 0.01, **kw: Any):
    return SimpleNamespace(
        title=title, doc_type=doc_type, rrf_score=score,
        content=kw.get("content", f"{title} 的正文内容"),
        source=kw.get("source", "project"),
    )


def case_dict(**kw: Any) -> dict[str, Any]:
    base = {
        "id": str(uuid.uuid4()), "question": "q", "expected_name": "标准名",
        "expected_type": "condition", "category": "c", "alternative_names": [],
        "answer_keywords": [], "citation_names": [], "expected_refusal": False,
    }
    base.update(kw)
    return base


# ====================================================================
# 一、纯判定
# ====================================================================


def test_acceptable_names_union() -> None:
    names = es.acceptable_names(
        case_dict(expected_name="高血压：健康教育与就医警示",
                  alternative_names=["高血压", "  ", "HBP"])
    )
    assert names == {"高血压：健康教育与就医警示", "高血压", "HBP"}


def test_distinct_by_title_keeps_first_order() -> None:
    rows = [hit("A"), hit("B"), hit("A"), hit("C")]
    out = es.distinct_by_title(rows)
    assert [r.title for r in out] == ["A", "B", "C"]


def test_retrieval_hit_rank_and_alias() -> None:
    results = [hit("X"), hit("高血压", "guideline"), hit("高血压：健康教育与就医警示", "guideline")]
    got = es.build_retrieval_result(
        results, case_dict(expected_name="高血压：健康教育与就医警示",
                           expected_type="guideline",
                           alternative_names=["高血压"]),
    )
    # 别名以 rank2 命中，Top1 判负但 Recall@3 通过；
    # 窗口内「高血压」(别名) 与「高血压：健康教育与就医警示」(全名) 是两条
    # 不同 title 的资料，relevant_count 如实计 2。
    assert got["expected_rank"] == 2
    assert got["passed"] is True
    assert got["relevant_count"] == 2


def test_retrieval_wrong_type_not_hit() -> None:
    got = es.build_retrieval_result(
        [hit("同名资料", "guideline")],
        case_dict(expected_name="同名资料", expected_type="condition"),
    )
    assert got["expected_rank"] is None and got["passed"] is False


def test_retrieval_miss_beyond_window() -> None:
    results = [hit(f"n{i}") for i in range(2)] + [hit("目标")]
    got = es.build_retrieval_result(results, case_dict(expected_name="目标"))
    assert got["expected_rank"] == 3 and got["passed"] is True
    far = [hit(f"n{i}") for i in range(3)] + [hit("目标")]
    got2 = es.build_retrieval_result(far, case_dict(expected_name="目标"))
    assert got2["expected_rank"] == 4 and got2["passed"] is False
    # 窗口截断：retrieved_count 恒 ≤ 3
    assert got2["retrieved_count"] == 3


def test_quality_refusal_and_coverage() -> None:
    r1 = es.build_quality_result([], case_dict(expected_refusal=True))
    assert r1["refusal_correct"] is True and r1["answer_correct"] is True
    r2 = es.build_quality_result([hit("资料A")], case_dict(expected_refusal=True))
    assert r2["refusal_correct"] is False
    r3 = es.build_quality_result(
        [hit("甲", content="本资料含 剂量 说明")],
        case_dict(expected_name="甲", answer_keywords=["剂量"]),
    )
    assert r3["answer_correct"] is True and r3["citation_correct"] is True


# ==================== 聚合与门禁 ====================


def test_aggregate_retrieval_values_and_zero_denominator() -> None:
    rows = [
        {"expected_rank": 1, "passed": True, "relevant_count": 1, "retrieved_count": 2,
         "question": "a", "category": "x"},
        {"expected_rank": None, "passed": False, "relevant_count": 0, "retrieved_count": 0,
         "question": "b", "category": "y"},
        {"expected_rank": 2, "passed": True, "relevant_count": 1, "retrieved_count": 3,
         "question": "c", "category": "x"},
    ]
    agg = es.aggregate_retrieval(rows)
    assert agg["total_count"] == 3
    assert agg["top1_accuracy"] == round(1 / 3, 4)
    assert agg["recall_at_3"] == round(2 / 3, 4) and agg["precision_at_3"] == round(2 / 5, 4)
    assert agg["by_category"]["x"]["top1_accuracy"] == 0.5
    empty = es.aggregate_retrieval([])
    assert empty["top1_accuracy"] is None and empty["precision_at_3"] is None


def test_apply_gate_levels() -> None:
    cur = {"top1_accuracy": 0.7, "knowledge_hash": "h1"}
    prev = {"top1_accuracy": 0.82, "knowledge_hash": "h1"}
    g = es.apply_gate(cur, prev)
    assert g["status"] == "compared" and g["metrics"]["top1_accuracy"]["level"] == "attention"
    assert es.apply_gate(cur, None)["status"] == "baseline"
    assert es.apply_gate({"knowledge_hash": "a"}, {"knowledge_hash": "b"})["status"] == "changed"
    mild = es.apply_gate({"top1_accuracy": 0.75, "knowledge_hash": "h"},
                         {"top1_accuracy": 0.80, "knowledge_hash": "h"})
    assert mild["metrics"]["top1_accuracy"]["level"] == "warning"
    up = es.apply_gate({"top1_accuracy": 0.9, "knowledge_hash": "h"},
                       {"top1_accuracy": 0.8, "knowledge_hash": "h"})
    assert up["metrics"]["top1_accuracy"]["level"] == "improved"


def test_ragas_score_normalization() -> None:
    assert normalize_ragas_score("0.12345") == 0.1235
    assert normalize_ragas_score(None) is None
    assert normalize_ragas_score(float("nan")) is None
    assert normalize_ragas_score(math.inf) is None
    avg = calculate_ragas_averages(
        [{"faithfulness": 1.0}, {"faithfulness": None, "context_recall": 0.5}]
    )
    assert avg["faithfulness"] == 1.0
    assert avg["answer_relevancy"] is None and avg["context_recall"] == 0.5


# ==================== 种子与运行（带库） ====================


async def test_seed_eval_cases_idempotent(engine) -> None:
    async with open_session(engine) as db:
        first = await seed_eval_cases(db)
        await db.commit()
        assert first["eval_cases_created"] == len(EVAL_CASES)
        assert first["eval_cases_skipped"] == 0
    async with open_session(engine) as db:
        second = await seed_eval_cases(db)
        await db.commit()
        assert second["eval_cases_created"] == 0
        assert second["eval_cases_skipped"] == len(EVAL_CASES)
    with engine.connect() as conn:
        n = conn.execute(sa.text("SELECT COUNT(*) FROM eval_cases")).scalar_one()
    assert n == len(EVAL_CASES)


async def test_seed_cases_reference_real_titles() -> None:
    """题库的 expected_name 必须能在种子知识文档里找到（防指标失真）。"""
    from app.seed.knowledge import _ALL_SEEDS

    titles = {s.title for s in _ALL_SEEDS}
    for spec in EVAL_CASES:
        if spec["expected_refusal"]:
            continue
        assert spec["expected_name"] in titles, spec["expected_name"]


class FakeRetriever:
    """按问题文本查预置结果；命中不了就是空列表。"""

    def __init__(self, mapping: dict[str, list[Any]]) -> None:
        self.mapping = mapping
        self.calls = 0

    async def retrieve(self, session: AsyncSession, query: str, **kw: Any):
        self.calls += 1
        results = self.mapping.get(query, [])
        return SimpleNamespace(results=results)


async def test_run_retrieval_evaluation_end_to_end(engine) -> None:
    async with open_session(engine) as db:
        db.add_all([
            EvalCase(**{k: v for k, v in case_dict(
                question="怎么治高血压？", expected_name="高血压：健康教育与就医警示",
                expected_type="guideline", category="健教").items() if k != "id"}),
            EvalCase(**{k: v for k, v in case_dict(
                question="咳嗽看哪科？", expected_name="呼吸内科常见病概览",
                category="专科").items() if k != "id"}),
            EvalCase(**{k: v for k, v in case_dict(
                question="医保报销比例？", expected_refusal=True,
                category="拒答校验").items() if k != "id"}),
        ])
        await db.commit()

    fake = FakeRetriever({
        "怎么治高血压？": [hit("呼吸内科常见病概览"), hit("高血压：健康教育与就医警示", "guideline")],
        "咳嗽看哪科？": [hit("呼吸内科常见病概览")],
    })
    async with open_session(engine) as db:
        run = await es.run_retrieval_evaluation(db, retriever=fake)
    assert run.status == EvalRunStatus.COMPLETED.value
    # 拒答题被排除：只评 2 题、只调 2 次检索
    assert run.total_count == 2 and fake.calls == 2
    assert run.top1_accuracy == 0.5      # 呼吸内科 rank1，高血压 rank2
    assert run.recall_at_3 == 1.0
    # 精度：两题合计检索 3 条（第一题返回 2 条，第二题 1 条），命中 2 条
    assert run.precision_at_3 == round(2 / 3, 4)
    assert run.metrics["by_category"]["健教"]["recall_at_3"] == 1.0
    assert run.metrics["gate"]["status"] == "baseline"
    # results 按题库排序返回，按题目定位而非依赖下标顺序
    by_q = {r["question"]: r for r in run.results}
    assert len(by_q) == 2
    assert by_q["怎么治高血压？"]["expected_rank"] == 2
    assert by_q["咳嗽看哪科？"]["expected_rank"] == 1

    # 第二次运行：与上一条 completed 比 → compared 档
    async with open_session(engine) as db:
        run2 = await es.run_retrieval_evaluation(db, retriever=fake)
    assert run2.metrics["gate"]["status"] == "compared"

    async with open_session(engine) as db:
        runs, total = await es.list_runs(db, kind="retrieval")
        assert total == 2 and all(r.results is None for r in runs)  # 列表瘦身
        trend = await es.run_trend(db, "retrieval")
        assert len(trend["points"]) == 2 and trend["baseline"]["run_id"] == str(run.id)
        full = await es.get_run(db, run.id)
        assert full.results is not None


async def test_dispatch_ragas_success_and_duplicate(engine, monkeypatch) -> None:
    dispatched: list[str] = []

    def fake_dispatch(name, *args, queue=None, task_id=None, **kw):  # noqa: ANN001
        dispatched.append(task_id)
        return SimpleNamespace(id=task_id)

    import app.tasks.celery_app  # noqa: F401

    ca_mod = sys.modules["app.tasks.celery_app"]
    monkeypatch.setattr(ca_mod, "dispatch_task", fake_dispatch)

    async with open_session(engine) as db:
        run, task = await es.dispatch_ragas_evaluation(db, sample_size=5)
        assert run.kind == EvalKind.RAGAS.value and run.status == "pending"
        assert task.task_type == es.TASK_TYPE_RAGAS and run.ai_task_id == task.id
        assert dispatched == [str(task.id)]

    async with open_session(engine) as db:
        with pytest.raises(DuplicateTaskError):
            await es.dispatch_ragas_evaluation(db)


async def test_dispatch_ragas_broker_down(engine, monkeypatch) -> None:
    import app.tasks.celery_app  # noqa: F401  确保模块对象已存在

    ca_mod = sys.modules["app.tasks.celery_app"]
    monkeypatch.setattr(ca_mod, "dispatch_task", lambda *a, **kw: None)
    async with open_session(engine) as db:
        with pytest.raises(TaskQueueUnavailableError):
            await es.dispatch_ragas_evaluation(db)
    async with open_session(engine) as db:
        _, total = await es.list_runs(db, kind="ragas", status="failed")
        assert total == 1
        task = (await db.execute(sa.select(AITask))).scalars().first()
        assert task.status == AITaskStatus.FAILED.value


# ==================== API ====================


def as_user(role: str) -> User:
    user = User(
        id=uuid.uuid4(), username=f"u_{role}", real_name="测试",
        password_hash="", role_id=ROLE_IDS[role], is_active=True,
    )
    user._role_name = role  # type: ignore[attr-defined]
    return user


@pytest.fixture()
def client_factory(engine):
    import app.imcs_main as imcs_main_module

    @asynccontextmanager
    async def _build(user: User):
        from app.core.database import async_session_factory  # noqa: F401

        local = build_session_factory(engine)

        async def override_db() -> AsyncIterator[AsyncSession]:
            async with local() as session:
                yield session

        async def override_user() -> User:
            return user

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[get_current_user] = override_user
        originals = (imcs_main_module.async_session_factory,)
        imcs_main_module.async_session_factory = local
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield client
        finally:
            app.dependency_overrides.clear()
            (imcs_main_module.async_session_factory,) = originals
            await local.kw["bind"].dispose()

    yield _build


async def test_api_case_crud_and_trigger(client_factory, monkeypatch) -> None:
    monkeypatch.setattr(
        eval_api, "RAGRetriever",
        lambda: FakeRetriever({"q1": [hit("甲")]}),
    )
    async with client_factory(as_user("admin")) as client:
        created = await client.post(
            "/api/v1/evaluation/cases",
            json={"question": "q1", "expected_name": "甲",
                  "expected_type": "condition", "category": "t"},
        )
        assert created.status_code == 201, created.text
        cid = created.json()["id"]

        listing = await client.get("/api/v1/evaluation/cases")
        assert listing.json()["total"] == 1

        upd = await client.put(
            f"/api/v1/evaluation/cases/{cid}", json={"category": "t2"}
        )
        assert upd.json()["category"] == "t2"

        bad = await client.put(f"/api/v1/evaluation/cases/{'x'*12}", json={"category": "z"})
        assert bad.status_code == 404

        run = await client.post("/api/v1/evaluation/runs", params={"kind": "retrieval"})
        assert run.status_code == 201, run.text
        body = run.json()
        assert body["top1_accuracy"] == 1.0 and body["results"]

        bad_kind = await client.post("/api/v1/evaluation/runs", params={"kind": "ragas"})
        assert bad_kind.status_code == 404

        runs = await client.get("/api/v1/evaluation/runs")
        assert runs.json()["total"] == 1
        detail = await client.get(f"/api/v1/evaluation/runs/{body['id']}")
        assert detail.json()["results"][0]["expected_rank"] == 1
        trend = await client.get("/api/v1/evaluation/trends", params={"kind": "retrieval"})
        assert len(trend.json()["points"]) == 1

        deleted = await client.delete(f"/api/v1/evaluation/cases/{cid}")
        assert deleted.status_code == 204


async def test_api_ragas_dispatch_202(client_factory, monkeypatch) -> None:
    import app.tasks.celery_app  # noqa: F401

    ca_mod = sys.modules["app.tasks.celery_app"]
    monkeypatch.setattr(
        ca_mod, "dispatch_task",
        lambda name, *a, queue=None, task_id=None, **kw: SimpleNamespace(id=task_id),
    )
    async with client_factory(as_user("admin")) as client:
        resp = await client.post("/api/v1/evaluation/ragas", json={"sample_size": 3})
        assert resp.status_code == 202, resp.text
        assert resp.json()["sample_size"] == 3
        # 重复派发 409
        again = await client.post("/api/v1/evaluation/ragas", json={})
        assert again.status_code == 409
        assert error_code_of(again) == "task_already_running"


@pytest.mark.parametrize("role", ["doctor", "patient", "pharmacist"])
async def test_api_non_admin_forbidden(client_factory, role: str) -> None:
    async with client_factory(as_user(role)) as client:
        for url in (
            "/api/v1/evaluation/cases",
            "/api/v1/evaluation/runs",
            "/api/v1/evaluation/trends?kind=retrieval",
        ):
            resp = await client.get(url)
            assert resp.status_code == 403, f"{role} {url} -> {resp.status_code}"
        resp = await client.post(
            "/api/v1/evaluation/runs", params={"kind": "retrieval"}
        )
        assert resp.status_code == 403
