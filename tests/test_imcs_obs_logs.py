"""阶段 6 批次 A：可观测埋点链路单元测试（检索日志 + 工作流追踪）。

覆盖三件事：

1. **写入契约**：``log_retrieval`` 落 rag_retrieval_logs 的字段、
   ``project_candidates`` 的精简投影、``start/finish_workflow_trace`` 的
   running→completed 生命周期与 node_count 修正。
2. **best-effort 语义**（本批次最关键的红线）：埋点写库失败绝不能让
   用户的检索、分诊、报告解读报错。用假会话工厂抛异常来验证「异常被吞成
   warning、主流程照常返回」。
3. **不回归既有业务语义**：分诊急诊分支仍然零 LLM 调用、报告解读的 CAS
   幂等仍然只跑一次——埋点只是旁观者。

为什么全部打在 ``app.services.observability`` 模块属性上：该模块统一
``from app.core.database import async_session_factory``，埋点函数内部引用的
是模块级名字，所以替换这一处即可同时覆盖 retriever / triage / exam 三条链路
经它落库的路径；打在自己的 engine 文件库上，与生产 PG 无关。
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient  # noqa: F401  - 与其他测试保持导入一致
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.services.observability as obs
from app.models.ai import RAGRetrievalLog, WorkflowTrace
from app.models.base import Base

UTC = timezone.utc

# 埋点涉及的表：业务表 + 四张可观测表（工作流/检索日志的落点）
OBS_TABLES: tuple[str, ...] = (
    "roles",
    "permissions",
    "role_permissions",
    "departments",
    "users",
    "patient_profiles",
    "triage_results",
    "ai_tasks",
    "llm_call_logs",
    "rag_retrieval_logs",
    "workflow_traces",
    "ai_metrics",
)


def _import_sqlite_patch() -> None:
    import tests.imcs_sqlite  # noqa: F401  - 注册 PGUUID / BigInteger 的 SQLite DDL 编译器


@pytest.fixture()
def engine(tmp_path):
    """文件型 SQLite：同步建表 + 异步被测会话共用同一份数据。

    刻意不用内存库：``sqlite://`` 的库生命周期绑定单条连接，埋点函数自开的
    异步会话拿不到同步引擎建好的表，首个 INSERT 就是 no such table。
    """
    _import_sqlite_patch()
    db_path = tmp_path / "imcs_obs.db"
    test_engine = sa.create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    tables = [Base.metadata.tables[name] for name in OBS_TABLES]
    with test_engine.begin() as conn:
        for table in tables:
            table.create(conn, checkfirst=True)
    yield test_engine
    test_engine.dispose()


@pytest.fixture()
def wired(engine, monkeypatch):
    """把埋点模块的会话工厂接到测试库上。

    返回 (engine, factory)，用例直接用它读回落库的行。
    teardown 的 dispose 可能撞上「循环已关」——同步夹具无法复用 async
    用例的事件循环，跨循环关闭 aiosqlite 连接本来就报错，而这些连接
    随进程结束自然释放，静默吞掉即可。
    """
    from tests.imcs_sqlite import build_session_factory

    factory = build_session_factory(engine)
    monkeypatch.setattr(obs, "async_session_factory", factory)
    yield engine, factory
    try:
        asyncio.run(_dispose(factory))
    except Exception:  # noqa: BLE001 - 清理失败不影响用例结论
        pass


async def _dispose(factory) -> None:  # noqa: ANN001
    await factory.kw["bind"].dispose()


def rows_of(engine, table_cls):  # noqa: ANN001
    """用同步引擎直读某张表的全部行（避开异步会话的 identity map 缓存）。"""
    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(sa.text(f"SELECT * FROM {table_cls.__tablename__}")).fetchall()]


# ====================================================================
# 一、log_retrieval：字段落库与 best-effort
# ====================================================================


async def test_log_retrieval_persists_fields(wired) -> None:
    engine, _ = wired
    tid = str(uuid.uuid4())
    uid = uuid.uuid4()
    await obs.log_retrieval(
        trace_id=tid,
        user_id=uid,
        query_text="布洛芬一次吃多少",
        rewritten_query="布洛芬 成人 单次剂量",
        vector_hit_count=12,
        keyword_hit_count=7,
        both_hit_count=4,
        merged_count=15,
        final_count=5,
        is_zero_result=False,
        top_score=0.0328,
        vector_latency_ms=41,
        keyword_latency_ms=17,
        rrf_latency_ms=3,
        rerank_latency_ms=0,
        rerank_channel=None,
        rrf_k=60,
    )
    rows = rows_of(engine, RAGRetrievalLog)
    assert len(rows) == 1
    row = rows[0]
    assert row["trace_id"] == tid
    # SQLite 里 PGUUID 存 32 位 hex，读回也是 hex，两种方言语义一致
    assert row["user_id"] == uid.hex
    assert row["vector_hit_count"] == 12
    assert row["both_hit_count"] == 4
    assert row["final_count"] == 5
    # 裸 SQL 直读绕过了 ORM 的 Boolean 转换，SQLite 返回 0/1 而不是 True/False
    assert not row["is_zero_result"]
    assert row["rrf_k"] == 60
    assert row["rerank_channel"] is None


async def test_log_retrieval_accepts_uuid_string_and_ignores_unknown_keys(wired) -> None:
    """Celery kwargs 传的是字符串 id；多传参数必须被忽略而不是炸。"""
    engine, _ = wired
    uid = uuid.uuid4()
    await obs.log_retrieval(
        trace_id="t-1",
        user_id=str(uid),
        query_text="q",
        this_key_does_not_exist="whatever",
        session=None,  # 调用方误传请求 session，必须被剔除而不是当成列
    )
    row = rows_of(engine, RAGRetrievalLog)[0]
    assert row["user_id"] == uid.hex
    assert row["trace_id"] == "t-1"


async def test_log_retrieval_user_id_garbage_degrades_to_null(wired) -> None:
    """非法 user_id 宁可留空，也不能让整条日志写失败。"""
    engine, _ = wired
    await obs.log_retrieval(trace_id="t-2", user_id="not-a-uuid", query_text="q")
    assert rows_of(engine, RAGRetrievalLog)[0]["user_id"] is None


async def test_log_retrieval_swallows_db_failure(wired, monkeypatch) -> None:
    """**红线**：写库异常只 warning，绝不向上抛。"""
    _, factory = wired

    class _Boom:
        def __call__(self):  # pragma: no cover - 直接抛，不产生会话
            raise RuntimeError("db down")

    monkeypatch.setattr(obs, "async_session_factory", _Boom())
    # 不应抛任何异常
    await obs.log_retrieval(trace_id="t-3", query_text="q")
    assert factory is not None


async def test_log_retrieval_backfills_trace_id_from_contextvar(wired) -> None:
    """调用方没传 trace_id 时，从 contextvar 兜底——这是链路能串起来的关键。"""
    engine, _ = wired
    from app.core.trace import use_trace_id

    with use_trace_id("ctx-trace-1"):
        await obs.log_retrieval(query_text="q")
    assert rows_of(engine, RAGRetrievalLog)[0]["trace_id"] == "ctx-trace-1"


# ====================================================================
# 二、project_candidates：只存元数据、不塞正文
# ====================================================================


class _Hit:
    """模拟 RetrievalResult：属性式访问。"""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def test_project_candidates_keeps_metadata_and_drops_body() -> None:
    hits = [
        _Hit(
            chunk_id=uuid.uuid4(),
            content="这是不该被复制进日志的整段正文" * 50,
            rrf_score=0.0163934426,
            title="布洛芬片说明书",
            doc_type="drug",
            source="drug:BLF001",
            source_tier=1,
            route_details={"vector": {"rank": 1, "score": 0.91}},
        )
    ]
    projected = obs.project_candidates(hits)
    assert len(projected) == 1
    entry = projected[0]
    assert "content" not in entry
    assert set(entry) == {
        "chunk_id", "rank", "rrf_score", "title", "doc_type",
        "source", "source_tier", "routes",
    }
    assert entry["rank"] == 1
    assert entry["rrf_score"] == 0.016393
    assert entry["routes"]["vector"]["rank"] == 1


def test_project_candidates_truncates_and_caps_count() -> None:
    hits = [
        _Hit(
            chunk_id=uuid.uuid4(),
            rrf_score=0.01,
            title="长标题" * 200,
            source="s" * 500,
            doc_type="condition",
        )
        for _ in range(30)
    ]
    projected = obs.project_candidates(hits)
    # 条数上限：30 条只留 MAX_CANDIDATES 条
    assert len(projected) == obs.MAX_CANDIDATES
    assert projected[0]["title"].endswith("…")
    assert projected[0]["source"].endswith("…")
    # rank 从 1 连续递增，回放时顺序可信
    assert [p["rank"] for p in projected] == list(range(1, obs.MAX_CANDIDATES + 1))


def test_project_candidates_tolerates_dict_and_missing_fields() -> None:
    """dict 形态与字段缺失都不能抛——日志组装失败是可容忍的降级。"""
    projected = obs.project_candidates([
        {"chunk_id": "c-1", "rrf_score": "not-a-number", "title": None},
        _Hit(chunk_id=uuid.uuid4()),
    ])
    assert projected[0]["chunk_id"] == "c-1"
    assert projected[0]["rrf_score"] is None
    # 缺 route_details 时**不放 routes 键**（而不是放个 None），
    # 所以只能判成员存在性：先取键再比 None 会直接 KeyError。
    assert "routes" not in projected[1]


# ====================================================================
# 三、workflow_traces：生命周期
# ====================================================================


async def test_start_and_finish_workflow_trace_by_id(wired) -> None:
    engine, _ = wired
    tid = str(uuid.uuid4())
    rid = uuid.uuid4()
    trace_id = await obs.start_workflow_trace(
        tid, "triage", user_id=rid, ref_id=rid, node_count=5
    )
    assert isinstance(trace_id, int)
    row = rows_of(engine, WorkflowTrace)[0]
    assert row["status"] == "running"
    assert row["node_count"] == 5
    assert row["executed_node_count"] == 0
    assert row["completed_at"] is None

    steps = [
        obs.build_step("emergency_gate", "rule", latency_ms=2),
        obs.build_step("llm_extract", "llm", latency_ms=800, tokens=1200),
    ]
    await obs.finish_workflow_trace(
        trace_id,
        status="completed",
        steps=steps,
        total_latency_ms=802,
        total_tokens=1200,
        total_cost_usd=0.004,
        branch_taken="emergency",
        final_output={"urgency": "stat"},
    )
    row = rows_of(engine, WorkflowTrace)[0]
    assert row["status"] == "completed"
    assert row["executed_node_count"] == 2
    assert row["branch_taken"] == "emergency"
    assert row["total_tokens"] == 1200
    assert row["completed_at"] is not None
    steps_back = json.loads(row["steps"]) if isinstance(row["steps"], str) else row["steps"]
    assert [s["node"] for s in steps_back] == ["emergency_gate", "llm_extract"]


async def test_finish_by_trace_id_and_multiple_rows_targets_latest(wired) -> None:
    """同一 trace_id 多条（重复投递）时，收尾只更新最新一条。"""
    engine, _ = wired
    tid = "dup-trace"
    first = await obs.start_workflow_trace(tid, "triage", node_count=5)
    second = await obs.start_workflow_trace(tid, "triage", node_count=5)
    assert first != second

    # 按 trace_id 收尾：命中 id 更大的那条
    await obs.finish_workflow_trace(tid, status="failed", error_message="boom")
    by_id = {r["id"]: r for r in rows_of(engine, WorkflowTrace)}
    assert by_id[second]["status"] == "failed"
    assert by_id[second]["error_message"] == "boom"
    assert by_id[first]["status"] == "running"


async def test_finish_node_count_never_below_executed(wired) -> None:
    """node_count 修正不得小于实际执行数：否则执行率会算出 >100% 的假象。"""
    engine, _ = wired
    trace_id = await obs.start_workflow_trace("nc-trace", "triage", node_count=5)
    steps = [obs.build_step(f"n{i}", "llm") for i in range(7)]
    await obs.finish_workflow_trace(trace_id, status="completed", steps=steps, node_count=1)
    row = rows_of(engine, WorkflowTrace)[0]
    assert row["executed_node_count"] == 7
    assert row["node_count"] == 7


async def test_finish_missing_trace_is_noop(wired) -> None:
    """start 没建成（返回 None）时，finish 按 id/trace 都匹配不到行，必须静默跳过。"""
    engine, _ = wired
    await obs.finish_workflow_trace(999999, status="completed")
    await obs.finish_workflow_trace("never-existed", status="completed")
    assert rows_of(engine, WorkflowTrace) == []


async def test_workflow_trace_swallows_db_failure(wired, monkeypatch) -> None:
    """**红线**：追踪写失败不得让工作流报错，且 finish 收到 None 也要能继续。"""

    class _Boom:
        def __call__(self):  # pragma: no cover
            raise RuntimeError("db down")

    monkeypatch.setattr(obs, "async_session_factory", _Boom())
    assert await obs.start_workflow_trace("x", "triage") is None
    await obs.finish_workflow_trace(None, status="completed")


def test_build_step_snapshot_truncation_and_error_field() -> None:
    step = obs.build_step(
        "node_a",
        "llm",
        latency_ms=12.7,
        tokens=30,
        cost_usd=0.0001234567,
        input_snapshot={"q": "长" * 900},
        error="E" * 900,
    )
    assert step["node"] == "node_a"
    assert step["latency_ms"] == 12  # int 归一
    assert step["tokens"] == 30
    assert step["cost_usd"] == 0.000123
    assert len(step["input"]) == obs.SNAPSHOT_CHARS + 1  # 截断 + 省略号
    assert len(step["error"]) == 501
    assert set(step) >= {"node", "type", "latency_ms", "tokens", "input", "output", "error"}


def test_sum_usage_handles_stub_responses() -> None:
    """桩对象常缺 usage/cost，必须安全累加而不是 KeyError。"""

    class _Resp:
        def __init__(self, tokens: int, cost: Any = None) -> None:
            self.usage = type("U", (), {"total_tokens": tokens})()
            self.cost_usd = cost

    assert obs.sum_usage([_Resp(100, 0.001), _Resp(50)]) == (150, 0.001)
    assert obs.sum_usage([object(), _Resp("bad", "bad")]) == (0, 0.0)
    assert obs.sum_usage([]) == (0, 0.0)


# ====================================================================
# 四、Celery 侧同步封装
# ====================================================================


def test_sync_wrappers_run_without_event_loop(wired) -> None:
    """worker 任务体（无活循环）里用同步封装能真的落库。

    本用例刻意写成同步 def：pytest-asyncio 不给它事件循环，
    正好模拟 prefork 任务体的运行环境。
    """
    engine, _factory = wired
    obs.log_retrieval_sync(trace_id="sync-1", query_text="q")
    assert rows_of(engine, RAGRetrievalLog)[0]["trace_id"] == "sync-1"

    trace_id = asyncio.run(obs.start_workflow_trace("sync-wf", "report_interpret", node_count=3))
    obs.finish_workflow_trace_sync(trace_id, status="completed", steps=[])
    row = rows_of(engine, WorkflowTrace)[0]
    assert row["status"] == "completed"


def test_sync_wrapper_survives_exception(wired, monkeypatch) -> None:
    """同步封装的异常也走 best-effort：不能把 worker 任务带崩。"""

    class _Boom:
        def __call__(self):  # pragma: no cover
            raise RuntimeError("db down")

    monkeypatch.setattr(obs, "async_session_factory", _Boom())
    obs.log_retrieval_sync(trace_id="boom", query_text="q")


def test_sync_wrapper_from_thread_with_fresh_loop(wired) -> None:
    """从另一个线程（无循环）调用同步封装，也应各自建循环并落库。"""
    engine, _ = wired
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            obs.log_retrieval_sync(trace_id="from-thread", query_text="q")
        except BaseException as exc:  # noqa: BLE001 - 收集后断言
            errors.append(exc)

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join(timeout=30)
    assert not errors
    assert rows_of(engine, RAGRetrievalLog)[0]["trace_id"] == "from-thread"
