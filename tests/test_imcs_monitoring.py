"""AI 监控域单元测试（阶段 6 批次 A；文件型 SQLite，不打真实网络、不起容器）。

覆盖范围（对应派单口径的十类用例）：

1. **分位数口径钉死**（``metrics_service.nearest_rank_percentile`` / ``safe_ratio``）：
   空列表→None、单样本→该值、[1..100] 的 p50/p95/p99 按最近秩 ceiling、
   非整除位置（20 样本 p95 → rank=19）、重复值与乱序输入稳定、p 越界钳制。
2. **单桶聚合**（``aggregate_minute_bucket`` / ``compute_bucket_metrics``）：
   llm_call_count / llm_error_rate / llm_p50~p99 / llm_first_token_p95 /
   token_usage（按 model 打 tag）/ token_cost / degradation_rate /
   moderation_block_rate / retrieval_zero_result_rate / task_queue_backlog
   的值与样本一致；比率分母为 0 时不落库；左闭右开窗口衔接。
3. **幂等**：同桶跑两次行数不翻倍、值不累计（删桶重插，而非 ON CONFLICT）。
4. **previous_complete_bucket**：返回上一完整分钟，naive / aware / 跨时区边界。
5. **reap_stale_tasks**（``monitoring_tasks._reap_sync`` + ``_normalize_threshold``）：
   超时 running（按 started_at）/ pending（按 created_at）被标 failed 且
   error_message 带「被僵尸清理任务标记」；未超时与终态不动；阈值参数生效。
6. **replay_trace**：同一 trace_id 的多段日志齐全、LLM 调用按时间排序、
   汇总字段（tokens / cost / 降级数 / P95）正确；不存在→404；缺段→各段独立为 0。
7. **API 权限**：五个端点 admin 200、doctor / patient 403（矩阵只有 ADMIN 有 read_all）。
8. **list_llm_calls**：scenario / status / provider / model / trace / user 过滤 + 分页。
9. **query_metrics**：指标名与时间窗过滤、tags 子集匹配、limit 钳制。
10. **Celery 侧容错**：``run_aggregate`` DB 异常转结构化返回值而不外抛。

============================ 会话模型 ============================

被测 service 自带 commit，因此不共享长会话：每步用 ``open_session()`` 开一个连到
同一文件库的独立异步会话。HTTP 用例沿用 billing/exam 的做法，覆盖 ``get_db`` 与
``get_current_user``，并替换 ``imcs_main`` 模块级 ``async_session_factory`` /
``check_redis``，避免真实 PG 引擎在测试事件循环里建连接。

============================ 两处刻意偏离实现语义的断言（暴露疑似 bug） ============================

- ``test_bucket_endpoints_inclusive_at_upper_bound_is_the_known_gap``：实现是左闭右开，
  而派单口径写的是「窗口内」。这里把两种口径都写成断言，钉死当前行为，差异见报告。
- ``test_reap_running_with_null_started_at_is_left_alone``：pending 分支只按 created_at
  过滤，若出现「status=running 但 started_at IS NULL」的脏行，它既不进 running 命中集
  也不进 pending 命中集，会永远占位。用例按「不动」断言并在注释标注这是漏判面。

SQLite 已知差异（不影响结论）：DateTime 列不存时区偏移，读回是 naive，故不断言 tz 感知；
JSONB 列走 TEXT + JSON 序列化；``llm_call_logs.total_tokens`` 是 PG GENERATED 列，
SQLite 上退化成普通整数列，因此样本必须显式写入 prompt+completion 之和。
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.core.exceptions import ResourceNotFoundError
from app.imcs_main import app
from app.models.ai import (
    AIMetric,
    AITask,
    AITaskStatus,
    LLMCallLog,
    RAGRetrievalLog,
    TriageResult,
    WorkflowTrace,
)
from app.models.auth import User
from app.models.base import Base, utcnow
from app.services import metrics_service as ms
from app.services import monitoring_service as mon
from tests.imcs_sqlite import (
    build_session_factory,
    error_code_of,
    seed_role_permissions,
    seed_roles,
)

# ==================== 建表 ====================

MONITORING_TABLES: tuple[str, ...] = (
    "roles",
    "permissions",
    "role_permissions",
    "departments",
    "users",
    "patient_profiles",
    # AI 可观测性五表 + 分诊产出表（replay_trace 第四段）
    "ai_tasks",
    "llm_call_logs",
    "rag_retrieval_logs",
    "workflow_traces",
    "ai_metrics",
    "triage_results",
)

ROLE_IDS: dict[str, int] = {
    "patient": 1,
    "doctor": 2,
    "lab_tech": 3,
    "cashier": 4,
    "pharmacist": 5,
    "admin": 6,
}

# 聚合基准桶：固定时刻，避免用例依赖真实时钟
BUCKET_START = datetime(2026, 10, 14, 3, 5, 0, tzinfo=UTC)
BUCKET_END = BUCKET_START + timedelta(minutes=1)


@pytest.fixture()
def engine(tmp_path):
    """文件型 SQLite：同步建表 + 异步被测会话共用同一份数据（照抄 billing/exam）。

    不用内存库：``sqlite://`` 内存库生命周期绑定单条连接，同步引擎建完表后
    异步引擎拿到的是另一个空库，首个查询即 no such table。
    库文件名带 monitoring 前缀，与其他测试文件的临时库互不争抢。
    """
    import tests.imcs_sqlite  # noqa: F401  - 注册 PGUUID 的 SQLite DDL 编译器

    db_path = tmp_path / "imcs_monitoring.db"
    test_engine = sa.create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    tables = [Base.metadata.tables[name] for name in MONITORING_TABLES]
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


@asynccontextmanager
async def _session_scope(engine) -> AsyncIterator[AsyncSession]:
    """开一个连到同一文件库的独立异步会话，退出时释放其引擎。"""
    factory = build_session_factory(engine)
    try:
        async with factory() as session:
            yield session
    finally:
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


def add_llm_call(
    db: AsyncSession,
    *,
    scenario: str = "chat",
    provider: str = "dashscope",
    model: str = "qwen-max",
    requested_model: str | None = None,
    status: str = "success",
    latency_ms: int = 100,
    first_token_ms: int | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    cost_usd: float | Decimal | None = 0.001,
    degradation_level: int | None = 0,
    moderation_blocked: bool = False,
    is_stream: bool = False,
    trace_id: str | None = None,
    user_id: uuid.UUID | None = None,
    workflow_node: str | None = None,
    created_at: datetime | None = None,
) -> LLMCallLog:
    """向会话添加一条 LLM 调用日志（ORM add，随调用方事务提交）。

    total_tokens 是 Computed 生成列（PG 与新版 SQLite 都会按
    prompt+completion 自动算并拒绝显式写入），这里一律不赋值，
    读取时由数据库算出，两种方言取值天然一致。
    """
    row = LLMCallLog(
        trace_id=trace_id,
        scenario=scenario,
        provider=provider,
        model=model,
        requested_model=requested_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        first_token_ms=first_token_ms,
        status=status,
        degradation_level=degradation_level,
        moderation_blocked=moderation_blocked,
        is_stream=is_stream,
        user_id=user_id,
        workflow_node=workflow_node,
        created_at=created_at or BUCKET_START + timedelta(seconds=10),
    )
    db.add(row)
    return row


def add_rag_log(
    db: AsyncSession,
    *,
    trace_id: str | None = None,
    query_text: str = "高血压随访要点",
    rewritten_query: str | None = None,
    vector_hit_count: int = 5,
    keyword_hit_count: int = 3,
    both_hit_count: int = 2,
    merged_count: int = 6,
    final_count: int = 4,
    is_zero_result: bool = False,
    top_score: float | None = 0.812,
    candidates: list[Any] | None = None,
    created_at: datetime | None = None,
) -> RAGRetrievalLog:
    row = RAGRetrievalLog(
        trace_id=trace_id,
        query_text=query_text,
        rewritten_query=rewritten_query,
        vector_hit_count=vector_hit_count,
        keyword_hit_count=keyword_hit_count,
        both_hit_count=both_hit_count,
        merged_count=merged_count,
        final_count=final_count,
        is_zero_result=is_zero_result,
        top_score=top_score,
        candidates=candidates,
        created_at=created_at or BUCKET_START + timedelta(seconds=20),
    )
    db.add(row)
    return row


def add_workflow_trace(
    db: AsyncSession,
    *,
    trace_id: str,
    workflow_name: str = "triage",
    status: str = "completed",
    steps: list[Any] | None = None,
    node_count: int = 4,
    executed_node_count: int = 4,
    total_latency_ms: int = 900,
    total_tokens: int = 120,
    total_cost_usd: float | None = 0.004,
    branch_taken: str | None = "emergency",
    final_output: dict[str, Any] | None = None,
    error_message: str | None = None,
    created_at: datetime | None = None,
) -> WorkflowTrace:
    row = WorkflowTrace(
        trace_id=trace_id,
        workflow_name=workflow_name,
        status=status,
        steps=steps,
        node_count=node_count,
        executed_node_count=executed_node_count,
        total_latency_ms=total_latency_ms,
        total_tokens=total_tokens,
        total_cost_usd=total_cost_usd,
        branch_taken=branch_taken,
        final_output=final_output,
        error_message=error_message,
        created_at=created_at or BUCKET_START + timedelta(seconds=5),
    )
    db.add(row)
    return row


def insert_task(
    engine,  # noqa: ANN001
    *,
    status: str,
    task_type: str = "knowledge_rebuild",
    created_at: datetime | None = None,
    started_at: datetime | None = None,
    ref_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """直插一条 ai_tasks 行（僵尸清理走同步会话，样本也同步准备最干净）。"""
    tid = uuid.uuid4()
    now = utcnow()
    _insert(
        engine,
        "ai_tasks",
        {
            "id": tid.hex,
            "task_type": task_type,
            "ref_id": ref_id.hex if ref_id else None,
            "celery_task_id": f"celery-{tid.hex[:8]}",
            "status": status,
            "progress": 0,
            "retry_count": 0,
            "started_at": started_at,
            "created_at": created_at or now,
            "updated_at": now,
        },
    )
    return tid


def task_row(engine, task_id: uuid.UUID) -> dict[str, Any]:  # noqa: ANN001
    """直读任务行，断言库里真实发生了什么。"""
    with engine.begin() as conn:
        row = conn.execute(
            sa.text("SELECT * FROM ai_tasks WHERE id = :id"),  # noqa: S608
            {"id": task_id.hex},
        ).fetchone()
    assert row is not None, "ai_tasks 行不存在"
    return dict(row._mapping)


def count_rows(engine, table: str, where: str = "", params: dict[str, Any] | None = None) -> int:  # noqa: ANN001
    sql = f"SELECT COUNT(*) FROM {table}"  # noqa: S608 - 表名来自常量白名单
    if where:
        sql += f" WHERE {where}"
    with engine.begin() as conn:
        return int(conn.execute(sa.text(sql), params or {}).scalar_one())


async def metric_map(db: AsyncSession, bucket_start: datetime) -> dict[str, list[dict[str, Any]]]:
    """取某桶全部指标行，按 metric_name 分组（同名多 tag 行都在列表里）。"""
    rows = list(
        (
            await db.execute(
                sa.select(AIMetric).where(AIMetric.bucket_time == bucket_start)
            )
        ).scalars().all()
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.metric_name, []).append(
            {
                "value": float(row.metric_value),
                "sample_count": row.sample_count,
                "tags": row.tags or {},
            }
        )
    return grouped


def single(grouped: dict[str, list[dict[str, Any]]], name: str) -> dict[str, Any]:
    """取无 tag 维度的单行指标；缺失即断言失败。"""
    items = grouped.get(name)
    assert items, f"指标 {name} 未落库"
    assert len(items) == 1, f"指标 {name} 期望单行，实际 {len(items)} 行"
    return items[0]


# ====================================================================
# 一、分位数与比率口径（纯函数，钉死实现 docstring 里的三条边界）
# ====================================================================


class TestNearestRankPercentile:
    def test_empty_returns_none(self) -> None:
        """n=0 → 无定义返回 None（该指标不落库，而不是落 0 冒充延迟为 0）。"""
        assert ms.nearest_rank_percentile([], 0.5) is None
        assert ms.nearest_rank_percentile([], 0.95) is None

    def test_single_sample_any_percentile(self) -> None:
        """n=1 → 任何分位数都是那唯一样本。"""
        for p in (0.0, 0.5, 0.95, 0.99, 1.0):
            assert ms.nearest_rank_percentile([42], p) == 42

    def test_hundred_samples_ceiling_rank(self) -> None:
        """[1..100]：rank=ceil(p*n)，p95→第 95 个=95，p99→99，p50→50。"""
        values = list(range(1, 101))
        assert ms.nearest_rank_percentile(values, 0.50) == 50
        assert ms.nearest_rank_percentile(values, 0.95) == 95
        assert ms.nearest_rank_percentile(values, 0.99) == 99
        assert ms.nearest_rank_percentile(values, 1.0) == 100

    def test_non_integer_rank_rounds_up(self) -> None:
        """非整除位置向上取整：20 样本 p95 → rank=ceil(19)=19 → 第 19 个=19。"""
        values = list(range(1, 21))
        assert ms.nearest_rank_percentile(values, 0.95) == 19
        # n=2、p=0.95 → rank=ceil(1.9)=2 → 取较大值（docstring 明确登记例）
        assert ms.nearest_rank_percentile([100, 900], 0.95) == 900
        # n=10、p=0.95 → rank=ceil(9.5)=10 → 取最大值
        assert ms.nearest_rank_percentile(list(range(1, 11)), 0.95) == 10

    def test_result_is_a_real_sample_not_interpolated(self) -> None:
        """ceiling 口径保证结果一定是列表里真实存在的值（不像 percentile_cont 插值）。"""
        values = [10, 20, 900]
        result = ms.nearest_rank_percentile(values, 0.95)
        assert result in values
        assert result == 900

    def test_unordered_input_is_stable(self) -> None:
        """乱序输入内部先排序，结果与升序输入一致。"""
        shuffled = [900, 10, 500, 20, 100, 300, 700, 50, 400, 600]
        ordered = sorted(shuffled)
        for p in (0.5, 0.9, 0.95, 0.99):
            assert ms.nearest_rank_percentile(shuffled, p) == ms.nearest_rank_percentile(
                ordered, p
            )

    def test_duplicate_values(self) -> None:
        """重复值按出现次数占位：rank 落在哪次出现都取同一个值。"""
        assert ms.nearest_rank_percentile([5, 5, 5, 5], 0.95) == 5
        # n=5、p=0.95 → rank=ceil(4.75)=5 → 第 5 个=9
        assert ms.nearest_rank_percentile([1, 2, 3, 4, 9], 0.95) == 9
        # n=4、p=0.5 → rank=2 → 第 2 个（不是两个中间值的平均）
        assert ms.nearest_rank_percentile([1, 2, 8, 9], 0.5) == 2

    def test_out_of_range_percentile_is_clamped(self) -> None:
        """p<=0 钳到第 1 个、p>1 钳到最后 1 个，不越界 IndexError。"""
        values = [3, 1, 2]
        assert ms.nearest_rank_percentile(values, 0.0) == 1
        assert ms.nearest_rank_percentile(values, -0.5) == 1
        assert ms.nearest_rank_percentile(values, 1.5) == 3

    def test_does_not_mutate_input(self) -> None:
        """内部 sorted() 复制，不改调用方原列表（聚合侧还要复用同一批样本）。"""
        values = [5, 1, 3]
        ms.nearest_rank_percentile(values, 0.95)
        assert values == [5, 1, 3]


class TestSafeRatio:
    def test_zero_denominator_returns_none(self) -> None:
        """分母 0 → None：那一分钟一条日志都没写时，错误率显示 0% 是最坏结果。"""
        assert ms.safe_ratio(0, 0) is None
        assert ms.safe_ratio(3, 0) is None
        assert ms.safe_ratio(1, -1) is None

    def test_normal_and_rounding(self) -> None:
        assert ms.safe_ratio(1, 4) == 0.25
        # 1/3 收敛到 RATE_DECIMALS，避免断言依赖浮点噪声
        assert ms.safe_ratio(1, 3) == round(1 / 3, ms.RATE_DECIMALS)
        assert ms.safe_ratio(0, 5) == 0.0


# ====================================================================
# 二、previous_complete_bucket / floor_to_minute（时间桶边界）
# ====================================================================


class TestBucketBoundary:
    def test_returns_previous_full_minute(self) -> None:
        """给定时刻返回「上一个完整分钟」：终点是当前分钟起点。"""
        start, end = ms.previous_complete_bucket(datetime(2026, 10, 14, 3, 5, 47, tzinfo=UTC))
        assert start == datetime(2026, 10, 14, 3, 4, 0, tzinfo=UTC)
        assert end == datetime(2026, 10, 14, 3, 5, 0, tzinfo=UTC)

    def test_exactly_on_minute_boundary(self) -> None:
        """整分时刻仍取上一桶——当前分钟刚开启也还在写入。"""
        start, end = ms.previous_complete_bucket(datetime(2026, 10, 14, 3, 5, 0, tzinfo=UTC))
        assert (start, end) == (
            datetime(2026, 10, 14, 3, 4, tzinfo=UTC),
            datetime(2026, 10, 14, 3, 5, tzinfo=UTC),
        )

    def test_naive_input_treated_as_utc(self) -> None:
        """naive 按 UTC 理解（_as_utc），输出必须是 aware。"""
        start, end = ms.previous_complete_bucket(datetime(2026, 10, 14, 3, 5, 30))
        assert start.tzinfo is not None and end.tzinfo is not None
        assert start == datetime(2026, 10, 14, 3, 4, tzinfo=UTC)

    def test_other_timezone_converted_to_utc(self) -> None:
        """东八区 12:05 → UTC 04:05 → 上一桶 [04:04, 04:05)（先归一时区再取整）。"""
        start, end = ms.previous_complete_bucket(
            datetime(2026, 10, 14, 12, 5, 10, tzinfo=timezone_fixed(8))
        )
        assert start == datetime(2026, 10, 14, 4, 4, tzinfo=UTC)
        assert end == datetime(2026, 10, 14, 4, 5, tzinfo=UTC)

    def test_default_now_is_aware_and_aligned(self) -> None:
        """不给时刻则取当前 UTC，结果必为整分且跨度一分钟。"""
        start, end = ms.previous_complete_bucket()
        assert start.second == 0 and start.microsecond == 0
        assert end - start == timedelta(minutes=1)
        assert end <= datetime.now(UTC)

    def test_floor_to_minute(self) -> None:
        assert ms.floor_to_minute(datetime(2026, 10, 14, 3, 5, 59, 999999, tzinfo=UTC)) == (
            datetime(2026, 10, 14, 3, 5, tzinfo=UTC)
        )
        assert ms.floor_to_minute(datetime(2026, 10, 14, 3, 5)) == (
            datetime(2026, 10, 14, 3, 5, tzinfo=UTC)
        )


def timezone_fixed(offset_hours: int):
    """构造固定偏移时区（跨时区归一只需要一个非 UTC 的 tz）。"""
    from datetime import timezone

    return timezone(timedelta(hours=offset_hours))


# ====================================================================
# 三、compute_bucket_metrics 纯函数口径
# ====================================================================


def _row(**kw: Any) -> Any:
    """把 kwargs 包成命名元组，模拟 _fetch_llm_rows 的 Row 形态。"""
    from types import SimpleNamespace

    defaults = {
        "id": 1,
        "model": "qwen-max",
        "provider": "dashscope",
        "scenario": "chat",
        "status": "success",
        "latency_ms": 100,
        "first_token_ms": None,
        "total_tokens": 0,
        "cost_usd": None,
        "degradation_level": 0,
        "moderation_blocked": False,
    }
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class TestComputeBucketMetrics:
    def test_empty_bucket_only_counts_and_cost(self) -> None:
        """空桶：比率类与分位数都不落库（值 None 被 add() 过滤），只剩 0 值计数。"""
        rows = ms.compute_bucket_metrics([], 0, 0, 0)
        names = {r["metric_name"] for r in rows}
        for absent in (
            ms.METRIC_LLM_ERROR_RATE,
            ms.METRIC_DEGRADATION_RATE,
            ms.METRIC_MODERATION_BLOCK_RATE,
            ms.METRIC_RETRIEVAL_ZERO_RESULT_RATE,
            ms.METRIC_LLM_P50_LATENCY,
            ms.METRIC_LLM_P95_LATENCY,
            ms.METRIC_LLM_P99_LATENCY,
            ms.METRIC_LLM_FIRST_TOKEN_P95,
        ):
            assert absent not in names, f"{absent} 在无样本时不应落库"
        # 计数类是「确实发生了零次」，落 0
        by_name = {r["metric_name"]: r for r in rows}
        assert by_name[ms.METRIC_LLM_CALL_COUNT]["metric_value"] == 0.0
        assert by_name[ms.METRIC_TOKEN_COST]["metric_value"] == 0.0
        assert by_name[ms.METRIC_TASK_QUEUE_BACKLOG]["metric_value"] == 0.0

    def test_error_status_set(self) -> None:
        """error / timeout 计入失败率；degraded 不计（另有 degradation_rate）。"""
        rows = ms.compute_bucket_metrics(
            [
                _row(status="success"),
                _row(status="error"),
                _row(status="timeout"),
                _row(status="degraded"),
            ],
            0,
            0,
            0,
        )
        by_name = {r["metric_name"]: r for r in rows}
        assert by_name[ms.METRIC_LLM_ERROR_RATE]["metric_value"] == 0.5
        assert by_name[ms.METRIC_DEGRADATION_RATE]["metric_value"] == 0.25

    def test_degradation_union_of_status_and_level(self) -> None:
        """降级口径取并集：只标 status 或只填 level 都要算进来。"""
        rows = ms.compute_bucket_metrics(
            [
                _row(status="degraded", degradation_level=0),
                _row(status="success", degradation_level=2),
                _row(status="success", degradation_level=0),
                _row(status="success", degradation_level=None),
            ],
            0,
            0,
            0,
        )
        by_name = {r["metric_name"]: r for r in rows}
        assert by_name[ms.METRIC_DEGRADATION_RATE]["metric_value"] == 0.5

    def test_first_token_ignores_null_and_tagged_stream(self) -> None:
        """first_token_ms 为 NULL 的行不进样本，且 tags 标 scope=stream 说明口径。"""
        rows = ms.compute_bucket_metrics(
            [_row(first_token_ms=None), _row(first_token_ms=50), _row(first_token_ms=200)],
            0,
            0,
            0,
        )
        item = next(r for r in rows if r["metric_name"] == ms.METRIC_LLM_FIRST_TOKEN_P95)
        assert item["sample_count"] == 2
        assert item["tags"] == {"scope": "stream"}
        # n=2、p=0.95 → rank=2 → 取较大值
        assert item["metric_value"] == 200

    def test_token_usage_split_by_model(self) -> None:
        """token_usage 按 model 分行；model 为空归入 unknown。"""
        rows = ms.compute_bucket_metrics(
            [
                _row(model="qwen-max", total_tokens=100),
                _row(model="qwen-max", total_tokens=50),
                _row(model="gpt-4o", total_tokens=30),
                _row(model=None, total_tokens=7),
            ],
            0,
            0,
            0,
        )
        usage = {
            r["tags"]["model"]: r["metric_value"]
            for r in rows
            if r["metric_name"] == ms.METRIC_TOKEN_USAGE
        }
        assert usage == {"qwen-max": 150.0, "gpt-4o": 30.0, "unknown": 7.0}

    def test_cost_sums_decimals(self) -> None:
        """成本用 Decimal 累加（float 直接加会有二进制噪声），并按 6 位收敛。"""
        rows = ms.compute_bucket_metrics(
            [
                _row(cost_usd=Decimal("0.000001")),
                _row(cost_usd=Decimal("0.000002")),
                _row(cost_usd=None),
            ],
            0,
            0,
            0,
        )
        item = next(r for r in rows if r["metric_name"] == ms.METRIC_TOKEN_COST)
        assert item["metric_value"] == 0.000003
        # sample_count 只计非 NULL 成本的行
        assert item["sample_count"] == 2

    def test_retrieval_and_backlog_passthrough(self) -> None:
        rows = ms.compute_bucket_metrics([_row()], rag_total=4, rag_zero=1, backlog=7)
        by_name = {r["metric_name"]: r for r in rows}
        assert by_name[ms.METRIC_RETRIEVAL_ZERO_RESULT_RATE]["metric_value"] == 0.25
        assert by_name[ms.METRIC_RETRIEVAL_ZERO_RESULT_RATE]["sample_count"] == 4
        assert by_name[ms.METRIC_TASK_QUEUE_BACKLOG]["metric_value"] == 7.0


# ====================================================================
# 四、aggregate_minute_bucket：真实明细 → ai_metrics
# ====================================================================


async def _seed_bucket(db: AsyncSession) -> dict[str, Any]:
    """造一个信息量足够的桶：三种状态、三个模型、审核拦截、首 token、检索与积压。"""
    uid = uuid.uuid4()
    # 5 条 LLM 调用，延迟升序为 [100,200,300,400,1000]（分位数推导见下方断言注释）
    add_llm_call(
        db, scenario="chat", provider="dashscope", model="qwen-max", status="success",
        latency_ms=100, first_token_ms=40, prompt_tokens=100, completion_tokens=20,
        cost_usd=Decimal("0.001000"), created_at=BUCKET_START + timedelta(seconds=5),
    )
    add_llm_call(
        db, scenario="triage", provider="openai", model="gpt-4o", status="success",
        latency_ms=200, first_token_ms=60, prompt_tokens=50, completion_tokens=10,
        cost_usd=Decimal("0.002000"), user_id=uid,
        created_at=BUCKET_START + timedelta(seconds=20),
    )
    add_llm_call(
        db, scenario="chat", provider="dashscope", model="qwen-max", status="error",
        latency_ms=300, prompt_tokens=0, completion_tokens=0, cost_usd=None,
        created_at=BUCKET_START + timedelta(seconds=30),
    )
    add_llm_call(
        db, scenario="summary", provider="dashscope", model="qwen-plus",
        requested_model="qwen-max", status="degraded", degradation_level=1,
        latency_ms=400, prompt_tokens=30, completion_tokens=5,
        cost_usd=Decimal("0.000500"),
        created_at=BUCKET_START + timedelta(seconds=40),
    )
    add_llm_call(
        db, scenario="moderation", provider="dashscope", model="qwen-max", status="success",
        latency_ms=1000, first_token_ms=900, prompt_tokens=10, completion_tokens=0,
        cost_usd=Decimal("0.000100"), moderation_blocked=True,
        created_at=BUCKET_START + timedelta(seconds=50),
    )
    # 窗口外的两条：一条早于 start、一条正好落在 end（左闭右开 → 不属于本桶）
    add_llm_call(
        db, scenario="chat", model="qwen-max", latency_ms=9999,
        prompt_tokens=9999, completion_tokens=9999,
        created_at=BUCKET_START - timedelta(seconds=1),
    )
    add_llm_call(
        db, scenario="chat", model="qwen-max", latency_ms=8888,
        prompt_tokens=8888, completion_tokens=8888,
        created_at=BUCKET_END,
    )
    # 检索：3 条，其中 1 条零结果
    add_rag_log(db, is_zero_result=False, created_at=BUCKET_START + timedelta(seconds=8))
    add_rag_log(db, is_zero_result=True, created_at=BUCKET_START + timedelta(seconds=18))
    add_rag_log(db, is_zero_result=False, created_at=BUCKET_START + timedelta(seconds=28))
    # 积压：2 pending + 1 running + 1 completed（completed 不计入 backlog）
    for status in ("pending", "pending", "running", "completed"):
        db.add(
            AITask(
                id=uuid.uuid4(),
                task_type="knowledge_rebuild",
                status=status,
                progress=0,
                retry_count=0,
                created_at=BUCKET_START,
                updated_at=BUCKET_START,
            )
        )
    await db.commit()
    return {"user_id": uid}


async def test_aggregate_minute_bucket_values(db: AsyncSession, engine) -> None:
    """桶内 12 项指标逐一对齐样本（含窗口排除的两条慢调用）。"""
    await _seed_bucket(db)
    outcome = await ms.aggregate_minute_bucket(db, BUCKET_START, BUCKET_END)

    assert outcome["bucket_start"] == BUCKET_START.isoformat()
    assert outcome["written"] > 0
    assert outcome["llm_call_rows"] == 5, "窗口外两条必须被排除（左闭右开）"
    assert outcome["retrieval_rows"] == 3
    assert outcome["queue_backlog"] == 3, "backlog = pending(2)+running(1)，不含 completed"

    grouped = await metric_map(db, BUCKET_START)

    assert single(grouped, ms.METRIC_LLM_CALL_COUNT)["value"] == 5.0
    assert single(grouped, ms.METRIC_LLM_CALL_COUNT)["sample_count"] == 5
    # 1 error / 5 calls
    assert single(grouped, ms.METRIC_LLM_ERROR_RATE)["value"] == 0.2
    # 延迟升序 [100,200,300,400,1000]：p50 rank=ceil(2.5)=3 → 300
    assert single(grouped, ms.METRIC_LLM_P50_LATENCY)["value"] == 300.0
    # p95 rank=ceil(4.75)=5 → 1000；p99 rank=ceil(4.95)=5 → 1000
    assert single(grouped, ms.METRIC_LLM_P95_LATENCY)["value"] == 1000.0
    assert single(grouped, ms.METRIC_LLM_P99_LATENCY)["value"] == 1000.0
    # 首 token 样本 [40,60,900]：rank=ceil(2.85)=3 → 900
    assert single(grouped, ms.METRIC_LLM_FIRST_TOKEN_P95)["value"] == 900.0
    assert single(grouped, ms.METRIC_LLM_FIRST_TOKEN_P95)["tags"] == {"scope": "stream"}
    # 成本合计 0.001+0.002+0.0005+0.0001 = 0.0036（error 行 cost 为 NULL 不计样本）
    assert single(grouped, ms.METRIC_TOKEN_COST)["value"] == 0.0036
    assert single(grouped, ms.METRIC_TOKEN_COST)["sample_count"] == 4
    # 降级：status=degraded 或 level>0 → 1 条
    assert single(grouped, ms.METRIC_DEGRADATION_RATE)["value"] == 0.2
    # 审核拦截 1/5
    assert single(grouped, ms.METRIC_MODERATION_BLOCK_RATE)["value"] == 0.2
    # 检索零结果 1/3（RATE_DECIMALS 收敛）
    assert single(grouped, ms.METRIC_RETRIEVAL_ZERO_RESULT_RATE)["value"] == round(
        1 / 3, ms.RATE_DECIMALS
    )
    assert single(grouped, ms.METRIC_TASK_QUEUE_BACKLOG)["value"] == 3.0

    # token_usage 按 model 分行：qwen-max=100+20+0+10+0=130、gpt-4o=60、qwen-plus=35
    usage = {item["tags"]["model"]: item["value"] for item in grouped[ms.METRIC_TOKEN_USAGE]}
    assert usage == {"qwen-max": 130.0, "gpt-4o": 60.0, "qwen-plus": 35.0}


async def test_aggregate_rejects_inverted_window(db: AsyncSession) -> None:
    """bucket_end <= bucket_start 直接 ValueError（聚合方向写反属于调用方错误）。"""
    with pytest.raises(ValueError):
        await ms.aggregate_minute_bucket(db, BUCKET_END, BUCKET_START)
    same = BUCKET_START
    with pytest.raises(ValueError):
        await ms.aggregate_minute_bucket(db, same, same)


async def test_aggregate_accepts_naive_bucket_times(db: AsyncSession) -> None:
    """naive 桶时间按 UTC 归一，落库的 bucket_time 与 aware 版一致。"""
    naive_start = BUCKET_START.replace(tzinfo=None)
    naive_end = BUCKET_END.replace(tzinfo=None)
    outcome = await ms.aggregate_minute_bucket(db, naive_start, naive_end)
    assert outcome["bucket_start"] == BUCKET_START.isoformat()


async def test_aggregate_is_idempotent(db: AsyncSession, engine) -> None:
    """同桶跑两次：行数不翻倍、值不累计（删桶重插而非 ON CONFLICT）。"""
    await _seed_bucket(db)
    first = await ms.aggregate_minute_bucket(db, BUCKET_START, BUCKET_END)
    count_after_first = count_rows(engine, "ai_metrics")
    first_count = single(await metric_map(db, BUCKET_START), ms.METRIC_LLM_CALL_COUNT)["value"]

    second = await ms.aggregate_minute_bucket(db, BUCKET_START, BUCKET_END)
    count_after_second = count_rows(engine, "ai_metrics")
    second_count = single(await metric_map(db, BUCKET_START), ms.METRIC_LLM_CALL_COUNT)["value"]

    assert count_after_second == count_after_first, "重复聚合不得新增行数"
    assert second_count == first_count == 5.0, "重复聚合不得累计数值"
    assert second["written"] == first["written"]
    # 第二次执行会把第一次的行删掉重建
    assert second["deleted"] == first["written"]


async def test_aggregate_neighbour_buckets_do_not_overlap(db: AsyncSession, engine) -> None:
    """相邻两桶恰好衔接：边界时刻只归属后一桶，两桶计数相加等于总行数。"""
    prev_start = BUCKET_START - timedelta(minutes=1)
    db.add(
        LLMCallLog(
            scenario="chat", provider="dashscope", model="qwen-max",
            prompt_tokens=1, completion_tokens=1,
            latency_ms=10, status="success", created_at=prev_start,
        )
    )
    db.add(
        LLMCallLog(
            scenario="chat", provider="dashscope", model="qwen-max",
            prompt_tokens=1, completion_tokens=1,
            latency_ms=10, status="success", created_at=BUCKET_START,
        )
    )
    await db.commit()

    a = await ms.aggregate_minute_bucket(db, prev_start, BUCKET_START)
    b = await ms.aggregate_minute_bucket(db, BUCKET_START, BUCKET_END)
    assert a["llm_call_rows"] == 1
    assert b["llm_call_rows"] == 1, "12:05:00.000000 这条只属后一桶"

    grouped_prev = await metric_map(db, prev_start)
    grouped_cur = await metric_map(db, BUCKET_START)
    assert single(grouped_prev, ms.METRIC_LLM_CALL_COUNT)["value"] == 1.0
    assert single(grouped_cur, ms.METRIC_LLM_CALL_COUNT)["value"] == 1.0


async def test_aggregate_empty_bucket_skips_ratio_metrics(db: AsyncSession) -> None:
    """空桶：比率类指标整行不落库（而不是落 0 冒充「零错误率」）。"""
    await ms.aggregate_minute_bucket(db, BUCKET_START, BUCKET_END)
    grouped = await metric_map(db, BUCKET_START)
    for name in (
        ms.METRIC_LLM_ERROR_RATE,
        ms.METRIC_LLM_P95_LATENCY,
        ms.METRIC_LLM_FIRST_TOKEN_P95,
        ms.METRIC_DEGRADATION_RATE,
        ms.METRIC_MODERATION_BLOCK_RATE,
        ms.METRIC_RETRIEVAL_ZERO_RESULT_RATE,
    ):
        assert name not in grouped, f"{name} 在无样本时不应落库"
    assert ms.METRIC_LLM_CALL_COUNT in grouped


async def test_aggregate_tags_are_normalized_dict(db: AsyncSession) -> None:
    """tags 经 ORM 往返后仍是 dict，且维度键稳定（供看板下钻与后续加唯一约束）。"""
    await _seed_bucket(db)
    await ms.aggregate_minute_bucket(db, BUCKET_START, BUCKET_END)
    grouped = await metric_map(db, BUCKET_START)
    assert all(isinstance(item["tags"], dict) for items in grouped.values() for item in items)
    assert len(grouped[ms.METRIC_TOKEN_USAGE]) == 3


def test_normalize_tags_key_order_independent() -> None:
    """同一维度组合不同插入顺序必须归一化成同一串，否则看板出现两条重复曲线。"""
    a = ms._normalize_tags({"model": "qwen-max", "provider": "dashscope"})
    b = ms._normalize_tags({"provider": "dashscope", "model": "qwen-max"})
    assert a == b
    assert ms._normalize_tags(None) == ""
    assert ms._normalize_tags({}) == ""
    # 中文维度值不被转义成 \uXXXX，便于人工核对
    assert "急诊" in ms._normalize_tags({"分支": "急诊"})


# ====================================================================
# 五、run_aggregate：DB 异常转结构化返回值（旁路观测不搞崩宿主）
# ====================================================================


async def test_run_aggregate_swallows_db_errors(monkeypatch) -> None:
    """自开会话失败时不外抛，返回 ok=False + error + written=0。"""
    import app.services.metrics_service as module

    class _BoomFactory:
        def __call__(self):
            raise RuntimeError("db down")

    monkeypatch.setattr(module, "async_session_factory", _BoomFactory())

    result = await module.run_aggregate(BUCKET_START + timedelta(seconds=30))
    assert result["ok"] is False
    assert "RuntimeError" in result["error"]
    assert result["written"] == 0
    # 03:05:30 时「03:05 这一桶」还在写入，实现刻意只聚**上一个完整桶**
    # （03:04:00 ~ 03:05:00），否则看板会看到一根正在爬升的假曲线。
    assert result["bucket_start"] == (BUCKET_START - timedelta(minutes=1)).isoformat()
    assert result["bucket_end"] == BUCKET_START.isoformat()


async def test_run_aggregate_wraps_inner_failure_as_error_string(monkeypatch) -> None:
    """内层 aggregate_minute_bucket 抛异常也被收口，error 文案带类型名。

    只在 ``run_aggregate`` 这一层测容错：它自开事务并在最外层 try/except 收口，
    因此不需要真数据库——替换会话工厂与内层聚合即可覆盖「异常不外抛」这条契约。
    """
    import app.services.metrics_service as module

    async def _boom(*a, **kw):
        raise ValueError("bad window")

    # run_aggregate 用的是 ``async with async_session_factory()``，
    # 桩必须返回**异步**上下文管理器；同步 @contextmanager 包出来的对象
    # 不支持异步协议，会先抛 TypeError 把被测的 ValueError 盖掉。
    @asynccontextmanager
    async def _noop_session_cm():
        yield None

    def _factory(*a, **kw):
        return _noop_session_cm()

    monkeypatch.setattr(module, "aggregate_minute_bucket", _boom)
    monkeypatch.setattr(module, "async_session_factory", _factory)

    result = await module.run_aggregate(BUCKET_START + timedelta(seconds=30))
    assert result["ok"] is False
    assert "ValueError" in result["error"]
    assert result["written"] == 0
    assert result["bucket_start"] == (BUCKET_START - timedelta(minutes=1)).isoformat()


# ====================================================================
# 六、reap_stale_tasks：僵尸任务清理
# ====================================================================


def _sync_session(engine):  # noqa: ANN001
    """构造连到测试文件库的同步 Session（Celery worker 侧同款会话形态）。"""
    from sqlalchemy.orm import Session

    return Session(bind=engine)


@contextmanager
def _sync_db_context(engine):  # noqa: ANN001
    """把同步 Session 包成 get_sync_db 那样的上下文管理器。

    ``get_sync_db`` 是 ``@contextmanager`` 包装的生成器函数，调用形态是
    ``with get_sync_db() as session``；替换时必须保持同样的「无参可调用 +
    返回 CM」签名，否则被测任务体第一步就 TypeError。
    """
    session = _sync_session(engine)
    try:
        yield session
    finally:
        session.close()


def _patched_reap(engine, monkeypatch):  # noqa: ANN001
    """返回打好同步会话桩的 ``_reap_sync``。

    ``_reap_sync`` 内部用 ``get_sync_db``（绑全局 PG 引擎），测试环境没有 PG，
    必须替换成连到同一文件库的同步 Session——与
    ``test_reap_task_body_returns_structured_dict`` 同一手法，抽出来共用。
    """
    import app.tasks.monitoring_tasks as mt

    monkeypatch.setattr(mt, "get_sync_db", lambda: _sync_db_context(engine))
    return mt._reap_sync


def test_reap_marks_stale_running_and_pending(engine, monkeypatch) -> None:
    """超时 running（按 started_at）与 pending（按 created_at）都被标 failed 并留痕。"""
    old = utcnow() - timedelta(minutes=90)
    stale_running = insert_task(engine, status="running", created_at=old, started_at=old)
    stale_pending = insert_task(engine, status="pending", created_at=old, started_at=None)

    from app.tasks.monitoring_tasks import STALE_TASK_MESSAGE

    _reap_sync = _patched_reap(engine, monkeypatch)
    cutoff = utcnow() - timedelta(minutes=60)
    result = _reap_sync(cutoff, 60)

    assert result["ok"] is True
    assert result["reaped"] == 2
    assert result["reaped_running"] == 1
    assert result["reaped_pending"] == 1

    row = task_row(engine, stale_running)
    assert row["status"] == AITaskStatus.FAILED.value
    assert STALE_TASK_MESSAGE[:8] in (row["error_message"] or "")
    assert "被僵尸清理任务标记" in row["error_message"]
    assert row["completed_at"] is not None
    # duration_ms 由 started_at 推出（僵尸实际占用时长）
    assert row["duration_ms"] is not None and row["duration_ms"] >= 89 * 60_000
    # 清理不代表系统自动重试过，retry_count 不能被加上
    assert row["retry_count"] == 0

    pend = task_row(engine, stale_pending)
    assert pend["status"] == AITaskStatus.FAILED.value
    assert pend["completed_at"] is not None


def test_reap_leaves_fresh_and_terminal_tasks(engine, monkeypatch) -> None:
    """未超时 running、刚提交的 pending、以及终态任务都不动。"""
    now = utcnow()
    fresh_running = insert_task(
        engine, status="running", created_at=now - timedelta(minutes=5),
        started_at=now - timedelta(minutes=5),
    )
    fresh_pending = insert_task(engine, status="pending", created_at=now - timedelta(minutes=5))
    done = insert_task(
        engine, status="completed", created_at=now - timedelta(days=2),
        started_at=now - timedelta(days=2),
    )
    failed = insert_task(
        engine, status="failed", created_at=now - timedelta(days=2),
        started_at=now - timedelta(days=2),
    )
    retrying = insert_task(
        engine, status="retrying", created_at=now - timedelta(days=2),
        started_at=now - timedelta(days=2),
    )

    _reap_sync = _patched_reap(engine, monkeypatch)
    result = _reap_sync(now - timedelta(minutes=60), 60)
    assert result["reaped"] == 0

    for tid, expect in (
        (fresh_running, "running"),
        (fresh_pending, "pending"),
        (done, "completed"),
        (failed, "failed"),
        (retrying, "retrying"),
    ):
        assert task_row(engine, tid)["status"] == expect, f"{expect} 不该被动"
    assert task_row(engine, retrying)["error_message"] is None


def test_reap_threshold_parameter_changes_the_cut(engine, monkeypatch) -> None:
    """阈值参数生效：同一批任务，60 分钟不清理、10 分钟清理。"""
    borderline = insert_task(
        engine, status="running",
        created_at=utcnow() - timedelta(minutes=30),
        started_at=utcnow() - timedelta(minutes=30),
    )
    _reap_sync = _patched_reap(engine, monkeypatch)

    assert _reap_sync(utcnow() - timedelta(minutes=60), 60)["reaped"] == 0
    assert task_row(engine, borderline)["status"] == "running"

    assert _reap_sync(utcnow() - timedelta(minutes=10), 10)["reaped"] == 1
    assert task_row(engine, borderline)["status"] == "failed"


def test_reap_running_with_null_started_at_is_left_alone(engine, monkeypatch) -> None:
    """疑似漏判面：running 但 started_at IS NULL 的脏行两个分支都不命中，永远占位。

    实现里 running 分支要求 started_at 非 NULL，pending 分支只认 status=pending，
    于是这种行既不报错也不被清理。按现状断言「不动」，并在报告中标为待确认。
    """
    ghost = insert_task(
        engine, status="running", created_at=utcnow() - timedelta(days=3), started_at=None
    )
    _reap_sync = _patched_reap(engine, monkeypatch)

    result = _reap_sync(utcnow() - timedelta(minutes=60), 60)
    assert result["reaped"] == 0
    assert task_row(engine, ghost)["status"] == "running"


def test_reap_pending_uses_created_at_not_started_at(engine, monkeypatch) -> None:
    """pending 的 started_at 恒 NULL，必须按 created_at 判僵尸（否则前端一直转圈）。"""
    stuck = insert_task(
        engine, status="pending", created_at=utcnow() - timedelta(minutes=120), started_at=None
    )
    _reap_sync = _patched_reap(engine, monkeypatch)

    assert _reap_sync(utcnow() - timedelta(minutes=60), 60)["reaped_pending"] == 1
    assert task_row(engine, stuck)["status"] == "failed"


def test_normalize_threshold_falls_back_on_bad_input() -> None:
    """非法/非正阈值一律回落默认值：阈值是防误杀的安全阀。"""
    from app.tasks.monitoring_tasks import (
        STALE_TASK_THRESHOLD_MINUTES,
        _normalize_threshold,
    )

    assert _normalize_threshold(None) == STALE_TASK_THRESHOLD_MINUTES
    assert _normalize_threshold(0) == STALE_TASK_THRESHOLD_MINUTES
    assert _normalize_threshold(-5) == STALE_TASK_THRESHOLD_MINUTES
    assert _normalize_threshold("abc") == STALE_TASK_THRESHOLD_MINUTES
    assert _normalize_threshold(float("nan")) == STALE_TASK_THRESHOLD_MINUTES
    # 合法值（含 Beat 可能给的 float / 数字字符串）按原值取整
    assert _normalize_threshold("30") == 30
    assert _normalize_threshold(15.7) == 15
    assert _normalize_threshold(1) == 1


def test_reap_task_body_returns_structured_dict(engine, monkeypatch) -> None:
    """任务体经 _reap_sync 正常返回结构化摘要（含 cutoff 与截断标记）。

    ``monitoring_tasks`` 是 ``from app.core.database import get_sync_db`` 进来的
    模块级名字，因此补丁必须打在 monitoring_tasks 自身上，打 core.database 无效。
    用 ``.apply().get()`` 本地跑任务体；reap 的同步壳内部走 ``_reap_sync``，
    不新建事件循环，因此在 pytest-asyncio 用例里也能安全执行，不起 worker、不连 broker。
    """
    import app.tasks.monitoring_tasks as mt

    monkeypatch.setattr(mt, "get_sync_db", lambda: _sync_db_context(engine))

    insert_task(
        engine, status="running",
        created_at=utcnow() - timedelta(minutes=200),
        started_at=utcnow() - timedelta(minutes=200),
    )
    result = mt.reap_stale_tasks.apply(args=(60,)).get()
    assert result["ok"] is True
    assert result["threshold_minutes"] == 60
    assert result["reaped"] == 1
    assert result["truncated"] is False
    assert isinstance(result["task_ids"], list) and len(result["task_ids"]) == 1


def test_reap_task_body_swallows_errors(monkeypatch) -> None:
    """底层会话不可用时任务体返回 ok=False 而不是抛异常搞崩 worker。"""
    import app.tasks.monitoring_tasks as mt

    def _boom():
        raise RuntimeError("no sync db")

    monkeypatch.setattr(mt, "get_sync_db", _boom)

    result = mt.reap_stale_tasks.apply(args=(60,)).get()
    assert result["ok"] is False
    assert "RuntimeError" in result["error"]
    assert result["reaped"] == 0


def test_aggregate_task_body_falls_back_when_loop_exists(monkeypatch) -> None:
    """聚合任务体在 ``run_async`` 抛错时不外抛，收口成结构化返回值。

    ``run_async`` 内部是 ``asyncio.run``，在 pytest-asyncio 的活循环里调用必然
    RuntimeError。这正是「同步壳 + 异步芯」的已知边界：线上由 prefork 独占进程
    所以安全，测试侧只能断言**异常被收口成结构化返回值**这条契约。

    桩必须是**同步**函数：任务体里是 ``run_async(ms.run_aggregate())``，
    把协程传进来后由桩负责关闭。之前桩写成了 ``async def``，monkeypatch 后
    ``run_async(...)`` 只是造出一个永不执行的协程就返回，任务体拿着协程对象
    去 ``.get()`` 直接 AttributeError —— 收口分支根本没被走到，测的是假东西。
    """
    import app.tasks.monitoring_tasks as mt

    def _fake_run_async(coro):
        coro.close()  # 不让「coroutine was never awaited」警告污染输出
        raise RuntimeError("asyncio.run() cannot be called from a running event loop")

    monkeypatch.setattr(mt, "run_async", _fake_run_async)
    result = mt.aggregate_ai_metrics.apply().get()
    assert result["ok"] is False
    assert "RuntimeError" in result["error"]
    assert result["written"] == 0


def test_monitoring_tasks_registered() -> None:
    """两个任务必须挂 default 队列并完成注册（Beat 配置引用全限定名）。"""
    import sys

    from app.tasks import monitoring_tasks  # noqa: F401  - 触发注册

    mod = sys.modules["app.tasks.monitoring_tasks"]
    tasks = dict(mod.celery_app.tasks)
    assert "app.tasks.monitoring_tasks.aggregate_ai_metrics" in tasks
    assert "app.tasks.monitoring_tasks.reap_stale_tasks" in tasks
    assert tasks["app.tasks.monitoring_tasks.aggregate_ai_metrics"].queue == "default"
    assert tasks["app.tasks.monitoring_tasks.reap_stale_tasks"].queue == "default"


# ====================================================================
# 七、replay_trace：全链路回放
# ====================================================================


async def _seed_trace(db: AsyncSession, trace_id: str, *, user_id: uuid.UUID | None = None) -> dict[str, Any]:
    """一条链路：2 次 LLM 调用 + 1 次检索 + 1 条工作流 + 1 条分诊产出。"""
    t0 = BUCKET_START
    # 故意倒序 add：第二次调用 created_at 更早的行由排序负责纠正
    add_llm_call(
        db, scenario="summary", provider="dashscope", model="qwen-plus",
        requested_model="qwen-max", status="degraded", degradation_level=2,
        latency_ms=800, prompt_tokens=300, completion_tokens=70,
        cost_usd=Decimal("0.003000"), trace_id=trace_id, user_id=user_id,
        workflow_node="generate_advice",
        created_at=t0 + timedelta(seconds=30),
    )
    add_llm_call(
        db, scenario="triage", provider="openai", model="gpt-4o", status="success",
        latency_ms=200, prompt_tokens=120, completion_tokens=30,
        cost_usd=Decimal("0.001000"), trace_id=trace_id, user_id=user_id,
        workflow_node="extract_symptoms",
        created_at=t0 + timedelta(seconds=10),
    )
    add_rag_log(
        db, trace_id=trace_id, query_text="胸口疼还冒汗",
        rewritten_query="胸痛 心悸 鉴别诊断", vector_hit_count=7, keyword_hit_count=4,
        both_hit_count=3, merged_count=8, final_count=5, is_zero_result=False,
        candidates=[{"chunk_id": "c1", "rrf_score": 0.03}],
        created_at=t0 + timedelta(seconds=5),
    )
    add_workflow_trace(
        db, trace_id=trace_id, workflow_name="triage", status="completed",
        steps=[{"node": "extract_symptoms", "latency_ms": 200},
               {"node": "generate_advice", "latency_ms": 800}],
        node_count=4, executed_node_count=4, branch_taken="emergency",
        final_output={"urgency": "stat"},
        created_at=t0 + timedelta(seconds=2),
    )
    db.add(
        TriageResult(
            id=uuid.uuid4(),
            patient_id=uuid.uuid4(),
            user_id=user_id,
            trace_id=trace_id,
            chief_complaint="胸口疼还冒汗",
            extracted_symptoms=["胸痛", "出汗"],
            recommended_department_name="急诊科",
            urgency="stat",
            is_emergency=True,
            matched_emergency_keywords=["胸痛"],
            advice="立即前往急诊",
            confidence=0.91,
            decision_source="ai",
            llm_model="qwen-plus",
            created_at=t0 + timedelta(seconds=35),
            updated_at=t0 + timedelta(seconds=35),
        )
    )
    await db.commit()
    return {"t0": t0}


async def test_replay_trace_sections_and_order(db: AsyncSession) -> None:
    """三段齐全 + 分诊段，LLM 调用按 created_at 升序，counts 与实际一致。"""
    tid = str(uuid.uuid4())
    await _seed_trace(db, tid)

    data = await mon.replay_trace(db, tid)
    assert data["trace_id"] == tid
    assert data["counts"] == {
        "llm_calls": 2,
        "retrievals": 1,
        "workflows": 1,
        "triage_results": 1,
    }
    order = [item["workflow_node"] for item in data["llm_calls"]]
    assert order == ["extract_symptoms", "generate_advice"], "回放必须按时间升序"
    moments = [item["created_at"] for item in data["llm_calls"]]
    assert moments == sorted(moments)


async def test_replay_trace_summary_fields(db: AsyncSession) -> None:
    """汇总：总 tokens / 总成本 / 错误数 / 降级数 / 最大延迟 / P95。"""
    tid = str(uuid.uuid4())
    await _seed_trace(db, tid)

    summary = (await mon.replay_trace(db, tid))["summary"]
    assert summary["prompt_tokens"] == 420  # 120 + 300
    assert summary["completion_tokens"] == 100  # 30 + 70
    assert summary["total_tokens"] == 520
    assert summary["cost_usd"] == 0.004
    assert summary["call_count"] == 2
    assert summary["error_count"] == 0
    # degraded 或 level>0 的并集：第二条同时满足两个条件，只算一次
    assert summary["degraded_count"] == 1
    assert summary["max_latency_ms"] == 800
    # n=2、p=0.95 → rank=2 → 取较大值 800
    assert summary["p95_latency_ms"] == 800
    assert summary["providers"] == ["dashscope", "openai"]
    assert summary["models"] == ["gpt-4o", "qwen-plus"]
    assert summary["scenarios"] == ["summary", "triage"]
    assert summary["started_at"] is not None and summary["ended_at"] is not None


async def test_replay_trace_timeline_merged_sorted(db: AsyncSession) -> None:
    """时间线合并四类来源并按时刻升序，条目 kind 齐全。"""
    tid = str(uuid.uuid4())
    await _seed_trace(db, tid)

    data = await mon.replay_trace(db, tid)
    timeline = data["timeline"]
    assert len(timeline) == 5  # 2 llm + 1 retrieval + 1 workflow + 1 triage
    kinds = {e["kind"] for e in timeline}
    assert kinds == {"llm_call", "retrieval", "workflow", "triage_result"}
    # workflow(+2s) → retrieval(+5s) → llm(+10s) → llm(+30s) → triage(+35s)
    assert [e["kind"] for e in timeline] == [
        "workflow",
        "retrieval",
        "llm_call",
        "llm_call",
        "triage_result",
    ]
    assert all(e["summary"] for e in timeline)


async def test_replay_trace_workflow_fragment_content(db: AsyncSession) -> None:
    """工作流片段带 steps 与 branch_taken（急诊分支要能区分统计）。"""
    tid = str(uuid.uuid4())
    await _seed_trace(db, tid)

    data = await mon.replay_trace(db, tid)
    wf = data["workflows"][0]
    assert wf["branch_taken"] == "emergency"
    assert len(wf["steps"]) == 2
    assert wf["steps"][0]["node"] == "extract_symptoms"
    assert wf["final_output"] == {"urgency": "stat"}
    retr = data["retrievals"][0]
    assert retr["rewritten_query"] == "胸痛 心悸 鉴别诊断"
    assert retr["candidates"] and retr["candidates"][0]["chunk_id"] == "c1"
    assert retr["is_zero_result"] is False
    tri = data["triage_results"][0]
    assert tri["urgency"] == "stat" and tri["is_emergency"] is True
    assert tri["extracted_symptoms"] == ["胸痛", "出汗"]


async def test_replay_trace_missing_raises_404(db: AsyncSession) -> None:
    """完全不存在的 trace_id → ResourceNotFoundError（参数错误，不是空结果）。"""
    with pytest.raises(ResourceNotFoundError):
        await mon.replay_trace(db, str(uuid.uuid4()))


@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
async def test_replay_trace_blank_id_rejected(db: AsyncSession, bad: str) -> None:
    """空白 trace_id 同样按 404 处理，不去做无意义的 LIKE 匹配。"""
    with pytest.raises(ResourceNotFoundError):
        await mon.replay_trace(db, bad)


async def test_replay_trace_partial_sections_are_not_errors(db: AsyncSession) -> None:
    """缺段不是错误：只有 LLM 调用的链路，其余三段返回空数组且 counts 为 0。"""
    tid = str(uuid.uuid4())
    add_llm_call(db, trace_id=tid, scenario="chat", created_at=BUCKET_START)
    await db.commit()

    data = await mon.replay_trace(db, tid)
    assert data["counts"]["llm_calls"] == 1
    assert data["counts"]["retrievals"] == 0
    assert data["counts"]["workflows"] == 0
    assert data["counts"]["triage_results"] == 0
    assert data["retrievals"] == [] and data["workflows"] == []
    assert data["timeline"][0]["kind"] == "llm_call"


async def test_replay_trace_error_count_uses_error_statuses(db: AsyncSession) -> None:
    """error / timeout 计入 error_count，degraded 不计。"""
    tid = str(uuid.uuid4())
    add_llm_call(db, trace_id=tid, status="error", latency_ms=50,
                 created_at=BUCKET_START + timedelta(seconds=1))
    add_llm_call(db, trace_id=tid, status="timeout", latency_ms=60,
                 created_at=BUCKET_START + timedelta(seconds=2))
    add_llm_call(db, trace_id=tid, status="degraded", degradation_level=3, latency_ms=70,
                 created_at=BUCKET_START + timedelta(seconds=3))
    await db.commit()

    data = await mon.replay_trace(db, tid)
    assert data["summary"]["error_count"] == 2
    # degraded_count 数的是「发生过降级的调用条数」，不是 degradation_level 的值：
    # 三条里只有第三条是降级（level=3 表示降到确定性规则那一层），因此是 1。
    assert data["summary"]["degraded_count"] == 1


# ====================================================================
# 八、list_llm_calls：过滤与分页
# ====================================================================


async def _seed_calls(db: AsyncSession) -> None:
    base = BUCKET_START
    samples = [
        ("chat", "dashscope", "qwen-max", "success", base + timedelta(seconds=1)),
        ("chat", "dashscope", "qwen-max", "error", base + timedelta(seconds=2)),
        ("triage", "openai", "gpt-4o", "success", base + timedelta(seconds=3)),
        ("summary", "dashscope", "qwen-plus", "degraded", base + timedelta(seconds=4)),
        ("moderation", "dashscope", "qwen-max", "success", base + timedelta(seconds=5)),
    ]
    for scenario, provider, model, status, moment in samples:
        add_llm_call(
            db, scenario=scenario, provider=provider, model=model, status=status,
            latency_ms=100, prompt_tokens=10, completion_tokens=5,
            cost_usd=Decimal("0.000100"), created_at=moment,
        )
    await db.commit()


async def test_list_llm_calls_no_filter_returns_desc_order(db: AsyncSession) -> None:
    await _seed_calls(db)
    rows, total = await mon.list_llm_calls(db)
    assert total == 5
    assert len(rows) == 5
    moments = [r["created_at"] for r in rows]
    assert moments == sorted(moments, reverse=True), "默认最新在前"
    assert rows[0]["scenario"] == "moderation"


async def test_list_llm_calls_filters(db: AsyncSession) -> None:
    await _seed_calls(db)
    chat, n_chat = await mon.list_llm_calls(db, scenario="chat")
    assert n_chat == 2 and {r["scenario"] for r in chat} == {"chat"}

    errs, n_err = await mon.list_llm_calls(db, status_filter="error")
    assert n_err == 1 and errs[0]["status"] == "error"

    openai_rows, n_openai = await mon.list_llm_calls(db, provider="openai")
    assert n_openai == 1 and openai_rows[0]["model"] == "gpt-4o"

    modelled, n_modelled = await mon.list_llm_calls(db, model="qwen-max")
    assert n_modelled == 3

    combo, n_combo = await mon.list_llm_calls(
        db, scenario="chat", provider="dashscope", status_filter="success"
    )
    assert n_combo == 1 and combo[0]["status"] == "success"

    none_rows, n_none = await mon.list_llm_calls(db, scenario="not_exists")
    assert none_rows == [] and n_none == 0


async def test_list_llm_calls_pagination(db: AsyncSession) -> None:
    """page/page_size 翻页不重不漏；total 始终是过滤后的总数。"""
    await _seed_calls(db)
    p1, total = await mon.list_llm_calls(db, page=1, page_size=2)
    p2, _ = await mon.list_llm_calls(db, page=2, page_size=2)
    p3, _ = await mon.list_llm_calls(db, page=3, page_size=2)
    assert total == 5
    assert [len(x) for x in (p1, p2, p3)] == [2, 2, 1]
    ids = [r["id"] for r in (*p1, *p2, *p3)]
    assert len(ids) == len(set(ids)) == 5


async def test_list_llm_calls_page_size_clamped(db: AsyncSession) -> None:
    """page_size 超上限按 MAX_PAGE_SIZE 执行而不是报错。"""
    await _seed_calls(db)
    rows, _ = await mon.list_llm_calls(db, page_size=9999)
    assert len(rows) == 5  # 只有 5 条，钳制后仍能全取
    assert mon.MAX_PAGE_SIZE == 100


async def test_list_llm_calls_time_window_half_open(db: AsyncSession) -> None:
    """时间窗同样是左闭右开：end 时刻那条不属于窗口。"""
    await _seed_calls(db)
    rows, total = await mon.list_llm_calls(
        db,
        since=BUCKET_START + timedelta(seconds=2),
        until=BUCKET_START + timedelta(seconds=4),
    )
    assert total == 2, "含 2s、3s，不含 4s"
    assert {r["created_at"] for r in rows} == {
        BUCKET_START + timedelta(seconds=2),
        BUCKET_START + timedelta(seconds=3),
    }


async def test_list_llm_calls_date_only_window(db: AsyncSession) -> None:
    """纯日期参数：since 取当日 00:00，until 取次日 00:00（查到当天含全天）。"""
    from datetime import date

    await _seed_calls(db)
    rows, total = await mon.list_llm_calls(
        db, since=date(2026, 10, 14), until=date(2026, 10, 14)
    )
    assert total == 5, "until 是次日零点，故 14 号全天都在窗口内"
    assert len(rows) == 5


async def test_list_llm_calls_user_filter_and_bad_uuid(db: AsyncSession) -> None:
    """user_id 精确过滤；形态非法的 UUID 当作「查不到」而不是 400。"""
    uid = uuid.uuid4()
    add_llm_call(db, scenario="chat", user_id=uid, created_at=BUCKET_START)
    add_llm_call(db, scenario="triage", created_at=BUCKET_START + timedelta(seconds=1))
    await db.commit()

    rows, total = await mon.list_llm_calls(db, user_id=str(uid))
    assert total == 1 and rows[0]["scenario"] == "chat"
    assert rows[0]["user_id"] == str(uid)

    junk, junk_total = await mon.list_llm_calls(db, user_id="not-a-uuid")
    assert junk == [] and junk_total == 0, "乱码筛选条件应返回空而不是整块报错"


async def test_list_llm_calls_serialization_shape(db: AsyncSession) -> None:
    """明细行逐字段列出：UUID 对外是字符串，Numeric 转 float，布尔归一。"""
    uid = uuid.uuid4()
    add_llm_call(
        db, scenario="chat", status="success", cost_usd=Decimal("0.012345"),
        user_id=uid, moderation_blocked=True, is_stream=True,
        created_at=BUCKET_START,
    )
    await db.commit()
    rows, _ = await mon.list_llm_calls(db)
    row = rows[0]
    assert isinstance(row["id"], int)
    assert isinstance(row["cost_usd"], float) and row["cost_usd"] == 0.012345
    assert row["user_id"] == str(uid)
    assert row["moderation_blocked"] is True
    assert row["is_stream"] is True
    assert row["budget_exceeded"] is False
    assert "prompt" not in row and "completion" not in row, "响应不得泄露原文字段"


# ====================================================================
# 九、query_metrics：曲线查询
# ====================================================================


async def _seed_metrics(db: AsyncSession) -> None:
    """两个桶、两类指标、tag 维度齐备。"""
    b1 = BUCKET_START
    b2 = BUCKET_START + timedelta(minutes=1)
    for bucket, val in ((b1, 100.0), (b2, 200.0)):
        db.add(
            AIMetric(
                bucket_time=bucket, metric_name=ms.METRIC_LLM_P95_LATENCY,
                metric_value=val, sample_count=5,
            )
        )
        db.add(
            AIMetric(
                bucket_time=bucket, metric_name=ms.METRIC_TOKEN_USAGE,
                metric_value=val, sample_count=int(val), tags={"model": "qwen-max"},
            )
        )
        db.add(
            AIMetric(
                bucket_time=bucket, metric_name=ms.METRIC_TOKEN_USAGE,
                metric_value=1.0, sample_count=1, tags={"model": "gpt-4o"},
            )
        )
    await db.commit()


async def test_query_metrics_by_name_and_order(db: AsyncSession) -> None:
    await _seed_metrics(db)
    points = await mon.query_metrics(db, ms.METRIC_LLM_P95_LATENCY)
    assert len(points) == 2
    assert [p["bucket_time"] for p in points] == sorted(p["bucket_time"] for p in points)
    assert [p["metric_value"] for p in points] == [100.0, 200.0]
    assert all(isinstance(p["metric_value"], float) for p in points)
    assert points[0]["tags"] == {}


async def test_query_metrics_multi_names(db: AsyncSession) -> None:
    await _seed_metrics(db)
    points = await mon.query_metrics(
        db, [ms.METRIC_LLM_P95_LATENCY, ms.METRIC_TOKEN_USAGE]
    )
    assert len(points) == 6
    assert {p["metric_name"] for p in points} == {
        ms.METRIC_LLM_P95_LATENCY,
        ms.METRIC_TOKEN_USAGE,
    }


async def test_query_metrics_time_window(db: AsyncSession) -> None:
    await _seed_metrics(db)
    only_first = await mon.query_metrics(
        db, since=BUCKET_START, until=BUCKET_START + timedelta(minutes=1)
    )
    assert {p["bucket_time"] for p in only_first} == {BUCKET_START}
    assert len(only_first) == 3

    from_second = await mon.query_metrics(db, since=BUCKET_START + timedelta(minutes=1))
    assert {p["bucket_time"] for p in from_second} == {BUCKET_START + timedelta(minutes=1)}


async def test_query_metrics_tags_subset_match(db: AsyncSession) -> None:
    """tags 是子集匹配：只给 model 也能命中带更多维度的行。"""
    await _seed_metrics(db)
    qwen = await mon.query_metrics(db, ms.METRIC_TOKEN_USAGE, tags={"model": "qwen-max"})
    assert len(qwen) == 2
    assert all(p["tags"]["model"] == "qwen-max" for p in qwen)

    gpt = await mon.query_metrics(db, ms.METRIC_TOKEN_USAGE, tags={"model": "gpt-4o"})
    assert len(gpt) == 2 and all(p["metric_value"] == 1.0 for p in gpt)

    miss = await mon.query_metrics(db, ms.METRIC_TOKEN_USAGE, tags={"model": "none"})
    assert miss == []

    # 无 tag 行在被要求匹配 tag 时不出现
    strict = await mon.query_metrics(db, ms.METRIC_LLM_P95_LATENCY, tags={"model": "qwen-max"})
    assert strict == []

    # 数字与字符串混排比较前归一
    db.add(
        AIMetric(
            bucket_time=BUCKET_START, metric_name="probe", metric_value=1,
            sample_count=1, tags={"level": 2},
        )
    )
    await db.commit()
    numeric = await mon.query_metrics(db, "probe", tags={"level": "2"})
    assert len(numeric) == 1


async def test_query_metrics_limit_clamped(db: AsyncSession) -> None:
    """limit 超上限按 MAX_METRIC_LIMIT 执行；<=0 归一到至少 1。"""
    await _seed_metrics(db)
    limited = await mon.query_metrics(db, limit=2)
    assert len(limited) == 2
    zero = await mon.query_metrics(db, limit=0)
    assert len(zero) == 1, "limit 非法时至少返回 1 个点而不是空查询"
    huge = await mon.query_metrics(db, limit=999_999)
    assert len(huge) == 6


async def test_query_metrics_overview_reads_preagg(db: AsyncSession) -> None:
    """overview 从预聚合取读数；backlog 取最后一个桶而不是求和。"""
    now = utcnow()
    b1 = (now - timedelta(minutes=2)).replace(second=0, microsecond=0)
    b2 = (now - timedelta(minutes=1)).replace(second=0, microsecond=0)
    for bucket, backlog_val in ((b1, 5.0), (b2, 8.0)):
        db.add(AIMetric(bucket_time=bucket, metric_name=ms.METRIC_TASK_QUEUE_BACKLOG,
                        metric_value=backlog_val, sample_count=1))
        db.add(AIMetric(bucket_time=bucket, metric_name=ms.METRIC_LLM_CALL_COUNT,
                        metric_value=3.0, sample_count=3))
        db.add(AIMetric(bucket_time=bucket, metric_name=ms.METRIC_TOKEN_USAGE,
                        metric_value=10.0, sample_count=10, tags={"model": "qwen-max"}))
    await db.commit()

    data = await mon.overview(db, window_minutes=60)
    assert data["source"] == "preagg"
    assert data["bucket_count"] == 2
    # 积压取末桶快照：8 而不是 13
    assert data["metrics"]["task_queue_backlog"] == 8.0
    assert data["metrics"]["llm_call_count"] == 3.0
    # token_usage 是窗口内求和（两个桶各 10）
    assert data["metrics"]["token_usage_total"] == 20.0
    # 没落库的指标保持 None，不谎报 0
    assert data["metrics"]["llm_error_rate"] is None


async def test_overview_falls_back_to_live_when_no_preagg(db: AsyncSession) -> None:
    """预聚合缺席（新部署/Beat 停摆）时实时扫明细兜底，并标 source=live。"""
    now = utcnow()
    moment = now - timedelta(seconds=30)
    add_llm_call(db, scenario="chat", status="success", latency_ms=100,
                 prompt_tokens=10, completion_tokens=5, cost_usd=Decimal("0.001"),
                 created_at=moment)
    add_llm_call(db, scenario="chat", status="error", latency_ms=900,
                 prompt_tokens=0, completion_tokens=0, cost_usd=None,
                 created_at=moment + timedelta(seconds=1))
    await db.commit()

    data = await mon.overview(db, window_minutes=60)
    assert data["source"] == "live"
    assert data["bucket_count"] == 0
    m = data["metrics"]
    assert m["llm_call_count"] == 2.0
    assert m["llm_error_rate"] == 0.5
    assert m["llm_p95_latency_ms"] == 900.0
    assert m["token_usage_total"] == 15.0
    assert m["token_cost_usd"] == 0.001


async def test_overview_queue_snapshot_is_realtime(db: AsyncSession) -> None:
    """queue 段始终实时统计 AITask 状态分布（运维关心的即时事实）。"""
    for status in ("pending", "pending", "running", "completed"):
        db.add(
            AITask(id=uuid.uuid4(), task_type="triage", status=status, progress=0,
                   retry_count=0, created_at=utcnow(), updated_at=utcnow())
        )
    await db.commit()
    data = await mon.overview(db, window_minutes=5)
    assert data["queue"]["pending"] == 2
    assert data["queue"]["running"] == 1
    assert data["queue"]["completed"] == 1


async def test_overview_window_floor(db: AsyncSession) -> None:
    """window_minutes 非法（0/负数）归一到 1，不做零宽窗口查询。"""
    data = await mon.overview(db, window_minutes=0)
    assert data["window_minutes"] == 1
    assert data["source"] == "live"


# ====================================================================
# 十、HTTP 端到端：权限映射与响应装配
# ====================================================================

ADMIN = "admin"
DOCTOR_A = "doctor_a"
PATIENT1 = "patient1"

_WORLD_KEYS: dict[str, tuple[str, str]] = {
    ADMIN: ("admin01", "admin"),
    DOCTOR_A: ("doctor_a", "doctor"),
    PATIENT1: ("patient01", "patient"),
}


@pytest.fixture()
def world(engine) -> dict[str, Any]:
    """管理员 + 医生 + 患者各一账号（403 必须由真实存在的角色账号触发）。"""
    cardio = make_department(engine, "CARDIO", "心血管内科")
    return {
        "engine": engine,
        "cardio": cardio,
        "admin01": make_user(engine, role_name="admin", username="admin01", real_name="管理"),
        "doctor_a": make_user(engine, role_name="doctor", username="doctor_a",
                              real_name="张三", department_id=cardio),
        "patient01": make_user(engine, role_name="patient", username="patient_01"),
    }


def _resolve_identity(world: dict[str, Any], tag: str) -> tuple[uuid.UUID, str, str]:
    key, role_name = _WORLD_KEYS[tag]
    return world[key], key, role_name


@pytest.fixture()
async def client_factory(engine, world):
    """按身份标签构造已鉴权 ASGI 客户端；覆盖 get_db / get_current_user。

    ``require_permission`` 的非 admin 分支要回表查 role_permissions JOIN permissions，
    因此必须同时灌入 roles 与权限矩阵种子（engine 夹具里已做），否则 403 会假阳性。

    会话工厂在每次进入时新建、退出时 dispose：pytest-asyncio 用 function 作用域循环，
    跨用例复用同一 AsyncEngine 会把连接绑到已关闭的循环上（实测打挂过整套用例）。
    """
    import app.imcs_main as imcs_main_module

    @asynccontextmanager
    async def _build(tag: str | None = ADMIN):
        local_factory = build_session_factory(engine)

        async def override_get_db() -> AsyncIterator[AsyncSession]:
            async with local_factory() as session:
                yield session

        async def fake_check_redis() -> bool:
            return True

        app.dependency_overrides[get_db] = override_get_db
        if tag is not None:
            uid, username, role_name = _resolve_identity(world, tag)

            async def override_current_user():
                return as_user(uid, username, role_name)

            app.dependency_overrides[get_current_user] = override_current_user

        originals = (imcs_main_module.async_session_factory, imcs_main_module.check_redis)
        imcs_main_module.async_session_factory = local_factory
        imcs_main_module.check_redis = fake_check_redis

        transport = ASGITransport(app=app)
        try:
            async with AsyncClient(transport=transport, base_url="http://test") as http_client:
                yield http_client
        finally:
            app.dependency_overrides.clear()
            (
                imcs_main_module.async_session_factory,
                imcs_main_module.check_redis,
            ) = originals
            await local_factory.kw["bind"].dispose()

    yield _build


def _mounted(path_fragment: str) -> bool:
    try:
        paths = app.openapi()["paths"]
    except Exception:  # noqa: BLE001 - openapi 生成异常也当作未挂载
        return False
    return any(path_fragment in p for p in paths)


needs_monitoring = pytest.mark.skipif(
    not _mounted("/monitoring"), reason="monitoring router 尚未由主代理挂载到 api_router"
)

ENDPOINTS = (
    "/api/v1/monitoring/overview",
    "/api/v1/monitoring/metrics",
    "/api/v1/monitoring/llm-calls",
    "/api/v1/monitoring/traces/does-not-exist",
)


@needs_monitoring
@pytest.mark.parametrize("tag", [DOCTOR_A, PATIENT1])
async def test_http_non_admin_forbidden_everywhere(client_factory, tag: str) -> None:
    """矩阵里 ai_monitor 只有 ADMIN 一条 read_all → 医生/患者四个 GET 全 403。"""
    async with client_factory(tag) as client:
        for url in ENDPOINTS:
            resp = await client.get(url)
            assert resp.status_code == 403, f"{url} 应为 403，实际 {resp.status_code}"
            assert error_code_of(resp) == "auth_forbidden"
        post = await client.post("/api/v1/monitoring/aggregate", json={})
        assert post.status_code == 403, "POST /aggregate 同样挂 read_all 闸门"


@needs_monitoring
async def test_http_admin_gets_200_on_read_endpoints(client_factory, db) -> None:
    """admin 访问 overview / metrics / llm-calls 均 200，响应结构符合 schema。"""
    # 样本落在「当前分钟之前」：overview 的窗口是 [now-60min, now)，
    # 而明细查询按 created_at 倒序，用过去时刻可避免与并发写入争排序位置。
    add_llm_call(db, scenario="chat", created_at=utcnow() - timedelta(seconds=90))
    await db.commit()

    async with client_factory(ADMIN) as client:
        ov = await client.get("/api/v1/monitoring/overview")
        assert ov.status_code == 200, ov.text
        body = ov.json()
        assert set(body) == {
            "window_minutes", "window_start", "window_end",
            "source", "bucket_count", "metrics", "queue",
        }
        assert body["source"] in ("preagg", "live")
        assert set(body["metrics"]) == {
            "llm_call_count", "llm_error_rate", "llm_p95_latency_ms",
            "token_usage_total", "token_cost_usd", "degradation_rate",
            "moderation_block_rate", "retrieval_zero_result_rate",
            "task_queue_backlog",
        }

        met = await client.get("/api/v1/monitoring/metrics")
        assert met.status_code == 200, met.text
        assert met.json()["count"] == len(met.json()["items"])

        calls = await client.get("/api/v1/monitoring/llm-calls")
        assert calls.status_code == 200, calls.text
        cb = calls.json()
        assert cb["total"] >= 1
        assert cb["page"] == 1 and cb["page_size"] == 20


@needs_monitoring
async def test_http_admin_trace_404_and_error_shape(client_factory) -> None:
    """不存在 trace_id → 404 + 业务错误码（不是 500，也不是扁平 detail）。"""
    async with client_factory(ADMIN) as client:
        resp = await client.get(f"/api/v1/monitoring/traces/{uuid.uuid4()}")
    assert resp.status_code == 404, resp.text
    assert error_code_of(resp) == "resource_not_found"


@needs_monitoring
async def test_http_admin_trace_replay_roundtrip(client_factory, db) -> None:
    """回放端点装配完整响应：四段 + 时间线 + 汇总。"""
    tid = str(uuid.uuid4())
    await _seed_trace(db, tid)

    async with client_factory(ADMIN) as client:
        resp = await client.get(f"/api/v1/monitoring/traces/{tid}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trace_id"] == tid
    assert body["counts"] == {
        "llm_calls": 2, "retrievals": 1, "workflows": 1, "triage_results": 1
    }
    assert body["summary"]["total_tokens"] == 520
    assert body["summary"]["degraded_count"] == 1
    assert len(body["timeline"]) == 5
    assert body["workflows"][0]["branch_taken"] == "emergency"
    assert body["llm_calls"][0]["workflow_node"] == "extract_symptoms"


@needs_monitoring
async def test_http_metrics_endpoint_filters(client_factory, db) -> None:
    """/metrics 的 metric / metrics / tag / 时间窗参数落到 service 生效。"""
    db.add(AIMetric(bucket_time=BUCKET_START, metric_name=ms.METRIC_TOKEN_USAGE,
                    metric_value=42, sample_count=42, tags={"model": "qwen-max"}))
    db.add(AIMetric(bucket_time=BUCKET_START, metric_name=ms.METRIC_LLM_P95_LATENCY,
                    metric_value=123, sample_count=5))
    await db.commit()

    async with client_factory(ADMIN) as client:
        one = await client.get(
            "/api/v1/monitoring/metrics",
            params={
                "metric": ms.METRIC_TOKEN_USAGE,
                "start_date": "2026-10-14T00:00:00Z",
                "end_date": "2026-10-15T00:00:00Z",
            },
        )
        assert one.status_code == 200, one.text
        assert one.json()["metric_names"] == [ms.METRIC_TOKEN_USAGE]
        assert len(one.json()["items"]) == 1

        tagged = await client.get(
            "/api/v1/monitoring/metrics",
            params={"metrics": f"{ms.METRIC_TOKEN_USAGE},{ms.METRIC_LLM_P95_LATENCY}",
                    "tag": "model=qwen-max"},
        )
        assert tagged.status_code == 200, tagged.text
        items = tagged.json()["items"]
        assert len(items) == 1 and items[0]["metric_name"] == ms.METRIC_TOKEN_USAGE

        # 非法 tag（没有等号）忽略过滤条件而不是报错
        bad_tag = await client.get(
            "/api/v1/monitoring/metrics",
            params={"metric": ms.METRIC_LLM_P95_LATENCY, "tag": "oops"},
        )
        assert bad_tag.status_code == 200
        assert len(bad_tag.json()["items"]) == 1


@needs_monitoring
async def test_http_aggregate_endpoint_writes_bucket(client_factory, db, engine) -> None:
    """POST /aggregate 指定桶 → 同步返回写了多少行；重复调用只覆盖不新增。"""
    add_llm_call(db, scenario="chat", status="success", latency_ms=100,
                 prompt_tokens=10, completion_tokens=5, cost_usd=Decimal("0.001"),
                 created_at=BUCKET_START + timedelta(seconds=15))
    await db.commit()

    payload = {"bucket_start": BUCKET_START.isoformat()}
    async with client_factory(ADMIN) as client:
        first = await client.post("/api/v1/monitoring/aggregate", json=payload)
        assert first.status_code == 200, first.text
        fb = first.json()
        assert fb["ok"] is True
        assert len(fb["buckets"]) == 1
        assert fb["buckets"][0]["written"] > 0
        assert fb["total_written"] == fb["buckets"][0]["written"]
        assert "已聚合" in fb["message"]

        second = await client.post("/api/v1/monitoring/aggregate", json=payload)
        sb = second.json()
        assert sb["ok"] is True
        # 幂等：第二次把第一次的行删掉重建
        assert sb["buckets"][0]["deleted"] == fb["buckets"][0]["written"]

    # 独立会话回读：断言库里真实落了多少行，而不是复用写入会话的内存态。
    # engine 必须显式声明为夹具参数——裸写模块级的 ``world`` 拿到的是
    # FixtureFunctionDefinition 对象而不是它返回值，world["engine"] 会 TypeError。
    async with _session_scope(engine) as check_db:
        grouped = await metric_map(check_db, BUCKET_START)
    assert single(grouped, ms.METRIC_LLM_CALL_COUNT)["value"] == 1.0


@needs_monitoring
async def test_http_aggregate_lookback_respects_cap(client_factory) -> None:
    """lookback 超过 MAX_AGGREGATE_LOOKBACK_MINUTES 由 schema 拦成 422。"""
    from app.schemas.monitoring import MAX_AGGREGATE_LOOKBACK_MINUTES

    async with client_factory(ADMIN) as client:
        over = await client.post(
            "/api/v1/monitoring/aggregate",
            json={"lookback_minutes": MAX_AGGREGATE_LOOKBACK_MINUTES + 1},
        )
        assert over.status_code == 422, over.text

        small = await client.post(
            "/api/v1/monitoring/aggregate", json={"lookback_minutes": 2}
        )
        assert small.status_code == 200, small.text
        body = small.json()
        assert len(body["buckets"]) == 2, "lookback=2 补算两个完整桶"
        assert body["ok"] is True


@needs_monitoring
async def test_http_overview_window_validation(client_factory) -> None:
    """window_minutes 越界由 Query(ge=1, le=MAX) 拦下，不进 service。"""
    from app.schemas.monitoring import MAX_OVERVIEW_WINDOW_MINUTES

    async with client_factory(ADMIN) as client:
        too_big = await client.get(
            "/api/v1/monitoring/overview",
            params={"window_minutes": MAX_OVERVIEW_WINDOW_MINUTES + 1},
        )
        assert too_big.status_code == 422
        zero = await client.get("/api/v1/monitoring/overview", params={"window_minutes": 0})
        assert zero.status_code == 422


@needs_monitoring
async def test_http_llm_calls_endpoint_paginates(client_factory, db) -> None:
    """明细端点的 page/page_size 与过滤参数透传生效。"""
    for i in range(4):
        add_llm_call(db, scenario="chat" if i % 2 else "triage", status="success",
                     latency_ms=100 + i, prompt_tokens=1, completion_tokens=1,
                     created_at=BUCKET_START + timedelta(seconds=i))
    await db.commit()

    async with client_factory(ADMIN) as client:
        page = await client.get(
            "/api/v1/monitoring/llm-calls", params={"page": 1, "page_size": 2}
        )
        assert page.status_code == 200, page.text
        body = page.json()
        assert body["total"] == 4 and len(body["items"]) == 2

        filtered = await client.get(
            "/api/v1/monitoring/llm-calls", params={"scenario": "chat"}
        )
        assert filtered.json()["total"] == 2
        assert {i["scenario"] for i in filtered.json()["items"]} == {"chat"}

        oversized = await client.get(
            "/api/v1/monitoring/llm-calls", params={"page_size": 999}
        )
        assert oversized.status_code == 422, "page_size 超上限应被协议层拒绝"


# ====================================================================
# 十一、窗口口径对照（派单口径 vs 实现口径）
# ====================================================================


async def test_bucket_endpoints_inclusive_at_upper_bound_is_the_known_gap(
    db: AsyncSession,
) -> None:
    """把「左闭右开」钉成契约：end 时刻那条**不**属于本桶。

    派单口径写的是「窗口内造 llm_call_logs」，实现选的是左闭右开
    （相邻桶恰好衔接、边界时刻只归后一桶）。这里同时钉死两侧：
    end 时刻那条不进本桶，但进下一桶——避免将来有人改成双闭导致重复计数。
    """
    boundary = BUCKET_END  # 恰好等于本桶右端点
    db.add(
        LLMCallLog(
            scenario="chat", provider="dashscope", model="qwen-max",
            prompt_tokens=1, completion_tokens=1,
            latency_ms=10, status="success", created_at=boundary,
        )
    )
    await db.commit()

    current = await ms.aggregate_minute_bucket(db, BUCKET_START, BUCKET_END)
    assert current["llm_call_rows"] == 0, "左闭右开：右端点那条不属本桶"

    nxt = await ms.aggregate_minute_bucket(
        db, BUCKET_END, BUCKET_END + timedelta(minutes=1)
    )
    assert nxt["llm_call_rows"] == 1, "同一条应完整落入下一桶，不丢也不重"


# ====================================================================
# 十二、常量与登记清单
# ====================================================================


def test_metric_name_constants_unique_and_registered() -> None:
    """ALL_METRIC_NAMES 覆盖全部 METRIC_* 常量且不重名（看板与写入共用一份名字）。"""
    assert len(set(ms.ALL_METRIC_NAMES)) == len(ms.ALL_METRIC_NAMES)
    consts = {
        v for k, v in vars(ms).items() if k.startswith("METRIC_") and isinstance(v, str)
    }
    assert consts == set(ms.ALL_METRIC_NAMES)
    # models/ai.py 的列注释登记值必须能在常量表里找到（改一处必改两处会被抓到）
    for name in ("llm_call_count", "llm_error_rate", "llm_p95_latency",
                 "llm_first_token_p95", "token_usage", "token_cost",
                 "degradation_rate", "moderation_block_rate",
                 "retrieval_zero_result_rate", "task_queue_backlog"):
        assert name in consts


def test_backlog_and_error_status_sets() -> None:
    """积压只算 pending+running（retrying 由重试链路自管）；失败只算 error/timeout。"""
    assert set(ms.BACKLOG_STATUSES) == {"pending", "running"}
    assert "retrying" not in ms.BACKLOG_STATUSES
    assert set(ms.ERROR_STATUSES) == {"error", "timeout"}
    assert "degraded" not in ms.ERROR_STATUSES


def test_decimal_precision_constants() -> None:
    """精度常量与列定义同源：率值落 Numeric(16,4)、金额落同源 6 位。"""
    assert ms.RATE_DECIMALS == 4
    assert ms.COST_DECIMALS == 6
    assert round(1 / 3, ms.RATE_DECIMALS) == 0.3333
    assert math.isclose(ms.safe_ratio(2, 3), 0.6667, abs_tol=1e-9)
