"""监控与运维任务：指标聚合、僵尸任务清理。

这两个任务由 Celery Beat 定时触发，是 AI 可观测性的后台支撑：

**指标聚合**：监控看板不直接扫 ``llm_call_logs`` 明细表。明细表随调用量
线性增长，实时算 P95 会让看板越用越慢，且多个管理员同时打开看板会重复
计算同样的分位数。改为每分钟预聚合到 ``ai_metrics``，看板查预聚合结果，
响应时间与明细表规模解耦。具体口径（分位数算法、幂等实现）见
``app/services/metrics_service.py`` 模块 docstring——本文件只负责
「在 worker 里把异步服务跑起来 + 兜住异常」这层桥接职责。

**僵尸任务清理**：worker 崩溃或被 kill 时，正在执行的任务会永远停在
``running`` 状态——Celery 的 acks_late 会把消息重投给其他 worker，
但 ``ai_tasks`` 表里的业务状态没人更新。旧项目
``knowledge.py`` 靠「启动时扫描 stale job」处理，多副本下会重复扫描且
只覆盖重启时刻；改为定时扫描后，任何时刻的僵尸任务都能在一个周期内被纠正。

============================ 为什么两个任务都是同步 def ============================

Celery prefork 模式下每个 task 独占进程、进程内无既存事件循环，因此
``run_async()``（见 celery_app.py）可以安全地新建循环跑协程再关闭。
聚合逻辑本身是 async SQLAlchemy（与 API 侧共用同一套异步 service，
不允许出现「Beat 跑的聚合」和「手动触发的聚合」两套实现），
所以任务体是「同步壳 + run_async 驱动异步芯」。

注意**不能**在任务体里自己套 ``asyncio.run``：那会绕过 ``run_async`` 的
``aclose_adapter_clients`` 收尾，实测刷出成片的
``Task exception was never retrieved: Event loop is closed``。

============================ 失败绝不允许搞崩 worker ============================

两个任务体最外层都是 try/except 收口，异常转成
``{"ok": False, "error": ...}`` 结构化返回值并记日志，**任务本身正常返回**。
理由：这两个任务挂在 default 队列，同一条队列上还跑着用户可见的业务任务；
一个旁路观测逻辑抛异常会让 Celery 把它记成 FAILURE 并按配置重试/告警，
排查噪声远大于收益。下一轮 Beat 自然会补算，幂等保证补算不会重复计数。
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select

from app.core.database import get_sync_db
from app.models.ai import AITask, AITaskStatus
from app.services import metrics_service as ms
from app.tasks.celery_app import QUEUE_DEFAULT, celery_app, run_async

logger = logging.getLogger(__name__)

# 任务停留多久算僵尸。要显著大于最长任务的正常耗时，
# 否则会把正在跑的知识库重建误判为僵尸并标记失败。
STALE_TASK_THRESHOLD_MINUTES = 60

# 单次 reap 最多处理多少条。阈值调小（比如临时改成 1 分钟）时可能一次命中上万行，
# 全量 UPDATE 会长时间持锁并阻塞正在写 ai_tasks 的业务任务；
# 分批让每轮 Beat 消化一部分，积压会在几个周期内收敛。
REAP_BATCH_LIMIT = 500

# error_message 截断长度。列是 Text 没有上限，但这条信息要给前端展示与人工阅读，
# 超长堆栈属于日志不属于状态字段。
REAP_MESSAGE_MAX_CHARS = 500

# 僵尸标记文案。**必须包含「被僵尸清理任务标记」**：这是区分「LLM 真失败了」
# 与「worker 挂了没人收尾」的唯一线索，排障时看 error_message 第一眼就要能分辨。
STALE_TASK_MESSAGE = (
    "任务超时未完成，被僵尸清理任务标记（worker 可能已崩溃或被终止；可重新发起）"
)


@celery_app.task(
    bind=True,
    name="app.tasks.monitoring_tasks.aggregate_ai_metrics",
    queue=QUEUE_DEFAULT,
    max_retries=0,  # 定时任务无需重试，下个周期自然会再跑
    time_limit=300,
    soft_time_limit=240,
    # 同一时刻只允许一个聚合任务在跑。多个 Beat 副本或任务积压时，
    # 重复聚合会产生重复指标行，污染看板曲线。
    expires=60,
)
def aggregate_ai_metrics(self) -> dict:
    """按分钟桶聚合 AI 指标到 ai_metrics 表。

    聚合以下指标（口径全部钉死在 ``metrics_service``，此处只列清单）：
    - llm_call_count / llm_error_rate：调用量与失败率
    - llm_p50_latency / llm_p95_latency / llm_p99_latency：延迟分位数
      （最近秩 ceiling 法，Python 侧计算，不用 PG 专有 percentile_cont）
    - llm_first_token_p95：流式首 token 延迟（体感关键指标）
    - token_usage / token_cost：用量与成本，token_usage 按 model 维度打 tag
    - degradation_rate：降级率（status='degraded' 或 degradation_level>0 的并集占比）
    - moderation_block_rate：审核拦截率
    - retrieval_zero_result_rate：检索零结果率（知识库缺口信号）
    - task_queue_backlog：队列积压深度（pending+running 的瞬时快照）

    幂等性：``ai_metrics`` 上没有 (bucket_time, metric_name, tags) 唯一约束，
    因此不能用 ON CONFLICT；改为在同一事务内按 (bucket_time, metric_name)
    先删后插，整桶覆盖重写。重复执行同一分钟桶只覆盖不新增行数。

    只聚合**上一个完整分钟桶**（当前分钟还在写入，现在算会得到偏低的值，
    且看板曲线会出现一根正在爬升的假线）。
    """
    logger.info("AI 指标聚合任务触发")
    try:
        result = run_async(ms.run_aggregate())
    except Exception as exc:  # noqa: BLE001 - worker 顶层兜底，不让旁路观测搞崩进程
        logger.exception("AI 指标聚合任务异常退出")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500], "written": 0}

    ok = bool(result.get("ok", True))
    if not ok:
        logger.warning("AI 指标聚合未成功：%s", result.get("error"))
    return {"ok": ok, **result}


@celery_app.task(
    bind=True,
    name="app.tasks.monitoring_tasks.reap_stale_tasks",
    queue=QUEUE_DEFAULT,
    max_retries=0,
    time_limit=120,
    soft_time_limit=100,
    expires=120,
)
def reap_stale_tasks(self, threshold_minutes: int = STALE_TASK_THRESHOLD_MINUTES) -> dict:
    """清理超时僵尸任务：把超时的 pending/running 标成 failed。

    判据（三个条件同时成立才动）：
    1. ``status`` ∈ {pending, running}。retrying 也纳入考虑过但不纳入——它由
       ``knowledge_tasks`` 的重试链路自己管理（retry 前会显式写 retrying），
       清理它等于打断一个还在正常推进的重试。
    2. 参考时刻早于阈值：**running 取 started_at，pending 取 created_at**。
       这个区分很关键——pending 的 started_at 恒为 NULL，若统一按 started_at 过滤，
       「派发了但 worker 从没接手」（Redis 消息丢失、队列名写错）的任务
       永远不会被判僵尸，前端就一直转圈。
    3. 参考时刻非 NULL（NULL 视为数据异常，跳过而不是当作 0 时刻立刻标失败）。

    副作用：置 status=failed、写 completed_at 与 duration_ms（据此可算出这次僵尸
    实际占用了多久）、追加 error_message 说明是被清理的。**不改 retry_count**——
    清理不代表系统自动重试过，把它加上去会让「已重试 N 次」的展示说谎。

    保留可重试语义：状态机里 ``failed → pending`` 是唯一出边
    （见 state_machine.AI_TASK_TRANSITIONS），用户在前端点「重新发起」或
    业务侧重新派发即可回到 pending。因此标 failed 不是终局封棺，
    恰恰是把任务从「无人认领的 running」交回用户手里。

    阈值必须大于最长任务的正常耗时（知识库全量重建 time_limit 是 3600s），
    因此默认 60 分钟。误判的代价是把正在跑的任务标记失败，
    比漏判（僵尸永久占位）更可控——前者用户会重新触发，后者无人察觉。
    """
    minutes = _normalize_threshold(threshold_minutes)
    logger.info("僵尸任务清理触发 threshold_minutes=%d", minutes)

    cutoff = datetime.now(UTC) - timedelta(minutes=minutes)
    try:
        return _reap_sync(cutoff, minutes)
    except Exception as exc:  # noqa: BLE001 - worker 顶层兜底
        logger.exception("僵尸任务清理异常退出")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500], "reaped": 0}


def _normalize_threshold(raw: Any) -> int:
    """阈值归一为正整数分钟。

    Beat 传参、手动 ``apply_async`` 传参都可能给出 float / str / None。
    非法或非正值一律回落默认值：阈值是防误杀的安全阀，
    宁可保守（用默认 60 分钟）也不能因为一个参数类型错误就把正在跑的任务标失败。
    """
    try:
        minutes = int(float(raw))
    except (TypeError, ValueError):
        return STALE_TASK_THRESHOLD_MINUTES
    if minutes <= 0 or math.isnan(minutes):
        return STALE_TASK_THRESHOLD_MINUTES
    return minutes


def _reap_sync(cutoff: datetime, minutes: int) -> dict[str, Any]:
    """同步 Session 上执行一轮清理（Celery worker 侧数据库操作用同步会话）。"""
    reaped_ids: list[str] = []
    now = datetime.now(UTC)

    with get_sync_db() as session:
        # ---------- 1. running：以 started_at 为准 ----------
        running_rows = session.execute(
            select(AITask.id, AITask.started_at)
            .where(
                AITask.status == AITaskStatus.RUNNING.value,
                AITask.started_at.is_not(None),
                AITask.started_at < cutoff,
            )
            .limit(REAP_BATCH_LIMIT)
        ).all()
        running_ids = [str(row[0]) for row in running_rows]

        # ---------- 2. pending：以 created_at 为准（started_at 恒 NULL） ----------
        pending_rows = session.execute(
            select(AITask.id)
            .where(
                AITask.status == AITaskStatus.PENDING.value,
                AITask.created_at < cutoff,
            )
            .limit(REAP_BATCH_LIMIT)
        ).all()
        pending_ids = [str(row[0]) for row in pending_rows]

        # ---------- 3. 逐条带 ORM 改，以便正确维护 completed_at / duration_ms ----------
        # 用 UPDATE ... WHERE id IN (...) 一条 SQL 更快，但那样就得在 SQL 里
        # 复制 duration_ms = now - started_at 的计算，而 started_at 可能是 NULL
        # （pending 从未开始）。走 ORM 逐条改可以把「NULL 就不填 duration_ms」
        # 写成 Python 分支，语义清楚且不依赖方言函数。
        message = STALE_TASK_MESSAGE[:REAP_MESSAGE_MAX_CHARS]
        targets = session.execute(
            select(AITask).where(
                AITask.id.in_([row[0] for row in running_rows] + [row[0] for row in pending_rows])
            )
        ).scalars().all()

        for task in targets:
            task.status = AITaskStatus.FAILED.value
            task.error_message = message
            task.completed_at = now
            if task.started_at is not None:
                task.duration_ms = int((now - _ensure_utc(task.started_at)).total_seconds() * 1000)
            reaped_ids.append(str(task.id))

        session.commit()

    logger.info(
        "僵尸任务清理完成 阈值=%d分钟 截止=%s 标记=%d（running=%d pending=%d）",
        minutes,
        cutoff.isoformat(),
        len(reaped_ids),
        len(running_ids),
        len(pending_ids),
    )
    return {
        "ok": True,
        "threshold_minutes": minutes,
        "cutoff": cutoff.isoformat(),
        "reaped": len(reaped_ids),
        "reaped_running": len(running_ids),
        "reaped_pending": len(pending_ids),
        "task_ids": reaped_ids[:20],
        "truncated": len(reaped_ids) >= REAP_BATCH_LIMIT,
    }


def _ensure_utc(moment: datetime) -> datetime:
    """naive datetime 按 UTC 理解。

    库里 started_at 声明为 timezone-aware，但 SQLite 读回来是 naive
    （测试环境就是 SQLite）。不做这一步的话 ``aware - naive`` 直接 TypeError，
    而且这个 TypeError 只在测试里出现、线上 PG 永远不报——最难查的一类差异。
    """
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


__all__ = ["aggregate_ai_metrics", "reap_stale_tasks", "STALE_TASK_THRESHOLD_MINUTES"]
