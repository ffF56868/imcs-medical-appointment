"""监控与运维任务：指标聚合、僵尸任务清理。

这两个任务由 Celery Beat 定时触发，是 AI 可观测性的后台支撑：

**指标聚合**：监控看板不直接扫 ``llm_call_logs`` 明细表。明细表随调用量
线性增长，实时算 P95 会让看板越用越慢，且多个管理员同时打开看板会重复
计算同样的分位数。改为每分钟预聚合到 ``ai_metrics``，看板查预聚合结果，
响应时间与明细表规模解耦。

**僵尸任务清理**：worker 崩溃或被 kill 时，正在执行的任务会永远停在
``running`` 状态——Celery 的 acks_late 会把消息重投给其他 worker，
但 ``ai_tasks`` 表里的业务状态没人更新。旧项目
``knowledge.py`` 靠「启动时扫描 stale job」处理，多副本下会重复扫描且
只覆盖重启时刻；改为定时扫描后，任何时刻的僵尸任务都能在一个周期内被纠正。
"""

from __future__ import annotations

import logging

from app.tasks.celery_app import QUEUE_DEFAULT, celery_app

logger = logging.getLogger(__name__)

# 任务停留多久算僵尸。要显著大于最长任务的正常耗时，
# 否则会把正在跑的知识库重建误判为僵尸并标记失败。
STALE_TASK_THRESHOLD_MINUTES = 60


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

    阶段 6 实现完整逻辑，聚合以下指标：
    - llm_call_count / llm_error_rate：调用量与失败率
    - llm_p95_latency / llm_p50_latency / llm_p99_latency：延迟分位数
    - llm_first_token_p95：流式首 token 延迟（体感关键指标）
    - token_usage / token_cost：用量与成本，按 model 维度打 tag
    - degradation_rate：降级率，按 degradation_level 分布
    - moderation_block_rate：审核拦截率
    - retrieval_zero_result_rate：检索零结果率（知识库缺口信号）
    - task_queue_backlog：队列积压深度

    幂等性：按 (bucket_time, metric_name, tags) 唯一键 upsert，
    重复执行同一分钟桶只覆盖不新增。
    """
    logger.info("AI 指标聚合任务触发")
    raise NotImplementedError("阶段 6 实现：监控看板与指标聚合")


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
    """清理超时僵尸任务。

    把 started_at 早于阈值且仍为 running/pending 的任务标记为 failed，
    并写入 error_message 说明是被清理的。这样前端不会永远显示「处理中」，
    用户能重新触发。

    阈值必须大于最长任务的正常耗时（知识库全量重建 time_limit 是 3600s），
    因此默认 60 分钟。误判的代价是把正在跑的任务标记失败，
    比漏判（僵尸永久占位）更可控——前者用户会重新触发，后者无人察觉。
    """
    logger.info("僵尸任务清理触发 threshold_minutes=%d", threshold_minutes)
    raise NotImplementedError("阶段 6 实现：僵尸任务清理")


__all__ = ["aggregate_ai_metrics", "reap_stale_tasks", "STALE_TASK_THRESHOLD_MINUTES"]
