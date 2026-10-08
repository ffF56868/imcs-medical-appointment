"""AI 业务任务：报告解读、分诊工作流、RAGAS 评测。

这些任务的共同特征是**调用 LLM 且耗时不可控**，因此必须异步化：
报告解读一次可能十几秒，若同步执行，检验科录入报告的接口会长时间挂起，
前端超时而后台其实还在跑，用户重复提交又会产生多份解读。

异步化后的交互模型：录入报告 → 立即返回并派发任务 → 前端轮询或
WebSocket 推送拿结果。任务状态持久化在 ``ai_tasks`` 表，
Celery 的 result backend 只保留 1 小时，不能当长期状态源。
"""

from __future__ import annotations

import logging

from app.tasks.celery_app import (
    QUEUE_EVALUATE,
    QUEUE_INTERPRET,
    celery_app,
)

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="app.tasks.ai_tasks.interpret_exam_report",
    queue=QUEUE_INTERPRET,
    max_retries=2,
    default_retry_delay=30,  # 解读失败多因 LLM 抖动，30 秒后重试即可
    time_limit=120,  # 单份报告解读 2 分钟硬上限
    soft_time_limit=100,
    acks_late=True,
)
def interpret_exam_report(self, report_id: str, ai_task_id: str) -> dict:
    """检查报告 AI 解读。

    阶段 4 实现完整工作流：
    1. 读取报告的结构化指标与异常标记
    2. 异常项识别（规则优先，参考值区间比对）
    3. 针对异常项做 RAG 检索（RRF 融合召回医学知识）
    4. LLM 生成解读（走统一服务层，享受预算/审核/降级）
    5. 输出审核 + 结果回写 exam_reports

    重试耗尽后标记 failed 而**不阻塞业务**：医生仍可自行判读原始数据，
    AI 解读是辅助而非必需路径。这是医疗 AI 的基本设计原则。
    """
    logger.info("报告解读任务启动 report_id=%s ai_task_id=%s", report_id, ai_task_id)
    raise NotImplementedError("阶段 4 实现：接诊、检查与 AI 报告解读")


@celery_app.task(
    bind=True,
    name="app.tasks.ai_tasks.run_triage_workflow",
    queue=QUEUE_INTERPRET,
    max_retries=1,
    default_retry_delay=10,
    time_limit=60,
    soft_time_limit=45,
    acks_late=True,
)
def run_triage_workflow(self, triage_id: str, ai_task_id: str) -> dict:
    """智能分诊工作流（异步模式）。

    分诊主路径是**同步**的——患者在前端等结果，同步返回体验更好。
    这个异步版本用于两种场景：
    1. 批量分诊评估（跑历史数据算分诊准确率）
    2. 同步路径降级后的补偿：LLM 超时时先返回规则分诊结果，
       异步任务稍后补算 LLM 结果并更新记录
    """
    logger.info("分诊工作流任务启动 triage_id=%s", triage_id)
    raise NotImplementedError("阶段 3 实现：智能分诊与预约挂号")


@celery_app.task(
    bind=True,
    name="app.tasks.ai_tasks.run_ragas_evaluation",
    queue=QUEUE_EVALUATE,
    max_retries=1,
    default_retry_delay=60,
    time_limit=3600,  # RAGAS 要跑完整个评测集，允许 1 小时
    soft_time_limit=3500,
    acks_late=True,
)
def run_ragas_evaluation(self, eval_run_id: str) -> dict:
    """RAGAS 自动评测。

    从旧项目 ``evaluation.py`` 的 BackgroundTasks + threading.Lock 迁移过来。
    旧实现的问题：评测跑几十分钟，期间持有进程内锁，任何其他评测请求
    都被拒绝；进程重启则评测静默丢失，状态永远停在 running。

    独立队列 QUEUE_EVALUATE 且并发 1：评测是资源消耗大户，
    与在线业务任务混跑会挤占 LLM 配额，影响真实用户体验。
    """
    logger.info("RAGAS 评测任务启动 eval_run_id=%s", eval_run_id)
    raise NotImplementedError("阶段 6 实现：评测体系迁移")


__all__ = [
    "interpret_exam_report",
    "run_triage_workflow",
    "run_ragas_evaluation",
]
