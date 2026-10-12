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
import uuid

from sqlalchemy import select

from app.core.database import async_session_factory
from app.models.ai import TriageResult
from app.services.triage_workflow import assess
from app.tasks.celery_app import (
    QUEUE_EVALUATE,
    QUEUE_INTERPRET,
    celery_app,
    run_async,
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
def run_triage_workflow(self, triage_id: str, ai_task_id: str = "") -> dict:
    """智能分诊工作流（异步补偿）。

    分诊主路径是**同步**的——患者在前端等结果。本异步任务用于 docstring
    所述的两种场景：批量评估、以及规则先行结果的事后 LLM 补偿。
    输入是一条已存在的 TriageResult（通常由 batch-assess 的规则先行落库），
    重跑 assess 带 existing 把结论升级为 AI/hybrid。

    **幂等**：decision_source 已是 ai/hybrid 的记录跳过（不重复烧配额）；
    补偿后仍是 rule（LLM 仍降级）也计一次已处理，不无限重投。
    父任务进度用行锁读改写的 done 计数收敛到 completed，
    不依赖 decision_source（避免降级条目永远到不了终态）。
    """
    from datetime import UTC, datetime as dt

    from app.core.database import get_sync_db
    from app.services import knowledge_service as ks

    triage_uuid = _parse_id(triage_id)
    task_uuid = _parse_id(ai_task_id) if ai_task_id else None

    async def _compensate() -> dict[str, object]:
        async with async_session_factory() as db:
            record = await db.get(TriageResult, triage_uuid)
            if record is None:
                logger.error("分诊补偿找不到 triage id=%s", triage_uuid)
                return {"triage_id": str(triage_uuid), "status": "not_found"}
            if record.decision_source in ("ai", "hybrid"):
                return {
                    "triage_id": str(triage_uuid),
                    "status": "already_compensated",
                    "decision_source": record.decision_source,
                }
            outcome = await assess(
                db,
                chief_complaint=record.chief_complaint,
                user_id=record.user_id,
                patient_id=record.patient_id,
                ai_task_id=task_uuid,
                trace_id=record.trace_id,
                existing=record,
            )
            return {
                "triage_id": str(outcome.triage_id),
                "status": "compensated",
                "decision_source": outcome.decision_source,
                "degraded": outcome.degraded,
                "recommended_department_name": outcome.department_name,
            }

    result = run_async(_compensate())

    # ---------- 父任务进度收敛（批量场景）----------
    if task_uuid is not None:
        with get_sync_db() as sync_session:
            task = ks.load_task_for_update_sync(sync_session, task_uuid)
            if task is not None and task.status in ("pending", "running", "retrying"):
                payload = task.payload or {}
                total = int(payload.get("total", 0)) or 1
                done = int(payload.get("done", 0)) + 1
                payload["done"] = done
                task.payload = payload
                if task.status == "pending":
                    ks.set_task_status_sync(
                        sync_session, task, "running", progress=0,
                        progress_message=f"分诊补偿 {done}/{total}",
                    )
                if done >= total:
                    ks.set_task_status_sync(
                        sync_session, task, "completed", progress=100,
                        progress_message=f"分诊补偿完成 {done}/{total}",
                        result={"total": total, "done": done},
                    )
                else:
                    ks.set_task_status_sync(
                        sync_session, task, task.status,
                        progress=min(95, int(done / total * 100)),
                        progress_message=f"分诊补偿 {done}/{total}",
                    )
                sync_session.commit()

    return result


def _parse_id(raw: str) -> uuid.UUID:
    """把传入的 ID 转成 UUID。非法值抛 ValueError 让 Celery 记为任务失败。"""
    try:
        return uuid.UUID(str(raw))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"id 不是合法 UUID：{raw!r}") from exc


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
