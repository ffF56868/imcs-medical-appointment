"""知识库异步任务：向量化与增量重建。

替换旧项目 ``app/routers/knowledge.py`` 里的
``BackgroundTasks.add_task(run_rebuild_job, job.id)`` + ``threading.Lock``。

旧实现的三个问题：
1. ``threading.Lock`` 只在单进程有效，多副本部署时两个实例会同时重建，
   后完成的覆盖先完成的，索引状态不可预测
2. BackgroundTasks 在响应返回后执行，进程重启即丢失，且重建期间无法查进度
3. 旧 ``rebuild_vector_store()`` 是 drop 全部再重建，改一条资料也要重算所有向量

新实现用 Celery 队列 + ``ai_tasks`` 表持久化状态，并按 ``content_hash``
做**增量**向量化：只有新增或变更的切块才重新 embedding。

**本模块的事务边界**（为什么每篇文档提交一次）：一篇文档 = 删旧切块 + 建新切块
+ 填 tsv + 写向量，是一个自洽的最小单元。整体一个事务的话，跑到第 80 篇失败
会把前 79 篇的成果一起回滚，重试要从头再来；逐篇提交则失败点之后的文档留 pending，
下一轮 ``collect_pending_documents`` 自然补齐，已完成的不会重复调用 embedding。

**为什么状态流转要独立提交**：running/completed/failed 是前端进度的唯一依据。
如果它和文档写入同事务，业务异常导致 rollback 时会把 running 也一起撤销，
任务看起来「从没开始过」，僵尸清理也就无从判断。因此状态写库一律紧跟一次 commit。
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from celery.exceptions import Retry, SoftTimeLimitExceeded

from app.core.database import get_sync_db
from app.models.ai import AITask, AITaskStatus
from app.services import knowledge_service as ks
from app.tasks.celery_app import QUEUE_VECTORIZE, celery_app

logger = logging.getLogger(__name__)


def _parse_id(raw: str) -> uuid.UUID:
    """把传入的任务 ID 转成 UUID。

    非法 ID 直接抛 ValueError 让 Celery 记为任务失败，而不是拿 None 去查库——
    后者会伪装成「任务不存在」，掩盖真正的派发侧参数错误。
    """
    try:
        return uuid.UUID(str(raw))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"ai_task_id 不是合法 UUID：{raw!r}") from exc


def _load_task(sync_session: Any, task_id: uuid.UUID) -> AITask | None:
    """加行锁读取任务记录。"""
    return ks.load_task_for_update_sync(sync_session, task_id)


def _mark_failed(sync_session: Any, task: AITask, message: str) -> None:
    """把任务标 failed 并提交。

    单独抽出来是因为有三条不同的失败路径（软超时、重试耗尽、未预期异常）
    都要做这件事，各写一遍必然出现某条路径漏写 completed_at，
    而僵尸清理正是靠 started_at/completed_at 判超时。

    状态机不允许 failed→failed，重复标记会被 assert_transition 拒掉，
    因此这里吞掉非法迁移异常：任务已经是终态就是我们要的结果，
    不该因为「已经失败」再抛一个错把 worker 日志刷脏。
    """
    try:
        ks.set_task_status_sync(sync_session, task, AITaskStatus.FAILED.value, error_message=message)
    except Exception:  # noqa: BLE001 - 已是终态时静默接受
        logger.debug("任务 %s 已处于终态，跳过 failed 标记", task.id)
    sync_session.commit()


def _retry_or_fail(self: Any, sync_session: Any, task: AITask, exc: BaseException) -> Any:
    """可重试异常的处理：还有重试次数就 retry，耗尽则标 failed。

    ``self.retry`` 抛 ``Retry`` 异常来中断本次执行，因此必须在标 failed **之前**
    调用——反过来会把一个「稍后会重试」的任务永久标成失败。

    重试前把状态推进到 retrying 并递增 retry_count：用户在前端能看到
    「第 2 次重试中」，而不是干等一个毫无动静的 running。
    """
    retries_done = int(getattr(getattr(self, "request", None), "retries", 0) or 0)
    max_retries = int(getattr(self, "max_retries", 0) or 0)

    if retries_done < max_retries:
        ks.set_task_status_sync(
            sync_session,
            task,
            AITaskStatus.RETRYING.value,
            error_message=f"第 {retries_done + 1} 次重试：{exc}"[:2000],
        )
        ks.bump_retry_count_sync(sync_session, task)
        sync_session.commit()
        logger.warning(
            "任务 %s 执行失败，准备第 %d 次重试：%s", task.id, retries_done + 1, exc
        )
        return self.retry(exc=exc)

    logger.error("任务 %s 重试次数耗尽，标记失败：%s", task.id, exc)
    _mark_failed(sync_session, task, f"重试 {retries_done} 次后仍失败：{exc}"[:2000])
    return {
        "ai_task_id": str(task.id),
        "status": AITaskStatus.FAILED.value,
        "error": str(exc)[:500],
    }


@celery_app.task(
    bind=True,
    name="app.tasks.knowledge_tasks.vectorize_knowledge",
    queue=QUEUE_VECTORIZE,
    max_retries=3,
    default_retry_delay=300,  # 向量化失败通常是 embedding 服务抖动，5 分钟后再试
    time_limit=1800,  # 30 分钟硬超时，防止卡死占住队列
    soft_time_limit=1700,  # 软超时先于硬超时，留出时间把任务标记为 failed
    acks_late=True,
)
def vectorize_knowledge(self, ai_task_id: str) -> dict:
    """知识库增量向量化。

    阶段 2 实现完整逻辑：
    1. 计算各知识文档的 content sha256，与库中记录比对
    2. 对新增/变更文档做清洗、分块（复用旧项目 text_processing 的 700/120 策略）
    3. jieba 分词写入 tsvector，批量 embedding 写入 vector_embeddings
    4. 删除已不存在文档的切块与向量
    5. 更新文档的 index_status 与 indexed_at

    幂等性：任务入口先检查 ai_task 状态，已 completed 直接返回，
    避免重复触发时重复调用 embedding 接口浪费配额。
    """
    logger.info("知识库向量化任务启动 ai_task_id=%s", ai_task_id)

    task_uuid = _parse_id(ai_task_id)
    # with 而非手动 next()/close()：Celery 里任务抛异常是常态，
    # 手写 close() 一旦漏在某个 return 分支上就会泄漏一条数据库连接。
    with get_sync_db() as sync_session:
        # ---------- 入口幂等检查 ----------
        task = _load_task(sync_session, task_uuid)
        if task is None:
            logger.error("向量化任务找不到 ai_task 记录 id=%s", task_uuid)
            return {"ai_task_id": str(task_uuid), "status": "not_found"}

        if task.status == AITaskStatus.COMPLETED.value:
            logger.info("任务 %s 已完成，跳过重复执行", task.id)
            return {
                "ai_task_id": str(task.id),
                "status": task.status,
                "result": task.result,
                "idempotent_hit": True,
            }

        # cancelled 也是终态：管理员中途叫停的任务不能被重投的消息复活
        if task.status in (AITaskStatus.CANCELLED.value,):
            logger.info("任务 %s 已被取消，不执行", task.id)
            return {"ai_task_id": str(task.id), "status": task.status}

        # ---------- pending/retrying/failed → running ----------
        # failed 不在状态机的允许边里（failed 只能回 pending），因此遇到 failed
        # 说明这是人工重新触发的同一记录：先流转到 pending 再到 running，
        # 两步都走状态机，不绕过校验。
        if task.status == AITaskStatus.FAILED.value:
            ks.set_task_status_sync(sync_session, task, AITaskStatus.PENDING.value)
        ks.set_task_status_sync(sync_session, task, AITaskStatus.RUNNING.value, progress=0)
        sync_session.commit()

        report = ks.VectorizeReport()

        try:
            docs = ks.collect_pending_documents(sync_session)
            report.documents_scanned = len(docs)
            logger.info("本轮待向量化文档 %d 篇", len(docs))

            total = len(docs)
            for done, doc in enumerate(docs, start=1):
                stats = ks.vectorize_document(sync_session, doc, ai_task_id=task.id)
                report.chunks_deleted += stats.deleted_chunks
                report.chunks_created += stats.created_chunks
                report.vectors_created += stats.created_vectors
                if stats.skipped:
                    report.documents_skipped += 1
                else:
                    report.documents_indexed += 1
                report.per_document.append(
                    {
                        "document_id": str(stats.document_id),
                        "created_chunks": stats.created_chunks,
                        "deleted_chunks": stats.deleted_chunks,
                        "created_vectors": stats.created_vectors,
                        "skipped": stats.skipped,
                        "reason": stats.reason,
                    }
                )

                # 每篇一提交：见模块 docstring 的事务边界说明
                sync_session.commit()

                # 进度按已完成篇数折算，上限留 5%：收尾还有孤儿清理与统计落库，
                # 直接推到 100 会让用户在最后一段等待里看到假完成。
                if total:
                    ks.set_task_status_sync(
                        sync_session,
                        task,
                        AITaskStatus.RUNNING.value,
                        progress=min(95, int(done / total * 100)),
                        progress_message=f"已向量化 {done}/{total} 篇",
                    )
                    sync_session.commit()

            # ---------- 孤儿清理 ----------
            report.orphans_removed = ks.remove_orphaned(sync_session)
            sync_session.commit()

            ks.set_task_status_sync(
                sync_session,
                task,
                AITaskStatus.COMPLETED.value,
                result=report.to_result(),
                progress=100,
                progress_message=(
                    f"完成：{report.documents_indexed} 篇已索引，"
                    f"{report.chunks_created} 个切块"
                )[:200],
            )
            sync_session.commit()
            logger.info(
                "向量化任务完成 ai_task_id=%s 新建切块=%d 清理孤儿=%d",
                task.id,
                report.chunks_created,
                report.orphans_removed,
            )
            return {"ai_task_id": str(task.id), "status": task.status, "result": report.to_result()}

        except SoftTimeLimitExceeded as exc:
            # 软超时单独捕获且不重试：剩余几秒根本不够重跑一遍，
            # 在这里发起 retry 只会立刻被硬超时 SIGKILL，任务状态永远停在 running。
            # 标 failed 让用户能重新触发，已提交的文档成果保留。
            sync_session.rollback()
            logger.error("向量化任务软超时 ai_task_id=%s", task_uuid)
            _mark_failed(sync_session, task, f"执行超时（软超时上限）：{exc}"[:2000])
            return {
                "ai_task_id": str(task.id),
                "status": AITaskStatus.FAILED.value,
                "error": "soft_time_limit_exceeded",
            }

        except Retry:
            # self.retry 抛的控制流异常，不是业务失败，原样交给 Celery
            sync_session.rollback()
            raise

        except Exception as exc:
            sync_session.rollback()
            task = _load_task(sync_session, task_uuid) or task
            outcome = _retry_or_fail(self, sync_session, task, exc)
            if isinstance(outcome, dict):
                return outcome
            raise


@celery_app.task(
    bind=True,
    name="app.tasks.knowledge_tasks.rebuild_all_vectors",
    queue=QUEUE_VECTORIZE,
    max_retries=1,
    time_limit=3600,
    soft_time_limit=3500,
    acks_late=True,
)
def rebuild_all_vectors(self, ai_task_id: str, *, model_version: str) -> dict:
    """全量重建向量。

    仅在换 embedding 模型时使用——不同模型产出的向量不在同一语义空间，
    混用会让检索结果完全失真，因此必须全量重建而非增量。

    ``model_version`` 是必填参数而不是从配置读取：强制调用方显式声明
    目标模型，避免「配置改了但只重建了一半」这种难以发现的脏状态。

    与增量的区别：**只清向量表，保留切块与 tsv**。切块文本和分词结果与
    embedding 模型无关，跟着一起删就等于把不需要重算的东西也算进了重建成本，
    而且一旦中途失败，全文检索路会整条空掉。
    """
    logger.info(
        "全量向量重建启动 ai_task_id=%s model_version=%s", ai_task_id, model_version
    )

    task_uuid = _parse_id(ai_task_id)
    with get_sync_db() as sync_session:
        task = _load_task(sync_session, task_uuid)
        if task is None:
            logger.error("全量重建找不到 ai_task 记录 id=%s", task_uuid)
            return {"ai_task_id": str(task_uuid), "status": "not_found"}

        if task.status == AITaskStatus.COMPLETED.value:
            logger.info("任务 %s 已完成，跳过重复执行", task.id)
            return {
                "ai_task_id": str(task.id),
                "status": task.status,
                "result": task.result,
                "idempotent_hit": True,
            }
        if task.status == AITaskStatus.CANCELLED.value:
            return {"ai_task_id": str(task.id), "status": task.status}

        if task.status == AITaskStatus.FAILED.value:
            ks.set_task_status_sync(sync_session, task, AITaskStatus.PENDING.value)
        ks.set_task_status_sync(
            sync_session,
            task,
            AITaskStatus.RUNNING.value,
            progress=0,
            progress_message=f"全量重建目标模型 {model_version}",
        )
        sync_session.commit()

        report = ks.VectorizeReport()
        report.per_document = []

        try:
            # ---------- 1. 清空向量表（保留切块与 tsv）----------
            deleted_vectors = ks.truncate_vectors(sync_session)
            sync_session.commit()
            logger.info("已清空向量表，删除 %d 条向量，切块与全文索引保留", deleted_vectors)

            # ---------- 2. 逐文档重嵌 ----------
            docs = ks.iter_active_documents(sync_session)
            report.documents_scanned = len(docs)
            total = len(docs)

            for done, doc in enumerate(docs, start=1):
                # 不看 content_hash：reembed_document_chunks 对现有切块一律重算，
                # 目标就是「所有向量都换新模型」。若走增量的跳过分支，
                # 已是 indexed 且 hash 未变的文档会被略过，
                # 旧模型的向量就永久留在库里——这正是本任务要消除的状态。
                count = ks.reembed_document_chunks(sync_session, doc, ai_task_id=task.id)
                doc.index_status = ks.INDEX_STATUS_INDEXED
                doc.indexed_at = datetime.now(UTC)
                doc.content_hash = ks.content_sha256(doc.content)
                report.vectors_created += count
                report.documents_indexed += 1
                report.per_document.append(
                    {"document_id": str(doc.id), "reembedded_vectors": count}
                )
                sync_session.commit()

                if total:
                    ks.set_task_status_sync(
                        sync_session,
                        task,
                        AITaskStatus.RUNNING.value,
                        progress=min(95, int(done / total * 100)),
                        progress_message=f"已重嵌 {done}/{total} 篇",
                    )
                    sync_session.commit()

            report.orphans_removed = ks.remove_orphaned(sync_session)
            sync_session.commit()

            result = report.to_result()
            result["vectors_truncated"] = deleted_vectors
            result["model_version"] = model_version

            ks.set_task_status_sync(
                sync_session,
                task,
                AITaskStatus.COMPLETED.value,
                result=result,
                progress=100,
                progress_message=f"全量重建完成：{report.vectors_created} 条向量",
            )
            sync_session.commit()
            logger.info(
                "全量重建完成 ai_task_id=%s 向量=%d 模型=%s",
                task.id,
                report.vectors_created,
                model_version,
            )
            return {"ai_task_id": str(task.id), "status": task.status, "result": result}

        except SoftTimeLimitExceeded as exc:
            sync_session.rollback()
            logger.error("全量重建软超时 ai_task_id=%s", task_uuid)
            _mark_failed(sync_session, task, f"执行超时（软超时上限）：{exc}"[:2000])
            return {
                "ai_task_id": str(task.id),
                "status": AITaskStatus.FAILED.value,
                "error": "soft_time_limit_exceeded",
            }

        except Retry:
            sync_session.rollback()
            raise

        except Exception as exc:
            sync_session.rollback()
            task = _load_task(sync_session, task_uuid) or task
            outcome = _retry_or_fail(self, sync_session, task, exc)
            if isinstance(outcome, dict):
                return outcome
            raise


__all__ = ["vectorize_knowledge", "rebuild_all_vectors"]
