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
"""

from __future__ import annotations

import logging

from app.tasks.celery_app import QUEUE_VECTORIZE, celery_app

logger = logging.getLogger(__name__)


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
    raise NotImplementedError("阶段 2 实现：RRF 检索与知识库")


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
    """
    logger.info(
        "全量向量重建启动 ai_task_id=%s model_version=%s", ai_task_id, model_version
    )
    raise NotImplementedError("阶段 2 实现：RRF 检索与知识库")


__all__ = ["vectorize_knowledge", "rebuild_all_vectors"]
