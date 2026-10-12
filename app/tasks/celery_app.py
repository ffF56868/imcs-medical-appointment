"""Celery 应用与队列配置。

**队列划分**：按任务的资源特征分队列，而不是全部塞进默认队列。
向量化是 CPU + 网络密集且耗时可达数十分钟，报告解读要求低延迟响应，
两者混在一个队列里会导致「一次知识库重建把所有报告解读堵住」。
分队列后各队列可配独立并发度，互不干扰。

**worker 内的异步处理**：Celery prefork 模式下每个 task 独占一个进程，
进程内没有既存事件循环，因此 ``asyncio.run()`` 是安全的。数据库操作用
同步 Session（避免在同步上下文里维护事件循环），LLM / Embedding 这类
IO 密集调用用 ``asyncio.run()`` 包装异步客户端——全用同步会逼着把
worker 并发度开得很大才能补偿 IO 等待时间。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, TypeVar

from celery import Celery
from celery.signals import setup_logging

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 队列名常量。任务路由与 worker 启动命令都引用它们，避免字符串散落各处。
QUEUE_DEFAULT = "default"
QUEUE_VECTORIZE = "vectorize"
QUEUE_INTERPRET = "interpret"
QUEUE_EVALUATE = "evaluate"

ALL_QUEUES = (QUEUE_DEFAULT, QUEUE_VECTORIZE, QUEUE_INTERPRET, QUEUE_EVALUATE)


celery_app = Celery(
    "imcs",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    # 任务模块按需导入。用字符串而非直接 import，避免 worker 启动时
    # 加载全部业务代码（也规避循环导入）
    include=[
        "app.tasks.knowledge_tasks",
        "app.tasks.ai_tasks",
        "app.tasks.monitoring_tasks",
    ],
)

celery_app.conf.update(
    # ---------- 序列化 ----------
    # 只用 JSON。pickle 序列化允许任意对象反序列化执行代码，
    # 一旦 broker 被污染就是远程代码执行漏洞。
    task_serializer=settings.celery_task_serializer,
    result_serializer=settings.celery_task_serializer,
    accept_content=["json"],
    # ---------- 时区 ----------
    timezone="Asia/Shanghai",
    enable_utc=True,
    # ---------- 可靠性 ----------
    # 任务执行前才确认收到。默认 True 会导致 worker 崩溃时任务丢失，
    # 知识库重建这类长任务丢失后没人知道，索引就一直是旧的。
    task_acks_late=True,
    # 每个子进程处理一个任务后回收，防止长任务内存泄漏累积
    worker_max_tasks_per_child=200,
    worker_prefetch_multiplier=1,
    # 任务与结果都带唯一 ID，便于幂等判断
    task_track_started=True,
    result_expires=settings.celery_result_expires,
    # ---------- 并发 ----------
    worker_concurrency=settings.celery_worker_concurrency,
    # ---------- 任务路由 ----------
    task_routes={
        "app.tasks.knowledge_tasks.*": {"queue": QUEUE_VECTORIZE},
        "app.tasks.ai_tasks.interpret_exam_report": {"queue": QUEUE_INTERPRET},
        "app.tasks.ai_tasks.run_triage_workflow": {"queue": QUEUE_INTERPRET},
        "app.tasks.ai_tasks.*": {"queue": QUEUE_DEFAULT},
        "app.tasks.monitoring_tasks.*": {"queue": QUEUE_DEFAULT},
    },
    # 子进程处理满一定任务数后回收，与 worker_max_tasks_per_child 配合防止内存泄漏。
    # 队列级并发度不在这里配，而是由 worker 启动命令的 -Q 与 -c 决定：
    # 向量化队列并发低（单任务内部已用批量 embedding 打满 IO），
    # 解读队列并发高（用户在前端等结果）。见 docker-compose.yml 的 worker 服务。
    worker_queue_max_tasks_per_child=100,
    # ---------- 定时任务 ----------
    beat_schedule={
        # 每分钟聚合一次 AI 指标，供监控看板查询。
        # 看板不直接扫明细日志表：明细表增长快，实时聚合会让面板越用越慢。
        "aggregate-ai-metrics": {
            "task": "app.tasks.monitoring_tasks.aggregate_ai_metrics",
            "schedule": 60.0,
            "options": {"queue": QUEUE_DEFAULT},
        },
        # 每 5 分钟清理超时的僵尸任务（worker 崩溃后停留在 running 的任务）
        "reap-stale-ai-tasks": {
            "task": "app.tasks.monitoring_tasks.reap_stale_tasks",
            "schedule": 300.0,
            "options": {"queue": QUEUE_DEFAULT},
        },
    },
)


@setup_logging.connect
def _configure_celery_logging(**kwargs: Any) -> None:
    """接管 Celery 日志配置，让它与应用日志格式一致。

    不接管的话 Celery 会自己配一套 root logger，覆盖应用的日志设置，
    导致业务日志格式混乱、级别失效。

    必须带 ``**kwargs``：Celery 的 Signal._connect_signal 会强制校验接收器
    签名，不接受关键字参数就直接抛
    ``ValueError: Signal receiver must accept keyword arguments``，
    导致 worker 在 import 阶段崩溃、无限重启。派发时会传入 sender 等参数，
    本函数不需要它们，因此收下即弃。
    """
    del kwargs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """在同步 Celery 任务里执行异步协程。

    Celery prefork 模式下每个任务独占进程且无既存事件循环，
    ``asyncio.run()`` 会新建一个循环、跑完再关闭，是安全的。

    注意：不要在已经运行着事件循环的上下文里调用（如 FastAPI 路由内），
    那会抛 RuntimeError。路由内应直接 await 异步 service。

    **为什么每次先清适配器缓存**：``AsyncOpenAI`` 的 httpx 连接池绑定首次
    使用的事件循环，而这里每个任务新建一个循环。不清缓存的话，同一 worker
    进程里的第二个 LLM/embedding 任务会复用属于已关闭循环的连接。
    详见 ``provider.reset_adapter_clients`` 的说明。
    """
    from app.services.llm.provider import reset_adapter_clients

    reset_adapter_clients()
    return asyncio.run(coro)


def dispatch_task(
    task_name: str,
    *args: Any,
    queue: str | None = None,
    countdown: float | None = None,
    task_id: str | None = None,
    **kwargs: Any,
):
    """按名称派发任务。

    包一层而不是直接 ``celery_app.send_task``，是为了统一处理
    「broker 不可用」的情况：此时抛异常会让业务操作整体失败，
    但异步任务（如报告解读）失败不应该阻塞报告录入本身。
    调用方据返回值决定是否降级为同步执行或标记待重试。
    """
    options: dict[str, Any] = {}
    if queue:
        options["queue"] = queue
    if countdown is not None:
        options["countdown"] = countdown
    if task_id:
        # 显式指定 task_id 实现幂等：同一 ID 重复派发时 Celery 会去重，
        # 避免用户连点两次触发两遍向量化
        options["task_id"] = task_id
    try:
        return celery_app.send_task(task_name, args=args, kwargs=kwargs, **options)
    except Exception as exc:  # noqa: BLE001 - broker 故障不能让业务写入失败
        logger.error("派发 Celery 任务 %s 失败：%s", task_name, exc)
        return None


__all__ = [
    "celery_app",
    "run_async",
    "dispatch_task",
    "QUEUE_DEFAULT",
    "QUEUE_VECTORIZE",
    "QUEUE_INTERPRET",
    "QUEUE_EVALUATE",
    "ALL_QUEUES",
]
