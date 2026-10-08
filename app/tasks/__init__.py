"""Celery 异步任务包。

替换旧项目 ``BackgroundTasks + threading.Lock`` 的理由：
- BackgroundTasks 在响应返回后才执行，进程重启即丢失，且无法横向扩容
- 进程内 ``threading.Lock`` 只在单进程有效，多副本部署时锁形同虚设，
  会导致知识库被并发重建
- Celery 提供持久化队列、失败重试、任务路由、并发控制与可视化（Flower）
"""

from app.tasks.celery_app import celery_app

__all__ = ["celery_app"]
