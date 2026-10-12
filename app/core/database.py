"""IMCS 数据库访问层。

设计要点：**同一套 SQLAlchemy 2.0 声明式模型，同时服务两种运行时**。

- FastAPI 走 ``create_async_engine`` + ``AsyncSession``（psycopg v3 异步驱动）
- Celery worker 走 ``create_engine`` + ``Session``（psycopg v3 同步驱动）

psycopg v3 的同步与异步共用 ``postgresql+psycopg`` dialect，区别只在于把同一个
连接串交给 ``create_engine`` 还是 ``create_async_engine``，因此模型定义无需分叉。

旧版 ``app/database.py`` 里那套手写 ``SQLITE_COLUMN_MIGRATIONS`` 增量加列字典已被
Alembic 取代，不再需要。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator, Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings


def _engine_kwargs() -> dict[str, object]:
    """连接池参数。pool_pre_ping 防止使用被数据库端断开的陈旧连接。"""
    return {
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_pre_ping": True,
        "pool_recycle": 1800,
        "echo": settings.db_echo,
    }


# ---------- 异步引擎（FastAPI 使用）----------
async_engine: AsyncEngine = create_async_engine(
    settings.database_url,
    **_engine_kwargs(),  # type: ignore[arg-type]
)

async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


# ---------- 同步引擎（Celery worker 使用）----------
# Celery 不支持原生 async worker。任务里数据库操作用同步 Session，
# LLM / Embedding 这类 IO 密集调用用 asyncio.run() 包装异步客户端，
# 避免为了补偿 IO 等待而把 worker 并发度开得过大。
sync_engine = create_engine(
    settings.sync_database_url,
    **_engine_kwargs(),
)

sync_session_factory: sessionmaker[Session] = sessionmaker(
    bind=sync_engine,
    class_=Session,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：每个请求一个独立会话，退出时自动关闭。

    刻意**不在这里 commit**——事务边界由 service 层显式控制，
    避免依赖层隐式提交导致部分写入。
    """
    async with async_session_factory() as session:
        yield session


@contextmanager
def get_sync_db() -> Iterator[Session]:
    """Celery 任务用的同步会话上下文管理器。

    **必须是 context manager 而非裸生成器**：调用方一律写
    ``with get_sync_db() as session:``，普通生成器函数返回的 generator
    不支持上下文协议，worker 任务会在第一步就抛
    ``TypeError: 'generator' object does not support the context manager
    protocol``（实测向量化任务正是这样整批失败、状态永远停在 pending）。
    用 @contextmanager 包装后即支持 with，session_factory 的 with 仍负责
    退出时关闭会话。
    """
    with sync_session_factory() as session:
        yield session


async def dispose_async_engine() -> None:
    """应用关闭时释放连接池。"""
    await async_engine.dispose()


def dispose_sync_engine() -> None:
    """worker 关闭时释放同步连接池。"""
    sync_engine.dispose()
