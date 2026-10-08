"""Alembic 迁移环境配置。

为什么用同步引擎：Alembic 的迁移脚本本质是 DDL 语句序列，不涉及异步 I/O，
使用同步 psycopg 连接串（``postgresql+psycopg://``）最省事，也避免了
在迁移脚本里手动管理 async session 的复杂度。FastAPI 运行时仍走异步引擎。

为什么 URL 从 settings 注入而非写在 alembic.ini：本地开发数据库在 localhost:5433，
容器内在 postgres:5432，连接串随环境变化。统一由 ``app.core.config.settings``
管理，再允许环境变量覆盖，一份配置适配所有环境。
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Alembic Config 对象，提供对 .ini 文件的访问
config = context.config

# 配置 Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------- 动态注入数据库连接串 ----------
# 优先级：环境变量 SYNC_DATABASE_URL > DATABASE_URL > settings.sync_database_url
from app.core.config import settings  # noqa: E402

db_url = (
    os.environ.get("SYNC_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
    or settings.sync_database_url
)
config.set_main_option("sqlalchemy.url", db_url)

# ---------- 导入全部模型，触发 metadata 注册 ----------
# 必须在 target_metadata 赋值之前完成，否则 Alembic 看不到表定义
import app.models  # noqa: E402, F401
from app.models.base import Base  # noqa: E402

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """以「离线」模式运行迁移（仅生成 SQL 脚本，不连接数据库）。"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """以「在线」模式运行迁移（连接数据库并执行 DDL）。"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_schemas=False,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
