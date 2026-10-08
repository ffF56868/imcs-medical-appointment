"""模型基类与通用 Mixin。"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """全局声明式基类。Alembic 通过 ``Base.metadata`` 发现所有表。"""


def utcnow() -> datetime:
    """统一的时区感知当前时间。全项目禁止混用 naive datetime。"""
    return datetime.now(UTC)


class UUIDPrimaryKey:
    """UUID 主键 Mixin。

    用应用侧 ``default=uuid.uuid4`` 而非数据库侧 ``gen_random_uuid()``，
    这样在 SQLite 测试环境（无 pgcrypto）下同样可用。
    """

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )


class TimestampMixin:
    """创建/更新时间 Mixin。updated_at 由数据库侧触发，避免应用漏更新。"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
