"""IMCS 模型在 SQLite 上建表所需的兼容性处理（仅测试用）。

为什么需要这个模块：``app/models`` 全部主键与外键列都写成
``sqlalchemy.dialects.postgresql.UUID(as_uuid=True)``。该方言类型没有注册
SQLite 编译器，``Base.metadata.create_all(sqlite_engine)`` 会直接抛
``UnsupportedCompilationError``。而 ``models/types.py`` 里的 JSONB / ARRAY /
tsvector / pgvector 都是 TypeDecorator，非 PG 方言会自动退化为 TEXT，**不需要**干预。

因此这里只做两件事，且都只作用于测试引擎，不改任何生产模型定义：

1. 给 PG UUID 补一个 SQLite 编译实现（存 32 位 hex 文本，读回仍是 ``uuid.UUID``，
   对 service 层完全透明）。
2. 提供「认证链路所需表」的白名单建表函数。显式白名单而非「全表减去不兼容表」，
   是为了让建表范围一目了然，也不会因为某天新增一张 PG 专有表就让整套用例莫名变红。

另外一处必须知道的坑：``register_exception_handlers`` 把 IMCSError 转成
``HTTPException`` 后原样抛出，而 Starlette 的自定义 Exception 处理器并不美化
HTTPException —— 它会绕过 app 级中间件栈，导致响应体是
``{"detail": {"code": ..., "message": ...}}`` 而非 FastAPI 的正常校验错误结构。
测试断言据此写，见 ``error_code_of``。
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from sqlalchemy.dialects.postgresql import UUID as PGUUID

import app.models  # noqa: F401  - 触发全部 ORM 表注册，Base.metadata 才完整
from app.models.base import Base


# ==================== PG UUID → SQLite ====================


@sqlite.register_compile_context(PGUUID, PGUUID.__visit_name__)
def _compile_pg_uuid_on_sqlite(element: PGUUID, compiler, **kwargs) -> str:  # noqa: ANN001
    """SQLite 上没有原生 UUID 类型，DDL 退化为 CHAR(32)。

    PGUUID 继承自 ``sqlalchemy.Uuid``，其 ``bind_processor`` / ``result_processor``
    不依赖方言，读写行为保持 ``uuid.UUID`` ↔ hex 文本，因此只需补 DDL 编译。
    """
    del element, compiler, kwargs
    return "CHAR(32)"


# ==================== 建表白名单 ====================

AUTH_TABLES: tuple[str, ...] = (
    "roles",
    "permissions",
    "role_permissions",
    "departments",
    "users",
    "patient_profiles",
    "doctor_schedules",
)


def auth_tables() -> list[sa.Table]:
    """按外键依赖顺序返回认证链路涉及的表对象。"""
    return [Base.metadata.tables[name] for name in AUTH_TABLES]


def create_auth_schema(engine) -> None:  # noqa: ANN001
    """在同步引擎上建出认证相关表。"""
    Base.metadata.create_all(engine, tables=auth_tables(), checkfirst=True)


async def create_auth_schema_async(connection) -> None:  # noqa: ANN001
    """在异步连接上建表。

    ``run_sync`` 会把原始 DBAPI 连接交给同步函数，而 ``create_all`` 需要的是
    Engine 对象，所以这里逐表调用 ``table.create`` —— 它接受 Connection。
    """
    await connection.run_sync(_create_on)


def _create_on(sync_connection) -> None:  # noqa: ANN001
    """在既有连接上执行 DDL（供 run_sync / ``with engine.begin()`` 调用）。

    刻意用 ``table.create`` 而非 ``Base.metadata.create_all``：后者需要 Engine，
    而测试要把表建到**当前这条共享连接**上。PGUUID 的 SQLite 编译补丁已注册在
    方言全局，因此这里能直接产出可执行的 CHAR(32) DDL。
    """
    for table in auth_tables():
        table.create(sync_connection, checkfirst=True)


# ==================== 种子最小集 ====================

ROLE_IDS: dict[str, int] = {
    "patient": 1,
    "doctor": 2,
    "lab_tech": 3,
    "cashier": 4,
    "pharmacist": 5,
    "admin": 6,
}


def seed_roles(conn) -> None:  # noqa: ANN001
    """写入六类角色行。

    注册流程要按 ``Role.name == 'patient'`` 查 role_id，空 roles 表会让所有
    注册请求返回 503，所以这是最小必需种子。ID 与 rbac_matrix 保持一致。
    """
    existing = {row[0] for row in conn.execute(sa.text("SELECT id FROM roles")).fetchall()}
    for name, role_id in ROLE_IDS.items():
        if role_id in existing:
            continue
        conn.execute(
            sa.text("INSERT INTO roles (id, name, description) VALUES (:id, :name, NULL)"),
            {"id": role_id, "name": name},
        )


def build_session_factory(engine):  # noqa: ANN001
    """把同步引擎包成异步 Session 工厂，供测试覆盖 ``get_db`` 使用。

    为什么用 ``AsyncEngine(sync_engine)`` 而不是手写同步 Session + greenlet 桥接：
    SQLAlchemy 官方提供的这层包装会把同步连接池的操作交给 greenlet_spawn，
    业务代码拿到的就是标准 ``AsyncSession``，await 语义天然成立。
    这样被测 service 无需为测试做任何妥协。
    """
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

    async_engine = AsyncEngine(engine)
    return async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)


# ==================== 响应解析助手 ====================


def error_code_of(response) -> str | None:  # noqa: ANN001
    """从错误响应里取机器可读错误码。

    兼容两种形态：经 IMCSError 处理器产生的嵌套 ``detail.code``，
    以及 FastAPI 正常抛出的扁平 ``detail`` 字符串（此时返回 None）。
    """
    detail = response.json().get("detail")
    if isinstance(detail, dict):
        code = detail.get("code")
        return str(code) if code is not None else None
    return None


def error_message_of(response) -> str:  # noqa: ANN001
    """取人类可读错误信息，同样兼容嵌套与扁平两种形态。"""
    payload = response.json()
    detail = payload.get("detail")
    if isinstance(detail, dict):
        return str(detail.get("message", ""))
    if isinstance(detail, str):
        return detail
    return str(payload)


__all__ = [
    "AUTH_TABLES",
    "ROLE_IDS",
    "auth_tables",
    "build_session_factory",
    "create_auth_schema",
    "create_auth_schema_async",
    "error_code_of",
    "error_message_of",
    "seed_roles",
]
