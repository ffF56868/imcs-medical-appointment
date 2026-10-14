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

错误响应结构：``IMCSError`` 经 ``register_exception_handlers`` 转成
``JSONResponse``，形如 ``{"detail": {"code": ..., "message": ...}}``；
而 FastAPI 自身的校验错误是扁平的 ``{"detail": [ ... ]}``。两者形态不同，
因此测试统一通过 ``error_code_of`` 取值（扁平形态返回 None），
不要在用例里直接下标取 ``["detail"]["code"]``。
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.compiler import compiles

import app.models  # noqa: F401  - 触发全部 ORM 表注册，Base.metadata 才完整
from app.models.base import Base


# ==================== PG UUID → SQLite ====================


@compiles(PGUUID, "sqlite")
def _compile_pg_uuid_on_sqlite(element: PGUUID, compiler, **kwargs) -> str:  # noqa: ANN001
    """SQLite 上没有原生 UUID 类型，DDL 退化为 CHAR(32)。

    不注册这个编译器的话，``Base.metadata.create_all(sqlite_engine)`` 会抛
    ``UnsupportedCompilationError``：PGUUID 只为 postgresql 方言提供 DDL 渲染。

    **必须用 ``sqlalchemy.ext.compiler.compiles``**，它是 SQLAlchemy 提供的
    自定义 DDL 渲染扩展点，第二个参数是方言名。写成
    ``sqlite.register_compile_context`` 之类的形式并不存在，会在 import
    阶段直接 AttributeError 并中断整个测试文件的收集。

    只补 DDL、不碰 DML：PGUUID 继承自 ``sqlalchemy.Uuid``，其
    ``bind_processor`` / ``result_processor`` 不依赖方言，读写行为仍是
    ``uuid.UUID`` ↔ 32 位 hex 文本，因此 service 层拿到的类型完全一致。
    """
    del element, compiler, kwargs
    return "CHAR(32)"


@compiles(sa.BigInteger, "sqlite")
def _compile_big_integer_on_sqlite(element: sa.BigInteger, compiler, **kwargs) -> str:  # noqa: ANN001
    """BigInteger 在 SQLite 上渲染成 INTEGER。

    SQLite 只有类型名恰好为 ``INTEGER`` 的主键才是 rowid 别名、才能获得
    自增行为；``BIGINT`` 主键插入不给 id 会直接 ``NOT NULL constraint failed``。
    阶段 6 的可观测表（llm_call_logs / rag_retrieval_logs / workflow_traces /
    ai_metrics）主键都是 BigInteger，PG 上走 BIGSERIAL 没问题，SQLite 测试
    环境必须降级。与 PGUUID 补丁同理：只动 DDL 渲染，读写行为不变。
    """
    del element, compiler, kwargs
    return "INTEGER"


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
    """为同一套文件库构造异步 Session 工厂，供测试覆盖 ``get_db`` 使用。

    **为什么不能 ``AsyncEngine(sync_engine)``**：``AsyncEngine.__init__``
    首行就校验 ``sync_engine.dialect.is_async``——它接受的是「aiosqlite 这类
    异步方言的底层同步外观」，不是任意同步引擎。传纯同步引擎会直接抛
    ``InvalidRequestError``（实测打挂了全部 31 个用例）。

    正确做法：用同一文件库 DSN 换 ``sqlite+aiosqlite`` 方言重新
    ``create_async_engine``。同步引擎负责建表与 db_read 直读断言，
    异步引擎承载被测会话——两条连接读写同一文件库，串行性由 SQLite
    自身的锁与 busy timeout 保证，测试规模下足够。

    返回的 sessionmaker 其引擎挂在 ``kw["bind"]``，调用方须在 fixture
    结束时 dispose（``kw["bind"]`` 即该 AsyncEngine）。
    """
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    async_engine = create_async_engine(engine.url.set(drivername="sqlite+aiosqlite"))
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


def seed_role_permissions(conn) -> None:  # noqa: ANN001
    """按 rbac_matrix 的权限矩阵灌入 permissions 与 role_permissions。

    **为什么单列一个函数而并进 seed_roles**：认证测试用轻量 JWT 依赖，
    从不查权限表，最小种子（只有 roles）就够；知识管理测试走
    ``require_permission``，它按 (resource, action) 查 role_permissions JOIN
    permissions——空表时非 admin 角色**永远 403**（admin 被依赖工厂直接放行，
    所以只灌 roles 的旧用例全绿、唯独医生读用例翻车，实测复现）。
    数据源直接取 ``ROLE_PERMISSIONS`` 矩阵本身而不是手抄：测试验证的就是
    真实授权边界，矩阵改了测试种子自动跟随，不会出现两处漂移。
    permissions.id 用 enumerate 顺序号即可——权限 ID 没有跨表引用，
    只有 (resource, action) 唯一约束需要成立。
    """
    from app.services.rbac_matrix import ROLE_PERMISSIONS

    existing = {row[0] for row in conn.execute(sa.text("SELECT id FROM permissions")).fetchall()}
    if existing:
        return  # 已灌过则整体跳过：两表数据必须同进同退

    perm_ids: dict[tuple[str, str], int] = {}
    next_id = 1
    for role, permissions in ROLE_PERMISSIONS.items():
        for resource, action in sorted(permissions):
            key = (resource, action)
            if key in perm_ids:
                continue
            perm_ids[key] = next_id
            conn.execute(
                sa.text(
                    "INSERT INTO permissions (id, resource, action, description) "
                    "VALUES (:id, :resource, :action, NULL)"
                ),
                {"id": next_id, "resource": resource, "action": action},
            )
            next_id += 1

    link_existing = {
        (row[0], row[1])
        for row in conn.execute(
            sa.text("SELECT role_id, permission_id FROM role_permissions")
        ).fetchall()
    }
    for role, permissions in ROLE_PERMISSIONS.items():
        role_id = ROLE_IDS[role.value]
        for key in sorted(permissions):
            pair = (role_id, perm_ids[key])
            if pair in link_existing:
                continue
            conn.execute(
                sa.text(
                    "INSERT INTO role_permissions (role_id, permission_id) "
                    "VALUES (:role_id, :perm_id)"
                ),
                {"role_id": pair[0], "perm_id": pair[1]},
            )
            link_existing.add(pair)


__all__ = [
    "AUTH_TABLES",
    "ROLE_IDS",
    "auth_tables",
    "build_session_factory",
    "create_auth_schema",
    "create_auth_schema_async",
    "error_code_of",
    "error_message_of",
    "seed_role_permissions",
    "seed_roles",
]
