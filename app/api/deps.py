"""FastAPI 公共依赖：会话、当前用户、角色与权限校验。

三个依赖层层递进：
- ``get_db``：异步会话
- ``get_current_user``：解 JWT 拿到用户（不回表，payload 已含 role）
- ``require_permission``：精确到「资源 + 动作」的权限校验

``get_current_user`` 刻意**不查数据库**——JWT payload 里已带 role，
绝大多数接口只需角色判断。只有需要用户完整档案（如患者的过敏史）
或需要校验 ``is_active`` 实时状态时，才用 ``get_current_user_fresh`` 回表。
这是无状态 JWT 与实时性之间的取舍：账号被禁用后，已签发的 token 在
有效期内仍可用，生产环境需配合 Redis 黑名单缩短这个窗口。
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import (
    AuthenticationError,
    PermissionDeniedError,
    ResourceNotFoundError,
)
from app.core.security import TOKEN_TYPE_ACCESS, decode_token
from app.models.auth import Permission, Role, RolePermission, User

# auto_error=False：未携带凭证时由我们自己抛出统一格式的 401，
# 而不是让 FastAPI 返回它默认的错误结构
bearer_scheme = HTTPBearer(auto_error=False)

DbSession = Annotated[AsyncSession, Depends(get_db)]


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> User:
    """从 JWT 解析当前用户。

    返回的是一个**未绑定会话的轻量 User 对象**，只填充了鉴权所需字段。
    这样避免了每次请求都查库，代价是不能直接访问其 relationship 属性。
    """
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("缺少登录凭证")

    payload = decode_token(credentials.credentials, expected_type=TOKEN_TYPE_ACCESS)
    try:
        user_id = UUID(payload["sub"])
    except (KeyError, ValueError) as exc:
        raise AuthenticationError("登录凭证格式错误") from exc

    user = User(
        id=user_id,
        username=payload.get("username", ""),
        real_name="",
        password_hash="",
        role_id=int(payload.get("role_id", 0)),
        is_active=True,
    )
    # 把角色名挂在实例上，供 require_role 快速判断。
    # role 关系是 lazy="selectin"，直接赋值会触发 ORM 加载，故用私有属性承载。
    user._role_name = payload.get("role", "")  # type: ignore[attr-defined]
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_current_user_fresh(
    current_user: CurrentUser,
    db: DbSession,
) -> User:
    """需要用户实时状态或完整档案时使用，会回表查询。

    校验 ``is_active``：账号被管理员禁用后，此依赖能立即拒绝，
    而轻量版 ``get_current_user`` 无法感知实时状态。
    """
    user = await db.get(User, current_user.id)
    if user is None:
        raise ResourceNotFoundError("用户不存在")
    if not user.is_active:
        raise AuthenticationError("账号已被停用")
    return user


FreshCurrentUser = Annotated[User, Depends(get_current_user_fresh)]


def get_role_name(user: User) -> str:
    """取角色名。兼容轻量用户（来自 JWT）与回表用户（含 relationship）。"""
    role_name = getattr(user, "_role_name", None)
    if role_name:
        return str(role_name)
    if user.role is not None:
        return str(user.role.name)
    return ""


def require_role(*allowed_roles: str):
    """角色级鉴权依赖工厂。

    用法::

        @router.get("/schedule", dependencies=[Depends(require_role("doctor"))])

    适合粗粒度控制。涉及具体资源动作时用 ``require_permission``。
    """
    allowed = set(allowed_roles)

    async def checker(current_user: CurrentUser) -> User:
        if get_role_name(current_user) not in allowed:
            raise PermissionDeniedError(
                f"该操作仅限以下角色：{'、'.join(sorted(allowed))}"
            )
        return current_user

    return checker


async def _load_role_permissions(db: AsyncSession, role_id: int) -> set[tuple[str, str]]:
    """加载某角色的全部 (资源, 动作) 权限对。

    这里每次请求查一次库。若后续压测发现热点，可在 Redis 缓存
    role_id -> 权限集合，并在角色权限变更时主动失效——权限矩阵变更
    频率极低，缓存命中率会很高。当前阶段先保证正确性与可读性。
    """
    statement = (
        select(Permission.resource, Permission.action)
        .join(RolePermission, RolePermission.permission_id == Permission.id)
        .where(RolePermission.role_id == role_id)
    )
    rows = (await db.execute(statement)).all()
    return {(row[0], row[1]) for row in rows}


def require_permission(resource: str, action: str):
    """权限级鉴权依赖工厂：精确到「资源 + 动作」。

    超级管理员（admin）默认拥有全部权限，不在权限矩阵里逐条列举，
    避免每加一个资源都要补一条 admin 授权记录。
    """

    async def checker(current_user: CurrentUser, db: DbSession) -> User:
        role_name = get_role_name(current_user)
        if role_name == "admin":
            return current_user

        permissions = await _load_role_permissions(db, current_user.role_id)
        if (resource, action) not in permissions:
            raise PermissionDeniedError(
                f"缺少权限：{resource}:{action}"
            )
        return current_user

    return checker


async def get_client_ip(request: Request) -> str:
    """取客户端 IP，用于登录限流与审计。

    优先读 ``X-Forwarded-For`` 首段——经 Nginx 反代后 remote_addr 只会是
    容器网段地址。该头部可被伪造，因此只用于限流与日志，不作为安全判定依据。
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


ClientIP = Annotated[str, Depends(get_client_ip)]


__all__ = [
    "DbSession",
    "CurrentUser",
    "FreshCurrentUser",
    "ClientIP",
    "get_current_user",
    "get_current_user_fresh",
    "get_role_name",
    "require_role",
    "require_permission",
    "bearer_scheme",
]
