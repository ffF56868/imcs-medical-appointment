"""认证路由：注册、登录、刷新、登出、当前用户。

路由层保持薄：解析请求 → 调 ``auth_service`` → 组装响应。
唯一的例外是令牌吊销检查——它需要同时拿到「HTTP 凭证」和「解码后的 payload」，
而公共依赖 ``get_current_user`` 只向上返回 User 对象，因此在本模块内
自建一个返回 payload 的依赖（见 ``require_active_access_token``）。

为什么黑名单不放进公共依赖：``get_current_user`` 刻意做成纯 JWT 解析、
零额外 IO，这是无状态令牌的核心收益。把 Redis 往返塞进每个请求的必经之路，
等于推翻该设计。吊销检查只在 auth 自身的路由上有意义（验证待作废的令牌是否还活着），
业务接口要不要也查黑名单属于后续批次的取舍，届时按压测数据再决定。
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import ClientIP, DbSession, bearer_scheme
from app.core.exceptions import AuthenticationError, ResourceNotFoundError
from app.core.security import TOKEN_TYPE_ACCESS, decode_token
from app.models.auth import Role, User
from app.schemas.auth import (
    AuthResult,
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    RegisterRequest,
    TokenPair,
    UserOut,
)
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["认证"])


# ==================== 局部依赖 ====================


async def require_active_access_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> dict[str, Any]:
    """解析访问令牌并校验其未被吊销。

    返回 payload 而非 User：登出需要 jti 与 exp 才能算出黑名单 TTL，
    这两项只有解码后的载荷里有。
    """
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("缺少登录凭证")

    payload = decode_token(credentials.credentials, expected_type=TOKEN_TYPE_ACCESS)
    await auth_service.assert_not_revoked("登录凭证", payload)
    return payload


ActiveAccessToken = Annotated[dict[str, Any], Depends(require_active_access_token)]


async def require_active_fresh_user(
    token_payload: ActiveAccessToken,
    db: DbSession,
) -> User:
    """吊销校验 + 回表，供 auth 自身需要「令牌必须真的还活着」的接口使用。

    **为什么 ``/me`` 不能直接用 ``FreshCurrentUser``**：后者只做 JWT 解析与回表，
    不查黑名单，于是登出后的旧令牌仍能读到用户资料——「退出登录」形同虚设。
    实测中确实复现了这一点（logout 返回 204 后 /me 仍 200）。

    为什么不干脆给全局 ``get_current_user`` 加黑名单检查：那会让每个请求
    都多一次 Redis 往返，推翻「无状态令牌零 IO」的设计收益。这里只对
    语义上必须校验的接口付这个代价。业务接口的取舍见 auth_service.assert_not_revoked。
    """
    user = await db.get(User, UUID(str(token_payload["sub"])))
    if user is None:
        raise ResourceNotFoundError("用户不存在")
    if not user.is_active:
        raise AuthenticationError("账号已被停用")
    return user


ActiveFreshUser = Annotated[User, Depends(require_active_fresh_user)]


async def _role_name(db: AsyncSession, user: User) -> str:
    """取角色名用于响应展示。

    注册路径上新建的 User 尚未加载 role 关系（``lazy="selectin"`` 只在查询时生效，
    add/refresh 的对象不带），所以显式查一次而不是依赖懒加载。
    """
    name = (
        await db.execute(select(Role.name).where(Role.id == user.role_id))
    ).scalar_one_or_none()
    return str(name) if name else ""


# ==================== 端点 ====================


@router.post(
    "/register",
    response_model=AuthResult,
    status_code=status.HTTP_201_CREATED,
    summary="患者自助注册",
)
async def register(payload: RegisterRequest, db: DbSession) -> AuthResult:
    """注册即登录：直接返回令牌对，省掉「注册成功后再跳登录页」的一步。

    固定分配 patient 角色，其余角色须由管理员创建（见 auth_service.register）。
    """
    user = await auth_service.register(db, payload)
    tokens = await auth_service.issue_tokens_for(db, user)
    return AuthResult(**tokens, user=auth_service.to_user_out(user, await _role_name(db, user)))


@router.post("/login", response_model=AuthResult, summary="登录")
async def login(
    payload: LoginRequest,
    db: DbSession,
    client_ip: ClientIP,
) -> AuthResult:
    """账号支持手机号 / 邮箱 / 用户名三选一；失败次数超限会被临时锁定。

    ``client_ip`` 仅用于限流维度，经反代后取自 X-Forwarded-For，可被伪造，
    因此不作为安全判定依据（见 deps.get_client_ip 注释）。
    """
    user = await auth_service.authenticate(db, payload, client_ip)
    tokens = await auth_service.issue_tokens_for(db, user)
    return AuthResult(**tokens, user=auth_service.to_user_out(user, await _role_name(db, user)))


@router.post("/refresh", response_model=TokenPair, summary="刷新访问令牌")
async def refresh(payload: RefreshRequest, db: DbSession) -> TokenPair:
    """轮换式刷新：旧刷新令牌立即作废，防止长期有效的续期凭证被二次利用。"""
    tokens = await auth_service.refresh_tokens(db, payload.refresh_token)
    return TokenPair(**tokens)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, summary="退出登录")
async def logout(
    token_payload: ActiveAccessToken,
    body: LogoutRequest | None = None,
) -> None:
    """吊销当前访问令牌；请求体带上 refresh_token 时一并作废刷新令牌。

    用 204 而非 200：客户端此刻已不再持有有效凭证，返回任何业务负载都没有消费方。
    即使黑名单写入因 Redis 故障失败，本接口仍返回成功——不能让用户「退不出去」。
    """
    await auth_service.logout(token_payload, body.refresh_token if body else None)


@router.get("/me", response_model=UserOut, summary="当前用户信息")
async def me(current_user: ActiveFreshUser) -> UserOut:
    """回表取用户，因此能反映实时的 ``is_active`` 与完整档案字段。

    用 ``ActiveFreshUser`` 而非 ``FreshCurrentUser``：/me 是前端渲染头像、
    姓名、角色的数据来源，必须是真值；账号被停用时应立刻报错而不是继续展示旧资料。
    更关键的是它同时校验令牌未被吊销——否则登出后的旧令牌仍能读到用户资料，
    「退出登录」形同虚设（实测已复现该问题）。
    """
    return auth_service.to_user_out(current_user)


__all__ = ["router"]
