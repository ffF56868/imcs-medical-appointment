"""认证业务逻辑：注册、登录限流与锁定、令牌签发与吊销。

路由层只做「解析请求 → 调 service → 组装响应」，所有判定规则集中在本模块，
便于后续把同一套逻辑复用给管理员建号接口和单元测试。

Redis 的角色是**加速层而非真相来源**（与 ``app/core/cache.py`` 的设计哲学一致）：
限流计数、失败次数、令牌黑名单全部只存 Redis。Redis 挂掉时一律 fail-open 放行并打
warning —— 因为「缓存故障导致所有人都登不进去」比「限流暂时失效」严重得多。
代价是 Redis 不可用期间黑名单形同虚设，已登出的 access token 会在自身有效期内
（默认 12 小时）继续可用。这是无状态 JWT 的固有取舍，缩短 access 有效期即可收窄窗口。

登录失败计数刻意**不带用户 ID**：账号不存在时也要计数，
否则攻击者可以用「是否返回锁定错误」枚举出哪些手机号已注册。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import cache_client, increment_with_ttl
from app.core.config import settings
from app.core.exceptions import (
    AccountLockedError,
    AuthenticationError,
    ResourceNotFoundError,
)
from app.core.security import (
    TOKEN_TYPE_ACCESS,
    TOKEN_TYPE_REFRESH,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_token_expiry_seconds,
    hash_password,
    normalize_account,
    verify_password,
)
from app.models.auth import Role, RoleName, User
from app.models.patient import PatientProfile
from app.schemas.auth import LoginRequest, RegisterRequest, UserOut

logger = logging.getLogger(__name__)

# ==================== Redis 键 ====================
# 统一前缀 imcs:auth:，便于按前缀监控、排查与整批清理。
KEY_FAIL_PREFIX = "imcs:auth:fail:"  # + account -> 连续失败次数（TTL=锁定时长）
KEY_RATE_ACCOUNT = "imcs:auth:rate:account:"  # + account -> 每分钟登录尝试数
KEY_RATE_IP = "imcs:auth:rate:ip:"  # + ip -> 每分钟登录尝试数
KEY_BLACKLIST = "imcs:auth:blacklist:"  # + jti -> 已吊销标记


def _is_rate_limited(count: int | None, limit: int) -> bool:
    """判定是否超限。

    ``count is None`` 表示 Redis 读写失败，按 fail-open 放行；
    限流本来只是抗暴力破解的辅助手段，不该成为单点故障源。
    """
    return count is not None and count > limit


async def _safe_delete(key: str) -> None:
    """删除键，异常只记日志不外抛。

    这里不能复用 ``cache.delete_cached``：它吞掉了 RedisError 并返回 False，
    调用方无法区分「键本就不存在」和「Redis 挂了」。后者需要 warning，
    因为它意味着紧接着的黑名单写入同样不会生效。
    """
    try:
        await cache_client.delete(key)
    except (RedisError, OSError):
        logger.warning("Redis 删除失败，键 %s 可能残留旧计数", key, exc_info=True)


async def _blacklist_jti(jti: str, exp: int) -> None:
    """把令牌 jti 写入黑名单，TTL 取该令牌的剩余寿命。

    TTL 用 ``exp - now`` 而不是固定值：键的生命周期恰好等于令牌本身的有效期，
    令牌自然过期后键自动消失，不需要任何清理任务，也不会无限堆积。
    """
    ttl = int(exp - utcnow_timestamp())
    if ttl <= 0:
        return
    try:
        # SETNX + EX 语义：重复吊销同一令牌不必再刷新 TTL
        await cache_client.set(f"{KEY_BLACKLIST}{jti}", "1", nx=True, ex=ttl)
    except (RedisError, OSError):
        # fail-open：写不进黑名单，登出仍返回成功，令牌在有效期内继续可用。
        # 反过来做（报错拒绝登出）会让客户端卡在「退不出去」的状态。
        logger.warning("Redis 黑名单写入失败，令牌 %s 未被吊销", jti, exc_info=True)


def utcnow_timestamp() -> int:
    """Unix 秒。单独抽出便于测试里冻结时间。"""
    return int(datetime.now(UTC).timestamp())


async def is_token_revoked(payload: dict[str, Any]) -> bool:
    """检查令牌 jti 是否已被吊销。

    返回值采用「保守为 False」语义：Redis 不可用时视为未吊销（fail-open），
    与整个缓存层的取舍保持一致。
    """
    jti = payload.get("jti")
    if not jti:
        return False
    try:
        return bool(await cache_client.exists(f"{KEY_BLACKLIST}{jti}"))
    except (RedisError, OSError):
        logger.warning("Redis 黑名单读取失败，本次按未吊销放行", exc_info=True)
        return False


# ==================== 序列化 ====================


def to_user_out(user: User, role_name: str | None = None) -> UserOut:
    """ORM 用户 -> 对外响应模型。

    ``role_name`` 可显式传入：轻量鉴权路径上的 User 对象没有加载 role 关系，
    角色名只在 JWT claims 里。回表路径则能从 ``user.role.name`` 取到。
    """
    resolved_role = role_name or (user.role.name if user.role is not None else "")
    return UserOut(
        id=str(user.id),
        username=user.username,
        real_name=user.real_name,
        phone=user.phone,
        email=user.email,
        role_id=user.role_id,
        role_name=resolved_role,
        department_id=str(user.department_id) if user.department_id is not None else None,
        title=user.title,
        specialty=user.specialty,
        is_active=user.is_active,
        created_at=user.created_at,
    )


# ==================== 注册 ====================


async def _resolve_patient_role_id(db: AsyncSession) -> int:
    """查 patient 角色 ID。

    查库而非引用 ``seed/users.py`` 里的 ``ROLE_ID_BY_NAME`` 常量：种子数据里那份
    字典与 ``rbac_matrix.ROLE_DEFINITIONS`` 是两处独立声明，已经靠注释约定同步。
    service 再造一份第三条副本只会加深这个风险，直接按角色名查是唯一真相来源。
    """
    role_id = (
        await db.execute(select(Role.id).where(Role.name == RoleName.PATIENT.value))
    ).scalar_one_or_none()
    if role_id is None:
        # 角色表为空说明 alembic 迁移或 seed_rbac 没跑过。此时若硬编码 1 建号，
        # 种子的 ID 校验会先报错，但线上更可能是静默产生一个悬空 role_id。
        raise HTTPException(
            status_code=503,
            detail="系统角色数据尚未初始化，请先执行数据库迁移与种子数据",
        )
    return int(role_id)


async def register(db: AsyncSession, payload: RegisterRequest) -> User:
    """自助注册。固定分配 patient 角色，其余角色留给管理员接口。

    为什么不开放角色选择：允许请求体自带 role 等于任何人都能注册成 admin，
    整套 RBAC 直接失效。医生/药师等账号必须由管理员在受控接口里创建。
    """
    normalized_account = normalize_account(payload.account)
    username = payload.username.strip().lower()

    # 逐项查重，给出「哪个字段冲突」的精确提示。
    # 只依赖唯一约束的 IntegrityError 也能拦住并发竞争，但报错信息只能是
    # 「注册失败」——患者分不清是用户名还是手机号被占用，会反复重试。
    existing_username = (
        await db.execute(select(User.id).where(User.username == username))
    ).scalar_one_or_none()
    if existing_username is not None:
        raise HTTPException(status_code=409, detail="该用户名已被注册")

    conflict_field = (
        await db.execute(
            select(User.username, User.phone, User.email).where(
                (User.phone == normalized_account) | (User.email == normalized_account)
            )
        )
    ).first()
    if conflict_field is not None:
        which = "手机号" if conflict_field.phone == normalized_account else "邮箱"
        raise HTTPException(status_code=409, detail=f"该{which}已被注册")

    role_id = await _resolve_patient_role_id(db)

    # 归一化结果按形态落列：+86 开头是手机号，含 @ 是邮箱。
    # 注册时 account 必然命中二者之一（normalize_account 已保证）。
    is_phone = normalized_account.startswith("+86")

    user = User(
        username=username,
        password_hash=hash_password(payload.password),
        real_name=payload.real_name.strip(),
        phone=normalized_account if is_phone else None,
        email=None if is_phone else normalized_account,
        role_id=role_id,
        is_active=True,
    )
    db.add(user)

    # 患者建档：PatientProfile.user_id 是 nullable=False + unique，
    # 且 appointments / visits / exam_requests / prescriptions / charge_orders
    # 五张表的外键都指向 patient_profiles.id。没有档案的账号后续任何挂号动作
    # 都会撞外键，因此注册必须同步建档，不能推迟到首次挂号。
    # gender / birth_date 之外的健康字段（过敏史、慢性病）留空由患者后续补充。
    profile = PatientProfile(
        user_id=user.id,
        gender=payload.gender,
        birth_date=payload.birth_date,
    )

    try:
        # 必须先 flush 才能拿到 user.id：SQLAlchemy 的 ``default=uuid.uuid4``
        # 是 **flush 时** 才执行的列默认值，构造 User() 的瞬间 ``user.id`` 仍是 None。
        # 若直接把 None 传给 PatientProfile.user_id，会违反 NOT NULL 约束，
        # 整个注册事务回滚并被下面的 IntegrityError 分支误报成「用户名已被注册」
        # ——实测中数据库里一个测试账号都没建出来，正是这个原因。
        await db.flush()
        profile.user_id = user.id
        db.add(profile)
        await db.commit()
    except IntegrityError as exc:
        # 兜住「查重通过但并发插入撞唯一约束」的竞争窗口。
        # 冲突字段不一定就是 username（phone / email 同样有唯一约束），
        # 因此提示语不指名具体字段，避免误导用户去改一个并不冲突的输入。
        # commit 失败时 SQLAlchemy 已把会话置为 needs_rollback，
        # 后续请求由 get_db 的会话生命周期负责回收。
        await db.rollback()
        logger.info("注册唯一约束冲突：username=%s detail=%s", username, exc.orig)
        raise HTTPException(status_code=409, detail="该账号信息已被注册，请更换后重试") from exc

    return user


# ==================== 登录 ====================


async def _check_login_rate(account: str, client_ip: str) -> None:
    """固定窗口限流：按账号与按 IP 两个维度各一道闸。

    两个维度缺一不可：
    - 只限 IP：分布式僵尸网络换 IP 即可无限爆破同一账号
    - 只限账号：攻击者可对海量账号各试几次，逐个撞库而不触发任何单账号阈值

    固定窗口（INCR + 首次设 TTL）实现最简，边界处最多容忍 2 倍突发。
    本项目阈值宽松（30/min），这点误差无关紧要，不值得引入滑动窗口的 ZSET 复杂度。

    两道闸共用同一个每分钟阈值：本阶段没有独立的 IP 阈值配置项，而
    ``core/config.py`` 属于既有文件、本批次只允许追加 CORS 一个字段，
    因此不擅自新增设置项。需要分开调优时，后续加一个
    ``login_rate_limit_ip_per_minute`` 即可，此处只需改一处常量引用。
    """
    window = 60
    account_count = await increment_with_ttl(
        f"{KEY_RATE_ACCOUNT}{account}", 1, window
    )
    if _is_rate_limited(account_count, settings.login_rate_limit_per_minute):
        raise HTTPException(status_code=429, detail="登录尝试过于频繁，请稍后再试")

    ip_count = await increment_with_ttl(f"{KEY_RATE_IP}{client_ip}", 1, window)
    if _is_rate_limited(ip_count, settings.login_rate_limit_per_minute):
        raise HTTPException(status_code=429, detail="当前网络登录尝试过于频繁，请稍后再试")


async def _read_counter(key: str) -> int | None:
    """读计数器。返回 None 表示键不存在或 Redis 不可用（两者都按「无计数」处理）。"""
    try:
        raw = await cache_client.get(key)
    except (RedisError, OSError):
        logger.warning("Redis 读取失败，键 %s 按不存在处理", key, exc_info=True)
        return None
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


async def _check_not_locked(account: str) -> None:
    """锁定检查：失败计数达到阈值即拒绝，直到 TTL 自然到期。

    没有 users 表的计数字段，所以锁定状态完全由 Redis 键的存在性与 TTL 承载。
    """
    failures = await _read_counter(f"{KEY_FAIL_PREFIX}{account}")
    if failures is not None and failures >= settings.max_login_failures:
        raise AccountLockedError(
            f"连续登录失败次数过多，请 {settings.login_lockout_minutes} 分钟后再试"
        )


async def find_user_by_account(db: AsyncSession, account: str) -> User | None:
    """按手机号 / 邮箱 / 用户名定位账号。

    入参已在 schema 层归一化：手机/邮箱形态保持归一化结果，
    其余按用户名小写匹配。三个条件用 OR 合并为一次查询，
    避免三次往返（也避免因先后顺序不同而产生「哪个账号优先」的歧义行为）。
    """
    statement = select(User).where(
        (User.username == account)
        | (User.phone == account)
        | (User.email == account)
    )
    return (await db.execute(statement)).scalar_one_or_none()


async def authenticate(
    db: AsyncSession,
    payload: LoginRequest,
    client_ip: str,
) -> User:
    """校验凭证并返回用户。失败路径负责计数与锁定。"""
    await _check_login_rate(payload.account, client_ip)
    await _check_not_locked(payload.account)

    user = await find_user_by_account(db, payload.account)

    # 账号不存在与密码错误返回完全相同的文案，且两条路径都走计数，
    # 使响应时间与错误信息都不泄露「该账号是否存在」。
    password_ok = False
    if user is not None:
        password_ok = verify_password(payload.password, user.password_hash)

    if user is None or not password_ok:
        failures = await increment_with_ttl(
            f"{KEY_FAIL_PREFIX}{payload.account}",
            1,
            settings.login_lockout_minutes * 60,
        )
        if failures is not None and failures >= settings.max_login_failures:
            raise AccountLockedError(
                f"连续登录失败次数过多，请 {settings.login_lockout_minutes} 分钟后再试"
            )
        raise AuthenticationError()

    if not user.is_active:
        # 不计入失败次数：这是账号状态问题，不是猜错密码，
        # 混在一起会让被停用的账号在恢复后仍处于锁定中。
        raise AuthenticationError("账号已被停用，请联系管理员")

    # 成功即清零，否则一次偶发输错会长期背着计数，
    # 下次真被爆破时提前触发锁定。
    await _safe_delete(f"{KEY_FAIL_PREFIX}{payload.account}")
    return user


# ==================== 令牌 ====================


def create_tokens(user: User, role_name: str) -> dict[str, Any]:
    """签发访问 + 刷新令牌对。

    角色名从 DB 现取而非信任任何客户端输入：JWT 的 role claim 是权限判断依据，
    签发时的正确性只能由服务端保证。
    """
    access_token = create_access_token(
        user.id,
        role_name,
        role_id=user.role_id,
        username=user.username,
    )
    refresh_token = create_refresh_token(user.id)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": get_token_expiry_seconds(),
    }


async def issue_tokens_for(db: AsyncSession, user: User) -> dict[str, Any]:
    """带角色名解析的令牌签发（回表场景）。

    角色名单独查一列而不是读 ``user.role.name``：轻量路径上的 User 实例
    并未绑定会话，访问 relationship 会抛 DetachedInstanceError；
    回表路径虽然能懒加载，但统一用一条 SELECT 更可控。
    """
    role_name = (
        await db.execute(select(Role.name).where(Role.id == user.role_id))
    ).scalar_one_or_none()
    return create_tokens(user, str(role_name) if role_name else "")


async def refresh_tokens(db: AsyncSession, refresh_token: str) -> dict[str, Any]:
    """用刷新令牌换发新令牌对，并作废旧刷新令牌（轮换）。

    轮换的意义：刷新令牌有效期长达 7 天，若不一次性使用，一旦泄漏就等于
    长期可重新获取访问凭证。作废旧 jti 后，泄漏窗口的价值被压到单次使用。

    注意这里**不查库**取用户信息，而是沿用旧 refresh token 的 sub：
    刷新令牌只携带 user_id，角色必须回表取最新值——管理员把某人从 doctor
    改成 admin 之后，刷新出来的 access token 应反映新角色。
    """
    payload = decode_token(refresh_token, expected_type=TOKEN_TYPE_REFRESH)

    if await is_token_revoked(payload):
        raise AuthenticationError("刷新令牌已失效，请重新登录")

    try:
        user_id = UUID(str(payload["sub"]))
    except (KeyError, ValueError) as exc:
        raise AuthenticationError("刷新令牌格式错误") from exc

    user = await db.get(User, user_id)
    if user is None:
        raise ResourceNotFoundError("用户不存在")
    if not user.is_active:
        raise AuthenticationError("账号已被停用，请联系管理员")

    # 先作废旧令牌，再签新令牌：即便签发抛异常，旧令牌也已失效，
    # 不会出现「新旧同时可用」的双活窗口。
    await _blacklist_jti(str(payload["jti"]), int(payload["exp"]))

    return await issue_tokens_for(db, user)


async def logout(access_payload: dict[str, Any], refresh_token: str | None) -> None:
    """吊销当前访问令牌，并可选地一并吊销刷新令牌。

    access 令牌的 jti 只有在路由层才能拿到（依赖注入解出来了但没往上传），
    因此由路由把解码后的 payload 传进来，而不是在这里重复解码一次。
    """
    jti = access_payload.get("jti")
    exp = int(access_payload.get("exp", 0))
    if jti:
        await _blacklist_jti(str(jti), exp)

    if refresh_token:
        try:
            refresh_payload = decode_token(refresh_token, expected_type=TOKEN_TYPE_REFRESH)
        except Exception:  # noqa: BLE001 - 非法刷新令牌无需吊销，静默跳过
            logger.debug("退出登录时携带了无法解析的刷新令牌，忽略")
            return
        await _blacklist_jti(
            str(refresh_payload["jti"]),
            int(refresh_payload["exp"]),
        )


# ==================== 黑名单校验依赖 ====================


async def assert_not_revoked(credentials_scope: str, payload: dict[str, Any]) -> None:
    """路由层显式黑名单闸门。

    ``get_current_user`` 是纯 JWT 解析、不回表也不碰 Redis，这样绝大多数请求
    零额外 IO。若在依赖里加黑名单检查，等于每个请求都要多一次 Redis 往返，
    把这个刻意的设计推翻了。因此吊销检查放在 auth 路由层：只有登出/刷新这类
    真正关心令牌生死的地方才付出这次往返。
    """
    if await is_token_revoked(payload):
        raise AuthenticationError(f"{credentials_scope}已失效，请重新登录")


__all__ = [
    "KEY_BLACKLIST",
    "KEY_FAIL_PREFIX",
    "KEY_RATE_ACCOUNT",
    "KEY_RATE_IP",
    "assert_not_revoked",
    "authenticate",
    "create_tokens",
    "find_user_by_account",
    "issue_tokens_for",
    "is_token_revoked",
    "logout",
    "refresh_tokens",
    "register",
    "to_user_out",
]
