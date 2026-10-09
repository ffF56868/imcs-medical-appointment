"""IMCS 认证层端到端测试（内存 SQLite + httpx ASGI 传输）。

结论：**这套用例可以在 SQLite 上跑**，不需要 PostgreSQL。依据三点：
1. ``models/types.py`` 的 JSONBText / TextArray / TSVectorText 与
   ``models/knowledge.py`` 的 VectorType 都是 TypeDecorator，非 PG 方言自动退化为 TEXT；
2. 唯一真正不兼容的是 ``dialects.postgresql.UUID`` 缺少 SQLite 的 DDL 编译器，
   由 ``tests/imcs_sqlite.py`` 注册一个退化实现解决，只影响测试进程，不改生产模型；
3. 建表走「认证相关表白名单」，绕开日志表里的 ``Computed`` 生成列与
   ``BigInteger`` 自增主键 —— 它们在 SQLite 语义不同，但认证链路完全不涉及。

Redis 用假客户端替换：认证层对 Redis 是 fail-open 设计，因此既要覆盖
「计数生效并触发锁定」，也要覆盖「Redis 挂掉时仍能登录」。
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import StaticPool

from app.core.database import get_db
from app.imcs_main import app
from app.services import auth_service
from tests.imcs_sqlite import (
    build_session_factory,
    create_auth_schema,
    error_code_of,
    error_message_of,
    seed_roles,
)

PASSWORD = "Imcs2026demo"


# ==================== 假 Redis ====================


class FakeRedis:
    """最小可用的内存 Redis，只实现认证层用到的命令。

    不复现 Redis 的单线程原子性——测试关注键值行为与 TTL 判定，不是并发语义。
    """

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.expires: dict[str, float] = {}
        self.fail_mode = False

    def _alive(self, key: str) -> bool:
        expire_at = self.expires.get(key)
        if expire_at is not None and expire_at <= datetime.now(UTC).timestamp():
            self.store.pop(key, None)
            self.expires.pop(key, None)
            return False
        return key in self.store

    def _down(self) -> None:
        if self.fail_mode:
            from redis.exceptions import ConnectionError as RedisConnectionError

            raise RedisConnectionError("fake redis is down")

    async def get(self, key: str) -> str | None:
        self._down()
        return self.store.get(key) if self._alive(key) else None

    async def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None) -> bool:
        self._down()
        if nx and self._alive(key):
            return False
        self.store[key] = value
        if ex:
            self.expires[key] = datetime.now(UTC).timestamp() + ex
        return True

    async def delete(self, *keys: str) -> int:
        self._down()
        removed = 0
        for key in keys:
            if self.store.pop(key, None) is not None:
                removed += 1
            self.expires.pop(key, None)
        return removed

    async def exists(self, key: str) -> int:
        self._down()
        return 1 if self._alive(key) else 0

    async def incrby(self, key: str, amount: int) -> int:
        self._down()
        self._alive(key)
        current = int(self.store.get(key, "0")) + amount
        self.store[key] = str(current)
        return current

    async def ttl(self, key: str) -> int:
        self._down()
        expire_at = self.expires.get(key)
        if expire_at is None:
            # -2 键不存在 / -1 存在但无过期时间，与 Redis 返回值一致
            return -1 if key in self.store else -2
        return int(expire_at - datetime.now(UTC).timestamp())

    async def expire(self, key: str, seconds: int) -> bool:
        self._down()
        if key not in self.store:
            return False
        self.expires[key] = datetime.now(UTC).timestamp() + seconds
        return True


@pytest.fixture()
def fake_redis() -> Iterator[FakeRedis]:
    """替换认证服务持有的 cache_client。

    打补丁的目标是 ``auth_service.cache_client`` 而不是 ``core.cache.cache_client``：
    auth_service 用的是 ``from ... import`` ，持有的是导入时刻绑定的模块属性引用，
    改源模块属性不会影响已导入的名字。
    """
    client = FakeRedis()
    original = auth_service.cache_client
    auth_service.cache_client = client
    try:
        yield client
    finally:
        auth_service.cache_client = original


# ==================== 数据库 ====================


@pytest.fixture()
def engine():
    """内存 SQLite 引擎 + 认证表 + 角色种子。

    StaticPool 让所有会话共用同一条连接：``sqlite://`` 内存库的生命周期绑定在
    连接上，换连接等于换了个空库。
    """
    test_engine = sa.create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with test_engine.begin() as conn:
        create_auth_schema(conn)
        seed_roles(conn)
    yield test_engine
    test_engine.dispose()


@pytest.fixture()
async def client(engine, fake_redis) -> AsyncIterator[AsyncClient]:
    """指向被测应用的异步客户端，并把 ``get_db`` 重定向到内存库。

    用 dependency_overrides 而非改 settings：core.database 在 import 时就建好了
    PG 引擎，改配置只能改代码；覆盖依赖则完全绕开它。
    会话工厂用 ``AsyncEngine(sync_engine)`` 包装同步引擎，业务代码拿到的仍是
    标准 AsyncSession，await 语义天然成立。
    """
    session_factory = build_session_factory(engine)

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client

    app.dependency_overrides.clear()


@pytest.fixture()
def db_read(engine):
    """以同步方式直读库，用于断言落库结果。

    刻意绕过 ORM：断言要验证的是「数据真的写进去了」，
    用另一条路径读比复用被测代码的同一套映射更有说服力。
    """

    def read(sql: str, **params) -> list[tuple[Any, ...]]:  # noqa: ANN001
        with engine.connect() as conn:
            return conn.execute(sa.text(sql), params).fetchall()

    return read


# ==================== 工具 ====================


def register_payload(**overrides) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "username": "zhangsan",
        "real_name": "张三",
        "account": "13800001111",
        "password": PASSWORD,
    }
    payload.update(overrides)
    return payload


async def register_user(client: AsyncClient, **overrides) -> dict[str, Any]:
    response = await client.post("/api/v1/auth/register", json=register_payload(**overrides))
    assert response.status_code == 201, response.text
    return response.json()


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ==================== 静态冒烟（不碰数据库）====================


def test_routes_mounted() -> None:
    """确认路由挂载完整。即使建表方案在某环境不成立，也能在收集阶段暴露 import 错误。"""
    paths = {route.path for route in app.routes}
    for expected in (
        "/api/v1/auth/register",
        "/api/v1/auth/login",
        "/api/v1/auth/refresh",
        "/api/v1/auth/logout",
        "/api/v1/auth/me",
        "/health",
    ):
        assert expected in paths, f"缺少路由 {expected}"


def test_cors_origins_split() -> None:
    """逗号分隔配置应拆成列表，且尾随逗号不产生空 origin。"""
    from app.imcs_main import cors_origins

    origins = cors_origins()
    assert isinstance(origins, list)
    assert all(origin for origin in origins)


# ==================== 注册 ====================


@pytest.mark.asyncio
async def test_register_success(client: AsyncClient) -> None:
    data = await register_user(client)

    assert data["token_type"] == "bearer"
    assert data["access_token"] and data["refresh_token"]
    # expires_in 来自 get_token_expiry_seconds()，随配置变化，不能写死常量
    assert data["expires_in"] > 0
    assert data["user"]["role_name"] == "patient"
    assert data["user"]["username"] == "zhangsan"
    # 手机号入库被归一化成 +86 前缀，避免同一号码因输入格式不同产生多个账号
    assert data["user"]["phone"] == "+8613800001111"
    # 响应绝不能带出密码哈希
    assert "password_hash" not in data["user"]


@pytest.mark.asyncio
async def test_register_duplicate_username(client: AsyncClient) -> None:
    await register_user(client)
    response = await client.post(
        "/api/v1/auth/register",
        json=register_payload(account="13800002222"),
    )
    assert response.status_code == 409
    assert "用户名" in error_message_of(response)


@pytest.mark.asyncio
async def test_register_duplicate_phone_normalized(client: AsyncClient) -> None:
    await register_user(client)
    # 同一号码的不同写法必须撞到同一个账号
    response = await client.post(
        "/api/v1/auth/register",
        json=register_payload(username="lisi", account="+86 138-0000-1111"),
    )
    assert response.status_code == 409
    assert "手机号" in error_message_of(response)


@pytest.mark.asyncio
async def test_register_weak_password(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register",
        json=register_payload(password="abcdefgh"),  # 有字母无数字
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_short_password(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register",
        json=register_payload(password="Ab1"),
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_invalid_account(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register",
        json=register_payload(account="not-a-phone-or-email"),
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_forced_patient_role(client: AsyncClient, db_read) -> None:
    """请求体里没有 role 字段可填；即使伪造也被 Pydantic 忽略，落库必为 patient。"""
    await client.post(
        "/api/v1/auth/register",
        json={**register_payload(), "role": "admin", "role_id": 6},
    )
    rows = db_read("SELECT u.role_id, r.name FROM users u JOIN roles r ON r.id = u.role_id")
    assert rows == [(1, "patient")]


@pytest.mark.asyncio
async def test_register_creates_patient_profile(client: AsyncClient, db_read) -> None:
    """注册必须同步建档：挂号/就诊/处方/收费的外键都指向 patient_profiles.id。"""
    await register_user(client, gender="f", birth_date="1990-05-06")

    rows = db_read(
        "SELECT pp.gender, pp.birth_date FROM patient_profiles pp "
        "JOIN users u ON u.id = pp.user_id WHERE u.username = :username",
        username="zhangsan",
    )
    assert len(rows) == 1, "注册后应存在且仅存在一份患者档案"
    assert rows[0].gender == "F"


# ==================== 登录 ====================


@pytest.mark.asyncio
async def test_login_by_username_and_phone(client: AsyncClient) -> None:
    await register_user(client)

    for account in ("zhangsan", "13800001111", "+8613800001111"):
        response = await client.post(
            "/api/v1/auth/login",
            json={"account": account, "password": PASSWORD},
        )
        assert response.status_code == 200, f"{account} 应能登录：{response.text}"
        assert response.json()["user"]["role_name"] == "patient"


@pytest.mark.asyncio
async def test_login_wrong_password_counts_failure(client: AsyncClient, fake_redis: FakeRedis) -> None:
    await register_user(client)
    response = await client.post(
        "/api/v1/auth/login",
        json={"account": "zhangsan", "password": "Wrongpass1"},
    )
    assert response.status_code == 401
    assert error_code_of(response) == "auth_invalid_credentials"
    assert fake_redis.store.get(f"{auth_service.KEY_FAIL_PREFIX}zhangsan") == "1"


@pytest.mark.asyncio
async def test_login_unknown_account_does_not_leak_existence(client: AsyncClient) -> None:
    """账号不存在与密码错误必须返回同样的信息，否则可枚举已注册手机号。"""
    await register_user(client)
    unknown = await client.post(
        "/api/v1/auth/login",
        json={"account": "nobody", "password": "Wrongpass1"},
    )
    wrong = await client.post(
        "/api/v1/auth/login",
        json={"account": "zhangsan", "password": "Wrongpass1"},
    )
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json()


@pytest.mark.asyncio
async def test_login_lockout_after_max_failures(
    client: AsyncClient,
    monkeypatch,
    fake_redis: FakeRedis,
) -> None:
    """连续失败达到阈值后返回 423，且锁定期内正确密码也进不去。"""
    from app.core.config import settings

    # 真实阈值 5 次要跑 5 轮 scrypt(n=16384)，单轮约 50-100ms；调小只为跑得动
    monkeypatch.setattr(settings, "max_login_failures", 2)
    monkeypatch.setattr(settings, "login_rate_limit_per_minute", 50)

    await register_user(client)
    last = None
    for _ in range(2):
        last = await client.post(
            "/api/v1/auth/login",
            json={"account": "zhangsan", "password": "Wrongpass1"},
        )
    assert last is not None and last.status_code == 423
    assert error_code_of(last) == "auth_account_locked"

    locked = await client.post(
        "/api/v1/auth/login",
        json={"account": "zhangsan", "password": PASSWORD},
    )
    assert locked.status_code == 423


@pytest.mark.asyncio
async def test_successful_login_clears_failure_count(client: AsyncClient, fake_redis: FakeRedis) -> None:
    await register_user(client)
    await client.post("/api/v1/auth/login", json={"account": "zhangsan", "password": "Wrongpass1"})
    ok = await client.post("/api/v1/auth/login", json={"account": "zhangsan", "password": PASSWORD})
    assert ok.status_code == 200
    assert fake_redis.store.get(f"{auth_service.KEY_FAIL_PREFIX}zhangsan") is None


@pytest.mark.asyncio
async def test_login_rate_limit_triggers_429(client: AsyncClient, monkeypatch) -> None:
    """固定窗口限流：超过每分钟阈值返回 429。"""
    from app.core.config import settings

    monkeypatch.setattr(settings, "login_rate_limit_per_minute", 3)
    monkeypatch.setattr(settings, "max_login_failures", 20)

    await register_user(client)
    statuses = [
        (
            await client.post(
                "/api/v1/auth/login",
                json={"account": "zhangsan", "password": PASSWORD},
            )
        ).status_code
        for _ in range(8)
    ]
    assert 429 in statuses, f"应触发限流，实际状态码={statuses}"


@pytest.mark.asyncio
async def test_login_fails_open_when_redis_down(client: AsyncClient, fake_redis: FakeRedis) -> None:
    """Redis 故障不得阻断登录：限流与锁定是辅助手段，不该成为单点故障源。"""
    await register_user(client)
    fake_redis.fail_mode = True
    response = await client.post(
        "/api/v1/auth/login",
        json={"account": "zhangsan", "password": PASSWORD},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_inactive_account_rejected(client: AsyncClient, engine) -> None:
    await register_user(client)
    with engine.begin() as conn:
        conn.execute(sa.text("UPDATE users SET is_active = 0 WHERE username = 'zhangsan'"))
    response = await client.post(
        "/api/v1/auth/login",
        json={"account": "zhangsan", "password": PASSWORD},
    )
    assert response.status_code == 401
    assert "停用" in error_message_of(response)


# ==================== 刷新轮换 ====================


@pytest.mark.asyncio
async def test_refresh_rotates_tokens(client: AsyncClient) -> None:
    data = await register_user(client)
    old_refresh = data["refresh_token"]

    response = await client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert response.status_code == 200
    rotated = response.json()
    assert rotated["refresh_token"] != old_refresh, "刷新必须发新令牌"
    assert rotated["access_token"]

    # 旧刷新令牌立即作废：泄漏窗口的价值被压到单次使用
    replay = await client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert replay.status_code == 401


@pytest.mark.asyncio
async def test_refresh_rejects_access_token(client: AsyncClient) -> None:
    """类型混淆防护：access token 不能当 refresh token 用。"""
    data = await register_user(client)
    response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": data["access_token"]},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_refresh_survives_redis_outage(client: AsyncClient, fake_redis: FakeRedis) -> None:
    """黑名单读不到时按未吊销放行（fail-open），不能因为缓存故障把所有人踢下线。"""
    data = await register_user(client)
    fake_redis.fail_mode = True
    response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": data["refresh_token"]},
    )
    assert response.status_code == 200


# ==================== 登出吊销 ====================


@pytest.mark.asyncio
async def test_logout_revokes_access_token(client: AsyncClient) -> None:
    data = await register_user(client)
    headers = bearer(data["access_token"])

    assert (await client.get("/api/v1/auth/me", headers=headers)).status_code == 200

    logout = await client.post("/api/v1/auth/logout", headers=headers)
    assert logout.status_code == 204

    after = await client.get("/api/v1/auth/me", headers=headers)
    assert after.status_code == 401, "登出后旧 access token 应立即失效"


@pytest.mark.asyncio
async def test_logout_revokes_refresh_token(client: AsyncClient) -> None:
    data = await register_user(client)
    await client.post(
        "/api/v1/auth/logout",
        headers=bearer(data["access_token"]),
        json={"refresh_token": data["refresh_token"]},
    )
    # 只登出 access token 而留着 refresh token，客户端下次刷新又能拿到凭证，等于没退干净
    refreshed = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": data["refresh_token"]},
    )
    assert refreshed.status_code == 401


@pytest.mark.asyncio
async def test_logout_blacklist_key_ttl_matches_token_life(client: AsyncClient, fake_redis: FakeRedis) -> None:
    """黑名单键的 TTL 应等于令牌剩余寿命，令牌自然过期后键自动消失，无需清理任务。"""
    data = await register_user(client)
    await client.post("/api/v1/auth/logout", headers=bearer(data["access_token"]))

    keys = [k for k in fake_redis.store if k.startswith(auth_service.KEY_BLACKLIST)]
    assert len(keys) == 1, f"应恰好写入一条黑名单，实际={keys}"
    ttl = fake_redis.expires[keys[0]] - datetime.now(UTC).timestamp()
    assert 0 < ttl <= 60 * 60 * 24 * 7


@pytest.mark.asyncio
async def test_logout_succeeds_when_redis_down(client: AsyncClient, fake_redis: FakeRedis) -> None:
    """Redis 挂了也要让登出成功：报错拒绝登出会把用户卡在「退不出去」的状态。"""
    data = await register_user(client)
    fake_redis.fail_mode = True
    response = await client.post("/api/v1/auth/logout", headers=bearer(data["access_token"]))
    assert response.status_code == 204


# ==================== /me ====================


@pytest.mark.asyncio
async def test_me_requires_auth(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/auth/me")).status_code == 401


@pytest.mark.asyncio
async def test_me_rejects_garbage_token(client: AsyncClient) -> None:
    response = await client.get("/api/v1/auth/me", headers=bearer("not.a.jwt"))
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_reflects_realtime_deactivation(client: AsyncClient, engine) -> None:
    """/me 走 FreshCurrentUser 回表，因此管理员停用账号后能立刻反映。

    轻量版 get_current_user 只看 JWT，感知不到实时的 is_active —— 这正是
    deps.py 注释里说的取舍，本用例把它钉下来。
    """
    data = await register_user(client)
    headers = bearer(data["access_token"])

    with engine.begin() as conn:
        conn.execute(sa.text("UPDATE users SET is_active = 0 WHERE username = 'zhangsan'"))

    response = await client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_returns_full_profile(client: AsyncClient) -> None:
    data = await register_user(client)
    response = await client.get("/api/v1/auth/me", headers=bearer(data["access_token"]))
    assert response.status_code == 200
    body = response.json()
    assert body["real_name"] == "张三"
    assert body["id"] == data["user"]["id"]
    assert uuid.UUID(body["id"])  # id 必须是合法 UUID 字符串


# ==================== 健康检查 ====================


@pytest.mark.asyncio
async def test_health_reports_degraded_not_5xx_when_redis_down(client: AsyncClient, fake_redis: FakeRedis) -> None:
    """Redis 不可用只报 degraded，绝不返回 5xx。

    compose 的 api.healthcheck 与 Nginx 探活只看状态码，503 会触发容器重启，
    把一个性能降级放大成服务中断。
    """
    fake_redis.fail_mode = True
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["cache"] == "unavailable"
    assert body["status"] in {"degraded", "ok"}


@pytest.mark.asyncio
async def test_health_database_field_present(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["database"] in {"connected", "unavailable"}
    assert body["version"]


# ==================== 安全响应头 ====================


@pytest.mark.asyncio
async def test_auth_paths_are_no_store(client: AsyncClient) -> None:
    """令牌属敏感凭证，/auth/ 响应禁止任何缓存。"""
    response = await client.post(
        "/api/v1/auth/login",
        json={"account": "nobody", "password": "Wrongpass1"},
    )
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


@pytest.mark.asyncio
async def test_non_auth_paths_have_security_headers(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "Cache-Control" not in response.headers


# ==================== 时间助手自检 ====================


def test_time_helper_sanity() -> None:
    """FakeRedis 的 TTL 判定依赖 UTC aware datetime，先自证可用。"""
    now = datetime.now(UTC)
    delta = (now + timedelta(seconds=60)).timestamp() - now.timestamp()
    assert delta == pytest.approx(60, abs=1)
