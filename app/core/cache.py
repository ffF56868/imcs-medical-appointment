"""异步 Redis 客户端与缓存工具。

Redis 在本系统里是**加速层与计数器，不是真相来源**：
- Token 日配额计数：Redis 不可用时退化为「放行但记日志」，不能因为缓存挂了
  就拒绝所有 AI 请求——这是 fail-open 而非 fail-closed 的取舍
- 登录限流：同样 fail-open，限流失效的影响远小于把用户锁在门外
- 语义缓存：不可用时直接穿透到 LLM

旧项目 ``app/cache.py`` 用的是同步 ``redis.Redis``，在 async 路由里会阻塞事件循环。
这里改用 ``redis.asyncio``，旧模块保留给 legacy 代码使用。
"""

from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import settings

# decode_responses=True：本系统的键值都是文本（JSON、计数器），
# 不需要处理二进制。向量数据走 PostgreSQL，不经过 Redis。
redis_client: Redis = Redis.from_url(
    settings.redis_url,
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1,
    health_check_interval=30,
)

# 语义缓存与限流用独立的库，避免与 Celery broker（db0）、result backend（db1）
# 的键空间混在一起，便于按库清空与监控。
# db= 参数会覆盖 URL 里解析出的库号。
CACHE_DB_INDEX = 2
cache_client: Redis = Redis.from_url(
    settings.redis_url,
    db=CACHE_DB_INDEX,
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1,
    health_check_interval=30,
)


async def check_redis() -> bool:
    """健康检查。失败不抛异常，返回 False 让 /health 报 degraded。"""
    try:
        return bool(await redis_client.ping())
    except (RedisError, OSError):
        return False


async def get_cached_json(key: str) -> Any | None:
    try:
        value = await cache_client.get(key)
        return json.loads(value) if value is not None else None
    except (RedisError, TypeError, ValueError, OSError):
        return None


async def set_cached_json(key: str, value: Any, ttl_seconds: int) -> bool:
    try:
        await cache_client.setex(
            key,
            max(1, ttl_seconds),
            json.dumps(value, ensure_ascii=False, default=str),
        )
        return True
    except (RedisError, TypeError, ValueError, OSError):
        return False


async def delete_cached(key: str) -> bool:
    try:
        await cache_client.delete(key)
        return True
    except (RedisError, OSError):
        return False


async def increment_with_ttl(
    key: str,
    amount: int,
    ttl_seconds: int,
) -> int | None:
    """原子自增并在首次创建时设置过期时间。

    用 pipeline 把 INCRBY 与 EXPIRE 合并为一次往返。
    返回自增后的值，Redis 不可用时返回 None（调用方按 fail-open 处理）。

    注意这里存在一个理论上的竞态：INCRBY 成功但 EXPIRE 未执行时进程崩溃，
    键会永不过期。用 ``ttl`` 命令判断只在键没有过期时间时才补设，
    可避免每次自增都刷新 TTL 导致日配额窗口被无限延长。
    """
    try:
        pipe = cache_client.pipeline(transaction=False)
        pipe.incrby(key, amount)
        # 只在键尚无过期时间时设置，防止窗口被滑动延长
        pipe.ttl(key)
        results = await pipe.execute()
        current = int(results[0])
        remaining_ttl = int(results[1])
        if remaining_ttl < 0:
            await cache_client.expire(key, ttl_seconds)
        return current
    except (RedisError, ValueError, TypeError, OSError):
        return None


async def decrement(key: str, amount: int) -> None:
    """回滚预扣的额度。失败只记日志——额度多扣一点优于少扣。"""
    try:
        await cache_client.decrby(key, amount)
    except (RedisError, OSError):
        return None


async def close_redis() -> None:
    """应用关闭时释放连接池。两个客户端各自独立，都要关。"""
    for client in (redis_client, cache_client):
        try:
            await client.aclose()
        except (RedisError, OSError, AttributeError):
            # 兼容旧版 redis-py 没有 aclose 的情况
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass
