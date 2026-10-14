"""IMCS 独立应用入口。

**与旧 ``app/main.py`` 完全分离**：旧入口是 MySQL + Milvus 的 RAG 问答系统，
本入口是 PostgreSQL/pgvector 的智能医疗预约诊疗系统。两套代码共存于同一仓库，
各自可独立启动、互不影响，因此这里绝不 import 旧模块（连 ``app.database``、
``app.routers`` 都不能碰），否则 IMCS 镜像会被迫装上 pymilvus / sqlmodel。

启动：``uvicorn app.imcs_main:app --port 8010``
（docker-compose.imcs.yml 的 api.command 由主代理同步改为指向本模块）。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.v1 import api_router
from app.core.cache import check_redis, close_redis
from app.core.config import settings
from app.core.database import async_session_factory, dispose_async_engine, dispose_sync_engine
from app.core.exceptions import register_exception_handlers
from app.core.trace import TRACE_HEADER, get_trace_id
from app.core.trace_middleware import TraceIdMiddleware
from app.services.llm.logger import call_logger

logger = logging.getLogger(__name__)


def cors_origins() -> list[str]:
    """把逗号分隔的配置串拆成允许源列表。

    拆分时过滤空串：环境变量里常见的尾随逗号（``"a,b,"``）若原样传给
    CORSMiddleware，会产生一个空 origin 规则，行为不可预期。
    """
    return [origin.strip() for origin in settings.cors_allow_origins.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """启动不建表，关闭必释放连接池。

    不在启动时 ``create_all``：schema 由 Alembic 迁移独占管理。自动建表会与
    迁移版本产生两套真相——模型改了但忘了生成迁移时，本地能跑、线上炸，
    这类 bug 极难定位。compose 里 init-db 服务已保证迁移先于 api 成功完成。

    关闭时释放两个引擎：Celery worker 与本进程可能同容器部署，
    只关异步引擎会漏掉同步连接池，滚动重启时留下连接堆积。

    LLM 调用日志缓冲区的生命周期也挂在这里：``call_logger`` 采用
    「攒够批量或到点才刷盘」的策略，**必须显式 start() 才会有定时刷盘任务**。
    没接 lifespan 的后果是低频场景下日志长期滞留在内存里不落库；
    而关闭时不 stop()，最后一段未攒满批次的日志会随进程一起消失。
    start/stop 都包了 try/except —— 可观测性组件故障不允许阻断应用启停。
    """
    logger.info("%s v%s 启动，environment=%s", settings.app_name, settings.app_version, settings.environment)
    try:
        call_logger.start()
    except Exception:  # noqa: BLE001 - 刷盘任务起不来也不能阻止服务启动
        logger.warning("LLM 调用日志定时刷盘任务启动失败，日志将仅在攒满批次时落库", exc_info=True)
    yield
    # 必须放在 dispose_async_engine 之前：stop() 要用异步 session 把尾部日志刷进库，
    # 引擎一旦释放就再也写不进去，这批日志直接丢。
    try:
        await call_logger.stop()
    except Exception:  # noqa: BLE001 - 关闭阶段不能因日志落库失败而中断资源释放
        logger.warning("LLM 调用日志刷盘收尾失败", exc_info=True)
    await close_redis()
    await dispose_async_engine()
    dispose_sync_engine()
    logger.info("%s 已释放数据库与 Redis 连接池", settings.app_name)


# 生产环境关闭交互式文档端点。本项目定位为演示/面试作品，默认 development
# 时保留 docs；一旦 ENVIRONMENT=production 就把攻击面（可直接发起请求的 UI）收掉。
_is_production = settings.environment == "production"

app = FastAPI(
    title=settings.app_name,
    description="智能医疗预约诊疗系统：分诊导诊、挂号排班、就诊病历、检查报告、处方发药与收费闭环",
    version=settings.app_version,
    lifespan=lifespan,
    # 文档端点统一挂在 /api 前缀下：Nginx 只反代 /api/，若 docs 留在 /docs
    # 则经网关访问会 404（实测已复现），只有直连 8010 才能打开
    docs_url=None if _is_production else "/api/docs",
    redoc_url=None if _is_production else "/api/redoc",
    openapi_url=None if _is_production else "/api/openapi.json",
)

# ---------- 异常处理 ----------
# 必须在挂路由之前注册：register_exception_handlers 走 app.exception_handler，
# 与路由注册顺序无关，但放在最前面能让「业务异常 → HTTP 响应」这条链路一眼可见。
register_exception_handlers(app)


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """未预期异常的兜底处理器。

    为什么不只依赖 ``register_exception_handlers``：它只覆盖 IMCSError 子类。
    DB 驱动抛出的 OperationalError、第三方 SDK 的异常等都会落到 Starlette 的
    默认处理器上，返回纯文本 "Internal Server Error" —— 前端按统一 JSON 结构
    解析 detail 会直接崩。这里补齐同构响应，同时**不回显异常原文**，
    避免 SQL 片段或文件路径经错误信息泄漏给客户端。真实堆栈只进服务端日志。
    """
    logger.exception(
        "未处理异常 %s：%s %s（trace=%s）",
        type(exc).__name__,
        request.method,
        request.url.path,
        get_trace_id(),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "internal_error", "message": "服务器内部错误，请稍后重试"}},
        # 兜底处理器返回的响应不经过任何中间件包装层，这里手动补 trace 头：
        # 客户端拿到 500 时能直接把 trace_id 报给研发，按 ID 回放整条链路。
        headers={TRACE_HEADER: get_trace_id()} if get_trace_id() else None,
    )


# ---------- 中间件 ----------
# trace 中间件最后 add_middleware，因此在栈最外层最先执行：Starlette 的
# 中间件是「后添加者先执行」。只有它先跑，后续的安全头中间件、CORS、业务路由
# 以及兜底异常处理器派生的日志与 LLM 调用，才能从 contextvar 读到同一个 trace_id。
# 反过来若放在内层，最需要追踪的失败请求（在外层就抛异常的那些）反而会缺链路 ID。
app.add_middleware(TraceIdMiddleware)

# CORS 必须在安全头中间件之前 add_middleware：Starlette 的中间件栈是
# 后添加者先执行，而预检请求（OPTIONS）不经过业务逻辑，需要 CORS 尽早应答。
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    # 认证走 Bearer token 而非 Cookie，不需要携带凭据，因此保持默认 False：
    # 开启 allow_credentials 会把 Origin 反射进响应头，扩大 CSRF 面
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    # X-Trace-Id 必须显式放行：浏览器跨域时自定义头不在 CORS safelist 里，
    # 不加的话前端自带的链路 ID 会在预检阶段被拦掉，链路在网关处断头。
    allow_headers=["Authorization", "Content-Type", "X-Request-Id", "X-Trace-Id"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """沿用旧项目已验证的安全响应头策略。

    - X-Content-Type-Options: nosniff —— 禁止浏览器猜测响应类型，
      防止上传的文本被当成脚本执行
    - X-Frame-Options: DENY —— 禁止被嵌套，防点击劫持
    - Referrer-Policy —— 跨域跳转时不外泄完整路径（含患者 ID 这类信息）
    - Permissions-Policy —— 后端 API 用不到摄像头/麦克风/定位，一律拒绝，
      避免被恶意页面借本域发起授权弹窗
    - 令牌属敏感凭证，认证接口响应禁止任何缓存（no-store），
      否则代理或浏览器可能把 access_token 留在缓存里被后续读取
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    # 用「路径中含 /auth/」而非前缀匹配：认证路由实际挂在
    # /api/v1/auth/... 下，早先写成 startswith("/auth/") 永远不命中，
    # 导致登录响应缺失 no-store（实测复现）。子串匹配对前缀调整也更稳健。
    if "/auth/" in request.url.path:
        response.headers["Cache-Control"] = "no-store"
    return response


# ---------- 业务路由 ----------
app.include_router(api_router)


# ---------- 系统端点 ----------
# 健康检查必须整体返回 200，包括 Redis 不可用时。原因：
# compose 的 api.healthcheck 与 Nginx 的上游探活都只看 HTTP 状态码，
# 若因缓存抖动返回 503，编排器会判定容器不健康并反复重启，
# 把一个「性能降级」放大成「服务不可用」。Redis 状态通过 body 里的字段表达。
@app.get("/health", tags=["系统"], summary="存活与依赖健康检查")
async def health() -> dict[str, object]:
    """探测数据库与 Redis。数据库真挂了才报 down，Redis 只报 degraded。"""
    db_ok = True
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - 健康检查不得因任何底层异常中断上报
        db_ok = False
        logger.warning("健康检查：数据库不可用", exc_info=True)

    redis_ok = await check_redis()

    status_value = "ok" if db_ok and redis_ok else ("down" if not db_ok else "degraded")
    return {
        "status": status_value,
        "service": settings.app_name.lower(),
        "version": settings.app_version,
        "environment": settings.environment,
        "database": "connected" if db_ok else "unavailable",
        "cache": "connected" if redis_ok else "unavailable",
    }


@app.get("/", include_in_schema=False, tags=["系统"])
async def index() -> dict[str, str]:
    """根路径仅返回服务标识，静态前端由 Nginx 直接托管，不经后端。"""
    return {"service": settings.app_name, "version": settings.app_version, "docs": "/docs"}


__all__ = ["app", "cors_origins", "health", "lifespan"]
