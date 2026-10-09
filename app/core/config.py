"""IMCS 全局配置。

使用 pydantic-settings 统一读取环境变量，替代旧代码里散落的 ``os.getenv``。
所有配置集中在 ``settings`` 单例，业务代码不再直接读环境变量。
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """IMCS 运行时配置。

    端口约定：本机同时运行着其他项目，已占用 5432（另一个 PostgreSQL）、
    6379（Redis）、8000/8002/8003（API）、3000（前端）、9000-9001（MinIO）、
    19530-19531（Milvus）。IMCS 一律使用下列独立端口，避免冲突：
    PostgreSQL 5433、Redis 6380、API 8010、Flower 5556、Nginx 8080。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- 应用 ----------
    app_name: str = "IMCS"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", description="development / production")
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8010

    # ---------- 数据库 ----------
    # 容器内地址；宿主机映射到 5433 以便本地客户端连接
    database_url: str = Field(
        default="postgresql+psycopg://imcs:imcs_password@postgres:5432/imcs",
        description="SQLAlchemy 异步连接串（FastAPI 使用）",
    )
    sync_database_url: str = Field(
        default="postgresql+psycopg://imcs:imcs_password@postgres:5432/imcs",
        description="SQLAlchemy 同步连接串（Celery worker 使用）",
    )
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_echo: bool = False

    postgres_db: str = "imcs"
    postgres_user: str = "imcs"
    postgres_password: str = "imcs_password"
    postgres_host_port: int = 5433

    # ---------- Redis ----------
    redis_url: str = "redis://redis:6379/0"
    redis_host_port: int = 6380
    # broker 与 result backend 分库，避免任务队列键和结果键互相干扰
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    # ---------- 认证 ----------
    jwt_secret_key: str = Field(
        default="change-me-in-production",
        description="生产环境必须通过环境变量覆盖",
    )
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=60 * 12, ge=5, le=60 * 24 * 7)
    refresh_token_expire_days: int = Field(default=7, ge=1, le=30)
    max_login_failures: int = Field(default=5, ge=1, le=20)
    login_lockout_minutes: int = Field(default=15, ge=1, le=1440)
    login_rate_limit_per_minute: int = Field(default=30, ge=1, le=300)

    # ---------- CORS ----------
    # 逗号分隔的允许源。开发期前端是 Vite dev server（默认 5173），
    # 生产环境由 Nginx 同源提供静态资源，通常不需要再加别的源。
    # 用字符串而非 list[str]：pydantic-settings 解析环境变量时，
    # 列表类型要求 JSON 字面量（'["a","b"]'），在 .env / compose 里极易写错；
    # 逗号串在 shell 和 compose 里都是自然写法。拆分逻辑见 imcs_main.cors_origins()。
    cors_allow_origins: str = "http://localhost:5173"

    # ---------- LLM 供应商 ----------
    # DashScope 提供 OpenAI 兼容端点，因此两个供应商共用 openai SDK
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 主供应商与备用供应商；主供应商失败时按此顺序降级
    llm_primary_provider: str = "openai"
    llm_fallback_provider: str = "dashscope"

    # 场景 -> 主模型；备用模型用于降级链第一级
    llm_model_chat: str = "gpt-4.1-mini"
    llm_model_chat_fallback: str = "qwen-turbo"
    llm_model_triage: str = "gpt-4.1-mini"
    llm_model_triage_fallback: str = "qwen-turbo"
    llm_model_interpret: str = "gpt-4.1"
    llm_model_interpret_fallback: str = "qwen-plus"
    embedding_model: str = "text-embedding-3-small"
    # text-embedding-3-small 输出 1536 维；换供应商模型时必须同步改向量列定义
    embedding_dim: int = 1536
    rerank_model: str = "gte-rerank"

    # 单价（美元 / 1K token），用于成本核算
    llm_price_input_per_1k: float = 0.0004
    llm_price_output_per_1k: float = 0.0016
    embedding_price_per_1k: float = 0.00002

    # ---------- LLM 超时与重试 ----------
    llm_connect_timeout: float = Field(default=5.0, gt=0)
    llm_read_timeout: float = Field(default=30.0, gt=0)
    # 流式：首 token 慢通常代表排队过载，应快速降级；chunk 间隔慢可能只是长文本生成，不该误杀
    llm_stream_first_token_timeout: float = Field(default=10.0, gt=0)
    llm_stream_idle_timeout: float = Field(default=15.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)
    llm_retry_backoff_base: float = Field(default=1.0, gt=0)
    llm_retry_backoff_cap: float = Field(default=30.0, gt=0)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # ---------- Token 预算 ----------
    # 请求级上限按场景设置，避免单次问答吃掉过多上下文
    token_budget_triage: int = 2000
    token_budget_report_interpret: int = 8000
    token_budget_chat: int = 1500
    token_budget_default: int = 4000
    # 日配额（按用户）；超限后降级到小模型，不直接拒绝
    token_daily_limit_per_user: int = Field(default=200_000, ge=1000)
    token_daily_limit_global: int = Field(default=5_000_000, ge=10_000)
    # 中文 1 token ≈ 1.5 字符，tiktoken 不可用时的兜底估算系数
    token_char_ratio_fallback: float = Field(default=1.5, gt=0)

    # ---------- 调用日志批量落库 ----------
    # 同步写库会拖慢流式响应，改为内存缓冲 + 定时/定量刷盘
    llm_log_batch_size: int = Field(default=50, ge=1, le=1000)
    llm_log_flush_interval: float = Field(default=5.0, gt=0)

    # ---------- RAG 检索 ----------
    rag_vector_fetch_count: int = 20
    rag_keyword_fetch_count: int = 20
    rag_rrf_k: int = Field(default=60, ge=1, description="RRF 平滑常数，论文推荐 60")
    rag_result_count: int = 5
    rag_min_score: float = 0.0
    chunk_size: int = 700
    chunk_overlap: int = 120

    # ---------- Celery ----------
    celery_task_serializer: str = "json"
    celery_result_expires: int = 3600
    celery_worker_concurrency: int = 4


@lru_cache
def get_settings() -> Settings:
    """返回缓存的配置单例。测试中可用 ``get_settings.cache_clear()`` 重载。"""
    return Settings()


settings = get_settings()
