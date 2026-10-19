# IMCS 智能医疗预约诊疗系统

面向患者、医生、检验科、收费员、药师、超级管理员六类角色的全流程门诊业务系统，打通「智能分诊 → 预约挂号 → 接诊病历 → 检查报告 → AI 解读 → 处方 → 收费 → 发药」闭环，并内置 RAG 知识底座、AI 可观测性与评测体系。

技术栈：FastAPI + SQLAlchemy 2.0（异步）+ PostgreSQL 16 / pgvector + Celery + Redis + Nginx + Vue 3。

> [!WARNING]
> 本项目是工程演示系统，不提供诊断、处方或紧急医疗替代方案。系统中的检验危急值阈值、用药校验规则均为演示口径，实际部署必须按所在机构的 LIS 目录与药事规定逐条核定。出现呼吸困难、持续胸痛、意识改变、抽搐、单侧肢体无力等紧急症状时，应立即联系急救服务或就近医疗机构。

---

## 目录

- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [系统架构](#系统架构)
- [业务闭环与状态机](#业务闭环与状态机)
- [角色与权限](#角色与权限)
- [快速开始](#快速开始)
- [前端开发](#前端开发)
- [API 概览](#api-概览)
- [数据库与迁移](#数据库与迁移)
- [异步任务与队列](#异步任务与队列)
- [RAG 知识库](#rag-知识库)
- [AI 工程化](#ai-工程化)
- [测试](#测试)
- [部署与安全](#部署与安全)
- [项目结构](#项目结构)
- [故障排查](#故障排查)
- [遗留系统](#遗留系统)

---

## 核心特性

**门诊业务闭环**

- 智能分诊：急症关键词拦截（不走模型）→ 结构化症状提取 → RAG 证据召回 → 科室推荐；LLM 不可用时自动退回规则分诊，结果标注 `decision_source`。
- 预约挂号：号源乐观锁扣减（单条 UPDATE 完成扣号与版本推进）+ 唯一索引兜底，杜绝超卖与重复挂号。
- 接诊病历：待诊队列、接诊工作台、病历书写与定稿、就诊结束。
- 检查报告：医生开单 → 检验科接单 → 录入实测值 → 规则比对参考区间生成异常旗标与危急值 → AI 解读（异步）。
- 处方与用药安全：纯规则校验过敏史冲突（阻断）、重复用药、剂量，不依赖 LLM；冲突需带 override 理由才能强行开具。
- 收费与发药：处方与挂号费合并成收费单，先缴费后发药，药师先审后发，已发药禁止退费。
- 健康咨询：基于知识库的 SSE 流式问答，带引用出处、并发准入控制与心跳。

**知识底座**

- pgvector 向量检索 + PostgreSQL 全文检索（应用层 jieba 分词写入 tsvector）双路召回，RRF 融合。
- 知识文档按类型与来源可信度分级，支持审核、版本化重建与检索诊断。

**AI 工程化**

- 统一 LLM 服务层：供应商适配（OpenAI / DashScope）、Token 预算预扣与校正、超时重试、内容审核、失败降级、调用日志、成本核算。
- 全链路 `trace_id`：LLM 调用、检索、工作流分步日志落库，支持按 `trace_id` 回放整条链路。
- 指标分钟桶聚合 + 监控看板；僵尸任务定时清理。
- 三类评测：检索质量（Top1 / Recall@3 / Precision@3）、回答质量（证据覆盖 / 引用准确率 / 拒答准确率）、RAGAS 四指标（异步队列执行），带回归趋势与门禁。

**工程质量**

- 27 张业务表由 Alembic 独占管理 schema，应用启动不建表。
- 六类角色 RBAC 权限矩阵，写库前自检矩阵合法性（拼写错误立即失败而非静默失效）。
- JWT 认证、scrypt 加盐哈希、登录限流与失败锁定、统一安全响应头、生产环境关闭交互式文档。
- 700+ 单元/集成测试用例，业务状态机流转穷举校验。

---

## 技术栈

| 分类 | 技术 | 用途 |
| --- | --- | --- |
| 后端 | Python 3.12、FastAPI、Uvicorn | HTTP API、SSE 流式输出 |
| ORM 与迁移 | SQLAlchemy 2.0（asyncio）、Alembic、psycopg 3 | 异步数据访问、schema 版本管理 |
| 关系数据库 | PostgreSQL 16 + pgvector | 业务数据、知识切块、向量索引、全文检索 |
| 缓存与队列 | Redis 7、Celery 5 | broker / result backend / 缓存与限流；异步任务 |
| 任务监控 | Flower、Celery Beat | 队列面板、定时聚合与清理 |
| 大模型 | OpenAI SDK（兼容 DashScope）、tiktoken | 对话生成、Embedding、精确 Token 计数 |
| 中文处理 | jieba、langchain-text-splitters | 全文检索分词、递归分块 |
| 文档解析 | pypdf、python-docx、openpyxl、BeautifulSoup4 | PDF / Word / Excel / 网页 |
| 前端 | Vue 3、TypeScript、Vite、Naive UI、Pinia、Vue Router、ECharts、Axios | 六角色工作台、指标图表 |
| 网关 | Nginx 1.27 | 静态资源托管、API 反代、SSE 不缓冲 |
| 评测 | RAGAS 0.2.15 | faithfulness / relevancy / precision / recall |
| 测试与 CI | Pytest、pytest-asyncio、httpx、GitHub Actions | 自动化测试与持续集成 |
| 容器化 | Docker、Docker Compose | 一键部署全部服务 |

---

## 系统架构

```text
浏览器
   │
   ▼
Nginx (:8080) ──── /api/* ───▶ FastAPI (:8010)
   │                              │
   └── / （Vue 3 静态产物）        ├── PostgreSQL 16 + pgvector (:5433)
                                  │     业务表 · 知识切块 · 向量索引 · 全文索引 · 日志与指标
                                  ├── Redis 7 (:6380)
                                  │     db0 Celery broker · db1 result backend · db2 缓存与限流
                                  ├── OpenAI / DashScope
                                  │     对话 · Embedding · 报告解读
                                  └── Celery
                                        worker-vectorize  知识库向量化（并发 2）
                                        worker-online     报告解读 / 分诊补偿 / 指标聚合（并发 4）
                                        worker-evaluate   RAGAS 评测（并发 1，profile）
                                        beat              定时调度（profile）
```

**请求链路**

```text
客户端 → Nginx → TraceIdMiddleware（生成/透传 trace_id）
      → CORS → 安全响应头 → 业务路由
      → RBAC 权限校验 → Service 层（状态机校验 + 业务规则）
      → PostgreSQL / Redis / LLM 服务层
      → 统一 JSON 响应（含 X-Trace-Id）
```

未预期异常由兜底处理器转为同构 JSON，不回显异常原文，真实堆栈只进服务端日志。

---

## 业务闭环与状态机

```text
患者：智能分诊 ──▶ 预约挂号 ──▶ 到院签到 ──▶ 医生接诊
                                                  │
                              ┌───────────────────┼───────────────────┐
                              ▼                   ▼                   ▼
                          书写病历            开检查申请            开处方
                              │                   │                   │
                              │        检验科接单 → 录入报告            │
                              │        → 危急值判定 → AI 解读           │
                              │                   │                   │
                              │                   └───── 生成收费单 ───┤
                              ▼                                       ▼
                          就诊结束                          收费员收款 → 药师审核 → 发药
```

| 实体 | 合法迁移 | 终态 |
| --- | --- | --- |
| 预约 `appointment` | `booked` → `checked_in` / `cancelled` / `no_show`；`checked_in` → `in_progress` / `no_show` / `cancelled`；`in_progress` → `completed` | `completed`、`cancelled`、`no_show` |
| 就诊 `visit` | `in_progress` → `documented` / `closed`；`documented` → `closed` | `closed` |
| 检查申请 `exam_request` | `ordered` → `received` / `processing`；`received` → `processing`；`processing` → `reported`；`reported` → `reviewed` | `reviewed` |
| 处方 `prescription` | `draft` → `confirmed` / `cancelled`；`confirmed` → `paid` / `cancelled`；`paid` → `dispensed` | `dispensed`、`cancelled` |
| 收费单 `charge_order` | `pending` → `paid` / `cancelled`；`paid` → `refunded` | `refunded`、`cancelled` |
| AI 任务 `ai_task` | `pending` → `running` / `cancelled` / `failed`；`running` → `completed` / `failed` / `retrying` / `cancelled`；`retrying` → `running` / `failed` | `completed`、`cancelled` |

已缴费的处方与收费单不能直接作废，必须走退费流程——这是财务留痕要求，不是技术限制。

检查申请紧急度分 `routine`（常规）/ `urgent`（紧急）/ `stat`（立即，危急值复测），检验科工作台按此排序。

状态流转在 Service 层统一校验（`app/services/state_machine.py`），非法迁移返回带当前状态与目标状态的明确错误；所有写入都经过 Service 层，不存在旁路。

---

## 角色与权限

权限模型为 `roles` / `permissions` / `role_permissions` 三表，权限项以 `(资源, 动作)` 表达，路由通过 `require_permission` 校验。

| 角色 | 标识 | 主要职责 |
| --- | --- | --- |
| 患者 | `patient` | 智能分诊、健康咨询、预约挂号、查看病历/报告/处方/缴费 |
| 医生 | `doctor` | 接诊、书写病历、开检查申请与处方 |
| 检验科 | `lab_tech` | 接收检查申请、录入报告、触发 AI 解读 |
| 收费员 | `cashier` | 核对费用、收费与退费 |
| 药师 | `pharmacist` | 审核处方、发药、用药安全把关 |
| 超级管理员 | `admin` | 用户与排班管理、知识库维护、AI 运行监控与评测 |

超级管理员刻意**不授予**临床业务权限（就诊、处方、检查报告的读写）：管理员能改配置和账号，但不应能篡改诊疗记录，否则审计日志失去意义。

---

## 快速开始

### 1. 前置条件

- Docker Desktop 已启动，可在 WSL / Linux 终端执行 `docker`
- Docker Compose v2
- 至少一个可用的 LLM API Key（OpenAI 或 DashScope），且网络可访问对应端点

### 2. 配置环境变量

以 IMCS 模板创建本机私有配置：

```bash
cp .env.example.imcs .env
```

必填项：

```dotenv
OPENAI_API_KEY=你的_OpenAI_Key
DASHSCOPE_API_KEY=你的_DashScope_Key          # 备用供应商，主供应商故障时接管
IMCS_JWT_SECRET_KEY=<强随机值>                 # 生成：python -c "import secrets; print(secrets.token_urlsafe(48))"
```

常用配置：

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `LLM_PRIMARY_PROVIDER` / `LLM_FALLBACK_PROVIDER` | 否 | 主备供应商，默认 `openai` / `dashscope`；国内部署建议主用 `dashscope` |
| `LLM_MODEL_CHAT` / `_TRIAGE` / `_INTERPRET` | 否 | 分场景模型：对话与分诊用轻量模型，报告解读用强模型 |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | 否 | 默认 `text-embedding-3-small` / `1536`。**改动必须同步改维度并全量重建向量**，不同模型的向量不在同一语义空间 |
| `TOKEN_DAILY_LIMIT_PER_USER` / `_GLOBAL` | 否 | 日 Token 配额，超限后降级到小模型而非拒绝服务 |
| `RAG_RESULT_COUNT` / `RAG_RRF_K` | 否 | 送 LLM 的检索条数（默认 5）与 RRF 平滑常数（默认 60） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 否 | 中文递归分块，默认 700 / 120 |
| `IMCS_POSTGRES_PASSWORD` | 否 | 首次建卷时生效，改这里不会重置已有数据卷 |
| `IMCS_ENVIRONMENT` | 否 | `development` / `production`；生产环境自动关闭 `/api/docs` |
| `IMCS_API_WORKERS` | 否 | API 进程数，CPU 核数少时设为 1 以免与 worker 抢资源 |
| `DOCKER_HTTP_PROXY` / `DOCKER_HTTPS_PROXY` | 否 | 仅构建期拉 pip 包用，运行时不注入容器 |

`DATABASE_URL`、`REDIS_URL`、`CELERY_*` 在 `docker-compose.imcs.yml` 中被显式覆盖为容器内网地址，模板里的值只用于**在宿主机直接跑 alembic / pytest / 种子脚本**的场景。`.env` 已被 Git 忽略，不要提交或分享其中的 Key 与密码。

> [!IMPORTANT]
> `IMCS_` 前缀的变量（`IMCS_JWT_SECRET_KEY`、`IMCS_ENVIRONMENT`、`IMCS_DEBUG`、`IMCS_*_HOST_PORT`、`IMCS_API_WORKERS`）是**给 Compose 插值用的**，由 compose 映射成容器内无前缀的 `JWT_SECRET_KEY` / `ENVIRONMENT` / `DEBUG` 后再交给应用。
>
> `app/core/config.py` 的 `Settings` 没有设置 `env_prefix`，字段名即环境变量名。因此在宿主机直接运行 alembic / pytest / 种子脚本时，`.env` 里的 `IMCS_JWT_SECRET_KEY` **不会**被识别，需要同时提供无前缀的 `JWT_SECRET_KEY`。容器部署不受此影响。

### 3. 启动

```bash
# 完整启动：PostgreSQL/pgvector、Redis、迁移+种子、API、两类 Celery worker、Nginx
docker compose -f docker-compose.imcs.yml up -d --build

# 可选：Celery Beat 定时聚合 + Flower 监控面板
docker compose -f docker-compose.imcs.yml --profile monitoring up -d

# 可选：RAGAS 评测队列 worker（跑 RAGAS 评测时才需要）
docker compose -f docker-compose.imcs.yml --profile evaluate up -d
```

`init-db` 是一次性初始化服务，会在 API 启动前执行 `alembic upgrade head` 与种子数据写入；两者均幂等，重启栈会重复执行但不产生脏数据。迁移失败会阻断整个栈启动，避免 API 带着残缺 schema 上线。

### 4. 访问

| 地址 | 用途 |
| --- | --- |
| http://localhost:8080 | Nginx 统一入口，托管前端工作台并反代 `/api/*` |
| http://localhost:8080/api/docs | Swagger 接口文档（生产环境关闭） |
| http://localhost:8080/api/redoc | ReDoc 接口文档 |
| http://localhost:8010/health | API 健康检查（直连，不经网关） |
| http://localhost:8080/healthz | Nginx 自身探活 |
| http://localhost:5556 | Flower 任务监控面板（需 `monitoring` profile） |
| localhost:5433 | PostgreSQL（宿主机映射端口） |
| localhost:6380 | Redis（宿主机映射端口） |

端口刻意避开常见默认值（5432 / 6379 / 8000 / 80），便于与本机其他项目共存。可通过 `IMCS_*_HOST_PORT` 系列变量调整。

`/health` 在 Redis 不可用时仍返回 200，状态字段为 `degraded`；只有数据库不可用才返回 `down`。这是刻意设计——编排器与网关探活只看状态码，缓存抖动不应被放大成服务不可用。

### 5. 演示账号

种子数据写入以下账号，密码统一 `Imcs2026demo`：

| 账号 | 角色 |
| --- | --- |
| `patient01`、`patient02` | 患者 |
| `doctor01`（心血管）、`doctor02`（内科）、`doctor03`（神经内科） | 医生 |
| `lab01` | 检验科 |
| `cashier01` | 收费员 |
| `pharmacist01` | 药师 |
| `admin` | 超级管理员 |

> [!CAUTION]
> 以上为公开演示凭据。生产部署前必须强制改密或删除演示账号，并把 `IMCS_JWT_SECRET_KEY` 换成强随机值——JWT 密钥泄露等于任何人都能伪造医生身份。

---

## 前端开发

前端位于 `frontend/`，Vue 3 + TypeScript + Naive UI，构建产物 `frontend/dist/` 由 Nginx 直接挂载（无需重建镜像）。

```bash
cd frontend
npm install
npm run dev        # 开发服务器 http://localhost:5173，/api 代理到 localhost:8010
npm run build      # 类型检查 + 构建到 dist/
npm run typecheck  # 仅类型检查
```

路由表即菜单：在 `frontend/src/router/routes.ts` 登记页面并声明 `meta.roles`，侧栏自动生成，无需两处维护。

| 角色 | 页面 |
| --- | --- |
| 通用 | 工作台 |
| 患者 | 智能分诊、健康咨询、预约挂号、我的病历、检查报告、我的处方、我的缴费 |
| 医生 | 待诊队列、接诊工作台、我的病历 |
| 检验科 | 检验工作台、全部报告 |
| 收费员 | 收费台 |
| 药师 | 处方审核与发药、发药记录 |
| 管理员 | AI 运行监控、AI 评测中心、知识库管理 |

开发期与生产期都走 `/api` 同源代理且不 rewrite 前缀（后端路由前缀本身就是 `/api/v1`），两边行为一致，切换部署无需改 `baseURL`。跨域开发时需保证 `CORS_ALLOW_ORIGINS` 包含 `http://localhost:5173`（默认值）。

---

## API 概览

全部接口挂在 `/api/v1` 前缀下，Bearer Token 认证。完整参数与在线调试见 http://localhost:8080/api/docs。

| 分组 | 路径前缀 | 说明 |
| --- | --- | --- |
| 认证 | `/api/v1/auth` | 登录、刷新令牌、退出、当前用户 |
| 知识库 | `/api/v1/knowledge` | 文档 CRUD、向量化任务、混合检索 |
| 号源排班 | `/api/v1/schedules` | 分页查询可挂号源 |
| 预约挂号 | `/api/v1/appointments` | 挂号、我的预约列表与详情、签到、取消 |
| 智能分诊 | `/api/v1/triage` | 分诊请求、分诊历史与详情 |
| 接诊病历 | `/api/v1/consult` | 待诊队列、接诊、病历书写与定稿、结束就诊 |
| 检查报告 | `/api/v1/exams` | 开单、检验科工作台、报告录入、AI 解读、报告列表与详情 |
| 处方 | `/api/v1/prescriptions` | 开方、合理用药校验、确认开具、override |
| 收费 | `/api/v1/billing` | 生成收费单、收款、退费、收费单列表 |
| 发药 | `/api/v1/dispensing` | 药师审核、发药、取药记录 |
| 健康咨询 | `/api/v1/health-qa` | SSE 流式问答（`POST /health-qa/stream`） |
| AI 监控 | `/api/v1/monitoring` | 看板汇总、指标时间序列、LLM 调用明细、按 `trace_id` 回放、手动聚合 |
| AI 评测 | `/api/v1/evaluation` | 题库 CRUD、检索/质量同步评测、RAGAS 异步评测、运行详情、回归趋势 |

统一错误响应结构为 `{"detail": {"code": ..., "message": ...}}`；FastAPI 自身的参数校验错误是扁平的 `{"detail": [...]}`，客户端取值需兼容两种形态。

SSE 端点用 `POST` 而非 `GET`：`EventSource` 不能携带 `Authorization` 头，而该端点必须鉴权。Nginx 侧已关闭 `proxy_buffering`，否则 token 会被缓冲到响应结束一次性吐出，流式效果完全失效。

---

## 数据库与迁移

Schema 由 Alembic 独占管理，应用启动**不执行** `create_all`。自动建表会与迁移版本形成两套真相——模型改了但忘记生成迁移时，本地能跑、线上炸，这类问题极难定位。

```bash
# 容器内执行迁移
docker compose -f docker-compose.imcs.yml exec api alembic upgrade head

# 宿主机直接执行（需 .env 中 DATABASE_URL 指向 localhost:5433）
alembic upgrade head
alembic revision --autogenerate -m "描述"

# 重写种子数据（科室 → 角色权限 → 用户与患者档案 → 知识文档 → 评测题库 → 医生排班）
python -m app.seed.run_all
python -m app.seed.run_all --skip-schedules   # 只建账号与权限
```

`alembic.ini` 的 `sqlalchemy.url` 留空，真实连接串由 `migrations/env.py` 从 `app.core.config.settings` 动态注入，因此本机（`localhost:5433`）与容器内（`postgres:5432`）共用同一份配置。

种子写入在单个事务内提交，任何一步失败都回滚，不会留下「角色建好了但账号没建」的半成品状态。

27 张表按业务域拆分：

| 模块 | 表 |
| --- | --- |
| 认证与组织 | `departments`、`roles`、`permissions`、`role_permissions`、`users`、`doctor_schedules` |
| 患者 | `patient_profiles` |
| 诊疗 | `appointments`、`visits`、`exam_requests`、`exam_reports` |
| 药事 | `prescriptions`、`prescription_items`、`dispensing_records` |
| 收费 | `charge_orders` |
| 知识库 | `knowledge_documents`、`knowledge_chunks`、`vector_embeddings` |
| AI 可观测 | `ai_tasks`、`llm_call_logs`、`rag_retrieval_logs`、`workflow_traces`、`ai_metrics`、`triage_results` |
| AI 评测 | `eval_cases`、`eval_runs` |
| 健康咨询 | `health_qa_logs` |

主键统一使用 UUID，避免自增 ID 暴露业务量。PostgreSQL 专有类型（JSONB、ARRAY、tsvector、vector）在 `app/models/types.py` 中封装为 `TypeDecorator`，在非 PG 后端自动退化为 TEXT——这是单元测试无需真实 PostgreSQL 的基础。

---

## 异步任务与队列

队列按业务特性物理隔离，避免相互挤占：

| 队列 | Worker | 并发 | 承载任务 | 隔离原因 |
| --- | --- | --- | --- | --- |
| `vectorize` | `worker-vectorize` | 2 | 知识库向量化 | 重建可能跑数十分钟并吃满 embedding 配额，与在线任务同队列会堵死所有患者的 AI 解读。并发刻意压低：单任务内部已用批量 embedding 打满 IO，提高进程数只会互相抢配额 |
| `interpret` + `default` | `worker-online` | 4 | 报告解读、分诊补偿、指标聚合、普通任务 | 用户在前端等结果，需要较高并发 |
| `evaluate` | `worker-evaluate` | 1 | RAGAS 评测 | LLM 配额消耗大户，与在线业务并行会挤占真实用户额度；挂在 profile 下，日常开发不启动 |

Redis 单实例按 db 号隔离用途：db0 broker、db1 result backend、db2 缓存与限流。broker 与 result 分库可避免任务队列键和结果键互相干扰。

Celery Beat 驱动两个周期任务（全栈只能有一个 beat 实例，否则定时任务会被重复触发）：

| 任务 | 周期 | 作用 |
| --- | --- | --- |
| `aggregate-ai-metrics` | 60 秒 | 聚合 AI 指标到分钟桶。看板不直接扫明细日志表——明细增长快，实时聚合会让面板越用越慢 |
| `reap-stale-ai-tasks` | 300 秒 | 清理 worker 崩溃后停留在 `running` 的僵尸任务 |

Worker 子进程处理满 100 个任务后回收，防止内存泄漏累积。

---

## RAG 知识库

**检索链路**

```text
知识文档写入（JSON 创建 / 文本导入）
   → 文本清洗 → 中文递归分块（700 字符，120 重叠）
   → Embedding → vector_embeddings（pgvector）
   → jieba 分词写入 tsvector（全文索引）

检索请求
   → 向量检索（LIMIT 20） + 全文检索（LIMIT 20）   两路顺序执行，任一路失败降级为空结果
   → RRF 融合（k=60）
   → 截断到前 5 条
   → 写 rag_retrieval_logs（带 trace_id）
   → 返回带标题、来源、来源等级、页码的引用信息
```

中文分词放在应用层而非定制 PostgreSQL 镜像：`zhparser` / `pg_jieba` 都需自行编译 PG 扩展，部署成本高得多。

两路检索刻意不并发：共用 `AsyncSession` 并发会触发 `InterfaceError`，而派生独立异步引擎的收益不足以抵消构造风险；两路都是 LIMIT 20 的索引查询，顺序执行的开销可忽略。

**文档类型与来源分级**

| 字段 | 取值 |
| --- | --- |
| `doc_type` | `guideline`（指南）、`drug_manual`（药品说明书）、`lab_reference`（检验参考）、`condition`（病症）、`drug`（药物）、`textbook`（教材） |
| `source_tier` | `authoritative`、`professional`、`general`、`unverified` |

合法值以 `Literal` 在 schema 层显式声明，并在导入时断言与 service 层常量一致——两处漂移会让校验静默失效。

**安全边界**

- 急症拦截是第 0 步且不走模型：LLM 配额耗尽、供应商宕机、模型幻觉都不能延误真正的急症患者。
- 危急值与异常旗标全部由「实测值 × 参考区间」的规则比对得出，一次 LLM 都不调。低钾血症可致恶性心律失常，这类判断必须可复算、可审计、断网也能生效。
- 用药安全校验（过敏史冲突、重复用药、剂量）同为纯规则。模型幻觉可能让「对青霉素过敏的人被开了阿莫西林」被说成「无冲突」。
- 检索结果零命中或知识库未向量化都不阻塞分诊，只在响应中标注证据缺失。

---

## AI 工程化

### 统一 LLM 服务层

业务代码只依赖 `LLMService` 一个入口，横切关注点收敛在一处：

```text
输入审核 → Token 预算检查（预扣） → 逐供应商重试调用
        → 输出审核 → 真实用量校正 → 落库调用日志 → 成本核算
        → 全部失败则走降级链
```

没有这一层，超时/重试/日志/降级会散落在每个业务路由里，写法各不相同且极易漏项。收敛后任何 AI 能力的改进自动覆盖全系统。

| 机制 | 说明 |
| --- | --- |
| 供应商适配 | OpenAI 与 DashScope 共用 OpenAI SDK，切换只改 `base_url` 与 `api_key` |
| 分场景模型 | 对话/分诊用轻量模型（高频、低延迟），报告解读用强模型（低频、高准确） |
| 超时策略 | 流式首 token 超时 10 秒（慢通常代表供应商排队过载，应快速降级）；chunk 间隔超时 15 秒（长文本生成中，不该误杀） |
| Token 预算 | 按场景设请求级上限；日配额超限后降级到小模型而非拒绝服务。tiktoken 不可用时按「中文 1 token ≈ 1.5 字符」估算 |
| 重试 | 指数退避（基数 1 秒，上限 30 秒），最多 3 次；区分可重试与不可重试错误 |
| 温度 | 固定 0，医疗问答要求稳定可复现 |
| 调用日志 | 内存缓冲 + 攒够 50 条或 5 秒定时刷盘。同步写库会拖慢流式响应；缓冲区的 `start()` / `stop()` 挂在应用 lifespan 上，否则低频场景日志长期滞留内存、关闭时尾部日志随进程消失 |

### 可观测性

`trace_id` 由中间件在请求最外层生成或透传，贯穿 HTTP 响应头（`X-Trace-Id`）、LLM 调用日志、检索日志与工作流分步日志。客户端拿到 500 时可直接把 `trace_id` 报给研发，按 ID 回放整条链路。

中间件注册顺序是刻意的：Starlette 中间件「后添加者先执行」，`TraceIdMiddleware` 最后 `add_middleware` 因此位于栈最外层最先执行。只有它先跑，最需要追踪的失败请求（在外层就抛异常的那些）才不会缺链路 ID。

监控接口提供看板汇总、指标时间序列、LLM 调用明细、按 `trace_id` 回放、手动触发聚合。可观测性组件的故障不允许阻断主路径——所有落库函数内部吞异常。

### 评测体系

| 类型 | 指标 | 执行方式 | LLM 调用 |
| --- | --- | --- | --- |
| 检索质量 `retrieval` | Top1 准确率、Recall@3、Precision@3 | 同步 | 零调用 |
| 回答质量 `quality` | 证据覆盖、引用准确率、拒答准确率 | 同步 | 零调用 |
| RAGAS `ragas` | faithfulness、answer_relevancy、context_precision、context_recall | 异步（`evaluate` 队列） | 生成回答 + 评估 |

前两层是纯确定性判定，题库几十题、每题一次 RRF 检索即可在一次请求内跑完，因此结果稳定且几乎不消耗配额。

`answer_relevancy` 使用定制的中文医疗提示词而非 RAGAS 默认英文提示词：默认提示词会在回答出现任何「回避」标签时直接给 0 分，而「仅供参考」「不代替医生诊断」这类医疗安全话术会把本来直接的回答误判为回避。定制口径只校准这个二分类，算法本身不变。

检索评测复用日常问答的同一套混合检索链路，并做资料级去重——一篇文档会被切成多块，一次检索可能命中同一 `title` 的多个 chunk，不去重会让 Precision@3 把「同一份资料排了三次」算成三条有效结果，指标虚高。Recall / Precision 分母为 0 时返回 `None` 而非 `0.0`：「一题都没检到」与「检到了但都不对」是两种完全不同的失败。

评测运行支持前后对比与回归趋势查询，可设置门禁防止指标回退。

---

## 测试

```bash
# 隔离容器测试（临时 SQLite，不连正式 PostgreSQL / Redis）
bash scripts/docker-test.sh

# 宿主机直接跑（需先 pip install -r requirements.txt）
pytest -q                          # IMCS 全量用例
pytest tests/test_imcs_exam.py -q  # 单个业务域
pytest tests/legacy                # 遗留系统用例（默认不收集）
```

`pytest.ini` 默认 `--ignore=tests/legacy`：遗留测试依赖 MySQL + Milvus 且是同步 SQLModel 风格，与 IMCS 的 PostgreSQL/pgvector 异步栈不兼容。不隔离的话 pytest 会为每个用例加载 `tests/legacy/conftest.py`，进而去连根本不存在的 Milvus，所有用例卡在连接超时上。

`asyncio_mode = auto` 是必需的：IMCS 测试大量使用 `async def` 测试函数与异步 fixture，不显式声明模式时 pytest-asyncio 不会接管协程，用例会被静默跳过——表现为「N passed」但一个断言都没跑。

单元测试通过 `tests/imcs_sqlite.py` 在 SQLite 上建表：为 PostgreSQL UUID 方言注册 SQLite 编译实现，其余 PG 专有类型由 `TypeDecorator` 自动退化。因此大部分业务逻辑无需真实 PostgreSQL 即可覆盖。

**端到端验证**（需栈已启动，对运行中的容器发真实 HTTP 请求）：

| 脚本 | 覆盖链路 |
| --- | --- |
| `scripts/e2e_stage5.py` | 处方 → 收费 → 发药全链路，含过敏拦截、override、重复关联拒绝、越权访问 |
| `scripts/e2e_stage6a.py` | AI 可观测：`X-Trace-Id`、检索与分诊链路落库、按 `trace_id` 回放、手动聚合、权限拦截 |
| `scripts/e2e_stage6b.py` | AI 评测：题库、检索三指标与门禁、趋势对比、RAGAS 异步派发与轮询 |
| `scripts/e2e_stage7_sse.py` | 健康咨询 SSE：事件顺序、引用与用量字段、经 Nginx 反代不缓冲、并发上限 409 |
| `scripts/e2e_stage7b_doctor.py` | 医生端全链路：挂号 → 签到 → 队列 → 接诊 → 病历 → 检查 → 结案 → 越权 |
| `scripts/e2e_stage7c_ops.py` | 药师审核发药 + 收费台收款 + 检验接单出报告 |
| `scripts/e2e_stage7d_admin.py` | 管理端三页面（监控 / 评测 / 知识库）接口链路 |

```bash
python scripts/e2e_stage7b_doctor.py            # 默认打 http://127.0.0.1:8010
python scripts/e2e_stage5.py --base http://127.0.0.1:8010
```

`e2e_stage6b.py` 需要 `--profile evaluate` 启动评测 worker；`e2e_stage7d_admin.py` 刻意不触发 RAGAS 与索引重建（烧配额且长占 worker）。

**持续集成**：`.github/workflows/tests.yml` 在每次 push 与 Pull Request 时用 Python 3.12 安装 `requirements.txt` 并执行 `pytest -q`。

---

## 部署与安全

`docker-compose.imcs.yml` 适合本机与单机部署，对外只暴露 Nginx 的 8080 端口。业务数据存于 `imcs_pgdata` / `imcs_redisdata` 两个 Docker Volume，升级 API 镜像不会删除数据。

上线前至少完成：

1. 删除或强制改密全部演示账号，`IMCS_JWT_SECRET_KEY` 换成强随机值。
2. 设置 `IMCS_ENVIRONMENT=production`：自动关闭 `/api/docs`、`/api/redoc`、`/api/openapi.json`，收紧错误详情。
3. 用 Nginx / Caddy / 云负载均衡器配置 HTTPS 域名。
4. 不要把 PostgreSQL（5433）、Redis（6380）、Flower（5556）暴露到公网；用防火墙或私有网络限制。生产环境可直接移除这些端口映射，容器间走内部网络。
5. `DB_ECHO` 必须为 `false`：SQL 日志量极大且会泄露健康数据。
6. 按所在机构实际标准核定检验危急值阈值表与用药校验规则，并把规则表迁到配置或数据库字典表。
7. 定期备份 `imcs_pgdata`，并在非生产环境验证恢复流程。
8. 对医疗知识资料做人工审核，记录可靠来源与更新时间；不要把模型回答当作医疗结论。

已内建的安全措施：

- 密码使用 scrypt 加盐哈希（`scrypt$16384$8$1$salt$hash`），内存硬函数抗 GPU 暴力破解优于同参数 bcrypt；输出格式带参数前缀，便于将来升级。
- JWT HS256，访问令牌 720 分钟、刷新令牌 7 天；认证接口响应强制 `Cache-Control: no-store`，避免代理或浏览器缓存令牌。
- 登录限流每分钟 30 次（按账号与来源 IP），连续失败 5 次锁定 15 分钟。
- 安全响应头：`X-Content-Type-Options: nosniff`、`X-Frame-Options: DENY`、`Referrer-Policy`、`Permissions-Policy`（拒绝摄像头/麦克风/定位）。后端与 Nginx 各设一遍但不叠加——Nginx 只在静态资源 location 内设置，`/api/` 的响应由后端中间件负责，两层叠加会产生 `nosniff, nosniff` 这样的重复头。
- CORS 保持 `allow_credentials=False`：认证走 Bearer Token 而非 Cookie，开启会把 Origin 反射进响应头，扩大 CSRF 面。`X-Trace-Id` 需显式放行，否则跨域预检阶段就被拦掉，链路在网关处断头。
- 容器以非 root 用户（uid 10001）运行，降低容器逃逸影响面。
- 未预期异常不回显原文，避免 SQL 片段或文件路径经错误信息泄漏。

---

## 项目结构

```text
.
├── app/
│   ├── imcs_main.py            # IMCS 应用入口（中间件、异常处理、健康检查）
│   ├── api/
│   │   ├── deps.py             # 认证与权限依赖（CurrentUser、require_permission）
│   │   └── v1/                 # 13 个业务域路由，聚合到 /api/v1
│   ├── core/
│   │   ├── config.py           # pydantic-settings 配置（环境变量映射）
│   │   ├── database.py         # 异步/同步引擎与会话工厂
│   │   ├── cache.py            # Redis 客户端、缓存与限流
│   │   ├── security.py         # scrypt 密码哈希、JWT 签发与校验
│   │   ├── exceptions.py       # 业务异常体系与统一响应转换
│   │   ├── trace.py            # trace_id 上下文变量
│   │   └── trace_middleware.py # 链路 ID 中间件
│   ├── models/                 # SQLAlchemy 2.0 声明式模型，按业务域拆分（27 张表）
│   │   ├── base.py             # Base、UUID 主键、时间戳 Mixin
│   │   ├── types.py            # PG 专有类型的 TypeDecorator 封装（非 PG 后端退化为 TEXT）
│   │   ├── auth.py             # 科室、角色、权限、用户、排班
│   │   ├── clinical.py         # 预约、就诊、检查申请与报告（含状态流转表）
│   │   ├── pharmacy.py         # 处方、处方明细、发药记录
│   │   ├── billing.py          # 收费单
│   │   ├── knowledge.py        # 知识文档、切块、向量
│   │   ├── ai.py               # AI 任务、调用日志、检索日志、工作流轨迹、指标、分诊结果
│   │   ├── evaluation.py       # 评测题库与运行记录
│   │   └── health_qa.py        # 健康咨询日志
│   ├── schemas/                # Pydantic 请求/响应模型
│   ├── services/
│   │   ├── llm/                # 统一 LLM 服务层
│   │   │   ├── service.py      # 主入口：审核 → 预算 → 重试 → 降级 → 日志 → 成本
│   │   │   ├── provider.py     # 供应商适配（OpenAI / DashScope）
│   │   │   ├── retry.py        # 超时与重试（含流式双超时）
│   │   │   ├── token_budget.py # 预算预扣、校正与日配额
│   │   │   ├── moderation.py   # 输入输出内容审核
│   │   │   ├── degradation.py  # 降级链与兜底话术
│   │   │   ├── cost.py         # 成本核算
│   │   │   └── logger.py       # 调用日志批量缓冲落库
│   │   ├── rag/                # 向量检索、全文检索、RRF 融合、检索编排、分块
│   │   ├── safety/             # 问答安全护栏
│   │   ├── triage_workflow.py  # 智能分诊：急症拦截 → 提取 → 召回 → 推荐
│   │   ├── appointment_service.py  # 号源乐观锁扣减与挂号闭环
│   │   ├── consult_service.py  # 接诊与病历
│   │   ├── exam_service.py     # 检查申请、报告录入、危急值规则判定
│   │   ├── prescription_service.py # 处方与合理用药纯规则校验
│   │   ├── billing_service.py  # 收费与退费边界
│   │   ├── dispense_service.py # 先审后发闭环
│   │   ├── knowledge_service.py    # 知识文档管理与向量化
│   │   ├── health_qa_service.py    # SSE 流式问答
│   │   ├── observability.py    # 日志落库统一收敛点（内部吞异常）
│   │   ├── metrics_service.py  # 指标分钟桶聚合
│   │   ├── monitoring_service.py   # 监控查询与链路回放
│   │   ├── evaluation_service.py   # 检索与质量评测
│   │   ├── eval_answer.py      # 回答质量断言
│   │   ├── ragas_runner.py     # RAGAS 四指标
│   │   ├── state_machine.py    # 状态流转统一校验入口
│   │   ├── rbac_matrix.py      # 角色权限矩阵与自检
│   │   └── auth_service.py     # 登录、锁定、令牌
│   ├── tasks/
│   │   ├── celery_app.py       # Celery 配置、队列路由、beat 调度
│   │   ├── knowledge_tasks.py  # 向量化任务
│   │   ├── ai_tasks.py         # 报告解读、分诊补偿、RAGAS
│   │   └── monitoring_tasks.py # 指标聚合、僵尸任务清理
│   ├── seed/                   # 幂等种子数据（科室/权限/用户/知识/题库/排班）
│   ├── legacy/                 # 遗留系统的模型与 schema
│   └── *.py                    # 遗留 RAG 系统模块（main.py、vector_store.py 等）
├── frontend/
│   ├── src/
│   │   ├── api/                # 按业务域拆分的接口封装、SSE 客户端
│   │   ├── views/              # 按角色组织的页面（patient/doctor/lab/cashier/pharmacist/admin）
│   │   ├── components/         # 布局与通用组件
│   │   ├── composables/        # 菜单生成等组合式函数
│   │   ├── router/             # 路由表（即菜单来源）
│   │   ├── stores/             # Pinia：认证与布局状态
│   │   ├── styles/             # 设计变量与 Naive UI 覆写
│   │   ├── types/              # 枚举与业务类型
│   │   └── utils/              # 错误处理、金额与格式化
│   ├── dist/                   # 构建产物（入库，Nginx 直接挂载）
│   └── vite.config.ts          # 开发代理、分包策略
├── migrations/
│   ├── env.py                  # 从 settings 动态注入连接串
│   └── versions/               # 4 个迁移版本
├── deploy/nginx/
│   ├── nginx.conf
│   └── conf.d/
│       ├── imcs.conf           # 站点配置：API 反代、SSE 不缓冲、静态资源缓存策略
│       └── security-headers.inc
├── scripts/
│   ├── docker-test.sh          # 隔离容器测试
│   ├── e2e_stage*.py           # 端到端业务链路验证（7 个）
│   ├── seed_custom_evaluation_cases.py
│   └── seed_retrieval_gap_knowledge.py
├── tests/
│   ├── test_imcs_*.py          # 14 个业务域测试文件（700+ 用例）
│   ├── imcs_sqlite.py          # SQLite 建表兼容层（仅测试用）
│   └── legacy/                 # 遗留系统测试（默认不收集）
├── sample_documents/           # 可导入的示例资料
├── docker-compose.imcs.yml     # IMCS 容器编排
├── docker-compose.test.yml     # 隔离测试环境
├── docker-compose.yml          # 遗留系统编排
├── Dockerfile.imcs             # IMCS 镜像（API 与 worker 共用）
├── Dockerfile                  # 遗留系统镜像
├── alembic.ini
├── pytest.ini
├── requirements-imcs.txt       # IMCS 运行时依赖
├── requirements.txt            # 全量依赖（IMCS + 遗留系统，CI 使用）
├── .env.example.imcs           # IMCS 环境变量模板
└── .env.example                # 遗留系统环境变量模板
```

---

## 故障排查

**构建时拉不到 pip 包**

`Dockerfile.imcs` 固定走清华镜像源（PyPI 官方源从国内容器内直连不稳定，实测出现过 fastapi 成功后 pydantic 断连），并带 `--retries 5 --timeout 60` 兜住偶发抖动。部署到海外服务器时删掉 `-i` 那一行即回到官方源。若需通过代理拉包：

```bash
echo $HTTPS_PROXY   # 先确认 WSL 内代理变量存在，端口以代理软件当前显示为准
export HTTPS_PROXY=http://127.0.0.1:<当前代理端口>
export HTTP_PROXY=$HTTPS_PROXY
docker compose -f docker-compose.imcs.yml up -d --build
```

`scripts/*.sh` 会把 WSL 的本机代理地址转换为容器可访问的 `host.docker.internal`。构建期代理不注入运行时容器环境，避免应用把内网代理地址带进 LLM 请求。

**`/api/*` 全部返回 502**

Nginx 的 `upstream` 块主机名只在启动时解析一次，容器重建换 IP 后会一直连旧地址。`deploy/nginx/conf.d/imcs.conf` 已改用 Docker 内置 DNS（`resolver 127.0.0.11`）+ 变量承载上游地址，实现每次请求重新解析。若自行改回 upstream 块，重建 API 容器后必须重启 Nginx。

**SSE 流式没有打字机效果**

检查网关链路上是否有关闭 `proxy_buffering`。Nginx 侧已设 `proxy_buffering off`、`proxy_cache off`、`chunked_transfer_encoding on`；读写超时设为 300 秒（长报告解读可能持续数分钟，默认 60 秒会中途断流）。若前面还有其他代理，需同样配置。

**改了 Embedding 模型后检索结果异常**

不同模型产出的向量不在同一语义空间，混用会让检索完全失真。改 `EMBEDDING_MODEL` 必须同步改 `EMBEDDING_DIM`（需生成迁移调整 `vector_embeddings.embedding` 列定义）并执行全量向量重建。

**权限授予后仍返回 403**

权限矩阵里的资源名或动作名拼写错误会让授权静默失效：记录写进去了，但 `require_permission` 查的是正确拼写。`seed_rbac` 在写库前调用 `validate_matrix()` 自检并直接抛错，遇到该错误先修正 `app/services/rbac_matrix.py` 中的拼写。

**pytest 显示 passed 但断言没跑**

检查 `pytest.ini` 的 `asyncio_mode = auto` 是否被覆盖。协程测试函数未被 pytest-asyncio 接管时会被静默跳过。

**容器日志里出现 `LLM 调用日志刷盘收尾失败`**

`call_logger.stop()` 要用异步 session 把尾部日志刷进库，必须在 `dispose_async_engine()` 之前执行。该警告不影响请求处理，但说明最后一批未攒满批次的调用日志未落库。

> [!CAUTION]
> 不要执行 `docker compose -f docker-compose.imcs.yml down -v` 或删除 `imcs_pgdata` / `imcs_redisdata` 数据卷，这会清空全部业务数据与向量索引。

---

## 遗留系统

本仓库同时保留重构前的「医疗健康助手」（纯 RAG 问答系统），代码完整、可独立运行，但已不属于 IMCS 的演进路线。

| 项 | IMCS | 遗留系统 |
| --- | --- | --- |
| 入口 | `app/imcs_main.py` | `app/main.py` |
| 编排 | `docker-compose.imcs.yml` | `docker-compose.yml` |
| 镜像 | `Dockerfile.imcs` + `requirements-imcs.txt` | `Dockerfile` + `requirements.txt` |
| 数据库 | PostgreSQL 16 + pgvector | MySQL 8.4 |
| 向量库 | pgvector | Milvus 2.4（+ etcd、MinIO） |
| 端口 | API 8010 / Nginx 8080 | API 8003 |
| 测试 | `tests/test_imcs_*.py` | `tests/legacy/`（默认不收集） |
| 运维脚本 | `docker compose -f docker-compose.imcs.yml ...` | `scripts/docker-manage.sh` |

两套代码共存于同一仓库但完全隔离：`app/imcs_main.py` 绝不 import 遗留模块（连 `app.database`、`app.routers` 都不碰），否则 IMCS 镜像会被迫装上 pymilvus / sqlmodel。两者端口不重叠，可同时运行以便对比。

```bash
# 启动遗留系统
bash scripts/docker-manage.sh start     # 或 docker compose up -d
bash scripts/docker-manage.sh status    # 查看状态与 /health
bash scripts/docker-manage.sh logs      # 跟随 API 日志
bash scripts/docker-manage.sh stop      # 停止但保留数据卷
```

> [!NOTE]
> `scripts/docker-manage.sh`、`scripts/docker-up.sh`、`使用说明.txt`、`.env.example` 均服务于遗留系统，操作 IMCS 时请使用 `docker compose -f docker-compose.imcs.yml` 与 `.env.example.imcs`。`scripts/docker-test.sh` 是例外——它按 `pytest.ini` 收集，实际运行的是 IMCS 测试。
