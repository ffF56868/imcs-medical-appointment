# IMCS 智能医疗预约诊疗系统（重构进行中）

> 本仓库正在从「医疗健康助手」（纯 RAG 问答）升级为 **IMCS 智能医疗预约诊疗系统**：
> FastAPI + SQLAlchemy 2.0 异步 + PostgreSQL/pgvector + Celery + Redis + OpenAI/DashScope，
> 面向患者、医生、检验科、收费员、药师、超级管理员六类角色，
> 打通智能分诊、预约挂号、接诊、检查报告、AI 解读、病历、处方、收费、发药全流程。

## 重构进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0 运行底座 | Alembic 迁移、Docker 编排、IMCS 入口、认证层、种子数据、测试 | ✅ 完成 |
| 1 架构地基 | 24 张表模型、统一 LLM 服务层、RBAC、状态机、Celery 队列 | ✅ 完成 |
| 2 RAG 知识底座 | pgvector + PG 全文检索 + RRF 融合、知识库向量化、知识管理接口 | ✅ 完成 |
| 3 分诊预约 | 工作流编排引擎、智能分诊、排班号源乐观锁、预约挂号闭环 | ✅ 完成 |
| 4 接诊报告 | 接诊病历、检查申请、报告录入、危急值、真实 AI 解读（qwen-plus） | ✅ 完成 |
| 5 处方收费发药 | 处方与合理用药校验、收费单与退费边界、先审后发闭环 | ✅ 完成 |
| 6A AI 可观测 | trace_id 贯穿全链路、LLM/检索/工作流日志落库、分钟桶指标聚合、监控接口与按 trace_id 回放、僵尸任务清理 | ✅ 完成 |
| 6B AI 评测 | 评测题库与运行表、检索三指标（Top1/Recall@3/Precision@3）、RAGAS 迁移 Celery | 待做 |
| 7 前端 | Vue 3 + TypeScript + Element Plus | 待做 |

## IMCS 启动方式（阶段 0 已可运行）

```bash
# 完整启动：PostgreSQL/pgvector、Redis、迁移+种子、API、两类 Celery worker、Nginx
docker compose -f docker-compose.imcs.yml up -d --build

# 可选：任务监控面板（Flower）与定时聚合
docker compose -f docker-compose.imcs.yml --profile monitoring up -d
```

| 地址 | 用途 |
| --- | --- |
| http://localhost:8080 | Nginx 统一入口（当前为占位页，阶段 7 接入前端） |
| http://localhost:8080/api/docs | Swagger 接口文档 |
| http://localhost:8010/health | API 健康检查 |
| http://localhost:5433 | PostgreSQL（宿主机映射端口） |

演示账号（种子数据，密码统一 `Imcs2026demo`）：`patient01`、`doctor01`、`lab01`、`cashier01`、`pharmacist01`、`admin`。生产部署前必须改密并删除演示账号。

环境变量：参照 `.env.example.imcs` 填写 `.env`，至少填 `OPENAI_API_KEY` 与 `DASHSCOPE_API_KEY`，并把 `IMCS_JWT_SECRET_KEY` 换成强随机值。

测试：`bash scripts/docker-test.sh`（全量 610 个单元用例；阶段 5 端到端业务闭环可用 `python scripts/e2e_stage5.py` 对运行中的栈验证）。

---

以下为旧「医疗健康助手」（RAG 问答系统）的原始说明。旧系统代码完整保留、可独立运行（`docker compose up -d`，端口 8003），但已不属于 IMCS 的演进路线。

# 医疗健康助手

> 基于 FastAPI、LangChain、OpenAI、Milvus、MySQL 和 Redis 的医疗健康 RAG 知识库问答系统。

医疗健康助手用于将病症、药物、PDF、Word、Excel、Markdown、文本和公开网页整理为可检索的知识库。已登录用户可以基于资料进行问答、查看引用来源和历史会话；管理员可以维护知识库、重建向量索引、查看质量评测和运行监控。

> [!WARNING]
> 本项目仅提供健康知识检索与健康教育信息，不提供诊断、处方或紧急医疗替代方案。出现呼吸困难、持续胸痛、意识改变、抽搐、单侧肢体无力等紧急症状时，应立即联系急救服务或就近医疗机构。

## 功能

- 病症与药物资料的增、查、改、删。
- 上传 `.md`、`.txt`、`.pdf`、`.docx`、`.xlsx`、`.xlsm` 文件；导入公开 HTML 网页。
- PDF 按可读取文字页拆分并保留页码，回答可展示引用资料与位置。
- 文本清洗、中文递归分块、OpenAI Embedding 向量化、Milvus 持久化检索。
- MySQL 关键词检索 + Milvus 向量检索的混合检索，并进行去重与规则重排序。
- 流式回答、会话历史、对话导出、回答反馈和问答监控。
- 分层记忆：会话短期窗口 + 压缩摘要，以及用户明确确认的长期记忆。
- 手机号或邮箱注册登录，普通用户与管理员权限隔离。
- 知识库版本快照、异步重建任务、资料审核队列、检索诊断与评测历史。
- 自定义检索评测题、Top1 准确率、Recall@3、Precision@3、回答证据与引用质量评估。

## 技术栈

| 分类 | 技术 | 用途 |
| --- | --- | --- |
| 后端 | Python 3.12、FastAPI、Uvicorn | HTTP API、网页服务、SSE 流式输出 |
| 数据模型 | SQLModel、PyMySQL | 数据表建模与 MySQL 访问 |
| 大模型 | OpenAI API、LangChain | 对话生成、Embedding、文档与向量检索抽象 |
| 向量数据库 | Milvus 2.4 | 保存向量切块并进行语义相似度搜索 |
| 关系数据库 | MySQL 8.4 | 用户、病症、药物、知识文档、会话、反馈和评测记录 |
| 缓存与限流 | Redis 7 | 知识库状态缓存与登录频率限制 |
| 文档处理 | pypdf、python-docx、openpyxl、BeautifulSoup | PDF、Word、Excel、网页解析 |
| 容器化 | Docker、Docker Compose | 一键部署全部服务 |
| 测试与 CI | Pytest、GitHub Actions | 自动化测试与推送后的持续集成 |

## 系统架构

```text
浏览器页面
    |
    v
FastAPI API (8003)
    |-- MySQL: 用户、原始资料、会话、反馈、评测、审计记录
    |-- Redis: 知识库状态缓存、登录限流
    |-- Milvus: 文档切块的向量索引
    |     |-- etcd: Milvus 元数据
    |     `-- MinIO: Milvus 对象存储
    `-- OpenAI API: Embedding 与回答生成
```

### RAG 问答流程

```text
上传资料 / 新增病症药物
    -> MySQL 保存原始内容和元数据
    -> 管理员重建知识库
    -> 清洗文本 -> 分块（700 字符，120 重叠） -> OpenAI Embedding -> Milvus

用户提问
    -> 登录校验与紧急症状安全拦截
    -> 读取长期记忆、会话摘要和最近 6 条对话
    -> 只用会话摘要与最近对话构造检索问题，避免个人偏好污染知识检索
    -> Milvus 向量检索 + MySQL 关键词检索
    -> 合并、去重、重排序，保留前 3 条资料
    -> OpenAI 根据资料生成回答
    -> 返回流式文本、引用来源和检索信息
```

## 快速开始

### 1. 前置条件

- Docker Desktop 已启动，且可以在 WSL / Linux 终端执行 `docker`。
- Docker Compose v2。
- 一个可用的 OpenAI API Key，以及能够访问对应 API 的网络。

在 Windows 上，推荐打开 WSL 终端后进入项目目录：

```bash
cd /mnt/e/医疗健康
docker --version
docker compose version
```

从 GitHub 克隆项目时，目录名称可以不同：

```bash
git clone <你的仓库地址>
cd <项目目录>
```

### 2. 配置环境变量

复制公开模板，创建本机私有配置：

```bash
cp .env.example .env
```

在 VS Code 中打开 `.env`，至少填写下面这一项：

```dotenv
OPENAI_API_KEY=你的真实_OpenAI_API_Key
```

常用配置如下。`.env` 已被 Git 忽略，**不要提交或分享其中的 Key、密码和连接地址**。

| 变量 | 是否必填 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 是 | OpenAI API Key |
| `OPENAI_BASE_URL` | 否 | 仅在使用代理或兼容 API 服务时填写，通常以 `/v1` 结尾 |
| `CHAT_MODEL` | 否 | 对话模型，默认 `gpt-4.1-mini` |
| `EMBEDDING_MODEL` | 否 | 向量模型，默认 `text-embedding-3-small` |
| `MYSQL_*` | 否 | MySQL 数据库名、账号和密码；首次部署前建议改成强密码 |
| `SESSION_DAYS` | 否 | 登录有效期，默认 7 天 |
| `LOGIN_RATE_LIMIT_PER_MINUTE` | 否 | 单账号与来源 IP 的每分钟登录次数限制，默认 30 |
| `REBUILD_JOB_TIMEOUT_SECONDS` | 否 | 知识库重建超时秒数，默认 1800 |

Docker Compose 会自动向 API 注入 MySQL、Redis 和 Milvus 的容器内地址。正常 Docker 部署时，不需要手动修改 `DATABASE_URL`、`REDIS_URL`、`MILVUS_HOST` 或 `MILVUS_PORT`。

### 3. 启动服务

推荐使用项目脚本启动：

```bash
bash scripts/docker-manage.sh start
```

它会自动完成镜像构建、首次数据卷创建和容器启动。启动后打开：

| 地址 | 用途 |
| --- | --- |
| [http://127.0.0.1:8003](http://127.0.0.1:8003) | Web 应用 |
| [http://127.0.0.1:8003/docs](http://127.0.0.1:8003/docs) | FastAPI / OpenAPI 接口文档 |
| [http://127.0.0.1:8003/health](http://127.0.0.1:8003/health) | 服务健康检查 |

检查运行状态：

```bash
bash scripts/docker-manage.sh status
```

正常时 `api`、`mysql`、`redis` 均应显示为 `healthy` 或 `Up`，`/health` 会返回 JSON 状态。

## 首次使用

### 注册并设置管理员

1. 打开 Web 应用，点击右上角“我的”，注册一个中国手机号或邮箱账号。
2. 手机号会统一保存为 `+86` 格式；密码需要是 8-32 位且同时包含英文字母和数字。
3. 新注册账号默认是普通用户，只能问答和访问被授权的资料。
4. 首次部署时，使用 MySQL 将一个已注册账号提升为管理员。

在 WSL / Linux 终端运行：

```bash
docker compose exec mysql mysql -u root -p
```

输入 `.env` 内的 `MYSQL_ROOT_PASSWORD`，然后执行以下 SQL。示例手机号 `13812345678` 在系统中会保存为 `+8613812345678`：

```sql
USE medical_health;
UPDATE `user` SET is_admin = 1 WHERE account = '+8613812345678';
SELECT account, is_admin FROM `user`;
EXIT;
```

重新登录该账号后，即可看到资料管理、重建知识库、评测和监控等管理员功能。

### 建立第一份知识库

管理员完成登录后：

1. 在“管理资料”添加病症、药物，或上传文件、导入公开网页。
2. 为资料填写可靠的来源、链接和可信度；医疗资料应人工核验。
3. 点击“重建知识库”，等待异步任务状态变为“完成”。
4. 回到聊天页面提问，并在回答下方查看引用来源、页码和检索路径。

新部署的数据库默认没有业务资料。必须先添加资料并重建，RAG 才有内容可以检索。

### 分层记忆

问答上下文分为两层：

- **短期记忆**：当前会话保留最近 6 条消息；超过 8 条后，较早消息会写入 MySQL 的会话摘要，原始消息仍保留在会话记录中。
- **长期记忆**：只保存用户明确表达的称呼、语言/回答偏好、过敏史、病史或长期用药等信息。普通的症状描述不会自动保存为个人病史。
- **写入控制**：长期记忆要求重要性达到阈值；优先使用 OpenAI Embedding 做相似度去重，没有可用 Key 时退回确定性的词重叠比较。相同记忆键出现矛盾时默认不写入，只有用户明确说“更正/改成”等才替换旧值。
- **管理接口**：登录后访问 `GET /memory?conversation_id=default` 查看当前用户的摘要与长期记忆，使用 `DELETE /memory/{memory_id}` 删除某条长期记忆。

长期记忆只用于个性化回答，不作为医疗事实或诊断依据。

## 文档导入规则

| 类型 | 行为 | 注意事项 |
| --- | --- | --- |
| Markdown / TXT | 读取 UTF-8 文本 | 单文件最大 5 MB |
| PDF | 每个有文字的页面保存为一条资料并保留页码 | 扫描版 PDF 需要先 OCR |
| Word | 读取段落和表格 | 仅支持 `.docx` |
| Excel | 每个非空工作表保存为一条资料 | 支持 `.xlsx`、`.xlsm` |
| 公开网页 | 抓取 HTML 正文并保存原始 URL | 不支持登录后页面、PDF 链接或动态网页 |

单次最多上传 20 个文件，批量总大小不超过 20 MB。每次新增、编辑或删除资料后，知识库都会变为“待重建”；不重建则 Milvus 中仍是旧索引。

## 权限与数据隔离

| 能力 | 普通用户 | 管理员 |
| --- | --- | --- |
| 基于已授权资料问答、查看历史、提交反馈 | 是 | 是 |
| 访问公开资料和自己的私有资料 | 是 | 是 |
| 新增、编辑、删除知识资料 | 否 | 是 |
| 重建、恢复知识库版本 | 否 | 是 |
| 查看评测、监控、审计记录 | 否 | 是 |

认证使用 Bearer Token。密码使用 `scrypt` 加盐哈希保存，Token 在数据库中只保存哈希值；Redis 负责短时间登录限流，MySQL 负责连续失败后的锁定记录。

## RAG 评测与监控

管理员可以在“检索评估”页面维护自定义题并运行评测。当前项目的检索评测只使用数据库内的自定义题，不再混入默认题。

| 指标 | 含义 |
| --- | --- |
| Top1 准确率 | 正确资料排在第 1 条的题目比例 |
| Recall@3 | 正确资料是否进入前 3 条结果的比例 |
| Precision@3 | 所有前 3 条有效结果中，相关资料所占比例 |
| 回答准确率 | 检索到的证据是否包含预设答案关键词 |
| 引用准确率 | 目标引用是否命中且带有来源或页码等位置 |
| 拒答准确率 | 知识库无资料时是否正确拒答 |

评测使用和日常问答相同的混合检索与重排序链路，但不会为每一题调用对话模型生成文本，因此结果更稳定且 API 消耗更低。

## 常用运维命令

所有命令都在项目根目录的 WSL / Linux 终端执行。

```bash
# 查看状态和健康检查
bash scripts/docker-manage.sh status

# 实时查看 API 日志，按 Ctrl+C 退出日志，不会停止服务
bash scripts/docker-manage.sh logs

# 修改 Python、前端、Docker 或 .env 后，重新构建并启动
bash scripts/docker-manage.sh rebuild

# 停止容器，但保留 MySQL、Redis、Milvus 等 Docker Volume 中的数据
bash scripts/docker-manage.sh stop

# 运行隔离的自动化测试，不影响正式资料库
bash scripts/docker-manage.sh test
```

也可以直接使用 Docker Compose：

```bash
docker compose ps
docker compose logs -f api
docker compose up -d --build
docker compose down
```

> [!CAUTION]
> 不要随意执行 `docker compose down -v` 或删除 `medical-health-*` 数据卷，这会清空 MySQL、Redis、Milvus 等持久化数据。

### 迁移旧 SQLite 数据（可选）

如果项目目录中有旧版 `medical_health.db`，首次启动脚本会将其备份到数据卷。需要迁移到 MySQL 时执行：

```bash
docker compose exec api python scripts/migrate_sqlite_to_mysql.py
```

该脚本复制数据，不会删除原始 SQLite 文件。迁移完成后，在网页中重建知识库。

## 部署说明

本仓库的 `docker-compose.yml` 适合本机和单机 Docker 部署，API 对外端口为 `8003`。持久化数据使用 Docker Volume，因此升级 API 镜像不会自动删除业务数据。

部署到云服务器或公网前，至少完成以下事项：

1. 修改 `.env` 中所有 MySQL 默认密码，并妥善保管 `OPENAI_API_KEY`。
2. 使用 Nginx、Caddy 或云负载均衡器为 API 配置 HTTPS 域名。
3. 不要将 MySQL、Redis、Milvus、MinIO 的管理端口直接暴露到公网；应通过防火墙或私有网络限制访问。
4. 定期备份 MySQL 与 Milvus 对应的数据卷，并在非生产环境验证恢复流程。
5. 对医疗资料进行人工审核，记录可靠来源和更新时间；不要把模型回答当作医疗结论。

## 代理与网络排查

如果 Docker 构建时无法拉取镜像或调用 OpenAI API 失败，先确认 WSL 内代理变量是否存在：

```bash
echo $HTTPS_PROXY
```

若你使用本机代理，可在当前 WSL 终端设置它，再执行启动命令。端口应填写代理软件当前显示的端口，不要写死旧端口：

```bash
export HTTPS_PROXY=http://127.0.0.1:<当前代理端口>
export HTTP_PROXY=$HTTPS_PROXY
bash scripts/docker-manage.sh start
```

项目脚本会把 WSL 的本机代理地址转换为 Docker 容器可访问的 `host.docker.internal` 地址。电脑重启后代理端口变化时，重新设置当前端口并执行 `start` 或 `rebuild` 即可。

## API 概览

完整接口参数和可在线调试的请求示例，请访问 [http://127.0.0.1:8003/docs](http://127.0.0.1:8003/docs)。主要接口分组如下：

| 分组 | 路径前缀 | 用途 |
| --- | --- | --- |
| 认证 | `/auth` | 注册、登录、退出、当前用户与审计日志 |
| 问答 | `/ask` | 普通问答与 `/ask/stream` 流式问答 |
| 病症 / 药物 | `/conditions`、`/drugs` | 结构化健康资料 CRUD |
| 文档 | `/documents` | 新增资料、文件上传、网页导入、编辑和删除 |
| 知识库 | `/knowledge` | 状态、异步重建、版本、审核与资料搜索 |
| 对话与反馈 | `/conversations`、`/feedback` | 历史会话、导出与用户反馈 |
| 分层记忆 | `/memory` | 查看和删除当前用户的长期记忆、会话摘要状态 |
| 评测与监控 | `/evaluation`、`/monitoring` | 检索评测、质量评估、诊断与运行指标 |

## 项目结构

```text
.
├── app/
│   ├── main.py                 # FastAPI 入口与健康检查
│   ├── routers/                # 认证、问答、资料、知识库、评测等接口
│   ├── static/                 # HTML、CSS、浏览器端 JavaScript
│   ├── hybrid_search.py        # MySQL 关键词 + Milvus 向量混合检索
│   ├── reranker.py             # 候选资料重排序
│   ├── vector_store.py         # 向量索引构建与 Milvus 访问
│   ├── document_parsers.py     # 多格式文档和网页解析
│   └── text_processing.py      # 文本清洗与分块策略
├── scripts/
│   ├── docker-manage.sh        # 启动、状态、日志、重建、测试快捷命令
│   ├── docker-up.sh            # 首次启动与数据卷初始化
│   └── migrate_sqlite_to_mysql.py
├── tests/                      # Pytest 自动化测试
├── sample_documents/           # 可用于导入测试的示例资料
├── docker-compose.yml          # 正式运行的容器编排
├── docker-compose.test.yml     # 隔离测试环境
├── Dockerfile                  # API 镜像构建说明
├── .env.example                # 环境变量模板
└── 使用说明.txt                 # 面向学习的详细文件说明
```

## 开发与测试

GitHub Actions 会在每次 `push` 和 Pull Request 时运行：

```bash
pytest -q
```

本机 Docker 测试使用临时 SQLite 与临时向量目录，不会连接正式 MySQL、Redis 或 Milvus。提交代码前建议执行：

```bash
bash scripts/docker-manage.sh test
```

## 相关文档

- [使用说明.txt](使用说明.txt)：按文件解释项目代码与学习顺序。
- [`.env.example`](.env.example)：可公开提交的环境变量模板。
- [FastAPI Docs](http://127.0.0.1:8003/docs)：运行后查看完整 API 文档。

