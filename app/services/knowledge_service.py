"""知识库服务层：向量化（worker 同步侧）与文档管理（API 异步侧）。

**为什么一个模块里混两种会话形态**：向量化跑在 Celery worker 里，只能用同步
Session（见 ``app/tasks/celery_app.py`` 模块 docstring 的进程模型说明）；文档
CRUD 跑在 FastAPI 请求里，只能用 AsyncSession。把两者放进同一个领域模块而不是
拆成两个文件，是因为它们共享同一套业务规则——content_hash 口径、切块与向量
必须同生同灭、索引状态取值集合。拆成两份会让「改一处规则」变成两处同步，
而这类不一致在测试里很难暴露（两侧各自都能跑通）。

分界靠命名与前缀注释明确：``# ==================== worker 侧`` 之下全部接受
``Session``，``# ==================== API 侧`` 之下全部接受 ``AsyncSession``。
跨侧调用只允许 worker 侧被 Celery 任务调用、API 侧被路由调用，不允许
在异步函数里包 ``run_async``，也不允许在同步函数里 await。

**批次 B 的边界**：本模块只处理「已经是纯文本」的内容（JSON 创建 + .md/.txt
上传）。PDF / docx 的解析属于后续批次，届时应在写库前把二进制转成
``content`` 字符串，向量化链路本身不需要改动。
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import (
    DuplicateTaskError,
    InvalidStateTransitionError,
    ResourceNotFoundError,
    TaskQueueUnavailableError,
)
from app.models.ai import AITask, AITaskStatus
from app.models.knowledge import KnowledgeChunk, KnowledgeDocument, VectorEmbedding
from app.services.rag.text_processing import split_text, tokenize_for_tsv
from app.services.state_machine import assert_transition, validate_status_value

logger = logging.getLogger(__name__)

# ==================== 常量 ====================

# AITask.task_type 取值。models/ai.py 的注释里登记的就是这个字符串，
# 派发侧与任务侧必须共用常量，否则 idx_ai_task_type_status 上的重复任务检查会失效。
TASK_TYPE_VECTORIZE = "knowledge_vectorize"
TASK_TYPE_REBUILD = "knowledge_rebuild"

# running/pending/retrying 都算「占位中」：判定重复派发时三者都要拦，
# 只拦 running 的话队列积压期间的重复点击仍会排进第二个重建任务。
ACTIVE_TASK_STATUSES: frozenset[str] = frozenset(
    {
        AITaskStatus.PENDING.value,
        AITaskStatus.RUNNING.value,
        AITaskStatus.RETRYING.value,
    }
)

# knowledge_documents.index_status 取值（models/knowledge.py 注释登记的四种）。
# 没有 CHECK 约束兜底，因此写入一律走 INDEX_STATUS_* 常量，避免大小写或拼写漂移
# 让 collect_pending_documents 静默漏掉整批文档。
INDEX_STATUS_PENDING = "pending"
INDEX_STATUS_INDEXED = "indexed"
INDEX_STATUS_STALE = "stale"
INDEX_STATUS_FAILED = "failed"

# 需要（重新）向量化的状态集合。failed 也在其中：上一轮失败的文档
# 必须在下一次任务里被重新捞起，否则一次 embedding 抖动就把文档永久卡在失败态。
PENDING_INDEX_STATUSES: tuple[str, ...] = (
    INDEX_STATUS_PENDING,
    INDEX_STATUS_STALE,
    INDEX_STATUS_FAILED,
)

CHUNKING_STRATEGY = "recursive-chinese-v1"

# 可注入的异步向量化函数签名：入参文本列表，出参向量列表。
EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


def content_sha256(content: str) -> str:
    """文档内容的 SHA-256 指纹。

    **必须与 ``app/seed/knowledge.py::_content_hash`` 保持同一实现口径**：
    种子写入的 hash 若与这里的算法不同（例如一边对清洗后的文本取哈希），
    向量化任务会把全部种子文档误判为「已变更」，每轮都全量重嵌一遍，
    增量策略形同虚设。两边都是 sha256(原始 utf-8 content)。
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ChunkStats:
    """单篇文档的向量化产出统计，用于聚合成任务结果。"""

    document_id: uuid.UUID
    created_chunks: int = 0
    deleted_chunks: int = 0
    created_vectors: int = 0
    skipped: bool = False
    reason: str | None = None


@dataclass(slots=True)
class VectorizeReport:
    """一轮向量化的汇总统计，落到 ``ai_tasks.result``。

    JSONB 里存的是这批数字而不是日志文本：前端要展示「新增 N 条切块」，
    监控也要据此判断某轮重建是否异常（比如删除数远大于新增数说明
    文档被批量清空）。日志只能给人看，没法被程序消费。
    """

    documents_scanned: int = 0
    documents_indexed: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    chunks_deleted: int = 0
    vectors_created: int = 0
    orphans_removed: int = 0
    per_document: list[dict[str, Any]] = field(default_factory=list)

    def to_result(self) -> dict[str, Any]:
        """转成可写入 JSONB 的字典（UUID 转字符串）。"""
        return {
            "documents_scanned": self.documents_scanned,
            "documents_indexed": self.documents_indexed,
            "documents_skipped": self.documents_skipped,
            "documents_failed": self.documents_failed,
            "chunks_created": self.chunks_created,
            "chunks_deleted": self.chunks_deleted,
            "vectors_created": self.vectors_created,
            "orphans_removed": self.orphans_removed,
            "per_document": self.per_document,
        }


# ==================== worker 侧（同步 Session，供 Celery 任务调用）====================


def collect_pending_documents(sync_session: Session) -> list[KnowledgeDocument]:
    """找出需要向量化的文档。

    两个条件取并集：

    1. ``index_status`` 属于 pending/stale/failed —— 显式的「待索引」标记
    2. 已标 indexed 但 ``content_hash`` 与实际内容不符 —— 有人绕过 service
       直接改了 content（数据修复脚本、SQL 手工订正都会这样），此时状态还是
       indexed，只看状态就会永久漏掉这篇文档

    第 2 类只能在 Python 侧比对：hash 是应用层算的，PG 里没有对应表达式，
    写不成一条 SQL。为避免「为找差异而全表扫描」，这里只扫
    ``is_active=True`` 的文档并按 id 稳定排序；知识库规模在万级以内时
    这点内存比对的代价可以接受，真要上量应该改成数据库生成列 + 触发器维护
    hash，届时本函数即可简化为纯状态过滤。

    ``is_active=False`` 的文档不参与：下线资料不该进入检索结果，给它建向量
    只是浪费 embedding 配额。
    """
    rows = (
        sync_session.execute(
            select(KnowledgeDocument)
            .where(KnowledgeDocument.is_active.is_(True))
            .order_by(KnowledgeDocument.id)
        )
        .scalars()
        .all()
    )

    pending: list[KnowledgeDocument] = []
    for doc in rows:
        if doc.index_status in PENDING_INDEX_STATUSES:
            pending.append(doc)
            continue
        if doc.index_status == INDEX_STATUS_INDEXED and doc.content_hash != content_sha256(
            doc.content
        ):
            logger.info(
                "文档 %s 状态为 indexed 但内容指纹已变化，纳入本轮重建", doc.title
            )
            pending.append(doc)
    return list(pending)


def populate_chunk_tsv(sync_session: Session, chunk_ids: list[uuid.UUID], tokens: list[str]) -> None:
    """用原生 SQL 把 jieba 分词结果写入 ``knowledge_chunks.tsv``。

    **为什么要单独一个函数走原生 SQL**：tsv 列在 PG 上是 TSVECTOR。
    SQLAlchemy 的 TypeDecorator（``models/types.py::TSVectorText``）只做值透传，
    传普通字符串进去时 PG 会按 tsvector 的字面量语法解析，遇到中文词串直接报
    ``syntax error at end of input`` 或把整串当成一个 lexeme —— 批次 A 的容器
    冒烟实测确认了正确写法只能是 ``to_tsvector('simple', :tok)``。

    用 ``simple`` 配置而非 ``chinese``/``english``：写入侧已经用
    ``tokenize_for_tsv`` 分好词，PG 再切一次会破坏 token 对齐，导致检索静默零命中
    （``services/rag/text_processing.py`` 模块 docstring 详述）。

    SQLite 环境退化为 TEXT，直接把分词串当普通文本写入即可，因此非 PG 方言下
    走 ORM 赋值分支。两侧行为一致（都能存能读），只是 PG 上多了 GIN 索引可用性。
    """
    if not chunk_ids:
        return

    dialect_name = sync_session.bind.dialect.name if sync_session.bind else ""
    if dialect_name != "postgresql":
        # SQLite：tsv 是普通 TEXT 列。**不能用裸 text("... WHERE id = :cid") 绑 UUID**——
        # 标准 sqlite3 驱动不认识 uuid.UUID 对象，会抛
        # ProgrammingError: type 'UUID' is not supported（实测打挂 6 个 worker 侧用例）。
        # @compiles 只补了建表 DDL 的渲染，管不到原生 SQL 的参数绑定。
        # 改用 ORM update()：KnowledgeChunk.id 是 PGUUID 列，其 bind_processor
        # 会把 UUID 正确序列化成 hex 文本，与批次 A 冒烟的写法等价且跨方言安全。
        for chunk_id, tokenized in zip(chunk_ids, tokens, strict=True):
            sync_session.execute(
                update(KnowledgeChunk)
                .where(KnowledgeChunk.id == chunk_id)
                .values(tsv=tokenized)
            )
        return

    for chunk_id, tokenized in zip(chunk_ids, tokens, strict=True):
        sync_session.execute(
            text("UPDATE knowledge_chunks SET tsv = to_tsvector('simple', :tok) WHERE id = :cid"),
            {"tok": tokenized, "cid": chunk_id},
        )


def _delete_existing_chunks(sync_session: Session, doc_id: uuid.UUID) -> int:
    """删除某文档现有切块及其向量，返回删除的切块数。

    向量表的外键是 ``ondelete="CASCADE"``，理论上删切块就能连带删向量；
    但 CASCADE 只在**数据库真正执行 DELETE** 时生效，而 ORM 的 bulk delete
    在某些 flush 时序下并不会触发级联（比如同一事务里先插后删的孤儿向量）。
    这里显式两条 DELETE：既保证幂等（重跑任务不残留旧向量），也让
    「删了几条切块」这个数字可控可统计。
    """
    chunk_ids = list(
        sync_session.execute(
            select(KnowledgeChunk.id).where(KnowledgeChunk.document_id == doc_id)
        )
        .scalars()
        .all()
    )
    if not chunk_ids:
        return 0

    sync_session.execute(
        delete(VectorEmbedding).where(VectorEmbedding.chunk_id.in_(chunk_ids))
    )
    sync_session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.id.in_(chunk_ids)))
    return len(chunk_ids)


def _build_chunks(
    doc: KnowledgeDocument,
) -> tuple[list[KnowledgeChunk], list[str]]:
    """切块并构造 ORM 对象，同时返回对应的分词串。

    切块的溯源字段（title/doc_type/source/source_tier）从文档反规范化复制：
    检索侧要按 source_tier 过滤权威资料，JOIN 文档表会让 LIMIT 语义变复杂，
    冗余存储换取查询简单（``models/knowledge.py`` 注释说明了这个取舍）。

    page_number/page_range 留空：本批次只有纯文本来源，页码信息要等 PDF 解析
    批次才有值可填。与其猜一个默认页码污染引用展示，不如保持 NULL，
    让前端明确知道「这条引用没有页码」。
    """
    pieces = split_text(doc.content)
    chunks: list[KnowledgeChunk] = []
    tokens: list[str] = []

    for piece in pieces:
        chunk_text = str(piece["content"])
        chunk = KnowledgeChunk(
            document_id=doc.id,
            chunk_index=int(piece["chunk_index"]),
            content=chunk_text,
            content_hash=content_sha256(chunk_text),
            token_count=len(chunk_text),
            chunk_start_index=int(piece["chunk_start_index"]),
            chunk_end_index=int(piece["chunk_end_index"]),
            chunking_strategy=CHUNKING_STRATEGY,
            title=doc.title,
            doc_type=doc.doc_type,
            source=doc.source,
            source_tier=doc.source_tier,
            extra_metadata=None,
        )
        chunks.append(chunk)
        tokens.append(tokenize_for_tsv(chunk_text))

    return chunks, tokens


def embed_texts(
    texts: list[str],
    *,
    ai_task_id: uuid.UUID | None = None,
    embed_fn: "EmbedFn | None" = None,
) -> list[list[float]]:
    """调 LLM 服务批量取向量（同步包装）。

    **唯一异步点**：embedding 客户端是异步的，worker 进程内没有既存事件循环，
    因此在这里用 ``asyncio.run`` 收口，而不是让每个调用方自己包一层——
    多处包 asyncio.run 容易漏，且嵌套调用（外层已有 loop）会直接 RuntimeError。

    延迟导入 celery_app.run_async：knowledge_service 会被 API 侧一起导入，
    顶层导入 tasks 包等于让 Web 进程也加载 Celery 应用配置（broker 连接参数等），
    没必要。同理 ``get_llm_service`` 也在函数内取——LLM 层带着供应商客户端与
    Redis 日志缓冲，不该在模块导入期被拉起来。

    ``embed_fn`` 是为可测性留的口子：维度校验、数量校验这些分支若只能靠真 API
    触发，单元测试就得打网络。注入后测试可以精确构造「返回数量不符」「维度不符」
    两种坏输入，而不牺牲生产路径（默认 None 仍走全局单例）。

    维度校验放在这里而不是调用方：所有写 vector_embeddings 的路径都必须经过本函数，
    集中一处校验才能杜绝「某条路径没校验就落库」——pgvector 的 vector(1536) 列
    会在 INSERT 时报错，但那时脏数据已经在事务里，报错信息也不会指向真正的根因。
    """
    from app.tasks.celery_app import run_async  # 延迟导入，避免 Web 进程加载 Celery 配置

    if not texts:
        return []

    if embed_fn is None:
        from app.services.llm.service import get_llm_service

        llm = get_llm_service()

        async def _default_embed(items: list[str]) -> list[list[float]]:
            vectors, _usage = await llm.embed(items, ai_task_id=ai_task_id)
            return vectors

        embed_fn = _default_embed

    vectors = run_async(embed_fn(texts))

    if len(vectors) != len(texts):
        raise ValueError(f"embedding 返回数量 {len(vectors)} 与输入数量 {len(texts)} 不一致")

    for index, vector in enumerate(vectors):
        if len(vector) != settings.embedding_dim:
            raise ValueError(
                f"第 {index} 条向量维度 {len(vector)} 与配置 embedding_dim="
                f"{settings.embedding_dim} 不一致，请核对 embedding 模型"
            )
    return vectors


def vectorize_document(
    sync_session: Session,
    doc: KnowledgeDocument,
    *,
    ai_task_id: uuid.UUID | None = None,
    force: bool = False,
    embed_fn: "EmbedFn | None" = None,
) -> ChunkStats:
    """单篇文档：清洗 → 切块 → 全文索引 → 批量 embedding → 落库。

    **幂等性**：``force=False`` 时，若文档已是 indexed 且 content_hash 与实际
    内容一致，直接跳过并返回空统计。这让「Celery 重试整个任务」变得安全——
    重跑只会补完没做完的文档，已完成的不会二次调用 embedding（那是真金白银的配额）。

    **为什么不复用旧切块做 diff**：按切块级 content_hash 只重嵌变化的那几块，
    理论上更省配额，但切块边界本身会随分块参数漂移，diff 出的「相同 hash 切块」
    可能位置已不同，反而更难推理。当前阶段选择「整篇重写」：语义简单、
    幂等性由文档级 hash 保证，成本上限是一篇文档的切块数。

    失败处理：任何一步抛异常都把文档标 failed 后再向外抛。标 failed 的意义在于
    ``collect_pending_documents`` 会把 failed 重新捞起，下一轮自动补偿；
    如果只抛异常不改状态，一篇永远切不出内容的坏文档会被反复选中却毫无进展。
    """
    stats = ChunkStats(document_id=doc.id)

    current_hash = content_sha256(doc.content)
    if not force and doc.index_status == INDEX_STATUS_INDEXED and doc.content_hash == current_hash:
        stats.skipped = True
        stats.reason = "内容指纹未变化"
        return stats

    try:
        stats.deleted_chunks = _delete_existing_chunks(sync_session, doc.id)

        chunks, tokens = _build_chunks(doc)
        if not chunks:
            # 内容为空或清洗后为空：不建切块，直接标 indexed（chunk_count=0）。
            # 标 pending 会让它每轮都被重新捞起，形成空转死循环。
            doc.index_status = INDEX_STATUS_INDEXED
            doc.chunk_count = 0
            doc.content_hash = current_hash
            doc.indexed_at = datetime.now(UTC)
            sync_session.flush()
            stats.skipped = True
            stats.reason = "清洗后无可索引内容"
            return stats

        sync_session.add_all(chunks)
        sync_session.flush()  # 拿到 chunk.id 才能填 tsv 与建向量外键

        populate_chunk_tsv(sync_session, [c.id for c in chunks], tokens)

        vectors = embed_texts([c.content for c in chunks], ai_task_id=ai_task_id)
        sync_session.add_all(
            [
                VectorEmbedding(
                    chunk_id=chunk.id,
                    embedding=vector,
                    model_version=settings.embedding_model,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
        )

        doc.index_status = INDEX_STATUS_INDEXED
        doc.chunk_count = len(chunks)
        doc.content_hash = current_hash
        doc.indexed_at = datetime.now(UTC)
        sync_session.flush()

        stats.created_chunks = len(chunks)
        stats.created_vectors = len(vectors)
        return stats

    except Exception as exc:
        # **先 rollback 再单独标 failed 提交**。只 flush 的话，任务层捕获本异常后
        # 会先 rollback（清理这篇文档写坏的半截切块），failed 标记连同回滚一起消失，
        # 库里状态倒退回 pending——前端看不到「这篇失败了」，排障只能翻 worker 日志
        # （单测实测复现：rollback 后读到的是 pending）。
        # 回滚后会话对象全部过期，再访问属性会重新加载，此时置 failed 并 commit，
        # 失败状态就跨过了回滚存活下来。
        logger.exception("文档向量化失败 id=%s title=%s", doc.id, doc.title)
        sync_session.rollback()
        doc.index_status = INDEX_STATUS_FAILED
        sync_session.commit()
        raise RuntimeError(f"文档 {doc.title} 向量化失败：{exc}") from exc


def remove_orphaned(sync_session: Session) -> int:
    """删除「所属文档已不存在」的切块与向量，返回清理掉的切块数。

    正常情况下不该有孤儿：文档表的外键是 ON DELETE CASCADE，删文档会连带删切块，
    切块删了向量也会连带删。但这个函数防的是另一类情况——迁移脚本、数据修复
    或直接 SQL 操作绕过了级联（例如先 TRUNCATE documents 再灌数据）。
    孤儿切块会出现在检索结果里，引用指向一篇不存在的文档，前端点开就是 404，
    属于「用户能看见但没人负责」的脏数据，因此每轮重建收尾都清一次。
    """
    orphan_ids = list(
        sync_session.execute(
            select(KnowledgeChunk.id).where(
                ~select(KnowledgeDocument.id)
                .where(KnowledgeDocument.id == KnowledgeChunk.document_id)
                .correlate(KnowledgeChunk)
                .exists()
            )
        )
        .scalars()
        .all()
    )
    if not orphan_ids:
        return 0

    sync_session.execute(delete(VectorEmbedding).where(VectorEmbedding.chunk_id.in_(orphan_ids)))
    sync_session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.id.in_(orphan_ids)))
    logger.info("清理孤儿切块 %d 条及其向量", len(orphan_ids))
    return len(orphan_ids)


def mark_document_stale(sync_session: Session, doc_id: uuid.UUID) -> bool:
    """内容变更后把文档标 stale，等待下一轮向量化。

    返回是否真的改了状态。已经是 pending/stale 的文档不再改写，
    避免把「尚未索引」覆盖成「已过期」这种倒退（两者的语义差别会影响
    前端提示：前者是排队中，后者是需要重建）。
    """
    doc = sync_session.get(KnowledgeDocument, doc_id)
    if doc is None:
        return False
    if doc.index_status in (INDEX_STATUS_PENDING, INDEX_STATUS_STALE):
        return False
    doc.index_status = INDEX_STATUS_STALE
    sync_session.flush()
    return True


def clear_document_vectors_sync(sync_session: Session, doc_id: uuid.UUID) -> int:
    """删除一篇文档的全部切块与向量，返回删除的切块数。

    给「文档删除」和「全量重建向量（保留切块）」复用。注意 rebuild 场景
    只删向量不删切块：切块的 tsv 与文本无关，换 embedding 模型时全文索引
    仍然有效，重算向量即可。
    """
    chunk_ids = list(
        sync_session.execute(
            select(KnowledgeChunk.id).where(KnowledgeChunk.document_id == doc_id)
        )
        .scalars()
        .all()
    )
    if not chunk_ids:
        return 0
    sync_session.execute(delete(VectorEmbedding).where(VectorEmbedding.chunk_id.in_(chunk_ids)))
    sync_session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.id.in_(chunk_ids)))
    return len(chunk_ids)


def reembed_document_chunks(
    sync_session: Session,
    doc: KnowledgeDocument,
    *,
    ai_task_id: uuid.UUID | None = None,
) -> int:
    """保留切块与 tsv，只为现有切块重算向量。

    全量重建（换 embedding 模型）专用：切块文本没变，分词结果也没变，
    变的只是向量。删掉切块重来会把 tsv 也一起丢掉，那才是真的浪费——
    全文索引重建不需要任何外部 API 调用，但重嵌需要。
    """
    chunks = list(
        sync_session.execute(
            select(KnowledgeChunk).where(KnowledgeChunk.document_id == doc.id)
        )
        .scalars()
        .all()
    )
    if not chunks:
        return 0

    vectors = embed_texts([c.content for c in chunks], ai_task_id=ai_task_id)
    sync_session.add_all(
        [
            VectorEmbedding(
                chunk_id=chunk.id,
                embedding=vector,
                model_version=settings.embedding_model,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
    )
    return len(vectors)


def truncate_vectors(sync_session: Session) -> int:
    """清空 vector_embeddings 全表，保留切块与 tsv，返回删除条数。

    用 ``DELETE`` 而非 ``TRUNCATE``：TRUNCATE 在 PG 上会取 ACCESS EXCLUSIVE 锁
    并**隐式提交当前事务**，把同一事务里刚写的任务状态一起提交掉；而且它带
    ``CASCADE`` 语义，一旦被补上就会顺着「切块是向量表的外键父表」这条链
    反向牵连删表。DELETE 走普通行锁、可回滚，rowcount 也如实反映删除量，
    便于写进任务统计。

    全量重建中途失败时这个选择尤其重要：DELETE 可回滚，TRUNCATE 不可。
    """
    result = sync_session.execute(delete(VectorEmbedding))
    return int(getattr(result, "rowcount", 0) or 0)


def iter_active_documents(sync_session: Session) -> list[KnowledgeDocument]:
    """列出全部启用中的文档，供全量重建逐篇遍历。

    与 ``collect_pending_documents`` 的区别是本函数不看 index_status 也不比 hash：
    换模型时所有文档都要重嵌，包括那些已经是 indexed 的。
    """
    return list(
        sync_session.execute(
            select(KnowledgeDocument)
            .where(KnowledgeDocument.is_active.is_(True))
            .order_by(KnowledgeDocument.id)
        )
        .scalars()
        .all()
    )


def load_task_for_update_sync(sync_session: Session, task_id: uuid.UUID) -> AITask | None:
    """加行锁读取任务。

    ``with_for_update`` 只在 PG 生效（SQLite 无行锁，测试里等价于普通 SELECT）。
    加锁的目的是防止两个 worker 副本同时处理同一个任务：acks_late 下消息可能被
    重投，不加锁就会出现「两个进程都在给同一篇文档删旧建新」，
    最终留下半套向量。
    """
    return sync_session.execute(
        select(AITask).where(AITask.id == task_id).with_for_update()
    ).scalar_one_or_none()


def set_task_status_sync(
    sync_session: Session,
    task: AITask,
    target: str,
    *,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
    progress: int | None = None,
    progress_message: str | None = None,
) -> AITask:
    """流转任务状态，非法迁移抛 ``InvalidStateTransitionError``。

    时间戳在这里统一维护：started_at 首次进入 running 时写，completed_at 在
    终态写，duration_ms 据此算。放到调用方各写一遍必然出现「标 failed 忘了写
    completed_at」，而僵尸任务清理（reap_stale_tasks）正是依赖这两个字段判断超时。

    **target == current 是进度刷新，不是状态迁移**：长任务每处理完一篇文档都要
    更新 progress/progress_message，此时目标状态仍是 running。而转换矩阵刻意
    不含 running→running 自环（自环会让「重复提交同一终态」这种 bug 静默通过），
    若进度刷新也走 assert_transition，第一篇文档的刷新就会抛
    InvalidStateTransitionError——实测向量化任务因此每轮重试只能推进一篇、
    24 篇拖成两小时。所以同状态直接落进度字段，跨状态才校验迁移。
    """
    if task.status != target:
        task.status = assert_transition("ai_task", task.status, target)

    now = datetime.now(UTC)
    if target == AITaskStatus.RUNNING.value and task.started_at is None:
        task.started_at = now
        task.queued_ms = int((now - task.created_at).total_seconds() * 1000) if task.created_at else None
    if target in (
        AITaskStatus.COMPLETED.value,
        AITaskStatus.FAILED.value,
        AITaskStatus.CANCELLED.value,
    ):
        task.completed_at = now
        if task.started_at is not None:
            task.duration_ms = int((now - task.started_at).total_seconds() * 1000)
    if result is not None:
        task.result = result
    if error_message is not None:
        task.error_message = error_message[:2000]
    if progress is not None:
        task.progress = max(0, min(100, progress))
    if progress_message is not None:
        task.progress_message = progress_message[:200]

    sync_session.flush()
    return task


def bump_retry_count_sync(sync_session: Session, task: AITask) -> AITask:
    """任务进入 retrying 时递增计数。

    ``retry_count`` 是业务侧的重试次数，与 Celery 自己的 request.retries 分开：
    前者给用户看（「已重试 2 次」），后者归 Flower。两者可能不一致
    （worker 崩溃重投不计入 Celery retries），以数据库为准更安全。
    """
    task.retry_count += 1
    sync_session.flush()
    return task


# ==================== API 侧（异步 session，供路由调用）====================


DOC_TYPES = (
    "guideline",
    "drug_manual",
    "lab_reference",
    "condition",
    "drug",
    "textbook",
)
SOURCE_TIERS = ("authoritative", "professional", "general", "unverified")


async def create_document(
    db: AsyncSession,
    *,
    title: str,
    content: str,
    doc_type: str,
    source: str,
    source_url: str | None = None,
    source_tier: str = "unverified",
    department_id: uuid.UUID | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> KnowledgeDocument:
    """新建知识文档，初始状态 pending，等向量化任务接手。

    **不在这里同步切块/embedding**：一次 embedding 调用可能几十秒，
    放在请求线程里会让接口超时，且失败后无法重试。创建只负责落原文，
    索引交给异步任务，这也是 index_status=pending 存在的意义。

    不做内容长度限制：长文档正是需要切片的场景，限制长度等于逼调用方
    自己切片，那会把分块策略散得到处都是。
    """
    doc = KnowledgeDocument(
        title=title,
        doc_type=doc_type,
        content=content,
        content_hash=content_sha256(content),
        source=source,
        source_url=source_url,
        source_tier=source_tier,
        department_id=department_id,
        extra_metadata=extra_metadata,
        is_reviewed=False,
        is_active=True,
        index_status=INDEX_STATUS_PENDING,
        chunk_count=0,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    logger.info("知识文档已创建 id=%s title=%s", doc.id, doc.title)
    return doc


def _to_uuid(value: Any) -> uuid.UUID | None:
    """把路径/查询参数里的字符串转 UUID，非法值返回 None 交由调用方报错。"""
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


async def get_document(db: AsyncSession, doc_id: uuid.UUID) -> KnowledgeDocument:
    """按 ID 取文档，不存在则 404。"""
    doc = await db.get(KnowledgeDocument, doc_id)
    if doc is None:
        raise ResourceNotFoundError("知识文档不存在")
    return doc


async def list_documents(
    db: AsyncSession,
    *,
    keyword: str | None = None,
    doc_type: str | None = None,
    index_status: str | None = None,
    include_inactive: bool = False,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[KnowledgeDocument], int]:
    """分页列出文档，返回 (当前页记录, 总数)。

    关键词只匹配 title 与 source，**不匹配 content**：content 是 Text 列，
    ILIKE '%kw%' 在全表上无法用索引，知识库灌满后这个查询会变成慢查询源头。
    要做正文检索应该走 /knowledge/search（全文 + 向量两路），
    列表接口的定位是「管理后台翻目录」。

    count 用子查询包一层而非 ``func.count()`` 直接聚合：filter 条件里含
    OR 与 ILIKE 时，SQLAlchemy 生成的 count 语句在某些版本会漏掉 WHERE，
    导致 total 恒等于全表行数——分页总数错了前端就会显示出不存在的页码。
    """
    conditions = []
    if not include_inactive:
        conditions.append(KnowledgeDocument.is_active.is_(True))
    if doc_type:
        conditions.append(KnowledgeDocument.doc_type == doc_type)
    if index_status:
        conditions.append(KnowledgeDocument.index_status == index_status)
    if keyword:
        pattern = f"%{keyword}%"
        conditions.append(
            or_(
                KnowledgeDocument.title.ilike(pattern),
                KnowledgeDocument.source.ilike(pattern),
            )
        )

    stmt = select(KnowledgeDocument)
    if conditions:
        stmt = stmt.where(*conditions)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = int((await db.execute(count_stmt)).scalar_one())

    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    page_stmt = (
        stmt.order_by(KnowledgeDocument.created_at.desc(), KnowledgeDocument.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = list((await db.execute(page_stmt)).scalars().all())
    return items, total


async def update_document(
    db: AsyncSession,
    doc_id: uuid.UUID,
    *,
    title: str | None = None,
    content: str | None = None,
    doc_type: str | None = None,
    source: str | None = None,
    source_url: str | None = None,
    source_tier: str | None = None,
    is_reviewed: bool | None = None,
    is_active: bool | None = None,
) -> KnowledgeDocument:
    """更新文档；content 变化时重算 hash 并标 stale。

    标 stale 是本函数最关键的一步：漏掉它的话，改了内容的文档仍是 indexed，
    ``collect_pending_documents`` 的状态过滤不会捞它（虽然 hash 比对那条兜底
    路径最终也能发现，但那要等到下一轮全量扫描），检索继续返回旧切块——
    医生看到的还是上一版指南，这在医疗场景是实打实的临床风险。
    """
    doc = await get_document(db, doc_id)

    if title is not None:
        doc.title = title
    if doc_type is not None:
        doc.doc_type = doc_type
    if source is not None:
        doc.source = source
    if source_url is not None:
        doc.source_url = source_url
    if source_tier is not None:
        doc.source_tier = source_tier
    if is_reviewed is not None:
        doc.is_reviewed = is_reviewed
    if is_active is not None:
        doc.is_active = is_active

    content_changed = content is not None and content != doc.content
    if content_changed:
        assert content is not None
        doc.content = content
        doc.content_hash = content_sha256(content)

    # 标题/来源分级变了但内容没变时，切块的反规范化字段就成了旧值，
    # 引用展示会显示过期的标题。这些字段由切块承载，因此同样要重建。
    meta_changed = any(
        value is not None
        for value in (title, source, source_tier, doc_type)
    )

    if content_changed or meta_changed:
        doc.index_status = INDEX_STATUS_STALE

    await db.commit()
    await db.refresh(doc)
    return doc


async def delete_document(db: AsyncSession, doc_id: uuid.UUID) -> None:
    """删除文档，连带清理切块与向量。

    显式删切块与向量而不是只删文档：ORM 的 relationship cascade 会处理切块，
    向量则靠数据库 FK CASCADE。但 SQLite 测试环境默认没开
    ``PRAGMA foreign_keys``，级联不会触发，孤儿向量会留在库里让检索返回
    指向已删文档的结果。显式 DELETE 让两侧行为一致，不依赖数据库特性。
    """
    doc = await get_document(db, doc_id)

    chunk_ids = list(
        (
            await db.execute(
                select(KnowledgeChunk.id).where(KnowledgeChunk.document_id == doc.id)
            )
        )
        .scalars()
        .all()
    )
    if chunk_ids:
        await db.execute(delete(VectorEmbedding).where(VectorEmbedding.chunk_id.in_(chunk_ids)))
        await db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.id.in_(chunk_ids)))
    await db.delete(doc)
    await db.commit()
    logger.info("知识文档已删除 id=%s 连带清理切块 %d 条", doc_id, len(chunk_ids))


async def find_active_task(
    db: AsyncSession, task_type: str
) -> AITask | None:
    """查同型任务里是否已有处于占位状态的记录。

    用于派发前的重复检查。不带 FOR UPDATE：异步侧拿不到跨请求的行锁语义
    （真正的并发防护靠下面 start_vectorization 里的 PG advisory lock）。
    """
    stmt = (
        select(AITask)
        .where(AITask.task_type == task_type, AITask.status.in_(ACTIVE_TASK_STATUSES))
        .order_by(AITask.created_at.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def acquire_dispatch_lock(db: AsyncSession, key: str) -> bool:
    """跨进程互斥锁，防并发派发产生两个重建任务。

    **为什么不能只用「查有没有 running 任务」这一招**：检查和插入之间有竞态窗口，
    管理员连点两次按钮，两个请求都在对方插入前查完了，双双通过检查，
    于是两个任务同时跑——它们都会执行「删旧切块再建新切块」，
    交错执行的结果是留下半套向量，比慢一点严重得多。

    PG 上用事务级 advisory lock：锁名由 key 的哈希算出两个 int，
    不同 key 不互相阻塞。获锁后在同一事务里查重 + 插入，另一个请求要么等锁
    要么拿不到锁，从而把「检查—写入」压成一个原子段。

    SQLite（测试环境）没有该函数，捕获异常按「无锁」放行，
    由上面的查重承担尽力而为的防护——测试本来就是单连接串行。
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    lock_left = int.from_bytes(digest[:4], "big", signed=True)
    lock_right = int.from_bytes(digest[4:8], "big", signed=True)
    try:
        result = await db.execute(
            text("SELECT pg_try_advisory_xact_lock(:left, :right)"),
            {"left": lock_left, "right": lock_right},
        )
        return bool(result.scalar_one())
    except Exception:  # noqa: BLE001 - 非 PG 后端无 advisory lock，降级为无锁
        logger.debug("advisory lock 不可用（非 PostgreSQL 后端），本次派发仅做查重")
        return True


async def start_vectorization(
    db: AsyncSession,
    user_id: uuid.UUID | None,
    *,
    task_type: str = TASK_TYPE_VECTORIZE,
    payload: dict[str, Any] | None = None,
) -> AITask:
    """建 AITask 记录并派发 Celery 任务，返回任务对象。

    先落库再派发（而不是反过来）：派发成功但落库失败的话，任务在跑但没有任何
    业务侧记录，前端查不到进度，僵尸清理也无从下手。反过来（先落库、派发失败）
    只是留下一条 pending 记录，可由用户重新触发，损失更小。

    派发失败（broker 挂了，``dispatch_task`` 返回 None）时把任务标 failed 并抛
    503：此时库里已有一条永远不会被执行的 pending 记录，不标失败它就会一直
    占着「同型任务运行中」的位置，把后续所有重建请求都挡在门外。
    """
    task_payload = dict(payload or {})

    locked = await acquire_dispatch_lock(db, f"knowledge:{task_type}")
    if not locked:
        await db.rollback()
        raise DuplicateTaskError(
            "同类知识库任务正在派发中，请稍后重试", task_type=task_type
        )

    existing = await find_active_task(db, task_type)
    if existing is not None:
        # 必须在 rollback **之前**取 id：rollback 会使会话中全部对象过期
        # （expire_on_commit=False 也挡不住 rollback 的过期），此时访问
        # existing.id 会触发隐式刷新——AsyncSession 里同步 IO 直接抛
        # MissingGreenlet，把本该 409 的重复派发变成 500（实测复现）。
        existing_task_id = str(existing.id)
        await db.rollback()
        raise DuplicateTaskError(
            "已有知识库索引任务在执行，请等待其完成后再触发",
            task_type=task_type,
            task_id=existing_task_id,
        )

    task = AITask(
        task_type=task_type,
        status=AITaskStatus.PENDING.value,
        payload=task_payload,
        progress=0,
        progress_message="已排队，等待 worker 接手",
        idempotency_key=f"{task_type}:{uuid.uuid4().hex}",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    celery_task_name = (
        "app.tasks.knowledge_tasks.rebuild_all_vectors"
        if task_type == TASK_TYPE_REBUILD
        else "app.tasks.knowledge_tasks.vectorize_knowledge"
    )

    from app.tasks.celery_app import QUEUE_VECTORIZE, dispatch_task

    kwargs: dict[str, Any] = {}
    if task_type == TASK_TYPE_REBUILD:
        kwargs["model_version"] = str(task_payload.get("model_version") or settings.embedding_model)

    async_result = dispatch_task(
        celery_task_name,
        str(task.id),
        queue=QUEUE_VECTORIZE,
        task_id=str(task.id),
        **kwargs,
    )

    if async_result is None:
        task.status = assert_transition("ai_task", task.status, AITaskStatus.FAILED.value)
        task.error_message = "任务队列不可用，未能派发"
        task.completed_at = datetime.now(UTC)
        await db.commit()
        raise TaskQueueUnavailableError("任务队列暂时不可用，请稍后重试")

    task.celery_task_id = str(getattr(async_result, "id", task.id))[:64]
    await db.commit()
    logger.info(
        "知识库任务已派发 type=%s ai_task_id=%s celery_id=%s user_id=%s",
        task_type,
        task.id,
        task.celery_task_id,
        user_id,
    )
    return task


async def get_task(db: AsyncSession, task_id: uuid.UUID) -> AITask:
    """按 ID 取任务，供状态查询端点使用。

    校验状态值合法性：历史数据或人工写库可能塞进一个不在枚举里的状态串，
    前端 switch 到未知分支会渲染成空白状态。这里提前拦住并报 404 之外的错误，
    比让前端猜要可靠。
    """
    task = await db.get(AITask, task_id)
    if task is None:
        raise ResourceNotFoundError("任务不存在")
    try:
        validate_status_value("ai_task", task.status)
    except InvalidStateTransitionError as exc:
        logger.warning("任务 %s 状态值非法：%s", task_id, exc.message)
        raise ResourceNotFoundError("任务状态数据异常，请联系管理员") from exc
    return task


__all__ = [
    "ACTIVE_TASK_STATUSES",
    "CHUNKING_STRATEGY",
    "DOC_TYPES",
    "INDEX_STATUS_FAILED",
    "INDEX_STATUS_INDEXED",
    "INDEX_STATUS_PENDING",
    "INDEX_STATUS_STALE",
    "PENDING_INDEX_STATUSES",
    "SOURCE_TIERS",
    "TASK_TYPE_REBUILD",
    "TASK_TYPE_VECTORIZE",
    "ChunkStats",
    "VectorizeReport",
    "acquire_dispatch_lock",
    "bump_retry_count_sync",
    "clear_document_vectors_sync",
    "collect_pending_documents",
    "content_sha256",
    "create_document",
    "delete_document",
    "embed_texts",
    "find_active_task",
    "get_document",
    "get_task",
    "list_documents",
    "load_task_for_update_sync",
    "mark_document_stale",
    "populate_chunk_tsv",
    "reembed_document_chunks",
    "remove_orphaned",
    "set_task_status_sync",
    "start_vectorization",
    "update_document",
    "vectorize_document",
]
