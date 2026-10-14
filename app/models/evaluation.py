"""AI 评测域模型：评测题库与评测运行记录。

阶段 6B 的地基。旧系统（``app/legacy/models.py``）有四张评测表：
``RAGEvaluationCase``、``RAGEvaluationRun``、``RAGQualityEvaluationRun``、
``RAGASAutoEvaluationRun``。这里重新设计为**两张**：

**一张题库表** ``eval_cases``。字段口径沿用旧表（question / expected_name /
expected_type / category / alternative_names / answer_keywords /
citation_names / expected_refusal），但两处改造：
1. JSON 列从「Text 存 json 字符串」升级为 JSONB 原生 list——旧系统用 SQLModel
   把数组塞进 ``*_json`` 字符串列，读写全靠手工 ``json.loads``，
   新架构统一走 ``JSONBText``（TypeDecorator，PG 上是 JSONB、SQLite 退化 TEXT，
   应用侧始终是 Python 对象，见 models/types.py）。
2. ``expected_type`` 的取值域必须是 IMCS 知识模型的真实 ``doc_type``
   （``condition`` / ``guideline``），而不是旧系统的 drug/document 口径——
   命中判定拿它和检索结果的 ``doc_type`` 直接比对，口径错位会让全量指标归零。

**一张运行表** ``eval_runs``，用 ``kind`` 判别列合并旧的三张 run 表。
理由：三张表的差异只在「产出哪些指标列」，而指标本身是稀疏的——
一次检索评测没有 faithfulness，一次 RAGAS 没有 refusal_rate。
拆三张表等于让「哪个字段属于哪种运行」这个知识散落在表结构里；
合并后统一用 ``metrics`` JSONB 存各类汇总值，表结构只保留
跨类型共用的生命周期列（status / started_at / completed_at / error_message），
新增一种评测类型不需要加列也不需要新迁移。

``results`` 存逐题明细（含每题的期望排名、命中资料、缺失关键词、四指标得分），
排障与「回归在哪一题」的定位全靠它，因此不设数据库长度上限（JSONB 天然不截断），
但单题明细的体积由服务层投影控制，见 evaluation_service 的序列化函数。
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKey
from app.models.types import JSONBText


class EvalKind(str, enum.Enum):
    """评测运行类型。与旧三层评测一一对应。"""

    RETRIEVAL = "retrieval"  # 检索质量：Top1 / Recall@3 / Precision@3
    QUALITY = "quality"  # 回答质量：证据覆盖 / 引用准确率 / 拒答准确率
    RAGAS = "ragas"  # RAGAS 四指标：faithfulness / relevancy / precision / recall


class EvalRunStatus(str, enum.Enum):
    """评测运行状态。

    与旧实现的关键区别：旧版用 ``BackgroundTasks`` + 进程内 ``threading.Lock``
    跑评测，API 进程重启就静默丢失、状态永远停在 running。这里状态落库、
    执行体搬到 Celery（queue=evaluate、acks_late），配合 monitoring 侧的
    僵尸任务清理，running 不再可能永久占位。
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvalCase(UUIDPrimaryKey, TimestampMixin, Base):
    """一条评测题：问什么问题、期望检索到哪份资料。"""

    __tablename__ = "eval_cases"
    __table_args__ = (
        Index("idx_eval_case_category", "category"),
        Index("idx_eval_case_type", "expected_type"),
    )

    question: Mapped[str] = mapped_column(Text, nullable=False)
    # 期望命中的资料标题（对应 knowledge_chunks.title）
    expected_name: Mapped[str] = mapped_column(String(300), nullable=False)
    # condition / guideline —— 必须与 KnowledgeChunk.doc_type 取值域一致
    expected_type: Mapped[str] = mapped_column(String(30), nullable=False)
    category: Mapped[str] = mapped_column(
        String(50), default="未分类", nullable=False,
        comment="分类维度，指标按类下钻（药物检索/病症检索/急症警示…）",
    )
    # 同一资料的等价写法：简称、全称、带书名号等。命中判定取并集，
    # 避免「同一份资料换个叫法就算检索失败」这种假阴性。
    alternative_names: Mapped[list] = mapped_column(
        JSONBText(), default=list, nullable=False
    )
    # 回答质量层用：答案证据里必须出现的关键词。为空时服务层回退用
    # expected_name（旧系统同一口径）。
    answer_keywords: Mapped[list] = mapped_column(
        JSONBText(), default=list, nullable=False
    )
    citation_names: Mapped[list] = mapped_column(
        JSONBText(), default=list, nullable=False
    )
    # 知识库覆盖不到的问题应设为 True：检索不到东西才是正确行为。
    # 这类题是「幻觉率」的反向度量——不设它，系统会把瞎编当作有能力。
    expected_refusal: Mapped[bool] = mapped_column(
        default=False, nullable=False
    )


class EvalRun(UUIDPrimaryKey, Base):
    """一次评测运行的状态与产出。"""

    __tablename__ = "eval_runs"
    __table_args__ = (
        Index("idx_eval_run_kind_created", "kind", "created_at"),
        Index("idx_eval_run_status", "status", "created_at"),
    )

    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default=EvalRunStatus.PENDING.value, nullable=False
    )
    # 本次运行覆盖的题量（抽样的话就是抽样后的条数）
    total_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # ---------- 检索层指标（kind=retrieval）----------
    top1_accuracy: Mapped[float | None] = mapped_column()
    recall_at_3: Mapped[float | None] = mapped_column()
    precision_at_3: Mapped[float | None] = mapped_column()

    # ---------- 质量层指标（kind=quality）----------
    answer_accuracy: Mapped[float | None] = mapped_column()
    citation_accuracy: Mapped[float | None] = mapped_column()
    refusal_accuracy: Mapped[float | None] = mapped_column()

    # ---------- RAGAS 四指标（kind=ragas）----------
    faithfulness: Mapped[float | None] = mapped_column()
    answer_relevancy: Mapped[float | None] = mapped_column()
    context_precision: Mapped[float | None] = mapped_column()
    context_recall: Mapped[float | None] = mapped_column()

    # 逐类下钻的指标（category -> 该类的三指标/四指标），
    # 与旧系统的分类统计同源：整体 0.8 可能是某一类 1.0 另一类 0.2，
    # 只报整体值会掩盖知识库缺口在哪一类。
    metrics: Mapped[dict | None] = mapped_column(JSONBText())
    # 逐题明细，用于定位「回归发生在哪一题」
    results: Mapped[list | None] = mapped_column(JSONBText())

    # 运行时的检索配置快照：指标必须能回答「当时用的是什么参数」。
    # 不快照的话，改了 rag_result_count 之后历史指标就失去了可解释性。
    retrieval_config: Mapped[dict | None] = mapped_column(JSONBText())
    # 知识库指纹（内容哈希 + 文档数），指标下滑时先排除「知识变了」这个因素
    knowledge_hash: Mapped[str | None] = mapped_column(String(64))
    knowledge_document_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    error_message: Mapped[str | None] = mapped_column(Text)
    ai_task_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), comment="RAGAS 异步任务的 AITask id，便于跳查执行详情"
    )
    triggered_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), comment="发起人 user id，评测是消耗 LLM 配额的动作"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


__all__ = [
    "EvalCase",
    "EvalKind",
    "EvalRun",
    "EvalRunStatus",
]
