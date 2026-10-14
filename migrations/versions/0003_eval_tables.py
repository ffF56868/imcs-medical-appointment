"""阶段 6B：AI 评测体系建表 —— eval_cases 与 eval_runs。

两张表承载「题库 + 评测运行」。运行表用 ``kind`` 判别列合并旧系统的
``RAGEvaluationRun`` / ``RAGQualityEvaluationRun`` / ``RAGASAutoEvaluationRun``
三张表：三者的生命周期列完全同构，差异只在产出哪些指标，而指标是稀疏的
（一次检索评测没有 faithfulness，一次 RAGAS 没有 refusal_accuracy）。
合并成一张表后，新增评测类型不必加列、不必新迁移，见 models/evaluation.py 说明。

``metrics`` / ``results`` / ``retrieval_config`` 是 JSONB；模型侧用
``JSONBText`` TypeDecorator（非 PG 后端退化 TEXT），迁移里按物理类型写 JSONB，
与 0001 中 ``structured_data`` / ``emergency_contact`` 的写法一致。

``expected_type`` 的取值域是 IMCS 知识模型的 ``doc_type``（condition /
guideline），不是旧系统的 drug/document——命中判定直接拿它和检索结果的
doc_type 比对，口径错位会让全量指标归零，因此注释显式钉死取值域。
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "eval_cases",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("expected_name", sa.String(300), nullable=False),
        sa.Column(
            "expected_type", sa.String(30), nullable=False,
            comment="取值域 = KnowledgeChunk.doc_type：condition / guideline",
        ),
        sa.Column("category", sa.String(50), nullable=False, server_default="未分类"),
        sa.Column("alternative_names", JSONB(), nullable=False, server_default="[]"),
        sa.Column("answer_keywords", JSONB(), nullable=False, server_default="[]"),
        sa.Column("citation_names", JSONB(), nullable=False, server_default="[]"),
        sa.Column("expected_refusal", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_eval_case_category", "eval_cases", ["category"])
    op.create_index("idx_eval_case_type", "eval_cases", ["expected_type"])

    op.create_table(
        "eval_runs",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("kind", sa.String(20), nullable=False, comment="retrieval / quality / ragas"),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("total_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_count", sa.Integer(), nullable=False, server_default="0"),
        # 检索层
        sa.Column("top1_accuracy", sa.Numeric(6, 4), nullable=True),
        sa.Column("recall_at_3", sa.Numeric(6, 4), nullable=True),
        sa.Column("precision_at_3", sa.Numeric(6, 4), nullable=True),
        # 质量层
        sa.Column("answer_accuracy", sa.Numeric(6, 4), nullable=True),
        sa.Column("citation_accuracy", sa.Numeric(6, 4), nullable=True),
        sa.Column("refusal_accuracy", sa.Numeric(6, 4), nullable=True),
        # RAGAS 四指标
        sa.Column("faithfulness", sa.Numeric(6, 4), nullable=True),
        sa.Column("answer_relevancy", sa.Numeric(6, 4), nullable=True),
        sa.Column("context_precision", sa.Numeric(6, 4), nullable=True),
        sa.Column("context_recall", sa.Numeric(6, 4), nullable=True),
        sa.Column("metrics", JSONB(), nullable=True, comment="按 category 下钻的指标"),
        sa.Column("results", JSONB(), nullable=True, comment="逐题明细，定位回归在哪一题"),
        sa.Column("retrieval_config", JSONB(), nullable=True, comment="运行时检索参数快照"),
        sa.Column("knowledge_hash", sa.String(64), nullable=True),
        sa.Column("knowledge_document_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("ai_task_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("triggered_by", PGUUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_eval_run_kind_created", "eval_runs", ["kind", "created_at"])
    op.create_index("idx_eval_run_status", "eval_runs", ["status", "created_at"])


def downgrade() -> None:
    op.drop_index("idx_eval_run_status", table_name="eval_runs")
    op.drop_index("idx_eval_run_kind_created", table_name="eval_runs")
    op.drop_table("eval_runs")
    op.drop_index("idx_eval_case_type", table_name="eval_cases")
    op.drop_index("idx_eval_case_category", table_name="eval_cases")
    op.drop_table("eval_cases")
