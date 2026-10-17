"""阶段 7 批次 0：健康咨询问答日志表 —— health_qa_logs。

配合 ``POST /api/v1/health-qa/stream``（SSE 流式健康咨询）落地审计记录。

设计取舍见 models/health_qa.py 的模块 docstring，这里只钉两条与迁移相关的：

1. ``references`` / ``urgent_keywords`` 用 JSONB 物理类型，模型侧是
   ``JSONBText`` TypeDecorator（非 PG 后端退化 TEXT），与 0001 的
   ``structured_data``、0003 的 ``metrics`` 写法一致。
2. ``is_urgent_short_circuit`` 建索引：「有多少咨询被急症拦截直接转就医指引」
   是医疗安全的关键指标，看板要按它 group by 出曲线，不能每次解析 JSON。

``trace_id`` 用 String(36) 而非 UUID 类型，与 llm_call_logs /
rag_retrieval_logs / workflow_traces 保持同一口径——监控接口按 trace_id
跨这四张表做全链路回放，类型不一致会让联查条件失效。
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "health_qa_logs",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("trace_id", sa.String(36), nullable=True),
        sa.Column("user_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer_text", sa.Text(), nullable=True),
        # 不叫 references：PostgreSQL 保留字，手写 SQL 需额外引号
        sa.Column(
            "ref_sources", JSONB(), nullable=True,
            comment="引用来源投影 [{chunk_id,title,source,doc_type,rrf_score}]",
        ),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("first_token_ms", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column(
            "degradation_level", sa.Integer(), nullable=True,
            comment="取值同 DegradationLevel：0 正常 … 4 兜底话术",
        ),
        sa.Column("urgent_keywords", JSONB(), nullable=True),
        sa.Column(
            "is_urgent_short_circuit", sa.Boolean(), nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "is_zero_result", sa.Boolean(), nullable=False, server_default=sa.false(),
            comment="检索零结果：医疗安全取向下不给通用建议，直接引导就诊",
        ),
        sa.Column(
            "client_disconnected", sa.Boolean(), nullable=False, server_default=sa.false(),
            comment="客户端提前断开，与生成失败区分，避免失败率失真",
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
    )
    op.create_index("idx_health_qa_trace_id", "health_qa_logs", ["trace_id"])
    op.create_index("idx_health_qa_user_id", "health_qa_logs", ["user_id"])
    op.create_index("idx_health_qa_urgent", "health_qa_logs", ["is_urgent_short_circuit"])
    op.create_index("idx_health_qa_created_at", "health_qa_logs", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_health_qa_created_at", table_name="health_qa_logs")
    op.drop_index("idx_health_qa_urgent", table_name="health_qa_logs")
    op.drop_index("idx_health_qa_user_id", table_name="health_qa_logs")
    op.drop_index("idx_health_qa_trace_id", table_name="health_qa_logs")
    op.drop_table("health_qa_logs")
