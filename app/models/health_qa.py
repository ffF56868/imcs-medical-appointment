"""健康咨询问答日志模型（阶段 7 批次 0）。

``health_qa_logs`` 记录每一次流式健康咨询的问答全貌：问题、最终回答、
引用来源、token 用量、首 token 延迟、是否走了急症短路。

**定位是 best-effort 审计日志，不是计费依据**：写入发生在 SSE 流结束后的
BackgroundTask 里（见 api/v1/health_qa.py），客户端中途断开时 Starlette 仍会
执行 background task，但 worker 被强杀时这条记录会丢。医疗问答的可追溯性
需要它，但把「必须落库」的强一致要求压在一条流式旁路日志上并不划算——
真正的用量与成本核算走 ``llm_call_logs``（由 LLM 服务层同步写入，不依赖本表）。

急症短路（``is_urgent_short_circuit``）单独建列而不是塞进 JSON：这是医疗安全
的关键指标，「有多少咨询被急症拦截直接转就医指引」需要能直接 group by 统计，
放 JSON 里就得每次解析，看板上做不出曲线。
"""

from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKey
from app.models.types import JSONBText


class HealthQALog(UUIDPrimaryKey, TimestampMixin, Base):
    """一次健康咨询问答的审计记录。"""

    __tablename__ = "health_qa_logs"

    # 与 llm_call_logs / rag_retrieval_logs / workflow_traces 同一个 trace_id，
    # 监控接口据此把「这次咨询」的检索、LLM 调用、工作流串成一条链路回放。
    trace_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), index=True, nullable=True
    )

    question: Mapped[str] = mapped_column(Text(), nullable=False)
    # 最终回答全文。流式场景下由服务端把逐 chunk 的 text 累加后落库，
    # 前端不需要回传——回传会让「患者看到的」与「系统记录的」可能不一致。
    answer_text: Mapped[str | None] = mapped_column(Text(), nullable=True)

    # 引用来源投影：[{chunk_id,title,source,doc_type,rrf_score}]。
    # 只存展示所需字段，不存切块正文（正文在 knowledge_chunks，按 chunk_id 可回查）。
    # 列名刻意不叫 ``references``：那是 PostgreSQL 保留字，SQLAlchemy 虽会自动
    # 加引号，但之后任何手写 SQL / 看板查询都得记得引号，是个无谓的坑。
    ref_sources: Mapped[list | None] = mapped_column(JSONBText(), nullable=True)

    prompt_tokens: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    cost_usd: Mapped[float | None] = mapped_column(Float(), nullable=True)
    first_token_ms: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)

    # 实际使用的模型名（流式 final chunk 才给真实值，降级时可能与请求值不同）
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # 降级层级，取值同 DegradationLevel（0 正常 … 4 兜底话术）
    degradation_level: Mapped[int | None] = mapped_column(Integer(), nullable=True)

    # 命中的急症关键词（JSON 数组）；为空表示未触发短路
    urgent_keywords: Mapped[list | None] = mapped_column(JSONBText(), nullable=True)
    is_urgent_short_circuit: Mapped[bool] = mapped_column(
        Boolean(), nullable=False, default=False, index=True
    )
    # 检索零结果：医疗安全取向下不给通用建议，直接引导就诊，需要单独统计
    is_zero_result: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)

    # 客户端提前断开（前端 abort / 关页面）。与「生成失败」区分开：
    # 前者是用户行为，后者才是质量信号，混在一起会让失败率失真。
    client_disconnected: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)
    error_message: Mapped[str | None] = mapped_column(Text(), nullable=True)
