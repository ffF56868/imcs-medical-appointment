"""初始全量建表迁移。

按外键依赖顺序创建全部 24 张业务表，并建立所有索引、约束与 PG 扩展。
向量索引用 ivfflat（lists=100），全文检索用 GIN 索引。

Revision ID: 0001
Revises: None
Create Date: 2026-10-08
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID as PGUUID

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ---------- PG 扩展 ----------
    # vector 扩展提供 pgvector 类型与距离运算符
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ========== 1. 基础字典表（无外键依赖） ==========

    # --- departments ---
    op.create_table(
        "departments",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(50), nullable=False, unique=True),
        sa.Column("code", sa.String(20), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("symptom_keywords", ARRAY(sa.Text()), nullable=True),
        sa.Column("parent_id", PGUUID(as_uuid=True), sa.ForeignKey("departments.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_departments_created_at", "departments", ["created_at"])

    # --- roles ---
    op.create_table(
        "roles",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("name", sa.String(30), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
    )

    # --- permissions ---
    op.create_table(
        "permissions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("resource", sa.String(50), nullable=False),
        sa.Column("action", sa.String(20), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.UniqueConstraint("resource", "action", name="uq_permission_resource_action"),
    )

    # --- role_permissions（关联表） ---
    op.create_table(
        "role_permissions",
        sa.Column("role_id", sa.Integer(), sa.ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("permission_id", sa.Integer(), sa.ForeignKey("permissions.id", ondelete="CASCADE"), primary_key=True),
    )

    # ========== 2. 用户与排班 ==========

    # --- users ---
    op.create_table(
        "users",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(50), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("real_name", sa.String(50), nullable=False),
        sa.Column("phone", sa.String(20), nullable=True, unique=True),
        sa.Column("email", sa.String(200), nullable=True, unique=True),
        sa.Column("role_id", sa.Integer(), sa.ForeignKey("roles.id"), nullable=False),
        sa.Column("department_id", PGUUID(as_uuid=True), sa.ForeignKey("departments.id"), nullable=True),
        sa.Column("title", sa.String(50), nullable=True),
        sa.Column("specialty", sa.String(200), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_users_created_at", "users", ["created_at"])
    op.create_index("idx_users_role_active", "users", ["role_id", "is_active"])
    op.create_index("idx_users_department", "users", ["department_id"])

    # --- doctor_schedules ---
    op.create_table(
        "doctor_schedules",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("doctor_id", PGUUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("department_id", PGUUID(as_uuid=True), sa.ForeignKey("departments.id"), nullable=False),
        sa.Column("schedule_date", sa.Date(), nullable=False),
        sa.Column("time_slot", sa.String(20), nullable=False),
        sa.Column("slot_period", sa.String(10), nullable=False, server_default=sa.text("'am'"), comment="am / pm，便于前端按上下午分组展示"),
        sa.Column("total_slots", sa.Integer(), nullable=False, server_default=sa.text("30")),
        sa.Column("booked_slots", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("status", sa.String(10), nullable=False, server_default=sa.text("'open'")),
        sa.Column("registration_fee", sa.Integer(), nullable=False, server_default=sa.text("0"), comment="挂号费，单位分，避免浮点误差"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("doctor_id", "schedule_date", "time_slot", name="uq_schedule_doctor_date_slot"),
        sa.CheckConstraint("booked_slots >= 0", name="ck_booked_slots_non_negative"),
        sa.CheckConstraint("total_slots > 0", name="ck_total_slots_positive"),
        sa.CheckConstraint("booked_slots <= total_slots", name="ck_booked_within_total"),
    )
    op.create_index("ix_doctor_schedules_created_at", "doctor_schedules", ["created_at"])
    op.create_index("idx_schedule_date_status", "doctor_schedules", ["schedule_date", "status"])

    # ========== 3. 患者档案 ==========

    # --- patient_profiles ---
    op.create_table(
        "patient_profiles",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", PGUUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("gender", sa.String(1), nullable=True),
        sa.Column("birth_date", sa.Date(), nullable=True),
        sa.Column("id_card_no", sa.String(18), nullable=True),
        sa.Column("blood_type", sa.String(5), nullable=True),
        sa.Column("allergy_history", ARRAY(sa.Text()), nullable=True),
        sa.Column("chronic_diseases", ARRAY(sa.Text()), nullable=True),
        sa.Column("medical_notes", sa.Text(), nullable=True),
        sa.Column("emergency_contact", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("gender IS NULL OR gender IN ('M', 'F')", name="ck_patient_gender_valid"),
    )
    op.create_index("ix_patient_profiles_created_at", "patient_profiles", ["created_at"])

    # ========== 4. 临床业务 ==========

    # --- appointments ---
    op.create_table(
        "appointments",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("patient_id", PGUUID(as_uuid=True), sa.ForeignKey("patient_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("doctor_id", PGUUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("department_id", PGUUID(as_uuid=True), sa.ForeignKey("departments.id"), nullable=False),
        sa.Column("schedule_id", PGUUID(as_uuid=True), sa.ForeignKey("doctor_schedules.id"), nullable=False),
        sa.Column("appointment_date", sa.Date(), nullable=False),
        sa.Column("time_slot", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'booked'")),
        sa.Column("source", sa.String(20), nullable=False, server_default=sa.text("'self'")),
        sa.Column("triage_summary", sa.Text(), nullable=True),
        sa.Column("triage_result_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("chief_complaint", sa.Text(), nullable=True, comment="患者主诉"),
        sa.Column("registration_fee", sa.Integer(), nullable=False, server_default=sa.text("0"), comment="挂号费，单位分"),
        sa.Column("checked_in_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_appointments_created_at", "appointments", ["created_at"])
    op.create_index("idx_appt_patient_date", "appointments", ["patient_id", "appointment_date"])
    op.create_index("idx_appt_doctor_date_status", "appointments", ["doctor_id", "appointment_date", "status"])
    # 部分唯一索引：只约束未取消的预约
    op.create_index(
        "uq_appt_patient_slot_active",
        "appointments",
        ["patient_id", "schedule_id"],
        unique=True,
        postgresql_where=sa.text("status NOT IN ('cancelled', 'no_show')"),
    )

    # --- visits ---
    op.create_table(
        "visits",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("appointment_id", PGUUID(as_uuid=True), sa.ForeignKey("appointments.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("patient_id", PGUUID(as_uuid=True), sa.ForeignKey("patient_profiles.id"), nullable=False),
        sa.Column("doctor_id", PGUUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("chief_complaint", sa.Text(), nullable=False),
        sa.Column("present_illness", sa.Text(), nullable=True),
        sa.Column("past_history", sa.Text(), nullable=True),
        sa.Column("physical_exam", sa.Text(), nullable=True),
        sa.Column("diagnosis", sa.Text(), nullable=True),
        sa.Column("icd_codes", ARRAY(sa.Text()), nullable=True),
        sa.Column("treatment_advice", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'in_progress'")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_visits_created_at", "visits", ["created_at"])
    op.create_index("idx_visit_doctor_status", "visits", ["doctor_id", "status"])
    op.create_index("idx_visit_patient", "visits", ["patient_id"])

    # --- exam_requests ---
    op.create_table(
        "exam_requests",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("visit_id", PGUUID(as_uuid=True), sa.ForeignKey("visits.id", ondelete="CASCADE"), nullable=False),
        sa.Column("patient_id", PGUUID(as_uuid=True), sa.ForeignKey("patient_profiles.id"), nullable=False),
        sa.Column("doctor_id", PGUUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("exam_type", sa.String(50), nullable=False),
        sa.Column("exam_name", sa.String(100), nullable=False),
        sa.Column("body_part", sa.String(100), nullable=True),
        sa.Column("clinical_info", sa.Text(), nullable=True, comment="临床诊断信息，供检验科参考"),
        sa.Column("urgency", sa.String(10), nullable=False, server_default=sa.text("'routine'")),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'ordered'")),
        sa.Column("is_charged", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("fee_amount", sa.Integer(), nullable=False, server_default=sa.text("0"), comment="检查费，单位分"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_exam_requests_created_at", "exam_requests", ["created_at"])
    op.create_index("idx_exam_request_visit", "exam_requests", ["visit_id"])
    op.create_index("idx_exam_request_status", "exam_requests", ["status"])
    op.create_index("idx_exam_request_lab_status", "exam_requests", ["status", "created_at"])

    # --- exam_reports ---
    op.create_table(
        "exam_reports",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("request_id", PGUUID(as_uuid=True), sa.ForeignKey("exam_requests.id", ondelete="CASCADE"), nullable=False),
        sa.Column("patient_id", PGUUID(as_uuid=True), sa.ForeignKey("patient_profiles.id"), nullable=False),
        sa.Column("lab_tech_id", PGUUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("raw_content", sa.Text(), nullable=False),
        sa.Column("structured_data", JSONB(), nullable=True),
        sa.Column("abnormal_flags", ARRAY(sa.Text()), nullable=True),
        sa.Column("conclusion", sa.Text(), nullable=True, comment="检验科结论"),
        sa.Column("ai_interpretation", sa.Text(), nullable=True),
        sa.Column("ai_summary", sa.Text(), nullable=True, comment="AI 解读摘要"),
        sa.Column("ai_suggestions", ARRAY(sa.Text()), nullable=True, comment="AI 给出的后续建议"),
        sa.Column("ai_confidence", sa.Numeric(4, 3), nullable=True),
        sa.Column("ai_model", sa.String(50), nullable=True),
        sa.Column("ai_provider", sa.String(20), nullable=True),
        sa.Column("ai_task_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("ai_status", sa.String(20), nullable=False, server_default=sa.text("'pending'"), comment="pending / processing / completed / failed / skipped"),
        sa.Column("interpreted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("request_id", name="uq_exam_report_request"),
    )
    op.create_index("ix_exam_reports_created_at", "exam_reports", ["created_at"])
    op.create_index("idx_exam_report_lab_tech", "exam_reports", ["lab_tech_id"])

    # ========== 5. 处方与发药 ==========

    # --- prescriptions ---
    op.create_table(
        "prescriptions",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("visit_id", PGUUID(as_uuid=True), sa.ForeignKey("visits.id", ondelete="CASCADE"), nullable=False),
        sa.Column("patient_id", PGUUID(as_uuid=True), sa.ForeignKey("patient_profiles.id"), nullable=False),
        sa.Column("doctor_id", PGUUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'draft'")),
        sa.Column("diagnosis", sa.Text(), nullable=True, comment="处方对应的临床诊断"),
        sa.Column("notes", sa.Text(), nullable=True, comment="医嘱备注"),
        sa.Column("total_amount", sa.Integer(), nullable=False, server_default=sa.text("0"), comment="处方总金额，单位分"),
        sa.Column("safety_checks", JSONB(), nullable=True, comment="过敏史与相互作用校验结果"),
        sa.Column("safety_override_reason", sa.Text(), nullable=True, comment="医生强制开具时填写的理由"),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_prescriptions_created_at", "prescriptions", ["created_at"])
    op.create_index("idx_prescription_visit", "prescriptions", ["visit_id"])
    op.create_index("idx_prescription_status_created", "prescriptions", ["status", "created_at"])
    # 部分索引：已缴费待发药
    op.create_index(
        "idx_prescription_pending_dispense",
        "prescriptions",
        ["created_at"],
        postgresql_where=sa.text("status = 'paid'"),
    )

    # --- prescription_items ---
    op.create_table(
        "prescription_items",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("prescription_id", PGUUID(as_uuid=True), sa.ForeignKey("prescriptions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("drug_code", sa.String(30), nullable=False),
        sa.Column("drug_name", sa.String(100), nullable=False),
        sa.Column("specification", sa.String(100), nullable=True, comment="规格，如 0.25g*24片"),
        sa.Column("dosage", sa.String(50), nullable=True),
        sa.Column("frequency", sa.String(30), nullable=True),
        sa.Column("route", sa.String(30), nullable=True),
        sa.Column("duration_days", sa.Integer(), nullable=True),
        sa.Column("quantity", sa.Numeric(10, 2), nullable=False),
        sa.Column("unit", sa.String(20), nullable=False, comment="单位，如 盒/瓶/支"),
        sa.Column("unit_price", sa.Numeric(10, 2), nullable=False, server_default=sa.text("0")),
        sa.Column("usage_instructions", sa.Text(), nullable=True, comment="用药指导"),
        sa.CheckConstraint("quantity > 0", name="ck_rx_item_quantity_positive"),
        sa.UniqueConstraint("prescription_id", "drug_code", name="uq_rx_item_drug"),
    )
    op.create_index("idx_prescription_item_rx", "prescription_items", ["prescription_id"])

    # --- dispensing_records ---
    op.create_table(
        "dispensing_records",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("prescription_id", PGUUID(as_uuid=True), sa.ForeignKey("prescriptions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("pharmacist_id", PGUUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("verified", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("verification_notes", sa.Text(), nullable=True),
        sa.Column("dispensed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("prescription_id", name="uq_dispensing_prescription"),
    )
    op.create_index("ix_dispensing_records_created_at", "dispensing_records", ["created_at"])
    op.create_index("idx_dispensing_pharmacist", "dispensing_records", ["pharmacist_id"])

    # ========== 6. 收费 ==========

    # --- charge_orders ---
    op.create_table(
        "charge_orders",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("visit_id", PGUUID(as_uuid=True), sa.ForeignKey("visits.id", ondelete="CASCADE"), nullable=False),
        sa.Column("patient_id", PGUUID(as_uuid=True), sa.ForeignKey("patient_profiles.id"), nullable=False),
        sa.Column("items", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("total_amount", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("paid_amount", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("payment_channel", sa.String(20), nullable=True),
        sa.Column("cashier_id", PGUUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("paid_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("refunded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("refund_reason", sa.Text(), nullable=True),
        sa.Column("receipt_no", sa.String(40), nullable=True, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("total_amount >= 0", name="ck_charge_amount_non_negative"),
    )
    op.create_index("ix_charge_orders_created_at", "charge_orders", ["created_at"])
    op.create_index("idx_charge_visit", "charge_orders", ["visit_id"])
    op.create_index("idx_charge_patient_status", "charge_orders", ["patient_id", "status"])
    # 部分索引：待缴费
    op.create_index(
        "idx_charge_pending",
        "charge_orders",
        ["created_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )

    # ========== 7. 知识库 ==========

    # --- knowledge_documents ---
    op.create_table(
        "knowledge_documents",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("doc_type", sa.String(30), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("source", sa.String(200), nullable=False, server_default=sa.text("'未标注来源'")),
        sa.Column("source_url", sa.String(2000), nullable=True),
        sa.Column("source_tier", sa.String(20), nullable=False, server_default=sa.text("'unverified'")),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("department_id", PGUUID(as_uuid=True), sa.ForeignKey("departments.id"), nullable=True),
        sa.Column("extra_metadata", JSONB(), nullable=True),
        sa.Column("is_reviewed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("index_status", sa.String(20), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_knowledge_documents_created_at", "knowledge_documents", ["created_at"])
    op.create_index("idx_knowledge_doc_type", "knowledge_documents", ["doc_type"])
    op.create_index("idx_knowledge_doc_content_hash", "knowledge_documents", ["content_hash"])

    # --- knowledge_chunks ---
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", PGUUID(as_uuid=True), sa.ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("page_number", sa.Integer(), nullable=True, comment="PDF 页码"),
        sa.Column("page_range", sa.String(20), nullable=True, comment="页码区间，如 12-15"),
        sa.Column("chunk_start_index", sa.Integer(), nullable=True),
        sa.Column("chunk_end_index", sa.Integer(), nullable=True),
        sa.Column("chunking_strategy", sa.String(40), nullable=False, server_default=sa.text("'recursive-chinese-v1'")),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("doc_type", sa.String(30), nullable=False),
        sa.Column("source", sa.String(200), nullable=False),
        sa.Column("source_tier", sa.String(20), nullable=False),
        sa.Column("extra_metadata", JSONB(), nullable=True),
        sa.Column("tsv", TSVECTOR(), nullable=True),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunk_doc_index"),
    )
    op.create_index("idx_chunk_document", "knowledge_chunks", ["document_id"])
    op.create_index("idx_chunk_content_hash", "knowledge_chunks", ["content_hash"])
    # GIN 索引用于 tsvector 全文检索
    op.execute("CREATE INDEX idx_chunk_tsv_gin ON knowledge_chunks USING gin (tsv)")

    # --- vector_embeddings ---
    op.create_table(
        "vector_embeddings",
        sa.Column("chunk_id", PGUUID(as_uuid=True), sa.ForeignKey("knowledge_chunks.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("embedding", Vector(dim=1536), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_vector_chunk", "vector_embeddings", ["chunk_id"])
    op.create_index("idx_vector_model_version", "vector_embeddings", ["model_version"])
    # ivfflat 向量索引（余弦距离），lists=100 适合中等规模数据
    op.execute(
        "CREATE INDEX idx_vector_embedding_ivfflat "
        "ON vector_embeddings USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )

    # ========== 8. AI 可观测性 ==========

    # --- ai_tasks ---
    op.create_table(
        "ai_tasks",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("task_type", sa.String(40), nullable=False),
        sa.Column("ref_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("celery_task_id", sa.String(64), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("progress", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("progress_message", sa.String(200), nullable=True),
        sa.Column("payload", JSONB(), nullable=True, comment="任务入参快照"),
        sa.Column("result", JSONB(), nullable=True, comment="任务产出"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("idempotency_key", sa.String(128), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("queued_ms", sa.Integer(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_ai_tasks_created_at", "ai_tasks", ["created_at"])
    op.create_index("idx_ai_task_type_status", "ai_tasks", ["task_type", "status"])
    op.create_index("idx_ai_task_ref", "ai_tasks", ["task_type", "ref_id"])
    op.create_index("idx_ai_task_celery_id", "ai_tasks", ["celery_task_id"])
    op.create_index("ix_ai_tasks_idempotency_key", "ai_tasks", ["idempotency_key"])

    # --- llm_call_logs ---
    op.create_table(
        "llm_call_logs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("trace_id", sa.String(36), nullable=True),
        sa.Column("scenario", sa.String(30), nullable=False),
        sa.Column("provider", sa.String(20), nullable=False, comment="openai / dashscope"),
        sa.Column("model", sa.String(50), nullable=False),
        sa.Column("requested_model", sa.String(50), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("total_tokens", sa.Integer(), sa.Computed("prompt_tokens + completion_tokens", persisted=True), nullable=False),
        sa.Column("cost_usd", sa.Numeric(10, 6), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("first_token_ms", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'success'")),
        sa.Column("error_type", sa.String(60), nullable=True),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("degradation_level", sa.Integer(), nullable=True),
        sa.Column("moderation_blocked", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("budget_exceeded", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_stream", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("user_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("workflow_node", sa.String(60), nullable=True),
        sa.Column("ai_task_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_llm_log_created", "llm_call_logs", ["created_at"])
    op.create_index("idx_llm_log_scenario", "llm_call_logs", ["scenario", "created_at"])
    op.create_index("idx_llm_log_provider_model", "llm_call_logs", ["provider", "model"])
    op.create_index("idx_llm_log_trace", "llm_call_logs", ["trace_id"])
    op.create_index("idx_llm_log_status", "llm_call_logs", ["status", "created_at"])
    op.create_index("idx_llm_log_user", "llm_call_logs", ["user_id", "created_at"])
    op.create_index("ix_llm_call_logs_trace_id", "llm_call_logs", ["trace_id"])

    # --- rag_retrieval_logs ---
    op.create_table(
        "rag_retrieval_logs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("trace_id", sa.String(36), nullable=True),
        sa.Column("user_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=True),
        sa.Column("rewritten_query", sa.Text(), nullable=True),
        sa.Column("vector_hit_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("keyword_hit_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("both_hit_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("merged_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("final_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("is_zero_result", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("top_score", sa.Numeric(8, 6), nullable=True),
        sa.Column("vector_latency_ms", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("keyword_latency_ms", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("rrf_latency_ms", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("rerank_latency_ms", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("rerank_channel", sa.String(20), nullable=True),
        sa.Column("rrf_k", sa.Integer(), nullable=False, server_default=sa.text("60")),
        sa.Column("candidates", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_rag_log_created", "rag_retrieval_logs", ["created_at"])
    op.create_index("idx_rag_log_trace", "rag_retrieval_logs", ["trace_id"])
    op.create_index("ix_rag_retrieval_logs_trace_id", "rag_retrieval_logs", ["trace_id"])

    # --- workflow_traces ---
    op.create_table(
        "workflow_traces",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("trace_id", sa.String(36), nullable=False),
        sa.Column("workflow_name", sa.String(40), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'running'")),
        sa.Column("steps", JSONB(), nullable=True),
        sa.Column("node_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("executed_node_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("total_latency_ms", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("total_cost_usd", sa.Numeric(10, 6), nullable=True),
        sa.Column("branch_taken", sa.String(40), nullable=True),
        sa.Column("final_output", JSONB(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("user_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("ref_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_wf_trace_workflow", "workflow_traces", ["workflow_name", "created_at"])
    op.create_index("idx_wf_trace_trace_id", "workflow_traces", ["trace_id"])
    op.create_index("idx_wf_trace_status", "workflow_traces", ["status", "created_at"])

    # --- ai_metrics ---
    op.create_table(
        "ai_metrics",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("bucket_time", sa.DateTime(timezone=True), nullable=False, comment="分钟级时间桶，向下取整到分钟"),
        sa.Column("metric_name", sa.String(50), nullable=False),
        sa.Column("metric_value", sa.Numeric(16, 4), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("tags", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_ai_metric_name_time", "ai_metrics", ["metric_name", "bucket_time"])
    op.create_index("idx_ai_metric_bucket", "ai_metrics", ["bucket_time"])

    # --- triage_results ---
    op.create_table(
        "triage_results",
        sa.Column("id", PGUUID(as_uuid=True), primary_key=True),
        sa.Column("patient_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("user_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("trace_id", sa.String(36), nullable=True),
        sa.Column("chief_complaint", sa.Text(), nullable=False),
        sa.Column("extracted_symptoms", JSONB(), nullable=True),
        sa.Column("recommended_department_id", PGUUID(as_uuid=True), sa.ForeignKey("departments.id"), nullable=True),
        sa.Column("recommended_department_name", sa.String(50), nullable=True),
        sa.Column("urgency", sa.String(10), nullable=False, server_default=sa.text("'routine'")),
        sa.Column("is_emergency", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("matched_emergency_keywords", JSONB(), nullable=True),
        sa.Column("advice", sa.Text(), nullable=True, comment="给患者的就医建议"),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=True),
        sa.Column("decision_source", sa.String(20), nullable=False, server_default=sa.text("'ai'")),
        sa.Column("ai_task_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("llm_model", sa.String(50), nullable=True),
        sa.Column("converted_appointment_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("actual_department_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("feedback_rating", sa.Integer(), nullable=True, comment="患者对分诊的评分 1-5"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_triage_results_created_at", "triage_results", ["created_at"])
    op.create_index("idx_triage_patient", "triage_results", ["patient_id", "created_at"])
    op.create_index("idx_triage_urgency", "triage_results", ["urgency", "created_at"])
    op.create_index("ix_triage_results_trace_id", "triage_results", ["trace_id"])


def downgrade() -> None:
    # 逆序 drop 全部表（先 drop 有外键依赖的子表）
    op.drop_table("triage_results")
    op.drop_table("ai_metrics")
    op.drop_table("workflow_traces")
    op.drop_table("rag_retrieval_logs")
    op.drop_table("llm_call_logs")
    op.drop_table("ai_tasks")
    op.drop_table("vector_embeddings")
    op.drop_table("knowledge_chunks")
    op.drop_table("knowledge_documents")
    op.drop_table("charge_orders")
    op.drop_table("dispensing_records")
    op.drop_table("prescription_items")
    op.drop_table("prescriptions")
    op.drop_table("exam_reports")
    op.drop_table("exam_requests")
    op.drop_table("visits")
    op.drop_table("appointments")
    op.drop_table("patient_profiles")
    op.drop_table("doctor_schedules")
    op.drop_table("users")
    op.drop_table("role_permissions")
    op.drop_table("permissions")
    op.drop_table("roles")
    op.drop_table("departments")

    # 最后移除扩展
    op.execute("DROP EXTENSION IF EXISTS vector")
