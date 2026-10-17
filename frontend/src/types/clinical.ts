/**
 * 业务域类型：分诊、排班、预约、就诊、检查、处方、收费、发药、监控、评测。
 * 字段名逐字对齐后端 app/schemas/*，错一个字段前端就静默拿不到值。
 */

import type { AmountFen, PageReq, WithTransitions } from './common'

// ==================== 分诊 ====================
export interface EvidenceOut {
  title: string
  source: string
  source_tier: string
  snippet: string
  rrf_score: number
}
export interface SuggestedSlot {
  schedule_id: string
  doctor_name: string
  doctor_title: string | null
  schedule_date: string
  time_slot: string
  remaining_slots: number
  registration_fee: AmountFen
}
export interface TriageResponse {
  triage_id: string
  urgency: 'routine' | 'urgent' | 'stat'
  is_emergency: boolean
  matched_emergency_keywords: string[]
  recommended_department_id: string | null
  recommended_department_name: string | null
  recommended_department_code: string | null
  confidence: number | null
  reason: string | null
  advice: string | null
  decision_source: 'ai' | 'rule' | 'hybrid' | 'emergency'
  degraded: boolean
  extracted_symptoms: Record<string, unknown> | null
  evidence: EvidenceOut[]
  llm_model: string | null
  trace_id: string | null
  suggested_slots: SuggestedSlot[]
}

// ==================== 排班 ====================
export interface ScheduleWithDoctor {
  id: string
  doctor_id: string
  doctor_name: string
  doctor_title: string | null
  doctor_specialty: string | null
  department_id: string
  department_name: string
  department_code: string
  schedule_date: string
  time_slot: string
  slot_period: 'am' | 'pm'
  total_slots: number
  booked_slots: number
  remaining_slots: number
  status: 'open' | 'full' | 'closed'
  registration_fee: AmountFen
  version: number
}

// ==================== 预约 ====================
export interface Appointment extends WithTransitions {
  id: string
  patient_id: string
  doctor_id: string
  doctor_name: string
  department_id: string | null
  department_name: string | null
  schedule_id: string
  appointment_date: string
  time_slot: string
  status: 'booked' | 'checked_in' | 'in_progress' | 'completed' | 'cancelled' | 'no_show'
  source: 'self' | 'triage_referral'
  chief_complaint: string | null
  triage_summary: string | null
  triage_result_id: string | null
  registration_fee: AmountFen
  checked_in_at: string | null
  cancelled_at: string | null
  created_at: string
  updated_at: string
}

// ==================== 就诊 / 病历 ====================
export interface Visit extends WithTransitions {
  id: string
  appointment_id: string | null
  patient_id: string
  doctor_id: string
  doctor_name: string | null
  chief_complaint: string | null
  present_illness: string | null
  past_history: string | null
  physical_exam: string | null
  diagnosis: string | null
  icd_codes: string[]
  treatment_advice: string | null
  status: 'in_progress' | 'documented' | 'closed'
  started_at: string | null
  ended_at: string | null
  created_at: string
  updated_at: string
  exam_request_count: number
}

// ==================== 检查 ====================
export interface AbnormalItem {
  name: string
  value: string | number
  unit: string | null
  ref_low: number | null
  ref_high: number | null
  flag: string | null
}
export interface AIInterpretation {
  ai_status: 'pending' | 'processing' | 'completed' | 'failed' | 'skipped'
  summary: string | null
  suggestions: string[]
  severity: 'low' | 'medium' | 'high' | null
  confidence: number | null
  model: string | null
  provider: string | null
  ai_task_id: string | null
  interpreted_at: string | null
  need_doctor_review: boolean
}
export interface ExamReport {
  id: string
  request_id: string
  patient_id: string
  lab_tech_id: string | null
  exam_type: string
  exam_name: string
  urgency: 'routine' | 'urgent' | 'stat'
  request_status: string
  raw_content: string
  structured_data: Record<string, unknown> | null
  abnormal_flags: string[]
  abnormal_items: AbnormalItem[]
  critical_values: unknown[]
  critical: boolean
  critical_message: string | null
  conclusion: string | null
  created_at: string
  ai: AIInterpretation
}
export interface ExamRequest {
  id: string
  visit_id: string
  patient_id: string
  doctor_id: string
  exam_type: string
  exam_name: string
  body_part: string | null
  clinical_info: string | null
  urgency: 'routine' | 'urgent' | 'stat'
  status: 'ordered' | 'received' | 'processing' | 'reported' | 'reviewed'
  created_at: string
  report_id: string | null
  ai_status: string | null
  /** 检查费（分）。表外项目为 0，由收费侧核价。 */
  fee_amount?: number
}

// ==================== 处方 ====================
export interface SafetyCheck {
  rule: 'allergy_conflict' | 'duplicate_therapy' | 'daily_dose_excess'
  level: 'block' | 'warning'
  drug_code: string
  drug_name: string
  message: string
  detail: Record<string, unknown>
}
export interface PrescriptionItem {
  id?: string
  drug_code: string
  drug_name: string
  specification: string | null
  dosage: string | null
  frequency: string | null
  route: string | null
  duration_days: number | null
  quantity: number
  unit: string
  unit_price: number // 元（唯一用元的字段）
  usage_instructions: string | null
}
export interface Prescription extends WithTransitions {
  id: string
  visit_id: string
  patient_id: string
  doctor_id: string
  doctor_name: string | null
  status: 'draft' | 'confirmed' | 'paid' | 'dispensed' | 'cancelled'
  diagnosis: string | null
  notes: string | null
  total_amount: AmountFen
  items: PrescriptionItem[]
  safety_checks: SafetyCheck[]
  blocked_count: number
  warning_count: number
  safety_override_reason: string | null
  confirmed_at: string | null
  cancelled_at: string | null
  created_at: string
  updated_at: string
}

// ==================== 收费 ====================
export interface ChargeItem {
  type: 'prescription' | 'exam' | 'other'
  prescription_id: string | null
  exam_request_id: string | null
  name: string | null
  amount: AmountFen
}
export interface ChargeOrder extends WithTransitions {
  id: string
  visit_id: string
  patient_id: string
  status: 'pending' | 'paid' | 'refunded' | 'cancelled'
  items: ChargeItem[]
  total_amount: AmountFen
  paid_amount: AmountFen
  payment_channel: string | null
  cashier_id: string | null
  paid_at: string | null
  refunded_at: string | null
  refund_reason: string | null
  receipt_no: string | null
  created_at: string
  updated_at: string
}

// ==================== 发药 ====================
export interface DispensingRecord {
  id: string
  prescription_id: string
  pharmacist_id: string | null
  verified: boolean
  verification_notes: string | null
  dispensed_at: string | null
  notes: string | null
  created_at: string
  updated_at: string
  patient_id: string
  prescription_status: string
  total_amount: AmountFen
}

// ==================== 监控 ====================
export interface MetricPoint {
  bucket_time: string
  metric_name: string
  metric_value: number
  sample_count: number
  tags: Record<string, unknown>
}
export interface LLMCall {
  id: number
  trace_id: string | null
  scenario: string
  provider: string
  model: string
  requested_model: string | null
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  cost_usd: number | null
  latency_ms: number
  first_token_ms: number | null
  status: string
  error_type: string | null
  degradation_level: number | null
  moderation_blocked: boolean
  is_stream: boolean
  user_id: string | null
  workflow_node: string | null
  created_at: string
}

// ==================== 评测 ====================
export interface EvalRun {
  id: string
  kind: 'retrieval' | 'quality' | 'ragas'
  status: string
  total_count: number
  completed_count: number
  top1_accuracy: number | null
  recall_at_3: number | null
  precision_at_3: number | null
  answer_accuracy: number | null
  citation_accuracy: number | null
  refusal_accuracy: number | null
  faithfulness: number | null
  answer_relevancy: number | null
  context_precision: number | null
  context_recall: number | null
  metrics: Record<string, unknown> | null
  created_at: string
}

export type { PageReq }
