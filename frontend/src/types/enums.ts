/**
 * 全局枚举与状态字典。
 *
 * 取值逐字对齐后端（app/models 与 app/schemas），错一个字就会静默失效——
 * 比如状态标签匹配不上就渲染成空白。用 `as const` 对象 + 派生联合类型，
 * 既能当类型用（`AppointmentStatus`），又能在运行时遍历（做 label/color 映射）。
 *
 * 每个枚举配一个 LABELS 中文映射：界面一律显示中文，不显示 status 原值。
 */

// ==================== 角色 ====================
export const RoleName = {
  PATIENT: 'patient',
  DOCTOR: 'doctor',
  LAB_TECH: 'lab_tech',
  CASHIER: 'cashier',
  PHARMACIST: 'pharmacist',
  ADMIN: 'admin',
} as const
export type RoleName = (typeof RoleName)[keyof typeof RoleName]

export const ROLE_LABELS: Record<string, string> = {
  patient: '患者',
  doctor: '医生',
  lab_tech: '检验科技师',
  cashier: '收费员',
  pharmacist: '药师',
  admin: '系统管理员',
}

// ==================== 预约状态 ====================
export const AppointmentStatus = {
  BOOKED: 'booked',
  CHECKED_IN: 'checked_in',
  IN_PROGRESS: 'in_progress',
  COMPLETED: 'completed',
  CANCELLED: 'cancelled',
  NO_SHOW: 'no_show',
} as const
export type AppointmentStatus = (typeof AppointmentStatus)[keyof typeof AppointmentStatus]

export const APPOINTMENT_STATUS_LABELS: Record<string, string> = {
  booked: '已预约',
  checked_in: '已签到',
  in_progress: '就诊中',
  completed: '已完成',
  cancelled: '已取消',
  no_show: '未到访',
}

// ==================== 就诊状态 ====================
export const VisitStatus = {
  IN_PROGRESS: 'in_progress',
  DOCUMENTED: 'documented',
  CLOSED: 'closed',
} as const
export type VisitStatus = (typeof VisitStatus)[keyof typeof VisitStatus]

export const VISIT_STATUS_LABELS: Record<string, string> = {
  in_progress: '接诊中',
  documented: '已写病历',
  closed: '已结束',
}

// ==================== 检查申请状态 ====================
export const ExamRequestStatus = {
  ORDERED: 'ordered',
  RECEIVED: 'received',
  PROCESSING: 'processing',
  REPORTED: 'reported',
  REVIEWED: 'reviewed',
} as const
export type ExamRequestStatus = (typeof ExamRequestStatus)[keyof typeof ExamRequestStatus]

export const EXAM_REQUEST_STATUS_LABELS: Record<string, string> = {
  ordered: '已开单',
  received: '已接收',
  processing: '检查中',
  reported: '已出报告',
  reviewed: '已审核',
}

// ==================== AI 解读状态 ====================
export const AiStatus = {
  PENDING: 'pending',
  PROCESSING: 'processing',
  COMPLETED: 'completed',
  FAILED: 'failed',
  SKIPPED: 'skipped',
} as const
export type AiStatus = (typeof AiStatus)[keyof typeof AiStatus]

export const AI_STATUS_LABELS: Record<string, string> = {
  pending: '待解读',
  processing: '解读中',
  completed: '已完成',
  failed: '解读失败',
  skipped: '已跳过',
}

// ==================== 处方状态 ====================
export const PrescriptionStatus = {
  DRAFT: 'draft',
  CONFIRMED: 'confirmed',
  PAID: 'paid',
  DISPENSED: 'dispensed',
  CANCELLED: 'cancelled',
} as const
export type PrescriptionStatus = (typeof PrescriptionStatus)[keyof typeof PrescriptionStatus]

export const PRESCRIPTION_STATUS_LABELS: Record<string, string> = {
  draft: '草稿',
  confirmed: '已开具',
  paid: '已缴费',
  dispensed: '已发药',
  cancelled: '已作废',
}

// ==================== 收费单状态 ====================
export const ChargeOrderStatus = {
  PENDING: 'pending',
  PAID: 'paid',
  REFUNDED: 'refunded',
  CANCELLED: 'cancelled',
} as const
export type ChargeOrderStatus = (typeof ChargeOrderStatus)[keyof typeof ChargeOrderStatus]

export const CHARGE_ORDER_STATUS_LABELS: Record<string, string> = {
  pending: '待缴费',
  paid: '已缴费',
  refunded: '已退费',
  cancelled: '已作废',
}

// ==================== 处方安全校验 ====================
export const SafetyLevel = {
  BLOCK: 'block',
  WARNING: 'warning',
} as const
export type SafetyLevel = (typeof SafetyLevel)[keyof typeof SafetyLevel]

export const SAFETY_LEVEL_LABELS: Record<string, string> = {
  block: '阻断',
  warning: '提醒',
}

export const SAFETY_RULE_LABELS: Record<string, string> = {
  allergy_conflict: '过敏冲突',
  duplicate_therapy: '重复用药',
  daily_dose_excess: '剂量超限',
}

// ==================== 紧急度 ====================
export const Urgency = {
  ROUTINE: 'routine',
  URGENT: 'urgent',
  STAT: 'stat',
} as const
export type Urgency = (typeof Urgency)[keyof typeof Urgency]

export const URGENCY_LABELS: Record<string, string> = {
  routine: '常规',
  urgent: '紧急',
  stat: '危急',
}

// ==================== 排班状态 / 时段 ====================
export const ScheduleStatus = {
  OPEN: 'open',
  FULL: 'full',
  CLOSED: 'closed',
} as const
export type ScheduleStatus = (typeof ScheduleStatus)[keyof typeof ScheduleStatus]

export const SCHEDULE_STATUS_LABELS: Record<string, string> = {
  open: '可预约',
  full: '已约满',
  closed: '已停诊',
}

export const SLOT_PERIOD_LABELS: Record<string, string> = {
  am: '上午',
  pm: '下午',
}

// ==================== 支付渠道 ====================
export const PAYMENT_CHANNEL_LABELS: Record<string, string> = {
  mock: '模拟支付',
  cash: '现金',
  insurance: '医保',
}

// ==================== 分诊 ====================
export const URGENCY_TRIAGE_LABELS: Record<string, string> = {
  routine: '常规',
  urgent: '尽快就诊',
  stat: '立即急诊',
}

export const DECISION_SOURCE_LABELS: Record<string, string> = {
  ai: 'AI 分诊',
  rule: '规则匹配',
  hybrid: 'AI + 规则',
  emergency: '急症拦截',
}

// ==================== 知识库 ====================
export const DOC_TYPE_LABELS: Record<string, string> = {
  guideline: '诊疗指南',
  drug_manual: '药品说明书',
  lab_reference: '检验参考',
  condition: '疾病知识',
  drug: '药物',
  textbook: '教材',
}

export const SOURCE_TIER_LABELS: Record<string, string> = {
  authoritative: '权威',
  professional: '专业',
  general: '一般',
  unverified: '未核实',
}

export const INDEX_STATUS_LABELS: Record<string, string> = {
  pending: '待索引',
  indexed: '已索引',
  stale: '需重建',
  failed: '索引失败',
}

// ==================== 监控 / 评测 ====================
export const LLM_CALL_STATUS_LABELS: Record<string, string> = {
  success: '成功',
  degraded: '降级',
  error: '错误',
  timeout: '超时',
}

export const EVAL_KIND_LABELS: Record<string, string> = {
  retrieval: '检索质量',
  quality: '回答质量',
  ragas: 'RAGAS',
}

export const EVAL_RUN_STATUS_LABELS: Record<string, string> = {
  pending: '排队中',
  running: '运行中',
  completed: '已完成',
  failed: '失败',
}

// ==================== AI 任务状态 ====================
export const AI_TASK_STATUS_LABELS: Record<string, string> = {
  pending: '排队中',
  running: '执行中',
  completed: '已完成',
  failed: '失败',
  retrying: '重试中',
}
