/**
 * 接诊与病历 API（医生侧）。对齐 app/api/v1/consult.py。
 * 患者侧只读的 listMyVisits/getVisit 也在这里，同域不拆两个文件。
 */
import { api } from './client'
import type { Visit, ExamRequest } from '@/types/clinical'

/** 今日待诊队列的一行（后端返回预约，visit 可能尚不存在）。 */
export interface PendingAppointment {
  appointment_id: string
  patient_id: string
  time_slot: string
  appointment_date: string
  status: string
  chief_complaint: string | null
  triage_summary: string | null
  checked_in_at: string | null
  visit_id: string | null
  visit_status: string | null
}

export interface PendingQueueResp {
  doctor_id: string
  date: string
  items: PendingAppointment[]
  total: number
}

export interface VisitListResp {
  items: Visit[]
  total: number
  page: number
  page_size: number
}

export interface ConsultStartResp {
  visit: Visit
  appointment_status: string
  prior_visits: Visit[]
}

export interface RecordSaveResp {
  visit: Visit
  transitioned: boolean
}

export interface CloseResp {
  visit: Visit
  appointment_id: string
  appointment_status: string
}

export interface MedicalRecordPayload {
  present_illness?: string
  past_history?: string
  physical_exam?: string
  diagnosis?: string
  icd_codes?: string[]
  treatment_advice?: string
}

export interface ExamOrderPayload {
  exam_type: string
  exam_name: string
  body_part?: string
  clinical_info?: string
  urgency: 'routine' | 'urgent' | 'stat'
}

export function listPendingQueue(date?: string): Promise<PendingQueueResp> {
  return api.get<PendingQueueResp>('/consult/appointments/pending', {
    params: date ? { date } : {},
  })
}

export function startConsultation(appointmentId: string): Promise<ConsultStartResp> {
  return api.post<ConsultStartResp>(`/consult/appointments/${appointmentId}/start`)
}

export function saveMedicalRecord(visitId: string, payload: MedicalRecordPayload): Promise<RecordSaveResp> {
  return api.put<RecordSaveResp>(`/consult/visits/${visitId}/record`, payload)
}

export function closeVisit(visitId: string): Promise<CloseResp> {
  return api.patch<CloseResp>(`/consult/visits/${visitId}/close`)
}

export function orderExam(visitId: string, payload: ExamOrderPayload): Promise<ExamRequest> {
  return api.post<ExamRequest>(`/consult/visits/${visitId}/exams`, payload)
}

export function listVisitExams(visitId: string): Promise<ExamRequest[]> {
  return api.get<ExamRequest[]>(`/consult/visits/${visitId}/exams`)
}

export function listVisits(params: {
  status?: string
  patient_id?: string
  page?: number
  page_size?: number
}): Promise<VisitListResp> {
  return api.get<VisitListResp>('/consult/visits', { params })
}

/** 患者视图别名：同一端点，后端按角色行级隔离。 */
export const listMyVisits = listVisits

export function getVisit(id: string): Promise<Visit> {
  return api.get<Visit>(`/consult/visits/${id}`)
}
