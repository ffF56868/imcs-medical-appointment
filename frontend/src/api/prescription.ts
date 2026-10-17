/**
 * 处方 API。患者读自己的，医生读写自己的，药师/收费员读全部。
 * 对齐 app/api/v1/prescription.py。
 */
import { api } from './client'
import type { PageReq, PageResp } from '@/types/common'
import type { Prescription } from '@/types/clinical'

export interface PrescriptionQuery extends PageReq {
  status?: string
}

/** 患者视图：只看自己的处方。 */
export function listMyPrescriptions(params: PrescriptionQuery): Promise<PageResp<Prescription>> {
  return api.get<PageResp<Prescription>>('/prescriptions', { params })
}

/** 全域视图（药师/收费员，需 prescription:read_all）。 */
export function listPrescriptionsAll(params: PrescriptionQuery): Promise<PageResp<Prescription>> {
  // 后端同一端点按角色行级过滤，read_all 角色拿到的即全院数据
  return api.get<PageResp<Prescription>>('/prescriptions', { params })
}

export function getPrescription(id: string): Promise<Prescription> {
  return api.get<Prescription>(`/prescriptions/${id}`)
}

/** 某次就诊下的全部处方（医生工作台展开用）。 */
export function listVisitPrescriptions(visitId: string): Promise<Prescription[]> {
  return api.get<Prescription[]>(`/prescriptions/by-visit/${visitId}`)
}

export interface PrescriptionItemIn {
  drug_code: string
  drug_name: string
  specification?: string | null
  dosage?: string | null
  frequency?: string | null
  route?: string | null
  duration_days?: number | null
  quantity: number
  unit: string
  unit_price: number
  usage_instructions?: string | null
}

export interface PrescriptionCreatePayload {
  visit_id: string
  diagnosis?: string | null
  notes?: string | null
  items: PrescriptionItemIn[]
}

export function createPrescription(payload: PrescriptionCreatePayload): Promise<Prescription> {
  return api.post<Prescription>('/prescriptions', payload)
}

export function updatePrescription(id: string, payload: PrescriptionCreatePayload): Promise<Prescription> {
  return api.put<Prescription>(`/prescriptions/${id}`, payload)
}

export interface ConfirmResp {
  prescription: Prescription
  overridden: boolean
}

/**
 * 确认开具。命中阻断项时后端 422（ApiError.detail 带 findings），
 * 医生复核后可带 safety_override_reason 重试。
 */
export function confirmPrescription(id: string, reason?: string): Promise<ConfirmResp> {
  return api.post<ConfirmResp>(`/prescriptions/${id}/confirm`, reason ? { safety_override_reason: reason } : {})
}

export function cancelPrescription(id: string, reason?: string): Promise<{ prescription: Prescription }> {
  return api.patch<{ prescription: Prescription }>(`/prescriptions/${id}/cancel`, reason ? { reason } : {})
}
