/**
 * 处方 API（患者读自己的处方）。对齐 app/api/v1/prescription.py。
 */
import { api } from './client'
import type { PageReq, PageResp } from '@/types/common'
import type { Prescription } from '@/types/clinical'

export interface PrescriptionQuery extends PageReq {
  status?: string
}

export function listMyPrescriptions(params: PrescriptionQuery): Promise<PageResp<Prescription>> {
  return api.get<PageResp<Prescription>>('/prescriptions', { params })
}

export function getPrescription(id: string): Promise<Prescription> {
  return api.get<Prescription>(`/prescriptions/${id}`)
}
