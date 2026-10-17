/**
 * 发药与药师审核 API。对齐 app/api/v1/dispensing.py + prescription.py。
 */
import { api } from './client'
import type { PageReq, PageResp } from '@/types/common'
import type { Prescription, DispensingRecord } from '@/types/clinical'

/** 药师工作台用的处方行（后端按角色隔离，read_all 拿到全院）。 */
export type PrescriptionRow = Prescription

export interface DispenseQuery extends PageReq {
  status?: string
}

/** 待审核/待发药的处方：付费后的处方（status=paid）进入药师队列。 */
export function listPrescriptionsAll(params: DispenseQuery): Promise<PageResp<PrescriptionRow>> {
  return api.get<PageResp<PrescriptionRow>>('/prescriptions', { params })
}

export function getPrescription(id: string): Promise<PrescriptionRow> {
  return api.get<PrescriptionRow>(`/prescriptions/${id}`)
}

/** 审核处方（四查十对，notes 必填）。 */
export function approvePrescription(id: string, body: { notes: string }): Promise<DispensingRecord> {
  return api.post<DispensingRecord>(`/dispensing/prescriptions/${id}/approve`, body)
}

/** 发药（需已审核，处方 paid→dispensed）。 */
export function dispensePrescription(id: string, body: { notes: string | null }): Promise<DispensingRecord> {
  return api.post<DispensingRecord>(`/dispensing/prescriptions/${id}/dispense`, body)
}

/** 患者视角：自己的取药记录。 */
export function listMyRecords(params: PageReq): Promise<PageResp<DispensingRecord>> {
  return api.get<PageResp<DispensingRecord>>('/dispensing/records', { params })
}

/** 药师视角：全院发药记录。 */
export function listRecordsAll(params: PageReq): Promise<PageResp<DispensingRecord>> {
  return api.get<PageResp<DispensingRecord>>('/dispensing/records/all', { params })
}
