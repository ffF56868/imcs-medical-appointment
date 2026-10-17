/**
 * 就诊与病历 API（患者读自己的病历）。对齐 app/api/v1/consult.py。
 */
import { api } from './client'
import type { PageReq, PageResp } from '@/types/common'
import type { Visit } from '@/types/clinical'

export interface VisitQuery extends PageReq {
  status?: string
}

export function listMyVisits(params: VisitQuery): Promise<PageResp<Visit>> {
  return api.get<PageResp<Visit>>('/consult/visits', { params })
}

export function getVisit(id: string): Promise<Visit> {
  return api.get<Visit>(`/consult/visits/${id}`)
}
