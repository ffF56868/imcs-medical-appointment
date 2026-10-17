/**
 * 分诊 API。对齐 app/api/v1/triage.py。
 * 注意：/triage/results 返回裸数组（无分页包裹），这是全站唯一例外。
 */
import { api } from './client'
import type { TriageResponse } from '@/types/clinical'

export interface TriageRequest {
  chief_complaint: string
  include_slots?: boolean
}

export function assess(payload: TriageRequest): Promise<TriageResponse> {
  return api.post<TriageResponse>('/triage/assess', payload)
}

/** 返回裸数组，无分页包裹。 */
export function listResults(): Promise<unknown[]> {
  return api.get<unknown[]>('/triage/results')
}

export function getResult(id: string): Promise<unknown> {
  return api.get<unknown>(`/triage/results/${id}`)
}
