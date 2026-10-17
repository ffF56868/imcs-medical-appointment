/**
 * 检查与报告 API。患者读自己的报告，医生开单与读协作报告，检验科执行全流程。
 * 对齐 app/api/v1/exam.py。
 */
import { api } from './client'
import type { PageReq, PageResp } from '@/types/common'
import type { ExamReport, ExamRequest } from '@/types/clinical'

export function listMyReports(params: PageReq): Promise<PageResp<ExamReport>> {
  return api.get<PageResp<ExamReport>>('/exams/reports', { params })
}

export function getReport(id: string): Promise<ExamReport> {
  return api.get<ExamReport>(`/exams/reports/${id}`)
}

/* ==================== 检验科工作台 ==================== */

/** queue 响应带 counts（按紧急度分布），用于顶部分布标签。 */
export interface LabQueueResp {
  items: ExamRequest[]
  total: number
  page: number
  page_size: number
  counts: Record<string, number>
}

/** 在办队列（received / processing）。 */
export function labQueue(params: { status?: string; page?: number; page_size?: number }): Promise<LabQueueResp> {
  return api.get<LabQueueResp>('/exams/queue', { params })
}

/** 待接收队列（ordered，医生刚开出的新单）。 */
export function labIncoming(params: { page?: number; page_size?: number }): Promise<LabQueueResp> {
  return api.get<LabQueueResp>('/exams/requests', { params })
}

export function receiveRequest(id: string): Promise<ExamRequest> {
  return api.post<ExamRequest>(`/exams/requests/${id}/receive`, {})
}

export function startProcessing(id: string): Promise<ExamRequest> {
  return api.post<ExamRequest>(`/exams/requests/${id}/process`, {})
}

/** 录入报告（后端自动派发 AI 解读，不等 LLM）。 */
export interface SubmitReportResp {
  report: ExamReport
  dispatch: { dispatched: boolean; reason: string | null; ai_task_id: string | null }
}

export function submitReport(
  requestId: string,
  body: {
    raw_content: string
    structured_data?: Record<string, unknown> | null
    conclusion?: string | null
    auto_interpret?: boolean
  },
): Promise<SubmitReportResp> {
  return api.post<SubmitReportResp>(`/exams/requests/${requestId}/reports`, body)
}

/** 全域报告列表（检验科）。 */
export function listAllReports(params: PageReq): Promise<PageResp<ExamReport>> {
  return api.get<PageResp<ExamReport>>('/exams/reports/all', { params })
}

/** 手动（重）触发 AI 解读。 */
export function triggerInterpret(id: string, reason?: string): Promise<unknown> {
  return api.post(`/exams/reports/${id}/interpret`, reason ? { reason } : {})
}
