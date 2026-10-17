/**
 * 检查报告 API（患者读自己的报告，含 AI 解读）。对齐 app/api/v1/exam.py。
 */
import { api } from './client'
import type { PageReq, PageResp } from '@/types/common'
import type { ExamReport } from '@/types/clinical'

export function listMyReports(params: PageReq): Promise<PageResp<ExamReport>> {
  return api.get<PageResp<ExamReport>>('/exams/reports', { params })
}

export function getReport(id: string): Promise<ExamReport> {
  return api.get<ExamReport>(`/exams/reports/${id}`)
}
