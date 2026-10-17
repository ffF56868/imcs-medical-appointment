/**
 * AI 评测 API（管理端）。对齐 app/api/v1/evaluation.py。
 * 注意：retrieval/quality 同步跑（秒级），ragas 异步派发后要轮询 runs/{id}。
 */
import { api } from './client'
import type {
  EvalCase,
  EvalCaseDraft,
  EvalCaseListResp,
  EvalRun,
  EvalRunListResp,
  EvalTrend,
  RagasDispatch,
} from '@/types/admin'

// ---- 题库 ----

export function listCases(params: {
  category?: string
  page?: number
  page_size?: number
}): Promise<EvalCaseListResp> {
  return api.get('/evaluation/cases', { params })
}

export function createCase(draft: EvalCaseDraft): Promise<EvalCase> {
  return api.post('/evaluation/cases', draft)
}

export function updateCase(caseId: string, patch: Partial<EvalCaseDraft>): Promise<EvalCase> {
  return api.put(`/evaluation/cases/${caseId}`, patch)
}

export function deleteCase(caseId: string): Promise<void> {
  return api.delete(`/evaluation/cases/${caseId}`)
}

// ---- 运行 ----

/** 同步评测：kind=retrieval|quality，请求内跑完直接带指标返回。 */
export function triggerRun(kind: 'retrieval' | 'quality', category?: string): Promise<EvalRun> {
  return api.post('/evaluation/runs', null, { params: { kind, category } })
}

/** RAGAS 异步评测：202 排队，之后轮询 getRun。409=已有活跃任务（防双跑烧配额）。 */
export function dispatchRagas(sampleSize?: number): Promise<RagasDispatch> {
  return api.post('/evaluation/ragas', { sample_size: sampleSize })
}

export function listRuns(params: {
  kind?: string
  status?: string
  page?: number
  page_size?: number
}): Promise<EvalRunListResp> {
  return api.get('/evaluation/runs', { params })
}

export function getRun(runId: string): Promise<EvalRun> {
  return api.get(`/evaluation/runs/${runId}`)
}

export function getTrend(kind: string): Promise<EvalTrend> {
  return api.get('/evaluation/trends', { params: { kind } })
}
