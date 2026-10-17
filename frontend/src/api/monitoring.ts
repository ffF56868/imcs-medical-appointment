/**
 * AI 监控 API（管理端）。对齐 app/api/v1/monitoring.py，权限 ai_monitor:read_all。
 */
import { api } from './client'
import type {
  AggregateResult,
  LlmCallListResp,
  MetricSeriesResponse,
  MonitoringOverview,
  TraceReplay,
} from '@/types/admin'

export function getOverview(windowMinutes = 60): Promise<MonitoringOverview> {
  return api.get('/monitoring/overview', { params: { window_minutes: windowMinutes } })
}

export function getMetricSeries(params: {
  metric?: string
  metrics?: string
  start_date?: string
  end_date?: string
  tag?: string
  limit?: number
}): Promise<MetricSeriesResponse> {
  return api.get('/monitoring/metrics', { params })
}

export function listLlmCalls(params: {
  scenario?: string
  model?: string
  status_filter?: string
  trace_id?: string
  start_date?: string
  end_date?: string
  page?: number
  page_size?: number
}): Promise<LlmCallListResp> {
  return api.get('/monitoring/llm-calls', { params })
}

export function replayTrace(traceId: string): Promise<TraceReplay> {
  return api.get(`/monitoring/traces/${encodeURIComponent(traceId)}`)
}

/** 手动聚合：演示/断档补数用。lookback_minutes 最多回溯，缺省只算上一分钟桶。 */
export function triggerAggregate(payload: {
  bucket_start?: string
  lookback_minutes?: number
}): Promise<AggregateResult> {
  return api.post('/monitoring/aggregate', payload)
}
