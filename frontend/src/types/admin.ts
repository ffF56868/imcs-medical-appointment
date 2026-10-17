/**
 * 管理端三块（监控 / 评测 / 知识库）的类型定义。
 * 字段与 app/schemas/{monitoring,evaluation,knowledge}.py 逐一对齐。
 */

// ==================== AI 监控 ====================

export interface MetricPoint {
  bucket_time: string
  metric_name: string
  metric_value: number
  sample_count: number
  tags: Record<string, unknown>
}

export interface MetricSeriesResponse {
  items: MetricPoint[]
  count: number
  limit: number
  metric_names: string[]
}

export interface LlmCallRow {
  id: number
  trace_id: string | null
  scenario: string
  provider: string
  model: string
  requested_model: string | null
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  cost_usd: number | null
  latency_ms: number
  first_token_ms: number | null
  status: string
  error_type: string | null
  http_status: number | null
  retry_count: number
  degradation_level: number | null
  moderation_blocked: boolean
  budget_exceeded: boolean
  is_stream: boolean
  user_id: string | null
  workflow_node: string | null
  ai_task_id: string | null
  created_at: string | null
}

export interface LlmCallListResp {
  items: LlmCallRow[]
  total: number
  page: number
  page_size: number
}

export interface OverviewMetrics {
  llm_call_count: number | null
  llm_error_rate: number | null
  token_usage_total: number | null
  token_cost_usd: number | null
  degradation_rate: number | null
  moderation_block_rate: number | null
  retrieval_zero_result_rate: number | null
  task_queue_backlog: number | null
}

export interface MonitoringOverview {
  window_minutes: number
  window_start: string
  window_end: string
  /** aggregated=预聚合表 realtime=现场扫明细 */
  source: string
  bucket_count: number
  metrics: OverviewMetrics
  queue: Record<string, number>
}

export interface RetrievalFragment {
  id: number
  trace_id: string | null
  query_text: string | null
  rewritten_query: string | null
  vector_hit_count: number
  keyword_hit_count: number
  both_hit_count: number
  merged_count: number
  final_count: number
  is_zero_result: boolean
  top_score: number | null
  vector_latency_ms: number
  keyword_latency_ms: number
  rrf_latency_ms: number
  rerank_latency_ms: number
  rerank_channel: string | null
  candidates: unknown[]
  created_at: string | null
}

export interface WorkflowFragment {
  id: number
  trace_id: string | null
  workflow_name: string
  status: string
  steps: Array<{ node?: string; status?: string; latency_ms?: number; [k: string]: unknown }>
  node_count: number
  executed_node_count: number
  total_latency_ms: number
  total_tokens: number
  total_cost_usd: number | null
  branch_taken: string | null
  final_output: Record<string, unknown> | null
  error_message: string | null
  created_at: string | null
  completed_at: string | null
}

export interface TriageFragment {
  id: string | null
  chief_complaint: string | null
  extracted_symptoms: string[]
  recommended_department_name: string | null
  urgency: string | null
  is_emergency: boolean
  advice: string | null
  confidence: number | null
  decision_source: string | null
  created_at: string | null
}

export interface TraceSummary {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  cost_usd: number
  call_count: number
  error_count: number
  degraded_count: number
  max_latency_ms: number | null
  started_at: string | null
  ended_at: string | null
  providers: string[]
  models: string[]
  scenarios: string[]
}

export interface TraceReplay {
  trace_id: string
  counts: { llm_calls: number; retrievals: number; workflows: number; triage_results: number }
  summary: TraceSummary
  llm_calls: LlmCallRow[]
  retrievals: RetrievalFragment[]
  workflows: WorkflowFragment[]
  triage_results: TriageFragment[]
  timeline: Array<{ kind: string; at: string | null; summary: string; ref: Record<string, unknown> }>
}

export interface AggregateResult {
  ok: boolean
  buckets: Array<{ bucket_start: string; written: number; ok: boolean; error?: string | null }>
  total_written: number
  message: string
}

// ==================== 评测 ====================

export interface EvalCase {
  id: string
  question: string
  expected_name: string
  expected_type: string
  category: string
  alternative_names: string[]
  answer_keywords: string[]
  citation_names: string[]
  expected_refusal: boolean
  created_at: string
}

export interface EvalCaseListResp {
  items: EvalCase[]
  total: number
  page: number
  page_size: number
}

export type EvalCaseDraft = Omit<EvalCase, 'id' | 'created_at'>

export interface EvalRun {
  id: string
  kind: string
  status: string
  total_count: number
  completed_count: number
  answer_accuracy: number | null
  citation_accuracy: number | null
  refusal_accuracy: number | null
  faithfulness: number | null
  answer_relevancy: number | null
  context_precision: number | null
  context_recall: number | null
  metrics: Record<string, unknown> | null
  results: Array<Record<string, unknown>> | null
  knowledge_hash: string | null
  knowledge_document_count: number
  error_message: string | null
  triggered_by: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
}

export interface EvalRunListResp {
  items: EvalRun[]
  total: number
  page: number
  page_size: number
}

export interface RagasDispatch {
  eval_run_id: string
  ai_task_id: string | null
  status: string
  sample_size: number
  message: string
}

export interface EvalTrend {
  kind: string
  points: Array<Record<string, unknown>>
  baseline: Record<string, unknown> | null
}

// ==================== 知识库 ====================

export interface KnowledgeDocSummary {
  id: string
  title: string
  doc_type: string
  source: string
  source_tier: string
  index_status: string
  chunk_count: number
  is_reviewed: boolean
  is_active: boolean
  content_length: number
  created_at: string | null
  indexed_at: string | null
}

export interface KnowledgeDoc extends KnowledgeDocSummary {
  content: string
  content_hash: string
  source_url: string | null
  published_at: string | null
}

export interface DocListResp {
  items: KnowledgeDocSummary[]
  total: number
  page: number
  page_size: number
}

export interface DocDraft {
  title: string
  content: string
  doc_type: string
  source: string
  source_url?: string | null
  source_tier: string
}

export interface KbTask {
  id: string
  task_type: string
  status: string
  progress: number
  progress_message: string | null
  error_message: string | null
  result: Record<string, unknown> | null
  started_at: string | null
  completed_at: string | null
  duration_ms: number | null
}

export interface RebuildAccepted {
  task_id: string
  task_type: string
  status: string
  message: string
}

export interface SearchHit {
  chunk_id: string
  content: string
  rrf_score: number
  title: string
  doc_type: string
  source: string
  source_tier: string
  page_range: string | null
}

export interface SearchResp {
  query: string
  results: SearchHit[]
  diagnostics: {
    vector_hit_count: number
    keyword_hit_count: number
    both_hit_count: number
    merged_count: number
    is_zero_result: boolean
  }
}
