/**
 * 健康咨询 API。对齐 app/api/v1/health_qa.py。
 * 走 SSE 流式，不经 axios，用 consumeSse（fetch + ReadableStream）。
 */
import { consumeSse, type SseHandlers } from './sse'
import { useAuthStore } from '@/stores/auth'

export interface ConsultTurn {
  role: 'user' | 'assistant'
  content: string
}

export interface ConsultMetadata {
  trace_id: string
  model: string | null
  references: Array<{
    chunk_id: string
    title: string
    source: string
    doc_type: string
    source_tier: string
    page_number: number | null
    rrf_score: number
  }>
  retrieved_count: number
  is_zero_result: boolean
  guard: 'emergency' | 'crisis' | null
}

export interface ConsultDone {
  usage: { prompt_tokens: number; completion_tokens: number }
  first_token_ms: number | null
  latency_ms: number
  model: string | null
  degradation: string | null
  disclaimer: string
}

/**
 * 发起一次流式健康咨询。
 * @returns 一个 abort 函数，供组件卸载或用户点「停止」时调用。
 */
export function askHealthQuestion(
  question: string,
  history: ConsultTurn[],
  handlers: SseHandlers,
): () => void {
  const auth = useAuthStore()
  const controller = new AbortController()
  consumeSse({
    body: { question, history },
    token: auth.accessToken,
    signal: controller.signal,
    ...handlers,
  })
  return () => controller.abort()
}

export type { ConsultMetadata as Metadata, ConsultDone as Done }
