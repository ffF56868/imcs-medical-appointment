/**
 * SSE 流式消费器。健康咨询的「打字机」效果就靠它。
 *
 * **为什么不用原生 EventSource**：EventSource 不能设置请求头，
 * 而本系统的 /health-qa/stream 需要 Authorization: Bearer 鉴权。
 * 因此改用 fetch + ReadableStream 手动解析 SSE 帧。
 *
 * SSE 帧格式（与后端 app/api/v1/health_qa.py 的 _sse_frame 对齐）：
 *   event: <name>\n
 *   data: <json>\n
 *   \n
 * 心跳是注释行 `: heartbeat\n\n`，解析时跳过。
 */

import { API_BASE } from './client'

export type SseEventName = 'metadata' | 'token' | 'done' | 'error'

export interface SseEvent {
  event: SseEventName | string
  data: unknown
}

export interface SseHandlers {
  /** metadata：引用来源、trace_id、guard 类型。先于正文到达。 */
  onMetadata?: (data: any) => void
  /** token：增量文本，逐片回调实现打字机。 */
  onToken?: (text: string) => void
  /** done：用量、免责声明、降级标记。正常结束。 */
  onDone?: (data: any) => void
  /** error：出错结束（HTTP 仍 200，因为流已开始无法改状态码）。 */
  onError?: (data: { code?: string; message?: string }) => void
}

export interface SseStreamOptions extends SseHandlers {
  /** 请求体（POST）。 */
  body: Record<string, unknown>
  /** 鉴权 token。 */
  token: string | null
  /** 可选：外部传入的 trace-id，便于前端主动串联日志。 */
  traceId?: string
  /** 中止信号：组件卸载或用户点「停止」时 abort。 */
  signal?: AbortSignal
}

/**
 * 发起一次 SSE 请求并消费事件流。返回的 Promise 在流正常结束、出错或被
 * 中止时 resolve（不 reject）——调用方通过 onError / signal.aborted 判断结局。
 * 这样设计是因为「用户主动停止」不是错误，不该触发 catch 里的错误提示。
 */
export async function consumeSse(opts: SseStreamOptions): Promise<void> {
  const { body, token, traceId, signal } = opts
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  }
  if (token) headers.Authorization = `Bearer ${token}`
  if (traceId) headers['X-Trace-Id'] = traceId

  let response: Response
  try {
    response = await fetch(`${API_BASE}/health-qa/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal,
    })
  } catch (err) {
    // fetch 本身失败（网络断、被 abort）
    if (signal?.aborted) return
    opts.onError?.({ code: 'network_error', message: '无法连接到服务器，请检查网络' })
    return
  }

  // HTTP 层错误（401/403/409/422/503）：此时还没进入流，body 是 JSON 错误体
  if (!response.ok || !response.body) {
    let detail: any = {}
    try {
      detail = await response.json()
    } catch {
      // 忽略：错误体解析失败就用状态码兜底
    }
    const d = detail?.detail
    const code = typeof d === 'object' ? d?.code : ''
    const message =
      (typeof d === 'object' ? d?.message : d) || `请求失败（HTTP ${response.status}）`
    opts.onError?.({ code: code || `http_${response.status}`, message })
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // 按 SSE 事件分隔符 \n\n 切分；最后一段可能不完整，留在 buffer 等下一块
      let sepIndex: number
      while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
        const rawEvent = buffer.slice(0, sepIndex)
        buffer = buffer.slice(sepIndex + 2)
        dispatchFrame(rawEvent, opts)
      }
    }
    // flush 残余（正常以 \n\n 结尾时 buffer 为空）
    if (buffer.trim()) dispatchFrame(buffer, opts)
  } catch (err) {
    if (!signal?.aborted) {
      opts.onError?.({ code: 'stream_error', message: '连接中断，请重试' })
    }
  } finally {
    reader.releaseLock()
  }
}

/** 解析单个 SSE 帧并分发到对应 handler。 */
function dispatchFrame(rawEvent: string, opts: SseHandlers): void {
  let eventName = 'message'
  const dataLines: string[] = []

  for (const line of rawEvent.split('\n')) {
    if (line.startsWith(':')) continue // 心跳等注释行，跳过
    if (line.startsWith('event:')) {
      eventName = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim())
    }
  }

  if (dataLines.length === 0) return // 纯注释帧（心跳）

  const raw = dataLines.join('\n')
  let data: unknown
  try {
    data = JSON.parse(raw)
  } catch {
    data = raw // 不是 JSON 就当纯文本（理论上不会发生，兜底）
  }

  switch (eventName) {
    case 'metadata':
      opts.onMetadata?.(data)
      break
    case 'token':
      opts.onToken?.((data as { text?: string })?.text ?? '')
      break
    case 'done':
      opts.onDone?.(data)
      break
    case 'error':
      opts.onError?.(data as { code?: string; message?: string })
      break
    default:
      break
  }
}
