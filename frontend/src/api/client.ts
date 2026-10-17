/**
 * Axios 实例与拦截器。全站 HTTP 请求的唯一出口。
 *
 * 三件事在这里统一做，业务 API 文件只管定义端点：
 *   1. 请求注入 Authorization: Bearer 与 X-Trace-Id
 *   2. 401 静默刷新一次再重放；刷新也失败才跳登录
 *   3. 非 2xx 把 detail.code 转成中文提示并抛出统一错误
 *
 * **无成功包裹层**：后端成功直接返回资源体，所以响应拦截器返回
 * response.data 本身，业务侧拿到的就是 UserOut / PageResp<T> 等，
 * 不需要再 .data.data。
 */

import axios, {
  type AxiosInstance,
  type AxiosRequestConfig,
  type InternalAxiosRequestConfig,
} from 'axios'
import { parseApiError } from '@/utils/errors'

export const API_BASE = '/api/v1'

/** 统一错误类型：业务侧 catch 到它就能拿到 code 与中文 message。 */
export class ApiError extends Error {
  code: string
  status: number
  constructor(code: string, message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
  }
}

// 延迟到运行时从 auth store 取 token，避免 client 与 store 循环依赖。
// 由 main.ts 在创建 store 后注入这两个访问器。
let tokenGetter: () => string | null = () => null
let unauthorizedHandler: () => void = () => {}
let refreshFn: (() => Promise<boolean>) | null = null

export function configureAuthHooks(opts: {
  getToken: () => string | null
  onUnauthorized: () => void
  refresh?: () => Promise<boolean>
}) {
  tokenGetter = opts.getToken
  unauthorizedHandler = opts.onUnauthorized
  refreshFn = opts.refresh ?? null
}

/** 生成一次请求的 trace-id。与后端 X-Trace-Id 对齐，便于全链路排障。 */
function newTraceId(): string {
  // 优先用原生 randomUUID；不支持时退化为时间戳+随机数（够用即可）
  const c = globalThis.crypto as Crypto | undefined
  if (c?.randomUUID) return c.randomUUID()
  return `${Date.now().toString(16)}${Math.random().toString(16).slice(2, 10)}`
}

export const http: AxiosInstance = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

// 401 刷新期间的并发请求挂起队列：避免多个请求同时触发多次刷新
let isRefreshing = false
let pendingQueue: Array<(token: string | null) => void> = []

function flushQueue(token: string | null) {
  pendingQueue.forEach((resolve) => resolve(token))
  pendingQueue = []
}

http.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = tokenGetter()
  if (token) config.headers.Authorization = `Bearer ${token}`
  // 每个请求带独立 trace-id；若调用方已指定则尊重之（便于前端主动串联）
  if (!config.headers['X-Trace-Id']) config.headers['X-Trace-Id'] = newTraceId()
  return config
})

http.interceptors.response.use(
  (response) => response.data, // 拆掉 axios 外壳，直接给资源体
  async (error) => {
    const { config, response } = error as {
      config?: InternalAxiosRequestConfig & { _retry?: boolean }
      response?: { status: number; data: unknown }
    }

    // 401：尝试静默刷新一次。刷新接口本身 401 不再递归刷新。
    const isRefreshCall = config?.url?.includes('/auth/refresh')
    if (response?.status === 401 && config && !config._retry && !isRefreshCall && refreshFn) {
      if (isRefreshing) {
        // 已有刷新在跑，挂起当前请求，等刷新结果出来后重放
        return new Promise((resolve, reject) => {
          pendingQueue.push((token) => {
            if (!token) {
              reject(toApiError(error))
              return
            }
            config._retry = true
            config.headers.Authorization = `Bearer ${token}`
            resolve(http(config as AxiosRequestConfig))
          })
        })
      }
      config._retry = true
      isRefreshing = true
      try {
        const ok = await refreshFn()
        const token = ok ? tokenGetter() : null
        flushQueue(token)
        if (ok && token) {
          config.headers.Authorization = `Bearer ${token}`
          return http(config as AxiosRequestConfig)
        }
        unauthorizedHandler()
        return Promise.reject(toApiError(error))
      } catch {
        flushQueue(null)
        unauthorizedHandler()
        return Promise.reject(toApiError(error))
      } finally {
        isRefreshing = false
      }
    }

    if (response?.status === 401) unauthorizedHandler()
    return Promise.reject(toApiError(error))
  },
)

function toApiError(error: unknown): ApiError {
  const resp = (error as { response?: { status: number; data: unknown } })?.response
  const { code, message } = parseApiError(resp?.data)
  return new ApiError(code, message, resp?.status ?? 0)
}

/** 便捷方法：返回类型友好的 get/post/put/patch/delete。 */
export const api = {
  get: <T>(url: string, config?: AxiosRequestConfig): Promise<T> =>
    http.get(url, config) as unknown as Promise<T>,
  post: <T>(url: string, body?: unknown, config?: AxiosRequestConfig): Promise<T> =>
    http.post(url, body, config) as unknown as Promise<T>,
  put: <T>(url: string, body?: unknown, config?: AxiosRequestConfig): Promise<T> =>
    http.put(url, body, config) as unknown as Promise<T>,
  patch: <T>(url: string, body?: unknown, config?: AxiosRequestConfig): Promise<T> =>
    http.patch(url, body, config) as unknown as Promise<T>,
  delete: <T>(url: string, config?: AxiosRequestConfig): Promise<T> =>
    http.delete(url, config) as unknown as Promise<T>,
}
