/**
 * 跨模块通用类型：认证、用户、分页、错误。
 *
 * 字段名逐字对齐后端 schema（app/schemas/auth.py 等）。
 * 后端**无成功包裹层**——成功直接返回资源体，失败是 { detail: { code, message } }，
 * 因此这里的类型都是「裸资源」形态，不带 code/data 外壳。
 */

import type { RoleName } from './enums'

// ==================== 认证 ====================

/** 登录响应：TokenPair + user。 */
export interface AuthResult {
  access_token: string
  refresh_token: string
  token_type: string // 恒为 "bearer"
  expires_in: number // 秒
  user: UserOut
}

export interface TokenPair {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
}

/** 当前用户。role_name 是路由守卫与菜单过滤的依据。 */
export interface UserOut {
  id: string
  username: string
  real_name: string | null
  phone: string | null
  email: string | null
  role_id: number
  role_name: RoleName
  department_id: string | null
  department_name?: string | null
  title: string | null
  specialty: string | null
  is_active: boolean
  created_at: string
}

// ==================== 分页 ====================

/** 列表请求参数。page 从 1 开始，page_size 上限 100。 */
export interface PageReq {
  page?: number
  page_size?: number
}

/** 列表响应。全站统一这四字段（/triage/results 是例外，返回裸数组）。 */
export interface PageResp<T> {
  items: T[]
  total: number
  page: number
  page_size: number
}

// ==================== 错误 ====================

/** 后端统一错误体：detail.code 供前端做分支（如 401 跳登录）。 */
export interface ApiErrorDetail {
  code: string
  message: string
  [key: string]: unknown
}

export interface ApiErrorBody {
  detail: ApiErrorDetail | string
}

// ==================== 通用 ====================

/** 状态机资源普遍带 allowed_transitions，前端据此决定按钮可用性。 */
export interface WithTransitions {
  allowed_transitions?: string[]
}

/** 金额：后端 registration_fee/total_amount/paid_amount/amount 都是 int 分。
 *  处方明细 unit_price 例外，是元（float）。展示统一走 utils/currency。 */
export type AmountFen = number
