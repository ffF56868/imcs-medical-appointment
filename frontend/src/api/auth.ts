/**
 * 认证 API。对齐后端 app/api/v1/auth.py。
 *
 * 登录请求体字段是 account（可传手机号/邮箱/用户名），不是 username——
 * 这点在阶段 5 的 e2e 里踩过坑，务必对齐。
 */

import { api } from './client'
import type { AuthResult, TokenPair, UserOut } from '@/types/common'

export function login(account: string, password: string): Promise<AuthResult> {
  return api.post<AuthResult>('/auth/login', { account, password })
}

export function getMe(): Promise<UserOut> {
  return api.get<UserOut>('/auth/me')
}

export function refresh(refresh_token: string): Promise<TokenPair> {
  return api.post<TokenPair>('/auth/refresh', { refresh_token })
}

export function logout(refresh_token?: string | null): Promise<void> {
  return api.post<void>('/auth/logout', { refresh_token: refresh_token ?? null })
}
