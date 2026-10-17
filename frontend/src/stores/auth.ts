/**
 * 认证状态。token 持久化策略：
 *   - access_token → sessionStorage：关标签页即失效，降低被盗用窗口
 *   - refresh_token → localStorage：支持下次打开静默续期
 * 这是安全与体验的折中：完全放 localStorage 会让 access_token 长期驻留，
 * 完全放 sessionStorage 则每次开新标签都要重新登录。
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import * as authApi from '@/api/auth'
import type { UserOut } from '@/types/common'
import type { RoleName } from '@/types/enums'

const ACCESS_KEY = 'imcs.access_token'
const REFRESH_KEY = 'imcs.refresh_token'

export const useAuthStore = defineStore('auth', () => {
  const accessToken = ref<string | null>(sessionStorage.getItem(ACCESS_KEY))
  const refreshToken = ref<string | null>(localStorage.getItem(REFRESH_KEY))
  const user = ref<UserOut | null>(null)

  const isLoggedIn = computed(() => !!accessToken.value)
  const role = computed<RoleName | null>(() => user.value?.role_name ?? null)
  const displayName = computed(
    () => user.value?.real_name || user.value?.username || '',
  )

  function setTokens(access: string, refresh: string) {
    accessToken.value = access
    refreshToken.value = refresh
    sessionStorage.setItem(ACCESS_KEY, access)
    localStorage.setItem(REFRESH_KEY, refresh)
  }

  function clearAll() {
    accessToken.value = null
    refreshToken.value = null
    user.value = null
    sessionStorage.removeItem(ACCESS_KEY)
    localStorage.removeItem(REFRESH_KEY)
  }

  async function login(account: string, password: string) {
    const result = await authApi.login(account, password)
    setTokens(result.access_token, result.refresh_token)
    user.value = result.user
    return result.user
  }

  /** 拉当前用户。刷新页面后 user 为空但 token 还在时用它恢复。 */
  async function fetchMe() {
    if (!accessToken.value) return null
    user.value = await authApi.getMe()
    return user.value
  }

  /** 静默刷新。成功返回 true 并更新 token，失败清空登录态返回 false。 */
  async function tryRefresh(): Promise<boolean> {
    if (!refreshToken.value) return false
    try {
      const pair = await authApi.refresh(refreshToken.value)
      setTokens(pair.access_token, pair.refresh_token)
      return true
    } catch {
      clearAll()
      return false
    }
  }

  async function logout() {
    // 通知后端作废 refresh_token；失败也要清本地，不能卡住登出
    try {
      await authApi.logout(refreshToken.value)
    } catch {
      // 忽略：登出以本地清理为准
    }
    clearAll()
  }

  /** 供路由守卫调用：是否有权访问某角色集合。 */
  function hasRole(roles: RoleName[]): boolean {
    if (!role.value) return false
    return roles.includes(role.value)
  }

  return {
    accessToken,
    refreshToken,
    user,
    isLoggedIn,
    role,
    displayName,
    login,
    fetchMe,
    tryRefresh,
    logout,
    clearAll,
    hasRole,
    setTokens,
  }
})
