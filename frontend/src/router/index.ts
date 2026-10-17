/**
 * 路由实例与全局守卫。
 *
 * 守卫逻辑：
 *   1. 白名单（login/403/404）直接放行
 *   2. 未登录 → 跳登录并记录 redirect，登录后回跳
 *   3. 已登录但 user 为空（刷新页面）→ 先 fetchMe 恢复，失败则尝试静默刷新
 *   4. 角色不匹配 meta.roles → 跳 403（不泄露资源是否存在）
 */

import { createRouter, createWebHistory } from 'vue-router'
import { routes } from './routes'
import { useAuthStore } from '@/stores/auth'

const router = createRouter({
  history: createWebHistory(),
  routes,
  // 切换路由回到顶部，避免长表格页残留滚动位置
  scrollBehavior: () => ({ top: 0 }),
})

const WHITELIST = new Set(['login', 'forbidden', 'not-found'])

router.beforeEach(async (to) => {
  const auth = useAuthStore()

  if (WHITELIST.has(to.name as string)) {
    // 已登录用户访问登录页，直接回工作台
    if (to.name === 'login' && auth.isLoggedIn) return { name: 'home' }
    return true
  }

  // 未登录：记录来路，登录后回跳
  if (!auth.isLoggedIn) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }

  // 刷新页面后 user 丢失但 token 还在：先恢复用户信息
  if (!auth.user) {
    try {
      await auth.fetchMe()
    } catch {
      const ok = await auth.tryRefresh()
      if (ok) {
        try {
          await auth.fetchMe()
        } catch {
          auth.clearAll()
          return { name: 'login', query: { redirect: to.fullPath } }
        }
      } else {
        return { name: 'login', query: { redirect: to.fullPath } }
      }
    }
  }

  // 角色校验
  const roles = to.meta.roles
  if (roles && roles.length > 0 && !auth.hasRole(roles)) {
    return { name: 'forbidden' }
  }

  return true
})

export default router
