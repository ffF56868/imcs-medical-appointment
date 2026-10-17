/**
 * 侧栏菜单：从路由表派生，按当前用户角色过滤。
 *
 * 路由即菜单——加页面只在 routes.ts 登记并给 meta.roles，这里自动出现，
 * 避免「路由加了但菜单忘了加」这类两处维护的漂移。
 */

import { computed } from 'vue'
import type { MenuOption } from 'naive-ui'
import { routes } from '@/router/routes'
import { useAuthStore } from '@/stores/auth'
import { ROLE_LABELS } from '@/types/enums'
import { SideMenuIcon } from '@/components/layout/SideMenuIcon'

export function useMenu() {
  const auth = useAuthStore()

  const menuOptions = computed<MenuOption[]>(() => {
    // 主布局是 routes 里 path='/' 那条，业务页都在它的 children
    const layout = routes.find((r) => r.path === '/')
    const children = layout?.children ?? []
    const role = auth.role

    return children
      .filter((r) => {
        if (r.meta?.hidden) return false
        const roles = r.meta?.roles
        if (!roles || roles.length === 0) return true
        return role ? roles.includes(role) : false
      })
      .map((r) => ({
        label: r.meta?.title ?? (r.name as string),
        key: r.name as string,
        icon: r.meta?.icon ? SideMenuIcon(r.meta.icon) : undefined,
      }))
  })

  const roleLabel = computed(() =>
    auth.role ? ROLE_LABELS[auth.role] ?? auth.role : '',
  )

  return { menuOptions, roleLabel }
}
