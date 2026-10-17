<script setup lang="ts">
/**
 * 顶栏：折叠按钮 + 页面标题 + 用户区（角色标签、登出）。
 * 高度固定 --header-height，白底细下边框，与侧栏深色形成克制的对比。
 */
import { computed, h } from 'vue'
import { NIcon, NButton, NTag, NDropdown, useDialog } from 'naive-ui'
import { MenuOutline, PersonCircleOutline, LogOutOutline } from '@vicons/ionicons5'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useLayoutStore } from '@/stores/layout'
import { useMenu } from '@/composables/useMenu'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const layout = useLayoutStore()
const { roleLabel } = useMenu()
const dialog = useDialog()

const pageTitle = computed(() => (route.meta.title as string) || '')

const userMenuOptions = [
  {
    label: '退出登录',
    key: 'logout',
    icon: () => h(NIcon, null, { default: () => h(LogOutOutline) }),
  },
]

function handleUserAction(key: string) {
  if (key === 'logout') {
    dialog.warning({
      title: '退出登录',
      content: '确定要退出当前账号吗？',
      positiveText: '退出',
      negativeText: '取消',
      onPositiveClick: async () => {
        await auth.logout()
        router.replace({ name: 'login' })
      },
    })
  }
}
</script>

<template>
  <header class="header">
    <div class="header__left">
      <n-button quaternary circle @click="layout.toggle()">
        <template #icon>
          <n-icon :size="20">
            <menu-outline />
          </n-icon>
        </template>
      </n-button>
      <span class="header__title">{{ pageTitle }}</span>
    </div>
    <div class="header__right">
      <n-tag :bordered="false" type="info" size="small">{{ roleLabel }}</n-tag>
      <n-dropdown :options="userMenuOptions" @select="handleUserAction">
        <div class="header__user">
          <n-icon :size="20" color="#1b5e7b">
            <person-circle-outline />
          </n-icon>
          <span class="header__username">{{ auth.displayName }}</span>
        </div>
      </n-dropdown>
    </div>
  </header>
</template>

<style scoped>
.header {
  height: var(--header-height);
  background: var(--gray-1);
  border-bottom: 1px solid var(--border-color-light);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 var(--space-5);
  flex-shrink: 0;
}
.header__left {
  display: flex;
  align-items: center;
  gap: var(--space-3);
}
.header__title {
  font-size: var(--font-lg);
  font-weight: 600;
  color: var(--gray-9);
}
.header__right {
  display: flex;
  align-items: center;
  gap: var(--space-3);
}
.header__user {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  cursor: pointer;
  padding: var(--space-1) var(--space-2);
  border-radius: var(--radius-md);
}
.header__user:hover {
  background: var(--gray-3);
}
.header__username {
  font-size: var(--font-base);
  color: var(--gray-8);
}
</style>
