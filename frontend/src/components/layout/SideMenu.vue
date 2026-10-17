<script setup lang="ts">
/**
 * 侧栏：品牌区 + 角色过滤后的菜单。
 *
 * 折叠态只显示图标（宽度收到 --sidebar-width-collapsed），
 * 展开态显示图标 + 文字。折叠状态由 layout store 持有，刷新后保留。
 */
import { computed } from 'vue'
import { NMenu, NIcon } from 'naive-ui'
import { useRouter } from 'vue-router'
import { useMenu } from '@/composables/useMenu'
import { useLayoutStore } from '@/stores/layout'
import { MedkitOutline } from '@vicons/ionicons5'

const router = useRouter()
const layout = useLayoutStore()
const { menuOptions } = useMenu()

const collapsed = computed(() => layout.sidebarCollapsed)

function handleSelect(key: string) {
  router.push({ name: key })
}
</script>

<template>
  <div class="sidebar" :class="{ 'sidebar--collapsed': collapsed }">
    <div class="sidebar__brand">
      <n-icon :size="22" color="#ffffff">
        <medkit-outline />
      </n-icon>
      <span v-show="!collapsed" class="sidebar__brand-text">IMCS 诊疗系统</span>
    </div>
    <n-menu
      class="sidebar__menu"
      :collapsed="collapsed"
      :collapsed-width="56"
      :collapsed-icon-size="20"
      :options="menuOptions"
      :value="(router.currentRoute.value.name as string)"
      @update:value="handleSelect"
    />
  </div>
</template>

<style scoped>
.sidebar {
  width: var(--sidebar-width);
  height: 100%;
  background: var(--brand-7);
  display: flex;
  flex-direction: column;
  transition: width var(--transition-base);
  overflow: hidden;
}
.sidebar--collapsed {
  width: var(--sidebar-width-collapsed);
}
.sidebar__brand {
  height: var(--header-height);
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: 0 var(--space-4);
  color: #fff;
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
  flex-shrink: 0;
}
.sidebar__brand-text {
  font-size: var(--font-base);
  font-weight: 600;
  white-space: nowrap;
}
.sidebar__menu {
  flex: 1;
  overflow-y: auto;
  padding-top: var(--space-2);
}
/* 深色侧栏上的菜单文字与选中态 */
.sidebar :deep(.n-menu .n-menu-item-content) {
  color: rgba(255, 255, 255, 0.72);
}
.sidebar :deep(.n-menu .n-menu-item-content:hover) {
  color: #fff;
  background: rgba(255, 255, 255, 0.08);
}
.sidebar :deep(.n-menu .n-menu-item-content--selected) {
  color: #fff;
  background: var(--brand-6);
}
</style>
