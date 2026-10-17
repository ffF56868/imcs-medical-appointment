/**
 * 布局状态：侧栏折叠。用 localStorage 持久化，刷新后保留用户偏好。
 */
import { defineStore } from 'pinia'
import { ref } from 'vue'

const KEY = 'imcs.sidebar_collapsed'

export const useLayoutStore = defineStore('layout', () => {
  const sidebarCollapsed = ref<boolean>(localStorage.getItem(KEY) === '1')

  function toggle() {
    sidebarCollapsed.value = !sidebarCollapsed.value
    localStorage.setItem(KEY, sidebarCollapsed.value ? '1' : '0')
  }

  return { sidebarCollapsed, toggle }
})
