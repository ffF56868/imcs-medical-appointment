<script setup lang="ts">
/**
 * 主布局：左侧栏 + 右内容区（顶栏 + 路由出口）。
 * 六角色共用这一套外壳，内容区由各角色页面填充。
 */
import SideMenu from './SideMenu.vue'
import HeaderBar from './HeaderBar.vue'
</script>

<template>
  <div class="layout">
    <aside class="layout__sidebar">
      <side-menu />
    </aside>
    <div class="layout__main">
      <header-bar />
      <main class="layout__content">
        <router-view v-slot="{ Component }">
          <keep-alive :max="6">
            <component :is="Component" />
          </keep-alive>
        </router-view>
      </main>
    </div>
  </div>
</template>

<style scoped>
.layout {
  display: flex;
  height: 100%;
  overflow: hidden;
}
.layout__sidebar {
  flex-shrink: 0;
  height: 100%;
}
.layout__main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0; /* 允许内容区在窄屏收缩，避免撑破布局 */
}
.layout__content {
  flex: 1;
  overflow-y: auto;
  background: var(--gray-3);
}
</style>
