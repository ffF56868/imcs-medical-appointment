<script setup lang="ts">
/**
 * 占位页模板。批次 2-4 的角色页面尚未实现，先用它占位，
 * 保证 vite build 的动态导入都能解析（缺文件会直接构建失败）。
 * 每批实现时替换对应文件即可。
 */
import { NCard, NEmpty } from 'naive-ui'

defineProps<{ title: string; note?: string }>()
</script>

<template>
  <div class="page-container">
    <n-card :title="title" :bordered="false">
      <n-empty :description="note || '该页面将在后续批次实现'" class="placeholder" />
    </n-card>
  </div>
</template>

<style scoped>
.placeholder { margin-top: var(--space-12); }
</style>
