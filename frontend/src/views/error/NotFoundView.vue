<script setup lang="ts">
import { NResult, NButton } from 'naive-ui'
import { useRouter } from 'vue-router'
const router = useRouter()
</script>

<template>
  <div class="center-page">
    <n-result status="404" title="页面不存在" description="你访问的页面不存在或已被移除。">
      <template #footer>
        <n-button @click="router.replace({ name: 'home' })">返回工作台</n-button>
      </template>
    </n-result>
  </div>
</template>

<style scoped>
.center-page {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
}
</style>
