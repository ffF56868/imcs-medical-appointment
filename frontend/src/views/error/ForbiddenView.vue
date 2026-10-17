<script setup lang="ts">
import { NResult, NButton } from 'naive-ui'
import { useRouter } from 'vue-router'
const router = useRouter()
</script>

<template>
  <div class="center-page">
    <n-result status="403" title="无访问权限" description="当前账号没有访问该页面的权限，如需访问请联系管理员。">
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
