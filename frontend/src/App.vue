<script setup lang="ts">
/**
 * 根组件。只做两件事：注入 Naive 主题、挂全局消息/对话框 provider。
 *
 * NConfigProvider 注入临床克制风主题（见 styles/naive-overrides.ts）；
 * n-message-provider / n-dialog-provider 让任意组件能用 useMessage/useDialog，
 * 不必层层传 prop。中文语言包统一在此设置。
 */
import { NConfigProvider, NMessageProvider, NDialogProvider, NLoadingBarProvider, zhCN, dateZhCN } from 'naive-ui'
import { themeOverrides } from '@/styles/naive-overrides'
</script>

<template>
  <n-config-provider :theme-overrides="themeOverrides" :locale="zhCN" :date-locale="dateZhCN">
    <n-loading-bar-provider>
      <n-message-provider>
        <n-dialog-provider>
          <router-view />
        </n-dialog-provider>
      </n-message-provider>
    </n-loading-bar-provider>
  </n-config-provider>
</template>
