<script setup lang="ts">
/**
 * 工作台首页。按角色展示快捷入口与关键提示。
 * 克制：不用大卡片网格堆砌，按角色给「今天要做什么」的直达入口。
 */
import { computed } from 'vue'
import { NCard, NGrid, NGridItem, NButton, NEmpty } from 'naive-ui'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { ROLE_LABELS } from '@/types/enums'

const router = useRouter()
const auth = useAuthStore()

interface QuickEntry {
  title: string
  desc: string
  route: string
}

const entries = computed<QuickEntry[]>(() => {
  switch (auth.role) {
    case 'patient':
      return [
        { title: '智能分诊', desc: '描述症状，推荐科室与号源', route: 'patient-triage' },
        { title: '健康咨询', desc: '基于院内知识库的流式问答', route: 'patient-consult' },
        { title: '预约挂号', desc: '查看号源并预约', route: 'patient-appointments' },
        { title: '我的报告', desc: '查看检查报告与 AI 解读', route: 'patient-reports' },
      ]
    case 'doctor':
      return [
        { title: '待诊队列', desc: '今日签到患者，开始接诊', route: 'doctor-queue' },
        { title: '我的病历', desc: '历史接诊记录', route: 'doctor-visits' },
      ]
    case 'pharmacist':
      return [
        { title: '处方审核与发药', desc: '四查十对，先审后发', route: 'pharmacist-review' },
        { title: '发药记录', desc: '已发药流水', route: 'pharmacist-records' },
      ]
    case 'cashier':
      return [{ title: '收费台', desc: '待收费单、收款与退费', route: 'cashier-orders' }]
    case 'lab_tech':
      return [
        { title: '检验工作台', desc: '接收申请、录入报告', route: 'lab-queue' },
        { title: '全部报告', desc: '已出报告与 AI 解读状态', route: 'lab-reports' },
      ]
    case 'admin':
      return [
        { title: 'AI 运行监控', desc: '指标曲线与全链路回放', route: 'admin-monitor' },
        { title: 'AI 评测中心', desc: '检索质量、RAGAS 与回归门禁', route: 'admin-evaluation' },
        { title: '知识库管理', desc: '文档与索引维护', route: 'admin-knowledge' },
      ]
    default:
      return []
  }
})

const greeting = computed(() => {
  const name = auth.displayName
  const role = auth.role ? ROLE_LABELS[auth.role] : ''
  return `${name}${role ? '（' + role + '）' : ''}`
})

function go(route: string) {
  router.push({ name: route })
}
</script>

<template>
  <div class="page-container home">
    <div class="home-head">
      <h1 class="page-title home-title">你好，{{ greeting }}</h1>
      <p class="page-subtitle home-subtitle">下面是你常用功能的快捷入口</p>
    </div>

    <n-grid v-if="entries.length" cols="1 s:2 l:2 xl:4" responsive="screen" :x-gap="20" :y-gap="20" class="home-grid">
      <n-grid-item v-for="e in entries" :key="e.route">
        <n-card class="entry-card" hoverable @click="go(e.route)">
          <div class="entry-card__title">{{ e.title }}</div>
          <div class="entry-card__desc">{{ e.desc }}</div>
          <n-button text type="primary" @click.stop="go(e.route)">
            进入 →
          </n-button>
        </n-card>
      </n-grid-item>
    </n-grid>
    <n-empty v-else description="暂无可用功能入口" />
  </div>
</template>

<style scoped>
.home {
  padding: var(--space-8) var(--space-8);
}
.home-head {
  margin-bottom: var(--space-8);
}
.home-title {
  font-size: 28px;
  font-weight: 700;
  color: var(--gray-9);
  margin: 0 0 var(--space-2);
}
.home-subtitle {
  font-size: var(--font-lg);
  color: var(--gray-7);
  margin: 0;
}
.home-grid {
  max-width: 1600px;
}
.entry-card {
  cursor: pointer;
  height: 100%;
  min-height: 200px;
  border: 1px solid var(--border-color-light);
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.entry-card:hover {
  border-color: var(--brand-5);
  box-shadow: var(--shadow-md);
}
.entry-card__title {
  font-size: 20px;
  font-weight: 600;
  color: var(--gray-9);
  margin-bottom: var(--space-3);
}
.entry-card__desc {
  font-size: var(--font-base);
  color: var(--gray-7);
  margin-bottom: var(--space-5);
  min-height: 44px;
  line-height: 1.7;
}
</style>
