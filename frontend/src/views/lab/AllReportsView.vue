<script setup lang="ts">
/**
 * 检验科「全部报告」。全域列表 + 详情弹窗。
 *
 * AI 解读是异步的：录入返回时多半还在排队，详情弹窗按 2s→8s 退避轮询
 * GET /exams/reports/{id}，终态（completed/failed/skipped）停止。
 * 危急值是服务端规则当场算出的，列表行直接标红，不等 AI。
 */
import { ref, computed, onMounted, onBeforeUnmount, h } from 'vue'
import {
  NCard, NDataTable, NButton, NTag, NModal, NAlert,
  NDescriptions, NDescriptionsItem, useMessage,
} from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import { listAllReports, getReport, triggerInterpret } from '@/api/exam'
import { ApiError } from '@/api/client'
import type { ExamReport } from '@/types/clinical'
import { AI_STATUS_LABELS, URGENCY_LABELS } from '@/types/enums'

const message = useMessage()

const loading = ref(false)
const rows = ref<ExamReport[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = 20

async function load() {
  loading.value = true
  try {
    const resp = await listAllReports({ page: page.value, page_size: pageSize })
    rows.value = resp.items
    total.value = resp.total
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '加载报告失败')
  } finally {
    loading.value = false
  }
}
onMounted(load)

/* ==================== 详情 + AI 轮询 ==================== */
const detail = ref<ExamReport | null>(null)
const detailVisible = computed({
  get: () => detail.value !== null,
  set: (v: boolean) => { if (!v) { stopPolling(); detail.value = null } },
})

let pollTimer: ReturnType<typeof setTimeout> | null = null
let pollDelay = 2000
const POLL_MAX_MS = 120_000
let pollElapsed = 0

function stopPolling() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null }
  pollDelay = 2000
  pollElapsed = 0
}

const AI_TERMINAL = new Set(['completed', 'failed', 'skipped'])

async function openDetail(row: ExamReport) {
  stopPolling()
  detail.value = row
  await refreshDetail(row.id)
}

async function refreshDetail(id: string) {
  try {
    const r = await getReport(id)
    if (detail.value?.id !== id) return // 弹窗已切换/关闭
    detail.value = r
    if (!AI_TERMINAL.has(r.ai.ai_status) && pollElapsed < POLL_MAX_MS) {
      pollTimer = setTimeout(() => {
        pollElapsed += pollDelay
        pollDelay = Math.min(pollDelay * 1.6, 8000)
        refreshDetail(id)
      }, pollDelay)
    }
  } catch {
    /* 轮询失败静默：用户可手动刷新 */
  }
}

onBeforeUnmount(stopPolling)

async function doReinterpret() {
  if (!detail.value) return
  try {
    await triggerInterpret(detail.value.id, '检验科人工重触发')
    message.success('已重新排队解读')
    await refreshDetail(detail.value.id)
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '触发失败')
  }
}

function fmtTime(v: string | null): string {
  return (v || '').slice(0, 16).replace('T', ' ')
}

function aiTag(s: string) {
  const t = s === 'completed' ? 'success' : s === 'failed' ? 'error' : s === 'processing' ? 'info' : 'default'
  return h(NTag, { size: 'small', bordered: false, type: t }, { default: () => AI_STATUS_LABELS[s] ?? s })
}

const columns: DataTableColumns<ExamReport> = [
  {
    title: '报告', key: 'exam_name', ellipsis: { tooltip: true },
    render: (r) => h('span', {}, [
      r.critical
        ? h(NTag, { size: 'tiny', bordered: false, type: 'error', class: 'crit-tag' }, { default: () => '危急' })
        : null,
      ` ${r.exam_name || r.exam_type}`,
    ]),
  },
  {
    title: '紧急度', key: 'urgency', width: 80,
    render: (r) => (r.urgency ? URGENCY_LABELS[r.urgency] ?? r.urgency : '—'),
  },
  {
    title: '申请状态', key: 'request_status', width: 90,
    render: (r) => r.request_status || '—',
  },
  {
    title: 'AI 解读', key: 'ai_status', width: 90,
    render: (r) => aiTag(r.ai.ai_status),
  },
  {
    title: '结论', key: 'conclusion', ellipsis: { tooltip: true },
    render: (r) => r.conclusion || '—',
  },
  { title: '时间', key: 'created_at', width: 130, className: 'num', render: (r) => fmtTime(r.created_at) },
  {
    title: '操作', key: 'a', width: 70,
    render: (r) => h('a', { onClick: () => openDetail(r) }, '查看'),
  },
]

const pagination = computed(() => ({
  page: page.value,
  pageSize,
  itemCount: total.value,
  onChange: (p: number) => { page.value = p; load() },
  prefix: ({ itemCount }: { itemCount?: number }) => `共 ${itemCount ?? 0} 份`,
}))
</script>

<template>
  <div class="page-container">
    <n-card :bordered="false">
      <template #header>全部报告</template>
      <template #header-extra>
        <n-button size="small" quaternary @click="load">刷新</n-button>
      </template>
      <n-data-table
        :columns="columns" :data="rows" :loading="loading" :bordered="false"
        size="small" remote :row-key="(r: ExamReport) => r.id" :pagination="pagination"
      />
    </n-card>

    <n-modal v-model:show="detailVisible" preset="card" title="报告详情" style="max-width: 720px">
      <template v-if="detail">
        <n-alert v-if="detail.critical" type="error" :bordered="false" class="rep-block">
          {{ detail.critical_message || '出现危急值，请立即联系临床' }}
        </n-alert>

        <n-descriptions :column="2" label-placement="left" size="small" bordered class="rep-block">
          <n-descriptions-item label="项目">{{ detail.exam_name || detail.exam_type }}</n-descriptions-item>
          <n-descriptions-item label="时间">{{ fmtTime(detail.created_at) }}</n-descriptions-item>
          <n-descriptions-item label="结论" :span="2">{{ detail.conclusion || '—' }}</n-descriptions-item>
        </n-descriptions>

        <div class="rep-raw">
          <div class="rep-raw__label">报告内容</div>
          <div class="rep-raw__text">{{ detail.raw_content }}</div>
        </div>

        <div v-if="detail.abnormal_items.length" class="rep-block">
          <div class="rep-raw__label">异常指标（规则判定）</div>
          <div v-for="(a, i) in detail.abnormal_items" :key="i" class="abn-line">
            <span>{{ a.flag }}</span>
            <span class="abn-line__v">{{ a.value }}{{ a.unit ? ` ${a.unit}` : '' }}
              （参考 {{ a.ref_low ?? '—' }}–{{ a.ref_high ?? '—' }}）</span>
          </div>
        </div>

        <div class="rep-ai rep-block">
          <div class="rep-ai__head">
            <span class="rep-raw__label">AI 解读</span>
            <n-tag size="tiny" :bordered="false" :type="detail.ai.ai_status === 'completed' ? 'success' : detail.ai.ai_status === 'failed' ? 'error' : 'default'">
              {{ AI_STATUS_LABELS[detail.ai.ai_status] }}
            </n-tag>
            <n-button v-if="['failed', 'skipped'].includes(detail.ai.ai_status)" size="tiny" @click="doReinterpret">
              重新解读
            </n-button>
          </div>
          <div v-if="detail.ai.ai_status === 'pending' || detail.ai.ai_status === 'processing'" class="rep-ai__pending">
            解读生成中，页面自动刷新…
          </div>
          <template v-else-if="detail.ai.summary">
            <p class="rep-ai__summary">{{ detail.ai.summary }}</p>
            <ul v-if="detail.ai.suggestions.length" class="rep-ai__list">
              <li v-for="(s, i) in detail.ai.suggestions" :key="i">{{ s }}</li>
            </ul>
            <n-alert type="warning" :bordered="false" class="rep-ai__note">
              AI 内容为辅助解读，需医生审核后方可作为诊疗依据。
            </n-alert>
          </template>
        </div>
      </template>
    </n-modal>
  </div>
</template>

<style scoped>
.crit-tag { margin-right: 2px; }
.rep-block { margin-bottom: var(--space-4); }
.rep-raw__label { font-size: var(--font-sm); color: var(--gray-6); margin-bottom: var(--space-2); }
.rep-raw__text { white-space: pre-wrap; color: var(--gray-8); font-size: var(--font-base); line-height: 1.6; }
.abn-line { display: flex; justify-content: space-between; padding: var(--space-1) 0; font-size: var(--font-sm); border-bottom: 1px dashed var(--border-color-light); }
.abn-line__v { color: var(--gray-6); }
.rep-ai__head { display: flex; align-items: center; gap: var(--space-2); margin-bottom: var(--space-2); }
.rep-ai__pending { color: var(--gray-6); font-size: var(--font-sm); }
.rep-ai__summary { margin: 0 0 var(--space-2); color: var(--gray-8); line-height: 1.6; }
.rep-ai__list { margin: 0; padding-left: var(--space-5); color: var(--gray-7); font-size: var(--font-sm); }
.rep-ai__note { margin-top: var(--space-3); }
</style>
