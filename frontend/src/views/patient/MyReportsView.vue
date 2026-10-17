<script setup lang="ts">
/**
 * 我的检查报告。患者只读自己的报告，含 AI 解读结果。
 * AI 解读按状态差异化展示：completed 显示摘要与建议，processing/pending
 * 显示进行中，failed 提示可重试（重试由检验科触发，患者端只读状态）。
 */
import { ref, onMounted, h } from 'vue'
import {
  NCard, NDataTable, NTag, NSpin, NEmpty, useMessage, NModal,
  NDescriptions, NDescriptionsItem, NAlert, NList, NListItem,
} from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import { listMyReports } from '@/api/exam'
import { ApiError } from '@/api/client'
import type { ExamReport } from '@/types/clinical'
import { AI_STATUS_LABELS, URGENCY_LABELS } from '@/types/enums'

const message = useMessage()
const loading = ref(false)
const rows = ref<ExamReport[]>([])
const detail = ref<ExamReport | null>(null)

async function load() {
  loading.value = true
  try {
    const resp = await listMyReports({ page: 1, page_size: 50 })
    rows.value = resp.items
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '加载报告失败')
  } finally {
    loading.value = false
  }
}

function aiTagType(status: string): 'success' | 'info' | 'warning' | 'error' | 'default' {
  switch (status) {
    case 'completed': return 'success'
    case 'processing':
    case 'pending': return 'info'
    case 'failed': return 'error'
    default: return 'default'
  }
}

const columns: DataTableColumns<ExamReport> = [
  { title: '检查项目', key: 'exam_name' },
  { title: '类型', key: 'exam_type' },
  {
    title: '紧急度', key: 'urgency',
    render: (r) => h(NTag, { size: 'small', bordered: false, type: r.urgency === 'stat' ? 'error' : r.urgency === 'urgent' ? 'warning' : 'default' }, { default: () => URGENCY_LABELS[r.urgency] }),
  },
  { title: '报告时间', key: 'created_at', className: 'num', render: (r) => (r.created_at || '').slice(0, 16).replace('T', ' ') },
  {
    title: 'AI 解读', key: 'ai_status',
    render: (r) => h(NTag, { size: 'small', bordered: false, type: aiTagType(r.ai?.ai_status ?? 'pending') }, { default: () => AI_STATUS_LABELS[r.ai?.ai_status ?? 'pending'] }),
  },
  {
    title: '操作', key: 'action',
    render: (r) => h('a', { onClick: () => { detail.value = r } }, '查看'),
  },
]

onMounted(load)
</script>

<template>
  <div class="page-container">
    <n-card title="检查报告" :bordered="false">
      <n-spin :show="loading">
        <n-data-table
          :columns="columns"
          :data="rows"
          :bordered="false"
          size="small"
          :row-key="(r: ExamReport) => r.id"
          :pagination="{ pageSize: 10 }"
        />
        <n-empty v-if="!loading && !rows.length" description="暂无检查报告" class="empty-block" />
      </n-spin>
    </n-card>

    <n-modal
      :show="detail !== null"
      preset="card"
      title="报告详情"
      style="max-width: 720px"
      @update:show="(v: boolean) => { if (!v) detail = null }"
    >
      <template v-if="detail">
        <n-descriptions :column="2" label-placement="left" bordered size="small">
          <n-descriptions-item label="检查项目">{{ detail.exam_name }}</n-descriptions-item>
          <n-descriptions-item label="类型">{{ detail.exam_type }}</n-descriptions-item>
          <n-descriptions-item label="结论" :span="2">{{ detail.conclusion || '—' }}</n-descriptions-item>
        </n-descriptions>

        <n-alert v-if="detail.critical" type="error" :bordered="false" class="block-gap">
          危急值提醒：{{ detail.critical_message || '本报告含危急值，请立即联系医生' }}
        </n-alert>

        <div v-if="detail.abnormal_items?.length" class="block-gap">
          <div class="section-title">异常项</div>
          <n-data-table
            :columns="[
              { title: '项目', key: 'name' },
              { title: '结果', key: 'value', align: 'right', className: 'cell-number' },
              { title: '单位', key: 'unit', render: (r: any) => r.unit || '—' },
              { title: '参考范围', key: 'ref', render: (r: any) => (r.ref_low != null && r.ref_high != null ? `${r.ref_low} ~ ${r.ref_high}` : '—'), className: 'num' },
              { title: '提示', key: 'flag', render: (r: any) => r.flag || '—' },
            ]"
            :data="detail.abnormal_items"
            :bordered="false"
            size="small"
            :row-key="(r: any) => r.name"
          />
        </div>

        <div class="block-gap">
          <div class="section-title">AI 解读</div>
          <n-alert v-if="detail.ai?.ai_status === 'completed'" type="info" :bordered="false">
            <div class="ai-summary">{{ detail.ai.summary }}</div>
            <n-list v-if="detail.ai.suggestions?.length" class="ai-suggest">
              <n-list-item v-for="(s, i) in detail.ai.suggestions" :key="i">{{ s }}</n-list-item>
            </n-list>
            <div class="ai-note">
              模型 {{ detail.ai.model || '—' }} · 严重度 {{ detail.ai.severity || '—' }} ·
              此解读仅供医生参考，最终诊断以医生意见为准
            </div>
          </n-alert>
          <n-alert v-else-if="detail.ai?.ai_status === 'processing' || detail.ai?.ai_status === 'pending'" type="default" :bordered="false">
            AI 解读生成中，请稍后刷新查看。
          </n-alert>
          <n-alert v-else-if="detail.ai?.ai_status === 'failed'" type="warning" :bordered="false">
            AI 解读生成失败，可请检验科重新触发。
          </n-alert>
          <n-alert v-else type="default" :bordered="false">暂无 AI 解读。</n-alert>
        </div>
      </template>
    </n-modal>
  </div>
</template>

<style scoped>
.empty-block { margin-top: var(--space-8); }
.block-gap { margin-top: var(--space-4); }
.section-title {
  font-size: var(--font-sm);
  font-weight: 600;
  color: var(--gray-9);
  margin-bottom: var(--space-2);
}
.ai-summary { white-space: pre-wrap; line-height: 1.6; }
.ai-suggest { margin-top: var(--space-2); }
.ai-note {
  margin-top: var(--space-2);
  font-size: var(--font-xs);
  color: var(--gray-7);
}
</style>
