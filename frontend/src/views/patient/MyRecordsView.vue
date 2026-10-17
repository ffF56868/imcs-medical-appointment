<script setup lang="ts">
/**
 * 我的病历。患者只读自己的就诊记录（后端行级隔离，越权返回 404）。
 */
import { ref, computed, onMounted, h } from 'vue'
import {
  NCard, NDataTable, NTag, NSpin, NEmpty, useMessage,
  NDescriptions, NDescriptionsItem, NModal,
} from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import { listMyVisits } from '@/api/consult'
import { ApiError } from '@/api/client'
import type { Visit } from '@/types/clinical'
import { VISIT_STATUS_LABELS } from '@/types/enums'

const message = useMessage()
const loading = ref(false)
const rows = ref<Visit[]>([])
const detail = ref<Visit | null>(null)
const detailVisible = computed({
  get: () => detail.value !== null,
  set: (v: boolean) => { if (!v) detail.value = null },
})

async function load() {
  loading.value = true
  try {
    const resp = await listMyVisits({ page: 1, page_size: 50 })
    rows.value = resp.items
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '加载病历失败')
  } finally {
    loading.value = false
  }
}

function fmtTime(v: string | null): string {
  return (v || '').slice(0, 16).replace('T', ' ')
}

const columns: DataTableColumns<Visit> = [
  { title: '就诊时间', key: 'started_at', className: 'num', render: (r) => fmtTime(r.started_at || r.created_at) },
  { title: '医生', key: 'doctor_name', render: (r) => r.doctor_name || '—' },
  { title: '主诉', key: 'chief_complaint', ellipsis: { tooltip: true }, render: (r) => r.chief_complaint || '—' },
  { title: '诊断', key: 'diagnosis', ellipsis: { tooltip: true }, render: (r) => r.diagnosis || '—' },
  {
    title: '状态', key: 'status',
    render: (r) => h(NTag, { size: 'small', bordered: false }, { default: () => VISIT_STATUS_LABELS[r.status] }),
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
    <n-card title="我的病历" :bordered="false">
      <n-spin :show="loading">
        <n-data-table
          :columns="columns"
          :data="rows"
          :bordered="false"
          size="small"
          :row-key="(r: Visit) => r.id"
          :pagination="{ pageSize: 10 }"
        />
        <n-empty v-if="!loading && !rows.length" description="暂无就诊记录" class="empty-block" />
      </n-spin>
    </n-card>

    <n-modal v-model:show="detailVisible" preset="card" title="病历详情" style="max-width: 640px">
      <n-descriptions v-if="detail" :column="1" label-placement="left" bordered size="small">
        <n-descriptions-item label="主诉">{{ detail.chief_complaint || '—' }}</n-descriptions-item>
        <n-descriptions-item label="现病史">{{ detail.present_illness || '—' }}</n-descriptions-item>
        <n-descriptions-item label="既往史">{{ detail.past_history || '—' }}</n-descriptions-item>
        <n-descriptions-item label="体格检查">{{ detail.physical_exam || '—' }}</n-descriptions-item>
        <n-descriptions-item label="诊断">{{ detail.diagnosis || '—' }}</n-descriptions-item>
        <n-descriptions-item label="ICD 编码">{{ detail.icd_codes?.join('、') || '—' }}</n-descriptions-item>
        <n-descriptions-item label="处理建议">{{ detail.treatment_advice || '—' }}</n-descriptions-item>
      </n-descriptions>
    </n-modal>
  </div>
</template>

<style scoped>
.empty-block { margin-top: var(--space-8); }
</style>
