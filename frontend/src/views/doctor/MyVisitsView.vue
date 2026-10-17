<script setup lang="ts">
/**
 * 医生侧「我的病历」：名下全部就诊记录，可按状态筛选。
 * 未结案的病历提供「继续书写」入口，已结案只读。
 */
import { ref, computed, onMounted, h, watch } from 'vue'
import {
  NCard, NDataTable, NButton, NTag, NSpace, NSelect, useMessage,
} from 'naive-ui'
import type { DataTableColumns, SelectOption } from 'naive-ui'
import { useRouter } from 'vue-router'
import { listVisits } from '@/api/consult'
import { ApiError } from '@/api/client'
import type { Visit } from '@/types/clinical'
import { VISIT_STATUS_LABELS } from '@/types/enums'

const router = useRouter()
const message = useMessage()

const loading = ref(false)
const rows = ref<Visit[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(20)
const statusFilter = ref<string | null>(null)

const statusOptions: SelectOption[] = [
  { label: '全部状态', value: '' },
  ...Object.entries(VISIT_STATUS_LABELS).map(([value, label]) => ({ label, value })),
]

async function load() {
  loading.value = true
  try {
    const resp = await listVisits({
      status: statusFilter.value || undefined,
      page: page.value,
      page_size: pageSize.value,
    })
    rows.value = resp.items
    total.value = resp.total
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '加载病历失败')
  } finally {
    loading.value = false
  }
}

watch([page, statusFilter], load)
onMounted(load)

const tagType = (s: Visit['status']) =>
  s === 'closed' ? 'success' : s === 'documented' ? 'info' : 'warning'

const columns: DataTableColumns<Visit> = [
  {
    title: '就诊时间', key: 'started_at', width: 150, className: 'num',
    render: (r) => (r.started_at || r.created_at || '').slice(0, 16).replace('T', ' ') || '—',
  },
  { title: '主诉', key: 'chief_complaint', ellipsis: { tooltip: true }, render: (r) => r.chief_complaint || '—' },
  { title: '诊断', key: 'diagnosis', ellipsis: { tooltip: true }, render: (r) => r.diagnosis || '—' },
  {
    title: '检查', key: 'exam_request_count', width: 70, align: 'right', className: 'num',
    render: (r) => (r.exam_request_count ? `${r.exam_request_count} 项` : '—'),
  },
  {
    title: '状态', key: 'status', width: 100,
    render: (r) => h(NTag, { size: 'small', bordered: false, type: tagType(r.status) },
      { default: () => VISIT_STATUS_LABELS[r.status] }),
  },
  {
    title: '操作', key: 'action', width: 110,
    render: (r) =>
      r.status === 'closed'
        ? h('span', { class: 'cell-muted' }, '已结案')
        : h(NButton, {
            size: 'tiny',
            onClick: () => router.push({ name: 'doctor-visit', params: { visitId: r.id } }),
          }, { default: () => (r.status === 'in_progress' ? '继续接诊' : '补写病历') }),
  },
]

const pagination = computed(() => ({
  page: page.value,
  pageSize: pageSize.value,
  itemCount: total.value,
  showSizePicker: true,
  pageSizes: [20, 50],
  prefix: ({ itemCount }: { itemCount?: number }) => `共 ${itemCount ?? 0} 条`,
  onChange: (p: number) => { page.value = p },
  onUpdatePageSize: (ps: number) => { pageSize.value = ps; page.value = 1 },
}))
</script>

<template>
  <div class="page-container">
    <n-card :bordered="false">
      <template #header>我的病历</template>
      <template #header-extra>
        <n-space align="center" :size="12">
          <n-select
            v-model:value="statusFilter"
            :options="statusOptions"
            size="small"
            placeholder="全部状态"
            style="width: 140px"
            @update:value="page = 1"
          />
        </n-space>
      </template>
      <n-data-table
        :columns="columns"
        :data="rows"
        :loading="loading"
        :bordered="false"
        size="small"
        remote
        :row-key="(r: Visit) => r.id"
        :pagination="pagination"
      />
    </n-card>
  </div>
</template>

<style scoped>
.cell-muted { color: var(--gray-5); font-size: var(--font-sm); }
</style>
