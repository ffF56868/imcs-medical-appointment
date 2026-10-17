<script setup lang="ts">
/**
 * 发药记录（药师全域视图）。已审核/已发药的处方流水，支持按天翻查。
 */
import { ref, computed, onMounted } from 'vue'
import {
  NCard, NDataTable, NButton, NTag, useMessage,
} from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import { listRecordsAll } from '@/api/dispensing'
import { ApiError } from '@/api/client'
import type { DispensingRecord } from '@/types/clinical'
import { PRESCRIPTION_STATUS_LABELS } from '@/types/enums'
import { formatFen } from '@/utils/currency'

const message = useMessage()
const loading = ref(false)
const rows = ref<DispensingRecord[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = 20

async function load() {
  loading.value = true
  try {
    const resp = await listRecordsAll({ page: page.value, page_size: pageSize })
    rows.value = resp.items
    total.value = resp.total
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '加载发药记录失败')
  } finally {
    loading.value = false
  }
}
onMounted(load)

function fmtTime(v: string | null): string {
  return (v || '').slice(0, 16).replace('T', ' ')
}

const columns: DataTableColumns<DispensingRecord> = [
  {
    title: '发药时间', key: 'dispensed_at', width: 150, className: 'num',
    render: (r) => fmtTime(r.dispensed_at) || '—',
  },
  {
    title: '处方', key: 'prescription_id', width: 100, className: 'num',
    render: (r) => r.prescription_id.slice(0, 8),
  },
  {
    title: '金额', key: 'total_amount', width: 100, align: 'right', className: 'num',
    render: (r) => (r.total_amount != null ? formatFen(r.total_amount) : '—'),
  },
  {
    title: '处方状态', key: 'prescription_status', width: 100,
    render: (r) => {
      const t = r.prescription_status === 'dispensed' ? 'success' : 'default'
      return h(NTag, { size: 'small', bordered: false, type: t },
        { default: () => PRESCRIPTION_STATUS_LABELS[r.prescription_status] ?? r.prescription_status })
    },
  },
  {
    title: '审核意见', key: 'verification_notes', ellipsis: { tooltip: true },
    render: (r) => r.verification_notes || '—',
  },
  {
    title: '发药备注', key: 'notes', ellipsis: { tooltip: true },
    render: (r) => r.notes || '—',
  },
]

import { h } from 'vue'

const pagination = computed(() => ({
  page: page.value,
  pageSize,
  itemCount: total.value,
  onChange: (p: number) => { page.value = p; load() },
  prefix: ({ itemCount }: { itemCount?: number }) => `共 ${itemCount ?? 0} 条`,
}))
</script>

<template>
  <div class="page-container">
    <n-card :bordered="false">
      <template #header>发药记录</template>
      <template #header-extra>
        <n-button size="small" quaternary @click="load">刷新</n-button>
      </template>
      <n-data-table
        :columns="columns" :data="rows" :loading="loading" :bordered="false"
        size="small" remote :row-key="(r: DispensingRecord) => r.id" :pagination="pagination"
      />
    </n-card>
  </div>
</template>
