<script setup lang="ts">
/**
 * 我的处方。患者只读自己的处方，含安全校验结果与发药状态。
 * safety_checks 里的 block/warning 对患者展示为「用药提醒」，不暴露 override 细节。
 */
import { ref, onMounted, h } from 'vue'
import {
  NCard, NDataTable, NTag, NSpin, NEmpty, useMessage, NModal,
  NDescriptions, NDescriptionsItem, NAlert,
} from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import { listMyPrescriptions } from '@/api/prescription'
import { ApiError } from '@/api/client'
import type { Prescription, PrescriptionItem } from '@/types/clinical'
import { PRESCRIPTION_STATUS_LABELS } from '@/types/enums'
import { formatFen, formatYuan } from '@/utils/currency'

const message = useMessage()
const loading = ref(false)
const rows = ref<Prescription[]>([])
const detail = ref<Prescription | null>(null)

async function load() {
  loading.value = true
  try {
    const resp = await listMyPrescriptions({ page: 1, page_size: 50 })
    rows.value = resp.items
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '加载处方失败')
  } finally {
    loading.value = false
  }
}

function statusType(s: string): 'success' | 'info' | 'warning' | 'default' {
  switch (s) {
    case 'dispensed': return 'success'
    case 'paid':
    case 'confirmed': return 'info'
    case 'cancelled': return 'default'
    default: return 'warning'
  }
}

const columns: DataTableColumns<Prescription> = [
  { title: '开方时间', key: 'created_at', className: 'num', render: (r) => (r.created_at || '').slice(0, 16).replace('T', ' ') },
  { title: '医生', key: 'doctor_name', render: (r) => r.doctor_name || '—' },
  { title: '诊断', key: 'diagnosis', ellipsis: { tooltip: true }, render: (r) => r.diagnosis || '—' },
  { title: '药品数', key: 'item_count', align: 'right', className: 'cell-number', render: (r) => r.items?.length ?? 0 },
  { title: '金额', key: 'total_amount', align: 'right', className: 'cell-number', render: (r) => formatFen(r.total_amount) },
  {
    title: '状态', key: 'status',
    render: (r) => h(NTag, { size: 'small', bordered: false, type: statusType(r.status) }, { default: () => PRESCRIPTION_STATUS_LABELS[r.status] }),
  },
  {
    title: '操作', key: 'action',
    render: (r) => h('a', { onClick: () => { detail.value = r } }, '查看'),
  },
]

const itemColumns: DataTableColumns<PrescriptionItem> = [
  { title: '药品', key: 'drug_name' },
  { title: '规格', key: 'specification', render: (r) => r.specification || '—' },
  { title: '用法', key: 'usage', render: (r) => [r.dosage, r.frequency].filter(Boolean).join(' ') || '—' },
  { title: '数量', key: 'quantity', align: 'right', className: 'cell-number', render: (r) => `${r.quantity}${r.unit || ''}` },
  { title: '单价', key: 'unit_price', align: 'right', className: 'cell-number', render: (r) => formatYuan(r.unit_price) },
]

onMounted(load)
</script>

<template>
  <div class="page-container">
    <n-card title="我的处方" :bordered="false">
      <n-spin :show="loading">
        <n-data-table
          :columns="columns"
          :data="rows"
          :bordered="false"
          size="small"
          :row-key="(r: Prescription) => r.id"
          :pagination="{ pageSize: 10 }"
        />
        <n-empty v-if="!loading && !rows.length" description="暂无处方记录" class="empty-block" />
      </n-spin>
    </n-card>

    <n-modal
      :show="detail !== null"
      preset="card"
      title="处方详情"
      style="max-width: 760px"
      @update:show="(v: boolean) => { if (!v) detail = null }"
    >
      <template v-if="detail">
        <n-descriptions :column="2" label-placement="left" bordered size="small">
          <n-descriptions-item label="诊断">{{ detail.diagnosis || '—' }}</n-descriptions-item>
          <n-descriptions-item label="医生">{{ detail.doctor_name || '—' }}</n-descriptions-item>
          <n-descriptions-item label="状态">
            <n-tag size="small" bordered :type="statusType(detail.status)">{{ PRESCRIPTION_STATUS_LABELS[detail.status] }}</n-tag>
          </n-descriptions-item>
          <n-descriptions-item label="合计">{{ formatFen(detail.total_amount) }}</n-descriptions-item>
          <n-descriptions-item v-if="detail.notes" label="备注" :span="2">{{ detail.notes }}</n-descriptions-item>
        </n-descriptions>

        <n-alert
          v-for="(sc, i) in detail.safety_checks"
          :key="i"
          :type="sc.level === 'block' ? 'error' : 'warning'"
          :bordered="false"
          class="block-gap"
        >
          {{ sc.message }}
        </n-alert>

        <div class="section-title block-gap">药品明细</div>
        <n-data-table
          :columns="itemColumns"
          :data="detail.items"
          :bordered="false"
          size="small"
          :row-key="(r: PrescriptionItem) => r.drug_code"
        />
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
}
</style>
