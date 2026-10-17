<script setup lang="ts">
/**
 * 我的缴费。患者只读自己的收费单，含缴费状态与收据号。
 * 支付动作在收费员端完成，患者端只展示结果（本系统不做患者线上支付）。
 */
import { ref, onMounted, h } from 'vue'
import {
  NCard, NDataTable, NTag, NSpin, NEmpty, useMessage, NModal, NList, NListItem,
} from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import { listMyOrders } from '@/api/billing'
import { ApiError } from '@/api/client'
import type { ChargeOrder } from '@/types/clinical'
import { CHARGE_ORDER_STATUS_LABELS, PAYMENT_CHANNEL_LABELS } from '@/types/enums'
import { formatFen } from '@/utils/currency'

const message = useMessage()
const loading = ref(false)
const rows = ref<ChargeOrder[]>([])
const detail = ref<ChargeOrder | null>(null)

async function load() {
  loading.value = true
  try {
    const resp = await listMyOrders({ page: 1, page_size: 50 })
    rows.value = resp.items
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '加载缴费记录失败')
  } finally {
    loading.value = false
  }
}

function statusType(s: string): 'success' | 'warning' | 'default' | 'error' {
  switch (s) {
    case 'paid': return 'success'
    case 'pending': return 'warning'
    case 'refunded': return 'error'
    default: return 'default'
  }
}

const columns: DataTableColumns<ChargeOrder> = [
  { title: '开单时间', key: 'created_at', className: 'num', render: (r) => (r.created_at || '').slice(0, 16).replace('T', ' ') },
  { title: '项目数', key: 'count', align: 'right', className: 'cell-number', render: (r) => r.items?.length ?? 0 },
  { title: '应收', key: 'total_amount', align: 'right', className: 'cell-number', render: (r) => formatFen(r.total_amount) },
  { title: '实收', key: 'paid_amount', align: 'right', className: 'cell-number', render: (r) => formatFen(r.paid_amount) },
  { title: '收据号', key: 'receipt_no', className: 'num', render: (r) => r.receipt_no || '—' },
  {
    title: '状态', key: 'status',
    render: (r) => h(NTag, { size: 'small', bordered: false, type: statusType(r.status) }, { default: () => CHARGE_ORDER_STATUS_LABELS[r.status] }),
  },
  {
    title: '操作', key: 'action',
    render: (r) => h('a', { onClick: () => { detail.value = r } }, '明细'),
  },
]

onMounted(load)
</script>

<template>
  <div class="page-container">
    <n-card title="我的缴费" :bordered="false">
      <n-spin :show="loading">
        <n-data-table
          :columns="columns"
          :data="rows"
          :bordered="false"
          size="small"
          :row-key="(r: ChargeOrder) => r.id"
          :pagination="{ pageSize: 10 }"
        />
        <n-empty v-if="!loading && !rows.length" description="暂无缴费记录" class="empty-block" />
      </n-spin>
    </n-card>

    <n-modal
      :show="detail !== null"
      preset="card"
      title="缴费明细"
      style="max-width: 600px"
      @update:show="(v: boolean) => { if (!v) detail = null }"
    >
      <template v-if="detail">
        <n-list bordered>
          <n-list-item v-for="(it, i) in detail.items" :key="i">
            <div class="item-row">
              <span>{{ it.name || (it.type === 'prescription' ? '处方药费' : it.type === 'exam' ? '检查费' : '其他') }}</span>
              <span class="num">{{ formatFen(it.amount) }}</span>
            </div>
          </n-list-item>
          <n-list-item>
            <div class="item-row total">
              <span>合计</span>
              <span class="num">{{ formatFen(detail.total_amount) }}</span>
            </div>
          </n-list-item>
        </n-list>
        <div v-if="detail.status === 'paid'" class="pay-info">
          支付渠道：{{ PAYMENT_CHANNEL_LABELS[detail.payment_channel || ''] || detail.payment_channel || '—' }}
          <span v-if="detail.paid_at"> · 支付时间：{{ detail.paid_at.slice(0, 16).replace('T', ' ') }}</span>
        </div>
        <div v-if="detail.status === 'refunded'" class="pay-info refund">
          已退费<span v-if="detail.refund_reason"> · 原因：{{ detail.refund_reason }}</span>
        </div>
      </template>
    </n-modal>
  </div>
</template>

<style scoped>
.empty-block { margin-top: var(--space-8); }
.item-row {
  display: flex;
  justify-content: space-between;
  width: 100%;
}
.item-row.total {
  font-weight: 600;
  color: var(--gray-9);
}
.pay-info {
  margin-top: var(--space-3);
  font-size: var(--font-sm);
  color: var(--gray-7);
}
.pay-info.refund { color: var(--danger); }
</style>
