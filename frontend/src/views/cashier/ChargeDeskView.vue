<script setup lang="ts">
/**
 * 收费台。收费员看全院收费单，处理收款 / 退费 / 作废。
 *
 * 状态机（后端强制）：pending →(收款)→ paid →(退费)→ refunded；
 * pending →(作废)→ cancelled。含处方项的单一旦支付不可退（处方状态机
 * 没有回退边），409 的文案后端已给出「先作废处方走人工」指引，直接展示。
 */
import { ref, computed, onMounted, h } from 'vue'
import type { VNode } from 'vue'
import {
  NCard, NDataTable, NButton, NTag, NSpace, NSelect, NInput, NModal,
  NFormItem, NPopconfirm, NDescriptions, NDescriptionsItem, useMessage,
} from 'naive-ui'
import type { DataTableColumns, SelectOption } from 'naive-ui'
import {
  listAllOrders, payOrder, refundOrder, cancelOrder,
  type ChargeOrderRow,
} from '@/api/billing'
import { ApiError } from '@/api/client'
import { CHARGE_ORDER_STATUS_LABELS, PAYMENT_CHANNEL_LABELS } from '@/types/enums'
import { formatFen } from '@/utils/currency'

const message = useMessage()

const loading = ref(false)
const rows = ref<ChargeOrderRow[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = 20
const statusFilter = ref<string | null>(null)

const statusOptions: SelectOption[] = [
  { label: '全部', value: '' },
  { label: '待缴费', value: 'pending' },
  { label: '已缴费', value: 'paid' },
  { label: '已退费', value: 'refunded' },
  { label: '已作废', value: 'cancelled' },
]

async function load() {
  loading.value = true
  try {
    const resp = await listAllOrders({
      status: statusFilter.value || undefined,
      page: page.value,
      page_size: pageSize,
    })
    rows.value = resp.items
    total.value = resp.total
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '加载收费单失败')
  } finally {
    loading.value = false
  }
}

onMounted(load)

/* ==================== 收款 ==================== */
const payTarget = ref<ChargeOrderRow | null>(null)
const channel = ref('mock')
const paying = ref(false)
/** v-model 只能绑成员表达式，弹窗开关用可写 computed 包装。 */
const payVisible = computed({
  get: () => payTarget.value !== null,
  set: (v: boolean) => { if (!v) payTarget.value = null },
})
const refundVisible = computed({
  get: () => refundTarget.value !== null,
  set: (v: boolean) => { if (!v) refundTarget.value = null },
})
const CHANNEL_OPTIONS: SelectOption[] = [
  { label: '模拟支付', value: 'mock' },
  { label: '现金', value: 'cash' },
  { label: '医保', value: 'insurance' },
]

async function submitPay() {
  if (!payTarget.value) return
  paying.value = true
  try {
    const order = await payOrder(payTarget.value.id, { channel: channel.value })
    message.success(`收款成功，小票号 ${order.receipt_no}`)
    payTarget.value = null
    await load()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '收款失败')
  } finally {
    paying.value = false
  }
}

/* ==================== 退费 ==================== */
const refundTarget = ref<ChargeOrderRow | null>(null)
const refundReason = ref('')
const refunding = ref(false)

async function submitRefund() {
  if (!refundTarget.value) return
  if (!refundReason.value.trim()) {
    message.warning('退费原因必填（资金流出要留痕）')
    return
  }
  refunding.value = true
  try {
    await refundOrder(refundTarget.value.id, { reason: refundReason.value.trim() })
    message.success('已退费')
    refundTarget.value = null
    await load()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '退费失败')
  } finally {
    refunding.value = false
  }
}

async function doCancel(row: ChargeOrderRow) {
  try {
    await cancelOrder(row.id)
    message.success('收费单已作废')
    await load()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '作废失败')
  }
}

function fmtTime(v: string | null): string {
  return (v || '').slice(0, 16).replace('T', ' ')
}

/** 明细人话版：处方/检查项显示类型，自定义项显示名称。 */
function itemsBrief(row: ChargeOrderRow): string {
  return row.items
    .map((it) => {
      if (it.type === 'prescription') return '处方药费'
      if (it.type === 'exam') return '检查费'
      return it.name || '其他'
    })
    .join(' + ')
}

const columns: DataTableColumns<ChargeOrderRow> = [
  {
    title: '单号', key: 'id', width: 100, className: 'num',
    render: (r) => h('a', { onClick: () => { detail.value = r } }, r.id.slice(0, 8)),
  },
  { title: '费用项', key: 'items', ellipsis: { tooltip: true }, render: (r) => itemsBrief(r) },
  {
    title: '应收', key: 'total_amount', width: 100, align: 'right', className: 'num',
    render: (r) => formatFen(r.total_amount),
  },
  {
    title: '实收', key: 'paid_amount', width: 100, align: 'right', className: 'num',
    render: (r) => (r.paid_amount ? formatFen(r.paid_amount) : '—'),
  },
  {
    title: '渠道', key: 'payment_channel', width: 90,
    render: (r) => (r.payment_channel ? PAYMENT_CHANNEL_LABELS[r.payment_channel] ?? r.payment_channel : '—'),
  },
  {
    title: '状态', key: 'status', width: 90,
    render: (r) => {
      const t = r.status === 'paid' ? 'success' : r.status === 'pending' ? 'warning'
        : r.status === 'refunded' ? 'info' : 'default'
      return h(NTag, { size: 'small', bordered: false, type: t },
        { default: () => CHARGE_ORDER_STATUS_LABELS[r.status] ?? r.status })
    },
  },
  {
    title: '创建时间', key: 'created_at', width: 130, className: 'num',
    render: (r) => fmtTime(r.created_at),
  },
  {
    title: '操作', key: 'action', width: 150,
    render: (r) => {
      const btns: VNode[] = []
      const trans = r.allowed_transitions ?? []
      if (trans.includes('paid')) {
        btns.push(h(NButton, { size: 'small', type: 'primary', onClick: () => { payTarget.value = r; channel.value = 'mock' } },
          { default: () => '收款' }))
      }
      if (trans.includes('refunded')) {
        btns.push(h(NButton, { size: 'small', onClick: () => { refundTarget.value = r; refundReason.value = '' } },
          { default: () => '退费' }))
      }
      if (trans.includes('cancelled')) {
        btns.push(h(NPopconfirm, { onPositiveClick: () => doCancel(r) }, {
          trigger: () => h(NButton, { size: 'small', quaternary: true, type: 'error' }, { default: () => '作废' }),
          default: () => '作废后患者需重新建单，确认作废这张待缴费单？',
        }))
      }
      if (!btns.length) return h('span', { class: 'cell-muted' }, '—')
      return h(NSpace, { size: 6 }, { default: () => btns })
    },
  },
]

const detail = ref<ChargeOrderRow | null>(null)
const detailVisible = computed({
  get: () => detail.value !== null,
  set: (v: boolean) => { if (!v) detail.value = null },
})

const pagination = computed(() => ({
  page: page.value,
  pageSize,
  itemCount: total.value,
  onChange: (p: number) => { page.value = p; load() },
  prefix: ({ itemCount }: { itemCount?: number }) => `共 ${itemCount ?? 0} 单`,
}))
</script>

<template>
  <div class="page-container">
    <n-card :bordered="false">
      <template #header>收费台</template>
      <template #header-extra>
        <n-space align="center" :size="12">
          <n-select
            v-model:value="statusFilter"
            :options="statusOptions"
            size="small"
            placeholder="全部"
            style="width: 130px"
            @update:value="() => { page = 1; load() }"
          />
          <n-button size="small" quaternary @click="load">刷新</n-button>
        </n-space>
      </template>

      <n-data-table
        :columns="columns"
        :data="rows"
        :loading="loading"
        :bordered="false"
        size="small"
        remote
        :row-key="(r: ChargeOrderRow) => r.id"
        :pagination="pagination"
      />
    </n-card>

    <!-- 收款 -->
    <n-modal v-model:show="payVisible" preset="card" title="确认收款" style="max-width: 480px">
      <template v-if="payTarget">
        <n-descriptions :column="1" label-placement="left" size="small" bordered>
          <n-descriptions-item label="费用项">{{ itemsBrief(payTarget) }}</n-descriptions-item>
          <n-descriptions-item label="应收金额">
            <b class="pay-amount">{{ formatFen(payTarget.total_amount) }}</b>
          </n-descriptions-item>
        </n-descriptions>
        <p class="pay-note">整单同生同灭：处方项将同步推进为已缴费并进入待发药队列。</p>
        <n-form-item label="支付渠道" label-placement="top">
          <n-select v-model:value="channel" :options="CHANNEL_OPTIONS" />
        </n-form-item>
        <n-space justify="end">
          <n-button @click="payTarget = null">取消</n-button>
          <n-button type="primary" :loading="paying" @click="submitPay">确认收款</n-button>
        </n-space>
      </template>
    </n-modal>

    <!-- 退费 -->
    <n-modal v-model:show="refundVisible" preset="card" title="退费" style="max-width: 480px">
      <template v-if="refundTarget">
        <p class="pay-note">单号 {{ refundTarget.id.slice(0, 8) }} · {{ formatFen(refundTarget.paid_amount) }}</p>
        <n-input
          v-model:value="refundReason"
          type="textarea"
          :autosize="{ minRows: 2, maxRows: 4 }"
          placeholder="退费原因（必填）"
          maxlength="500"
        />
        <n-space justify="end" class="modal-gap">
          <n-button @click="refundTarget = null">取消</n-button>
          <n-button type="primary" :loading="refunding" @click="submitRefund">确认退费</n-button>
        </n-space>
      </template>
    </n-modal>

    <!-- 明细 -->
    <n-modal v-model:show="detailVisible" preset="card" title="收费单明细" style="max-width: 560px">
      <template v-if="detail">
        <div v-for="(it, i) in detail.items" :key="i" class="charge-item">
          <span>{{ it.type === 'prescription' ? '处方药费' : it.type === 'exam' ? '检查费' : it.name || '其他' }}</span>
          <span class="charge-item__amt num">{{ formatFen(it.amount) }}</span>
        </div>
        <div class="charge-item charge-item--total">
          <span>合计</span>
          <span class="num">{{ formatFen(detail.total_amount) }}</span>
        </div>
        <n-descriptions :column="2" label-placement="left" size="small" bordered class="modal-gap">
          <n-descriptions-item label="状态">{{ CHARGE_ORDER_STATUS_LABELS[detail.status] }}</n-descriptions-item>
          <n-descriptions-item label="小票号">{{ detail.receipt_no || '—' }}</n-descriptions-item>
          <n-descriptions-item label="支付时间">{{ fmtTime(detail.paid_at) || '—' }}</n-descriptions-item>
          <n-descriptions-item label="退费原因">{{ detail.refund_reason || '—' }}</n-descriptions-item>
        </n-descriptions>
      </template>
    </n-modal>
  </div>
</template>

<style scoped>
.cell-muted { color: var(--gray-5); }
.pay-amount { font-size: var(--font-lg); color: var(--brand); }
.pay-note { color: var(--gray-6); font-size: var(--font-sm); margin: var(--space-3) 0; }
.charge-item { display: flex; justify-content: space-between; padding: var(--space-2) 0; border-bottom: 1px solid var(--border-color-light); }
.charge-item--total { font-weight: 600; border-bottom: none; }
.charge-item__amt { color: var(--gray-8); }
.modal-gap { margin-top: var(--space-4); }
</style>
