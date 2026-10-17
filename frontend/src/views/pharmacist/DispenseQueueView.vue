<script setup lang="ts">
/**
 * 药师工作台：待审核（已缴费处方）→ 四查十对审核 → 发药。
 *
 * 状态流（后端强制）：confirmed →(收费员收款)→ paid →(药师审核)→ paid+verified
 * →(发药)→ dispensed。本页面就是 paid/dispensed 这一段的操作台。
 * 审核意见必填——审核是责任行为，后端同样强校验，留空点确认会被 422 打回。
 */
import { ref, computed, onMounted, h } from 'vue'
import type { VNode } from 'vue'
import {
  NCard, NTabs, NTabPane, NDataTable, NButton, NTag, NSpace, NModal, NInput,
  NAlert, NDescriptions, NDescriptionsItem, useMessage,
} from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import {
  listPrescriptionsAll, approvePrescription, dispensePrescription, listRecordsAll,
} from '@/api/dispensing'
import type { PrescriptionRow } from '@/api/dispensing'
import { ApiError } from '@/api/client'
import type { DispensingRecord, Prescription, SafetyCheck } from '@/types/clinical'
import { PRESCRIPTION_STATUS_LABELS, SAFETY_RULE_LABELS } from '@/types/enums'
import { formatFen, formatYuan } from '@/utils/currency'

const message = useMessage()

const pending = ref<PrescriptionRow[]>([])
const records = ref<DispensingRecord[]>([])
const loading = ref(false)
const tab = ref<'review' | 'dispense'>('review')

/** prescription_id → 审核记录，决定行按钮语义（先审后发是后端硬约束）。 */
const verifiedMap = computed(() => {
  const m = new Map<string, DispensingRecord>()
  for (const r of records.value) m.set(r.prescription_id, r)
  return m
})

async function load() {
  loading.value = true
  try {
    const [rxs, recs] = await Promise.all([
      listPrescriptionsAll({ status: 'paid', page: 1, page_size: 100 }),
      listRecordsAll({ page: 1, page_size: 100 }),
    ])
    pending.value = rxs.items
    records.value = recs.items
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '加载处方失败')
  } finally {
    loading.value = false
  }
}

onMounted(load)

// ==================== 审核 ====================
const reviewRx = ref<PrescriptionRow | null>(null)
const reviewNotes = ref('')
const reviewing = ref(false)
/** v-model 只能绑成员表达式，弹窗开关用可写 computed 包装。 */
const reviewVisible = computed({
  get: () => reviewRx.value !== null,
  set: (v: boolean) => { if (!v) reviewRx.value = null },
})

function openReview(rx: PrescriptionRow) {
  reviewRx.value = rx
  reviewNotes.value = ''
}

async function submitReview() {
  if (!reviewRx.value) return
  if (!reviewNotes.value.trim()) {
    message.warning('审核意见必填（四查十对结论请写清楚）')
    return
  }
  reviewing.value = true
  try {
    await approvePrescription(reviewRx.value.id, { notes: reviewNotes.value.trim() })
    message.success('审核通过，可发药')
    reviewRx.value = null
    await load()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '审核失败')
  } finally {
    reviewing.value = false
  }
}

// ==================== 发药 ====================
const dispenseRx = ref<PrescriptionRow | null>(null)
const dispenseNotes = ref('')
const dispensing = ref(false)
const dispenseVisible = computed({
  get: () => dispenseRx.value !== null,
  set: (v: boolean) => { if (!v) dispenseRx.value = null },
})

function openDispense(rx: PrescriptionRow) {
  dispenseRx.value = rx
  dispenseNotes.value = ''
}

async function submitDispense() {
  if (!dispenseRx.value) return
  dispensing.value = true
  try {
    await dispensePrescription(dispenseRx.value.id, { notes: dispenseNotes.value.trim() || null })
    message.success('已发药')
    dispenseRx.value = null
    await load()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '发药失败')
  } finally {
    dispensing.value = false
  }
}

function fmtTime(v: string | null): string {
  return (v || '').slice(0, 16).replace('T', ' ')
}

/** 处方明细摘要：药名×数量，超过 3 味折叠。 */
function itemsSummary(rx: Prescription): string {
  const names = rx.items.map((it) => `${it.drug_name}×${it.quantity}${it.unit}`)
  return names.length > 3 ? `${names.slice(0, 3).join('、')} 等${names.length}味` : names.join('、')
}

/** 医生开药时带理由压过的阻断项，在发药前最后一道人工关口提示药师重点核对。 */
function overriddenBlock(rx: Prescription): SafetyCheck[] {
  if (!rx.safety_override_reason) return []
  return rx.safety_checks.filter((c) => c.level === 'block')
}

function rowButtons(rx: PrescriptionRow) {
  const rec = verifiedMap.value.get(rx.id)
  const btns: VNode[] = []
  if (!rec?.verified) {
    btns.push(h(NButton, { size: 'small', type: 'primary', onClick: () => openReview(rx) },
      { default: () => '审核' }))
  } else {
    btns.push(h(NButton, { size: 'small', onClick: () => openDispense(rx) },
      { default: () => '发药' }))
  }
  return h(NSpace, { size: 6 }, { default: () => btns })
}

const baseColumns: DataTableColumns<PrescriptionRow> = [
  { title: '开方医生', key: 'doctor_name', width: 100 },
  {
    title: '诊断', key: 'diagnosis', width: 150, ellipsis: { tooltip: true },
    render: (r) => r.diagnosis || '—',
  },
  {
    title: '药品', key: 'items', ellipsis: { tooltip: true },
    render: (r) => itemsSummary(r),
  },
  {
    title: '金额', key: 'total_amount', width: 90, align: 'right', className: 'num',
    render: (r) => formatFen(r.total_amount),
  },
  {
    title: '安全校验', key: 'safety', width: 110,
    render: (r) => {
      const blocked = overriddenBlock(r)
      if (blocked.length) {
        return h(NTag, { size: 'small', bordered: false, type: 'error' },
          { default: () => `带理由开具 ×${blocked.length}` })
      }
      if (r.warning_count) {
        return h(NTag, { size: 'small', bordered: false, type: 'warning' },
          { default: () => `提醒 ${r.warning_count} 项` })
      }
      return h('span', { class: 'cell-muted' }, '通过')
    },
  },
  {
    title: '开方时间', key: 'created_at', width: 130, className: 'num',
    render: (r) => fmtTime(r.created_at),
  },
  { title: '操作', key: 'action', width: 110, render: (r) => rowButtons(r as PrescriptionRow) },
]

/** 待发放 Tab：只列已审核通过的处方。 */
const reviewColumns = baseColumns
const dispenseReady = computed(() =>
  pending.value.filter((rx) => verifiedMap.value.get(rx.id)?.verified),
)

const historyColumns: DataTableColumns<DispensingRecord> = [
  {
    title: '发药时间', key: 'dispensed_at', width: 150, className: 'num',
    render: (r) => fmtTime(r.dispensed_at),
  },
  { title: '处方金额', key: 'total_amount', width: 100, align: 'right', className: 'num',
    render: (r) => (r.total_amount != null ? formatFen(r.total_amount) : '—') },
  {
    title: '处方状态', key: 'prescription_status', width: 100,
    render: (r) => PRESCRIPTION_STATUS_LABELS[r.prescription_status] ?? r.prescription_status,
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

function checkedInDetail(rx: PrescriptionRow | null): SafetyCheck[] {
  return rx ? rx.safety_checks : []
}
</script>

<template>
  <div class="page-container">
    <n-card :bordered="false">
      <template #header>处方审核与发药</template>
      <template #header-extra>
        <n-button size="small" quaternary @click="load">刷新</n-button>
      </template>

      <n-tabs v-model:value="tab" type="line" size="small">
        <n-tab-pane name="review" :tab="`待审核（${pending.length - dispenseReady.length}）`">
          <n-alert type="info" :bordered="false" class="queue-hint">
            流程口径：处方须经收费员收款（已缴费）后才进入审核；审核与发药分两步，先审后发。
          </n-alert>
          <n-data-table
            :columns="reviewColumns"
            :data="pending.filter((rx) => !verifiedMap.get(rx.id)?.verified)"
            :loading="loading"
            :bordered="false"
            size="small"
            :row-key="(r: PrescriptionRow) => r.id"
            :pagination="false"
          />
        </n-tab-pane>

        <n-tab-pane name="dispense" :tab="`待发放（${dispenseReady.length}）`">
          <n-data-table
            :columns="baseColumns"
            :data="dispenseReady"
            :loading="loading"
            :bordered="false"
            size="small"
            :row-key="(r: PrescriptionRow) => r.id"
            :pagination="false"
          />
        </n-tab-pane>

        <n-tab-pane name="history" :tab="`发药记录（${records.length}）`">
          <n-data-table
            :columns="historyColumns"
            :data="records"
            :loading="loading"
            :bordered="false"
            size="small"
            :row-key="(r: DispensingRecord) => r.id"
            :pagination="records.length > 15 ? { pageSize: 15 } : false"
          />
        </n-tab-pane>
      </n-tabs>
    </n-card>

    <!-- 审核弹窗：明细 + 校验结果 + 意见必填 -->
    <n-modal v-model:show="reviewVisible" preset="card" title="处方审核（四查十对）" style="max-width: 640px">
      <template v-if="reviewRx">
        <n-descriptions :column="2" label-placement="left" size="small" bordered class="rx-meta">
          <n-descriptions-item label="开方医生">{{ reviewRx.doctor_name }}</n-descriptions-item>
          <n-descriptions-item label="诊断">{{ reviewRx.diagnosis || '—' }}</n-descriptions-item>
          <n-descriptions-item label="金额">{{ formatFen(reviewRx.total_amount) }}</n-descriptions-item>
          <n-descriptions-item label="患者档案">{{ reviewRx.patient_id.slice(0, 8) }}…</n-descriptions-item>
        </n-descriptions>

        <div class="rx-items">
          <div v-for="it in reviewRx.items" :key="it.drug_code" class="rx-item">
            <div class="rx-item__name">
              {{ it.drug_name }}
              <span v-if="it.specification" class="rx-item__spec">{{ it.specification }}</span>
            </div>
            <div class="rx-item__usage">
              {{ [it.dosage, it.frequency, it.route, it.duration_days ? `${it.duration_days}天` : null]
                .filter(Boolean).join(' · ') || '用法未注明' }}
              <span class="rx-item__qty">{{ it.quantity }}{{ it.unit }} × {{ formatYuan(Number(it.unit_price)) }}</span>
            </div>
          </div>
        </div>

        <n-alert
          v-for="(c, i) in checkedInDetail(reviewRx)"
          :key="i"
          :type="c.level === 'block' ? 'error' : 'warning'"
          :bordered="false"
          class="rx-check"
        >
          {{ SAFETY_RULE_LABELS[c.rule] ?? c.rule }}：{{ c.message }}
          <template v-if="c.level === 'block' && reviewRx.safety_override_reason">
            （医生已带理由开具：{{ reviewRx.safety_override_reason }}）
          </template>
        </n-alert>

        <n-input
          v-model:value="reviewNotes"
          type="textarea"
          :autosize="{ minRows: 2, maxRows: 4 }"
          placeholder="审核意见（必填），如：用法用量合理，无配伍禁忌，同意调配"
          maxlength="1000"
        />
        <n-space justify="end" class="modal-actions">
          <n-button @click="reviewRx = null">取消</n-button>
          <n-button type="primary" :loading="reviewing" @click="submitReview">审核通过</n-button>
        </n-space>
      </template>
    </n-modal>

    <!-- 发药弹窗 -->
    <n-modal v-model:show="dispenseVisible" preset="card" title="发药确认" style="max-width: 560px">
      <template v-if="dispenseRx">
        <p class="dispense-line">{{ dispenseRx.doctor_name }} 开具 · {{ itemsSummary(dispenseRx) }}</p>
        <p class="dispense-line">应收金额：<b>{{ formatFen(dispenseRx.total_amount) }}</b>（已收款）</p>
        <n-input
          v-model:value="dispenseNotes"
          type="textarea"
          :autosize="{ minRows: 2, maxRows: 4 }"
          placeholder="发药备注（可选）：用药交代、患者签收情况"
          maxlength="1000"
        />
        <n-space justify="end" class="modal-actions">
          <n-button @click="dispenseRx = null">取消</n-button>
          <n-button type="primary" :loading="dispensing" @click="submitDispense">确认发药</n-button>
        </n-space>
      </template>
    </n-modal>
  </div>
</template>

<style scoped>
.queue-hint { margin-bottom: var(--space-3); }
.cell-muted { color: var(--gray-5); font-size: var(--font-sm); }
.rx-meta { margin-bottom: var(--space-4); }
.rx-items { margin-bottom: var(--space-4); border: 1px solid var(--border-color-light); border-radius: var(--radius-md); }
.rx-item { padding: var(--space-3); border-bottom: 1px solid var(--border-color-light); }
.rx-item:last-child { border-bottom: none; }
.rx-item__name { font-weight: 600; color: var(--gray-9); font-size: var(--font-base); }
.rx-item__spec { margin-left: var(--space-2); color: var(--gray-6); font-weight: 400; font-size: var(--font-sm); }
.rx-item__usage { color: var(--gray-7); font-size: var(--font-sm); margin-top: 2px; display: flex; justify-content: space-between; }
.rx-item__qty { color: var(--gray-8); font-weight: 500; }
.rx-check { margin-bottom: var(--space-2); }
.dispense-line { margin: 0 0 var(--space-3); color: var(--gray-8); font-size: var(--font-base); }
.modal-actions { margin-top: var(--space-4); }
</style>
