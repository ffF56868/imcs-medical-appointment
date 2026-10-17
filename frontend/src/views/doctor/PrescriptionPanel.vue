<script setup lang="ts">
/**
 * 医生工作台·处方面板。草稿 → 确认开具（合理用药校验）→ 缴费后流转药师。
 *
 * 确认开具撞安全拦截时后端返回 422（detail.findings 列出命中规则），
 * 面板把风险项原样展示，医生复核后可填理由强行开具（override 落库，
 * 药师端可见）——这不是绕过安全，而是医师处方权的正规通道。
 *
 * 药品目录：IMCS 无药品库端点，前端内置常用药预设（演示口径），
 * 编码/单价对齐过敏与剂量规则能演示的效果。
 */
import { ref, computed, onMounted, h } from 'vue'
import {
  NButton, NTag, NSpace, NModal, NSelect, NInput, NInputNumber, NDataTable,
  NAlert, NPopconfirm, useMessage,
} from 'naive-ui'
import type { DataTableColumns, SelectOption } from 'naive-ui'
import {
  listVisitPrescriptions, createPrescription, updatePrescription,
  confirmPrescription, cancelPrescription, type PrescriptionItemIn,
} from '@/api/prescription'
import { ApiError } from '@/api/client'
import type { Prescription, PrescriptionItem, SafetyCheck } from '@/types/clinical'
import { PRESCRIPTION_STATUS_LABELS, SAFETY_RULE_LABELS, SAFETY_LEVEL_LABELS } from '@/types/enums'
import { formatFen } from '@/utils/currency'

const props = defineProps<{
  visitId: string
  diagnosis: string | null
  visitClosed: boolean
}>()

const message = useMessage()

const rows = ref<Prescription[]>([])
const loading = ref(false)

async function load() {
  loading.value = true
  try {
    rows.value = await listVisitPrescriptions(props.visitId)
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '加载处方失败')
  } finally {
    loading.value = false
  }
}
onMounted(load)
defineExpose({ load })

/* ==================== 药品预设 ==================== */
interface DrugPreset {
  code: string
  name: string
  spec: string
  unit: string
  price: number
  dosage: string
  frequency: string
  route: string
}
const DRUG_PRESETS: DrugPreset[] = [
  { code: 'D0001', name: '阿莫西林胶囊', spec: '0.25g*24粒', unit: '盒', price: 18.5, dosage: '0.5g', frequency: 'tid', route: '口服' },
  { code: 'D0002', name: '布洛芬缓释胶囊', spec: '0.3g*20粒', unit: '盒', price: 19.9, dosage: '0.3g', frequency: 'bid', route: '口服' },
  { code: 'D0003', name: '青霉素V钾片', spec: '0.25g*24片', unit: '盒', price: 12.0, dosage: '0.5g', frequency: 'q6h', route: '口服' },
  { code: 'D0004', name: '苯磺酸氨氯地平片', spec: '5mg*28片', unit: '盒', price: 32.8, dosage: '5mg', frequency: 'qd', route: '口服' },
  { code: 'D0005', name: '二甲双胍缓释片', spec: '0.5g*30片', unit: '盒', price: 21.6, dosage: '0.5g', frequency: 'bid', route: '口服' },
  { code: 'D0006', name: '阿司匹林肠溶片', spec: '100mg*30片', unit: '盒', price: 15.9, dosage: '100mg', frequency: 'qd', route: '口服' },
  { code: 'D0007', name: '阿托伐他汀钙片', spec: '20mg*7片', unit: '盒', price: 27.5, dosage: '20mg', frequency: 'qn', route: '口服' },
  { code: 'D0008', name: '复方磺胺甲噁唑片', spec: '0.4g*10片', unit: '盒', price: 8.9, dosage: '2片', frequency: 'bid', route: '口服' },
  { code: 'D0009', name: '硝酸甘油片', spec: '0.5mg*100片', unit: '瓶', price: 29.0, dosage: '0.5mg', frequency: '舌下含服', route: '舌下' },
  { code: 'D0010', name: '氯化钾缓释片', spec: '0.5g*24片', unit: '盒', price: 9.8, dosage: '1g', frequency: 'tid', route: '口服' },
]
const drugOptions: SelectOption[] = DRUG_PRESETS.map((d) => ({
  label: `${d.name}（${d.spec}）`,
  value: d.code,
}))
const FREQ_OPTIONS: SelectOption[] = ['qd', 'bid', 'tid', 'qid', 'qn'].map((f) => ({ label: f, value: f }))

/* ==================== 编辑弹窗 ==================== */
interface DraftItem {
  drug_code: string | null
  quantity: number
  dosage: string
  frequency: string
  route: string
  duration_days: number | null
}

const editing = ref<Prescription | null>(null)
const creating = ref(false)
const items = ref<DraftItem[]>([])
const notes = ref('')
const saving = ref(false)

function blankItem(): DraftItem {
  return { drug_code: null, quantity: 1, dosage: '', frequency: 'bid', route: '口服', duration_days: null }
}

function openCreate() {
  editing.value = null
  items.value = [blankItem()]
  notes.value = ''
  creating.value = true
}

function openEdit(rx: Prescription) {
  editing.value = rx
  items.value = rx.items.map((it) => ({
    drug_code: it.drug_code,
    quantity: Number(it.quantity),
    dosage: it.dosage ?? '',
    frequency: it.frequency ?? 'bid',
    route: it.route ?? '口服',
    duration_days: it.duration_days,
  }))
  notes.value = rx.notes ?? ''
  creating.value = true
}

function onPickDrug(row: DraftItem, code: string) {
  const preset = DRUG_PRESETS.find((d) => d.code === code)
  if (!preset) return
  row.dosage = preset.dosage
  row.frequency = preset.frequency
  row.route = preset.route
}

function toPayload(): { visit_id: string; diagnosis: string | null; notes: string | null; items: PrescriptionItemIn[] } | null {
  const list: PrescriptionItemIn[] = []
  for (const it of items.value) {
    const preset = DRUG_PRESETS.find((d) => d.code === it.drug_code)
    if (!preset) continue
    list.push({
      drug_code: preset.code,
      drug_name: preset.name,
      specification: preset.spec,
      dosage: it.dosage || null,
      frequency: it.frequency || null,
      route: it.route || null,
      duration_days: it.duration_days,
      quantity: it.quantity,
      unit: preset.unit,
      unit_price: preset.price,
    })
  }
  if (!list.length) return null
  return {
    visit_id: props.visitId,
    diagnosis: props.diagnosis,
    notes: notes.value || null,
    items: list,
  }
}

const totalPreview = computed(() => {
  let sum = 0
  for (const it of items.value) {
    const p = DRUG_PRESETS.find((d) => d.code === it.drug_code)
    if (p) sum += p.price * it.quantity * 100
  }
  return Math.round(sum)
})

async function saveDraft() {
  const payload = toPayload()
  if (!payload) {
    message.warning('至少选择一味药')
    return
  }
  saving.value = true
  try {
    if (editing.value) {
      await updatePrescription(editing.value.id, payload)
      message.success('草稿已更新')
    } else {
      const rx = await createPrescription(payload)
      message.success('草稿已创建')
      await tryConfirm(rx.id)
    }
    creating.value = false
    await load()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '保存失败')
  } finally {
    saving.value = false
  }
}

/* ==================== 确认开具 + 拦截 override ==================== */
const blockModal = ref(false)
const blockRxId = ref('')
const blockFindings = ref<SafetyCheck[]>([])
const overrideReason = ref('')
const confirming = ref(false)

async function tryConfirm(rxId: string, reason?: string) {
  confirming.value = true
  try {
    const resp = await confirmPrescription(rxId, reason)
    message.success(resp.overridden ? '已带理由开具（药师将看到你的理由）' : '处方已开具')
    blockModal.value = false
    await load()
  } catch (err) {
    if (err instanceof ApiError && (err.code === 'prescription_allergy_conflict' || err.code === 'prescription_drug_interaction')) {
      const findings = (err.detail as { findings?: SafetyCheck[] })?.findings ?? []
      blockRxId.value = rxId
      blockFindings.value = findings
      overrideReason.value = ''
      blockModal.value = true
    } else {
      message.error(err instanceof ApiError ? err.message : '确认开具失败')
    }
  } finally {
    confirming.value = false
  }
}

async function submitOverride() {
  if (!overrideReason.value.trim()) {
    message.warning('请填写复核理由（落库供药师审核与事后追责）')
    return
  }
  await tryConfirm(blockRxId.value, overrideReason.value.trim())
}

async function doCancel(rx: Prescription) {
  try {
    await cancelPrescription(rx.id, '医生作废')
    message.success('处方已作废')
    await load()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '作废失败')
  }
}

const statusTag = (s: string) => {
  const t = s === 'dispensed' ? 'success' : s === 'paid' ? 'info' : s === 'confirmed' ? 'primary'
    : s === 'cancelled' ? 'default' : 'warning'
  return h(NTag, { size: 'small', bordered: false, type: t }, { default: () => PRESCRIPTION_STATUS_LABELS[s] ?? s })
}

function itemsBrief(rx: Prescription) {
  const names = rx.items.map((it: PrescriptionItem) => it.drug_name)
  return names.length > 2 ? `${names.slice(0, 2).join('、')} 等${names.length}味` : names.join('、')
}

const columns: DataTableColumns<Prescription> = [
  { title: '药品', key: 'items', ellipsis: { tooltip: true }, render: (r) => itemsBrief(r) },
  { title: '金额', key: 'total_amount', width: 90, align: 'right', className: 'num', render: (r) => formatFen(r.total_amount) },
  { title: '状态', key: 'status', width: 84, render: (r) => statusTag(r.status) },
  {
    title: '操作', key: 'a', width: 150,
    render: (r) => {
      if (r.status === 'draft') {
        return h(NSpace, { size: 6 }, { default: () => [
          h(NButton, { size: 'tiny', onClick: () => openEdit(r) }, { default: () => '编辑' }),
          h(NButton, { size: 'tiny', type: 'primary', loading: confirming.value, onClick: () => tryConfirm(r.id) }, { default: () => '开具' }),
          h(NPopconfirm, { onPositiveClick: () => doCancel(r) }, {
            trigger: () => h(NButton, { size: 'tiny', quaternary: true }, { default: () => '废' }),
            default: () => '作废这张草稿处方？',
          }),
        ] })
      }
      return h('span', { class: 'cell-muted' }, r.status === 'confirmed' ? '待缴费' : '—')
    },
  },
]

/** 编辑弹窗里的药品明细列 */
const editColumns: DataTableColumns<DraftItem> = [
  {
    title: '药品', key: 'drug_code', width: 210,
    render: (row) => h(NSelect, {
      value: row.drug_code, options: drugOptions, size: 'small', placeholder: '选择药品',
      'onUpdate:value': (v: string) => onPickDrug(row, v),
    }),
  },
  {
    title: '单次剂量', key: 'dosage', width: 92,
    render: (row) => h(NInput, {
      value: row.dosage, size: 'small', placeholder: '0.5g',
      'onUpdate:value': (v: string) => { row.dosage = v },
    }),
  },
  {
    title: '频次', key: 'frequency', width: 88,
    render: (row) => h(NSelect, {
      value: row.frequency, options: FREQ_OPTIONS, size: 'small',
      'onUpdate:value': (v: string) => { row.frequency = v },
    }),
  },
  {
    title: '数量', key: 'quantity', width: 84,
    render: (row) => h(NInputNumber, {
      value: row.quantity, min: 1, max: 10000, size: 'small', showButton: false,
      'onUpdate:value': (v: number | null) => { row.quantity = v ?? 1 },
    }),
  },
  {
    title: '', key: 'del', width: 36,
    render: (row) => items.value.length > 1
      ? h(NButton, { size: 'tiny', quaternary: true, onClick: () => { items.value = items.value.filter((x) => x !== row) } }, { default: () => '×' })
      : null,
  },
]
</script>

<template>
  <div class="rx-panel">
    <div class="rx-panel__head">
      <span class="rx-panel__title">处方</span>
      <n-button size="tiny" type="primary" :disabled="visitClosed" @click="openCreate">新建处方</n-button>
    </div>
    <n-data-table
      :columns="columns" :data="rows" :loading="loading" :bordered="false"
      size="small" :row-key="(r: Prescription) => r.id"
    />
    <div v-if="!loading && !rows.length" class="cell-muted">尚未开具处方</div>

    <!-- 明细编辑 -->
    <n-modal v-model:show="creating" preset="card" :title="editing ? '编辑处方草稿' : '新建处方'" style="width: 640px">
      <n-data-table :columns="editColumns" :data="items" :bordered="false" size="small" />
      <n-button size="tiny" dashed class="rx-add" @click="items.push(blankItem())">+ 添加药品</n-button>
      <n-input v-model:value="notes" type="textarea" :autosize="{ minRows: 1, maxRows: 3 }"
        placeholder="医嘱备注（可选）" class="rx-notes" maxlength="2000" />
      <div class="rx-total">合计预估：<b>{{ formatFen(totalPreview) }}</b>
        <span class="rx-total__hint">实际以服务端计价为准；开具时自动执行合理用药校验</span>
      </div>
      <n-space justify="end">
        <n-button @click="creating = false">取消</n-button>
        <n-button type="primary" :loading="saving" @click="saveDraft">{{ editing ? '保存草稿' : '开具处方' }}</n-button>
      </n-space>
    </n-modal>

    <!-- 安全拦截 -->
    <n-modal v-model:show="blockModal" preset="card" title="用药安全拦截" style="max-width: 560px">
      <n-alert type="error" :bordered="false" class="blk-hint">
        以下风险项阻止了本次开具。请逐条核对患者情况——如临床上已复核风险（过敏史有误、脱敏治疗等），
        可填写理由强行开具；理由将落库，药师审核与事后追责都能看到。
      </n-alert>
      <div v-for="(f, i) in blockFindings" :key="i" class="blk-item">
        <n-tag size="tiny" :bordered="false" :type="f.level === 'block' ? 'error' : 'warning'">
          {{ SAFETY_LEVEL_LABELS[f.level] }} · {{ SAFETY_RULE_LABELS[f.rule] ?? f.rule }}
        </n-tag>
        <span>{{ f.message }}</span>
      </div>
      <n-input v-model:value="overrideReason" type="textarea" :autosize="{ minRows: 2, maxRows: 4 }"
        placeholder="复核理由（强行开具时必填），如：过敏史为家属口误，皮试阴性" maxlength="500" class="blk-reason" />
      <n-space justify="end">
        <n-button @click="blockModal = false">返回修改</n-button>
        <n-button type="error" :loading="confirming" @click="submitOverride">带理由开具</n-button>
      </n-space>
    </n-modal>
  </div>
</template>

<style scoped>
.rx-panel { margin-top: var(--space-4); }
.rx-panel__head { display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--space-2); }
.rx-panel__title { font-size: var(--font-md); font-weight: 600; color: var(--gray-9); }
.cell-muted { color: var(--gray-5); font-size: var(--font-sm); padding: var(--space-2) 0; }
.rx-add { margin-top: var(--space-2); }
.rx-notes { margin-top: var(--space-3); }
.rx-total { margin: var(--space-3) 0; font-size: var(--font-base); color: var(--gray-8); }
.rx-total__hint { display: block; font-size: var(--font-sm); color: var(--gray-5); margin-top: 2px; }
.blk-hint { margin-bottom: var(--space-3); }
.blk-item { display: flex; align-items: center; gap: var(--space-2); padding: var(--space-2) 0; border-bottom: 1px dashed var(--border-color-light); font-size: var(--font-sm); color: var(--gray-8); }
.blk-reason { margin-top: var(--space-3); }
</style>
