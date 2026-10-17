<script setup lang="ts">
/**
 * 检验科工作台。上 Tab「待接收」是医生刚开出的 ordered 新单（点击接收进在办队列），
 * 下 Tab「在办」是已接收/检查中的单子（开始执行 → 录入报告）。
 * 报告录入后服务端自动派发 AI 解读，本页不等 LLM；危急值在录入瞬间由规则算出并弹窗。
 */
import { ref, computed, onMounted, h } from 'vue'
import {
  NCard, NTabs, NTabPane, NDataTable, NButton, NTag, NSpace, NModal, NInput,
  NFormItem, NAlert, NSwitch, useMessage, useDialog,
} from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import {
  labIncoming, labQueue, receiveRequest, startProcessing, submitReport,
} from '@/api/exam'
import { ApiError } from '@/api/client'
import type { ExamRequest } from '@/types/clinical'
import { EXAM_REQUEST_STATUS_LABELS, URGENCY_LABELS } from '@/types/enums'
import { formatFen } from '@/utils/currency'

const message = useMessage()
const dialog = useDialog()

const tab = ref<'incoming' | 'working'>('incoming')
const loading = ref(false)
const incoming = ref<ExamRequest[]>([])
const working = ref<ExamRequest[]>([])

async function load() {
  loading.value = true
  try {
    const [inc, work] = await Promise.all([
      labIncoming({ page: 1, page_size: 100 }),
      labQueue({ page: 1, page_size: 100 }),
    ])
    incoming.value = inc.items
    working.value = work.items
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '加载待办失败')
  } finally {
    loading.value = false
  }
}
onMounted(load)

async function doReceive(row: ExamRequest) {
  try {
    await receiveRequest(row.id)
    message.success('已接收，进入在办队列')
    await load()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '接收失败')
  }
}

async function doProcess(row: ExamRequest) {
  try {
    await startProcessing(row.id)
    message.success('已开始执行')
    await load()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '开始执行失败')
  }
}

/* ==================== 录入报告 ==================== */
const reportTarget = ref<ExamRequest | null>(null)
/** v-model 只能是成员表达式，布尔开关用可写 computed 包一层。 */
const reportVisible = computed({
  get: () => reportTarget.value !== null,
  set: (v: boolean) => { if (!v) reportTarget.value = null },
})
const rawContent = ref('')
const structuredText = ref('')
const conclusion = ref('')
const autoInterpret = ref(true)
const submitting = ref(false)

function openReport(row: ExamRequest) {
  reportTarget.value = row
  rawContent.value = ''
  structuredText.value = ''
  conclusion.value = ''
  autoInterpret.value = true
}

/** 结构化指标录入：契约 {"items":[{name,code,value,unit,ref_low,ref_high}]}。 */
function parseStructured(): Record<string, unknown> | null {
  const t = structuredText.value.trim()
  if (!t) return null
  return JSON.parse(t)
}

async function submitReportNow() {
  if (!reportTarget.value) return
  if (!rawContent.value.trim()) {
    message.warning('请填写报告原始内容')
    return
  }
  let structured: Record<string, unknown> | null = null
  try {
    structured = parseStructured()
  } catch {
    message.error('结构化指标 JSON 格式有误，请检查')
    return
  }
  submitting.value = true
  try {
    const resp = await submitReport(reportTarget.value.id, {
      raw_content: rawContent.value,
      structured_data: structured,
      conclusion: conclusion.value || null,
      auto_interpret: autoInterpret.value,
    })
    if (resp.report.critical) {
      dialog.error({
        title: '危急值告警',
        content: resp.report.critical_message || '出现危急值，请立即联系临床',
        positiveText: '已知悉',
      })
    } else {
      message.success('报告已录入' + (resp.dispatch.dispatched ? '，AI 解读排队中' : ''))
    }
    reportTarget.value = null
    await load()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '录入失败')
  } finally {
    submitting.value = false
  }
}

function urgencyTag(u: string) {
  const t = u === 'stat' ? 'error' : u === 'urgent' ? 'warning' : 'default'
  return h(NTag, { size: 'small', bordered: false, type: t }, { default: () => URGENCY_LABELS[u] ?? u })
}

const cols: DataTableColumns<ExamRequest> = [
  { title: '项目', key: 'exam_name', ellipsis: { tooltip: true } },
  { title: '部位', key: 'body_part', width: 90, render: (r) => r.body_part || '—' },
  { title: '紧急度', key: 'urgency', width: 80, render: (r) => urgencyTag(r.urgency) },
  {
    title: '费用', key: 'fee_amount', width: 90, align: 'right', className: 'num',
    render: (r) => (r.fee_amount != null ? formatFen(r.fee_amount) : '—'),
  },
  {
    title: '状态', key: 'status', width: 90,
    render: (r) => EXAM_REQUEST_STATUS_LABELS[r.status] ?? r.status,
  },
]

const incomingCols: DataTableColumns<ExamRequest> = [
  ...cols,
  {
    title: '操作', key: 'a', width: 90,
    render: (r) => h(NButton, { size: 'small', type: 'primary', onClick: () => doReceive(r) }, { default: () => '接收' }),
  },
]

const workingCols: DataTableColumns<ExamRequest> = [
  ...cols,
  {
    title: '操作', key: 'a', width: 170,
    render: (r) => {
      if (r.status === 'received') {
        return h(NSpace, { size: 6 }, { default: () => [
          h(NButton, { size: 'small', onClick: () => doProcess(r) }, { default: () => '开始执行' }),
        ] })
      }
      if (r.status === 'processing') {
        return h(NButton, { size: 'small', type: 'primary', onClick: () => openReport(r) }, { default: () => '录入报告' })
      }
      return h('span', { class: 'cell-muted' }, '—')
    },
  },
]
</script>

<template>
  <div class="page-container">
    <n-card :bordered="false">
      <template #header>检验工作台</template>
      <template #header-extra>
        <n-button size="small" quaternary @click="load">刷新</n-button>
      </template>
      <n-tabs v-model:value="tab" type="line" size="small">
        <n-tab-pane name="incoming" :tab="`待接收（${incoming.length}）`">
          <n-alert type="info" :bordered="false" class="hint">
            医生新开出的检查单，接收后进入在办队列。危急单自动插队排在最前。
          </n-alert>
          <n-data-table
            :columns="incomingCols" :data="incoming" :loading="loading" :bordered="false"
            size="small" :row-key="(r: ExamRequest) => r.id" :pagination="incoming.length > 15 ? { pageSize: 15 } : false"
          />
        </n-tab-pane>
        <n-tab-pane name="working" :tab="`在办（${working.length}）`">
          <n-data-table
            :columns="workingCols" :data="working" :loading="loading" :bordered="false"
            size="small" :row-key="(r: ExamRequest) => r.id" :pagination="working.length > 15 ? { pageSize: 15 } : false"
          />
        </n-tab-pane>
      </n-tabs>
    </n-card>

    <n-modal v-model:show="reportVisible" preset="card" title="录入检查报告" style="max-width: 640px">
      <template v-if="reportTarget">
        <p class="target-line">{{ reportTarget.exam_name }}
          <span v-if="reportTarget.body_part"> · {{ reportTarget.body_part }}</span>
        </p>
        <n-form-item label="报告内容（原始文本）" label-placement="top">
          <n-input v-model:value="rawContent" type="textarea" :autosize="{ minRows: 3, maxRows: 8 }" placeholder="影像/检验所见描述…" />
        </n-form-item>
        <n-form-item label-placement="top">
          <template #label>
            结构化指标（可选，JSON，服务端按参考区间自动算异常旗标与危急值）
          </template>
          <n-input v-model:value="structuredText" type="textarea" :autosize="{ minRows: 2, maxRows: 6 }"
            placeholder='{"items":[{"name":"血钾","code":"K","value":2.8,"unit":"mmol/L","ref_low":3.5,"ref_high":5.3}]}' />
        </n-form-item>
        <n-form-item label="结论" label-placement="top">
          <n-input v-model:value="conclusion" type="textarea" :autosize="{ minRows: 2, maxRows: 4 }" placeholder="检查结论（可留空）" />
        </n-form-item>
        <div class="switch-row">
          <span>自动派发 AI 解读</span>
          <n-switch v-model:value="autoInterpret" />
        </div>
        <n-space justify="end">
          <n-button @click="reportTarget = null">取消</n-button>
          <n-button type="primary" :loading="submitting" @click="submitReportNow">提交报告</n-button>
        </n-space>
      </template>
    </n-modal>
  </div>
</template>

<style scoped>
.hint { margin-bottom: var(--space-3); }
.cell-muted { color: var(--gray-5); }
.target-line { margin: 0 0 var(--space-3); color: var(--gray-7); }
.switch-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--space-3); color: var(--gray-8); font-size: var(--font-base); }
</style>
