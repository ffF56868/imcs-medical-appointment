<script setup lang="ts">
/**
 * AI 评测中心（管理端）。题库 CRUD + 三类评测触发 + 运行历史 + 指标趋势。
 *
 * 交互按后端语义设计：
 * - retrieval / quality 同步跑（零 LLM、秒级），按钮 loading 到返回；
 * - RAGAS 是 202 排队 + 轮询运行状态，已有活跃任务时后端 409（防双跑烧配额）；
 * - 运行详情展示 metrics.gate（回归门禁档位），趋势页画同类运行的指标线。
 */
import { ref, computed, onMounted, onBeforeUnmount, h } from 'vue'
import type { VNode } from 'vue'
import {
  NButton, NTag, NSpace, NDataTable, NModal, NInput, NFormItem,
  NInputNumber, NSelect, NSwitch, NTabs, NTabPane, NPopconfirm, NAlert,
  NDivider, useMessage,
} from 'naive-ui'
import MetricChart from '@/components/MetricChart.vue'
import {
  listCases, createCase, updateCase, deleteCase,
  triggerRun, dispatchRagas, listRuns, getRun, getTrend,
} from '@/api/evaluation'
import { ApiError } from '@/api/client'
import type { EvalCase, EvalCaseDraft, EvalRun, EvalTrend } from '@/types/admin'
import { EVAL_KIND_LABELS, EVAL_RUN_STATUS_LABELS } from '@/types/enums'
import { formatDateTime } from '@/utils/format'

const message = useMessage()
const activeTab = ref<'cases' | 'runs' | 'trend'>('cases')

// ==================== 题库 ====================
const cases = ref<EvalCase[]>([])
const casesTotal = ref(0)
const casesPage = ref(1)
const casesCategory = ref('')
const casesLoading = ref(false)

async function loadCases() {
  casesLoading.value = true
  try {
    const r = await listCases({
      category: casesCategory.value || undefined,
      page: casesPage.value,
      page_size: 20,
    })
    cases.value = r.items
    casesTotal.value = r.total
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '题库加载失败')
  } finally {
    casesLoading.value = false
  }
}

const typeOptions = [
  { value: 'medication', label: '药品' },
  { value: 'disease', label: '疾病' },
  { value: 'department', label: '科室' },
  { value: 'other', label: '其他' },
]

// ---- 题目编辑 ----
const caseModalVisible = ref(false)
const caseEditMode = ref<'create' | 'edit'>('create')
const caseEditId = ref<string | null>(null)
const caseDraft = ref<EvalCaseDraft>(emptyDraft())
const caseSaving = ref(false)

function emptyDraft(): EvalCaseDraft {
  return {
    question: '', expected_name: '', expected_type: 'medication',
    category: '未分类', alternative_names: [], answer_keywords: [],
    citation_names: [], expected_refusal: false,
  }
}

function openCaseCreate() {
  caseEditMode.value = 'create'
  caseEditId.value = null
  caseDraft.value = emptyDraft()
  caseModalVisible.value = true
}

function openCaseEdit(row: EvalCase) {
  caseEditMode.value = 'edit'
  caseEditId.value = row.id
  const { id: _id, created_at: _c, ...rest } = row
  caseDraft.value = rest
  caseModalVisible.value = true
}

/** 输入框里逗号/顿号分隔的字符串 → 数组，保存时统一。 */
const altText = ref('')
const kwText = ref('')
const citeText = ref('')

function splitList(raw: string): string[] {
  return raw.split(/[,，、;；]/).map((s) => s.trim()).filter(Boolean)
}

async function saveCase() {
  const d = caseDraft.value
  if (!d.question.trim() || (!d.expected_name.trim() && !d.expected_refusal)) {
    message.warning('题干必填；非拒答题时期望实体也必填')
    return
  }
  d.alternative_names = splitList(altText.value)
  d.answer_keywords = splitList(kwText.value)
  d.citation_names = splitList(citeText.value)
  caseSaving.value = true
  try {
    if (caseEditMode.value === 'create') {
      await createCase(d)
      message.success('已新增评测题')
    } else if (caseEditId.value) {
      await updateCase(caseEditId.value, d)
      message.success('已更新')
    }
    caseModalVisible.value = false
    await loadCases()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '保存失败')
  } finally {
    caseSaving.value = false
  }
}

async function removeCase(row: EvalCase) {
  try {
    await deleteCase(row.id)
    message.success('已删除')
    await loadCases()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '删除失败')
  }
}

const caseColumns = computed(() => [
  { title: '题干', key: 'question', ellipsis: { tooltip: true } },
  {
    title: '期望实体', key: 'expected_name', width: 180, ellipsis: { tooltip: true },
    render: (r: EvalCase) => r.expected_refusal ? h(NTag, { size: 'small', type: 'warning', bordered: false }, () => '应拒答') : r.expected_name,
  },
  { title: '类型', key: 'expected_type', width: 80 },
  { title: '分类', key: 'category', width: 90 },
  {
    title: '操作', key: 'ops', width: 120,
    render: (r: EvalCase): VNode =>
      h(NSpace, { size: 6 }, () => [
        h(NButton, { size: 'tiny', onClick: () => openCaseEdit(r) }, () => '编辑'),
        h(NPopconfirm, { onPositiveClick: () => removeCase(r) }, {
          trigger: () => h(NButton, { size: 'tiny', type: 'error', quaternary: true }, () => '删除'),
          default: () => '删除该题？历史运行结果不受影响',
        }),
      ]),
  },
])

// ==================== 运行 ====================
const runs = ref<EvalRun[]>([])
const runsTotal = ref(0)
const runsPage = ref(1)
const runsKind = ref<string | null>(null)
const runsLoading = ref(false)
const syncBusy = ref<string | null>(null)
const ragasBusy = ref(false)
const ragasSample = ref(10)

const KIND_OPTIONS = [
  { value: null as unknown as string, label: '类型：全部' },
  ...Object.entries(EVAL_KIND_LABELS).map(([v, l]) => ({ value: v, label: l })),
]

async function loadRuns() {
  runsLoading.value = true
  try {
    const r = await listRuns({
      kind: runsKind.value || undefined,
      page: runsPage.value,
      page_size: 15,
    })
    runs.value = r.items
    runsTotal.value = r.total
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '运行列表加载失败')
  } finally {
    runsLoading.value = false
  }
}

async function runSync(kind: 'retrieval' | 'quality') {
  syncBusy.value = kind
  try {
    const run = await triggerRun(kind)
    message.success(`${EVAL_KIND_LABELS[kind]} 评测完成：${run.total_count} 题`)
    runsPage.value = 1
    await loadRuns()
    await openRun(run.id)
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '评测执行失败')
  } finally {
    syncBusy.value = null
  }
}

async function runRagas() {
  ragasBusy.value = true
  try {
    const d = await dispatchRagas(ragasSample.value)
    message.info(d.message)
    startPollRun(d.eval_run_id)
    await loadRuns()
  } catch (err) {
    if (err instanceof ApiError && err.status === 409) {
      message.warning('已有 RAGAS 任务在跑（评测烧配额，禁止双跑），请等待完成')
    } else {
      message.error(err instanceof ApiError ? err.message : '派发失败')
    }
  } finally {
    ragasBusy.value = false
  }
}

// ---- RAGAS 轮询 ----
let pollTimer: ReturnType<typeof setTimeout> | null = null
const pollRunId = ref<string | null>(null)
const pollRun = ref<EvalRun | null>(null)

function stopPoll() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null }
  pollRunId.value = null
}
onBeforeUnmount(stopPoll)

function startPollRun(runId: string) {
  stopPoll()
  pollRunId.value = runId
  pollRun.value = null
  pollTick()
}

async function pollTick() {
  if (!pollRunId.value) return
  try {
    const run = await getRun(pollRunId.value)
    pollRun.value = run
    if (run.status === 'completed' || run.status === 'failed') {
      stopPoll()
      if (run.status === 'completed') message.success('RAGAS 评测完成')
      else message.error(`RAGAS 失败：${run.error_message ?? '未知原因'}`)
      await loadRuns()
      return
    }
    pollTimer = setTimeout(pollTick, 4000)
  } catch {
    stopPoll()
  }
}

// ---- 运行详情 ----
const runModalVisible = ref(false)
const runDetail = ref<EvalRun | null>(null)
const runDetailLoading = ref(false)

async function openRun(runId: string) {
  runDetailLoading.value = true
  try {
    runDetail.value = await getRun(runId)
    runModalVisible.value = true
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '详情加载失败')
  } finally {
    runDetailLoading.value = false
  }
}

const runStatusTag = (s: string) =>
  s === 'completed' ? 'success' : s === 'failed' ? 'error' : s === 'running' ? 'info' : 'default'

const pct = (v: number | null) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)

/** 每种 kind 展示各自有意义的指标列，避免整排空值。 */
const metricCols = (run: EvalRun): Array<{ label: string; value: string }> => {
  if (run.kind === 'retrieval') {
    const m = (run.metrics ?? {}) as Record<string, number>
    return [
      { label: 'Top1 命中', value: pct(m.top1_accuracy ?? null) },
      { label: 'Recall@3', value: pct(m.recall_at_3 ?? null) },
      { label: 'Precision@3', value: pct(m.precision_at_3 ?? null) },
    ]
  }
  if (run.kind === 'quality') {
    return [
      { label: '回答准确', value: pct(run.answer_accuracy) },
      { label: '引用准确', value: pct(run.citation_accuracy) },
      { label: '拒答正确', value: pct(run.refusal_accuracy) },
    ]
  }
  return [
    { label: '忠实度', value: pct(run.faithfulness) },
    { label: '相关度', value: pct(run.answer_relevancy) },
    { label: 'Context P', value: pct(run.context_precision) },
    { label: 'Context R', value: pct(run.context_recall) },
  ]
}

const runColumns = computed(() => [
  { title: '时间', key: 'created_at', width: 150, render: (r: EvalRun) => formatDateTime(r.created_at) },
  { title: '类型', key: 'kind', width: 90, render: (r: EvalRun) => EVAL_KIND_LABELS[r.kind] ?? r.kind },
  {
    title: '状态', key: 'status', width: 80,
    render: (r: EvalRun) => h(NTag, { size: 'small', type: runStatusTag(r.status) as never, bordered: false }, () => EVAL_RUN_STATUS_LABELS[r.status] ?? r.status),
  },
  { title: '题数', key: 'total_count', width: 60, render: (r: EvalRun) => `${r.completed_count}/${r.total_count}` },
  { title: '知识库文档', key: 'knowledge_document_count', width: 100 },
  {
    title: '主要指标', key: 'main', ellipsis: { tooltip: true },
    render: (r: EvalRun) => metricCols(r).map((c) => `${c.label} ${c.value}`).join(' · '),
  },
  {
    title: '操作', key: 'ops', width: 80,
    render: (r: EvalRun): VNode => h(NButton, { size: 'tiny', onClick: () => openRun(r.id) }, () => '详情'),
  },
])

const gateView = computed(() => {
  const gate = (runDetail.value?.metrics as Record<string, unknown> | null)?.gate as
    Record<string, unknown> | undefined
  if (!gate) return null
  const status = gate.status as string
  const metrics = gate.metrics as Record<string, { level?: string; delta?: number }> | undefined
  return {
    status,
    reason: gate.reason as string | undefined,
    items: Object.entries(metrics ?? {}).map(([k, v]) => ({ key: k, level: v.level ?? '—', delta: v.delta ?? 0 })),
  }
})

// ==================== 趋势 ====================
const trendKind = ref<'retrieval' | 'quality' | 'ragas'>('retrieval')
const trend = ref<EvalTrend | null>(null)
const trendLoading = ref(false)

const TREND_METRICS: Record<string, Array<{ key: string; label: string }>> = {
  retrieval: [
    { key: 'top1_accuracy', label: 'Top1' },
    { key: 'recall_at_3', label: 'Recall@3' },
    { key: 'precision_at_3', label: 'Precision@3' },
  ],
  quality: [
    { key: 'answer_accuracy', label: '回答准确' },
    { key: 'citation_accuracy', label: '引用准确' },
    { key: 'refusal_accuracy', label: '拒答正确' },
  ],
  ragas: [
    { key: 'faithfulness', label: '忠实度' },
    { key: 'answer_relevancy', label: '相关度' },
    { key: 'context_precision', label: 'CtxP' },
    { key: 'context_recall', label: 'CtxR' },
  ],
}

async function loadTrend() {
  trendLoading.value = true
  try {
    trend.value = await getTrend(trendKind.value)
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '趋势加载失败')
  } finally {
    trendLoading.value = false
  }
}

const trendSeries = computed(() => {
  if (!trend.value) return []
  const defs = TREND_METRICS[trend.value.kind] ?? []
  return defs
    .map((d) => ({
      name: d.label,
      points: trend.value!.points
        .filter((p) => typeof p[d.key] === 'number')
        .map((p) => [String(p.created_at ?? '').slice(5, 10), p[d.key] as number] as [string, number]),
    }))
    .filter((s) => s.points.length > 0)
})

function onTabChange(t: 'cases' | 'runs' | 'trend') {
  if (t === 'runs' && !runs.value.length) loadRuns()
  if (t === 'trend' && !trend.value) loadTrend()
}

onMounted(() => { loadCases(); loadRuns() })
</script>

<template>
  <div class="ev">
    <n-tabs v-model:value="activeTab" type="line" animated @update:value="onTabChange">
      <!-- 题库 -->
      <n-tab-pane name="cases" tab="题库">
        <div class="bar">
          <n-input v-model:value="casesCategory" size="small" placeholder="按分类过滤" clearable style="width: 160px" @keyup.enter="() => { casesPage = 1; loadCases() }" />
          <div class="spacer" />
          <n-button size="small" type="primary" @click="openCaseCreate">新增题目</n-button>
        </div>
        <n-data-table
          :columns="caseColumns" :data="cases" :loading="casesLoading" size="small"
          :row-key="(r: EvalCase) => r.id" remote
          :pagination="{ page: casesPage, pageSize: 20, itemCount: casesTotal, onUpdatePage: (p: number) => { casesPage = p; loadCases() } }"
        />
      </n-tab-pane>

      <!-- 运行 -->
      <n-tab-pane name="runs" tab="评测运行">
        <div class="bar">
          <n-button size="small" type="primary" :loading="syncBusy === 'retrieval'" @click="runSync('retrieval')">跑检索评测</n-button>
          <n-button size="small" type="primary" secondary :loading="syncBusy === 'quality'" @click="runSync('quality')">跑质量评测</n-button>
          <n-divider vertical />
          <span class="lbl">RAGAS 抽样</span>
          <n-input-number v-model:value="ragasSample" size="small" :min="1" :max="50" style="width: 90px" />
          <n-button size="small" :loading="ragasBusy" @click="runRagas">派发</n-button>
          <div class="spacer" />
          <n-select v-model:value="runsKind" :options="KIND_OPTIONS" size="small" style="width: 140px" @update:value="() => { runsPage = 1; loadRuns() }" />
        </div>
        <n-alert v-if="pollRun" type="info" :bordered="false" class="poll">
          RAGAS 运行中（{{ pollRun.completed_count }}/{{ pollRun.total_count }} 题完成），每 4 秒自动刷新…
        </n-alert>
        <n-data-table
          :columns="runColumns" :data="runs" :loading="runsLoading" size="small"
          :row-key="(r: EvalRun) => r.id" remote
          :pagination="{ page: runsPage, pageSize: 15, itemCount: runsTotal, onUpdatePage: (p: number) => { runsPage = p; loadRuns() } }"
        />
      </n-tab-pane>

      <!-- 趋势 -->
      <n-tab-pane name="trend" tab="指标趋势">
        <div class="bar">
          <n-select
            v-model:value="trendKind" size="small" style="width: 140px"
            :options="[{ value: 'retrieval', label: '检索评测' }, { value: 'quality', label: '质量评测' }, { value: 'ragas', label: 'RAGAS' }]"
            @update:value="loadTrend"
          />
          <n-button size="small" :loading="trendLoading" @click="loadTrend">刷新</n-button>
        </div>
        <metric-chart v-if="trendSeries.length" :series="trendSeries" :height="280" :y-axis-format="(v: number) => `${(v * 100).toFixed(0)}%`" />
        <div v-else class="empty">同类 completed 运行不足 2 次，无法绘制趋势线</div>
      </n-tab-pane>
    </n-tabs>
  </div>

  <!-- 题目编辑弹窗 -->
  <n-modal v-model:show="caseModalVisible" preset="card" :title="caseEditMode === 'create' ? '新增评测题' : '编辑评测题'" style="max-width: 640px">
    <n-form label-placement="top">
      <n-form-item label="题干">
        <n-input v-model:value="caseDraft.question" type="textarea" :autosize="{ minRows: 2 }" placeholder="如：成人原发性高血压首选药物？" />
      </n-form-item>
      <n-space>
        <n-form-item label="期望实体" style="width: 260px">
          <n-input v-model:value="caseDraft.expected_name" :disabled="caseDraft.expected_refusal" placeholder="如 氨氯地平" />
        </n-form-item>
        <n-form-item label="实体类型" style="width: 140px">
          <n-select v-model:value="caseDraft.expected_type" :options="typeOptions" />
        </n-form-item>
        <n-form-item label="分类" style="width: 130px">
          <n-input v-model:value="caseDraft.category" placeholder="高血压" />
        </n-form-item>
      </n-space>
      <n-space>
        <n-form-item label="可接受别名（逗号分隔）" style="width: 300px">
          <n-input v-model:value="altText" placeholder="络活喜, 氨氯地平片" />
        </n-form-item>
        <n-form-item label="答案关键词（逗号分隔）" style="width: 300px">
          <n-input v-model:value="kwText" placeholder="钙通道阻滞剂, 降压" />
        </n-form-item>
      </n-space>
      <n-space>
        <n-form-item label="期望引用来源（逗号分隔）" style="width: 300px">
          <n-input v-model:value="citeText" placeholder="中国高血压防治指南" />
        </n-form-item>
        <n-form-item label="期望拒答（超纲题）">
          <n-switch v-model:value="caseDraft.expected_refusal" />
        </n-form-item>
      </n-space>
    </n-form>
    <template #footer>
      <n-space justify="end">
        <n-button @click="caseModalVisible = false">取消</n-button>
        <n-button type="primary" :loading="caseSaving" @click="saveCase">保存</n-button>
      </n-space>
    </template>
  </n-modal>

  <!-- 运行详情弹窗 -->
  <n-modal v-model:show="runModalVisible" preset="card" :title="runDetail ? `${EVAL_KIND_LABELS[runDetail.kind]} · ${runDetail.total_count} 题` : '运行详情'" style="max-width: 720px">
    <template v-if="runDetail">
      <div class="detail-metrics">
        <div v-for="m in metricCols(runDetail)" :key="m.label" class="dm-cell">
          <span class="dm-label">{{ m.label }}</span>
          <span class="dm-value">{{ m.value }}</span>
        </div>
      </div>
      <template v-if="gateView">
        <div class="gate-head">回归门禁：{{ gateView.status }}<span v-if="gateView.reason" class="gate-reason">{{ gateView.reason }}</span></div>
        <div v-for="g in gateView.items" :key="g.key" class="gate-row">
          <n-tag size="small" :bordered="false" :type="g.level === 'attention' ? 'error' : g.level === 'warning' ? 'warning' : g.level === 'improved' ? 'success' : 'default'">{{ g.level }}</n-tag>
          <span class="gate-key">{{ g.key }}</span>
          <span class="gate-delta">{{ (g.delta * 100).toFixed(1) }}pp</span>
        </div>
      </template>
      <template v-if="runDetail.results?.length">
        <div class="res-head">逐题明细（前 30）</div>
        <div v-for="(item, i) in runDetail.results.slice(0, 30)" :key="i" class="res-row">
          <n-tag size="tiny" :bordered="false" :type="item.pass ? 'success' : 'error'">{{ item.pass ? '✓' : '✗' }}</n-tag>
          <span class="res-q">{{ item.question ?? item.query ?? JSON.stringify(item).slice(0, 80) }}</span>
        </div>
      </template>
    </template>
  </n-modal>
</template>

<style scoped>
.ev { background: #fff; }
.bar { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.bar .spacer { flex: 1; }
.lbl { font-size: 12.5px; color: #6b7280; }
.poll { margin-bottom: 10px; }
.empty { color: #9ca3af; font-size: 12.5px; padding: 40px 0; text-align: center; }
.detail-metrics { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
.dm-cell { border: 1px solid var(--imcs-border, #e5e7eb); border-radius: 4px; padding: 8px 14px; min-width: 110px; }
.dm-label { display: block; font-size: 11.5px; color: #6b7280; }
.dm-value { font-size: 17px; font-variant-numeric: tabular-nums; }
.gate-head { font-size: 13px; font-weight: 600; margin-bottom: 8px; }
.gate-reason { font-weight: 400; color: #6b7280; font-size: 12px; margin-left: 8px; }
.gate-row { display: flex; align-items: center; gap: 10px; font-size: 12.5px; padding: 3px 0; }
.gate-key { font-family: Consolas, monospace; }
.gate-delta { color: #6b7280; font-variant-numeric: tabular-nums; }
.res-head { font-size: 13px; font-weight: 600; margin: 14px 0 6px; }
.res-row { display: flex; align-items: center; gap: 8px; font-size: 12.5px; padding: 2px 0; }
.res-q { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
</style>
