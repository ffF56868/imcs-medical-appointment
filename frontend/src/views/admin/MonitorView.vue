<script setup lang="ts">
/**
 * AI 运行监控（管理端）。看板首屏 + 指标曲线 + 调用明细 + trace 全链路回放。
 * 视觉沿用临床克制：读数网格卡、单色细线曲线、trace 用时间线列表而非泳道图。
 */
import { ref, computed, onMounted, h } from 'vue'
import {
  NCard, NButton, NSelect, NDataTable, NModal, NTag, NInput,
  useMessage,
} from 'naive-ui'
import MetricChart from '@/components/MetricChart.vue'
import { getOverview, getMetricSeries, listLlmCalls, replayTrace, triggerAggregate } from '@/api/monitoring'
import { ApiError } from '@/api/client'
import type { MonitoringOverview, LlmCallRow, TraceReplay } from '@/types/admin'
import { LLM_CALL_STATUS_LABELS } from '@/types/enums'
import { formatDateTime } from '@/utils/format'

const message = useMessage()

interface Series { name: string; points: Array<[string, number]> }

// ==================== 窗口与首屏 ====================
const WINDOW_OPTIONS = [
  { value: 15, label: '近 15 分钟' },
  { value: 60, label: '近 1 小时' },
  { value: 240, label: '近 4 小时' },
  { value: 1440, label: '近 24 小时' },
]
const windowMinutes = ref(60)
const overview = ref<MonitoringOverview | null>(null)
const ovLoading = ref(false)
const aggLoading = ref(false)

async function loadOverview() {
  ovLoading.value = true
  try {
    overview.value = await getOverview(windowMinutes.value)
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '看板数据加载失败')
  } finally {
    ovLoading.value = false
  }
}
onMounted(loadOverview)

async function aggregateNow() {
  aggLoading.value = true
  try {
    const r = await triggerAggregate({ lookback_minutes: 30 })
    message.success(r.message)
    await loadOverview()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '聚合失败')
  } finally {
    aggLoading.value = false
  }
}

function ratio(v: number | null | undefined): string {
  if (v == null) return '—'
  return `${(v * 100).toFixed(2)}%`
}
const cards = computed(() => {
  const m = overview.value?.metrics
  if (!m) return []
  return [
    { label: '调用次数', value: m.llm_call_count == null ? '—' : String(Math.round(m.llm_call_count)) },
    { label: '错误率', value: ratio(m.llm_error_rate) },
    { label: '降级率', value: ratio(m.degradation_rate) },
    { label: '拦截率', value: ratio(m.moderation_block_rate) },
    { label: '零召回率', value: ratio(m.retrieval_zero_result_rate) },
    { label: 'Token 用量', value: m.token_usage_total == null ? '—' : Math.round(m.token_usage_total).toLocaleString() },
    { label: '成本 USD', value: m.token_cost_usd == null ? '—' : `$${m.token_cost_usd.toFixed(4)}` },
    { label: '队列积压', value: m.task_queue_backlog == null ? '—' : String(Math.round(m.task_queue_backlog)) },
  ]
})

// ==================== 指标曲线 ====================
const METRIC_META: Record<string, { label: string; ratio: boolean }> = {
  llm_error_rate: { label: '错误率', ratio: true },
  degradation_rate: { label: '降级率', ratio: true },
  retrieval_zero_result_rate: { label: '零召回率', ratio: true },
  llm_p95_latency: { label: 'P95 延迟', ratio: false },
  llm_first_token_p95: { label: '首 token P95', ratio: false },
  token_usage: { label: 'Token 用量', ratio: false },
}
const METRIC_OPTIONS = Object.entries(METRIC_META).map(([v, m]) => ({ value: v, label: m.label }))

const chosen = ref<string[]>(['llm_error_rate', 'retrieval_zero_result_rate'])
const series = ref<Series[]>([])
const chartLoading = ref(false)

const yAxisFormat = computed(() => {
  const allRatio = chosen.value.length > 0 && chosen.value.every((k) => METRIC_META[k]?.ratio)
  return allRatio ? (v: number) => `${(v * 100).toFixed(1)}%` : undefined
})

async function loadSeries() {
  if (chosen.value.length === 0) { series.value = []; return }
  chartLoading.value = true
  try {
    const end = new Date()
    const start = new Date(end.getTime() - 24 * 3600_000)
    const resp = await getMetricSeries({
      metrics: chosen.value.join(','),
      start_date: start.toISOString(),
      end_date: end.toISOString(),
      limit: 1440,
    })
    const byMetric = new Map<string, Array<[string, number]>>()
    for (const p of resp.items) {
      // 后端桶时间是 UTC ISO；取时分直接展示，曲线内口径一致
      const label = (p.bucket_time ?? '').slice(11, 16)
      if (!byMetric.has(p.metric_name)) byMetric.set(p.metric_name, [])
      byMetric.get(p.metric_name)!.push([label, p.metric_value])
    }
    series.value = [...byMetric.entries()].map(([name, points]) => ({
      name: METRIC_META[name]?.label ?? name,
      points,
    }))
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '曲线加载失败')
  } finally {
    chartLoading.value = false
  }
}

// ==================== 调用明细 ====================
const calls = ref<LlmCallRow[]>([])
const callsTotal = ref(0)
const callsPage = ref(1)
const callStatus = ref<string | null>(null)
const callsLoading = ref(false)

const STATUS_OPTIONS = [
  { value: null as unknown as string, label: '状态：全部' },
  ...Object.entries(LLM_CALL_STATUS_LABELS).map(([v, l]) => ({ value: v, label: l })),
]

async function loadCalls() {
  callsLoading.value = true
  try {
    const r = await listLlmCalls({
      status_filter: callStatus.value || undefined,
      page: callsPage.value,
      page_size: 15,
    })
    calls.value = r.items
    callsTotal.value = r.total
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '调用明细加载失败')
  } finally {
    callsLoading.value = false
  }
}
onMounted(loadCalls)

const statusTagType = (s: string) =>
  s === 'success' ? 'success' : s === 'degraded' ? 'warning' : s === 'timeout' ? 'info' : 'error'

const callColumns = computed(() => [
  { title: '时间', key: 'created_at', width: 150, render: (r: LlmCallRow) => r.created_at ? formatDateTime(r.created_at) : '—' },
  { title: '场景', key: 'scenario', width: 110 },
  { title: '模型', key: 'model', width: 130, ellipsis: { tooltip: true } },
  {
    title: '状态', key: 'status', width: 70,
    render: (r: LlmCallRow) => h(NTag, { size: 'small', type: statusTagType(r.status) as never, bordered: false }, () => LLM_CALL_STATUS_LABELS[r.status] ?? r.status),
  },
  { title: '延迟', key: 'latency_ms', width: 80, render: (r: LlmCallRow) => `${r.latency_ms.toLocaleString()} ms` },
  {
    title: '首token', key: 'first_token_ms', width: 90,
    render: (r: LlmCallRow) => (r.is_stream && r.first_token_ms != null ? `${r.first_token_ms.toLocaleString()} ms` : '—'),
  },
  { title: 'tokens', key: 'total_tokens', width: 80, render: (r: LlmCallRow) => r.total_tokens.toLocaleString() },
  { title: '成本', key: 'cost_usd', width: 80, render: (r: LlmCallRow) => (r.cost_usd == null ? '—' : `$${r.cost_usd.toFixed(5)}`) },
  {
    title: 'trace', key: 'trace_id', width: 100,
    render: (r: LlmCallRow) => r.trace_id
      ? h('span', { class: 'trace-link', onClick: () => doReplay(r.trace_id!) }, `${r.trace_id.slice(0, 8)}…`)
      : '—',
  },
])

// ==================== trace 回放 ====================
const replayInput = ref('')
const replay = ref<TraceReplay | null>(null)
const replayVisible = ref(false)
const replayLoading = ref(false)

async function doReplay(raw?: string) {
  const traceId = (raw ?? replayInput.value).trim()
  if (!traceId) { message.warning('请输入 trace_id'); return }
  replayLoading.value = true
  try {
    replay.value = await replayTrace(traceId)
    replayVisible.value = true
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '回放加载失败（trace 可能不存在）')
  } finally {
    replayLoading.value = false
  }
}

const KIND_LABELS: Record<string, string> = {
  llm_call: 'LLM 调用',
  retrieval: 'RAG 检索',
  workflow: '工作流',
  triage_result: '分诊',
}
</script>

<template>
  <div class="mon">
    <!-- 首屏 -->
    <n-card size="small" :bordered="false" class="mon-card">
      <div class="head-row">
        <n-select
          v-model:value="windowMinutes" :options="WINDOW_OPTIONS" size="small"
          style="width: 130px" @update:value="loadOverview"
        />
        <n-tag v-if="overview" size="small" :bordered="false" :type="overview.source === 'aggregated' ? 'success' : 'warning'">
          数据源 {{ overview.source }} · {{ overview.bucket_count }} 桶
        </n-tag>
        <div class="spacer" />
        <n-button size="small" :loading="ovLoading" @click="loadOverview">刷新</n-button>
        <n-button size="small" :loading="aggLoading" @click="aggregateNow">聚合近 30 分钟</n-button>
      </div>
      <div class="metric-grid">
        <div v-for="c in cards" :key="c.label" class="metric-cell">
          <span class="metric-label">{{ c.label }}</span>
          <span class="metric-value">{{ c.value }}</span>
        </div>
      </div>
      <div v-if="overview && Object.keys(overview.queue).length" class="queue-row">
        队列：<span v-for="(n, k) in overview.queue" :key="k" class="queue-item">{{ k }}={{ n }}</span>
      </div>
    </n-card>

    <!-- 曲线 -->
    <n-card size="small" title="指标曲线（近 24 小时）" :bordered="false" class="mon-card">
      <div class="head-row">
        <n-select
          v-model:value="chosen" :options="METRIC_OPTIONS" multiple size="small"
          style="min-width: 320px" placeholder="选择指标（可多选）"
        />
        <n-button size="small" type="primary" :loading="chartLoading" @click="loadSeries">查询</n-button>
      </div>
      <metric-chart v-if="series.length" :series="series" :height="240" :y-axis-format="yAxisFormat" />
      <div v-else class="chart-empty">选择至少一个指标后点击查询</div>
    </n-card>

    <!-- 明细 -->
    <n-card size="small" title="LLM 调用明细" :bordered="false" class="mon-card">
      <div class="head-row">
        <n-select v-model:value="callStatus" :options="STATUS_OPTIONS" size="small" style="width: 140px" @update:value="() => { callsPage = 1; loadCalls() }" />
        <div class="spacer" />
        <n-input v-model:value="replayInput" size="small" placeholder="trace_id" style="width: 260px" @keyup.enter="doReplay()" />
        <n-button size="small" :loading="replayLoading" @click="doReplay()">回放链路</n-button>
      </div>
      <n-data-table
        :columns="callColumns" :data="calls" :loading="callsLoading" size="small"
        :row-key="(r: LlmCallRow) => r.id" remote
        :pagination="{ page: callsPage, pageSize: 15, itemCount: callsTotal, onUpdatePage: (p: number) => { callsPage = p; loadCalls() } }"
      />
    </n-card>
  </div>

  <!-- 回放弹窗 -->
  <n-modal v-model:show="replayVisible" preset="card" title="trace 全链路回放" style="max-width: 760px">
    <template v-if="replay">
      <div class="replay-head">
        <span class="mono">{{ replay.trace_id }}</span>
        <n-tag size="small" :bordered="false">{{ replay.counts.llm_calls }} 调用</n-tag>
        <n-tag size="small" :bordered="false">{{ replay.counts.retrievals }} 检索</n-tag>
        <n-tag size="small" :bordered="false">{{ replay.counts.workflows }} 工作流</n-tag>
        <n-tag size="small" :bordered="false">{{ replay.counts.triage_results }} 分诊</n-tag>
      </div>
      <div class="replay-sum">
        tokens {{ replay.summary.total_tokens.toLocaleString() }} ·
        成本 ${{ replay.summary.cost_usd.toFixed(5) }} ·
        最大延迟 {{ replay.summary.max_latency_ms ?? '—' }} ms ·
        错误 {{ replay.summary.error_count }} / 降级 {{ replay.summary.degraded_count }}
      </div>
      <ul class="timeline">
        <li v-for="(e, i) in replay.timeline" :key="i">
          <span class="t-time">{{ e.at ? String(e.at).slice(11, 19) : '--:--:--' }}</span>
          <n-tag size="small" :bordered="false">{{ KIND_LABELS[e.kind] ?? e.kind }}</n-tag>
          <span class="t-summary">{{ e.summary }}</span>
        </li>
      </ul>
      <div v-if="!replay.timeline.length" class="chart-empty">该 trace 没有产生任何日志段</div>
    </template>
  </n-modal>
</template>

<style scoped>
.mon { display: flex; flex-direction: column; gap: 12px; }
.mon-card { background: #fff; border: 1px solid var(--imcs-border, #e5e7eb); border-radius: 6px; }
.head-row { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.head-row .spacer, .queue-row .spacer { flex: 1; }
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--imcs-border, #e5e7eb); border: 1px solid var(--imcs-border, #e5e7eb); }
.metric-cell { background: #fff; padding: 10px 14px; display: flex; flex-direction: column; gap: 4px; }
.metric-label { font-size: 12px; color: #6b7280; }
.metric-value { font-size: 18px; font-variant-numeric: tabular-nums; color: #111827; }
.queue-row { margin-top: 10px; font-size: 12px; color: #6b7280; }
.queue-item { margin-right: 12px; font-variant-numeric: tabular-nums; }
.chart-empty { color: #9ca3af; font-size: 12.5px; padding: 28px 0; text-align: center; }
.trace-link { color: var(--imcs-brand, #1B5E7B); cursor: pointer; }
.trace-link:hover { text-decoration: underline; }
.mono { font-family: Consolas, monospace; font-size: 12px; }
.replay-head { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
.replay-sum { font-size: 12.5px; color: #6b7280; margin-bottom: 12px; }
.timeline { list-style: none; margin: 0; padding: 0; border-left: 2px solid var(--imcs-border, #e5e7eb); }
.timeline li { display: flex; align-items: baseline; gap: 8px; padding: 6px 0 6px 14px; position: relative; }
.timeline li::before { content: ''; position: absolute; left: -5px; top: 13px; width: 8px; height: 8px; border-radius: 50%; background: var(--imcs-brand, #1B5E7B); }
.t-time { font-size: 12px; color: #9ca3af; font-variant-numeric: tabular-nums; width: 64px; flex-shrink: 0; }
.t-summary { font-size: 12.5px; color: #374151; }
</style>
