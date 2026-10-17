<script setup lang="ts">
/**
 * 智能分诊。患者描述症状 → 后端返回推荐科室、紧急度、依据与可挂号源。
 * 急症结果用醒目警示样式（红），常规用信息样式，避免弱化急症信号。
 */
import { ref, h } from 'vue'
import {
  NInput, NButton, NCard, NAlert, NTag, NDescriptions, NDescriptionsItem,
  NEmpty, NSpin, useMessage, NDataTable,
} from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import { assess } from '@/api/triage'
import { ApiError } from '@/api/client'
import type { TriageResponse, SuggestedSlot } from '@/types/clinical'
import { URGENCY_TRIAGE_LABELS, DECISION_SOURCE_LABELS, SOURCE_TIER_LABELS } from '@/types/enums'
import { formatFen } from '@/utils/currency'
import { useRouter } from 'vue-router'

const router = useRouter()
const message = useMessage()
const complaint = ref('')
const loading = ref(false)
const result = ref<TriageResponse | null>(null)

async function run() {
  const text = complaint.value.trim()
  if (text.length < 2) {
    message.warning('请至少描述 2 个字')
    return
  }
  loading.value = true
  result.value = null
  try {
    result.value = await assess({ chief_complaint: text, include_slots: true })
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '分诊失败，请稍后再试')
  } finally {
    loading.value = false
  }
}

const slotColumns: DataTableColumns<SuggestedSlot> = [
  { title: '医生', key: 'doctor_name' },
  { title: '职称', key: 'doctor_title', render: (r) => r.doctor_title || '—' },
  { title: '日期', key: 'schedule_date' },
  { title: '时段', key: 'time_slot' },
  { title: '余号', key: 'remaining_slots', align: 'right', className: 'cell-number' },
  {
    title: '挂号费',
    key: 'registration_fee',
    align: 'right',
    className: 'cell-number',
    render: (r) => formatFen(r.registration_fee),
  },
  {
    title: '操作',
    key: 'action',
    render: (r) =>
      h(
        NButton,
        { size: 'tiny', type: 'primary', onClick: () => goBook(r) },
        { default: () => '去预约' },
      ),
  },
]

function goBook(slot: SuggestedSlot) {
  router.push({
    name: 'patient-appointments',
    query: { schedule_id: slot.schedule_id },
  })
}
</script>

<template>
  <div class="page-container">
    <n-card title="智能分诊" :bordered="false">
      <template #header-extra>
        <span class="page-subtitle">描述症状，系统推荐就诊科室</span>
      </template>
      <n-input
        v-model:value="complaint"
        type="textarea"
        placeholder="例如：最近三天持续头晕，伴有恶心，早上起床时最明显"
        :autosize="{ minRows: 3, maxRows: 6 }"
        :disabled="loading"
      />
      <div class="triage__actions">
        <n-button type="primary" :loading="loading" @click="run">开始分诊</n-button>
      </div>
    </n-card>

    <n-spin :show="loading">
      <div v-if="result" class="triage__result">
        <n-alert
          :type="result.is_emergency ? 'error' : result.urgency === 'stat' ? 'error' : 'info'"
          :bordered="false"
          :title="result.is_emergency ? '急症提示' : '分诊建议'"
        >
          <div class="triage__advice">{{ result.advice || result.reason }}</div>
        </n-alert>

        <n-card :bordered="false" class="triage__detail">
          <n-descriptions :column="2" label-placement="left" bordered size="small">
            <n-descriptions-item label="推荐科室">
              {{ result.recommended_department_name || '—' }}
            </n-descriptions-item>
            <n-descriptions-item label="紧急度">
              <n-tag
                size="small"
                :type="result.urgency === 'stat' ? 'error' : result.urgency === 'urgent' ? 'warning' : 'default'"
              >
                {{ URGENCY_TRIAGE_LABELS[result.urgency] }}
              </n-tag>
            </n-descriptions-item>
            <n-descriptions-item label="决策来源">
              {{ DECISION_SOURCE_LABELS[result.decision_source] || result.decision_source }}
            </n-descriptions-item>
            <n-descriptions-item label="置信度">
              <span class="num">{{ result.confidence != null ? (result.confidence * 100).toFixed(0) + '%' : '—' }}</span>
            </n-descriptions-item>
            <n-descriptions-item v-if="result.matched_emergency_keywords?.length" label="命中警示词" :span="2">
              <n-tag v-for="k in result.matched_emergency_keywords" :key="k" size="small" type="error" :bordered="false" class="triage__kw">{{ k }}</n-tag>
            </n-descriptions-item>
          </n-descriptions>

          <div v-if="result.reason" class="triage__reason">
            <span class="triage__reason-label">分诊理由：</span>{{ result.reason }}
          </div>

          <template v-if="result.evidence?.length">
            <div class="triage__ev-title">参考依据</div>
            <div v-for="(e, i) in result.evidence" :key="i" class="triage__ev">
              <n-tag size="tiny" :bordered="false">{{ SOURCE_TIER_LABELS[e.source_tier] || e.source_tier }}</n-tag>
              <span class="triage__ev-name">《{{ e.title }}》</span>
              <span class="triage__ev-snippet">{{ e.snippet }}</span>
            </div>
          </template>
        </n-card>

        <n-card v-if="result.suggested_slots?.length" title="可挂号源" :bordered="false" class="triage__slots">
          <n-data-table
            :columns="slotColumns"
            :data="result.suggested_slots"
            :bordered="false"
            size="small"
            :row-key="(r: SuggestedSlot) => r.schedule_id"
          />
        </n-card>
      </div>
      <n-empty v-else-if="!loading" description="填写症状后点击开始分诊" class="triage__empty" />
    </n-spin>
  </div>
</template>

<style scoped>
.triage__actions {
  margin-top: var(--space-4);
  text-align: right;
}
.triage__result {
  margin-top: var(--space-5);
}
.triage__advice {
  white-space: pre-wrap;
  line-height: var(--leading-base);
}
.triage__detail {
  margin-top: var(--space-4);
}
.triage__kw {
  margin-right: var(--space-1);
}
.triage__reason {
  margin-top: var(--space-4);
  font-size: var(--font-sm);
  color: var(--gray-8);
  line-height: 1.6;
}
.triage__reason-label {
  color: var(--gray-7);
}
.triage__ev-title {
  margin-top: var(--space-4);
  font-size: var(--font-sm);
  font-weight: 600;
  color: var(--gray-9);
}
.triage__ev {
  margin-top: var(--space-2);
  font-size: var(--font-sm);
  line-height: 1.5;
}
.triage__ev-name {
  font-weight: 600;
  color: var(--gray-8);
  margin: 0 var(--space-2);
}
.triage__ev-snippet {
  color: var(--gray-7);
}
.triage__slots {
  margin-top: var(--space-4);
}
.triage__empty {
  margin-top: var(--space-12);
}
</style>
