<script setup lang="ts">
/**
 * 接诊工作台。单页完成：病历书写 → 开检查单 → 结束就诊。
 *
 * 病历用「渐进成文」心智：任何时候都能保存，首次保存后端自动
 * in_progress → documented；结案按钮只在 documented 后放行（后端硬约束），
 * 前端同步禁用并把原因写在 tooltip 里，不让医生点了才知道不行。
 */
import { ref, reactive, computed, onMounted, h } from 'vue'
import {
  NCard, NForm, NFormItem, NInput, NButton, NSpace, NTag, NIcon,
  NDataTable, NModal, NPopconfirm, NAlert, NDescriptions, NDescriptionsItem,
  NSelect, NRadioGroup, NRadioButton, NTooltip,
  useMessage,
} from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import { SaveOutline, FlaskOutline, ArrowBackOutline } from '@vicons/ionicons5'
import { useRoute, useRouter } from 'vue-router'
import {
  getVisit, saveMedicalRecord, closeVisit, orderExam, listVisitExams,
  type MedicalRecordPayload,
} from '@/api/consult'
import { ApiError } from '@/api/client'
import type { Visit, ExamRequest } from '@/types/clinical'
import {
  VISIT_STATUS_LABELS, EXAM_REQUEST_STATUS_LABELS, URGENCY_LABELS,
} from '@/types/enums'
import { formatFen } from '@/utils/currency'

const route = useRoute()
const router = useRouter()
const message = useMessage()

const visitId = computed(() => String(route.params.visitId))
const visit = ref<Visit | null>(null)
const loading = ref(false)
const saving = ref(false)
const closing = ref(false)

/** 表单草稿。icd_codes 输入用逗号分隔文本，提交时切分成数组。 */
const form = reactive({
  present_illness: '',
  past_history: '',
  physical_exam: '',
  diagnosis: '',
  icdText: '',
  treatment_advice: '',
})

const exams = ref<ExamRequest[]>([])

async function loadVisit() {
  loading.value = true
  try {
    visit.value = await getVisit(visitId.value)
    const v = visit.value
    form.present_illness = v.present_illness ?? ''
    form.past_history = v.past_history ?? ''
    form.physical_exam = v.physical_exam ?? ''
    form.diagnosis = v.diagnosis ?? ''
    form.icdText = (v.icd_codes ?? []).join(', ')
    form.treatment_advice = v.treatment_advice ?? ''
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '加载病历失败')
  } finally {
    loading.value = false
  }
}

async function loadExams() {
  try {
    exams.value = await listVisitExams(visitId.value)
  } catch {
    // 检查单列表失败不阻断病历书写，静默为空即可
    exams.value = []
  }
}

function buildPayload(): MedicalRecordPayload {
  const codes = form.icdText.split(/[,，\s]+/).map((s) => s.trim()).filter(Boolean)
  return {
    present_illness: form.present_illness || undefined,
    past_history: form.past_history || undefined,
    physical_exam: form.physical_exam || undefined,
    diagnosis: form.diagnosis || undefined,
    icd_codes: codes.length ? codes : undefined,
    treatment_advice: form.treatment_advice || undefined,
  }
}

const hasContent = computed(() =>
  Object.values(form).some((v) => String(v).trim().length > 0),
)

async function doSave() {
  saving.value = true
  try {
    const resp = await saveMedicalRecord(visitId.value, buildPayload())
    visit.value = resp.visit
    message.success(resp.transitioned ? '病历已定稿' : '已保存')
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '保存失败')
  } finally {
    saving.value = false
  }
}

async function doClose() {
  closing.value = true
  try {
    await closeVisit(visitId.value)
    message.success('就诊已结束')
    router.push({ name: 'doctor-queue' })
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '结束就诊失败')
  } finally {
    closing.value = false
  }
}

// ==================== 开检查单 ====================

/**
 * 与后端 consult_service.EXAM_FEE_TABLE_CENTS 逐字同步的八项演示价目。
 * 漂移的后果是医生看到的预估价与收费票据对不上，改动后端价目必须同步这里。
 */
const EXAM_PRESETS = [
  { code: 'blood_routine', name: '血常规', fee: 2500 },
  { code: 'urine_routine', name: '尿常规', fee: 1500 },
  { code: 'liver_function', name: '肝功能全套', fee: 6000 },
  { code: 'renal_function', name: '肾功能全套', fee: 6000 },
  { code: 'blood_glucose', name: '血糖', fee: 500 },
  { code: 'blood_lipid', name: '血脂四项', fee: 800 },
  { code: 'ecg', name: '心电图', fee: 3000 },
  { code: 'ultrasound_abdomen', name: '腹部B超', fee: 12000 },
]

const examModal = ref(false)
const examForm = reactive({ typeCode: '', examName: '', bodyPart: '', urgency: 'routine' as string })

const examTypeOptions = EXAM_PRESETS.map((p) => ({
  label: `${p.name}（¥${(p.fee / 100).toFixed(2)}）`,
  value: p.code,
}))

const feePreview = computed(() => {
  const hit = EXAM_PRESETS.find((p) => p.code === examForm.typeCode)
  return hit ? hit.fee : null
})

function openExamModal() {
  examForm.typeCode = ''
  examForm.examName = ''
  examForm.bodyPart = ''
  examForm.urgency = 'routine'
  examModal.value = true
}

async function submitExam() {
  const preset = EXAM_PRESETS.find((p) => p.code === examForm.typeCode)
  const name = examForm.examName.trim() || preset?.name
  if (!examForm.typeCode || !name) {
    message.warning('请选择检查项目')
    return
  }
  try {
    await orderExam(visitId.value, {
      exam_type: examForm.typeCode,
      exam_name: name,
      body_part: examForm.bodyPart || undefined,
      urgency: (examForm.urgency as 'routine' | 'urgent' | 'stat') ?? 'routine',
    })
    message.success('检查单已开出')
    examModal.value = false
    await loadExams()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '开单失败')
  }
}

const examColumns: DataTableColumns<ExamRequest> = [
  { title: '项目', key: 'exam_name' },
  { title: '部位', key: 'body_part', render: (r) => r.body_part || '—' },
  {
    title: '紧急度', key: 'urgency', width: 80,
    render: (r) => h(NTag, {
      size: 'small', bordered: false,
      type: r.urgency === 'stat' ? 'error' : r.urgency === 'urgent' ? 'warning' : 'default',
    }, { default: () => URGENCY_LABELS[r.urgency] }),
  },
  {
    title: '状态', key: 'status', width: 90,
    render: (r) => EXAM_REQUEST_STATUS_LABELS[r.status] ?? r.status,
  },
  {
    title: '费用', key: 'fee', width: 90, align: 'right', className: 'num',
    render: (r) => formatFen(r.fee_amount ?? 0),
  },
]

const statusType = computed(() => {
  const s = visit.value?.status
  return s === 'closed' ? 'success' : s === 'documented' ? 'info' : 'default'
})

/** 弹窗里选了预设项目但未手填名称时，占位提示显示预设名。 */
const presetName = computed(
  () => EXAM_PRESETS.find((p) => p.code === examForm.typeCode)?.name ?? '',
)

function goBack() {
  router.push({ name: 'doctor-queue' })
}

onMounted(async () => {
  await loadVisit()
  await loadExams()
})
</script>

<template>
  <div class="page-container" v-if="visit">
    <div class="wb-header">
      <div class="wb-header__left">
        <h2 class="wb-title">接诊工作台</h2>
        <n-tag :bordered="false" size="small" :type="statusType">
          {{ VISIT_STATUS_LABELS[visit.status] }}
        </n-tag>
      </div>
      <n-space :size="8">
        <n-button secondary @click="goBack">
          <template #icon><n-icon><arrow-back-outline /></n-icon></template>
          返回队列
        </n-button>
        <n-button @click="openExamModal" :disabled="visit.status === 'closed'">
          <template #icon><n-icon><flask-outline /></n-icon></template>
          开检查单
        </n-button>
        <n-popconfirm
          v-if="visit.status === 'documented'"
          positive-text="确认结案"
          negative-text="再等等"
          @positive-click="doClose"
        >
          <template #trigger>
            <n-button type="primary" :loading="closing">结束就诊</n-button>
          </template>
          结案后病历不可再修改，检查与收费仍会继续流转。确认结束本次就诊？
        </n-popconfirm>
        <n-button v-else type="primary" disabled>
          <n-tooltip trigger="hover">
            <template #trigger>结束就诊</template>
            需先保存病历（定稿后方可结案）
          </n-tooltip>
        </n-button>
      </n-space>
    </div>

    <n-alert
      v-if="visit.chief_complaint"
      type="info"
      :bordered="false"
      class="wb-complaint"
    >
      主诉：{{ visit.chief_complaint }}
      <span v-if="visit.exam_request_count"> · 已开检查 {{ visit.exam_request_count }} 项</span>
    </n-alert>

    <div class="wb-grid">
      <n-card title="病历书写" :bordered="false" size="small">
        <n-form label-placement="top" :disabled="visit.status === 'closed'">
          <n-form-item label="现病史">
            <n-input
              v-model:value="form.present_illness"
              type="textarea" :autosize="{ minRows: 3, maxRows: 8 }"
              placeholder="起病情况、发展演变、诊疗经过…"
              maxlength="5000" show-count
            />
          </n-form-item>
          <n-form-item label="既往史">
            <n-input
              v-model:value="form.past_history"
              type="textarea" :autosize="{ minRows: 2, maxRows: 5 }"
              placeholder="慢性病、手术、过敏史…" maxlength="3000" show-count
            />
          </n-form-item>
          <n-form-item label="体格检查">
            <n-input
              v-model:value="form.physical_exam"
              type="textarea" :autosize="{ minRows: 2, maxRows: 5 }"
              placeholder="阳性体征与重要阴性…" maxlength="3000" show-count
            />
          </n-form-item>
          <div class="wb-row2">
            <n-form-item label="诊断结论" class="wb-row2__main">
              <n-input v-model:value="form.diagnosis" placeholder="主要诊断" maxlength="1000" />
            </n-form-item>
            <n-form-item label="ICD-10 编码（逗号分隔）">
              <n-input v-model:value="form.icdText" placeholder="如 I10, E11.9" maxlength="200" />
            </n-form-item>
          </div>
          <n-form-item label="治疗建议 / 医嘱">
            <n-input
              v-model:value="form.treatment_advice"
              type="textarea" :autosize="{ minRows: 2, maxRows: 6 }"
              placeholder="用药与随访建议（具体处方在处方模块开具）" maxlength="5000" show-count
            />
          </n-form-item>
          <n-button
            type="primary"
            :loading="saving"
            :disabled="!hasContent || visit.status === 'closed'"
            @click="doSave"
          >
            <template #icon><n-icon><save-outline /></n-icon></template>
            保存病历
          </n-button>
          <span v-if="visit.status === 'in_progress' && hasContent" class="wb-hint">
            首次保存将定稿病历
          </span>
        </n-form>
      </n-card>

      <n-card title="本次检查单" :bordered="false" size="small">
        <n-data-table
          :columns="examColumns"
          :data="exams"
          :bordered="false"
          size="small"
          :row-key="(r: ExamRequest) => r.id"
        />
        <div v-if="!exams.length" class="wb-empty">尚未开具检查</div>
        <n-descriptions v-if="visit.status !== 'in_progress'" :column="1" label-placement="left" size="small" class="wb-meta">
          <n-descriptions-item label="病历状态">{{ VISIT_STATUS_LABELS[visit.status] }}</n-descriptions-item>
          <n-descriptions-item label="接诊时间">{{ (visit.started_at || '').slice(0, 16).replace('T', ' ') || '—' }}</n-descriptions-item>
        </n-descriptions>
      </n-card>
    </div>

    <n-modal
      v-model:show="examModal"
      preset="card"
      title="开具检查申请单"
      style="max-width: 480px"
    >
      <n-form label-placement="top">
        <n-form-item label="检查项目">
          <n-select
            v-model:value="examForm.typeCode"
            :options="examTypeOptions"
            filterable
            placeholder="选择常用项目"
          />
        </n-form-item>
        <n-form-item label="项目名称（可改，与收费票据一致）">
          <n-input v-model:value="examForm.examName" :placeholder="presetName || '默认取项目名'" maxlength="100" />
        </n-form-item>
        <n-form-item label="检查部位">
          <n-input v-model:value="examForm.bodyPart" placeholder="如 胸部（可不填）" maxlength="100" />
        </n-form-item>
        <n-form-item label="紧急度">
          <n-radio-group v-model:value="examForm.urgency">
            <n-radio-button value="routine">常规</n-radio-button>
            <n-radio-button value="urgent">紧急</n-radio-button>
            <n-radio-button value="stat">危急</n-radio-button>
          </n-radio-group>
        </n-form-item>
        <div class="wb-fee-hint">
          检查费预估：<b>{{ feePreview !== null ? formatFen(feePreview) : '表外项目·收费侧核价' }}</b>
        </div>
        <n-space justify="end" class="wb-modal-actions">
          <n-button @click="examModal = false">取消</n-button>
          <n-button type="primary" @click="submitExam">开单</n-button>
        </n-space>
      </n-form>
    </n-modal>
  </div>
  <div v-else class="page-container wb-loading">{{ loading ? '加载中…' : '病历不存在或无权查看' }}</div>
</template>

<style scoped>
.wb-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: var(--space-4);
}
.wb-header__left { display: flex; align-items: center; gap: var(--space-3); }
.wb-title { font-size: var(--font-xl); font-weight: 600; margin: 0; color: var(--gray-9); }
.wb-complaint { margin-bottom: var(--space-4); }
.wb-grid {
  display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: var(--space-4);
  align-items: start;
}
@media (max-width: 1100px) { .wb-grid { grid-template-columns: 1fr; } }
.wb-row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0 var(--space-4); }
.wb-hint { margin-left: var(--space-3); color: var(--gray-6); font-size: var(--font-sm); }
.wb-empty { color: var(--gray-5); font-size: var(--font-sm); padding: var(--space-3) 0; }
.wb-meta { margin-top: var(--space-4); }
.wb-fee-hint { color: var(--gray-7); font-size: var(--font-sm); margin-bottom: var(--space-3); }
.wb-modal-actions { margin-top: var(--space-2); }
.wb-loading { padding: var(--space-8); color: var(--gray-6); }
</style>
