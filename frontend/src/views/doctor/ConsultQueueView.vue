<script setup lang="ts">
/**
 * 医生待诊队列。今日已签到 / 就诊中的预约，按号别时段排序。
 * 行内主按钮二选一：无活跃病历 →「开始接诊」；有 →「继续书写」。
 * 后端已回填 visit_id/visit_status，前端不做二次关联查询。
 */
import { ref, computed, onMounted, h } from 'vue'
import {
  NCard, NDataTable, NButton, NTag, NSpace, NDatePicker, NIcon,
  useMessage, useDialog,
} from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import { RefreshOutline } from '@vicons/ionicons5'
import { useRouter } from 'vue-router'
import { listPendingQueue, startConsultation, type PendingAppointment } from '@/api/consult'
import { ApiError } from '@/api/client'
import { APPOINTMENT_STATUS_LABELS } from '@/types/enums'

const router = useRouter()
const message = useMessage()
const dialog = useDialog()

const loading = ref(false)
const rows = ref<PendingAppointment[]>([])
const dateValue = ref<number | null>(null)

/** NDatePicker 给时间戳，后端要 YYYY-MM-DD */
const dateParam = computed(() => {
  if (dateValue.value === null) return undefined
  const d = new Date(dateValue.value)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
})

async function load() {
  loading.value = true
  try {
    const resp = await listPendingQueue(dateParam.value)
    rows.value = resp.items
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '加载待诊队列失败')
  } finally {
    loading.value = false
  }
}

function goWorkbench(visitId: string) {
  router.push({ name: 'doctor-visit', params: { visitId } })
}

function doStart(row: PendingAppointment) {
  dialog.info({
    title: '开始接诊',
    content: `确认开始接诊 ${row.time_slot} 的患者？系统将创建病历并带入主诉。`,
    positiveText: '开始接诊',
    negativeText: '取消',
    onPositiveClick: async () => {
      try {
        const resp = await startConsultation(row.appointment_id)
        message.success('已开始接诊')
        goWorkbench(resp.visit.id)
      } catch (err) {
        if (err instanceof ApiError && err.status === 409 && err.code === 'appointment_duplicate') {
          // ApiError 不携带 detail 扩展字段；刷新队列即可拿到后端回填的
          // visit_id，该行按钮会自动变成「继续书写」。
          message.info('该患者已有进行中的病历，已在队列中为你标出')
          await load()
          return false
        }
        message.error(err instanceof ApiError ? err.message : '开始接诊失败')
        return false
      }
    },
  })
}

const columns: DataTableColumns<PendingAppointment> = [
  { title: '时段', key: 'time_slot', width: 90, className: 'num' },
  {
    title: '主诉', key: 'chief_complaint', ellipsis: { tooltip: true },
    render: (r) => r.chief_complaint || '—',
  },
  {
    title: '分诊摘要', key: 'triage_summary', ellipsis: { tooltip: true },
    render: (r) => r.triage_summary || '—',
  },
  {
    title: '签到时间', key: 'checked_in_at', width: 130, className: 'num',
    render: (r) => (r.checked_in_at || '').slice(11, 16) || '—',
  },
  {
    title: '状态', key: 'status', width: 90,
    render: (r) => h(NTag, {
      size: 'small',
      bordered: false,
      type: r.status === 'in_progress' ? 'info' : 'default',
    }, { default: () => APPOINTMENT_STATUS_LABELS[r.status] ?? r.status }),
  },
  {
    title: '操作', key: 'action', width: 130,
    render: (r) =>
      r.visit_id
        ? h(NButton, { size: 'small', onClick: () => goWorkbench(r.visit_id!) }, { default: () => '继续书写' })
        : h(NButton, { size: 'small', type: 'primary', onClick: () => doStart(r) }, { default: () => '开始接诊' }),
  },
]

onMounted(load)
</script>

<template>
  <div class="page-container">
    <n-card :bordered="false">
      <template #header>今日待诊</template>
      <template #header-extra>
        <n-space align="center" :size="12">
          <n-date-picker
            v-model:value="dateValue"
            type="date"
            clearable
            size="small"
            style="width: 150px"
            :actions="['clear']"
            @update:value="load"
          />
          <n-button size="small" quaternary @click="load">
            <template #icon><n-icon><refresh-outline /></n-icon></template>
            刷新
          </n-button>
        </n-space>
      </template>
      <n-data-table
        :columns="columns"
        :data="rows"
        :loading="loading"
        :bordered="false"
        size="small"
        :row-key="(r: PendingAppointment) => r.appointment_id"
        :pagination="rows.length > 10 ? { pageSize: 10 } : false"
      />
    </n-card>
  </div>
</template>
