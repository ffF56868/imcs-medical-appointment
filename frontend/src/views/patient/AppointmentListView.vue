<script setup lang="ts">
/**
 * 预约挂号。上半部号源列表（可按科室/日期筛选），下半部我的预约。
 * 从分诊页带 schedule_id 跳转时，自动滚动到该号源并高亮。
 */
import { ref, onMounted, computed, watch, h } from 'vue'
import type { VNode } from 'vue'
import {
  NCard, NDataTable, NButton, NTag, NSpace, NDatePicker,
  NInput, NEmpty, useMessage, useDialog, NTabs, NTabPane, NSpin,
} from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import { useRoute } from 'vue-router'
import { listSchedules, createAppointment, listAppointments, cancelAppointment } from '@/api/appointment'
import { ApiError } from '@/api/client'
import type { ScheduleWithDoctor, Appointment } from '@/types/clinical'
import { SCHEDULE_STATUS_LABELS, APPOINTMENT_STATUS_LABELS } from '@/types/enums'
import { formatFen } from '@/utils/currency'

const route = useRoute()
const message = useMessage()
const dialog = useDialog()

const loadingSchedules = ref(false)
const loadingAppts = ref(false)
const schedules = ref<ScheduleWithDoctor[]>([])
const appointments = ref<Appointment[]>([])
const keyword = ref('')
const deptCode = ref<string | null>(null)
const dateRange = ref<[number, number] | null>(null)
const highlightId = ref<string | null>(null)

const presetScheduleId = computed(() => (route.query.schedule_id as string) || '')

function toDateStr(ts: number): string {
  const d = new Date(ts)
  const m = `${d.getMonth() + 1}`.padStart(2, '0')
  const day = `${d.getDate()}`.padStart(2, '0')
  return `${d.getFullYear()}-${m}-${day}`
}

async function loadSchedules() {
  loadingSchedules.value = true
  try {
    const params: Record<string, unknown> = { page: 1, page_size: 50, status: 'open' }
    if (keyword.value) params.keyword = keyword.value
    if (deptCode.value) params.department_code = deptCode.value
    if (dateRange.value) {
      params.start_date = toDateStr(dateRange.value[0])
      params.end_date = toDateStr(dateRange.value[1])
    }
    const resp = await listSchedules(params)
    schedules.value = resp.items
    if (presetScheduleId.value) highlightId.value = presetScheduleId.value
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '加载号源失败')
  } finally {
    loadingSchedules.value = false
  }
}

async function loadAppointments() {
  loadingAppts.value = true
  try {
    const resp = await listAppointments({ page: 1, page_size: 50 })
    appointments.value = resp.items
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '加载预约失败')
  } finally {
    loadingAppts.value = false
  }
}

async function book(row: ScheduleWithDoctor) {
  if (row.remaining_slots <= 0) {
    message.warning('该号源已约满')
    return
  }
  dialog.info({
    title: '确认预约',
    content: `${row.doctor_name}（${row.department_name}）${row.schedule_date} ${row.time_slot}，挂号费 ${formatFen(row.registration_fee)}`,
    positiveText: '确认预约',
    negativeText: '取消',
    onPositiveClick: async () => {
      try {
        await createAppointment({ schedule_id: row.id })
        message.success('预约成功')
        await Promise.all([loadSchedules(), loadAppointments()])
      } catch (err) {
        message.error(err instanceof ApiError ? err.message : '预约失败')
      }
    },
  })
}

async function cancel(row: Appointment) {
  dialog.warning({
    title: '取消预约',
    content: `确定取消 ${row.appointment_date} ${row.time_slot} 的预约吗？`,
    positiveText: '确认取消',
    negativeText: '再想想',
    onPositiveClick: async () => {
      try {
        await cancelAppointment(row.id)
        message.success('已取消')
        await Promise.all([loadSchedules(), loadAppointments()])
      } catch (err) {
        message.error(err instanceof ApiError ? err.message : '取消失败')
      }
    },
  })
}

const scheduleColumns: DataTableColumns<ScheduleWithDoctor> = [
  { title: '科室', key: 'department_name' },
  { title: '医生', key: 'doctor_name' },
  { title: '职称', key: 'doctor_title', render: (r) => r.doctor_title || '—' },
  { title: '日期', key: 'schedule_date', className: 'num' },
  { title: '时段', key: 'time_slot', className: 'num' },
  { title: '余号', key: 'remaining_slots', align: 'right', className: 'cell-number' },
  { title: '挂号费', key: 'registration_fee', align: 'right', className: 'cell-number', render: (r) => formatFen(r.registration_fee) },
  {
    title: '状态',
    key: 'status',
    render: (r) => h(NTag, { size: 'small', bordered: false, type: r.status === 'open' ? 'success' : 'default' }, { default: () => SCHEDULE_STATUS_LABELS[r.status] }),
  },
  {
    title: '操作',
    key: 'action',
    render: (r) =>
      h(NButton, { size: 'tiny', type: 'primary', disabled: r.remaining_slots <= 0, onClick: () => book(r) }, { default: () => '预约' }),
  },
]

const apptColumns: DataTableColumns<Appointment> = [
  { title: '日期', key: 'appointment_date', className: 'num' },
  { title: '时段', key: 'time_slot', className: 'num' },
  { title: '科室', key: 'department_name', render: (r) => r.department_name || '—' },
  { title: '医生', key: 'doctor_name' },
  { title: '挂号费', key: 'registration_fee', align: 'right', className: 'cell-number', render: (r) => formatFen(r.registration_fee) },
  {
    title: '状态',
    key: 'status',
    render: (r) => h(NTag, { size: 'small', bordered: false }, { default: () => APPOINTMENT_STATUS_LABELS[r.status] }),
  },
  {
    title: '操作',
    key: 'action',
    render: (r) => {
      const btns: VNode[] = []
      if (r.status === 'booked') {
        // 签到由医生端叫号时确认（appointment:update 仅医生持有），患者页不提供
        btns.push(h(NButton, { size: 'tiny', type: 'error', quaternary: true, onClick: () => cancel(r) }, { default: () => '取消' }))
      }
      return h(NSpace, { size: 4 }, { default: () => btns })
    },
  },
]

const rowClassName = (row: ScheduleWithDoctor) =>
  row.id === highlightId.value ? 'row-highlight' : ''

onMounted(async () => {
  await Promise.all([loadSchedules(), loadAppointments()])
})

watch(() => route.query.schedule_id, () => {
  if (route.query.schedule_id) highlightId.value = route.query.schedule_id as string
})
</script>

<template>
  <div class="page-container">
    <n-tabs type="line" animated>
      <n-tab-pane name="book" tab="预约挂号">
        <n-card :bordered="false">
          <n-space :wrap="true" class="filter-bar">
            <n-input v-model:value="keyword" placeholder="医生 / 科室关键词" clearable style="width: 220px" />
            <n-date-picker v-model:value="dateRange" type="daterange" clearable style="width: 260px" />
            <n-button type="primary" @click="loadSchedules">查询</n-button>
          </n-space>
          <n-spin :show="loadingSchedules">
            <n-data-table
              :columns="scheduleColumns"
              :data="schedules"
              :bordered="false"
              size="small"
              :row-key="(r: ScheduleWithDoctor) => r.id"
              :row-class-name="rowClassName"
              :pagination="{ pageSize: 10 }"
            />
            <n-empty v-if="!loadingSchedules && !schedules.length" description="暂无可挂号源" class="empty-block" />
          </n-spin>
        </n-card>
      </n-tab-pane>

      <n-tab-pane name="mine" tab="我的预约">
        <n-card :bordered="false">
          <n-spin :show="loadingAppts">
            <n-data-table
              :columns="apptColumns"
              :data="appointments"
              :bordered="false"
              size="small"
              :row-key="(r: Appointment) => r.id"
              :pagination="{ pageSize: 10 }"
            />
            <n-empty v-if="!loadingAppts && !appointments.length" description="还没有预约记录" class="empty-block" />
          </n-spin>
        </n-card>
      </n-tab-pane>
    </n-tabs>
  </div>
</template>

<style scoped>
.filter-bar {
  margin-bottom: var(--space-4);
}
.empty-block {
  margin-top: var(--space-8);
}
:deep(.row-highlight) {
  background: var(--brand-1) !important;
}
</style>
