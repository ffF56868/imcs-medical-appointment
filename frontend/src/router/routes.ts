/**
 * 路由表。按角色组织，meta.roles 控制访问权。
 *
 * 菜单由本表过滤生成（见 composables/useMenu），因此路由即菜单：
 * 加一个页面只要在这里登记并给 meta.roles，侧栏自动出现，不必两处维护。
 */

import type { RouteRecordRaw } from 'vue-router'
import type { RoleName } from '@/types/enums'

declare module 'vue-router' {
  interface RouteMeta {
    /** 允许访问的角色；空表示仅登录即可（如个人设置） */
    roles?: RoleName[]
    /** 侧栏标题 */
    title?: string
    /** 侧栏图标名（Naive UI icon，见 layout/SideMenu） */
    icon?: string
    /** 是否在侧栏隐藏（如详情页） */
    hidden?: boolean
  }
}

const ALL_CLINICAL: RoleName[] = ['patient', 'doctor', 'lab_tech', 'cashier', 'pharmacist', 'admin']

export const routes: RouteRecordRaw[] = [
  {
    path: '/login',
    name: 'login',
    component: () => import('@/views/login/LoginView.vue'),
    meta: { hidden: true, title: '登录' },
  },
  {
    path: '/403',
    name: 'forbidden',
    component: () => import('@/views/error/ForbiddenView.vue'),
    meta: { hidden: true, title: '无访问权限' },
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: () => import('@/views/error/NotFoundView.vue'),
    meta: { hidden: true, title: '页面不存在' },
  },
  {
    path: '/',
    component: () => import('@/components/layout/AppLayout.vue'),
    redirect: '/home',
    children: [
      {
        path: 'home',
        name: 'home',
        component: () => import('@/views/common/HomeView.vue'),
        meta: { title: '工作台', icon: 'home', roles: ALL_CLINICAL },
      },

      // ---------- 患者端 ----------
      {
        path: 'patient/triage',
        name: 'patient-triage',
        component: () => import('@/views/patient/TriageView.vue'),
        meta: { title: '智能分诊', icon: 'triage', roles: ['patient'] },
      },
      {
        path: 'patient/consult',
        name: 'patient-consult',
        component: () => import('@/views/patient/HealthConsultView.vue'),
        meta: { title: '健康咨询', icon: 'chat', roles: ['patient'] },
      },
      {
        path: 'patient/appointments',
        name: 'patient-appointments',
        component: () => import('@/views/patient/AppointmentListView.vue'),
        meta: { title: '预约挂号', icon: 'calendar', roles: ['patient'] },
      },
      {
        path: 'patient/records',
        name: 'patient-records',
        component: () => import('@/views/patient/MyRecordsView.vue'),
        meta: { title: '我的病历', icon: 'record', roles: ['patient'] },
      },
      {
        path: 'patient/reports',
        name: 'patient-reports',
        component: () => import('@/views/patient/MyReportsView.vue'),
        meta: { title: '检查报告', icon: 'report', roles: ['patient'] },
      },
      {
        path: 'patient/prescriptions',
        name: 'patient-prescriptions',
        component: () => import('@/views/patient/MyPrescriptionsView.vue'),
        meta: { title: '我的处方', icon: 'prescription', roles: ['patient'] },
      },
      {
        path: 'patient/bills',
        name: 'patient-bills',
        component: () => import('@/views/patient/MyBillsView.vue'),
        meta: { title: '我的缴费', icon: 'bill', roles: ['patient'] },
      },

      // ---------- 医生端 ----------
      {
        path: 'doctor/queue',
        name: 'doctor-queue',
        component: () => import('@/views/doctor/ConsultQueueView.vue'),
        meta: { title: '待诊队列', icon: 'queue', roles: ['doctor'] },
      },
      {
        path: 'doctor/visits/:visitId',
        name: 'doctor-visit',
        component: () => import('@/views/doctor/VisitWorkbenchView.vue'),
        meta: { title: '接诊工作台', hidden: true, roles: ['doctor'] },
      },
      {
        path: 'doctor/visits',
        name: 'doctor-visits',
        component: () => import('@/views/doctor/MyVisitsView.vue'),
        meta: { title: '我的病历', icon: 'record', roles: ['doctor'] },
      },

      // ---------- 药师端 ----------
      {
        path: 'pharmacist/review',
        name: 'pharmacist-review',
        component: () => import('@/views/pharmacist/DispenseQueueView.vue'),
        meta: { title: '处方审核与发药', icon: 'pharmacy', roles: ['pharmacist'] },
      },
      {
        path: 'pharmacist/records',
        name: 'pharmacist-records',
        component: () => import('@/views/pharmacist/DispenseRecordsView.vue'),
        meta: { title: '发药记录', icon: 'record', roles: ['pharmacist'] },
      },

      // ---------- 收费员端 ----------
      {
        path: 'cashier/orders',
        name: 'cashier-orders',
        component: () => import('@/views/cashier/ChargeDeskView.vue'),
        meta: { title: '收费台', icon: 'bill', roles: ['cashier'] },
      },

      // ---------- 检验科端 ----------
      {
        path: 'lab/queue',
        name: 'lab-queue',
        component: () => import('@/views/lab/ExamQueueView.vue'),
        meta: { title: '检验工作台', icon: 'lab', roles: ['lab_tech'] },
      },
      {
        path: 'lab/reports',
        name: 'lab-reports',
        component: () => import('@/views/lab/AllReportsView.vue'),
        meta: { title: '全部报告', icon: 'report', roles: ['lab_tech'] },
      },

      // ---------- 管理员端 ----------
      {
        path: 'admin/monitor',
        name: 'admin-monitor',
        component: () => import('@/views/admin/MonitorView.vue'),
        meta: { title: 'AI 运行监控', icon: 'monitor', roles: ['admin'] },
      },
      {
        path: 'admin/evaluation',
        name: 'admin-evaluation',
        component: () => import('@/views/admin/EvaluationView.vue'),
        meta: { title: 'AI 评测中心', icon: 'eval', roles: ['admin'] },
      },
      {
        path: 'admin/knowledge',
        name: 'admin-knowledge',
        component: () => import('@/views/admin/KnowledgeView.vue'),
        meta: { title: '知识库管理', icon: 'knowledge', roles: ['admin'] },
      },
    ],
  },
]
