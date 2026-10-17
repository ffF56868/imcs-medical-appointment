/**
 * 排班与预约 API。对齐 app/api/v1/schedule.py 与 appointment.py。
 */
import { api } from './client'
import type { PageReq, PageResp } from '@/types/common'
import type { Appointment, ScheduleWithDoctor } from '@/types/clinical'

export interface ScheduleQuery extends PageReq {
  start_date?: string
  end_date?: string
  department_code?: string
  department_id?: string
  doctor_id?: string
  slot_period?: 'am' | 'pm'
  status?: 'open' | 'full' | 'closed'
  keyword?: string
}

export function listSchedules(params: ScheduleQuery): Promise<PageResp<ScheduleWithDoctor>> {
  return api.get<PageResp<ScheduleWithDoctor>>('/schedules', { params })
}

export interface AppointmentCreate {
  schedule_id: string
  chief_complaint?: string | null
  triage_result_id?: string | null
  triage_summary?: string | null
}

export function createAppointment(payload: AppointmentCreate): Promise<Appointment> {
  return api.post<Appointment>('/appointments', payload)
}

export interface AppointmentQuery extends PageReq {
  status?: string
  date_from?: string
  date_to?: string
  doctor_id?: string
}

export function listAppointments(params: AppointmentQuery): Promise<PageResp<Appointment>> {
  return api.get<PageResp<Appointment>>('/appointments', { params })
}

export function getAppointment(id: string): Promise<Appointment> {
  return api.get<Appointment>(`/appointments/${id}`)
}

export function cancelAppointment(id: string): Promise<void> {
  return api.delete<void>(`/appointments/${id}`)
}

export function checkIn(id: string): Promise<Appointment> {
  return api.patch<Appointment>(`/appointments/${id}/check-in`)
}
