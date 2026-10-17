/**
 * 收费 API。患者读自己的，医生开单，收费员收款/退费/作废。
 * 对齐 app/api/v1/billing.py。
 */
import { api } from './client'
import type { PageReq, PageResp } from '@/types/common'
import type { ChargeOrder } from '@/types/clinical'

export function listMyOrders(params: PageReq): Promise<PageResp<ChargeOrder>> {
  return api.get<PageResp<ChargeOrder>>('/billing/orders', { params })
}

export function getOrder(id: string): Promise<ChargeOrder> {
  return api.get<ChargeOrder>(`/billing/orders/${id}`)
}

export interface ChargeQuery extends PageReq {
  status?: string
  patient_id?: string
}

export type ChargeOrderRow = ChargeOrder

/** 收费员全域视图（charge_order:read_all）。 */
export function listAllOrders(params: ChargeQuery): Promise<PageResp<ChargeOrderRow>> {
  return api.get<PageResp<ChargeOrderRow>>('/billing/orders/all', { params })
}

export interface OrderItemIn {
  type: 'prescription' | 'exam' | 'other'
  prescription_id?: string
  exam_request_id?: string
  name?: string
  amount?: number
}

/** 医生为就诊建收费单；金额由服务端按源单据求值。 */
export function createOrder(body: { visit_id: string; items: OrderItemIn[] }): Promise<ChargeOrder> {
  return api.post<ChargeOrder>('/billing/orders', body)
}

/** 收款：pending→paid，处方项联动 confirmed→paid。 */
export function payOrder(id: string, body: { channel: string }): Promise<ChargeOrder> {
  return api.post<ChargeOrder>(`/billing/orders/${id}/pay`, body)
}

/** 退费：paid→refunded，reason 必填。 */
export function refundOrder(id: string, body: { reason: string }): Promise<ChargeOrder> {
  return api.post<ChargeOrder>(`/billing/orders/${id}/refund`, body)
}

/** 作废未缴单：pending→cancelled。 */
export function cancelOrder(id: string): Promise<ChargeOrder> {
  return api.post<ChargeOrder>(`/billing/orders/${id}/cancel`, {})
}
