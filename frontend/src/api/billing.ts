/**
 * 收费 API（患者读自己的缴费单）。对齐 app/api/v1/billing.py。
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
