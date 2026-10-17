/**
 * 发药 API（患者读自己的取药记录）。对齐 app/api/v1/dispensing.py。
 */
import { api } from './client'
import type { PageReq, PageResp } from '@/types/common'
import type { DispensingRecord } from '@/types/clinical'

export function listMyRecords(params: PageReq): Promise<PageResp<DispensingRecord>> {
  return api.get<PageResp<DispensingRecord>>('/dispensing/records', { params })
}
