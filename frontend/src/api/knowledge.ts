/**
 * 知识库管理 API（管理端）。对齐 app/api/v1/knowledge.py。
 */
import { api } from './client'
import type {
  DocDraft,
  DocListResp,
  KbTask,
  KnowledgeDoc,
  RebuildAccepted,
  SearchResp,
} from '@/types/admin'

export function listDocuments(params: {
  doc_type?: string
  source_tier?: string
  index_status?: string
  is_active?: boolean
  page?: number
  page_size?: number
}): Promise<DocListResp> {
  return api.get('/knowledge/documents', { params })
}

export function getDocument(docId: string): Promise<KnowledgeDoc> {
  return api.get(`/knowledge/documents/${docId}`)
}

export function createDocument(draft: DocDraft): Promise<KnowledgeDoc> {
  return api.post('/knowledge/documents', draft)
}

export function updateDocument(
  docId: string,
  patch: Partial<DocDraft> & { is_reviewed?: boolean; is_active?: boolean },
): Promise<KnowledgeDoc> {
  return api.patch(`/knowledge/documents/${docId}`, patch)
}

export function deleteDocument(docId: string): Promise<void> {
  return api.delete(`/knowledge/documents/${docId}`)
}

/** 触发索引重建（异步）。返回 task_id 后轮询 getTask。 */
export function rebuildIndex(): Promise<RebuildAccepted> {
  return api.post('/knowledge/rebuild')
}

export function getTask(taskId: string): Promise<KbTask> {
  return api.get(`/knowledge/tasks/${taskId}`)
}

/** 检索试用：管理员验证切块与召回效果。 */
export function search(payload: { query: string; trace_id?: string }): Promise<SearchResp> {
  return api.post('/knowledge/search', payload)
}
