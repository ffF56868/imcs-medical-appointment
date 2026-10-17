<script setup lang="ts">
/**
 * 知识库管理（管理端）。文档 CRUD + 索引重建 + 检索试用。
 *
 * 重建是异步任务：POST /rebuild 拿 task_id，进度条轮询 /tasks/{id}
 * 直到终态。检索试用面板直接打 /knowledge/search，把双路命中数与
 * 融合结果原样展示——管理员调库时最想看的就是「这句话到底召回了什么」。
 */
import { ref, computed, onMounted, onBeforeUnmount, h } from 'vue'
import type { VNode } from 'vue'
import {
  NDataTable, NButton, NTag, NSpace, NModal, NInput,
  NFormItem, NSelect, NSwitch, NProgress, NEmpty, NForm, NGrid, NGi,
  NDescriptions, NDescriptionsItem, NPopconfirm, NTabs, NTabPane, useMessage,
} from 'naive-ui'
import {
  listDocuments, createDocument, updateDocument, deleteDocument,
  rebuildIndex, getTask, search,
} from '@/api/knowledge'
import { ApiError } from '@/api/client'
import type { KnowledgeDoc, KnowledgeDocSummary, DocDraft, KbTask, SearchResp } from '@/types/admin'
import { DOC_TYPE_LABELS, SOURCE_TIER_LABELS, INDEX_STATUS_LABELS } from '@/types/enums'
import { formatDateTime } from '@/utils/format'

const message = useMessage()

const DOC_TYPE_OPTIONS = Object.entries(DOC_TYPE_LABELS).map(([v, label]) => ({ value: v, label }))
const TIER_OPTIONS = Object.entries(SOURCE_TIER_LABELS).map(([v, label]) => ({ value: v, label }))
const INDEX_OPTIONS = [
  { value: '', label: '索引状态：全部' },
  ...Object.entries(INDEX_STATUS_LABELS).map(([v, label]) => ({ value: v, label })),
]

// ==================== 列表 ====================
const rows = ref<KnowledgeDocSummary[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = ref(20)
const loading = ref(false)
const fDocType = ref<string | null>(null)
const fTier = ref<string | null>(null)
const fIndex = ref('')
const fActive = ref<string | null>(null)

async function load() {
  loading.value = true
  try {
    const r = await listDocuments({
      doc_type: fDocType.value || undefined,
      source_tier: fTier.value || undefined,
      index_status: fIndex.value || undefined,
      is_active: fActive.value === null ? undefined : fActive.value === 'yes',
      page: page.value,
      page_size: pageSize.value,
    })
    rows.value = r.items
    total.value = r.total
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '文档列表加载失败')
  } finally {
    loading.value = false
  }
}
onMounted(load)

function resetAndLoad() {
  page.value = 1
  load()
}

const tierTagType = (t: string) =>
  t === 'authoritative' ? 'success' : t === 'professional' ? 'info' : t === 'general' ? 'default' : 'warning'

const indexTagType = (s: string) =>
  s === 'indexed' ? 'success' : s === 'stale' ? 'warning' : s === 'failed' ? 'error' : 'default'

// ==================== 详情 ====================
const detailDoc = ref<KnowledgeDoc | null>(null)
const detailVisible = ref(false)

async function openDetail(row: KnowledgeDocSummary) {
  try {
    detailDoc.value = await import('@/api/knowledge').then((m) => m.getDocument(row.id))
    detailVisible.value = true
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '文档详情加载失败')
  }
}

// ==================== 编辑 ====================
const editVisible = ref(false)
const editMode = ref<'create' | 'edit'>('create')
const editTargetId = ref<string | null>(null)
const draft = ref<DocDraft & { is_reviewed?: boolean; is_active?: boolean }>({
  title: '', content: '', doc_type: 'guideline', source: '', source_url: '', source_tier: 'professional',
})
const saving = ref(false)

function openCreate() {
  editMode.value = 'create'
  editTargetId.value = null
  draft.value = { title: '', content: '', doc_type: 'guideline', source: '', source_url: '', source_tier: 'professional' }
  editVisible.value = true
}

function openEdit(row: KnowledgeDocSummary) {
  void openDetail(row)
  message.info('编辑将正文与元数据一起载入弹窗')
  editMode.value = 'edit'
  editTargetId.value = row.id
  // 正文不进列表响应，编辑前拉全文
  import('@/api/knowledge').then((m) => m.getDocument(row.id)).then((doc) => {
    draft.value = {
      title: doc.title, content: doc.content, doc_type: doc.doc_type,
      source: doc.source, source_url: doc.source_url, source_tier: doc.source_tier,
      is_reviewed: doc.is_reviewed, is_active: doc.is_active,
    }
    editVisible.value = true
  }).catch(() => message.error('加载全文失败'))
}

async function saveDoc() {
  if (!draft.value.title.trim() || !draft.value.content.trim() || !draft.value.source.trim()) {
    message.warning('标题、正文、来源都必填')
    return
  }
  saving.value = true
  try {
    if (editMode.value === 'create') {
      await createDocument(draft.value)
      message.success('已创建，索引进队列')
    } else if (editTargetId.value) {
      await updateDocument(editTargetId.value, draft.value)
      message.success('已更新')
    }
    editVisible.value = false
    await load()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '保存失败')
  } finally {
    saving.value = false
  }
}

async function removeDoc(row: KnowledgeDocSummary) {
  try {
    await deleteDocument(row.id)
    message.success(`已删除「${row.title}」`)
    await load()
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '删除失败')
  }
}

const columns = computed(() => [
  {
    title: '标题', key: 'title', ellipsis: { tooltip: true },
    render: (r: KnowledgeDocSummary) => h('span', { class: 'clickable', onClick: () => openDetail(r) }, r.title),
  },
  { title: '类型', key: 'doc_type', width: 100, render: (r: KnowledgeDocSummary) => DOC_TYPE_LABELS[r.doc_type] ?? r.doc_type },
  { title: '来源', key: 'source_tier', width: 80, render: (r: KnowledgeDocSummary) => h(NTag, { size: 'small', type: tierTagType(r.source_tier) as never, bordered: false }, () => SOURCE_TIER_LABELS[r.source_tier] ?? r.source_tier) },
  { title: '索引', key: 'index_status', width: 90, render: (r: KnowledgeDocSummary) => h(NTag, { size: 'small', type: indexTagType(r.index_status) as never, bordered: false }, () => INDEX_STATUS_LABELS[r.index_status] ?? r.index_status) },
  { title: '切块', key: 'chunk_count', width: 70 },
  { title: '字数', key: 'content_length', width: 80, render: (r: KnowledgeDocSummary) => r.content_length.toLocaleString() },
  { title: '审核', key: 'is_reviewed', width: 60, render: (r: KnowledgeDocSummary) => (r.is_reviewed ? '✓' : '—') },
  {
    title: '操作', key: 'ops', width: 150,
    render: (r: KnowledgeDocSummary): VNode =>
      h(NSpace, { size: 4 }, () => [
        h(NButton, { size: 'tiny', onClick: () => openEdit(r) }, () => '编辑'),
        h(NPopconfirm, { onPositiveClick: () => removeDoc(r) }, {
          trigger: () => h(NButton, { size: 'tiny', type: 'error', quaternary: true }, () => '删除'),
          default: () => `确认删除「${r.title}」？切块会一并清理`,
        }),
      ]),
  },
])

// ==================== 索引重建 ====================
const rebuilding = ref(false)
const rebuildTask = ref<KbTask | null>(null)
let pollTimer: ReturnType<typeof setTimeout> | null = null

function stopPoll() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null }
}
onBeforeUnmount(stopPoll)

async function startRebuild() {
  rebuilding.value = true
  stopPoll()
  try {
    const accepted = await rebuildIndex()
    message.info(`重建任务已排队（${accepted.task_id.slice(0, 8)}…）`)
    pollTask(accepted.task_id)
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '触发重建失败')
    rebuilding.value = false
  }
}

async function pollTask(taskId: string) {
  try {
    const t = await getTask(taskId)
    rebuildTask.value = t
    if (t.status === 'completed' || t.status === 'failed') {
      rebuilding.value = false
      if (t.status === 'completed') message.success('索引重建完成')
      else message.error(`重建失败：${t.error_message ?? '未知原因'}`)
      await load()
      return
    }
    pollTimer = setTimeout(() => pollTask(taskId), 2000)
  } catch {
    rebuilding.value = false
  }
}

// ==================== 检索试用 ====================
const sq = ref('')
const sBusy = ref(false)
const sResp = ref<SearchResp | null>(null)

async function runSearch() {
  if (sq.value.trim().length < 1) { message.warning('请输入检索问题'); return }
  sBusy.value = true
  try {
    sResp.value = await search({ query: sq.value.trim() })
    if (sResp.value.diagnostics.is_zero_result) message.warning('零召回——知识库可能缺相关资料')
  } catch (err) {
    message.error(err instanceof ApiError ? err.message : '检索失败')
  } finally {
    sBusy.value = false
  }
}

const activeTab = ref('docs')
</script>

<template>
  <n-tabs v-model:value="activeTab" type="line" animated pane-class="kb-pane">
    <n-tab-pane name="docs" tab="知识文档">
      <div class="kb-filter">
        <n-select v-model:value="fDocType" :options="[{ value: '', label: '类型：全部' }, ...DOC_TYPE_OPTIONS]" style="width: 140px" @update:value="resetAndLoad" />
        <n-select v-model:value="fTier" :options="[{ value: '', label: '来源：全部' }, ...TIER_OPTIONS]" style="width: 120px" @update:value="resetAndLoad" />
        <n-select v-model:value="fIndex" :options="INDEX_OPTIONS" style="width: 130px" @update:value="resetAndLoad" />
        <n-select v-model:value="fActive" :options="[{ value: '', label: '启用：全部' }, { value: 'yes', label: '仅启用' }, { value: 'no', label: '仅停用' }]" style="width: 110px" @update:value="resetAndLoad" />
        <div class="spacer" />
        <n-button size="small" @click="load">刷新</n-button>
        <n-button size="small" :loading="rebuilding" @click="startRebuild">重建索引</n-button>
        <n-button size="small" type="primary" @click="openCreate">新增文档</n-button>
      </div>

      <div v-if="rebuilding && rebuildTask" class="kb-progress">
        <n-progress type="line" :percentage="rebuildTask.progress" :height="8" processing />
        <span>{{ rebuildTask.progress_message ?? '排队中…' }}</span>
      </div>

      <n-data-table
        :columns="columns" :data="rows" :loading="loading" size="small"
        :row-key="(r: KnowledgeDocSummary) => r.id" remote
        :pagination="{ page, pageSize, itemCount: total, onUpdatePage: (p: number) => { page = p; load() } }"
      />
    </n-tab-pane>

    <n-tab-pane name="search" tab="检索试用">
      <div class="kb-search">
        <n-input v-model:value="sq" placeholder="输入临床问题，验证混合检索召回效果，如「成人社区获得性肺炎首选抗生素」" style="max-width: 560px" @keyup.enter="runSearch" />
        <n-button type="primary" :loading="sBusy" @click="runSearch">检索</n-button>
      </div>
      <template v-if="sResp">
        <div class="diag">
          向量命中 {{ sResp.diagnostics.vector_hit_count }} · 关键词命中 {{ sResp.diagnostics.keyword_hit_count }} ·
          双路都中 {{ sResp.diagnostics.both_hit_count }} · 融合后 {{ sResp.diagnostics.merged_count }}
          <n-tag v-if="sResp.diagnostics.is_zero_result" size="small" type="warning" :bordered="false">零召回</n-tag>
        </div>
        <div v-for="(hit, i) in sResp.results" :key="hit.chunk_id" class="hit">
          <div class="hit-head">
            <n-tag size="small" :bordered="false">{{ i + 1 }}</n-tag>
            <strong>{{ hit.title }}</strong>
            <n-tag size="small" :type="tierTagType(hit.source_tier) as never" :bordered="false">{{ SOURCE_TIER_LABELS[hit.source_tier] ?? hit.source_tier }}</n-tag>
            <span class="score">RRF {{ hit.rrf_score.toFixed(4) }}</span>
            <span v-if="hit.page_range" class="page">{{ hit.page_range }}</span>
          </div>
          <div class="hit-body">{{ hit.content }}</div>
        </div>
        <n-empty v-if="!sResp.results.length" description="没有召回任何切块" style="margin: 32px 0" />
      </template>
    </n-tab-pane>
  </n-tabs>

  <!-- 详情 -->
  <n-modal v-model:show="detailVisible" preset="card" :title="detailDoc?.title" style="max-width: 780px">
    <template v-if="detailDoc">
      <n-descriptions :column="3" size="small" label-placement="left" label-style="width:64px" class="det-desc">
        <n-descriptions-item label="类型">{{ DOC_TYPE_LABELS[detailDoc.doc_type] ?? detailDoc.doc_type }}</n-descriptions-item>
        <n-descriptions-item label="来源">{{ detailDoc.source }}</n-descriptions-item>
        <n-descriptions-item label="分级">{{ SOURCE_TIER_LABELS[detailDoc.source_tier] ?? detailDoc.source_tier }}</n-descriptions-item>
        <n-descriptions-item label="索引">{{ INDEX_STATUS_LABELS[detailDoc.index_status] ?? detailDoc.index_status }}</n-descriptions-item>
        <n-descriptions-item label="切块">{{ detailDoc.chunk_count }}</n-descriptions-item>
        <n-descriptions-item label="indexed">{{ detailDoc.indexed_at ? formatDateTime(detailDoc.indexed_at) : '—' }}</n-descriptions-item>
      </n-descriptions>
      <pre class="det-content">{{ detailDoc.content }}</pre>
    </template>
  </n-modal>

  <!-- 编辑 -->
  <n-modal v-model:show="editVisible" preset="card" :title="editMode === 'create' ? '新增知识文档' : '编辑文档'" style="max-width: 720px">
    <n-form label-placement="top">
      <n-grid :cols="2" :x-gap="12">
        <n-gi><n-form-item label="标题"><n-input v-model:value="draft.title" maxlength="300" /></n-form-item></n-gi>
        <n-gi><n-form-item label="资料类型"><n-select v-model:value="draft.doc_type" :options="DOC_TYPE_OPTIONS" /></n-form-item></n-gi>
        <n-gi><n-form-item label="来源机构"><n-input v-model:value="draft.source" placeholder="如 WHO / 中华医学会" /></n-form-item></n-gi>
        <n-gi><n-form-item label="来源可信度"><n-select v-model:value="draft.source_tier" :options="TIER_OPTIONS" /></n-form-item></n-gi>
      </n-grid>
      <n-form-item label="来源链接（可选）"><n-input v-model:value="draft.source_url" /></n-form-item>
      <n-form-item label="正文">
        <n-input v-model:value="draft.content" type="textarea" :autosize="{ minRows: 8, maxRows: 18 }" placeholder="Markdown 或纯文本；保存后自动切块进向量索引" />
      </n-form-item>
      <n-space v-if="editMode === 'edit'">
        <span class="sw-label">审核通过</span><n-switch v-model:value="draft.is_reviewed" />
        <span class="sw-label">启用</span><n-switch v-model:value="draft.is_active" />
      </n-space>
    </n-form>
    <template #footer>
      <n-space justify="end">
        <n-button @click="editVisible = false">取消</n-button>
        <n-button type="primary" :loading="saving" @click="saveDoc">保存</n-button>
      </n-space>
    </template>
  </n-modal>
</template>

<style scoped>
.kb-filter { display: flex; gap: 8px; margin-bottom: 12px; align-items: center; }
.kb-filter .spacer { flex: 1; }
.kb-progress { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; font-size: 12px; color: var(--imcs-text-secondary, #6b7280); }
.kb-progress :deep(.n-progress) { width: 240px; }
.kb-search { display: flex; gap: 10px; margin-bottom: 14px; }
.diag { font-size: 12.5px; color: #6b7280; margin-bottom: 12px; display: flex; gap: 8px; align-items: center; }
.hit { border: 1px solid var(--imcs-border, #e5e7eb); border-radius: 4px; padding: 10px 12px; margin-bottom: 8px; background: #fff; }
.hit-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.hit-head strong { font-size: 13.5px; }
.score { font-size: 12px; color: #9ca3af; font-variant-numeric: tabular-nums; }
.page { font-size: 12px; color: #9ca3af; margin-left: auto; }
.hit-body { font-size: 12.5px; color: #4b5563; line-height: 1.7; white-space: pre-wrap; max-height: 120px; overflow: auto; }
.det-desc { margin-bottom: 10px; }
.det-content { white-space: pre-wrap; font-family: inherit; font-size: 12.5px; line-height: 1.75; max-height: 420px; overflow: auto; background: #fafafa; border: 1px solid var(--imcs-border, #e5e7eb); border-radius: 4px; padding: 12px; margin: 0; }
.sw-label { font-size: 13px; color: #6b7280; align-self: center; margin-right: -4px; }
.clickable { color: var(--imcs-brand, #1B5E7B); cursor: pointer; }
.clickable:hover { text-decoration: underline; }
</style>
