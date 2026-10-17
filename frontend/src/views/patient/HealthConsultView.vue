<script setup lang="ts">
/**
 * 健康咨询（流式打字机）。批次 1 的核心体验页。
 *
 * 交互要点：
 *   - 提问后先显示 metadata 里的引用来源（先给出处，再吐正文）
 *   - token 逐片追加，末尾带闪烁光标（.stream-caret）
 *   - done 后展示用量与免责声明；guard 命中时用警示样式而非普通气泡
 *   - 生成中可「停止」；组件卸载自动 abort，避免连接泄漏
 */
import { ref, nextTick, onBeforeUnmount } from 'vue'
import {
  NInput, NButton, NAlert, NScrollbar, useMessage,
} from 'naive-ui'
import { askHealthQuestion, type ConsultMetadata, type ConsultDone } from '@/api/healthQa'

interface ChatMsg {
  role: 'user' | 'assistant'
  content: string
  // 助手消息的附属信息
  metadata?: ConsultMetadata
  done?: ConsultDone
  streaming?: boolean
  error?: string
}

const message = useMessage()
const question = ref('')
const sending = ref(false)
const messages = ref<ChatMsg[]>([])
const scrollRef = ref<InstanceType<typeof NScrollbar> | null>(null)
let abortFn: (() => void) | null = null

async function scrollToBottom() {
  await nextTick()
  const el = scrollRef.value?.$el?.querySelector('.n-scrollbar-container') as HTMLElement | null
  if (el) el.scrollTop = el.scrollHeight
}

function buildHistory(): { role: 'user' | 'assistant'; content: string }[] {
  // 只取最近 3 轮（与后端 MAX_HISTORY_TURNS 对齐），不含正在生成的这条
  const done = messages.value.filter((m) => !m.streaming && !m.error)
  return done.slice(-6).map((m) => ({ role: m.role, content: m.content }))
}

async function send() {
  const q = question.value.trim()
  if (!q) {
    message.warning('请输入你的健康问题')
    return
  }
  if (q.length < 2) {
    message.warning('问题至少 2 个字')
    return
  }
  if (sending.value) return

  const history = buildHistory()
  messages.value.push({ role: 'user', content: q })
  const assistant = ref<ChatMsg>({ role: 'assistant', content: '', streaming: true })
  messages.value.push(assistant.value)
  question.value = ''
  sending.value = true
  scrollToBottom()

  abortFn = askHealthQuestion(q, history, {
    onMetadata: (data) => {
      assistant.value.metadata = data as ConsultMetadata
      scrollToBottom()
    },
    onToken: (text) => {
      assistant.value.content += text
      scrollToBottom()
    },
    onDone: (data) => {
      assistant.value.done = data as ConsultDone
      assistant.value.streaming = false
      sending.value = false
      scrollToBottom()
    },
    onError: (data) => {
      assistant.value.streaming = false
      assistant.value.error = data.message || '回答生成失败'
      sending.value = false
      scrollToBottom()
    },
  })
}

function stop() {
  abortFn?.()
  const last = messages.value[messages.value.length - 1]
  if (last && last.streaming) {
    last.streaming = false
    if (!last.content) last.error = '已停止生成'
  }
  sending.value = false
}

onBeforeUnmount(() => {
  abortFn?.()
})

function guardType(m: ChatMsg): 'error' | 'warning' | null {
  const g = m.metadata?.guard
  if (g === 'crisis') return 'error'
  if (g === 'emergency') return 'warning'
  return null
}
</script>

<template>
  <div class="consult">
    <n-scrollbar ref="scrollRef" class="consult__scroll">
      <div class="consult__body">
        <div v-if="messages.length === 0" class="consult__empty">
          <div class="consult__empty-title">健康咨询</div>
          <p class="consult__empty-desc">
            基于院内知识库回答常见健康问题。以下为示例，点击可直接提问：
          </p>
          <div class="consult__samples">
            <n-button
              v-for="s in ['高血压日常怎么调理？', '布洛芬有什么副作用？', '体检前需要注意什么？']"
              :key="s"
              size="small"
              secondary
              @click="question = s"
            >
              {{ s }}
            </n-button>
          </div>
          <n-alert type="info" :bordered="false" class="consult__disclaimer-static">
            本服务仅供健康信息参考，不能替代医生的诊断与面诊。如有急重症状请立即就医或拨打 120。
          </n-alert>
        </div>

        <div v-for="(m, i) in messages" :key="i" class="msg" :class="`msg--${m.role}`">
          <div class="msg__role">{{ m.role === 'user' ? '我' : '咨询助手' }}</div>

          <!-- 用户消息 -->
          <div v-if="m.role === 'user'" class="msg__bubble msg__bubble--user">{{ m.content }}</div>

          <!-- 助手消息 -->
          <template v-else>
            <!-- 引用来源：先于正文渲染 -->
            <div v-if="m.metadata?.references?.length" class="msg__refs">
              <span class="msg__refs-label">参考来源：</span>
              <n-tag
                v-for="r in m.metadata.references"
                :key="r.chunk_id"
                size="small"
                :bordered="false"
                class="msg__ref"
              >
                《{{ r.title }}》
              </n-tag>
            </div>

            <!-- 急症 / 危机：警示样式 -->
            <n-alert
              v-if="guardType(m)"
              :type="guardType(m)!"
              :bordered="false"
              class="msg__guard"
            >
              <div class="msg__guard-text">{{ m.content }}</div>
            </n-alert>

            <!-- 普通回答 -->
            <div
              v-else
              class="msg__bubble msg__bubble--assistant"
              :class="{ 'stream-caret': m.streaming }"
            >{{ m.content }}<span v-if="m.streaming && !m.content">正在思考…</span></div>

            <!-- 错误 -->
            <n-alert v-if="m.error" type="error" :bordered="false" class="msg__error">
              {{ m.error }}
            </n-alert>

            <!-- 结束信息：用量 + 免责声明 -->
            <div v-if="m.done" class="msg__footer">
              <span class="msg__meta num">
                {{ m.done.usage.prompt_tokens + m.done.usage.completion_tokens }} tokens
                · {{ m.done.latency_ms }}ms
                <template v-if="m.done.degradation"> · 降级：{{ m.done.degradation }}</template>
              </span>
              <div class="msg__disclaimer">{{ m.done.disclaimer }}</div>
            </div>
          </template>
        </div>
      </div>
    </n-scrollbar>

    <div class="consult__input">
      <n-input
        v-model:value="question"
        type="textarea"
        placeholder="描述你的健康问题，例如「高血压日常怎么调理」"
        :autosize="{ minRows: 1, maxRows: 4 }"
        :disabled="sending"
        @keydown.enter.exact.prevent="send"
      />
      <div class="consult__actions">
        <n-button v-if="sending" secondary @click="stop">停止</n-button>
        <n-button type="primary" :loading="sending" :disabled="!question.trim()" @click="send">
          发送
        </n-button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.consult {
  display: flex;
  flex-direction: column;
  height: 100%;
}
.consult__scroll {
  flex: 1;
  min-height: 0;
}
.consult__body {
  padding: var(--space-5);
  max-width: 860px;
  margin: 0 auto;
}
.consult__empty {
  padding: var(--space-12) 0;
  text-align: center;
}
.consult__empty-title {
  font-size: var(--font-2xl);
  font-weight: 600;
  color: var(--gray-9);
}
.consult__empty-desc {
  color: var(--gray-7);
  font-size: var(--font-sm);
  margin: var(--space-3) 0 var(--space-4);
}
.consult__samples {
  display: flex;
  gap: var(--space-2);
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: var(--space-6);
}
.consult__disclaimer-static {
  max-width: 560px;
  margin: 0 auto;
  text-align: left;
}
.msg {
  margin-bottom: var(--space-5);
}
.msg__role {
  font-size: var(--font-xs);
  color: var(--gray-7);
  margin-bottom: var(--space-1);
}
.msg--user {
  text-align: right;
}
.msg__bubble {
  display: inline-block;
  max-width: 78%;
  text-align: left;
  padding: var(--space-3) var(--space-4);
  border-radius: var(--radius-md);
  font-size: var(--font-base);
  line-height: var(--leading-base);
  white-space: pre-wrap;
  word-break: break-word;
}
.msg__bubble--user {
  background: var(--brand-6);
  color: #fff;
}
.msg__bubble--assistant {
  background: var(--gray-1);
  border: 1px solid var(--border-color-light);
  color: var(--gray-8);
}
.msg__refs {
  margin-bottom: var(--space-2);
  font-size: var(--font-xs);
  text-align: left;
}
.msg__refs-label {
  color: var(--gray-7);
  margin-right: var(--space-1);
}
.msg__ref {
  margin-right: var(--space-1);
}
.msg__guard {
  max-width: 78%;
  text-align: left;
}
.msg__guard-text {
  white-space: pre-wrap;
  line-height: var(--leading-base);
}
.msg__error {
  max-width: 78%;
  margin-top: var(--space-2);
}
.msg__footer {
  margin-top: var(--space-2);
  text-align: left;
  max-width: 78%;
}
.msg__meta {
  font-size: var(--font-xs);
  color: var(--gray-7);
}
.msg__disclaimer {
  font-size: var(--font-xs);
  color: var(--gray-7);
  margin-top: var(--space-1);
  line-height: 1.5;
}
.consult__input {
  border-top: 1px solid var(--border-color-light);
  background: var(--gray-1);
  padding: var(--space-3) var(--space-5);
}
.consult__actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-2);
  margin-top: var(--space-2);
  max-width: 860px;
  margin-left: auto;
  margin-right: auto;
}
</style>
