<script setup>
import { nextTick, onMounted, ref, watch } from 'vue'
import { querySSE, listSessions, getSession } from '../api.js'
import { pendingSessionId, clearPending } from '../store.js'

const sessionId = ref(newSid())
const input = ref('')
const streaming = ref(false)
const log = ref([])
const sessions = ref([])

// 时间格式会话 id：s-YYYYMMDD-HHMMSS-ms（含毫秒防同一秒内并发冲突）
function newSid() {
  const d = new Date()
  const p = (n, l = 2) => String(n).padStart(l, '0')
  return `s-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}-${p(d.getMilliseconds(), 3)}`
}

// 每条消息：{ role, text, citations, refused, queries }
const messages = ref([])
const chatBox = ref(null)

function scrollToBottom() {
  nextTick(() => {
    if (chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight
  })
}

async function loadSessions() {
  try {
    sessions.value = (await listSessions()).sessions || []
  } catch (e) {
    sessions.value = []
  }
}

async function useSession(id) {
  sessionId.value = id
  messages.value = []
  log.value = []
  try {
    const d = await getSession(id)
    messages.value = (d.messages || []).map((m) => ({
      role: m.role,
      text: m.content,
      citations: [],
      queries: [],
      refused: false,
    }))
  } catch (e) {
    messages.value = []
  }
  scrollToBottom()
}

function newSession() {
  useSession(newSid())
  loadSessions()
}

function onPick(e) {
  const v = e.target.value
  e.target.value = ''
  if (!v) return
  if (v === '__new__') newSession()
  else useSession(v)
}

// 从会话管理跳转过来时载入对应历史会话
watch(pendingSessionId, (id) => {
  if (!id) return
  useSession(id)
  clearPending()
})

onMounted(loadSessions)

function renderMeta(m) {
  if (m.queries && m.queries.length) {
    m.meta = '改写意图：' + m.queries.join('；')
  }
  if (m.refused) m.meta = '已拒绝回答（知识库无足够相关）'
  else if (m.citations && m.citations.length) m.meta = '引用：' + m.citations.join(' → ')
}

async function send() {
  const q = input.value.trim()
  if (!q || streaming.value) return
  input.value = ''
  const userMsg = { role: 'user', text: q }
  messages.value.push(userMsg)
  const botMsg = { role: 'assistant', text: '', citations: [], queries: [], refused: false }
  messages.value.push(botMsg)
  streaming.value = true
  scrollToBottom()
  const ctrl = new AbortController()
  try {
    for await (const ev of querySSE({ session_id: sessionId.value, q }, ctrl.signal)) {
      if (ev.type === 'rewrite') botMsg.queries = ev.queries || []
      else if (ev.type === 'retrieval') log.value.push(`召回 ${ev.count} 段`)
      else if (ev.type === 'citations') botMsg.citations = (ev.citations || []).map((c) => c.text || '')
      else if (ev.type === 'refuse') botMsg.refused = true
      else if (ev.type === 'token') botMsg.text += ev.text
      renderMeta(botMsg)
      scrollToBottom()
    }
  } catch (e) {
    botMsg.text = '请求失败：' + e.message
  } finally {
    streaming.value = false
    renderMeta(botMsg)
    scrollToBottom()
  }
}
</script>

<template>
  <div class="chat">
    <div class="chat-head">
      <h2>对话</h2>
      <select class="picker" @change="onPick">
        <option value="">切换到会话…</option>
        <option value="__new__">＋ 新会话</option>
        <option v-for="s in sessions" :key="s.session_id" :value="s.session_id" :disabled="s.session_id === sessionId">
          {{ s.session_id }}（{{ s.turns }} 轮）
        </option>
      </select>
      <button @click="newSession">新会话</button>
      <span class="sid">session: {{ sessionId }}</span>
    </div>

    <div class="chat-box" ref="chatBox">
      <p v-if="messages.length === 0" class="empty">向知识库提问，例如「什么是注意力机制？」</p>
      <div v-for="(m, i) in messages" :key="i" :class="['msg', m.role]">
        <div class="bubble" v-if="m.role === 'user'">{{ m.text }}</div>
        <template v-else>
          <div class="bubble bot">{{ m.text || (m.refused ? '（未在知识库找到相关资料，已拒绝回答）' : '思考中…') }}</div>
          <div class="meta" v-if="m.meta">{{ m.meta }}</div>
        </template>
      </div>
    </div>

    <div class="chat-input">
      <textarea
        v-model="input"
        placeholder="输入问题，Enter 发送 / Shift+Enter 换行"
        @keydown.enter.exact.prevent="send"
      ></textarea>
      <button :disabled="streaming || !input.trim()" @click="send">
        {{ streaming ? '生成中…' : '发送' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.chat { display: flex; flex-direction: column; height: calc(100vh - 40px); }
.chat-head { display: flex; align-items: center; gap: 12px; }
.picker { padding: 6px 8px; border: 1px solid #d1d5db; border-radius: 6px; background: #fff; max-width: 240px; }
.chat-head .sid { color: #89909a; font-size: 12px; }
.chat-box { flex: 1; overflow: auto; background: #fff; border-radius: 8px; padding: 16px; margin: 12px 0; border: 1px solid #e1e4e8; }
.empty { color: #89909a; text-align: center; margin-top: 40px; }
.msg { margin-bottom: 14px; }
.msg.user { text-align: right; }
.bubble { display: inline-block; max-width: 82%; padding: 9px 12px; border-radius: 10px; white-space: pre-wrap; text-align: left; line-height: 1.6; }
.bubble.bot { background: #f1f4f7; }
.msg.user .bubble { background: #3b82f6; color: #fff; }
.meta { font-size: 11px; color: #6b7280; margin-top: 4px; }
.chat-input { display: flex; gap: 8px; }
.chat-input textarea { flex: 1; resize: none; min-height: 46px; padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; font-family: inherit; font-size: 14px; }
.chat-input button { width: 90px; border: none; border-radius: 8px; background: #3b82f6; color: #fff; cursor: pointer; font-size: 14px; }
.chat-input button:disabled { background: #b0c4e8; cursor: not-allowed; }
</style>