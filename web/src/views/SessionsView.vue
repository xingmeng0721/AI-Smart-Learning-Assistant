<script setup>
import { onMounted, ref } from 'vue'
import { listSessions, getSession, clearSession } from '../api.js'
import { openSession } from '../store.js'

const sessions = ref([])
const msg = ref('')
const err = ref('')
const detail = ref(null)

async function load() {
  try {
    sessions.value = (await listSessions()).sessions
    err.value = ''
  } catch (e) {
    err.value = e.message
  }
}

async function show(s) {
  try {
    detail.value = await getSession(s.session_id)
    err.value = ''
  } catch (e) {
    err.value = e.message
  }
}

async function clear(s) {
  if (!window.confirm(`清空会话「${s.session_id}」？`)) return
  try {
    await clearSession(s.session_id)
    if (detail.value && detail.value.session_id === s.session_id) detail.value = null
    msg.value = `已清空 ${s.session_id}`
    await load()
  } catch (e) {
    err.value = e.message
  }
}

onMounted(load)
</script>

<template>
  <div>
    <h2>会话管理</h2>
    <p class="note">多轮会话保存在服务内存中，可查看或清空。</p>
    <p v-if="msg" class="ok">{{ msg }}</p>
    <p v-if="err" class="err">{{ err }}</p>
    <button class="refresh" @click="load">刷新</button>

    <div class="sessions" v-if="sessions.length">
      <div class="session" v-for="s in sessions" :key="s.session_id" @click="show(s)">
        <div class="s-main">
          <div class="s-id">{{ s.session_id }}</div>
          <div class="s-last">{{ s.last }}</div>
        </div>
        <div class="s-meta"><span v-if="s.updated_at">{{ s.updated_at }}</span> · {{ s.turns }} 轮</div>
        <button class="resume" @click.stop="openSession(s.session_id)">继续对话</button>
        <button class="danger" @click.stop="clear(s)">清空</button>
      </div>
    </div>
    <p v-else class="empty">暂无会话记录。</p>

    <div class="card" v-if="detail">
      <h3>{{ detail.session_id }}</h3>
      <div class="m-row" v-for="(m, i) in detail.messages" :key="i" :class="m.role">
        <span class="m-role">{{ m.role }}</span>
        <span>{{ m.content }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.note { color: #6b7280; }
.refresh { padding: 6px 12px; border: 1px solid #d1d5db; border-radius: 6px; background: #fff; cursor: pointer; }
.ok { color: #15803d; }
.err { color: #d33; }
.sessions { display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }
.session { display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 10px 14px; cursor: pointer; }
.s-main { flex: 1; min-width: 0; }
.s-id { font-weight: 600; font-size: 14px; }
.s-last { color: #6b7280; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.s-meta { color: #9ca3af; font-size: 12px; }
.session button.resume { color: #3b82f6; border: 1px solid #bfdbfe; background: #fff; border-radius: 6px; padding: 5px 10px; cursor: pointer; }
.session button.danger { color: #d33; border: 1px solid #f3c1c1; background: #fff; border-radius: 6px; padding: 5px 10px; cursor: pointer; }
.empty { color: #89909a; }
.card { background: #fff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 14px; margin-top: 12px; }
.m-row { display: flex; gap: 8px; padding: 6px 8px; border-bottom: 1px solid #eef0f3; font-size: 14px; }
.m-row.user { background: #eef2ff; }
.m-row.assistant { background: #f1f4f7; }
.m-row .m-role { font-weight: 600; color: #3b82f6; min-width: 64px; }
</style>