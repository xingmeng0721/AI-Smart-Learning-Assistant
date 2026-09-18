<script setup>
import { onMounted, ref } from 'vue'
import { getHealth } from './api.js'
import ChatView from './views/ChatView.vue'
import AgentView from './views/AgentView.vue'
import BooksView from './views/BooksView.vue'
import SessionsView from './views/SessionsView.vue'
import { activeTab } from './store.js'

const tabs = [
  { key: 'chat', label: '对话' },
  { key: 'agent', label: 'Agent' },
  { key: 'books', label: '知识库' },
  { key: 'sessions', label: '会话管理' },
]
const health = ref(null)

onMounted(async () => {
  try {
    health.value = await getHealth()
  } catch (e) {
    health.value = null
  }
})
</script>

<template>
  <div class="layout">
    <aside class="sidebar">
      <h1 class="brand">AI 智能学习助教</h1>
      <nav>
        <button
          v-for="t in tabs"
          :key="t.key"
          :class="{ active: activeTab === t.key }"
          @click="activeTab = t.key"
        >
          {{ t.label }}
        </button>
      </nav>
      <div class="health" v-if="health">
        <div>状态：<b>{{ health.status }}</b></div>
        <div>chunks：{{ health.chunks }}</div>
        <div>mode：{{ health.mode }}</div>
        <div class="hint">LLM:{{ health.llm }} · Rerank:{{ health.reranker }}</div>
      </div>
      <div class="health" v-else>后端未连接</div>
    </aside>

    <main class="content">
      <ChatView v-if="activeTab === 'chat'" />
      <AgentView v-else-if="activeTab === 'agent'" />
      <BooksView v-else-if="activeTab === 'books'" />
      <SessionsView v-else />
    </main>
  </div>
</template>

<style>
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #f5f6f8; color: #1f2329; }
.layout { display: flex; height: 100vh; }
.sidebar { width: 220px; background: #20303f; color: #fff; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.sidebar .brand { font-size: 16px; margin: 0 0 8px; }
.sidebar nav { display: flex; flex-direction: column; gap: 6px; }
.sidebar nav button { text-align: left; padding: 9px 12px; border: none; border-radius: 6px; background: transparent; color: #cdd6df; cursor: pointer; font-size: 14px; }
.sidebar nav button:hover { background: rgba(255,255,255,0.08); }
.sidebar nav button.active { background: #3b82f6; color: #fff; }
.health { margin-top: auto; background: rgba(255,255,255,0.08); padding: 10px; border-radius: 6px; font-size: 12px; line-height: 1.6; }
.health .hint { color: #9fb0bf; font-size: 11px; }
.content { flex: 1; overflow: auto; padding: 20px; }
h2 { margin-top: 0; }
</style>