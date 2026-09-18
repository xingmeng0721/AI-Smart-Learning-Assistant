<script setup>
import { ref } from 'vue'
import { agentQuery } from '../api.js'

const input = ref('')
const maxSteps = ref(5)
const busy = ref(false)
const result = ref(null)
const error = ref('')

async function run() {
  const q = input.value.trim()
  if (!q || busy.value) return
  busy.value = true
  error.value = ''
  result.value = null
  try {
    result.value = await agentQuery({ q, max_steps: maxSteps.value })
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div class="agent">
    <h2>Agent Tool Use 问答</h2>
    <p class="desc">
      由 LLM 决策调用工具（笔记查询 / 资料检索 / 章节跳转）后再作答。
    </p>

    <div class="row">
      <textarea v-model="input" placeholder="输入问题，例如「查一下栈的概念」" @keydown.enter.exact.prevent="run"></textarea>
      <div class="controls">
        <label>max_steps <input type="number" v-model.number="maxSteps" min="1" max="10" /></label>
        <button :disabled="busy || !input.trim()" @click="run">{{ busy ? '运行中…' : '运行' }}</button>
      </div>
    </div>

    <p v-if="error" class="error">请求失败：{{ error }}</p>

    <div v-if="result" class="result">
      <div class="block">
        <div class="block-title">最终答案</div>
        <pre class="answer">{{ result.answer }}</pre>
      </div>
      <div class="block" v-if="result.steps && result.steps.length">
        <div class="block-title">执行步骤（{{ result.steps.length }}）</div>
        <div v-for="(s, i) in result.steps" :key="i" class="step">
          <div class="step-head">Step {{ i + 1 }} · 调用 <code>{{ s.tool_name }}</code></div>
          <pre>{{ JSON.stringify(s.tool_args, null, 2) }}</pre>
          <div class="obs">{{ s.observation }}</div>
        </div>
      </div>
      <div class="block" v-if="result.truncated">
        <div class="block-title warn">⚠ 已达最大步数，回答可能不完整</div>
      </div>
    </div>

    <p v-else-if="!busy" class="empty">运行后此处展示答案与工具调用轨迹。</p>
  </div>
</template>

<style scoped>
.desc { color: #6b7280; }
.row { display: flex; flex-direction: column; gap: 8px; margin: 12px 0; }
.row textarea { min-height: 54px; padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; font-family: inherit; font-size: 14px; resize: none; }
.controls { display: flex; align-items: center; gap: 12px; }
.controls label { display: flex; align-items: center; gap: 6px; font-size: 13px; }
.controls input { width: 52px; padding: 4px; }
.controls button { padding: 8px 18px; border: none; border-radius: 8px; background: #3b82f6; color: #fff; cursor: pointer; }
.controls button:disabled { background: #b0c4e8; cursor: not-allowed; }
.error { color: #d33; }
.result { background: #fff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 16px; }
.block { margin-bottom: 14px; }
.block-title { font-weight: 600; margin-bottom: 6px; }
.block-title.warn { color: #b54708; }
.answer { white-space: pre-wrap; background: #f8fafc; border-radius: 6px; padding: 10px; line-height: 1.6; }
.step { border: 1px solid #eef0f3; border-radius: 6px; padding: 8px; margin-bottom: 8px; background: #fbfcfd; }
.step-head { font-size: 13px; margin-bottom: 4px; }
.step code { background: #eef2ff; padding: 1px 5px; border-radius: 4px; }
.step pre { font-size: 12px; margin: 4px 0; background: #f1f4f7; padding: 6px; border-radius: 4px; white-space: pre-wrap; }
.obs { font-size: 13px; color: #374151; background: #f1f4f7; padding: 6px; border-radius: 4px; }
.empty { color: #89909a; }
</style>