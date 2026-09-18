<script setup>
import { onMounted, ref } from 'vue'
import { listDocuments, getDocument, deleteDocument, ingest, uploadFile } from '../api.js'

const docs = ref([])
const msg = ref('')
const err = ref('')
const selected = ref(null) // 详情面板展开的文档
const detail = ref(null)

const formDocId = ref('')
const formText = ref('')
const ingesting = ref(false)

// 文件上传
const fileRef = ref(null)
const uploadDocId = ref('')
const uploading = ref(false)

async function load() {
  try {
    docs.value = (await listDocuments()).docs
    err.value = ''
  } catch (e) {
    err.value = e.message
  }
}

async function showDetail(doc) {
  selected.value = doc
  try {
    detail.value = await getDocument(doc.doc_id)
    err.value = ''
  } catch (e) {
    err.value = e.message
  }
}

async function remove(doc) {
  if (!window.confirm(`确认删除文档「${doc.title}」(${doc.chunks} chunks)？`)) return
  try {
    await deleteDocument(doc.doc_id)
    if (selected.value && selected.value.doc_id === doc.doc_id) {
      selected.value = null
      detail.value = null
    }
    msg.value = `已删除 ${doc.doc_id}`
    await load()
  } catch (e) {
    err.value = e.message
  }
}

async function submitIngest() {
  const docId = formDocId.value.trim()
  const text = formText.value.trim()
  if (!docId || !text || ingesting.value) return
  ingesting.value = true
  msg.value = ''
  try {
    const r = await ingest(docId, text)
    msg.value = `导入成功：${docId}，${r.chunks} chunks，共 ${r.total}`
    formDocId.value = ''
    formText.value = ''
    await load()
  } catch (e) {
    err.value = e.message
  } finally {
    ingesting.value = false
  }
}

async function onFileChange(ev) {
  const file = ev.target.files && ev.target.files[0]
  if (!file || uploading.value) return
  uploading.value = true
  msg.value = ''
  try {
    const r = await uploadFile(file, uploadDocId.value.trim())
    msg.value = `上传成功：${file.name} → ${r.doc_id}，${r.chunks} chunks，共 ${r.total}`
    uploadDocId.value = ''
    await load()
  } catch (e) {
    err.value = e.message
  } finally {
    uploading.value = false
    if (fileRef.value) fileRef.value.value = ''
  }
}

onMounted(load)
</script>

<template>
  <div>
    <h2>知识库管理</h2>

    <div class="card">
      <h3>上传文件（Markdown / PDF / DOCX / TXT）</h3>
      <div class="form">
        <input v-model="uploadDocId" placeholder="doc_id（可选，默认取文件名）" />
        <input ref="fileRef" type="file" accept=".md,.markdown,.pdf,.docx,.txt" @change="onFileChange" />
        <button :disabled="uploading">{{ uploading ? '上传中…' : '上传' }}</button>
      </div>
    </div>

    <div class="card">
      <h3>导入文档（Markdown）</h3>
      <div class="form">
        <input v-model="formDocId" placeholder="doc_id（唯一标识）" />
        <textarea v-model="formText" placeholder="粘贴 Markdown 正文，一级标题将作为文档标题"></textarea>
        <button :disabled="ingesting || !formDocId.trim() || !formText.trim()" @click="submitIngest">
          {{ ingesting ? '导入中…' : '导入' }}
        </button>
      </div>
    </div>

    <p v-if="msg" class="ok">{{ msg }}</p>
    <p v-if="err" class="err">{{ err }}</p>
    <button class="refresh" @click="load">刷新列表</button>

    <div class="docs" v-if="docs.length">
      <div class="doc" v-for="d in docs" :key="d.doc_id">
        <div class="doc-main">
          <div class="doc-title">{{ d.title }}</div>
          <div class="doc-sub">{{ d.doc_id }} · {{ d.chunks }} chunks</div>
        </div>
        <div class="doc-actions">
          <button @click="showDetail(d)">查看分块</button>
          <button class="danger" @click="remove(d)">删除</button>
        </div>
      </div>
    </div>
    <p v-else class="empty">知识库为空，先导入文档。</p>

    <div class="card" v-if="detail">
      <h3>{{ selected && selected.title }} — 共 {{ detail.total }} 个分块</h3>
      <div class="chunk" v-for="(c, i) in detail.chunks" :key="c.chunk_id">
        <div class="chunk-head">{{ c.section_path || '（无章节）' }} <span class="cid">{{ c.chunk_id }}</span></div>
        <pre class="chunk-text">{{ c.text }}</pre>
      </div>
    </div>
  </div>
</template>

<style scoped>
.card { background: #fff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 14px; margin: 12px 0; }
.card h3 { margin-top: 0; }
.form { display: flex; gap: 8px; align-items: flex-start; }
.form input { flex: 1; padding: 8px; border: 1px solid #d1d5db; border-radius: 6px; }
.form textarea { flex: 2; min-height: 70px; padding: 8px; border: 1px solid #d1d5db; border-radius: 6px; resize: none; font-family: inherit; }
.form button { padding: 8px 18px; border: none; border-radius: 6px; background: #3b82f6; color: #fff; cursor: pointer; }
.form button:disabled { background: #b0c4e8; cursor: not-allowed; }
.ok { color: #15803d; }
.err { color: #d33; }
.refresh { margin-left: 8px; padding: 6px 12px; border: 1px solid #d1d5db; border-radius: 6px; background: #fff; cursor: pointer; }
.docs { display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }
.doc { display: flex; justify-content: space-between; align-items: center; background: #fff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px 14px; }
.doc-title { font-weight: 600; }
.doc-sub { color: #6b7280; font-size: 12px; margin-top: 3px; }
.doc-actions { display: flex; gap: 6px; }
.doc-actions button { padding: 6px 12px; border: 1px solid #d1d5db; border-radius: 6px; background: #fff; cursor: pointer; }
.doc-actions button.danger { color: #d33; border-color: #f3c1c1; }
.empty { color: #89909a; }
.chunk { border: 1px solid #eef0f3; border-radius: 6px; padding: 10px; margin-bottom: 8px; background: #fbfcfd; }
.chunk-head { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
.chunk-head .cid { color: #9ca3af; font-weight: 400; font-size: 11px; margin-left: 8px; }
.chunk-text { white-space: pre-wrap; font-family: inherit; font-size: 13px; line-height: 1.6; margin: 0; color: #374151; }
</style>