// 后端 API 封装：普通 JSON 请求 + SSE 流式问答。

async function j(url, options = {}) {
  const resp = await fetch(url, options)
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`
    try {
      const err = await resp.json()
      detail = err.detail || detail
    } catch (e) {
      /* ignore */
    }
    throw new Error(detail)
  }
  return resp.json()
}

export async function getHealth() {
  return j('/health')
}

export async function listDocuments() {
  return j('/documents')
}

export async function getDocument(docId) {
  return j(`/doc/${encodeURIComponent(docId)}`)
}

export async function deleteDocument(docId) {
  return j(`/doc/${encodeURIComponent(docId)}`, { method: 'DELETE' })
}

export async function ingest(docId, text) {
  return j('/ingest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ doc_id: docId, text }),
  })
}

export async function uploadFile(file, docId = '') {
  const form = new FormData()
  form.append('file', file)
  const url = docId ? `/upload?doc_id=${encodeURIComponent(docId)}` : '/upload'
  const resp = await fetch(url, { method: 'POST', body: form })
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`
    try {
      const err = await resp.json()
      detail = err.detail || detail
    } catch (e) {
      /* ignore */
    }
    throw new Error(detail)
  }
  return resp.json()
}

export async function listSessions() {
  return j('/sessions')
}

export async function getSession(sessionId) {
  return j(`/session/${encodeURIComponent(sessionId)}`)
}

export async function clearSession(sessionId) {
  return j(`/session/${encodeURIComponent(sessionId)}`, { method: 'DELETE' })
}

export async function agentQuery({ q, max_steps } = {}) {
  return j(`/agent/query?q=${encodeURIComponent(q || '')}&max_steps=${max_steps || 5}`)
}

// 流式问答：POST /query，解析 SSE 事件，逐个 yield {type, ...}
export async function* querySSE(params, signal) {
  const url = '/query?' + new URLSearchParams(params)
  const resp = await fetch(url, { method: 'POST', signal })
  if (!resp.ok || !resp.body) {
    throw new Error('HTTP ' + resp.status)
  }
  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let idx
    while ((idx = buf.indexOf('\n\n')) !== -1) {
      const block = buf.slice(0, idx)
      buf = buf.slice(idx + 2)
      for (const line of block.split('\n')) {
        if (line.startsWith('data: ')) {
          try {
            yield JSON.parse(line.slice(6))
          } catch (e) {
            /* 跳过无法解析的事件 */
          }
        }
      }
    }
  }
}