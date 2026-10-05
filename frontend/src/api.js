import { parseSSE } from './sse.js'

export async function api(path, { method = 'GET', body, token = '', signal } = {}) {
  const headers = token ? { Authorization: `Bearer ${token}` } : {}
  const form = body instanceof FormData
  if (body && !form) headers['Content-Type'] = 'application/json'
  const response = await fetch(`/api${path}`, { method, headers,
    body: body ? (form ? body : JSON.stringify(body)) : undefined, signal })
  let data
  try {
    data = await response.json()
  } catch (cause) {
    if (cause.name === 'AbortError') throw cause
    const error = new Error(response.ok ? '服务返回的数据格式无效' : `HTTP ${response.status}`)
    error.status = response.status
    throw error
  }
  if (!response.ok) {
    const error = new Error(data.error?.message || `HTTP ${response.status}`)
    error.status = response.status
    throw error
  }
  return data.data
}

export async function scopedApi(path, { getToken, ...options }) {
  const token = getToken()
  const result = await api(path, { ...options, token })
  if (getToken() !== token) throw new DOMException('登录会话已变化', 'AbortError')
  return result
}

export function eventSequence(id, runId) {
  const prefix = `${runId}:`
  if (!id.startsWith(prefix)) throw new Error('事件不属于当前运行')
  const value = Number(id.slice(prefix.length))
  if (!Number.isSafeInteger(value) || value < 1) throw new Error('无效事件序号')
  return value
}

function waitForRecovery(attempt, signal) {
  if (signal?.aborted) return Promise.reject(new DOMException('已取消恢复', 'AbortError'))
  return new Promise((resolve, reject) => {
    const finish = () => { signal?.removeEventListener('abort', abort); resolve() }
    const timer = setTimeout(finish, 2000 * 2 ** attempt)
    const abort = () => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', abort)
      reject(new DOMException('已取消恢复', 'AbortError'))
    }
    signal?.addEventListener('abort', abort, { once: true })
  })
}

export async function streamChat(body, token, onEvent, signal) {
  const run = body.run_id ? body : await api('/chat/runs', { method: 'POST', body, token, signal })
  onEvent('meta', run)
  let cursor = 0, completed = false
  for (let attempt = 0; attempt < 4 && !completed; attempt++) {
    try {
      const response = await fetch(`/api/runs/${run.run_id}/events?after=${cursor}`, {
        headers: { Authorization: `Bearer ${token}` }, signal,
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      for await (const frame of parseSSE(response.body)) {
        const sequence = eventSequence(frame.id, run.run_id)
        if (sequence <= cursor) continue
        if (sequence !== cursor + 1) throw new Error('事件序号不连续')
        const value = JSON.parse(frame.data)
        cursor = sequence
        onEvent(frame.event, value)
        if (frame.event === 'error') throw new Error(value.message)
        if (frame.event === 'done') completed = true
      }
    } catch (e) {
      if (signal?.aborted) throw e
    }
    if (!completed) {
      let state
      try {
        state = await api(`/runs/${run.run_id}`, { token, signal })
      } catch (error) {
        if (signal?.aborted || error.name === 'AbortError') throw error
        const transient = error instanceof TypeError || [408, 429, 500, 502, 503, 504].includes(error.status)
        if (!transient) throw error
        // The run was already submitted. Retry only replay/state reads within
        // the existing four-attempt limit; never submit or cancel another run.
        if (attempt < 3) await waitForRecovery(attempt, signal)
        continue
      }
      if (state.status === 'completed') {
        onEvent('draft_reset', { reason: '从运行记录恢复' })
        onEvent('citations', { sources: state.citations })
        onEvent('done', { answer: state.answer, run_id: run.run_id, rejected: state.rejected, artifacts: state.artifacts || [] })
        completed = true
      } else if (['failed', 'cancelled', 'timed_out'].includes(state.status)) {
        throw new Error(state.error || '运行已中断')
      } else {
        await new Promise(resolve => setTimeout(resolve, 250 * (attempt + 1)))
      }
    }
  }
  if (!completed) throw new Error('连接中断，后台继续执行；可从历史对话恢复')
}
