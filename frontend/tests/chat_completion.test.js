import test from 'node:test'
import assert from 'node:assert/strict'
import { streamChat, api, scopedApi } from '../src/api.js'
import { streamThenRefresh } from '../src/chat_completion.js'

test('verified answer, refusal flag and citations survive a history outage after done', async () => {
  const original = globalThis.fetch
  for (const rejected of [false, true]) {
    const message = { content: '', citations: [], provisional: true }
    globalThis.fetch = async url => url.includes('/events')
      ? new Response('id: run:1\nevent: citations\ndata: {"sources":[{"source_id":"S1"}]}\n\nid: run:2\nevent: done\ndata: ' + JSON.stringify({ answer: 'verified', rejected }) + '\n\n')
      : Response.json({ error: { message: 'temporary outage' } }, { status: 503 })
    try {
      const error = await streamThenRefresh(
        () => streamChat({ run_id: 'run' }, 'test', (name, data) => {
          if (name === 'citations') message.citations = data.sources
          if (name === 'done') Object.assign(message, { content: data.answer, provisional: false, rejected: data.rejected })
        }), () => api('/conversations?kb_id=kb'))
      assert.equal(error.status, 503)
      assert.equal(message.content, 'verified')
      assert.equal(message.provisional, false)
      assert.equal(message.rejected, rejected)
      assert.deepEqual(message.citations, [{ source_id: 'S1' }])
    } finally { globalThis.fetch = original }
  }
})

test('stream failure does not refresh history or accept draft tokens as an answer', async () => {
  const original = globalThis.fetch
  let refreshCalls = 0, accepted = false
  globalThis.fetch = async url => url.includes('/events')
    ? new Response('id: run:1\nevent: draft_token\ndata: {"text":"unverified draft"}\n\nid: run:2\nevent: error\ndata: {"message":"verification failed"}\n\n')
    : Response.json({ data: { status: 'failed', error: 'verification failed' } })
  try {
    await assert.rejects(streamThenRefresh(
      () => streamChat({ run_id: 'run' }, 'test', name => { if (name === 'done') accepted = true }),
      async () => { refreshCalls++ }), /verification failed/)
    assert.equal(accepted, false)
    assert.equal(refreshCalls, 0)
  } finally { globalThis.fetch = original }
})

test('explicit abort during streaming still propagates and never triggers history refresh', async () => {
  let refreshed = false
  const cancelled = new DOMException('cancelled', 'AbortError')
  await assert.rejects(streamThenRefresh(async () => { throw cancelled }, async () => { refreshed = true }),
    error => error === cancelled)
  assert.equal(refreshed, false)
})

test('login switch during late history request keeps scoped API isolation', async () => {
  const original = globalThis.fetch
  let token = 'alice', release, state = ['bob only']
  globalThis.fetch = () => new Promise(resolve => { release = () => resolve(Response.json({ data: ['alice private'] })) })
  try {
    const pending = streamThenRefresh(async () => {}, async () => {
      state = await scopedApi('/conversations', { getToken: () => token })
    })
    await new Promise(resolve => setImmediate(resolve))
    token = 'bob'
    release()
    const error = await pending
    assert.equal(error.name, 'AbortError')
    assert.deepEqual(state, ['bob only'])
  } finally { globalThis.fetch = original }
})

test('successful terminal stream refreshes history exactly once', async () => {
  const order = []
  assert.equal(await streamThenRefresh(async () => { order.push('verified done') }, async () => { order.push('history') }), null)
  assert.deepEqual(order, ['verified done', 'history'])
})
