import test from 'node:test'
import assert from 'node:assert/strict'
import { api, scopedApi, eventSequence, streamChat } from '../src/api.js'

test('event cursor rejects a different run and malformed sequences', () => {
  assert.equal(eventSequence('run:12', 'run'), 12)
  for (const id of ['other:1', 'run:0', 'run:1.5', 'run:NaN']) {
    assert.throws(() => eventSequence(id, 'run'))
  }
})

test('reconnect resumes the cursor and ignores duplicate frames', async () => {
  const original = globalThis.fetch
  const urls = [], events = []
  let calls = 0
  globalThis.fetch = async url => {
    urls.push(url)
    if (url.endsWith('/runs/run')) return Response.json({ data: { status: 'running' } })
    calls++
    return new Response(calls === 1
      ? 'id: run:1\nevent: token\ndata: {"text":"a"}\n\n'
      : 'id: run:1\nevent: token\ndata: {"text":"a"}\n\nid: run:2\nevent: done\ndata: {"answer":"a"}\n\n')
  }
  try {
    await streamChat({ run_id: 'run' }, 'test', (name, data) => events.push([name, data]))
    assert.equal(events.filter(([name]) => name === 'token').length, 1)
    assert.ok(urls.includes('/api/runs/run/events?after=1'))
    assert.equal(events.at(-1)[0], 'done')
  } finally { globalThis.fetch = original }
})

test('terminal record compensates a lost final event', async () => {
  const original = globalThis.fetch, events = []
  globalThis.fetch = async url => url.includes('/events') ? new Response('')
    : Response.json({ data: { status: 'completed', answer: 'verified', citations: [{ source_id: 'S1' }] } })
  try {
    await streamChat({ run_id: 'run' }, 'test', (name, data) => events.push([name, data]))
    assert.equal(events.at(-1)[1].answer, 'verified')
    assert.equal(events.find(([name]) => name === 'citations')[1].sources[0].source_id, 'S1')
    assert.ok(events.some(([name]) => name === 'draft_reset'))
  } finally { globalThis.fetch = original }
})

test('terminal compensation preserves refusal rather than presenting it as an accepted answer', async () => {
  const original = globalThis.fetch, events = []
  globalThis.fetch = async url => url.includes('/events') ? new Response('')
    : Response.json({ data: { status: 'completed', answer: 'insufficient evidence', citations: [], rejected: true } })
  try {
    await streamChat({ run_id: 'run' }, 'test', (name, data) => events.push([name, data]))
    assert.equal(events.at(-1)[1].rejected, true)
    assert.deepEqual(events.find(([name]) => name === 'citations')[1].sources, [])
  } finally { globalThis.fetch = original }
})

test('a late API response cannot populate the workspace after switching login accounts', async () => {
  const original = globalThis.fetch
  let token = 'alice', release
  globalThis.fetch = () => new Promise(resolve => { release = () => resolve(Response.json({ data: ['alice private kb'] })) })
  try {
    const pending = scopedApi('/knowledge-bases', { getToken: () => token })
    token = 'bob'; release()
    await assert.rejects(pending, error => error.name === 'AbortError')
  } finally { globalThis.fetch = original }
})

test('API errors distinguish expired login from temporary server outage', async () => {
  const original = globalThis.fetch
  try {
    for (const status of [401, 503]) {
      globalThis.fetch = async () => Response.json({ error: { message: 'unavailable' } }, { status })
      await assert.rejects(api('/knowledge-bases'), error => error.status === status)
    }
  } finally { globalThis.fetch = original }
})
