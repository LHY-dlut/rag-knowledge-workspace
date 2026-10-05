import test from 'node:test'
import assert from 'node:assert/strict'
import { api, streamChat } from '../src/api.js'

test('non-JSON proxy outage retains HTTP status without exposing HTML', async () => {
  const original = globalThis.fetch
  globalThis.fetch = async () => new Response('<html>private proxy details</html>', { status: 502 })
  try { await assert.rejects(api('/runs/run'), e => e.status === 502 && e.message === 'HTTP 502') }
  finally { globalThis.fetch = original }
})

test('invalid successful JSON remains an error, never invented successful data', async () => {
  const original = globalThis.fetch
  globalThis.fetch = async () => new Response('<html>unexpected</html>')
  try { await assert.rejects(api('/runs/run'), e => e.status === 200 && e.message.includes('格式无效')) }
  finally { globalThis.fetch = original }
})

test('state lookup outage reconnects from cursor with no resubmission or duplicate token', async () => {
  const original = globalThis.fetch, urls = [], events = []
  let streams = 0
  globalThis.fetch = async (url, options) => {
    urls.push([url, options.method || 'GET'])
    assert.notEqual(options.method, 'POST')
    if (url.endsWith('/runs/run')) return new Response('<html>Bad Gateway</html>', { status: 502 })
    streams++
    return new Response(streams === 1
      ? 'id: run:1\nevent: token\ndata: {"text":"draft"}\n\n'
      : 'id: run:1\nevent: token\ndata: {"text":"draft"}\n\nid: run:2\nevent: draft_reset\ndata: {}\n\nid: run:3\nevent: done\ndata: {"answer":"verified","rejected":false}\n\n')
  }
  try {
    await streamChat({ run_id: 'run' }, 'test', (name, data) => events.push([name, data]))
    assert.deepEqual(urls.map(([url]) => url), ['/api/runs/run/events?after=0', '/api/runs/run', '/api/runs/run/events?after=1'])
    assert.equal(events.filter(([name]) => name === 'token').length, 1)
    assert.equal(events.at(-1)[1].answer, 'verified')
  } finally { globalThis.fetch = original }
})

test('expired login state is not retried as an outage', async () => {
  const original = globalThis.fetch, urls = []
  globalThis.fetch = async url => {
    urls.push(url)
    return url.includes('/events') ? new Response('') : Response.json({ error: { message: '登录已过期' } }, { status: 401 })
  }
  try {
    await assert.rejects(streamChat({ run_id: 'run' }, 'test', () => {}), e => e.status === 401)
    assert.equal(urls.length, 2)
  } finally { globalThis.fetch = original }
})

test('timed-out terminal record never becomes done or a background-running promise', async () => {
  const original = globalThis.fetch, events = [], urls = []
  globalThis.fetch = async url => {
    urls.push(url)
    return url.includes('/events') ? new Response('') : Response.json({ data: { status: 'timed_out', error: '任务超时', answer: '' } })
  }
  try {
    await assert.rejects(streamChat({ run_id: 'run' }, 'test', (name) => events.push(name)), /任务超时/)
    assert.ok(!events.includes('done'))
    assert.equal(urls.length, 2)
  } finally { globalThis.fetch = original }
})

test('abort of state read propagates without retry or accepted answer', async () => {
  const original = globalThis.fetch, events = [], controller = new AbortController()
  let count = 0
  globalThis.fetch = async url => {
    count++
    if (url.includes('/events')) return new Response('')
    controller.abort()
    throw new DOMException('cancelled', 'AbortError')
  }
  try {
    await assert.rejects(streamChat({ run_id: 'run' }, 'test', name => events.push(name), controller.signal), e => e.name === 'AbortError')
    assert.equal(count, 2)
    assert.ok(!events.includes('done'))
  } finally { globalThis.fetch = original }
})

test('persistent outage stops at four read attempts without submitting or fabricating done', async () => {
  const original = globalThis.fetch, events = [], urls = []
  globalThis.fetch = async (url, options) => {
    assert.notEqual(options.method, 'POST')
    urls.push(url)
    return new Response('<html>unavailable</html>', { status: 503 })
  }
  try {
    await assert.rejects(streamChat({ run_id: 'run' }, 'test', name => events.push(name)), /连接中断，后台继续执行/)
    assert.equal(urls.filter(url => url.includes('/events')).length, 4)
    assert.equal(urls.filter(url => url.endsWith('/runs/run')).length, 4)
    assert.ok(!events.includes('done'))
  } finally { globalThis.fetch = original }
})
