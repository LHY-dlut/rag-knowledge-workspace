import test from 'node:test'
import assert from 'node:assert/strict'
import { streamChat } from '../src/api.js'

test('restart outages lasting beyond prior wait window recover with same cursor and run', async () => {
  const original = globalThis.fetch, events = [], urls = []
  let streams = 0
  globalThis.fetch = async (url, options) => {
    urls.push(url)
    assert.notEqual(options.method, 'POST')
    if (url.endsWith('/runs/run')) return new Response('<html>starting</html>', { status: 503 })
    streams++
    if (streams === 1) return new Response('id: run:1\nevent: token\ndata: {"text":"unverified draft"}\n\n')
    if (streams === 2) return new Response('starting', { status: 502 })
    return new Response('id: run:2\nevent: draft_reset\ndata: {}\n\nid: run:3\nevent: done\ndata: {"answer":"verified terminal"}\n\n')
  }
  try {
    await streamChat({ run_id: 'run' }, 'test', (name, data) => events.push([name, data]))
    assert.deepEqual(urls, ['/api/runs/run/events?after=0', '/api/runs/run', '/api/runs/run/events?after=1', '/api/runs/run', '/api/runs/run/events?after=1'])
    assert.equal(events.filter(([name]) => name === 'token').length, 1)
    assert.equal(events.at(-1)[1].answer, 'verified terminal')
  } finally { globalThis.fetch = original }
})

test('cancel during outage backoff stops before another replay or accepted answer', async () => {
  const original = globalThis.fetch, controller = new AbortController(), events = []
  let count = 0, timer
  globalThis.fetch = async url => {
    count++
    if (url.endsWith('/runs/run')) timer = setTimeout(() => controller.abort(), 25)
    return new Response('unavailable', { status: 502 })
  }
  try {
    await assert.rejects(streamChat({ run_id: 'run' }, 'test', name => events.push(name), controller.signal), e => e.name === 'AbortError')
    assert.equal(count, 2)
    assert.ok(!events.includes('done'))
  } finally { clearTimeout(timer); globalThis.fetch = original }
})
