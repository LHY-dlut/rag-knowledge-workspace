import test from 'node:test'
import assert from 'node:assert/strict'
import { parseSSE } from '../src/sse.js'

function stream(parts) {
  return new ReadableStream({ start(controller) {
    for (const part of parts) controller.enqueue(new TextEncoder().encode(part))
    controller.close()
  } })
}

test('SSE handles CRLF boundaries and multiline data', async () => {
  const frames = []
  for await (const frame of parseSSE(stream([
    ': heartbeat\r\n', 'id: 1\r\nevent: token\r', '\ndata: 中文\r\ndata: next\r\n\r', '\n',
    'event: done\ndata: {"ok":true}\n\n',
  ]))) frames.push(frame)
  assert.deepEqual(frames, [
    { event: 'token', data: '中文\nnext', id: '1' },
    { event: 'done', data: '{"ok":true}', id: '1' },
  ])
})

test('SSE ignores an uncommitted truncated frame', async () => {
  const frames = []
  for await (const frame of parseSSE(stream(['event: done\ndata: incomplete']))) frames.push(frame)
  assert.equal(frames.length, 0)
})

test('SSE decodes a Chinese UTF8 character split across bytes', async () => {
  const bytes = new TextEncoder().encode('event: token\ndata: 你\n\n')
  const input = new ReadableStream({ start(c) {
    for (const byte of bytes) c.enqueue(new Uint8Array([byte]))
    c.close()
  } })
  const frames = []
  for await (const frame of parseSSE(input)) frames.push(frame)
  assert.equal(frames[0].data, '你')
})
