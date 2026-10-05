import test from 'node:test'
import assert from 'node:assert/strict'
import { markInterruptedSteps, stepStatusText } from '../src/run_state.js'

test('explicit cancellation settles only the active node without inventing its duration', () => {
  const input = [{ node: 'generate', status: 'completed', elapsed_ms: 20 }, { node: 'check', status: 'running' }]
  const stopped = markInterruptedSteps(input, 'cancelled')
  assert.equal(stepStatusText(stopped[1]), '已取消')
  assert.equal(stopped[1].elapsed_ms, undefined)
  assert.deepEqual(stopped[0], input[0])
  assert.equal(input[1].status, 'running')
})

test('disconnect does not report a backend cancellation or failure', () => {
  const [step] = markInterruptedSteps([{ node: 'grade', status: 'running' }], 'network')
  assert.equal(step.status, 'interrupted')
  assert.equal(stepStatusText(step), '连接中断，待恢复')
})

test('a persisted failure survives an interrupted subscriber', () => {
  const [step] = markInterruptedSteps([{ node: 'check', status: 'failed', elapsed_ms: 30 }], 'cancelled')
  assert.equal(step.status, 'failed')
  assert.equal(stepStatusText(step), '已失败 · 30 ms')
})

test('restored authoritative steps replace local uncertainty and retain a valid zero duration', () => {
  const transient = markInterruptedSteps([{ node: 'rewrite', status: 'running' }], 'network')
  assert.equal(stepStatusText(transient[0]), '连接中断，待恢复')
  const authoritative = { node: 'rewrite', status: 'completed', elapsed_ms: 0 }
  assert.equal(stepStatusText(authoritative), '已完成 · 0 ms')
  assert.equal(stepStatusText({ status: 'completed' }), '已完成')
})
