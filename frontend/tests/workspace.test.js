import test from 'node:test'
import assert from 'node:assert/strict'
import { readWorkspace, saveWorkspace, clearWorkspace, restoredConversation } from '../src/workspace.js'

function storage() {
  const values = new Map()
  return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) }
}

test('refresh restores the completed conversation after the offline run unlocked it', () => {
  const session = storage()
  saveWorkspace(session, 'alice', 'kb', '', 'conversation')
  const saved = readWorkspace(session, 'alice')
  assert.equal(restoredConversation(saved, 'kb', '', [{ id: 'conversation', active_run_id: '' }]), 'conversation')
  assert.equal(readWorkspace(session, 'bob'), null)
  clearWorkspace(session, 'alice')
  assert.equal(readWorkspace(session, 'alice'), null)
})

test('a different knowledge base or application cannot inherit the remembered conversation', () => {
  const saved = { kb_id: 'kb', application_id: 'app', conversation_id: 'private' }
  assert.equal(restoredConversation(saved, 'other', 'app', [{ id: 'private' }]), null)
  assert.equal(restoredConversation(saved, 'kb', 'other', [{ id: 'private' }]), null)
})

test('new conversation and stale selection do not reopen unrelated completed history', () => {
  const session = storage()
  saveWorkspace(session, 'alice', 'kb', '', null)
  assert.equal(restoredConversation(readWorkspace(session, 'alice'), 'kb', '', [{ id: 'old' }]), null)
  assert.equal(restoredConversation({ kb_id: 'kb', application_id: null, conversation_id: 'deleted' }, 'kb', '', [{ id: 'old' }]), null)
})

test('malformed session navigation cannot break login or override authorized active history', () => {
  const session = storage()
  session.setItem('rag-workspace:alice', 'invalid')
  assert.equal(readWorkspace(session, 'alice'), null)
  assert.equal(restoredConversation(null, 'kb', '', [{ id: 'active', active_run_id: 'run' }]), 'active')
})
