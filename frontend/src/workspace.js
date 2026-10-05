// Tab-scoped navigation only. Never store document text, answers or passwords.
const prefix = 'rag-workspace:'

export function readWorkspace(storage, user) {
  try {
    const value = JSON.parse(storage.getItem(prefix + user))
    if (!value || typeof value.kb_id !== 'string') return null
    if (![value.application_id, value.conversation_id].every(v => v === null || typeof v === 'string')) return null
    return value
  } catch { return null }
}

export function saveWorkspace(storage, user, kb, application, conversation) {
  if (!user || !kb) return
  storage.setItem(prefix + user, JSON.stringify({ kb_id: kb, application_id: application || null, conversation_id: conversation || null }))
}

export function clearWorkspace(storage, user) { storage.removeItem(prefix + user) }

export function restoredConversation(saved, kb, application, conversations) {
  if (saved?.kb_id === kb && (saved.application_id || '') === (application || '')) {
    const previous = conversations.find(item => item.id === saved.conversation_id)
    if (previous) return previous.id
  }
  return conversations.find(item => item.active_run_id)?.id || null
}
