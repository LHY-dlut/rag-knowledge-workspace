<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts/core'
import { BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { api, scopedApi, streamChat } from './api.js'
import { readWorkspace, saveWorkspace, clearWorkspace, restoredConversation } from './workspace.js'
import { markInterruptedSteps, stepStatusText } from './run_state.js'
import { streamThenRefresh } from './chat_completion.js'
import ArtifactCard from './ArtifactCard.vue'
import { mergeArtifacts } from './artifact_view.js'

echarts.use([BarChart, GridComponent, TooltipComponent, CanvasRenderer])
// Tab-scoped session restores login after refresh; passwords are never persisted.
const token = ref(sessionStorage.getItem('rag-token') || ''), username = ref(sessionStorage.getItem('rag-user') || ''), password = ref(''), authBusy = ref(false)
const page = ref('chat'), kbs = ref([]), selectedId = ref(''), docs = ref([]), busy = ref(false)
const newKbName = ref(''), kbDialog = ref(false), health = ref({ provider: 'demo', mode: 'demo' })
const kb = computed(() => kbs.value.find(item => item.id === selectedId.value))
const config = ref({}), chunkDialog = ref(false), chunks = ref([]), editChunk = ref(null)
const query = ref(''), chatMessages = ref([]), conversationId = ref(null), conversations = ref([])
const applications = ref([]), selectedApplication = ref(''), appDialog = ref(false), appForm = ref({})
const activeApplication = computed(() => applications.value.find(a => a.id === selectedApplication.value))
const feedbackItems = ref([]), feedbackDialog = ref(false), feedbackForm = ref({})
const datasets = ref([]), datasetName = ref('手工测评集'), comparison = ref(null)
const presets = ref({}), preset = ref('custom'), debugStrategy = ref('custom')
const modelStatus = ref(null), modelTest = ref(null), modelBusy = ref(false)
const chatMode = ref('agent'), sending = ref(false), draft = ref(''), steps = ref([]), lastRunId = ref('')
const outputType = ref('answer'), artifactBusy = ref('')
const debugQuery = ref(''), debugResult = ref(null), debugBusy = ref(false)
const selectedDocs = ref([]), tagsFilter = ref(''), fileTypes = ref([])
const tools = ref([]), prompt = ref(''), evaluationCases = ref(JSON.stringify([{
  question: '出差报销申请需要几天内提交？', reference_answer: '7天内提交',
  reference_facts: ['7天内提交报销申请'], relevant_document_ids: [],
}], null, 2)), evaluation = ref(null), evaluationHistory = ref([]), evalBusy = ref(false)
const replacementDocument = ref(null), replacementInput = ref(null)
const chartEl = ref(null), traceDialog = ref(false), trace = ref(null), uploadInput = ref(null)
let controller = null, chart = null, poll = null, restoringSelection = false
const filters = computed(() => ({ document_ids: selectedDocs.value, file_types: fileTypes.value,
  tags: tagsFilter.value.split(',').map(t => t.trim()).filter(Boolean) }))
const nav = [
  ['chat', '知识问答', '01'], ['documents', '文档与片段', '02'], ['retrieval', '检索调试', '03'],
  ['strategy', '检索策略', '04'], ['applications', '应用管理', '05'], ['tools', '工具与提示词', '06'],
  ['evaluation', '质量评测', '07'], ['feedback', '对话反馈', '08'], ['models', '模型配置', '09'],
]
const title = computed(() => nav.find(n => n[0] === page.value)?.[1])
const switches = [
  ['query_rewrite', '指代消解', '结合历史补全当前问题'], ['multi_query', '多查询扩展', '使用不同问法扩大召回'],
  ['hyde', 'HyDE', '生成假设文本辅助向量召回'], ['hybrid', '混合检索', '向量与 BM25 双路融合'],
  ['rerank', '语义重排', '对子块候选重新判断相关性'], ['parent_retrieval', '父块回溯', '以完整父块作为回答证据'],
  ['metadata_filter', '元数据过滤', '应用文档、类型与标签条件'],
]
const numeric = [
  ['top_k', '每路召回数', 1, 100], ['candidate_k', '重排候选数', 1, 100], ['context_k', '证据块上限', 1, 20],
  ['rerank_k', '精排保留子块数', 1, 20], ['recursive_size', '递归块字符数', 100, 5000],
  ['recursive_overlap', '递归块重叠', 0, 200], ['short_size', '短片段字符数', 50, 1000], ['short_overlap', '短片段重叠', 0, 200],
  ['rrf_k', 'RRF 常数', 1, 1000], ['context_chars', '上下文字符预算', 500, 30000],
  ['parent_size', '父块字符数', 200, 5000], ['child_size', '子块字符数', 50, 1000],
  ['child_overlap', '子块重叠字符', 0, 200], ['grade_retry_limit', '重新检索次数', 0, 3],
  ['check_retry_limit', '答案重写次数', 0, 2],
]
function request(path, options = {}) { return scopedApi(path, { ...options, getToken: () => token.value }) }
function error(e) { if (e.name !== 'AbortError') ElMessage.error(e.message || String(e)) }
async function auth(register = false) {
  authBusy.value = true
  try {
    const result = await api(`/auth/${register ? 'register' : 'login'}`, {
      method: 'POST', body: { username: username.value, password: password.value },
    })
    token.value = result.access_token; password.value = ''
    sessionStorage.setItem('rag-token', token.value); sessionStorage.setItem('rag-user', username.value)
    await loadKbs()
  } catch (e) { error(e) } finally { authBusy.value = false }
}
async function loadKbs() {
  kbs.value = await request('/knowledge-bases')
  if (!selectedId.value && kbs.value.length) {
    const saved = readWorkspace(sessionStorage, username.value)
    selectedId.value = kbs.value.some(item => item.id === saved?.kb_id) ? saved.kb_id : kbs.value[0].id
  }
}
async function createKb() {
  try {
    const result = await request('/knowledge-bases', { method: 'POST', body: { name: newKbName.value } })
    await loadKbs(); selectedId.value = result.id; kbDialog.value = false; newKbName.value = ''
  } catch (e) { error(e) }
}
async function loadDocs() {
  if (!selectedId.value || !token.value) return
  docs.value = await request(`/knowledge-bases/${selectedId.value}/documents`)
}
async function loadConversations() { conversations.value = await request(`/conversations?kb_id=${selectedId.value}&application_id=${selectedApplication.value}`) }
async function loadApplications() { applications.value = await request(`/applications?kb_id=${selectedId.value}`) }
async function loadFeedback() { feedbackItems.value = await request(`/feedback?kb_id=${selectedId.value}`) }
async function loadDatasets() { datasets.value = await request(`/datasets?kb_id=${selectedId.value}`) }
async function loadModelStatus() { modelStatus.value = await request('/models') }
function applicationDialog(item = null) {
  appForm.value = item ? { ...item } : { kb_id: selectedId.value, name: '', description: '', strategy: 'hybrid', mode: 'rag',
    welcome: '你好，请提出与资料库有关的问题。', fallback: '资料库中暂无足够依据回答这个问题。', prompt: '用中文回答，先给结论，再列依据。', enabled: true }
  appDialog.value = true
}
async function saveApplication() {
  try {
    const { id, ...body } = appForm.value
    await request(id ? `/applications/${id}` : '/applications', { method: id ? 'PUT' : 'POST', body })
    await loadApplications(); appDialog.value = false; ElMessage.success('应用已保存')
  } catch (e) { error(e) }
}
async function feedback(message, helpful) {
  try {
    await request(`/runs/${message.run_id}/feedback`, { method: 'PUT', body: { helpful } })
    message.helpful = helpful; await loadFeedback(); ElMessage.success('反馈已记录')
  } catch (e) { error(e) }
}
function correctFeedback(item) {
  feedbackForm.value = { run_id: item.run_id, name: '反馈人工订正测评集', reference_answer: '', factsText: '[]' }
  feedbackDialog.value = true
}
async function saveFeedbackCase() {
  try {
    const { factsText, ...body } = feedbackForm.value
    await request('/datasets/from-feedback', { method: 'POST', body: { ...body, reference_facts: JSON.parse(factsText) } })
    feedbackDialog.value = false; await loadDatasets(); ElMessage.success('人工标准答案已保存为测评样本')
  } catch (e) { error(e) }
}
function applyPreset(name) { if (presets.value[name]) config.value = { ...config.value, ...presets.value[name] } }
async function testModels() {
  modelBusy.value = true
  try { modelTest.value = await request('/models/test', { method: 'POST' }) } catch (e) { error(e) }
  finally { modelBusy.value = false }
}
async function upload(file, replacement = null) {
  if (!file || !selectedId.value) return
  busy.value = true
  try {
    const body = new FormData(); body.append('file', file)
    const result = await request(replacement ? `/documents/${replacement.id}/file` : `/knowledge-bases/${selectedId.value}/documents`, { method: replacement ? 'PUT' : 'POST', body })
    ElMessage.success(result.duplicate ? '文档已存在' : '上传完成，正在解析与索引')
    await loadDocs()
  } catch (e) { error(e) } finally { busy.value = false; if (uploadInput.value) uploadInput.value.value = ''; if (replacementInput.value) replacementInput.value.value = '' }
}
async function loadSample() {
  const text = '# 公司差旅政策\n\n公司差旅政策规定，员工出差后必须在7天内提交报销申请。\n\n报销需要发票和审批单。住宿报销上限为每晚500元。\n\n# 年假政策\n\n工作满一年的员工享有5天带薪年假。'
  await upload(new File([text], '差旅与休假制度.md', { type: 'text/markdown' }))
}
async function reindex(doc) {
  try { await request(`/documents/${doc.id}/reindex`, { method: 'POST' }); await loadDocs() } catch (e) { error(e) }
}
async function deleteDocument(doc) {
  try { await request(`/documents/${doc.id}`, { method: 'DELETE' }); await loadDocs() } catch (e) { error(e) }
}
async function openChunks(doc) {
  try { chunks.value = await request(`/documents/${doc.id}/chunks`); editChunk.value = null; chunkDialog.value = true }
  catch (e) { error(e) }
}
async function saveChunk() {
  try {
    await request(`/chunks/${editChunk.value.id}`, { method: 'PUT', body: { content: editChunk.value.content } })
    ElMessage.success('修正已提交，正在重新索引'); chunkDialog.value = false; await loadDocs()
  } catch (e) { error(e) }
}
function rememberConversation() { saveWorkspace(sessionStorage, username.value, selectedId.value, selectedApplication.value, conversationId.value) }
function resetConversation() { conversationId.value = null; chatMessages.value = []; steps.value = []; draft.value = ''; lastRunId.value = '' }
function freshConversation() { resetConversation(); rememberConversation() }
async function openConversation(id) {
  if (sending.value) return
  try {
    chatMessages.value = await request(`/conversations/${id}/messages`); conversationId.value = id; steps.value = []
    rememberConversation()
    const last = chatMessages.value.findLast(message => message.role === 'assistant' && message.run_id)
    if (last) {
      const completed = await request(`/runs/${last.run_id}`)
      lastRunId.value = completed.id; steps.value = completed.steps; last.rejected = completed.rejected
    }
    const item = conversations.value.find(c => c.id === id)
    if (item?.active_run_id) {
      const run = await request(`/runs/${item.active_run_id}`)
      await send({ run_id: run.id, conversation_id: id, query: run.query })
    }
  }
  catch (e) { error(e) }
}
async function send(resume = null) {
  if (!resume?.run_id) resume = null
  if ((!resume && !query.value.trim()) || sending.value || !selectedId.value) return
  const text = resume?.query || query.value.trim(); query.value = ''; sending.value = true; draft.value = ''; steps.value = []
  chatMessages.value.push({ role: 'user', content: text })
  const answer = { role: 'assistant', content: '', citations: [], artifacts: [], provisional: true }
  chatMessages.value.push(answer)
  const chatSessionToken = token.value
  controller = new AbortController()
  try {
    const refreshError = await streamThenRefresh(() => streamChat(resume || { kb_id: selectedId.value, query: text, conversation_id: conversationId.value,
      mode: chatMode.value, output_type: outputType.value, application_id: selectedApplication.value || null, filters: filters.value }, token.value, (name, value) => {
      const message = chatMessages.value[chatMessages.value.length - 1]
      if (token.value !== chatSessionToken || !message) return
      if (name === 'meta') { conversationId.value = value.conversation_id; lastRunId.value = value.run_id; message.run_id = value.run_id; rememberConversation() }
      if (name === 'agent_step') {
        const index = steps.value.findIndex(step => step.ordinal === value.ordinal)
        if (index >= 0) steps.value[index] = { ...steps.value[index], ...value }
        else steps.value.push(value)
      }
      if (name === 'draft_start' || name === 'draft_reset') draft.value = ''
      if (name === 'draft_token') draft.value += value.text
      if (name === 'token') message.content += value.text
      if (name === 'citations') message.citations = value.sources
      if (name === 'artifact_ready') message.artifacts = mergeArtifacts(message.artifacts, [value.artifact])
      if (name === 'artifact_unavailable') message.artifactWarning = value.reason
      if (name === 'done') { message.content = value.answer; message.provisional = false; message.rejected = value.rejected; message.artifacts = mergeArtifacts(message.artifacts, value.artifacts || []) }
    }, controller.signal), loadConversations)
    if (refreshError && refreshError.name !== 'AbortError' && token.value === chatSessionToken) {
      ElMessage.warning('回答已完成，历史列表暂未更新；恢复连接后可刷新。')
    }
  } catch (e) {
    if (token.value !== chatSessionToken) return
    const message = chatMessages.value[chatMessages.value.length - 1]
    steps.value = markInterruptedSteps(steps.value, e.name === 'AbortError' ? 'cancelled' : 'interrupted')
    if (message) {
      message.content = e.name === 'AbortError' ? '已停止生成。' : `生成中断：${e.message}`
      message.provisional = false
    }
    if (e.name !== 'AbortError') error(e)
  } finally { sending.value = false; draft.value = ''; controller = null }
}
async function convertArtifact(message, type) {
  if (artifactBusy.value) return
  artifactBusy.value = `${message.run_id}:${type}`
  try {
    const item = await request(`/runs/${message.run_id}/artifacts`, { method: 'POST', body: { type } })
    message.artifacts = mergeArtifacts(message.artifacts, [item])
  } catch (e) { error(e) } finally { artifactBusy.value = '' }
}
async function cancelChat() {
  if (!lastRunId.value) return
  try { await request(`/runs/${lastRunId.value}/cancel`, { method: 'POST' }); controller?.abort() }
  catch (e) { error(e) }
}
function logout() {
  controller?.abort(); token.value = ''; selectedId.value = ''; kbs.value = []
  clearWorkspace(sessionStorage, username.value)
  sessionStorage.removeItem('rag-token'); sessionStorage.removeItem('rag-user')
}
async function debug() {
  debugBusy.value = true
  try { debugResult.value = await request('/retrieval/debug', { method: 'POST',
    body: { kb_id: selectedId.value, query: debugQuery.value, strategy: debugStrategy.value, filters: filters.value } }) }
  catch (e) { error(e) } finally { debugBusy.value = false }
}
async function saveStrategy() {
  try {
    await request(`/knowledge-bases/${selectedId.value}/config`, { method: 'PUT', body: config.value })
    await loadKbs(); ElMessage.success('策略已保存')
  } catch (e) { error(e) }
}
async function loadTools() {
  tools.value = await request('/tools'); prompt.value = (await request('/prompts/generation')).content
}
async function toggleTool(tool) {
  try { await request(`/tools/${tool.name}?enabled=${tool.enabled}`, { method: 'PUT' }) }
  catch (e) { tool.enabled = !tool.enabled; error(e) }
}
async function savePrompt() {
  try { await request('/prompts/generation', { method: 'PUT', body: { content: prompt.value } }); ElMessage.success('提示词已保存') }
  catch (e) { error(e) }
}
async function evaluate() {
  evalBusy.value = true
  comparison.value = null
  try {
    const cases = JSON.parse(evaluationCases.value)
    evaluation.value = await request('/evaluations', { method: 'POST', body: { kb_id: selectedId.value, cases } })
    await loadEvaluations(); await drawChart()
  } catch (e) { error(e) } finally { evalBusy.value = false }
}
async function compareStrategies() {
  evalBusy.value = true; evaluation.value = null
  try {
    comparison.value = await request('/evaluations/compare', { method: 'POST', body: { kb_id: selectedId.value, cases: JSON.parse(evaluationCases.value) } })
    await loadEvaluations()
  } catch (e) { error(e) } finally { evalBusy.value = false }
}
async function saveDataset() {
  try { await request('/datasets', { method: 'POST', body: { kb_id: selectedId.value, name: datasetName.value, cases: JSON.parse(evaluationCases.value) } }); await loadDatasets(); ElMessage.success('测评集已保存') }
  catch (e) { error(e) }
}
async function generateDataset() {
  evalBusy.value = true
  try {
    const result = await request('/datasets/generate', { method: 'POST', body: { kb_id: selectedId.value, count: 5, document_ids: selectedDocs.value } })
    evaluationCases.value = JSON.stringify(result.cases, null, 2); await loadDatasets()
    ElMessage.success('已生成 5 条带原文出处的样本，请人工复核标准答案')
  } catch (e) { error(e) } finally { evalBusy.value = false }
}
async function loadEvaluations() { evaluationHistory.value = await request(`/evaluations?kb_id=${selectedId.value}`) }
async function showEvaluation(id) {
  try { evaluation.value = await request(`/evaluations/${id}`); await drawChart() } catch (e) { error(e) }
}
async function drawChart() {
  await nextTick(); if (!chartEl.value || !evaluation.value?.metrics) return
  chart?.dispose(); chart = echarts.init(chartEl.value)
  const labels = ['Context Recall', 'Context Precision', 'Faithfulness', 'Answer Relevancy']
  const metrics = evaluation.value.metrics
  chart.setOption({ grid: { top: 12, right: 24, bottom: 30, left: 125 }, tooltip: {},
    xAxis: { type: 'value', min: 0, max: 1 }, yAxis: { type: 'category', data: labels, inverse: true },
    series: [{ type: 'bar', data: ['context_recall', 'context_precision', 'faithfulness', 'answer_relevancy'].map(k => metrics[k]),
      barWidth: 20, itemStyle: { color: '#42816e', borderRadius: [0, 5, 5, 0] } }],
  })
}
async function showTrace() {
  try { trace.value = await request(`/runs/${lastRunId.value}`); traceDialog.value = true } catch (e) { error(e) }
}
watch(selectedId, async () => {
  const saved = readWorkspace(sessionStorage, username.value)
  restoringSelection = true
  controller?.abort(); resetConversation(); debugResult.value = null; evaluation.value = null
  selectedDocs.value = []; selectedApplication.value = ''; comparison.value = null; preset.value = 'custom'; config.value = JSON.parse(JSON.stringify(kb.value?.config || {}))
  await nextTick()
  restoringSelection = false
  if (!selectedId.value || !token.value) { docs.value = []; conversations.value = []; return }
  try {
    await Promise.all([loadDocs(), loadConversations(), loadTools(), loadEvaluations(), loadApplications(), loadFeedback(), loadDatasets(), loadModelStatus(), request('/strategies').then(value => presets.value = value)])
    if (saved?.kb_id === selectedId.value && applications.value.some(item => item.id === saved.application_id)) {
      restoringSelection = true
      selectedApplication.value = saved.application_id
      await nextTick()
      restoringSelection = false
      await loadConversations()
    }
    const previous = restoredConversation(saved, selectedId.value, selectedApplication.value, conversations.value)
    if (previous) await openConversation(previous)
    else rememberConversation()
  } catch (e) { error(e) }
})
watch(selectedApplication, async () => {
  if (restoringSelection) return
  freshConversation(); if (activeApplication.value) chatMode.value = activeApplication.value.mode
  if (selectedId.value && token.value) { try { await loadConversations() } catch (e) { error(e) } }
})
watch(page, async () => { if (page.value === 'evaluation') await drawChart() })
onMounted(async () => {
  if (token.value) { try { await loadKbs() } catch (e) { if (e.status === 401) logout(); else error(e) } }
  try { health.value = await api('/health') } catch (_) { /* Login shows connection error when used. */ }
  poll = setInterval(() => {
    if (docs.value.some(d => ['queued', 'processing'].includes(d.status))) loadDocs().catch(error)
  }, 1500)
})
onUnmounted(() => { clearInterval(poll); controller?.abort(); chart?.dispose() })
</script>

<template>
  <div v-if="!token" class="login-layout">
    <div class="login-story"><div class="brand-mark">知</div><h1>让知识有据可循。</h1><p>从一份文档，到一次值得信任的回答。</p><div class="story-line"></div><span>知序 · 知识工作台</span></div>
    <div class="login-card"><div class="eyebrow">KNOWLEDGE WORKSPACE</div><h2>开始你的知识空间</h2><p class="muted">首次使用请注册，已有账号可直接登录。</p>
      <el-form @submit.prevent="auth(false)"><el-form-item label="用户名"><el-input v-model="username" placeholder="至少 3 位字母、数字或下划线" /></el-form-item><el-form-item label="密码"><el-input v-model="password" type="password" show-password placeholder="至少 8 位" /></el-form-item>
        <el-button type="primary" :loading="authBusy" @click="auth(false)">登录</el-button><el-button :loading="authBusy" @click="auth(true)">创建账号</el-button></el-form>
      <p class="login-note">{{ health.provider === 'demo' ? '当前使用词法演示模型，可免 Key 走通流程。' : '当前已接入百炼模型。' }}</p>
    </div>
  </div>
  <div v-else class="workspace">
    <aside class="sidebar"><a class="brand"><div class="brand-mark">知</div><div>知序<span>KNOWLEDGE STUDIO</span></div></a>
      <div class="sidebar-label">当前知识库</div><el-select v-model="selectedId" placeholder="选择知识库" :disabled="sending"><el-option v-for="item in kbs" :key="item.id" :label="item.name" :value="item.id" /></el-select>
      <button class="new-kb" @click="kbDialog = true">＋ 创建知识库</button>
      <nav><button v-for="[id, label, number] in nav" :key="id" :class="{ active: page === id }" @click="page = id"><span>{{ number }}</span>{{ label }}</button></nav>
      <div class="sidebar-footer"><div class="status-dot"></div>{{ health.provider === 'demo' ? '演示模式' : '百炼模型' }}<p>{{ username }}</p><button @click="logout">退出登录</button></div>
    </aside>
    <main><header class="topbar"><span>工作空间 / {{ title }}</span><span class="mode-pill">{{ health.provider === 'demo' ? '免 Key · 词法演示' : '已连接百炼' }}</span></header>
      <div class="page-heading"><div><div class="eyebrow">{{ kb?.name || 'YOUR KNOWLEDGE' }}</div><h1>{{ title }}</h1><p>{{ page === 'chat' ? '把问题交给资料，让回答带着出处。' : '查看每一步，让知识处理过程清晰可见。' }}</p></div><div class="stats"><b>{{ docs.filter(d => d.status === 'ready').length }}</b><span>已就绪文档</span></div></div>
      <div v-if="!selectedId" class="empty-card"><h2>先建立一个知识库</h2><p>创建后上传文档，即可检索、提问与评测。</p><el-button type="primary" @click="kbDialog = true">创建知识库</el-button></div>
      <template v-else>
        <div v-if="page === 'chat'" class="chat-layout">
          <section class="panel chat-panel"><div class="panel-title"><h3>对话</h3><div class="inline"><el-select v-model="selectedApplication" placeholder="直接使用知识库" style="width: 170px" :disabled="sending"><el-option label="直接使用知识库" value=""/><el-option v-for="item in applications.filter(a => a.enabled)" :key="item.id" :label="item.name" :value="item.id"/></el-select><el-select v-model="chatMode" style="width: 140px" :disabled="sending || !!selectedApplication"><el-option label="Agent 循环" value="agent"/><el-option label="单轮 RAG" value="rag"/></el-select><el-select v-model="outputType" style="width: 140px" :disabled="sending"><el-option label="引用回答" value="answer"/><el-option label="证据图表" value="chart"/><el-option label="引用报告" value="report"/><el-option label="网页预览" value="webpage"/></el-select><el-button :disabled="sending" @click="freshConversation">新对话</el-button></div></div>
            <div class="messages"><div v-if="!chatMessages.length" class="chat-empty"><div class="question-icon">？</div><h2>今天，想从资料里找到什么？</h2><p>{{ activeApplication?.welcome || "先上传文档，再开始提问。" }}</p><button @click="query = '出差报销申请需要几天内提交？'">出差报销申请需要几天内提交？ ↗</button></div>
              <article v-for="(message, i) in chatMessages" :key="i" :class="['message', message.role]"><div class="message-label">{{ message.role === 'user' ? '你' : '知序' }}<span v-if="message.rejected"> · 依据不足</span></div><div class="answer-text">{{ message.content || '正在查找资料…' }}</div><details v-for="source in message.citations" :key="source.source_id" class="citation"><summary>[{{ source.source_id }}] {{ source.filename }} · {{ source.location }}</summary><p>{{ source.content }}</p></details><ArtifactCard v-for="item in message.artifacts || []" :key="item.id" :item="item" :token="token"/><p v-if="message.artifactWarning" class="muted">{{ message.artifactWarning }}</p><div v-if="message.role === 'assistant' && message.run_id && !message.provisional && !message.rejected && message.citations?.length" class="artifact-actions"><el-button v-for="[kind, label] in [['chart','生成图表'],['report','生成报告'],['webpage','生成网页']]" :key="kind" size="small" :disabled="sending || !!artifactBusy" :loading="artifactBusy === `${message.run_id}:${kind}`" @click="convertArtifact(message, kind)">{{ label }}</el-button></div><div v-if="message.role === 'assistant' && message.run_id && !message.provisional" class="feedback-actions"><el-button text :type="message.helpful === true ? 'primary' : 'default'" @click="feedback(message, true)">有帮助</el-button><el-button text :type="message.helpful === false ? 'warning' : 'default'" @click="feedback(message, false)">没有帮助</el-button></div></article>
              <div v-if="draft" class="draft-box"><span>正在生成草稿 · 核验中</span><p>{{ draft }}</p></div>
            </div>
            <div class="composer"><el-input v-model="query" type="textarea" :rows="3" placeholder="输入问题，Ctrl / ⌘ + Enter 发送" @keydown.ctrl.enter.prevent="send" @keydown.meta.enter.prevent="send"/><div><span>回答将附来源；依据不足时明确说明。</span><el-button v-if="sending" @click="cancelChat">停止生成</el-button><el-button v-else type="primary" :disabled="!query.trim()" @click="send">发送 ↗</el-button></div></div>
        </section><aside class="chat-inspector"><section class="panel"><h3>执行过程</h3><p v-if="!steps.length" class="muted">发送问题后，查看检索、生成和核验过程。</p><div v-for="(step, i) in steps" :key="i" class="step"><i :class="step.status"></i><b>{{ step.node }}</b><span>{{ stepStatusText(step) }}</span></div><el-button v-if="lastRunId" text @click="showTrace">查看完整记录</el-button></section><section class="panel"><h3>历史对话</h3><button v-for="item in conversations" :key="item.id" class="history-item" :disabled="sending" @click="openConversation(item.id)">{{ item.title }}</button><p v-if="!conversations.length" class="muted">还没有历史对话。</p></section></aside>
        </div>
        <section v-if="page === 'documents'" class="panel"><div class="panel-title"><div><h3>文档管理</h3><p class="muted">PDF、DOCX、XLSX、Markdown、TXT</p></div><div><input ref="replacementInput" hidden type="file" accept=".pdf,.docx,.xlsx,.md,.txt" @change="upload($event.target.files[0], replacementDocument)"><input ref="uploadInput" hidden type="file" accept=".pdf,.docx,.xlsx,.md,.txt" @change="upload($event.target.files[0])"><el-button :loading="busy" @click="loadSample">载入示例</el-button><el-button type="primary" :loading="busy" @click="uploadInput.click()">＋ 上传文档</el-button></div></div>
          <el-table :data="docs" empty-text="上传第一份文档，建立可查询的资料库。"><el-table-column prop="filename" label="文档" min-width="200"/><el-table-column label="状态" width="120"><template #default="{ row }"><el-tag :type="row.status === 'ready' ? 'success' : row.status === 'failed' ? 'danger' : 'warning'">{{ row.status }}</el-tag></template></el-table-column><el-table-column prop="parent_count" label="父块" width="80"/><el-table-column prop="child_count" label="子块" width="80"/><el-table-column label="操作" width="330"><template #default="{ row }"><el-button text @click="openChunks(row)">片段</el-button><el-button text :disabled="!['ready','failed'].includes(row.status)" @click="replacementDocument = row; replacementInput.click()">覆盖文件</el-button><el-button text :disabled="!['ready','failed'].includes(row.status)" @click="reindex(row)">重建索引</el-button><el-popconfirm title="删除这份文档及其索引？" @confirm="deleteDocument(row)"><template #reference><el-button text type="danger">删除</el-button></template></el-popconfirm></template></el-table-column><el-table-column prop="error" label="处理信息" min-width="150" show-overflow-tooltip/></el-table>
        </section>
        <template v-if="page === 'retrieval'"><section class="panel"><h3>检索实验</h3><div class="search-row"><el-select v-model="debugStrategy" style="width: 170px"><el-option label="当前自定义策略" value="custom"/><el-option label="纯向量" value="dense"/><el-option label="混合检索" value="hybrid"/><el-option label="完整链路" value="full"/></el-select><el-input v-model="debugQuery" placeholder="输入查询，查看召回结果" @keyup.enter="debug"/><el-button type="primary" :loading="debugBusy" :disabled="!debugQuery.trim()" @click="debug">运行检索</el-button></div><div class="filter-row"><el-select v-model="selectedDocs" multiple collapse-tags placeholder="全部文档"><el-option v-for="doc in docs" :key="doc.id" :value="doc.id" :label="doc.filename"/></el-select><el-select v-model="fileTypes" multiple placeholder="全部类型"><el-option v-for="type in ['pdf','docx','xlsx','md','txt']" :key="type" :value="type" :label="type"/></el-select><el-input v-model="tagsFilter" placeholder="标签，逗号分隔"/></div><p class="muted">这些过滤条件也会用于知识问答。关闭元数据过滤开关后不生效。</p></section>
          <section v-if="debugResult" class="panel result-panel"><div class="panel-title"><h3>候选子块与分数</h3><span class="muted">{{ debugResult.diagnostics.elapsed_ms || 0 }} ms</span></div><el-table :data="debugResult.candidates"><el-table-column prop="content" label="子块文本" min-width="280" show-overflow-tooltip/><el-table-column prop="cosine_score" label="Cosine" width="100"/><el-table-column prop="bm25_score" label="BM25" width="100"/><el-table-column prop="rrf_score" label="RRF" width="120"/><el-table-column prop="rerank_score" label="Rerank" width="100"/></el-table><h3>最终证据上下文</h3><details v-for="source in debugResult.sources" :key="source.source_id" class="citation" open><summary>[{{ source.source_id }}] {{ source.filename }} · {{ source.location }}</summary><p>{{ source.content }}</p></details><p v-if="!debugResult.sources.length" class="muted">没有满足当前条件的证据。</p><details><summary>诊断详情</summary><pre>{{ JSON.stringify(debugResult.diagnostics, null, 2) }}</pre></details></section>
        </template>
        <section v-if="page === 'strategy'" class="panel"><div class="panel-title"><h3>七个检索开关</h3><el-select v-model="preset" style="width: 200px" @change="applyPreset"><el-option label="当前自定义策略" value="custom"/><el-option label="基线 · 纯向量" value="dense"/><el-option label="进阶 · 混合检索" value="hybrid"/><el-option label="完整链路" value="full"/></el-select><el-button type="primary" @click="saveStrategy">保存策略</el-button></div><div class="switch-grid"><div v-for="[key, label, description] in switches" :key="key" class="switch-item"><div><h4>{{ label }}</h4><p>{{ description }}</p></div><el-switch v-model="config[key]"/></div></div><h3>检索与分块参数</h3><el-select v-model="config.chunk_strategy" style="width: 220px"><el-option label="递归分块 · 默认500字" value="recursive"/><el-option label="递归短片段 · 默认250字" value="recursive_short"/><el-option label="父子分块 · 默认1200/300字" value="parent_child"/></el-select><el-alert title="分块参数以字符计数；修改后需重建已有文档索引。直接重建将恢复原始文档内容，覆盖人工片段修正。" type="info" :closable="false"/><div class="config-grid"><label v-for="[key, label, min, max] in numeric" :key="key">{{ label }}<el-input-number v-model="config[key]" :min="min" :max="max"/></label><label>重排相关性阈值<el-input-number v-model="config.rerank_threshold" :min="0" :max="1" :step="0.05" :precision="2"/></label><label>余弦阈值（纯向量档）<el-input-number v-model="config.cosine_threshold" :min="-1" :max="1" :step="0.05" :precision="2"/></label></div></section>
        <template v-if="page === 'tools'"><section class="panel"><h3>工具中心</h3><div v-for="tool in tools" :key="tool.name" class="switch-item"><div><h4>{{ tool.name }}</h4><p>{{ tool.description }}</p></div><el-switch v-model="tool.enabled" :disabled="tool.name === 'search_knowledge'" @change="toggleTool(tool)"/></div><p class="muted">工具只在当前用户授权范围内执行；新增外部工具需在后端注册实现。</p></section><section class="panel result-panel"><div class="panel-title"><h3>回答提示词</h3><el-button type="primary" @click="savePrompt">保存</el-button></div><el-input v-model="prompt" type="textarea" :rows="8"/></section></template>
        <template v-if="page === 'evaluation'"><section class="panel"><div class="panel-title"><div><h3>评测样本</h3><p class="muted">填写问题、参考答案和原子事实。参考事实用于判断检索覆盖率。</p></div><div class="inline"><el-button :loading="evalBusy" @click="generateDataset">从文档生成 5 条</el-button><el-button :loading="evalBusy" @click="compareStrategies">三档策略对比</el-button><el-button type="primary" :loading="evalBusy" @click="evaluate">运行评测</el-button></div></div><el-input v-model="evaluationCases" type="textarea" :rows="10" class="code-input"/><el-alert v-if="health.provider === 'demo'" title="演示模式只使用词法匹配，不代表 Qwen 或真实 RAG 质量。" type="warning" :closable="false"/></section><section class="panel result-panel"><div class="panel-title"><h3>可复用测评集</h3><div class="inline"><el-input v-model="datasetName" placeholder="测评集名称"/><el-button @click="saveDataset">保存当前样本</el-button></div></div><button v-for="item in datasets" :key="item.id" class="history-item" @click="evaluationCases = JSON.stringify(item.cases, null, 2)">{{ item.name }} · {{ item.origin }} · {{ item.cases.length }} 条（点击载入）</button></section>
<section v-if="comparison" class="panel result-panel"><h3>三档策略对比</h3><p class="muted">同一批问题、相同资料、单轮 RAG；实际测量质量与耗时。</p><el-table :data="comparison.comparisons"><el-table-column prop="strategy" label="策略"/><el-table-column v-for="key in ['context_recall','context_precision','faithfulness','answer_relevancy','elapsed_ms']" :key="key" :label="key" min-width="140"><template #default="{ row }">{{ row.metrics[key]?.toFixed(3) ?? '缺失 / 不适用' }}</template></el-table-column></el-table><details v-for="item in comparison.comparisons" :key="item.id"><summary>{{ item.strategy }} · 逐题回答与引用</summary><pre>{{ JSON.stringify(item.results, null, 2) }}</pre></details></section><section v-if="evaluation" class="panel result-panel"><h3>质量指标</h3><p v-if="evaluation.status === 'partial_missing'" class="muted">部分评测或 Judge 失败，缺失指标不计入均值。</p><div ref="chartEl" class="metric-chart"></div><el-table :data="evaluation.results"><el-table-column prop="question" label="问题" min-width="220"/><el-table-column prop="answer" label="回答" min-width="260" show-overflow-tooltip/><el-table-column label="Recall" width="90"><template #default="{ row }">{{ row.metrics.context_recall?.toFixed(2) ?? '缺失' }}</template></el-table-column><el-table-column label="Faithfulness" width="110"><template #default="{ row }">{{ row.metrics.faithfulness?.toFixed(2) ?? '缺失 / 不适用' }}</template></el-table-column></el-table></section><section class="panel result-panel"><h3>历史评测</h3><button v-for="item in evaluationHistory" :key="item.id" class="history-item" @click="showEvaluation(item.id)">{{ item.id.slice(0, 8) }} · {{ item.provider }} · {{ item.status }}</button></section></template>

        <section v-if="page === 'applications'" class="panel"><div class="panel-title"><div><h3>应用管理</h3><p class="muted">为同一知识库配置多个客服应用，再到知识问答中选择。</p></div><el-button type="primary" @click="applicationDialog()">创建应用</el-button></div><el-table :data="applications"><el-table-column prop="name" label="应用"/><el-table-column prop="strategy" label="检索策略"/><el-table-column prop="mode" label="运行模式"/><el-table-column prop="welcome" label="欢迎语" min-width="220" show-overflow-tooltip/><el-table-column label="操作"><template #default="{ row }"><el-button text @click="applicationDialog(row)">编辑</el-button></template></el-table-column></el-table></section>
        <section v-if="page === 'feedback'" class="panel"><h3>对话反馈</h3><p class="muted">把真实问题整理成测评集时，请人工填写标准答案和原子事实。</p><div v-for="item in feedbackItems" :key="item.id" class="chunk-card"><div class="panel-title"><b>{{ item.helpful ? '有帮助' : '没有帮助' }} · {{ item.question }}</b><el-button @click="correctFeedback(item)">订正为测评样本</el-button></div><p>{{ item.answer }}</p><p v-if="item.comment">{{ item.comment }}</p></div><p v-if="!feedbackItems.length" class="muted">回答后点击“有帮助 / 没有帮助”，这里会显示记录。</p></section>
        <section v-if="page === 'models'" class="panel"><div class="panel-title"><h3>模型配置与连通测试</h3><el-button :loading="modelBusy" @click="testModels">测试三类模型</el-button></div><template v-if="modelStatus"><div class="config-grid"><label>生成模型<b>{{ modelStatus.generation }}</b></label><label>向量模型<b>{{ modelStatus.embedding }} · {{ modelStatus.dimension }} 维</b></label><label>重排模型<b>{{ modelStatus.rerank }}</b></label></div><p class="muted">当前后端：{{ modelStatus.provider }}。真实模型的 Key、地域地址和模型名在部署环境 .env 中配置，修改后重启 API 与 worker 并重建向量。</p><pre>{{ modelStatus.fingerprint }}</pre></template><pre v-if="modelTest">{{ JSON.stringify(modelTest, null, 2) }}</pre></section>
      </template>
    </main>

    <el-dialog v-model="appDialog" title="客服应用配置" width="600"><el-form label-position="top"><el-form-item label="应用名称"><el-input v-model="appForm.name" placeholder="应用名称"/></el-form-item><el-form-item label="检索策略"><el-select v-model="appForm.strategy"><el-option label="纯向量" value="dense"/><el-option label="混合检索" value="hybrid"/><el-option label="完整链路" value="full"/><el-option label="知识库自定义" value="custom"/></el-select></el-form-item><el-form-item label="运行模式"><el-select v-model="appForm.mode"><el-option label="单轮 RAG" value="rag"/><el-option label="Agent 循环" value="agent"/></el-select></el-form-item><el-form-item label="欢迎语"><el-input v-model="appForm.welcome"/></el-form-item><el-form-item label="依据不足时的回复"><el-input v-model="appForm.fallback"/></el-form-item><el-form-item label="回答提示词"><el-input v-model="appForm.prompt" type="textarea" :rows="4"/></el-form-item><el-form-item label="启用"><el-switch v-model="appForm.enabled"/></el-form-item></el-form><template #footer><el-button type="primary" :disabled="!appForm.name?.trim()" @click="saveApplication">保存应用</el-button></template></el-dialog>
    <el-dialog v-model="feedbackDialog" title="人工订正标准答案" width="600"><p class="muted">请以原文为依据填写，模型曾给出的错误答案不会被自动用作标准。</p><el-input v-model="feedbackForm.name" placeholder="测评集名称"/><el-input v-model="feedbackForm.reference_answer" placeholder="正确的标准答案" type="textarea" :rows="4"/><el-input v-model="feedbackForm.factsText" placeholder='原子事实 JSON 数组，例如 ["保修期为24个月"]' type="textarea" :rows="4"/><template #footer><el-button type="primary" @click="saveFeedbackCase">保存测评样本</el-button></template></el-dialog>
    <el-dialog v-model="kbDialog" title="创建知识库" width="420"><el-input v-model="newKbName" placeholder="知识库名称" @keyup.enter="createKb"/><template #footer><el-button type="primary" :disabled="!newKbName.trim()" @click="createKb">创建</el-button></template></el-dialog>
    <el-drawer v-model="chunkDialog" title="父块片段管理" size="650"><div v-for="item in chunks" :key="item.id" class="chunk-card"><div class="panel-title"><b>片段 {{ item.ordinal + 1 }} · {{ item.metadata.location }}</b><el-button text @click="editChunk = { ...item }">编辑</el-button></div><p>{{ item.content }}</p></div><div v-if="editChunk" class="edit-box"><el-input v-model="editChunk.content" type="textarea" :rows="10"/><el-button type="primary" @click="saveChunk">保存并重新索引</el-button></div></el-drawer>
    <el-drawer v-model="traceDialog" title="运行记录" size="650"><template v-if="trace"><p>状态：{{ trace.status }} · 检索重试 {{ trace.grade_retries }} 次 · 答案重写 {{ trace.check_retries }} 次</p><div v-for="step in trace.steps" :key="step.ordinal" class="chunk-card"><b>{{ step.ordinal }} · {{ step.node }} · {{ step.elapsed_ms }} ms</b><pre>{{ JSON.stringify(step.output_summary, null, 2) }}</pre></div></template></el-drawer>
  </div>
</template>
