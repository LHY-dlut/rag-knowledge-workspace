<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import * as echarts from 'echarts/core'
import { BarChart, GraphChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { ElMessage } from 'element-plus'
import { downloadArtifact, graphOption, numericOption } from './artifact_view.js'

echarts.use([BarChart, GraphChart, GridComponent, TooltipComponent, CanvasRenderer])
const props = defineProps({ item: { type: Object, required: true }, token: { type: String, required: true } })
const title = computed(() => ({ chart: '证据图表', report: '引用报告', webpage: '网页预览' })[props.item.type])
const root = ref(null), expanded = ref(false), downloading = ref(false)
let instances = [], observer
async function draw() {
  await nextTick()
  instances.forEach(c => c.dispose()); instances = []
  if (props.item.type !== 'chart' || !root.value) return
  const payload = props.item.payload
  for (const [i, el] of [...root.value.querySelectorAll('.artifact-chart')].entries()) {
    const chart = echarts.init(el)
    chart.setOption(payload.charts.length ? numericOption(payload.charts[i]) : graphOption(payload.graph))
    instances.push(chart)
  }
}
async function download() {
  downloading.value = true
  try { await downloadArtifact(props.item, props.token) } catch (e) { ElMessage.error(e.message) }
  finally { downloading.value = false }
}
onMounted(async () => { await draw(); observer = new ResizeObserver(() => instances.forEach(c => c.resize())); observer.observe(root.value) })
watch(() => props.item, draw)
onUnmounted(() => { observer?.disconnect(); instances.forEach(c => c.dispose()) })
</script>

<template>
  <section ref="root" class="artifact-card">
    <div class="artifact-heading"><b>{{ title }}</b><el-button text :loading="downloading" @click="download">下载{{ item.type === 'chart' ? '数据' : 'HTML' }}</el-button></div>
    <p v-if="!item.current_evidence" class="artifact-warning">来源已更新或删除；此成果保留生成时的证据快照。</p>
    <p v-if="item.payload.verification === 'demo_exact_extraction_only'" class="muted">Demo 逐字摘录成果；尚未进行真实模型语义核验。</p>
    <template v-if="item.type === 'chart'">
      <p class="muted">{{ item.payload.charts.length ? '仅展示原文直接支持的数值，按单位分图；未进行汇总计算。' : '没有足够明确的数值，展示来源与已核验事实的引用关系。' }}</p>
      <template v-if="item.payload.charts.length"><div v-for="(group, i) in item.payload.charts" :key="i"><b>{{ group.unit }}</b><div class="artifact-chart"></div></div></template>
      <div v-else class="artifact-chart"></div>
    </template>
    <iframe v-else class="artifact-preview" :title="title" :srcdoc="item.preview_html" sandbox="" referrerpolicy="no-referrer"></iframe>
    <button class="artifact-evidence-toggle" @click="expanded = !expanded">{{ expanded ? '收起' : '查看' }}事实与引用依据</button>
    <div v-if="expanded">
      <div v-for="fact in item.payload.facts" :key="fact.span_id" class="artifact-fact"><p>{{ fact.text }}</p><blockquote v-for="(ref, i) in fact.evidence" :key="i">[{{ ref.source_id }}] {{ ref.original_quote }}<small>原文字符位置 {{ ref.source_start }}–{{ ref.source_end }}</small></blockquote></div>
      <details v-for="source in item.payload.citations" :key="source.source_id" class="citation"><summary>[{{ source.source_id }}] {{ source.filename }} · {{ source.location }} · 文档版本 {{ source.document_revision }}</summary><p>{{ source.content }}</p></details>
    </div>
  </section>
</template>

<style scoped>
.artifact-card{border:1px solid #c9dad3;background:#f7faf8;border-radius:12px;padding:16px;margin:16px 0}
.artifact-heading{display:flex;align-items:center;justify-content:space-between}.artifact-chart{height:380px;width:100%}
.artifact-preview{width:100%;height:460px;border:1px solid #dce4e2;border-radius:8px;background:white}
.artifact-warning{color:#95611a}.artifact-evidence-toggle{border:0;background:transparent;color:#357a65;padding:10px 0;cursor:pointer}
.artifact-fact{border-top:1px solid #dce4e2}blockquote{margin:12px 0;padding:8px 12px;border-left:3px solid #8baea0;white-space:pre-wrap}small{display:block;color:#667f76}
</style>
