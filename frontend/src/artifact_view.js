// Build chart options locally from the validated data. Never accept model JS,
// arbitrary ECharts options, HTML tooltips or a formatter function from payloads.
export function mergeArtifacts(current = [], incoming = []) {
  const items = new Map(current.map(item => [item.id, item]))
  for (const item of incoming) if (item?.id) items.set(item.id, item)
  return [...items.values()]
}

export function numericOption(group) {
  const points = (group.points || []).filter(p => typeof p.value === 'number' && Number.isFinite(p.value))
  return {
    tooltip: { trigger: 'axis', renderMode: 'richText' },
    grid: { top: 20, right: 30, bottom: 160, left: 65 },
    xAxis: { type: 'category', data: points.map(p => p.label), axisLabel: { rotate: 20, width: 160, overflow: 'break' } },
    yAxis: { type: 'value', name: String(group.unit), min: Math.min(0, ...points.map(p => p.value)) },
    series: [{ type: 'bar', data: points.map(p => p.value), label: { show: true, position: 'top' },
      itemStyle: { color: '#42816e', borderRadius: [5, 5, 0, 0] } }],
  }
}

export function graphOption(graph) {
  const nodes = (graph.nodes || []).map(n => ({ id: String(n.id), name: String(n.name),
    symbolSize: n.category === 'source' ? 42 : 26,
    itemStyle: { color: n.category === 'source' ? '#42816e' : '#466fa1' } }))
  const ids = new Set(nodes.map(n => n.id))
  const links = (graph.edges || []).filter(e => ids.has(String(e.source)) && ids.has(String(e.target)))
    .map(e => ({ source: String(e.source), target: String(e.target) }))
  return { tooltip: { renderMode: 'richText' }, series: [{ type: 'graph', layout: 'force',
    data: nodes, links, roam: true, force: { repulsion: 200, edgeLength: 130 },
    label: { show: true, width: 180, overflow: 'truncate' }, edgeSymbol: ['none', 'arrow'] }] }
}

export async function downloadArtifact(item, token) {
  if (!/^[0-9a-f-]{36}$/i.test(item.id) || !['chart', 'report', 'webpage'].includes(item.type)) throw new Error('无效成果')
  const response = await fetch(`/api/artifacts/${item.id}/download`, { headers: { Authorization: `Bearer ${token}` } })
  if (!response.ok) throw new Error(`下载失败：HTTP ${response.status}`)
  const url = URL.createObjectURL(await response.blob())
  const link = document.createElement('a')
  link.href = url; link.download = `rag-${item.type}-${item.id.slice(0, 8)}.${item.type === 'chart' ? 'json' : 'html'}`
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
