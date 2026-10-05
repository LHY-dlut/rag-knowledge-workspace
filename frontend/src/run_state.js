// A disconnected subscriber does not determine the backend job's final state.
export function markInterruptedSteps(steps, reason) {
  const status = reason === 'cancelled' ? 'cancelled' : 'interrupted'
  return steps.map(step => step.status === 'running' ? { ...step, status } : step)
}

export function stepStatusText(step) {
  if (step.status === 'running') return '执行中'
  if (step.status === 'cancelled') return '已取消'
  if (step.status === 'interrupted') return '连接中断，待恢复'
  const label = step.status === 'failed' ? '已失败' : '已完成'
  return Number.isFinite(step.elapsed_ms) ? `${label} · ${step.elapsed_ms} ms` : label
}
