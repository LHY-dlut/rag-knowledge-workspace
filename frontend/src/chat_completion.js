// A verified terminal answer is independent of the history-list request.
// Streaming failures still propagate to the existing interruption/cancel path.
export async function streamThenRefresh(stream, refresh) {
  await stream()
  try {
    await refresh()
    return null
  } catch (error) {
    return error
  }
}
