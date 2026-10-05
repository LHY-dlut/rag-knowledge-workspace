/** Parse SSE incrementally, including CRLF split between network chunks. */
export async function* parseSSE(stream) {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buffer = '', eventName = 'message', data = [], lastId = ''
  function parseLine(line) {
    if (!line) {
      const frame = data.length ? { event: eventName, data: data.join('\n'), id: lastId } : null
      eventName = 'message'; data = []
      return frame
    }
    if (line.startsWith(':')) return null
    const colon = line.indexOf(':')
    const field = colon < 0 ? line : line.slice(0, colon)
    let value = colon < 0 ? '' : line.slice(colon + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    if (field === 'event') eventName = value
    else if (field === 'data') data.push(value)
    else if (field === 'id' && !value.includes('\0')) lastId = value
    return null
  }
  try {
    while (true) {
      const { value, done } = await reader.read()
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true })
      if (buffer.length > 1024 * 1024) throw new Error('SSE frame exceeds size limit')
      let newline
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline).replace(/\r$/, '')
        buffer = buffer.slice(newline + 1)
        const frame = parseLine(line)
        if (frame) yield frame
      }
      if (done) break
    }
    // A valid SSE event ends with a blank line. Do not commit truncated frames.
  } finally {
    await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}
