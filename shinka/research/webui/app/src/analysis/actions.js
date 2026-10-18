// A finite data grammar, not JavaScript evaluation. Invalid suffixes reject the
// entire action before any effect occurs, including a valid-looking prefix.
export function parseAction(source, allowed, element, event) {
  const commands = []
  let rest = source.trim()
  while (rest) {
    const eventCall = rest.match(/^event\.(stopPropagation|preventDefault)\(\);?\s*/)
    if (eventCall) {
      commands.push({ name: '$event', args: [eventCall[1]] })
      rest = rest.slice(eventCall[0].length)
      continue
    }
    const call = rest.match(/^(?:window\.)?([\w$]+)\s*\(/)
    if (!call || !allowed.has(call[1])) return null
    rest = rest.slice(call[0].length)
    const args = []
    while (!rest.startsWith(')')) {
      const token = rest.match(/^\s*(?:'((?:\\.|[^'\\])*)'|"((?:\\.|[^"\\])*)"|(-?\d+(?:\.\d+)?)|(true|false|null)|(this(?:\.value|\.checked)?|event))\s*/)
      if (!token) return null
      let value
      if (token[1] !== undefined || token[2] !== undefined) value = (token[1] ?? token[2]).replace(/\\(['"\\])/g, '$1')
      else if (token[3] !== undefined) value = Number(token[3])
      else if (token[4] !== undefined) value = JSON.parse(token[4])
      else value = { this: element, 'this.value': element.value, 'this.checked': element.checked, event }[token[5]]
      args.push(value)
      rest = rest.slice(token[0].length)
      if (rest.startsWith(',')) rest = rest.slice(1).trim()
      else if (!rest.startsWith(')')) return null
    }
    rest = rest.slice(1).trim().replace(/^;/, '').trim()
    commands.push({ name: call[1], args })
  }
  return commands
}

export function plotURL(value, origin) {
  const url = new URL(String(value), origin)
  return url.origin === origin && url.pathname.startsWith('/plot_file/') ? url : null
}
