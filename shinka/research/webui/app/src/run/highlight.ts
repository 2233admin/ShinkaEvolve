import hljs from 'highlight.js/lib/core'
import cpp from 'highlight.js/lib/languages/cpp'
import javascript from 'highlight.js/lib/languages/javascript'
import json from 'highlight.js/lib/languages/json'
import julia from 'highlight.js/lib/languages/julia'
import python from 'highlight.js/lib/languages/python'
import rust from 'highlight.js/lib/languages/rust'
import swift from 'highlight.js/lib/languages/swift'
import typescript from 'highlight.js/lib/languages/typescript'
import { splitHighlighted } from './data'

// The languages Shinka evolves; the restored page registered Python only (D-R05).
hljs.registerLanguage('python', python)
hljs.registerLanguage('cpp', cpp)
hljs.registerLanguage('javascript', javascript)
hljs.registerLanguage('typescript', typescript)
hljs.registerLanguage('rust', rust)
hljs.registerLanguage('swift', swift)
hljs.registerLanguage('julia', julia)
hljs.registerLanguage('json', json)
hljs.registerAliases(['c', 'cuda', 'cu'], { languageName: 'cpp' })

const escape = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

/** Source as escaped HTML lines, highlighted when its language is known; highlight.js escapes the text itself. */
export function highlightLines(code: string, language: string | null | undefined): string[] {
  const name = language?.toLowerCase() ?? 'python'
  const html = hljs.getLanguage(name) ? hljs.highlight(code, { language: name, ignoreIllegals: true }).value : escape(code)
  return splitHighlighted(html)
}
