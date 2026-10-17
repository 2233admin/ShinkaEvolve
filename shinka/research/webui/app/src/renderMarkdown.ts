import MarkdownIt from 'markdown-it'

// [[页面名]] → 应用内链接（data-wiki），点击时由 Markdown 组件接管。
const md = new MarkdownIt({ html: false, linkify: true, breaks: false })

md.inline.ruler.before('link', 'wikilink', (state, silent) => {
  const src = state.src
  if (src.charCodeAt(state.pos) !== 0x5b || src.charCodeAt(state.pos + 1) !== 0x5b) return false
  const end = src.indexOf(']]', state.pos + 2)
  if (end < 0) return false
  const name = src.slice(state.pos + 2, end).split('|')[0].trim()
  if (!name || name.includes('\n')) return false
  if (!silent) {
    const open = state.push('link_open', 'a', 1)
    open.attrs = [['href', `#${encodeURIComponent(name)}`], ['data-wiki', name]]
    state.push('text', '', 0).content = src.slice(state.pos + 2, end).split('|').pop()!.trim()
    state.push('link_close', 'a', -1)
  }
  state.pos = end + 2
  return true
})

const fallback = md.renderer.rules.link_open ?? ((tokens, i, opts, _env, self) => self.renderToken(tokens, i, opts))
md.renderer.rules.link_open = (tokens, i, opts, env, self) => {
  const t = tokens[i]
  if (t.attrIndex('data-wiki') < 0) {
    t.attrSet('target', '_blank')
    t.attrSet('rel', 'noopener noreferrer')
  }
  return fallback(tokens, i, opts, env, self)
}

// 页面开头的 YAML 头信息（--- … ---）不当正文显示。
const stripFrontmatter = (text: string): string => text.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, '')

export const renderMarkdown = (text: string): string => md.render(stripFrontmatter(text))
