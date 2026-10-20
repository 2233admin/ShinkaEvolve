import { useMemo } from 'react'
import { renderMarkdown } from './renderMarkdown'

// html: false 下 markdown-it 不放行任何原始 HTML，链接协议也由它校验，所以直接注入是安全的。
// [[页面]] 链接是普通的应用内链接，点按由外壳路由接管。
export function Markdown({ text }: { text: string }) {
  const html = useMemo(() => renderMarkdown(text), [text])
  return <div className="md" dangerouslySetInnerHTML={{ __html: html }} />
}
