import { useMemo } from 'react'
import { renderMarkdown } from './markdown'

// html: false 下 markdown-it 不放行任何原始 HTML，链接协议也由它校验，所以直接注入是安全的。
export function Markdown({ text, onWiki }: { text: string; onWiki?: (name: string) => void }) {
  const html = useMemo(() => renderMarkdown(text), [text])
  return (
    <div
      className="md"
      onClick={(e) => {
        const a = (e.target as HTMLElement).closest('a[data-wiki]')
        if (!a) return
        e.preventDefault()
        onWiki?.(a.getAttribute('data-wiki')!)
      }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
