import { useState } from 'react'
import { InlineNotification, TreeNode, TreeView } from '@carbon/react'
import { Markdown } from './Markdown'
import type { Library } from './types'

const INDEX = '__index__'
const LOG = '__log__'

export function LibraryTab({ lib }: { lib: Library | null }) {
  const [sel, setSel] = useState(INDEX)
  if (lib === null)
    return (
      <p className="muted">
        还没有资料库——用 <code className="mono">shinka_research library add</code> 加资料。
      </p>
    )

  const names = new Set(lib.pages.map((p) => p.name))
  const page = lib.pages.find((p) => p.name === sel)
  const text = sel === INDEX ? lib.index : sel === LOG ? lib.log : (page?.text ?? '')
  const goto = (name: string) => {
    if (names.has(name)) setSel(name)
  }
  const shown = lib.pending.slice(0, 5).join('、') + (lib.pending.length > 5 ? ' 等' : '')

  return (
    <>
      {lib.pending.length > 0 && (
        <InlineNotification
          kind="warning"
          lowContrast
          hideCloseButton
          title={`有 ${lib.pending.length} 份资料待整理：`}
          subtitle={shown}
        />
      )}
      <div className="lib">
        <nav className="lib-list" aria-label="资料库页面">
          {/* Carbon TreeView 只在挂载时读 active/selected；按所选页换 key，[[链接]] 跳页时左侧选中跟着走。 */}
          <TreeView key={sel} label="资料库页面" hideLabel size="sm" active={sel} selected={[sel]} onSelect={(_e, node) => node.id && node.id !== '__pages__' && setSel(String(node.id))}>
            <TreeNode id={INDEX} label="目录" value={INDEX} />
            <TreeNode id={LOG} label="日志" value={LOG} />
            {lib.pages.length > 0 && (
              <TreeNode id="__pages__" label={`页面（${lib.pages.length}）`} value="__pages__" isExpanded>
                {lib.pages.map((p) => (
                  <TreeNode key={p.name} id={p.name} label={p.name} value={p.name} />
                ))}
              </TreeNode>
            )}
          </TreeView>
          {lib.pages.length === 0 && <p className="muted lib-empty">wiki 里还没有页面，研究员整理资料后会写到这里。</p>}
        </nav>
        <article className="lib-page">
          {text.trim() ? <Markdown text={text} onWiki={goto} /> : <p className="muted">这一页是空的。</p>}
        </article>
      </div>
    </>
  )
}
