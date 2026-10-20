import { InlineNotification, TreeNode, TreeView } from '@carbon/react'
import { Rich, useCopy } from './i18n/lang'
import { Markdown } from './Markdown'
import { PARTS } from './Term'
import type { Library } from './types'

const INDEX = '__index__'
const LOG = '__log__'

// The URL names the page: none is the index, `log` the log and `wiki/<name>` a wiki page.
const fromUrl = (page: string | null) => (page === null ? INDEX : page === 'log' ? LOG : page.replace(/^wiki\//, ''))
const toUrl = (sel: string) => (sel === INDEX ? null : sel === LOG ? 'log' : `wiki/${sel}`)

type Props = { lib: Library | null; page: string | null; onPage: (page: string | null) => void }

export function LibraryTab({ lib, page, onPage }: Props) {
  const { t } = useCopy()
  if (lib === null)
    return (
      <p className="muted">
        <Rich k="lib.empty" parts={PARTS} />
      </p>
    )

  const names = new Set(lib.pages.map((p) => p.name))
  const wanted = fromUrl(page)
  const sel = wanted === INDEX || wanted === LOG || names.has(wanted) ? wanted : INDEX
  const current = lib.pages.find((p) => p.name === sel)
  const text = sel === INDEX ? lib.index : sel === LOG ? lib.log : (current?.text ?? '')
  const goto = (name: string) => {
    if (names.has(name)) onPage(toUrl(name))
  }
  const list = lib.pending.slice(0, 5).join(t('lib.listSeparator'))
  const shown = lib.pending.length > 5 ? t('lib.pendingMore', { list }) : list
  return (
    <>
      {lib.pending.length > 0 && (
        <InlineNotification
          kind="warning"
          statusIconDescription={t('icon.warning')}
          lowContrast
          hideCloseButton
          title={t('lib.pending', { n: lib.pending.length })}
        >
          <span translate="no">{shown}</span>
        </InlineNotification>
      )}
      <div className="lib">
        <nav className="lib-list" aria-label={t('lib.pages')}>
          {/* Carbon TreeView 只在挂载时读 active/selected；按所选页换 key，[[链接]] 跳页时左侧选中跟着走。 */}
          <TreeView key={sel} label={t('lib.pages')} hideLabel size="sm" active={sel} selected={[sel]} onSelect={(_e, node) => node.id && node.id !== '__pages__' && node.id !== sel && onPage(toUrl(String(node.id)))}>
            <TreeNode id={INDEX} label={t('lib.index')} value={INDEX} />
            <TreeNode id={LOG} label={t('lib.log')} value={LOG} />
            {lib.pages.length > 0 && (
              <TreeNode id="__pages__" label={t('lib.pageGroup', { n: lib.pages.length })} value="__pages__" isExpanded>
                {lib.pages.map((p) => (
                  <TreeNode key={p.name} id={p.name} label={<span translate="no">{p.name}</span>} value={p.name} />
                ))}
              </TreeNode>
            )}
          </TreeView>
          {lib.pages.length === 0 && <p className="muted lib-empty">{t('lib.noPages')}</p>}
        </nav>
        <article className="lib-page">
          {text.trim()
            ? <div translate="no"><Markdown text={text} onWiki={goto} /></div>
            : <p className="muted">{t('lib.emptyPage')}</p>}
        </article>
      </div>
    </>
  )
}
