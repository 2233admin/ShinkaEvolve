import { Notice } from '../Notice'
import { useLayoutEffect, useRef, type ReactNode } from 'react'
import { Rich, useCopy } from '../i18n/lang'
import { Markdown } from '../Markdown'
import { routeHref } from '../shell/route'
import { PARTS } from '../Term'
import type { Library as LibraryData } from '../types'

type Props = {
  lib: LibraryData | null
  /** The URL's page: null is the index, `log` the log and `wiki/<name>` a wiki page. */
  page: string | null
}

/**
 * The shared library (award-redesign 4.3, P11): what is in it, what still waits to be compiled, and
 * one page at a time as sanitised Markdown. Every page is a link, so it can be shared, reopened and
 * reached from a `[[page]]` link anywhere in the workbench.
 */
export function Library({ lib, page }: Props) {
  const { t } = useCopy()
  const strip = useRef<HTMLElement>(null)
  const wiki = lib?.pages.find((p) => `wiki/${p.name}` === page) ?? null
  // A page the library does not have shows the index, with a note saying so.
  const missing = page !== null && page !== 'log' && wiki === null ? page.replace(/^wiki\//, '') : null
  const current = page === 'log' || wiki ? page : null

  // On phones the page list is one scrolling row; it starts at the open page.
  useLayoutEffect(() => {
    const nav = strip.current
    const here = nav?.querySelector<HTMLElement>('[aria-current="page"]')
    if (nav && here && nav.scrollWidth > nav.clientWidth) nav.scrollLeft = here.offsetLeft - 16
  }, [current])

  const header = (
    <header className="rhead">
      <h1>{t('head.library')}</h1>
      <p className="dmeta">
        <span className="dmeta-item">{t('lede.library')}</span>
        {lib && <span className="dmeta-item">{t('lib.meta', { n: lib.pages.length })}</span>}
        {lib && lib.pending.length > 0 && <span className="dmeta-item">{t('nav.pending', { n: lib.pending.length })}</span>}
      </p>
    </header>
  )
  if (lib === null)
    return (
      <>
        {header}
        <p className="muted lib-none">
          <Rich k="lib.empty" parts={PARTS} />
        </p>
      </>
    )

  const list = lib.pending.slice(0, 5).join(t('lib.listSeparator'))
  const shown = lib.pending.length > 5 ? t('lib.pendingMore', { list }) : list
  const text = page === 'log' ? lib.log : wiki ? wiki.text : lib.index
  const link = (target: string | null, label: ReactNode, data = false) => (
    <li key={target ?? ''}>
      <a href={routeHref({ view: 'library', page: target })} aria-current={current === target ? 'page' : undefined} translate={data ? 'no' : undefined}>
        {label}
      </a>
    </li>
  )

  return (
    <>
      {header}
      {lib.pending.length > 0 && (
        // Raw sources waiting to be compiled are work in progress, not a fault: an info notice, never the signal.
        <Notice kind="info" label={t('icon.info')} title={t('lib.pending', { n: lib.pending.length })}>
          <span className="lib-pending">
            <span className="mono" translate="no">{shown}</span>
            <span><Rich k="lib.pendingHint" parts={PARTS} /></span>
          </span>
        </Notice>
      )}
      {missing !== null && (
        <Notice kind="info" label={t('icon.info')} title={t('lib.missing')}>
          <span className="mono" translate="no">{missing}</span>
        </Notice>
      )}
      <div className="lib">
        <nav className="lib-nav" aria-label={t('lib.pages')} ref={strip}>
          <ul>
            {link(null, t('lib.index'))}
            {link('log', t('lib.log'))}
          </ul>
          <h2 className="lib-group">{t('lib.pageGroup', { n: lib.pages.length })}</h2>
          {lib.pages.length > 0 ? <ul>{lib.pages.map((p) => link(`wiki/${p.name}`, p.name, true))}</ul> : <p className="muted small lib-nopages">{t('lib.noPages')}</p>}
        </nav>
        <article className="lib-article">
          {text.trim() ? (
            <div translate="no">
              <Markdown text={text} />
            </div>
          ) : (
            <p className="muted">{t('lib.emptyPage')}</p>
          )}
        </article>
      </div>
    </>
  )
}
