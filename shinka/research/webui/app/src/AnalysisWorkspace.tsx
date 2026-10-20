import { useEffect, useRef, useState } from 'react'
import { InlineNotification } from '@carbon/react'
import type { CopyKey } from './i18n/copy'
import { Rich, useCopy, type Lang } from './i18n/lang'
import { PARTS } from './Term'

const PAGES: Record<string, true> = {
  '/analysis/index.html': true,
  '/viz_tree.html': true,
  '/compare.html': true,
}

type Selection = { path: string; label: string }[]

// A failure the host explains in the reader's language; anything else shows its own message.
class HostError extends Error {
  constructor(readonly key: CopyKey) {
    super(key)
  }
}

function selection(text: string | null): Selection {
  const value: unknown = JSON.parse(text ?? 'null')
  if (!Array.isArray(value) || value.length < 2 || !value.every((item) =>
    typeof item === 'object' && item !== null && typeof item.path === 'string' && typeof item.label === 'string')) {
    throw new HostError('an.compareSelection')
  }
  return value
}

function initialAnalysis(): { src: string | null; error: unknown } {
  try {
    const params = new URLSearchParams(window.location.search)
    const url = new URL(params.get('analysis') ?? '/analysis/index.html', window.location.origin)
    if (url.origin !== window.location.origin || !PAGES[url.pathname]) throw new HostError('an.badLink')
    if (url.pathname === '/compare.html') {
      url.searchParams.set('selection', JSON.stringify(selection(params.get('compare') ?? url.searchParams.get('selection'))))
    }
    return { src: url.pathname + url.search + url.hash, error: null }
  } catch (error) {
    return { src: null, error }
  }
}

// The recovered pages expose their own setters: viz_tree.html's applyLang also relabels its 中/EN button.
function setFrameLanguage(child: Window, lang: Lang) {
  const apply = Reflect.get(child, 'applyLang') ?? Reflect.get(child, 'shinkaSetLanguage')
  if (typeof apply === 'function') apply(lang)
}

export function AnalysisWorkspace({ light }: { light: boolean }) {
  const { lang, t } = useCopy()
  const [initial] = useState(initialAnalysis)
  const [error, setError] = useState<unknown>(initial.error)
  const frame = useRef<HTMLIFrameElement>(null)
  const theme = light ? 'light' : 'dark'
  useEffect(() => {
    const child = frame.current?.contentWindow
    const apply = child && Reflect.get(child, 'shinkaSetTheme')
    if (typeof apply === 'function') apply(theme)
  }, [theme])
  useEffect(() => {
    const child = frame.current?.contentWindow
    if (child) setFrameLanguage(child, lang)
  }, [lang])

  function connect() {
    const child = frame.current?.contentWindow
    if (!child) return
    try {
      const apply = Reflect.get(child, 'shinkaSetTheme')
      if (typeof apply !== 'function' || typeof Reflect.get(child, 'ShinkaHTML') !== 'function') {
        throw new HostError('an.notLoaded')
      }
      apply(theme)
      setFrameLanguage(child, lang)
      const sync = () => {
        const inside = new URL(child.location.href)
        if (inside.origin !== window.location.origin || !PAGES[inside.pathname]) {
          throw new HostError('an.leftWorkspace')
        }
        const outside = new URL(window.location.href)
        outside.searchParams.set('analysis', inside.pathname + inside.search + inside.hash)
        if (inside.pathname === '/compare.html') {
          outside.searchParams.set('compare', JSON.stringify(selection(inside.searchParams.get('selection'))))
        } else {
          outside.searchParams.delete('compare')
        }
        // Native frame navigation owns history. This is a URL projection, not a
        // second navigation: never push an extra entry or reset the iframe src.
        window.history.replaceState(null, '', outside)
      }
      for (const method of ['replaceState', 'pushState'] as const) {
        const original = child.history[method].bind(child.history)
        child.history[method] = (data: unknown, unused: string, url?: string | URL | null) => {
          original(data, unused, url)
          sync()
        }
      }
      child.addEventListener('popstate', sync)
      child.document.addEventListener('click', (event) => {
        const target = event.target as Element | null
        const anchor = target?.closest('a')
        if (!anchor) return
        const url = new URL(anchor.href, child.location.href)
        if (url.origin === window.location.origin && (url.pathname === '/' || url.pathname === '/index.html')) {
          event.preventDefault()
          child.location.assign('/analysis/index.html')
        }
      }, true)
      sync()
      setError(null)
    } catch (error) {
      setError(error)
    }
  }

  const message = error === null ? null : error instanceof HostError ? t(error.key) : error instanceof Error ? error.message : String(error)
  return (
    <section aria-label={t('an.section')}>
      <p className="analysis-context"><Rich k="an.context" parts={PARTS} /></p>
      {message && <InlineNotification kind="error" statusIconDescription={t('icon.error')} lowContrast hideCloseButton title={t('an.readError')} subtitle={message} />}
      {initial.src && (
        <div className="analysis-scroll">
          <iframe ref={frame} className="analysis-frame" title={t('an.section')} src={initial.src} onLoad={connect} />
        </div>
      )}
    </section>
  )
}
