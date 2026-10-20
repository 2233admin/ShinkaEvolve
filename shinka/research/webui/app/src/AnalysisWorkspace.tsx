import { useEffect, useRef, useState } from 'react'
import { InlineNotification } from '@carbon/react'
import { Rich, useCopy, type Lang } from './i18n/lang'
import { framePath, HostError, projectedHref, type HostedRoute } from './legacyFrame'
import { PARTS } from './Term'

// The recovered pages expose their own setters: viz_tree.html's applyLang also relabels its 中/EN button.
function setFrameLanguage(child: Window, lang: Lang) {
  const apply = Reflect.get(child, 'applyLang') ?? Reflect.get(child, 'shinkaSetLanguage')
  if (typeof apply === 'function') apply(lang)
}

type Props = {
  route: HostedRoute
  light: boolean
  /** The frame moved on its own; the URL already says where. */
  onProject: () => void
  /** A link in the frame leads to the run list, which the shell now draws itself. */
  onRunList: () => void
}

/** Hosts a restored analysis page for a route; mount once per navigation (key it by the router entry). */
export function AnalysisWorkspace({ route, light, onProject, onRunList }: Props) {
  const { lang, t } = useCopy()
  const [src] = useState(() => framePath(route))
  const [error, setError] = useState<unknown>(null)
  const frame = useRef<HTMLIFrameElement>(null)
  const latest = useRef({ route, onProject, onRunList })
  latest.current = { route, onProject, onRunList }
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
        if (inside.origin !== window.location.origin) throw new HostError('an.leftWorkspace')
        // Native frame navigation owns history. This is a URL projection, not a
        // second navigation: never push an extra entry or reset the iframe src.
        window.history.replaceState(null, '', projectedHref(inside, latest.current.route))
        latest.current.onProject()
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
          // The restored pages' "back to all runs" link: a navigation of the shell, so Back returns here.
          event.preventDefault()
          latest.current.onRunList()
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
      <div className="analysis-scroll">
        <iframe ref={frame} className="analysis-frame" title={t('an.section')} src={src} onLoad={connect} />
      </div>
    </section>
  )
}
