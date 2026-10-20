import { useCallback, useEffect, useRef, useState } from 'react'
import { parseRoute, type Route } from './route'

export type Router = {
  route: Route
  /** Changes on every navigation and history step, not when a hosted page projects its own state. */
  entry: number
  go: (href: string, options?: { replace?: boolean }) => void
  /** Re-read the URL after a hosted page rewrote it, without counting a navigation. */
  sync: () => void
}

/** Query-string routing with pushState: in-app links, Back and Forward stay on this page. */
export function useRouter(): Router {
  const [state, setState] = useState(() => ({ route: parseRoute(window.location.search), entry: 0 }))
  // The query the state was read from; a step that only changes the #fragment is not a navigation.
  const read = useRef(window.location.search)
  const go = useCallback((href: string, { replace = false } = {}) => {
    const url = new URL(href, window.location.href)
    if (replace) window.history.replaceState(null, '', url)
    else window.history.pushState(null, '', url)
    read.current = url.search
    setState((s) => ({ route: parseRoute(url.search), entry: s.entry + 1 }))
    if (!replace) window.scrollTo(0, 0)
  }, [])
  const sync = useCallback(() => {
    read.current = window.location.search
    setState((s) => ({ route: parseRoute(window.location.search), entry: s.entry }))
  }, [])

  useEffect(() => {
    const onPop = () => {
      if (window.location.search === read.current) return
      read.current = window.location.search
      setState((s) => ({ route: parseRoute(window.location.search), entry: s.entry + 1 }))
    }
    // Plain left clicks on links to this page navigate in place; new tabs and downloads stay native.
    const onClick = (event: MouseEvent) => {
      if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
      const anchor = event.target instanceof Element ? event.target.closest('a[href]') : null
      if (!(anchor instanceof HTMLAnchorElement) || (anchor.target && anchor.target !== '_self') || anchor.hasAttribute('download')) return
      // In-page anchors (#content) stay native.
      if (anchor.getAttribute('href')?.startsWith('#')) return
      const url = new URL(anchor.href)
      if (url.origin !== window.location.origin || url.pathname !== '/') return
      event.preventDefault()
      go(url.href)
    }
    window.addEventListener('popstate', onPop)
    document.addEventListener('click', onClick)
    return () => {
      window.removeEventListener('popstate', onPop)
      document.removeEventListener('click', onClick)
    }
  }, [go])

  return { ...state, go, sync }
}
