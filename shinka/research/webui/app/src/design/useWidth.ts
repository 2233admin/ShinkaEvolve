import { useLayoutEffect, useRef, useState } from 'react'

/**
 * The width of an element, kept current as the layout changes. It is read before the first paint, so
 * a chart has its full height by the time Back restores the scroll position below it.
 */
export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(0)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    setWidth(el.clientWidth)
    const observer = new ResizeObserver(() => setWidth(el.clientWidth))
    observer.observe(el)
    return () => observer.disconnect()
  }, [])
  return [ref, width] as const
}
