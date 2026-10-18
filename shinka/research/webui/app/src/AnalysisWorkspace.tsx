import { useEffect, useRef, useState } from 'react'
import { InlineNotification } from '@carbon/react'

const PAGES: Record<string, true> = {
  '/analysis/index.html': true,
  '/viz_tree.html': true,
  '/compare.html': true,
}

type Selection = { path: string; label: string }[]

function selection(text: string | null): Selection {
  const value: unknown = JSON.parse(text ?? 'null')
  if (!Array.isArray(value) || value.length < 2 || !value.every((item) =>
    typeof item === 'object' && item !== null && typeof item.path === 'string' && typeof item.label === 'string')) {
    throw new Error('比较链接缺少运行选择。请从“全部运行”选择至少两个运行。')
  }
  return value
}

function initialAnalysis(): { src: string | null; error: string | null } {
  try {
    const params = new URLSearchParams(window.location.search)
    const url = new URL(params.get('analysis') ?? '/analysis/index.html', window.location.origin)
    if (url.origin !== window.location.origin || !PAGES[url.pathname]) {
      throw new Error('分析链接无效。请从研究实验或“全部运行”重新进入。')
    }
    if (url.pathname === '/compare.html') {
      url.searchParams.set('selection', JSON.stringify(selection(params.get('compare') ?? url.searchParams.get('selection'))))
    }
    return { src: url.pathname + url.search + url.hash, error: null }
  } catch (error) {
    return { src: null, error: error instanceof Error ? error.message : String(error) }
  }
}

export function AnalysisWorkspace({ light }: { light: boolean }) {
  const [initial] = useState(initialAnalysis)
  const [error, setError] = useState(initial.error)
  const frame = useRef<HTMLIFrameElement>(null)
  const theme = light ? 'light' : 'dark'
  useEffect(() => {
    const child = frame.current?.contentWindow
    const apply = child && Reflect.get(child, 'shinkaSetTheme')
    if (typeof apply === 'function') apply(theme)
  }, [theme])

  function connect() {
    const child = frame.current?.contentWindow
    if (!child) return
    try {
      const apply = Reflect.get(child, 'shinkaSetTheme')
      if (typeof apply !== 'function' || typeof Reflect.get(child, 'ShinkaHTML') !== 'function') {
        throw new Error('分析资源未加载。请检查前端构建和静态资源后刷新页面。')
      }
      apply(theme)
      const sync = () => {
        const inside = new URL(child.location.href)
        if (inside.origin !== window.location.origin || !PAGES[inside.pathname]) {
          throw new Error('分析页面离开了工作区。请重新进入“全部运行”。')
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
      setError(error instanceof Error ? error.message : String(error))
    }
  }

  return (
    <section aria-label="运行浏览与实验分析">
      <p className="analysis-context">本页显示各运行的原始 <code>combined_score</code>，只供诊断；不同评估器的得分不可直接比较。研究结论和全方向试验的噪声基准以对应研究方向页为准，本页不另行推算。</p>
      {error && <InlineNotification kind="error" lowContrast hideCloseButton title="分析读取失败：" subtitle={error} />}
      {initial.src && (
        <div className="analysis-scroll">
          <iframe ref={frame} className="analysis-frame" title="运行浏览与实验分析" src={initial.src} onLoad={connect} />
        </div>
      )}
    </section>
  )
}
