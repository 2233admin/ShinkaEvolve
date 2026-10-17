import { useCallback, useEffect, useState } from 'react'
import { Button, InlineNotification, Theme } from '@carbon/react'
import { EventStream } from './EventStream'
import { HypothesisTable } from './HypothesisTable'
import { SpecCard } from './SpecCard'
import { Term } from './Term'
import { TopBar } from './TopBar'
import type { Snapshot } from './types'

const THEME_KEY = 'shinka-theme'

function initialLight(): boolean {
  try {
    return localStorage.getItem(THEME_KEY) === 'light'
  } catch {
    return false
  }
}

async function loadSnapshot(): Promise<Snapshot> {
  const res = await fetch('/api/snapshot', { cache: 'no-store' })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export default function App() {
  const [light, setLight] = useState(initialLight)
  const [snap, setSnap] = useState<Snapshot | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<string | null>(null)

  useEffect(() => {
    document.documentElement.dataset.theme = light ? 'light' : 'dark'
    document.body.classList.toggle('light-theme', light)
    try {
      localStorage.setItem(THEME_KEY, light ? 'light' : 'dark')
    } catch {
      /* 存不了就不记 */
    }
  }, [light])

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      setSnap(await loadSnapshot())
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh()
  }, [refresh])

  const current = snap?.hypotheses.find((h) => h.id === selected) ?? snap?.hypotheses[0] ?? null

  return (
    <Theme theme={light ? 'white' : 'g100'}>
      <TopBar snap={snap} light={light} onTheme={setLight} />
      <main className="page">
        <header className="page-head">
          <div className="page-head-row">
            <h1>{snap ? snap.direction : loading ? '载入中…' : '研究方向'}</h1>
            <Button kind="tertiary" size="sm" onClick={() => void refresh()} disabled={loading}>
              刷新
            </Button>
          </div>
          <p className="muted">
            每个假设在<Term k="insample">样本内</Term>进化，再由质疑者在<Term k="validation">验证段</Term>检验；
            <Term k="validation">验证段</Term>结果只在出结论后显示，<Term k="sealed">封存段</Term>只在最终确认时打开一次。
          </p>
        </header>

        {error && (
          <InlineNotification
            kind="error"
            lowContrast
            hideCloseButton
            title="读取失败："
            subtitle={`${error}。确认 shinka_research ui 还在运行，然后点刷新。`}
          />
        )}

        {snap && snap.hypotheses.length === 0 && (
          <InlineNotification
            kind="info"
            lowContrast
            hideCloseButton
            title="还没有假设："
            subtitle="用 shinka_research hypothesis add 添加第一个假设。"
          />
        )}

        {snap && current && (
          <div className="grid">
            <section className="area-table">
              <h2 className="section-title">假设</h2>
              <div className="table-wrap">
                <HypothesisTable hyps={snap.hypotheses} selected={current.id} onSelect={setSelected} />
              </div>
            </section>
            <aside className="area-side">
              <SpecCard h={current} />
            </aside>
            <section className="area-events">
              <h2 className="section-title">事件</h2>
              <EventStream events={snap.events} />
            </section>
          </div>
        )}
      </main>
    </Theme>
  )
}
