import { useEffect, useLayoutEffect, useState } from 'react'
import { Breadcrumb, BreadcrumbItem, Button, InlineNotification, Theme } from '@carbon/react'
import { AnalysisWorkspace } from './AnalysisWorkspace'
import { LibraryTab } from './LibraryTab'
import { ResearchTab } from './ResearchTab'
import { Term } from './Term'
import { TopBar } from './TopBar'
import { WorkspaceTable } from './WorkspaceTable'
import { WorkspaceNav, type WorkspaceView } from './WorkspaceNav'
import type { Snapshot, WorkspaceSnapshot } from './types'

const THEME_KEY = 'shinka-theme'

function initialLight(): boolean {
  try {
    return localStorage.getItem(THEME_KEY) === 'light'
  } catch {
    return false
  }
}


export default function App() {
  const [light, setLight] = useState(initialLight)
  const [snap, setSnap] = useState<Snapshot | null>(null)
  const [workspace, setWorkspace] = useState<WorkspaceSnapshot | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const params = new URLSearchParams(window.location.search)
  const [selected, setSelected] = useState<string | null>(() => params.get('hypothesis'))
  const [directionId] = useState(() => params.get('direction'))
  const requestedView = params.get('view')
  const view: WorkspaceView = requestedView === 'runs' || requestedView === 'library' ? requestedView : 'research'
  const [refreshVersion, setRefreshVersion] = useState(0)

  // layout effect：先于子组件读取主题色值的 passive effect 生效。
  useLayoutEffect(() => {
    document.documentElement.dataset.theme = light ? 'light' : 'dark'
    document.body.classList.toggle('light-theme', light)
    try {
      localStorage.setItem(THEME_KEY, light ? 'light' : 'dark')
    } catch {
      /* 存不了就不记 */
    }
  }, [light])

  useEffect(() => {
    const controller = new AbortController()
    const endpoint = directionId === null
      ? '/api/workspace'
      : `/api/snapshot?direction=${encodeURIComponent(directionId)}`
    setLoading(true)
    async function load() {
      try {
        const res = await fetch(endpoint, { cache: 'no-store', signal: controller.signal })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        if (controller.signal.aborted) return
        if (directionId === null) setWorkspace(data)
        else setSnap(data)
        setError(null)
      } catch (e) {
        if (!controller.signal.aborted) setError(e instanceof Error ? e.message : String(e))
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }
    void load()
    return () => controller.abort()
  }, [directionId, refreshVersion])

  const lib = snap?.library ?? workspace?.library ?? null
  const pending = lib?.pending.length ?? 0
  const current = snap?.hypotheses.find((h) => h.id === selected) ?? snap?.hypotheses[0] ?? null
  const researchParams = new URLSearchParams()
  if (directionId !== null) researchParams.set('direction', directionId)
  if (selected !== null) researchParams.set('hypothesis', selected)
  const researchUrl = `/?${researchParams}`
  function selectHypothesis(id: string) {
    setSelected(id)
    const url = new URL(window.location.href)
    url.searchParams.set('hypothesis', id)
    window.history.replaceState(null, '', url)
  }

  return (
    <Theme theme={light ? 'white' : 'g100'}>
      <TopBar snap={snap} light={light} onTheme={setLight} />
      <main className={`page${view === 'runs' ? ' analysis-page' : ''}`}>
        <WorkspaceNav view={view} pending={pending} />
        {directionId !== null && (
          <Breadcrumb noTrailingSlash>
            <BreadcrumbItem href="/">工作区</BreadcrumbItem>
            {view === 'runs'
              ? <BreadcrumbItem href={researchUrl}>{`返回研究 · ${snap?.direction ?? directionId}${selected ? ` · ${selected}` : ''}`}</BreadcrumbItem>
              : <BreadcrumbItem isCurrentPage>研究方向</BreadcrumbItem>}
            {view === 'runs' && <BreadcrumbItem isCurrentPage>实验分析</BreadcrumbItem>}
          </Breadcrumb>
        )}
        <header className="page-head">
          <div className="page-head-row">
            <h1>{view === 'runs' ? '运行与分析' : view === 'library' ? '共享资料库' : directionId !== null
              ? snap?.direction ?? (loading ? '载入中…' : '研究方向')
              : workspace ? `工作区 · ${workspace.name}` : loading ? '载入中…' : '工作区'}</h1>
            <Button kind="tertiary" size="sm" onClick={() => setRefreshVersion((v) => v + 1)} disabled={loading}>
              刷新
            </Button>
          </div>
          <p className="muted">
            {view === 'runs'
              ? <>浏览运行、比较结果并检查完整实验分析。研究由对话主导；网页只读，私有指标不公开。</>
              : view === 'library' ? <>所有研究方向共享同一份资料库。</> : directionId === null
              ? <>选择研究方向查看假设与结论。<Term k="trials">试验数</Term>按方向累计；
                <Term k="sealed">封存段</Term>状态按方向展示。同一工作区共用一个资料库，研究操作仍通过 CLI 执行。</>
              : <>每个假设在<Term k="insample">样本内</Term>进化，再由质疑者在<Term k="validation">验证段</Term>检验；
                <Term k="validation">验证段</Term>结果只在出结论后显示，<Term k="sealed">封存段</Term>只在最终确认时打开一次。</>}
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

        {view === 'research' && snap && snap.hypotheses.length === 0 && (
          <InlineNotification
            kind="info"
            lowContrast
            hideCloseButton
            title="还没有假设："
            subtitle="用 shinka_research hypothesis add 添加第一个假设。"
          />
        )}

        {view === 'runs'
          ? <AnalysisWorkspace key={refreshVersion} light={light} />
          : view === 'library'
            ? <LibraryTab lib={lib} />
            : workspace
              ? <WorkspaceTable directions={workspace.directions} />
              : current && snap && <ResearchTab snap={snap} current={current} light={light} onSelect={selectHypothesis} />}
      </main>
    </Theme>
  )
}
