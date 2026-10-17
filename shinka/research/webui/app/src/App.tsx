import { useEffect, useLayoutEffect, useState } from 'react'
import { Breadcrumb, BreadcrumbItem, Button, InlineNotification, Tab, TabList, TabPanel, TabPanels, Tabs, Tag, Theme } from '@carbon/react'
import { LibraryTab } from './LibraryTab'
import { ResearchTab } from './ResearchTab'
import { Term } from './Term'
import { TopBar } from './TopBar'
import { WorkspaceTable } from './WorkspaceTable'
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
  const [selected, setSelected] = useState<string | null>(null)
  const [directionId] = useState(() => new URLSearchParams(window.location.search).get('direction'))
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

  return (
    <Theme theme={light ? 'white' : 'g100'}>
      <TopBar snap={snap} light={light} onTheme={setLight} />
      <main className="page">
        {directionId !== null && (
          <Breadcrumb noTrailingSlash>
            <BreadcrumbItem href="/">工作区</BreadcrumbItem>
            <BreadcrumbItem isCurrentPage>研究方向</BreadcrumbItem>
          </Breadcrumb>
        )}
        <header className="page-head">
          <div className="page-head-row">
            <h1>{directionId !== null
              ? snap?.direction ?? (loading ? '载入中…' : '研究方向')
              : workspace ? `工作区 · ${workspace.name}` : loading ? '载入中…' : '工作区'}</h1>
            <Button kind="tertiary" size="sm" onClick={() => setRefreshVersion((v) => v + 1)} disabled={loading}>
              刷新
            </Button>
          </div>
          <p className="muted">
            {directionId === null
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

        {snap && snap.hypotheses.length === 0 && (
          <InlineNotification
            kind="info"
            lowContrast
            hideCloseButton
            title="还没有假设："
            subtitle="用 shinka_research hypothesis add 添加第一个假设。"
          />
        )}

        {(snap || workspace) && (
          <Tabs>
            <TabList aria-label="内容" contained={false}>
              <Tab>{directionId === null ? '工作区' : '研究'}</Tab>
              <Tab>
                <span className="tab-label">
                  资料库
                  {pending > 0 && (
                    <Tag size="sm" type="gray" title={`${pending} 份资料待整理`}>
                      {pending}
                    </Tag>
                  )}
                </span>
              </Tab>
            </TabList>
            <TabPanels>
              <TabPanel>
                {workspace
                  ? <WorkspaceTable directions={workspace.directions} />
                  : current && snap && <ResearchTab snap={snap} current={current} light={light} onSelect={setSelected} />}
              </TabPanel>
              <TabPanel>
                <LibraryTab lib={lib} />
              </TabPanel>
            </TabPanels>
          </Tabs>
        )}
      </main>
    </Theme>
  )
}
