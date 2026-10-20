import { useEffect, useState } from 'react'
import { Breadcrumb, BreadcrumbItem, Button, InlineNotification, Theme } from '@carbon/react'
import { AnalysisWorkspace } from './AnalysisWorkspace'
import { useTheme } from './design/theme'
import { LangProvider, Rich, useCopy, useLangState, type Lang } from './i18n/lang'
import { LibraryTab } from './LibraryTab'
import { ResearchTab } from './ResearchTab'
import { PARTS } from './Term'
import { TopBar } from './TopBar'
import { WorkspaceTable } from './WorkspaceTable'
import { WorkspaceNav, type WorkspaceView } from './WorkspaceNav'
import type { Snapshot, WorkspaceSnapshot } from './types'

export default function App() {
  const [lang, setLang] = useLangState()
  return (
    <LangProvider value={lang}>
      <Workspace onLang={setLang} />
    </LangProvider>
  )
}

function Workspace({ onLang }: { onLang: (lang: Lang) => void }) {
  const [theme, setTheme] = useTheme()
  const { t } = useCopy()
  const light = theme === 'light'
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
      <TopBar snap={snap} light={light} onTheme={(next) => setTheme(next ? 'light' : 'dark')} onLang={onLang} />
      <main className={`page${view === 'runs' ? ' analysis-page' : ''}`}>
        <WorkspaceNav view={view} pending={pending} />
        {directionId !== null && (
          <Breadcrumb noTrailingSlash aria-label={t('crumb.label')}>
            <BreadcrumbItem href="/">{t('crumb.workspace')}</BreadcrumbItem>
            {view === 'runs'
              ? <BreadcrumbItem href={researchUrl}>{t('crumb.back', { where: `${snap?.direction ?? directionId}${selected ? ` · ${selected}` : ''}` })}</BreadcrumbItem>
              : <BreadcrumbItem isCurrentPage>{t('crumb.direction')}</BreadcrumbItem>}
            {view === 'runs' && <BreadcrumbItem isCurrentPage>{t('crumb.analysis')}</BreadcrumbItem>}
          </Breadcrumb>
        )}
        <header className="page-head">
          <div className="page-head-row">
            <h1>
              {view === 'runs' ? t('head.runs') : view === 'library' ? t('head.library') : directionId !== null
                ? (snap ? <span translate="no">{snap.direction}</span> : loading ? t('state.loading') : t('head.direction'))
                : workspace
                  ? <Rich k="head.workspaceNamed" vars={{ name: workspace.name }} parts={{ name: (s) => <span translate="no">{s}</span> }} />
                  : loading ? t('state.loading') : t('head.workspace')}
            </h1>
            <Button kind="tertiary" size="sm" onClick={() => setRefreshVersion((v) => v + 1)} disabled={loading}>
              {t('action.refresh')}
            </Button>
          </div>
          <p className="muted">
            {view === 'runs'
              ? t('lede.runs')
              : view === 'library' ? t('lede.library') : directionId === null
              ? <Rich k="lede.workspace" parts={PARTS} />
              : <Rich k="lede.direction" parts={PARTS} />}
          </p>
        </header>

        {error && (
          <InlineNotification
            kind="error"
            statusIconDescription={t('icon.error')}
            lowContrast
            hideCloseButton
            title={t('error.read')}
            subtitle={t('error.readHint', { error })}
          />
        )}

        {view === 'research' && snap && snap.hypotheses.length === 0 && (
          <InlineNotification
            kind="info"
            statusIconDescription={t('icon.info')}
            lowContrast
            hideCloseButton
            title={t('empty.hypotheses')}
            subtitle={t('empty.hypothesesHint')}
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
