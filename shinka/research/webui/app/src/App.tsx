import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { InlineNotification, Theme } from '@carbon/react'
import { AnalysisWorkspace } from './AnalysisWorkspace'
import { DEMO_TAG, untagged } from './demo'
import { DirectionMeta } from './DirectionMeta'
import { useTheme } from './design/theme'
import { LangProvider, Rich, useCopy, useLangState, type Lang } from './i18n/lang'
import { LibraryTab } from './LibraryTab'
import { ResearchTab } from './ResearchTab'
import { Palette, type PaletteItem } from './shell/Palette'
import { Rail, TabBar } from './shell/Rail'
import { ALL_RUNS, routeHref, runRoute } from './shell/route'
import { TopBar, Updated, type Crumb } from './shell/TopBar'
import { useMatches } from './shell/useMatches'
import { useRouter } from './shell/useRouter'
import { useShellData } from './shell/useShellData'
import { Overview } from './overview/Overview'
import { PARTS } from './Term'

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
  const { lang, t } = useCopy()
  const light = theme === 'light'
  const { route, entry, go, sync } = useRouter()
  const data = useShellData(route.view === 'direction' ? route.direction : null)
  const [palette, setPalette] = useState(false)
  // A manual refresh also reloads a hosted analysis page; the automatic one leaves it alone.
  const [reload, setReload] = useState(0)
  // design.md › 响应式: the rail narrows below 1280px and on the analysis pages.
  const compact = useMatches('(max-width: 1279px)')

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setPalette(true)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const { workspace, snap, runs } = data
  const directions = workspace?.directions ?? []
  const names = useMemo(() => new Map((runs ?? []).map((run) => [run.path, run.name])), [runs])
  const lib = workspace?.library ?? null
  const refresh = () => {
    data.refresh()
    setReload((n) => n + 1)
  }
  const shown = { ...data, refresh }
  const demo = directions.some((d) => DEMO_TAG.test(d.direction))

  const titleOf = (id: string) => untagged(directions.find((d) => d.id === id)?.direction ?? id)
  const crumbs: Crumb[] = []
  const research = { label: t('nav.research'), href: routeHref({ view: 'directions' }) }
  const allRuns = { label: t('nav.runs'), href: routeHref(ALL_RUNS) }
  switch (route.view) {
    case 'directions':
      crumbs.push(research)
      break
    case 'direction':
      crumbs.push(research, { label: titleOf(route.direction), data: true })
      break
    case 'runs':
      crumbs.push(allRuns)
      break
    case 'compare':
      crumbs.push(allRuns, { label: t('crumb.compare', { n: route.runs.length }) })
      break
    case 'run':
      if (route.direction) {
        const back = routeHref({ view: 'direction', direction: route.direction, hypothesis: route.hypothesis })
        crumbs.push(research, { label: titleOf(route.direction), href: back, data: true }, { label: t('crumb.analysis') })
      } else {
        crumbs.push(allRuns, { label: names.get(route.run) ?? route.run, data: true })
      }
      break
    case 'library':
      crumbs.push({ label: t('nav.library'), href: routeHref({ view: 'library', page: null }) })
      if (route.page) crumbs.push({ label: route.page === 'log' ? t('lib.log') : route.page.replace(/^wiki\//, ''), data: route.page !== 'log' })
      break
  }
  const here = crumbs[crumbs.length - 1].label
  useEffect(() => {
    document.title = `${here} · Shinka Research`
  }, [here])

  const items: PaletteItem[] = [
    { key: 'v-research', group: 'pal.pages', label: t('nav.research'), act: () => go(routeHref({ view: 'directions' })) },
    { key: 'v-runs', group: 'pal.pages', label: t('nav.runs'), act: () => go(routeHref(ALL_RUNS)) },
    { key: 'v-library', group: 'pal.pages', label: t('nav.library'), act: () => go(routeHref({ view: 'library', page: null })) },
    ...directions.map((d): PaletteItem => ({
      key: `d-${d.id}`,
      group: 'pal.directions',
      label: untagged(d.direction),
      detail: d.id,
      data: true,
      act: () => go(routeHref({ view: 'direction', direction: d.id, hypothesis: null })),
    })),
    ...(runs ?? []).map((run): PaletteItem => ({
      key: `r-${run.path}`,
      group: 'pal.runs',
      label: run.name,
      detail: run.task,
      data: true,
      act: () => go(routeHref(runRoute(run.path))),
    })),
    ...(lib?.pages ?? []).map((page): PaletteItem => ({
      key: `p-${page.name}`,
      group: 'pal.library',
      label: page.name,
      data: true,
      act: () => go(routeHref({ view: 'library', page: `wiki/${page.name}` })),
    })),
    light
      ? { key: 'a-theme', group: 'pal.actions', label: t('pal.toDark'), act: () => setTheme('dark') }
      : { key: 'a-theme', group: 'pal.actions', label: t('pal.toLight'), act: () => setTheme('light') },
    lang === 'zh'
      ? { key: 'a-lang', group: 'pal.actions', label: t('pal.toEn'), act: () => onLang('en') }
      : { key: 'a-lang', group: 'pal.actions', label: t('pal.toZh'), act: () => onLang('zh') },
    { key: 'a-refresh', group: 'pal.actions', label: t('pal.refresh'), act: refresh },
  ]

  // The first load failed: say why. A later failure keeps the data and shows in the top bar.
  const readError = data.updated === null && data.error !== null && (
    <InlineNotification
      kind="error"
      statusIconDescription={t('icon.error')}
      lowContrast
      hideCloseButton
      title={t('error.read')}
      subtitle={t('error.readHint', { error: data.error })}
    />
  )
  const loading = <p className="muted">{t('state.loading')}</p>

  let page: ReactNode
  switch (route.view) {
    case 'directions':
      page = workspace ? <Overview workspace={workspace} freshness={<Updated data={shown} />} /> : <div className="home-empty">{readError || loading}</div>
      break
    case 'direction': {
      const current = snap?.hypotheses.find((h) => h.id === route.hypothesis) ?? snap?.hypotheses[0] ?? null
      const summary = directions.find((d) => d.id === route.direction)
      page = (
        <>
          <header className="page-head">
            <h1>{snap ? <span translate="no">{untagged(snap.direction)}</span> : t('head.direction')}</h1>
            {snap && <DirectionMeta snap={snap} entry={summary && summary.error === null ? summary : null} />}
            <p className="muted"><Rich k="lede.direction" parts={PARTS} /></p>
          </header>
          {data.error !== null && snap === null
            ? <InlineNotification kind="error" statusIconDescription={t('icon.error')} lowContrast hideCloseButton title={t('error.read')} subtitle={t('error.readHint', { error: data.error })} />
            : snap === null
              ? loading
              : snap.hypotheses.length === 0
                ? <InlineNotification kind="info" statusIconDescription={t('icon.info')} lowContrast hideCloseButton title={t('empty.hypotheses')} subtitle={t('empty.hypothesesHint')} />
                : current && (
                  <ResearchTab
                    snap={snap}
                    current={current}
                    light={light}
                    onSelect={(id) => go(routeHref({ view: 'direction', direction: route.direction, hypothesis: id }), { replace: true })}
                  />
                )}
        </>
      )
      break
    }
    case 'runs':
    case 'compare':
    case 'run':
      page = (
        <>
          <header className="page-head">
            <h1>{t('head.runs')}</h1>
            <p className="muted">{t('lede.runs')}</p>
          </header>
          {/* Selections carry run names as labels, so the run list and compare page wait for the listing. */}
          {route.view !== 'run' && !data.listed
            ? loading
            : <AnalysisWorkspace key={`${entry}:${reload}`} route={route} names={names} light={light} onProject={sync} />}
        </>
      )
      break
    case 'library':
      page = (
        <>
          <header className="page-head">
            <h1>{t('head.library')}</h1>
            <p className="muted">{t('lede.library')}</p>
          </header>
          {readError || (workspace
            ? <LibraryTab lib={lib} page={route.page} onPage={(next) => go(routeHref({ view: 'library', page: next }))} />
            : loading)}
        </>
      )
      break
  }

  const analysis = route.view === 'run' || route.view === 'compare'
  return (
    <Theme theme={light ? 'white' : 'g100'}>
      <a className="skip" href="#content">{t('shell.skip')}</a>
      <div className="shell">
        <Rail route={route} data={shown} narrow={analysis || compact} demo={demo} onSearch={() => setPalette(true)} />
        <div className="main">
          <TopBar crumbs={crumbs} data={shown} demo={demo} theme={theme} onTheme={setTheme} onLang={onLang} onSearch={() => setPalette(true)} />
          <main id="content" className={route.view === 'directions' ? 'page home-page' : analysis || route.view === 'runs' ? 'page analysis-page' : 'page'} tabIndex={-1}>
            {page}
          </main>
        </div>
      </div>
      <TabBar route={route} data={shown} />
      <Palette open={palette} onClose={() => setPalette(false)} items={items} />
    </Theme>
  )
}
