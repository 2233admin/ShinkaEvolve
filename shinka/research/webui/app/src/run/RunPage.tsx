import { useMemo } from 'react'
import { AnalysisWorkspace } from '../AnalysisWorkspace'
import { isCorrect } from '../compare/series'
import { Icon } from '../design/icons'
import { useSize } from '../design/useWidth'
import { int, money } from '../format'
import { useCopy } from '../i18n/lang'
import { Notice } from '../Notice'
import { runCost } from '../runs/data'
import { analysisView, routeHref, SHIPPED_VIEWS, viewRoute, type RunRoute } from '../shell/route'
import type { ShellData } from '../shell/useShellData'
import { bestOf, bestPrompts, generationOf, islandOf, metaUpdates, type RunProgram } from './data'
import { Inspector, paneOf, type Pane } from './Inspector'
import { Lineage, PANEL_SECTIONS, type PanelSection } from './Lineage'
import { LineagePanel } from './LineagePanel'
import { Scrubber } from './Scrubber'
import { useRunData } from './useRunData'

// The lane gutter and right margin (Lineage), so the scrubber's track lines up with the lanes' axis.
const LANE_LEFT = 128
const LANE_RIGHT = 28

/** The analysis views shipped so far, as the top bar's view tabs (design.md › 迁移与删除). */
export function ViewTabs({ route }: { route: RunRoute }) {
  const { t } = useCopy()
  const view = analysisView(route)
  return (
    <nav className="vtabs" aria-label={t('ra.views')}>
      {SHIPPED_VIEWS.map((tab) => (
        <a key={tab} href={routeHref(viewRoute(route, tab))} aria-current={view === tab ? 'page' : undefined}>
          {t('ra.view.lineage')}
        </a>
      ))}
    </nav>
  )
}

type Props = {
  route: RunRoute
  data: ShellData
  name: string
  go: (href: string, options?: { replace?: boolean }) => void
  sync: () => void
  light: boolean
  /** The router entry and the manual reload count: a new one remounts the restored page. */
  frameKey: string
  onRunList: () => void
}

/** One run's analysis (6.1, 6.2): the run bar, then Lineage with the resident inspector, or the restored page. */
export function RunPage({ route, data, name, go, sync, light, frameKey, onRunList }: Props) {
  const { t } = useCopy()
  const view = analysisView(route)
  const read = useRunData(route.run, data.runs)
  const run = read.data
  const programs = useMemo(() => run?.programs ?? [], [run])
  const byId = useMemo(() => new Map(programs.map((p) => [p.id, p])), [programs])
  const best = useMemo(() => bestOf(programs), [programs])
  const bestSet = useMemo(() => new Set(best.map((p) => p.id)), [best])
  const updates = useMemo(() => metaUpdates(run?.metaFiles ?? [], programs), [run, programs])
  const promptBest = useMemo(() => bestPrompts(run?.prompts ?? []), [run])
  const [canvas, size] = useSize<HTMLElement>()

  const replace = (next: RunRoute) => go(routeHref(next), { replace: true })
  const select = (id: string) => replace({ ...route, program: id })
  const maxGen = programs.reduce((m, p) => Math.max(m, generationOf(p)), 0)
  const asked = route.gen === null ? NaN : Number(route.gen)
  const upTo = Number.isInteger(asked) ? Math.max(0, Math.min(maxGen, asked)) : maxGen
  const pane: Pane = paneOf(route.pane)
  const panel = PANEL_SECTIONS.find((s) => s === route.panel) ?? null
  const shown: RunProgram | null = route.program ? (byId.get(route.program) ?? null) : (best[0] ?? null)
  const missing = route.program && run && !byId.has(route.program) ? route.program : null

  const valid = programs.filter(isCorrect).length
  const islands = new Set(programs.map(islandOf).filter((k) => k !== null)).size
  const generations = new Set(programs.map(generationOf)).size
  const cost = run ? runCost(run.stats) : null
  const legacy = view === 'legacy'
  const bar = (
    <div className="runbar" role="group" aria-label={t('ra.runbar')}>
      <span className="mono trunc" translate="no" title={route.run}>{name}</span>
      {run && (
        <span className="mono faint">
          {t('ra.programs', { n: int(programs.length) })} · {t('ra.valid', { n: int(valid) })} ·{' '}
          <span className={programs.length - valid > 0 ? 'sig' : undefined}>{t('ra.failedN', { n: int(programs.length - valid) })}</span> ·{' '}
          {t('ra.gens', { n: int(generations) })} · {t('ra.islands', { n: int(islands) })} · {t('ra.costIs', { cost: cost === null ? '—' : money(cost) })}
        </span>
      )}
      {read.error !== null && run !== null && <span className="small sig">{t('ra.stale', { error: read.error })}</span>}
      <a className="runbar-legacy" href={routeHref(viewRoute(route, 'legacy'))} aria-current={legacy ? 'page' : undefined} title={t('ra.legacyHint')}>
        {t('ra.legacy')}
        <Icon name="arrow" />
      </a>
    </div>
  )

  if (legacy) {
    return (
      <>
        <h1 className="sr-only">{t('ra.title')}</h1>
        {bar}
        <div className="legacy-frame">
          <AnalysisWorkspace key={frameKey} route={route} light={light} onProject={sync} onRunList={onRunList} />
        </div>
      </>
    )
  }

  let body
  if (run === null) {
    body = read.error ? (
      <div className="analysis-state">
        <Notice kind="error" label={t('icon.error')} title={t('ra.readFailed')} subtitle={read.error === 'HTTP 503' ? t('ra.busy') : read.error} />
      </div>
    ) : (
      <p className="analysis-state muted">{t('ra.loading')}</p>
    )
  } else if (programs.length === 0) {
    body = (
      <div className="analysis-state">
        <Notice kind="info" label={t('icon.info')} title={t('ra.empty')} subtitle={t('ra.emptyHint')} />
      </div>
    )
  } else {
    const openPanel = (section: PanelSection, extra: { prompt?: string; meta?: number } = {}) => {
      const toggle = section === panel && extra.prompt === undefined && extra.meta === undefined
      replace({
        ...route,
        panel: toggle ? null : section,
        prompt: extra.prompt ?? route.prompt,
        meta: extra.meta === undefined ? route.meta : String(extra.meta),
      })
    }
    const lanePanel = panel && (
      <LineagePanel
        section={panel}
        onClose={() => replace({ ...route, panel: null })}
        run={route.run}
        programs={programs}
        byId={byId}
        best={best}
        prompts={run.prompts}
        promptsError={run.promptsError}
        hasPromptEvo={run.stats.has_prompt_evo}
        bestPrompts={promptBest}
        promptId={route.prompt}
        onPrompt={(id) => replace({ ...route, prompt: id })}
        updates={updates}
        metaError={run.metaError}
        meta={route.meta === null ? null : Number(route.meta)}
        onMeta={(processed) => replace({ ...route, meta: String(processed) })}
        selected={shown?.id ?? null}
        onSelect={select}
        again={data.runs}
        maxWidth={Math.max(360, size.width - 200)}
      />
    )
    body = (
      <div className="analysis">
        <section className="canvas" ref={canvas} aria-label={t('ra.view.lineage')}>
          <Lineage
            programs={programs}
            byId={byId}
            upTo={upTo}
            maxGen={maxGen}
            selected={shown}
            onSelect={select}
            prompts={run.prompts}
            promptsError={run.promptsError}
            hasPromptEvo={run.stats.has_prompt_evo}
            bestPrompts={promptBest}
            updates={updates}
            metaError={run.metaError}
            panel={panel}
            onPanel={openPanel}
            panelNode={lanePanel}
            scrubber={
              <Scrubber
                max={maxGen}
                value={upTo}
                onChange={(g) => replace({ ...route, gen: g === maxGen ? null : String(g) })}
                count={programs.filter((p) => generationOf(p) <= upTo).length}
                left={LANE_LEFT}
                right={LANE_RIGHT}
              />
            }
          />
        </section>
        <Inspector
          run={route.run}
          program={shown}
          missing={missing}
          programs={programs}
          byId={byId}
          best={bestSet}
          pane={pane}
          onPane={(next) => replace({ ...route, pane: next === 'code' ? null : next })}
          onSelect={select}
          updates={updates}
          metaError={run.metaError}
          again={data.runs}
        />
      </div>
    )
  }

  return (
    <>
      <h1 className="sr-only">{t('ra.title')}</h1>
      {bar}
      {body}
    </>
  )
}
