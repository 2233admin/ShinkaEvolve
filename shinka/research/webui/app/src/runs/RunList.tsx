import { DefinitionTooltip, InlineNotification } from '@carbon/react'
import { useEffect, useReducer, useRef, type ReactNode } from 'react'
import { untagged } from '../demo'
import { directionState, STATE_LOOK } from '../direction'
import { Glyph, Icon } from '../design/icons'
import { fmt, int, money, when } from '../format'
import type { CopyKey } from '../i18n/copy'
import { useCopy } from '../i18n/lang'
import { routeHref, runRoute, type Route } from '../shell/route'
import type { ShellData } from '../shell/useShellData'
import type { RunEntry } from '../types'
import { groupRuns, matchesRun, runCost, runKind, runLabel, runTotals, runtimeText, RUN_SORTS, type RunKind, type RunSort, type RunStats, type RunTotals } from './data'
import { useRunStats } from './useRunStats'

type RunsRoute = Extract<Route, { view: 'runs' }>

type Props = { data: ShellData; route: RunsRoute; go: (href: string, options?: { replace?: boolean }) => void }

// P01-Q1's thresholds kept without a second colour: faint at or below the low line, text above it,
// strong above the high one (the signal orange is for budgets and failures only).
const STALL: [number, number] = [10, 50]
const COST: [number, number] = [1, 5]
const tier = (value: number, [low, high]: [number, number]) => (value > high ? 'is-high' : value > low ? undefined : 'faint')

const SORT_COPY: Record<RunSort, CopyKey> = { updated: 'runs.sortUpdated', name: 'runs.sortName', score: 'runs.sortScore', cost: 'runs.sortCost' }
const KINDS: ['' | RunKind, CopyKey][] = [['', 'runs.kindAll'], ['ordinary', 'runs.kindOrdinary'], ['experiment', 'runs.kindExperiment']]
const collapsedKey = (group: string) => `collapsed-task-${encodeURIComponent(group)}`
const epoch = (seconds: number | null) => (seconds === null ? '—' : when(new Date(seconds * 1000).toISOString()))

/** The run's statistics in the order of the columns; a run without programs says so in the first. */
function StatCells({ stats }: { stats: RunStats }) {
  const { t } = useCopy()
  const empty = stats.program_count === 0
  const failed = stats.program_count - stats.correct_count
  const stall = stats.gens_since_improvement
  const cost = runCost(stats)
  return (
    <>
      <span role="cell" className="num mono">
        {empty ? (
          <span className="rempty">
            <Glyph kind="dir-new" />
            {t('runs.empty')}
          </span>
        ) : (
          int(stats.program_count)
        )}
      </span>
      <span role="cell" className="num mono">{int(stats.correct_count)}</span>
      <span role="cell" className={failed ? 'num mono' : 'num mono faint'}>{int(failed)}</span>
      <span role="cell" className="num mono">{fmt(stats.best_score)}</span>
      <span role="cell" className="num mono">{empty || stats.max_generation === null ? '—' : int(stats.max_generation)}</span>
      <span role="cell" className="num mono">
        {empty || stall === null ? (
          '—'
        ) : stall === 0 ? (
          // The legacy list's ✓: the latest generation improved the best.
          <>
            <span aria-hidden="true">✓</span>
            <span className="sr-only">0</span>
          </>
        ) : (
          <span className={tier(stall, STALL)}>
            {int(stall)}
            {stall > STALL[1] && <em className="rstalled">{t('runs.stalled')}</em>}
          </span>
        )}
      </span>
      <span role="cell" className="num mono">
        {stats.has_prompt_evo ? (
          <DefinitionTooltip
            definition={stats.prompt_evo_cost === null ? t('runs.promptUnknown') : t('runs.promptCost', { cost: money(stats.prompt_evo_cost) })}
            openOnHover
            autoAlign
          >
            {int(stats.prompt_count)}
          </DefinitionTooltip>
        ) : (
          <span className="faint">—</span>
        )}
      </span>
      <span role="cell" className="num mono">
        {cost === null ? (
          <span className="faint" title={t('runs.costUnknown')}>—</span>
        ) : (
          <span className={tier(cost, COST)}>{money(cost)}</span>
        )}
      </span>
      <span role="cell" className="num mono">{runtimeText(stats.total_runtime_seconds)}</span>
      <span role="cell" className="num mono faint">{epoch(stats.last_update)}</span>
    </>
  )
}

/**
 * Every run the workspace can read (award-redesign 5.1, P01): grouped by evaluator — research
 * experiments under their direction, ordinary runs under their task — with search, kind and sort in
 * the URL, a selection that survives filtering and opens the compare page from two runs, collapsible
 * groups remembered as the legacy list remembered them, and totals over the rows shown.
 */
export function RunList({ data, route, go }: Props) {
  const { t } = useCopy()
  const search = useRef<HTMLInputElement>(null)
  // Collapsed groups live in localStorage under the legacy list's keys (P01-C4); a change redraws.
  const [, redraw] = useReducer((n: number) => n + 1, 0)
  const runs = data.runs
  const stats = useRunStats(runs)
  const sort: RunSort = RUN_SORTS.includes(route.sort as RunSort) ? (route.sort as RunSort) : 'updated'
  const kind: '' | RunKind = route.kind === 'ordinary' || route.kind === 'experiment' ? route.kind : ''
  const update = (next: Partial<RunsRoute>) => go(routeHref({ ...route, ...next }), { replace: true })

  // `/` focuses the search unless the reader is typing somewhere already (P01-C1).
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (event.key !== '/' || event.ctrlKey || event.metaKey || event.altKey) return
      if (target && (target.isContentEditable || target.closest('input, textarea, select'))) return
      event.preventDefault()
      search.current?.focus()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  if (!data.listed) return <p className="muted">{t('runs.loading')}</p>
  if (runs === null)
    return (
      <InlineNotification
        kind="error"
        statusIconDescription={t('icon.error')}
        lowContrast
        hideCloseButton
        title={t('runs.listError')}
        subtitle={t('runs.listErrorHint', { error: data.runsError ?? '' })}
      />
    )

  const entries = data.workspace?.directions ?? []
  const directions = new Set(entries.map((d) => d.id))
  const titles = new Map(entries.map((d) => [d.id, d] as const))
  const counts: Record<'' | RunKind, number> = { '': runs.length, ordinary: 0, experiment: 0 }
  for (const r of runs) counts[runKind(r, directions)] += 1
  const shown = runs.filter((r) => (kind === '' || runKind(r, directions) === kind) && matchesRun(r, route.q))
  const loaded = new Map([...stats].flatMap(([path, entry]) => (entry.stats === null ? [] : [[path, entry.stats] as const])))
  const groups = groupRuns(shown, sort, loaded, directions)
  // P01-Q2: the totals cover every run, whatever the filters show; a filter adds its own subtotal.
  const totals = runTotals(runs, loaded)
  const filtered = kind !== '' || route.q.trim() !== '' ? runTotals(shown, loaded) : null
  const unknown = totals.unknownCost + totals.missing
  const selected = new Set(route.select)
  const toggle = (path: string) => update({ select: selected.has(path) ? route.select.filter((p) => p !== path) : [...route.select, path] })
  const compare = route.select.length >= 2 ? routeHref({ view: 'compare', runs: route.select }) : null

  const row = (r: RunEntry) => {
    const label = runLabel(r)
    const on = selected.has(r.path)
    const s = stats.get(r.path)
    return (
      <div className={on ? 'rrow is-selected' : 'rrow'} role="row" key={r.path} aria-busy={s === undefined || undefined}>
        <span role="cell">
          <input type="checkbox" className="rcheck" checked={on} onChange={() => toggle(r.path)} aria-label={t('runs.select', { run: label })} />
        </span>
        <span role="cell" className="rname">
          <a className="rlink mono" href={routeHref(runRoute(r.path))} translate="no">
            {label}
          </a>
          {s?.failed && s.stats !== null && (
            // The latest read failed: the numbers on this row are from the previous one.
            <span className="rstale">
              <Glyph kind="error" signal />
              <span className="sig">{t('runs.readFailedStale')}</span>
            </span>
          )}
        </span>
        {s === undefined ? (
          <span role="cell" className="rstate faint">{t('runs.loadingRow')}</span>
        ) : s.stats === null ? (
          <span role="cell" className="rstate">
            <Glyph kind="error" signal />
            <span className="sig">{t('runs.readFailed')}</span>
          </span>
        ) : (
          <StatCells stats={s.stats} />
        )}
      </div>
    )
  }

  const group = (key: string, groupKind: RunKind, list: RunEntry[]) => {
    const collapsed = localStorage.getItem(collapsedKey(key)) === 'true'
    const dir = groupKind === 'experiment' ? titles.get(key) : undefined
    const look = dir && dir.error === null ? STATE_LOOK[directionState(dir)] : null
    const scores = list.flatMap((r) => {
      const b = loaded.get(r.path)?.best_score
      return b === null || b === undefined ? [] : [b]
    })
    const flip = () => {
      localStorage.setItem(collapsedKey(key), String(!collapsed))
      redraw()
    }
    return (
      <div className="rgroupwrap" key={key} role="rowgroup">
        <div className="rgroup" role="row">
          <span role="cell" className="rgroup-name">
            <button type="button" className="rgroup-toggle" aria-expanded={!collapsed} onClick={flip}>
              <span className={collapsed ? 'rchev is-shut' : 'rchev'}>
                <Icon name="chevron" />
              </span>
              {look && <Glyph kind={look.glyph} signal={look.signal} />}
              {dir ? <b translate="no">{untagged(dir.direction)}</b> : <b className="mono" translate="no">{key}</b>}
              {dir && <span className="mono faint" translate="no">{key}</span>}
            </button>
          </span>
          <span role="cell" className="rgroup-meta faint">
            {t(groupKind === 'experiment' ? 'runs.groupExperiments' : 'runs.groupOrdinary', { n: list.length })}
            {scores.length > 0 && <> · {t('runs.groupBest', { score: fmt(Math.max(...scores)) })}</>}
          </span>
        </div>
        {!collapsed && list.map(row)}
      </div>
    )
  }

  const totalsRow = (label: string, sums: RunTotals, className: string) => {
    const open = sums.unknownCost + sums.missing
    return (
      <div className={className} role="row">
        <span role="cell" />
        <span role="cell">{label}</span>
        <span role="cell" className="num mono">{int(sums.programs)}</span>
        <span role="cell" className="num mono">{int(sums.valid)}</span>
        <span role="cell" className="num mono">{int(sums.failed)}</span>
        <span role="cell" className="num mono faint">{fmt(sums.peak)}</span>
        <span role="cell" className="num faint">—</span>
        <span role="cell" className="num faint">—</span>
        <span role="cell" className="num mono">{int(sums.prompts)}</span>
        <span role="cell" className="num mono">{`${open > 0 ? '≥ ' : ''}${money(sums.cost)}`}</span>
        <span role="cell" className="num mono">{runtimeText(sums.runtime)}</span>
        <span role="cell" />
      </div>
    )
  }

  const header = (label: CopyKey, num = true): ReactNode => (
    <span role="columnheader" className={num ? 'num' : undefined}>
      {t(label)}
    </span>
  )

  return (
    <>
      <div className="rtoolbar">
        <label className="rsearch">
          <Icon name="search" />
          <input
            ref={search}
            type="search"
            value={route.q}
            placeholder={t('runs.search')}
            aria-label={t('runs.search')}
            aria-keyshortcuts="/"
            onChange={(event) => update({ q: event.currentTarget.value })}
          />
          <kbd aria-hidden="true">/</kbd>
        </label>
        <div className="seg" role="group" aria-label={t('runs.kind')}>
          {KINDS.map(([value, label]) => (
            <button key={value} type="button" aria-pressed={kind === value} onClick={() => update({ kind: value })}>
              {t(label)} <span className="mono">{counts[value]}</span>
            </button>
          ))}
        </div>
        <label className="select">
          {t('runs.sort')}
          <select value={sort} onChange={(event) => update({ sort: event.currentTarget.value === 'updated' ? '' : event.currentTarget.value })}>
            {RUN_SORTS.map((s) => (
              <option key={s} value={s}>
                {t(SORT_COPY[s])}
              </option>
            ))}
          </select>
        </label>
        <span className="selbar">
          <span>{t('runs.selectedN', { n: route.select.length })}</span>
          {compare ? (
            <a className="btn is-primary" href={compare}>
              {t('runs.compareSelected')}
              <Icon name="arrow" />
            </a>
          ) : (
            <span className="btn is-disabled" aria-disabled="true" title={t('runs.compareNeedsTwo')}>
              {t('runs.compareSelected')}
              <Icon name="arrow" />
            </span>
          )}
        </span>
      </div>
      <p className="caveat">
        <Glyph kind="challenged" />
        <span>{t('runs.caveat')}</span>
      </p>
      {data.runsError !== null && <p className="small sig">{t('runs.listStale', { error: data.runsError })}</p>}
      {runs.length === 0 ? (
        <div className="rnone">
          <p>{t('runs.none')}</p>
          <p className="muted small">{t('runs.noneHint')}</p>
        </div>
      ) : (
        <>
          <div className="hscroll">
            <div className="rtable" role="table" aria-label={t('nav.runs')}>
              <div className="rrow rhead2" role="row">
                <span role="columnheader">
                  <span className="sr-only">{t('runs.colSelect')}</span>
                </span>
                {header('runs.colRun', false)}
                {header('runs.colPrograms')}
                {header('runs.colValid')}
                {header('runs.colFailed')}
                {header('runs.colBest')}
                {header('runs.colGen')}
                {header('runs.colStall')}
                {header('runs.colPrompts')}
                {header('runs.colCost')}
                {header('runs.colRuntime')}
                {header('runs.colUpdated')}
              </div>
              {groups.length === 0 ? <p className="rnomatch muted">{t('runs.noMatch')}</p> : groups.map((g) => group(g.key, g.kind, g.runs))}
              {filtered && totalsRow(t('runs.subtotal', { n: filtered.runs }), filtered, 'rrow rtotal is-sub')}
              {totalsRow(t('runs.totals', { n: totals.runs }), totals, 'rrow rtotal')}
            </div>
          </div>
          <p className="rkeynote">
            {unknown > 0 && <>{t('runs.lowerBound', { n: unknown })} · </>}
            {t('runs.peakNote')} {t('runs.tiers')}
          </p>
        </>
      )}
    </>
  )
}
