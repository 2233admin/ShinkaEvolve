import { Notice } from '../Notice'
import { bisectCenter, curveStepAfter, line, max, scaleLinear } from 'd3'
import { Fragment, useId, useMemo, useState, type KeyboardEvent } from 'react'
import { untagged } from '../demo'
import { Glyph } from '../design/icons'
import { useWidth } from '../design/useWidth'
import { fmt, int, money, pct } from '../format'
import type { CopyKey } from '../i18n/copy'
import { useCopy } from '../i18n/lang'
import { runCost, runGroup, runLabel, type RunStats } from '../runs/data'
import { ALL_RUNS, routeHref, runRoute, type Route } from '../shell/route'
import type { ShellData } from '../shell/useShellData'
import { byGeneration, byGroup, bySpend, groupBoards, groupLeads, metricOf, METRICS, seriesStyle, stackLabels, stripDomain, unpricedCharge, valueAt, type Metric, type Point, type SeriesStyle as Style } from './series'
import { useCompareData } from './useCompareData'

type CompareRoute = Extract<Route, { view: 'compare' }>

type Props = { data: ShellData; route: CompareRoute; go: (href: string, options?: { replace?: boolean }) => void }

const METRIC_COPY: Record<Metric, CopyKey> = { best_score: 'cmp.metricBest', program_count: 'cmp.metricPrograms', correct_count: 'cmp.metricValid' }

/** A run's shape (hi-fi build.mjs › MARKS, extended to six): shape, never colour, tells runs apart. */
function MarkShape({ mark, muted, x, y, s }: { mark: number; muted: boolean; x: number; y: number; s: number }) {
  const ink = muted ? 'var(--muted)' : 'var(--text)'
  switch (mark) {
    case 0:
      return <circle cx={x} cy={y} r={s} fill={ink} />
    case 1:
      return <rect x={x - s + 0.8} y={y - s + 0.8} width={2 * s - 1.6} height={2 * s - 1.6} fill="var(--bg)" stroke={ink} strokeWidth={1.6} />
    case 2:
      return <path d={`M${x},${y - s - 1}l${s + 1},${2 * s}h${-2 * (s + 1)}z`} fill={ink} />
    case 3:
      return <path d={`M${x},${y - s - 0.6}l${s + 0.6},${s + 0.6}l${-s - 0.6},${s + 0.6}l${-s - 0.6},${-s - 0.6}z`} fill="var(--bg)" stroke={ink} strokeWidth={1.5} />
    case 4:
      return <path d={`M${x - s + 0.8},${y - s + 0.8}l${2 * s - 1.6},${2 * s - 1.6}m0,${-2 * s + 1.6}l${-2 * s + 1.6},${2 * s - 1.6}`} stroke={ink} strokeWidth={1.8} />
    default:
      return <path d={`M${x},${y + s + 1}l${s + 1},${-2 * s}h${-2 * (s + 1)}z`} fill="var(--bg)" stroke={ink} strokeWidth={1.5} />
  }
}

function RunMark({ style, size = 12, label }: { style: Style; size?: number; label?: string }) {
  return (
    <svg className="runmark" width={size} height={size} viewBox="0 0 14 14" role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
      <MarkShape mark={style.mark} muted={style.muted} x={7} y={7} s={4.2} />
    </svg>
  )
}

function LineSample({ style }: { style: Style }) {
  return (
    <svg width="22" height="8" viewBox="0 0 22 8" aria-hidden="true">
      <line x1="0" x2="22" y1="4" y2="4" stroke={style.muted ? 'var(--muted)' : 'var(--text)'} strokeWidth={1.6} strokeDasharray={style.dash || undefined} />
    </svg>
  )
}

type Series = { label: string; style: Style; points: Point[] }

const HEIGHT = 250
// Below this width the end labels leave the plot and sit under it.
const NARROW = 480
const CHAR = 6.9
const LABEL_MIN = 112

/** A long name keeps its start and its end, which is where run names differ. */
function shorten(text: string, chars: number): string {
  if (text.length <= chars) return text
  const head = Math.ceil((chars - 1) / 2)
  return `${text.slice(0, head)}…${text.slice(text.length - (chars - 1 - head))}`
}

/**
 * One step curve per run (P02-Q1, P02-Q2), each ending in its run's shape with its name and value.
 * Pointing at the chart, or stepping with ← →, reads every run's value at that point as the legacy
 * tooltips did. Counts start at zero, as the legacy charts did; the best keeps the range of its values.
 */
function CompareChart({ title, group, series, axis, format, zero }: { title: string; group: string | null; series: Series[]; axis: 'generation' | 'spend'; format: (v: number) => string; zero: boolean }) {
  const { t } = useCopy()
  const [box, width] = useWidth<HTMLElement>()
  const [active, setActive] = useState<number | null>(null)
  const help = useId()
  const drawn = series.filter((s) => s.points.length > 0)
  // A chart drawn for one task names it; the name is data, not interface text.
  const head = (
    <h3>
      {title}
      {group !== null && (
        <>
          {' · '}
          <span translate="no">{group}</span>
        </>
      )}
    </h3>
  )
  if (drawn.length === 0) {
    return (
      <section className="cchart" ref={box}>
        {head}
        <p className="cnone muted">{t('cmp.noPoints')}</p>
      </section>
    )
  }
  const narrow = width < NARROW
  const ends = drawn.map((s) => s.points[s.points.length - 1])
  const longest = Math.max(...drawn.map((s, i) => s.label.length + 1 + format(ends[i].y).length))
  const right = narrow ? 16 : Math.min(Math.max(LABEL_MIN, 18 + longest * CHAR), Math.round(width * 0.36))
  const fit = Math.floor((right - 18) / CHAR)
  const m = { t: 14, r: right, b: 36, l: 48 }
  const all = drawn.flatMap((s) => s.points)
  const x = scaleLinear([0, max(all, (p) => p.x) || 1], [m.l, Math.max(m.l + 1, width - m.r)]).nice()
  const lo = Math.min(...all.map((p) => p.y))
  const hi = Math.max(...all.map((p) => p.y))
  const y = scaleLinear(zero ? [0, hi || 1] : lo === hi ? [lo - 0.5, hi + 0.5] : [lo, hi], [HEIGHT - m.b, m.t]).nice()
  const path = line<Point>().curve(curveStepAfter).x((p) => x(p.x)).y((p) => y(p.y))
  // Labels keep 14px apart while they fit the plot's height; more runs than that close the gap.
  const gap = Math.min(14, (HEIGHT - m.b - m.t - 4) / Math.max(1, drawn.length - 1))
  const rows = stackLabels(ends.map((p) => y(p.y) + 4), gap, m.t + 4, HEIGHT - m.b)
  const xTicks = x.ticks(Math.max(2, Math.min(6, Math.floor((width - m.l - m.r) / 60)))).filter((v) => axis === 'spend' || Number.isInteger(v))
  const xText = (v: number) => (axis === 'generation' ? int(v) : money(v))
  // Amounts keep the money format, so on a spend axis of a few cents several ticks read the same (<$0.01): only
  // the first of a run of equal labels is written, the tick marks stay.
  const xLabels = xTicks.map((v, i) => (i > 0 && xText(v) === xText(xTicks[i - 1]) ? null : xText(v)))

  // Every x where some run has a point; the reading is each run's value there.
  const xs = [...new Set(all.map((p) => p.x))].sort((a, b) => a - b)
  const at = active === null ? null : xs[Math.min(active, xs.length - 1)]
  const reading = at === null ? null : drawn.map((s) => valueAt(s.points, at))
  const where = at === null ? '' : axis === 'generation' ? t('cmp.atGeneration', { g: int(at) }) : t('cmp.atSpend', { cost: money(at) })
  const spoken = reading === null ? '' : `${where}: ${drawn.map((s, i) => `${s.label} ${reading[i] === null ? '—' : format(reading[i])}`).join(', ')}`
  const pointAt = (clientX: number, element: Element) => {
    const bounds = element.getBoundingClientRect()
    setActive(bisectCenter(xs, x.invert(clientX - bounds.left)))
  }
  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      setActive(null)
      return
    }
    const end = xs.length - 1
    const from = active ?? end
    const moves: Record<string, number> = { ArrowRight: from + 1, ArrowLeft: from - 1, Home: 0, End: end }
    if (!(event.key in moves)) return
    event.preventDefault()
    setActive(active === null ? end : Math.max(0, Math.min(end, moves[event.key])))
  }
  const summary = `${group === null ? title : `${title} · ${group}`}: ${drawn.map((s, i) => `${s.label} ${format(ends[i].y)}`).join(', ')}`
  const left = at === null ? 0 : x(at)
  return (
    <section className="cchart" ref={box}>
      {head}
      {width > 0 && (
        <div className="cplot" tabIndex={0} role="group" aria-label={summary} aria-describedby={help} onKeyDown={onKey} onBlur={() => setActive(null)}>
          <svg className="chart" width={width} height={HEIGHT} viewBox={`0 0 ${width} ${HEIGHT}`} aria-hidden="true">
            {y.ticks(4).map((v) => (
              <g key={v}>
                <line x1={m.l} x2={width - m.r} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
                <text x={m.l - 8} y={y(v) + 4} textAnchor="end" className="axis">{format(v)}</text>
              </g>
            ))}
            {xTicks.map((v, i) => (
              <g key={v}>
                <line x1={x(v)} x2={x(v)} y1={HEIGHT - m.b + 2} y2={HEIGHT - m.b + 6} stroke="var(--stroke)" />
                {xLabels[i] !== null && <text x={x(v)} y={HEIGHT - m.b + 19} textAnchor="middle" className="axis">{xLabels[i]}</text>}
              </g>
            ))}
            <line x1={m.l} x2={width - m.r} y1={HEIGHT - m.b + 2} y2={HEIGHT - m.b + 2} stroke="var(--stroke)" />
            {at !== null && <line x1={left} x2={left} y1={m.t} y2={HEIGHT - m.b} stroke="var(--stroke)" strokeDasharray="2 3" />}
            {drawn.map((s, i) => (
              <g key={s.label}>
                <path d={path(s.points) ?? ''} fill="none" stroke={s.style.muted ? 'var(--muted)' : 'var(--text)'} strokeWidth={1.6} strokeDasharray={s.style.dash || undefined} />
                <MarkShape mark={s.style.mark} muted={s.style.muted} x={x(ends[i].x)} y={y(ends[i].y)} s={4} />
                {reading !== null && reading[i] !== null && <circle cx={left} cy={y(reading[i])} r={3} fill="var(--bg)" stroke={s.style.muted ? 'var(--muted)' : 'var(--text)'} strokeWidth={1.4} />}
              </g>
            ))}
            {!narrow &&
              drawn.map((s, i) => (
                <text key={s.label} x={width - m.r + 10} y={rows[i]} className={s.style.muted ? 'lbl muted' : 'lbl'}>
                  {shorten(s.label, Math.max(4, fit - format(ends[i].y).length - 1))} {format(ends[i].y)}
                </text>
              ))}
            <text x={width - m.r} y={HEIGHT - 2} textAnchor="end" className="axis-title">{t(axis === 'generation' ? 'cmp.generationAxis' : 'cmp.spendAxis')}</text>
            <rect
              x={m.l}
              y={0}
              width={Math.max(0, width - m.l - m.r)}
              height={HEIGHT - m.b}
              fill="transparent"
              onPointerMove={(event) => pointAt(event.clientX, event.currentTarget.ownerSVGElement!)}
              onPointerLeave={() => setActive(null)}
            />
          </svg>
          {reading !== null && (
            // The reading sits beside the guide, on whichever side has more room; screen readers hear it through the live line.
            <div className="ctip" aria-hidden="true" style={{ left: left + (left > width / 2 ? -12 : 12), transform: left > width / 2 ? 'translateX(-100%)' : undefined }}>
              <span className="faint">{where}</span>
              {drawn.map((s, i) => (
                <span key={s.label} className="ctip-row">
                  <RunMark style={s.style} />
                  <span className="mono">{s.label}</span>
                  <b className="mono">{reading[i] === null ? '—' : format(reading[i])}</b>
                </span>
              ))}
            </div>
          )}
        </div>
      )}
      {narrow && (
        <p className="clabels" aria-hidden="true">
          {drawn.map((s, i) => (
            <span key={s.label}>
              <RunMark style={s.style} /> <span className="mono">{s.label}</span> <b className="mono">{format(ends[i].y)}</b>
            </span>
          ))}
        </p>
      )}
      <p id={help} className="sr-only">{t('cmp.chartHelp')}</p>
      <p className="sr-only" aria-live="polite">{spoken}</p>
    </section>
  )
}

/** `scored` rows hold combined_score, which compares only between runs of the same task. */
type Row = { key: CopyKey; get: (s: RunStats) => number | null; format: (v: number) => string; better: 'max' | 'min' | null; unknownMayLead?: boolean; scored?: boolean }

// The hi-fi's run-level rows plus the legacy legend's maximum generation (P02-Q3). Better is bold; a
// cost that is unknown could be the lower one, so its rows are not judged.
const ROWS: Row[] = [
  { key: 'cmp.best', get: (s) => s.best_score, format: (v) => fmt(v), better: 'max', scored: true },
  { key: 'cmp.failRate', get: (s) => (s.program_count ? (s.program_count - s.correct_count) / s.program_count : null), format: (v) => pct(v, 1), better: 'min' },
  { key: 'cmp.cost', get: runCost, format: money, better: 'min', unknownMayLead: true },
  {
    key: 'cmp.costPerProgram',
    get: (s) => {
      const cost = runCost(s)
      return cost === null || !s.program_count ? null : cost / s.program_count
    },
    format: money,
    better: 'min',
    unknownMayLead: true,
  },
  { key: 'cmp.stall', get: (s) => (s.program_count ? s.gens_since_improvement : null), format: int, better: null },
  { key: 'cmp.programs', get: (s) => s.program_count, format: int, better: null },
  { key: 'cmp.maxGen', get: (s) => (s.program_count ? s.max_generation : null), format: int, better: null },
]

const STRIP = 150

/** Where each run sits on the row's own scale, by its shape. */
function Strip({ values, styles, format, label }: { values: (number | null)[]; styles: Style[]; format: (v: number) => string; label: string }) {
  const known = values.filter((v): v is number => v !== null)
  if (known.length === 0) return <span />
  const x = scaleLinear(stripDomain(known), [10, STRIP - 10]).nice()
  const [a, b] = x.domain()
  return (
    <svg width={STRIP} height={30} viewBox={`0 0 ${STRIP} 30`} role="img" aria-label={label}>
      <line x1={10} x2={STRIP - 10} y1={10} y2={10} stroke="var(--line-strong)" />
      {[a, b].map((v, i) => (
        <g key={i}>
          <line x1={x(v)} x2={x(v)} y1={7} y2={13} stroke="var(--stroke)" />
          <text x={x(v)} y={27} textAnchor={i ? 'end' : 'start'} className="axis">{format(v)}</text>
        </g>
      ))}
      {values.map((v, k) => (v === null ? null : <MarkShape key={k} mark={styles[k].mark} muted={styles[k].muted} x={x(v)} y={10} s={4} />))}
    </svg>
  )
}

/**
 * Two or more runs side by side (award-redesign 5.2, P02): their cumulative curves by generation and by
 * cumulative spend for the metric in the URL, their best valid programs, and their run-level numbers
 * with the better one bold. Scores compare only within a task, so across several tasks the score
 * curves, the leaderboard and the best's lead are drawn per task. The selection and the metric live
 * in the URL.
 */
export function Compare({ data, route, go }: Props) {
  const { t } = useCopy()
  const paths = useMemo(() => [...new Set(route.runs)], [route.runs])
  const metric = metricOf(route.metric)
  const read = useCompareData(paths, data.runs)
  const styles = paths.map((_, k) => seriesStyle(k))
  const entries = paths.map((path) => data.runs?.find((r) => r.path === path) ?? null)
  const keys = entries.map((e, k) => (e ? runGroup(e) : paths[k]))
  const groupKey = JSON.stringify(keys)
  // The groups follow the keys' text, not the array's identity.
  const groups = useMemo(() => byGroup(keys), [groupKey])

  const curves = useMemo(() => {
    if (read.runs === null) return null
    return {
      generation: read.runs.map((run) => byGeneration(run.programs, metric)),
      spend: read.runs.map((run) => bySpend(run, metric)),
      boards: groupBoards(read.runs.map((run) => run.programs), groups),
    }
  }, [read.runs, metric, groups])

  const heading = (
    <header className="chead">
      <h1>{t('crumb.compare', { n: paths.length })}</h1>
    </header>
  )
  if (paths.length < 2) {
    return (
      <>
        {heading}
        <Notice kind="info" label={t('icon.info')} title={t('cmp.none')} subtitle={t('cmp.noneHint')} />
        <p className="cto">
          <a className="btn" href={routeHref({ ...ALL_RUNS, select: paths })}>{t('cmp.toRuns')}</a>
        </p>
      </>
    )
  }
  // Run names come from the listing, so a listing that never arrived is the page's failure.
  if (data.listed && data.runs === null) {
    return (
      <>
        {heading}
        <Notice kind="error" label={t('icon.error')} title={t('runs.listError')} subtitle={t('runs.listErrorHint', { error: data.runsError ?? '' })} />
      </>
    )
  }
  const failed = read.error !== null && (
    <Notice kind="error" label={t('icon.error')} title={t('cmp.readFailed')} subtitle={read.runs === null ? read.error : t('cmp.stale', { error: read.error })} />
  )
  if (read.runs === null || curves === null) {
    return (
      <>
        {heading}
        {failed || <p className="muted">{t('cmp.loading')}</p>}
      </>
    )
  }

  const runs = read.runs
  const mixed = groups.length > 1
  // A run is its experiment or directory within its group. The header names the one group; across
  // several, surfaces that mix groups keep the group in the name so that two h001/e01 stay apart,
  // while a surface drawn per group names the group once in its title.
  const own = paths.map((path, k) => (entries[k] ? runLabel(entries[k]).replace(/^experiments\//, '') : path))
  const labels = own.map((name, k) => (mixed && entries[k] ? `${keys[k]}/${name}` : name))
  const titleOf = (key: string) => {
    const direction = data.workspace?.directions.find((d) => d.id === key)
    return direction ? untagged(direction.direction) : key
  }
  const direction = !mixed && data.workspace?.directions.some((d) => d.id === keys[0])
  const context = mixed ? t('cmp.tasks', { n: groups.length }) : titleOf(keys[0])
  const metricName = t(METRIC_COPY[metric])
  const format = metric === 'best_score' ? (v: number) => fmt(v) : int
  const all = paths.map((_, k) => k)
  // Counts and dollars compare across tasks; scores only within one.
  const chartGroups = metric === 'best_score' ? groups : [all]
  const series = (by: Point[][], group: number[]): Series[] =>
    group.map((k) => ({ label: chartGroups.length > 1 ? own[k] : labels[k], style: styles[k], points: by[k] }))
  // A run without programs has nothing to be better at; its numbers show, unjudged.
  const judged = (list: number[][]) => list.map((group) => group.filter((k) => runs[k].stats.program_count > 0))
  const columns = `120px repeat(${paths.length}, 72px) minmax(${STRIP}px, 1fr)`
  // The hi-fi's three runs fit its 560px metrics column; more runs need the full width.
  const wide = 120 + paths.length * 72 + STRIP + (paths.length + 1) * 14 + 12 > 560
  const boardHead = (
    <div className="cbrow cbhead" role="row">
      <span className="num" role="columnheader">{t('cmp.rank')}</span>
      <span role="columnheader">{t('cmp.run')}</span>
      <span role="columnheader">{t('cmp.program')}</span>
      <span role="columnheader">{t('cmp.id')}</span>
      <span className="num" role="columnheader">{t('cmp.generation')}</span>
      <span className="num" role="columnheader">{t('cmp.best')}</span>
      <span className="num" role="columnheader">{t('cmp.cost')}</span>
    </div>
  )

  return (
    <>
      <header className="chead">
        <h1>{t('crumb.compare', { n: paths.length })}</h1>
        <p className="muted">
          <span translate={mixed ? undefined : 'no'} className={direction || mixed ? undefined : 'mono'}>{context}</span> · {t('cmp.note')}
        </p>
        {mixed && (
          <p className="caveat">
            <Glyph kind="challenged" />
            <span>{t('runs.caveat')}</span>
          </p>
        )}
        <div className="chips">
          {paths.map((path, k) => (
            <a key={path} className="chip" href={routeHref(runRoute(path))} title={t('cmp.openRun', { run: labels[k] })}>
              <RunMark style={styles[k]} size={14} />
              <LineSample style={styles[k]} />
              <b className="mono" translate="no">{labels[k]}</b>
              <span className="faint">{t('cmp.programsN', { n: int(runs[k].stats.program_count) })}</span>
            </a>
          ))}
          <a className="chip is-add" href={routeHref({ ...ALL_RUNS, select: paths })}>+ {t('cmp.addRun')}</a>
        </div>
      </header>
      {failed}
      <div className="ctools">
        <span className="muted" id="cmp-metric">{t('cmp.metric')}</span>
        <div className="seg" role="group" aria-labelledby="cmp-metric">
          {METRICS.map((m) => (
            <button key={m} type="button" aria-pressed={metric === m} onClick={() => go(routeHref({ ...route, metric: m === 'best_score' ? '' : m }), { replace: true })}>
              {t(METRIC_COPY[m])}
            </button>
          ))}
        </div>
      </div>
      {chartGroups.map((groupRuns) => {
        const group = chartGroups.length > 1 ? titleOf(keys[groupRuns[0]]) : null
        // A run with an unpriced charge has an unknown spend from there on, so its spend curve stops early (bySpend).
        const cut = groupRuns.filter((k) => unpricedCharge(runs[k])).map((k) => labels[k])
        return (
          <Fragment key={groupRuns.join(',')}>
            <div className="ccharts">
              <CompareChart title={t('cmp.byGeneration', { metric: metricName })} group={group} series={series(curves.generation, groupRuns)} axis="generation" format={format} zero={metric !== 'best_score'} />
              <CompareChart title={t('cmp.bySpend', { metric: metricName })} group={group} series={series(curves.spend, groupRuns)} axis="spend" format={format} zero={metric !== 'best_score'} />
            </div>
            {cut.length > 0 && <p className="cnote small faint">{t('cmp.spendCut', { runs: cut.join(t('lib.listSeparator')) })}</p>}
          </Fragment>
        )
      })}
      <div className={wide ? 'cbottom is-wide' : 'cbottom'}>
        <section>
          <h3>{t('cmp.board')}</h3>
          {curves.boards.every((board) => board.length === 0) ? (
            <p className="muted">{t('cmp.noBoard')}</p>
          ) : (
            curves.boards.map((board, g) => (
              <div className="cboard-group" key={groups[g].join(',')}>
                {mixed && <h4 className="cgroup-name" translate="no">{titleOf(keys[groups[g][0]])}</h4>}
                {board.length === 0 ? (
                  <p className="muted">{t('cmp.noBoard')}</p>
                ) : (
                  <div className="hscroll">
                    <div className="cboard" role="table" aria-label={mixed ? `${t('cmp.board')} · ${titleOf(keys[groups[g][0]])}` : t('cmp.board')}>
                      {boardHead}
                      {board.map((e, i) => {
                        const name = untagged(String(e.program.metadata?.patch_name ?? '')) || e.program.id.slice(0, 8)
                        return (
                          <div className="cbrow" role="row" key={`${e.run}:${e.program.id}`}>
                            <span className="num mono faint" role="cell">{i + 1}</span>
                            <span className="runmk" role="cell">
                              <RunMark style={styles[e.run]} />
                              {/* Run directories differ at their ends, so a long name keeps both. */}
                              <span className="mono" translate="no" title={own[e.run]}>{shorten(own[e.run], 11)}</span>
                            </span>
                            <span role="cell" className="cbname">
                              <a className="cblink mono" href={routeHref({ ...runRoute(paths[e.run]), program: e.program.id, pane: 'code' })} translate="no">{name}</a>
                            </span>
                            <span className="mono faint" role="cell" translate="no">{e.program.id.slice(0, 8)}</span>
                            <span className="num mono" role="cell">{e.program.generation ?? 0}</span>
                            <span className="num mono" role="cell">{fmt(e.score)}</span>
                            <span className="num mono" role="cell">{e.cost === null ? '—' : money(e.cost)}</span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
        </section>
        <section>
          <h3>{t('cmp.metrics')}</h3>
          <div className="hscroll">
            <div className="cmetrics" role="table" aria-label={t('cmp.metrics')}>
              <div className="crow chd" role="row" style={{ gridTemplateColumns: columns }}>
                <span role="columnheader">{t('cmp.metricCol')}</span>
                {paths.map((path, k) => (
                  <span key={path} className="num" role="columnheader">
                    <RunMark style={styles[k]} label={labels[k]} />
                  </span>
                ))}
                <span role="columnheader">{t('cmp.scale')}</span>
              </div>
              {ROWS.map((row) => {
                const values = runs.map((run) => row.get(run.stats))
                const lead = row.better ? groupLeads(values, judged(row.scored ? groups : [all]), row.better, row.format, row.unknownMayLead) : null
                return (
                  <div className="crow" role="row" key={row.key} style={{ gridTemplateColumns: columns }}>
                    <span className="muted" role="rowheader">{t(row.key)}</span>
                    {values.map((v, k) => (
                      <span key={k} role="cell" className={`num mono${lead === null || lead[k] === null ? '' : lead[k] ? ' is-lead' : ' is-trail'}`}>
                        {v === null ? '—' : row.format(v)}
                      </span>
                    ))}
                    <span role="cell">
                      {/* One scale for every run's best would compare scores across tasks. */}
                      {row.scored && mixed ? <span /> : <Strip values={values} styles={styles} format={row.format} label={t(row.key)} />}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
          <p className="faint small">{t('cmp.leadNote')}</p>
        </section>
      </div>
    </>
  )
}
