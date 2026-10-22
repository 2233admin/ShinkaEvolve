import { curveStepAfter, line, scaleLinear } from 'd3'
import { Component, useId, useMemo, useState, type KeyboardEvent, type ReactNode } from 'react'
import { bySpend, DASHES, programCost, scoreOf, stackLabels } from '../compare/series'
import { useWidth } from '../design/useWidth'
import { fmt, int, money, pct } from '../format'
import type { CopyKey } from '../i18n/copy'
import { useCopy } from '../i18n/lang'
import { Notice } from '../Notice'
import { runCost, runtimeText, type RunStats } from '../runs/data'
import { generationAvgCost, modelLabels as costLabels, modelPatchCost, type ComboCost } from './costs'
import { generationOf, nameOf, type Prompt, type RunProgram } from './data'
import { Offspring } from './LineagePanel'
import { byComputeTime, byWallTime, metricKeys, metricSeries, promptCost, type GenerationRow } from './progressData'
import { llmStats, modelLabels, mutationStats, noveltyLlmStats, noveltyMutationStats, resampleLlmStats, resampleMutationStats, type ModelStats, type PatchTypeStat, type Tally } from './success'
import {
  completionsPerMinute,
  evalOccupancyDistribution,
  occupancyCurves,
  occupancyPercentCurves,
  poolRuntime,
  runtimeEmptyState,
  runtimeRows,
  runtimeTimeline,
  stageDurations,
  stageUtilization,
  throughputSummary,
  STAGES,
  type Stage,
} from './throughput'
import { codeBloat, correctnessRates, improvementRates, type RateSeries } from './trends'

// ---- A chart that fills its box: pixel width, so labels keep their size at every page width ----

const PATTERNS = [...DASHES, '8 3 1.5 3']
const STAGE_FILL: Record<Stage, string> = { sampling: 'var(--muted)', evaluation: 'var(--text)', postprocess: 'var(--stroke)' }
const STAGE_KEY: Record<Stage, CopyKey> = { sampling: 'rp.stage.sampling', evaluation: 'rp.stage.evaluation', postprocess: 'rp.stage.postprocess' }

function Sized({ children }: { children: (width: number) => ReactNode }) {
  const [box, width] = useWidth<HTMLDivElement>()
  return (
    <div ref={box} className="chart-box">
      {width > 0 && children(width)}
    </div>
  )
}

type Series = { key: string; label?: string; points: [number, number][]; stroke?: string; width?: number; dash?: string; step?: boolean }
type Dot = { id: string; x: number; y: number; hollow?: boolean }
type Bar = { x: number; y: number; w?: number; fill?: string }

type XYProps = {
  label: string
  height?: number
  x: [number, number]
  y: [number, number]
  series?: Series[]
  dots?: Dot[]
  bars?: Bar[]
  xFormat: (v: number) => string
  yFormat: (v: number) => string
  xTitle: string
  selected?: string | null
  onPick?: (id: string) => void
  /** What the chart says at an x the pointer or the arrow keys reach: one line, or several. */
  read: (x: number) => string | string[] | null
  /** The x values the arrow keys step through. */
  stops: number[]
  ends?: boolean
  /** Use the y range as given (an entered score-axis limit) instead of rounding it out to ticks. */
  exactY?: boolean
}

/**
 * The disclosures' line, step, dot and bar charts. Pointer and arrow keys read the value at an x; a
 * dot under the pointer, or the one the keys reach, opens its program (design.md › 联动与选择).
 */
function XYChart({ label, height = 168, x: xd, y: yd, series = [], dots = [], bars = [], xFormat, yFormat, xTitle, selected, onPick, read, stops, ends, exactY }: XYProps) {
  const { t } = useCopy()
  const [at, setAt] = useState<number | null>(null)
  // Marks outside an entered score range are cut at the plot's edge, not drawn over the axes.
  const clip = `xy-clip-${useId().replace(/:/g, '')}`
  return (
    <Sized>
      {(width) => {
        const m = { l: 56, r: ends ? 160 : 18, t: 10, b: 30 }
        const x = scaleLinear(xd[1] > xd[0] ? xd : [xd[0], xd[0] + 1], [m.l, Math.max(m.l + 1, width - m.r)])
        const yBase = scaleLinear(yd[1] > yd[0] ? yd : [yd[0] - 1, yd[0] + 1], [height - m.b, m.t])
        const y = exactY ? yBase : yBase.nice()
        const near = (px: number, py: number) => {
          let id: string | null = null
          let d = 7
          for (const p of dots) {
            const dist = Math.hypot(x(p.x) - px, y(p.y) - py)
            if (dist < d) {
              d = dist
              id = p.id
            }
          }
          return id
        }
        const stopAt = (px: number) => stops.reduce<number | null>((best, s) => (best === null || Math.abs(x(s) - px) < Math.abs(x(best) - px) ? s : best), null)
        const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
          if (stops.length === 0) return
          const i = at === null ? -1 : stops.indexOf(at)
          const moves: Record<string, number> = { ArrowRight: i + 1, ArrowLeft: Math.max(0, i - 1), Home: 0, End: stops.length - 1 }
          if (event.key === 'Escape') setAt(null)
          else if (event.key === 'Enter' && at !== null && onPick) {
            const dot = dots.find((d) => d.x === at)
            if (dot) onPick(dot.id)
          } else if (event.key in moves) setAt(stops[Math.max(0, Math.min(stops.length - 1, moves[event.key]))])
          else return
          event.preventDefault()
        }
        const reading = at === null ? null : read(at)
        const lines = reading === null ? [] : typeof reading === 'string' ? [reading] : reading
        // Only labelled lines take a place at the right edge.
        const labelled = ends ? series.filter((s) => s.label && s.points.length) : []
        const endYs = stackLabels(labelled.map((s) => y(s.points[s.points.length - 1][1])), 12, m.t, height - m.b)
        // The tip sits on the side of the reading with more room and wraps within it rather than leaving the chart.
        const tipX = at === null ? 0 : x(at)
        const flip = tipX > width / 2
        return (
          <div className="xy" tabIndex={stops.length ? 0 : -1} role="group" aria-label={label} onKeyDown={onKey} onBlur={() => setAt(null)}>
            <svg
              className="chart"
              width={width}
              height={height}
              viewBox={`0 0 ${width} ${height}`}
              aria-hidden="true"
              onPointerMove={(event) => {
                const r = event.currentTarget.getBoundingClientRect()
                const px = event.clientX - r.left
                setAt(px < m.l - 4 || px > width - m.r + 4 ? null : stopAt(px))
              }}
              onPointerLeave={() => setAt(null)}
              onClick={(event) => {
                if (!onPick) return
                const r = event.currentTarget.getBoundingClientRect()
                const id = near(event.clientX - r.left, event.clientY - r.top)
                if (id) onPick(id)
              }}
            >
              {y.ticks(3).map((v) => (
                <g key={v}>
                  <line x1={m.l} x2={width - m.r} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
                  <text x={m.l - 8} y={y(v) + 3.5} textAnchor="end" className="axis">{yFormat(v)}</text>
                </g>
              ))}
              <defs>
                <clipPath id={clip}>
                  <rect x={m.l - 6} y={m.t - 6} width={Math.max(0, width - m.l - m.r + 12)} height={Math.max(0, height - m.b - m.t + 12)} />
                </clipPath>
              </defs>
              <g clipPath={`url(#${clip})`}>
                {bars.map((b, k) => {
                  const w = b.w ?? Math.max(2, Math.min(14, (x(xd[0] + 1) - x(xd[0])) * 0.62))
                  return <rect key={k} x={x(b.x) - w / 2} y={y(Math.max(0, b.y))} width={w} height={Math.max(0, y(0) - y(Math.max(0, b.y)))} fill={b.fill ?? 'var(--stroke)'} />
                })}
                {dots.map((d) => (
                  <circle key={d.id} cx={x(d.x)} cy={y(d.y)} r={d.id === selected ? 5 : 1.8} fill={d.hollow || d.id === selected ? 'none' : 'var(--stroke)'} stroke={d.id === selected ? 'var(--text)' : d.hollow ? 'var(--stroke)' : 'none'} strokeWidth={d.id === selected ? 1.5 : 0.8} />
                ))}
                {series.map((s) => (
                  <path
                    key={s.key}
                    d={(s.step ? line<[number, number]>().curve(curveStepAfter) : line<[number, number]>()).x(([a]) => x(a)).y(([, b]) => y(b))(s.points) ?? ''}
                    fill="none"
                    stroke={s.stroke ?? 'var(--text)'}
                    strokeWidth={s.width ?? 1.4}
                    strokeDasharray={s.dash}
                  />
                ))}
              </g>
              {labelled.map((s, k) => (
                <text key={s.key} x={width - m.r + 8} y={endYs[k] + 3.5} className="axis">
                  {s.label} {yFormat(s.points[s.points.length - 1][1])}
                </text>
              ))}
              {at !== null && <line x1={x(at)} x2={x(at)} y1={m.t} y2={height - m.b} stroke="var(--stroke)" />}
              <line x1={m.l} x2={width - m.r} y1={height - m.b + 2} y2={height - m.b + 2} stroke="var(--stroke)" />
              {x.ticks(Math.max(2, Math.floor((width - m.l - m.r) / 90))).map((v) => (
                <text key={v} x={x(v)} y={height - m.b + 16} textAnchor="middle" className="axis">{xFormat(v)}</text>
              ))}
              <text x={width - m.r} y={height - 2} textAnchor="end" className="axis-title">{xTitle}</text>
            </svg>
            {lines.length > 0 && (
              <div className="lanes-tip ptip" style={{ left: tipX + (flip ? -12 : 12), top: 6, transform: flip ? 'translateX(-100%)' : undefined, boxSizing: 'border-box', maxWidth: Math.max(120, flip ? tipX - 12 : width - tipX - 12) }}>
                {lines.map((line, k) => (
                  <span key={k} className="mono">{line}</span>
                ))}
                {onPick && dots.some((d) => d.x === at) && <span className="faint">{t('ra.pointOpen')}</span>}
              </div>
            )}
            <p className="sr-only" aria-live="polite">{lines.join(' · ')}</p>
          </div>
        )
      }}
    </Sized>
  )
}

function Card({ title, note, children, wide }: { title: string; note?: ReactNode; children: ReactNode; wide?: boolean }) {
  return (
    <section className={wide ? 'pcard is-wide' : 'pcard'} aria-label={title}>
      <h4>{title}</h4>
      {children}
      {note && <p className="small faint">{note}</p>}
    </section>
  )
}

const lastOf = <T,>(xs: T[]): T | undefined => xs[xs.length - 1]

// ---- L04: one chart per public metric ----

/** L04: each numeric public metric over the generations: valid programs, the generation's highest value, and its running minimum or maximum. */
export function MetricsSection({ programs, upTo, selected, onSelect }: { programs: RunProgram[]; upTo: number; selected: RunProgram | null; onSelect: (id: string) => void }) {
  const { t } = useCopy()
  const keys = useMemo(() => metricKeys(programs), [programs])
  if (keys.length === 0) return <p className="small muted">{t('rp.noMetrics')}</p>
  return (
    <>
      <p className="small faint">{t('rp.metricsNote')}</p>
      <div className="pcards">
        {keys.map((key) => (
          <Metric key={key} metric={key} programs={programs} upTo={upTo} selected={selected} onSelect={onSelect} />
        ))}
      </div>
    </>
  )
}

function Metric({ metric, programs, upTo, selected, onSelect }: { metric: string; programs: RunProgram[]; upTo: number; selected: RunProgram | null; onSelect: (id: string) => void }) {
  const { t } = useCopy()
  // The restored Metrics view opened on the running minimum (viz_tree.html:13279); each chart switches on its own (L04-C1).
  const [mode, setMode] = useState<'min' | 'max'>('min')
  const { points, stats } = useMemo(() => metricSeries(programs, metric), [programs, metric])
  const shownPoints = points.filter((p) => p.generation <= upTo)
  const shownStats = stats.filter((s) => s.generation <= upTo)
  const values = points.map((p) => p.value)
  const maxGen = Math.max(1, ...points.map((p) => p.generation))
  const fmtV = (v: number) => (Math.abs(v) >= 100 ? int(v) : fmt(v, Math.abs(v) < 1 ? 3 : 2))
  return (
    <section className="pcard" aria-label={metric}>
      <div className="pcard-head">
        <h4 className="mono" translate="no">{metric}</h4>
        <span className="seg is-small" role="group" aria-label={t('rp.metricMode')}>
          <button type="button" aria-pressed={mode === 'min'} onClick={() => setMode('min')}>{t('rp.cumMin')}</button>
          <button type="button" aria-pressed={mode === 'max'} onClick={() => setMode('max')}>{t('rp.cumMax')}</button>
        </span>
      </div>
      <span className="legend">
        <span><i className="dot is-hollow" />{t('rp.validPrograms')}</span>
        <span><i className="ln" />{t('rp.genMax')}</span>
        <span><i className="ln is-dash" />{t(mode === 'min' ? 'rp.cumMin' : 'rp.cumMax')}</span>
      </span>
      <XYChart
        label={t('rp.metricLabel', { metric, n: int(shownPoints.length) })}
        x={[0, maxGen]}
        y={[Math.min(...values), Math.max(...values)]}
        dots={shownPoints.map((p) => ({ id: p.id, x: p.generation, y: p.value, hollow: true }))}
        series={[
          { key: 'max', points: shownStats.map((s) => [s.generation, s.max]), width: 1.6 },
          { key: 'cum', points: shownStats.map((s) => [s.generation, mode === 'min' ? s.cumulativeMin : s.cumulativeMax]), stroke: 'var(--muted)', dash: '4 3', step: true },
        ]}
        xFormat={(v) => int(v)}
        yFormat={fmtV}
        xTitle={t('ra.genAxis')}
        selected={selected?.id ?? null}
        onPick={onSelect}
        stops={shownStats.map((s) => s.generation)}
        read={(g) => {
          const s = shownStats.find((x) => x.generation === g)
          return s ? `${t('ra.gen', { g })} · ${t('rp.genMax')} ${fmtV(s.max)} · ${t(mode === 'min' ? 'rp.cumMin' : 'rp.cumMax')} ${fmtV(mode === 'min' ? s.cumulativeMin : s.cumulativeMax)}` : null
        }}
      />
    </section>
  )
}

// ---- L10-Q2/Q4, R01-Q5: score against spend ----

/** L10-Q2/Q4 and R01-Q5: the best valid score against cumulative spend, or against compute time when the run recorded no spend. */
export function SpendSection({ programs, prompts, stats, zeroCost, scoreRange }: { programs: RunProgram[]; prompts: Prompt[]; stats: RunStats; zeroCost: boolean; scoreRange: [number, number] }) {
  const { t } = useCopy()
  const spend = useMemo(() => bySpend({ programs, prompts, stats }, 'best_score'), [programs, prompts, stats])
  const time = useMemo(() => byComputeTime(programs), [programs])
  const total = runCost(stats)
  // An empty curve has one of two causes: no valid program, or a total already unknown at the first valid one.
  const anyValid = programs.some((p) => scoreOf(p) !== null)
  if (zeroCost) {
    const pts = time.points
    if (pts.length === 0) return <p className="small muted">{t(anyValid ? 'rp.computeUnknown' : 'rp.noValidScore')}</p>
    return (
      <Card title={t('rp.computeTitleFull')} note={time.complete ? t('rp.zeroCostNote') : t('rp.computeStopped', { v: runtimeText(lastOf(pts)?.x ?? 0) })} wide>
        <XYChart
          label={t('rp.computeTitleFull')}
          x={[0, lastOf(pts)?.x ?? 1]}
          y={scoreRange}
          exactY
          series={[{ key: 'best', points: pts.map((p) => [p.x, p.best]), step: true, width: 1.8 }]}
          xFormat={(v) => runtimeText(v)}
          yFormat={(v) => fmt(v)}
          xTitle={t('rp.computeAxis')}
          stops={pts.map((p) => p.x)}
          read={(v) => {
            const p = pts.find((q) => q.x === v)
            return p ? t('rp.computeRead', { x: runtimeText(p.x), y: fmt(p.best) }) : null
          }}
        />
      </Card>
    )
  }
  // The curve ends before the first charge without a price; with every charge priced but the run's
  // total unknown, it ends at the spend that was recorded.
  const unpriced = programs.some((p) => programCost(p) === null) || prompts.some((q) => promptCost(q) === null)
  if (spend.length === 0) return <p className="small muted">{t(anyValid ? 'rp.spendUnknown' : 'rp.noValidScore')}</p>
  const end = lastOf(spend)?.x ?? 0
  const note = unpriced ? t('rp.spendStopped', { v: money(end) }) : total === null ? t('rp.spendTotalUnknown', { v: money(end) }) : undefined
  return (
    <Card title={t('rp.spendTitle')} note={note} wide>
      <XYChart
        label={t('rp.spendTitle')}
        x={[0, end || 1]}
        y={scoreRange}
        exactY
        series={[{ key: 'best', points: spend.filter((p) => p.y !== null).map((p) => [p.x, p.y as number]), step: true, width: 1.8 }]}
        xFormat={(v) => money(v)}
        yFormat={(v) => fmt(v)}
        xTitle={t('rp.spendAxis')}
        stops={spend.map((p) => p.x)}
        read={(v) => {
          const p = [...spend].reverse().find((q) => q.x <= v)
          return p ? t('rp.spendRead', { x: money(p.x), y: fmt(p.y) }) : null
        }}
      />
    </Card>
  )
}

// ---- R10 and R01-Q6/Q8: runtime and throughput ----

/** R10 (all eight) with R01-Q6/Q8: Throughput is the runtime home (D-4). */
export function RuntimeSection({ programs, rows, selected, onSelect, scoreRange }: { programs: RunProgram[]; rows: GenerationRow[]; selected: RunProgram | null; onSelect: (id: string) => void; scoreRange: [number, number] }) {
  const { t } = useCopy()
  const base = useMemo(() => runtimeRows(programs), [programs])
  const pool = useMemo(() => poolRuntime(base.rows), [base])
  const timeline = useMemo(() => runtimeTimeline(base.rows, pool), [base, pool])
  const empty = runtimeEmptyState(programs.length, base.rows, pool)
  const wall = useMemo(() => byWallTime(programs), [programs])
  const [percent, setPercent] = useState(false)

  const perGen = rows.map((r) => r.computeTime)
  // The running total stops at the first generation whose time was not recorded.
  const cumulative: (number | null)[] = []
  for (const v of perGen) {
    const before = cumulative.length ? cumulative[cumulative.length - 1] : 0
    cumulative.push(before === null || v === null ? null : before + v)
  }

  const sections: ReactNode[] = []
  // R01-Q6: the best score against wall-clock time since the first program.
  if (wall.points.length) {
    const end = lastOf(wall.points)?.x ?? 1
    sections.push(
      <Card key="wall" title={t('rp.wallTitle')} wide>
        <XYChart
          label={t('rp.wallTitle')}
          x={[0, Math.max(end, ...wall.scores.map((s) => s.x))]}
          y={scoreRange}
          exactY
          dots={wall.scores.map((s) => ({ id: s.id, x: s.x, y: s.score, hollow: true }))}
          series={[{ key: 'best', points: wall.points.map((p) => [p.x, p.best]), step: true, width: 1.8 }]}
          xFormat={(v) => runtimeText(v)}
          yFormat={(v) => fmt(v)}
          xTitle={t('rp.wallAxis')}
          selected={selected?.id ?? null}
          onPick={onSelect}
          stops={wall.scores.map((s) => s.x)}
          read={(v) => {
            const s = wall.scores.find((q) => q.x === v)
            const best = [...wall.points].reverse().find((q) => q.x <= v)
            return s && best ? `${runtimeText(v)} · ${nameOf(programs.find((p) => p.id === s.id) as RunProgram)} ${fmt(s.score)} · ${t('rp.cumBest')} ${fmt(best.best)}` : null
          }}
        />
      </Card>,
    )
  }
  // R01-Q8: evaluation time per generation and its running total.
  if (perGen.some((v) => v !== null)) {
    const cumPairs = cumulative.flatMap((v, g): [number, number][] => (v === null ? [] : [[g, v]]))
    const most = Math.max(1, ...perGen.filter((v): v is number => v !== null))
    const cumMost = Math.max(1, ...cumPairs.map(([, v]) => v))
    sections.push(
      <Card key="gen" title={t('rp.genRuntimeTitle')} note={t('rp.genRuntimeNote', { total: runtimeText(lastOf(cumPairs)?.[1] ?? null) })}>
        <XYChart
          label={t('rp.genRuntimeTitle')}
          x={[0, Math.max(1, rows.length - 1)]}
          y={[0, most]}
          bars={perGen.flatMap((v, g) => (v === null ? [] : [{ x: g, y: v }]))}
          series={[{ key: 'cum', points: cumPairs.map(([g, v]) => [g, (v / cumMost) * most]), stroke: 'var(--muted)', dash: '1.5 2' }]}
          xFormat={(v) => int(v)}
          yFormat={(v) => runtimeText(v)}
          xTitle={t('ra.genAxis')}
          stops={rows.map((r) => r.generation)}
          read={(g) => (perGen[g] === null ? null : `${t('ra.gen', { g })} · ${runtimeText(perGen[g])} · ${t('rp.cumulative')} ${runtimeText(cumulative[g])}`)}
        />
      </Card>,
    )
  }
  if (empty === 'noPrograms' || empty === 'noRuntimeRows') {
    sections.push(<p key="none" className="small muted">{t('rp.noRuntime')}</p>)
    return <div className="pcards">{sections}</div>
  }

  // R10-Q1: the summary figures.
  const summary = pool ? throughputSummary(pool) : null
  if (summary) {
    sections.push(
      <Card key="sum" title={t('rp.sumTitle')} wide>
        <dl className="psum">
          <div><dt>{t('rp.sumWall')}</dt><dd className="mono">{runtimeText(summary.wallClockSeconds)}</dd></div>
          <div><dt>{t('rp.sumJobs')}</dt><dd className="mono">{int(summary.completedJobs)}</dd></div>
          <div><dt>{t('rp.sumRate')}</dt><dd className="mono">{fmt(summary.completionsPerMinute)}</dd></div>
          <div><dt>{t('rp.sumUtil')}</dt><dd className="mono">{summary.evalUtilizationPct === null ? '—' : `${fmt(summary.evalUtilizationPct, 1)}%`}</dd></div>
          <div><dt>{t('rp.sumFull')}</dt><dd className="mono">{summary.evalFullOccupancyPct === null ? '—' : `${fmt(summary.evalFullOccupancyPct, 1)}%`}</dd></div>
          <div><dt>{t('rp.sumIdle')}</dt><dd className="mono">{summary.evalIdlePct === null ? '—' : `${fmt(summary.evalIdlePct, 1)}%`}</dd></div>
        </dl>
      </Card>,
    )
  }
  // R10-Q2: one lane per worker slot (or the lanes the timestamps imply), the selected program outlined.
  if (timeline.mode !== 'empty') {
    sections.push(
      <Card key="tl" title={t('rp.timelineTitle')} note={empty === 'noPoolRows' ? t('rp.noPool') : timeline.mode === 'pool' ? STAGES.map((s) => t('rp.timelinePeak', { stage: t(STAGE_KEY[s]), peak: int(timeline.peaks[s]), cap: int(timeline.capacities[s]) })).join(' · ') : undefined} wide>
        <Timeline timeline={timeline} selected={selected?.id ?? null} onSelect={onSelect} />
      </Card>,
    )
  }
  if (pool) {
    const abs = occupancyCurves(pool)
    const rel = occupancyPercentCurves(pool)
    const curves = percent ? (rel?.curves ?? []) : abs.map((c) => ({ stage: c.stage, points: c.series.points }))
    const range: [number, number] = percent ? (rel?.range ?? [0, 1]) : [Math.min(...abs.map((c) => c.series.start)), Math.max(...abs.map((c) => c.series.end))]
    const top = percent ? 100 : Math.max(1, ...abs.map((c) => c.series.capacity))
    if (curves.length) {
      sections.push(
        <section key="occ" className="pcard is-wide" aria-label={t('rp.occTitle')}>
          <div className="pcard-head">
            <h4>{t('rp.occTitle')}</h4>
            <span className="seg is-small" role="group" aria-label={t('rp.occTitle')}>
              <button type="button" aria-pressed={!percent} onClick={() => setPercent(false)}>{t('rp.occAbs')}</button>
              <button type="button" aria-pressed={percent} onClick={() => setPercent(true)}>{t('rp.occPct')}</button>
            </span>
          </div>
          <span className="legend">
            {curves.map((c, k) => (
              <span key={c.stage}><i className="ln" style={{ borderTopStyle: k ? 'dashed' : 'solid', borderTopColor: STAGE_FILL[c.stage] }} />{t(STAGE_KEY[c.stage])}</span>
            ))}
          </span>
          <XYChart
            label={t('rp.occTitle')}
            x={range}
            y={[0, top]}
            series={curves.map((c, k) => ({ key: c.stage, points: c.points.map((p) => [p.t, p.value] as [number, number]), stroke: STAGE_FILL[c.stage], dash: PATTERNS[k % PATTERNS.length], step: true, width: 1.3 }))}
            xFormat={(v) => runtimeText(v)}
            yFormat={(v) => (percent ? `${int(v)}%` : int(v))}
            xTitle={t('rp.wallAxis')}
            stops={[...new Set(curves.flatMap((c) => c.points.map((p) => p.t)))].sort((a, b) => a - b)}
            read={(v) =>
              `${runtimeText(v)} · ${curves
                .map((c) => {
                  const p = [...c.points].reverse().find((q) => q.t <= v)
                  return `${t(STAGE_KEY[c.stage])} ${p ? (percent ? `${fmt(p.value, 0)}%` : int(p.value)) : '—'}`
                })
                .join(' · ')}`
            }
          />
        </section>,
      )
    }
    // R10-Q5: how long the evaluation pool spent at each number of busy workers.
    const dist = evalOccupancyDistribution(pool)
    if (dist && dist.length) {
      sections.push(
        <Card key="dist" title={t('rp.distTitle')}>
          <ol className="bars hbars">
            {dist.map((d) => (
              <li key={d.count}>
                <span className="mono faint">{t('rp.busyN', { n: d.count })}</span>
                <span className="bar" aria-hidden="true"><b style={{ width: `${d.sharePct}%`, background: d.full ? 'var(--text)' : 'var(--stroke)' }} /></span>
                <span className="mono small">{fmt(d.sharePct, 1)}%</span>
              </li>
            ))}
          </ol>
        </Card>,
      )
    }
  }
  // R10-Q6: evaluations completed per minute.
  const perMinute = completionsPerMinute(pool?.rows ?? base.rows)
  if (perMinute.length) {
    sections.push(
      <Card key="cpm" title={t('rp.rateTitle')}>
        <XYChart
          label={t('rp.rateTitle')}
          x={[perMinute[0].t, Math.max(perMinute[0].t + 1, lastOf(perMinute)?.t ?? 1)]}
          y={[0, Math.max(1, ...perMinute.map((p) => p.value))]}
          bars={perMinute.map((p) => ({ x: p.t, y: p.value, w: 4 }))}
          xFormat={(v) => t('rp.minuteN', { n: int(v) })}
          yFormat={(v) => int(v)}
          xTitle={t('rp.minuteAxis')}
          stops={perMinute.map((p) => p.t)}
          read={(v) => {
            const p = perMinute.find((q) => q.t === v)
            return p ? `${t('rp.minuteN', { n: int(p.t) })} · ${int(p.value)}` : null
          }}
        />
      </Card>,
    )
  }
  // R10-Q7/Q8: stage durations and resource utilisation.
  const durations = stageDurations(pool?.rows ?? base.rows)
  if (durations.length) {
    sections.push(
      <Card key="dur" title={t('rp.durTitle')}>
        <table className="itable">
          <thead>
            <tr><th>{t('rp.stage')}</th><th className="num">p50</th><th className="num">p95</th><th className="num">{t('rp.maxCol')}</th></tr>
          </thead>
          <tbody>
            {durations.map((d) => (
              <tr key={d.stage}><td>{t(STAGE_KEY[d.stage])}</td><td className="num mono">{seconds(d.p50)}</td><td className="num mono">{seconds(d.p95)}</td><td className="num mono">{seconds(d.max)}</td></tr>
            ))}
          </tbody>
        </table>
      </Card>,
    )
  }
  if (pool) {
    sections.push(
      <Card key="util" title={t('rp.utilTitle')}>
        <table className="itable">
          <thead>
            <tr><th>{t('rp.stage')}</th><th className="num">{t('rp.capacity')}</th><th className="num">{t('rp.avgBusy')}</th><th className="num">{t('rp.utilCol')}</th><th className="num">{t('rp.fullCol')}</th><th className="num">{t('rp.idleCol')}</th></tr>
          </thead>
          <tbody>
            {stageUtilization(pool).map((u) => (
              <tr key={u.stage}>
                <td>{t(STAGE_KEY[u.stage])}</td>
                <td className="num mono">{int(u.capacity)}</td>
                <td className="num mono">{u.avgOccupied === null ? '—' : fmt(u.avgOccupied)}</td>
                <td className="num mono">{u.utilizationPct === null ? '—' : `${fmt(u.utilizationPct, 1)}%`}</td>
                <td className="num mono">{u.fullOccupancyPct === null ? '—' : `${fmt(u.fullOccupancyPct, 1)}%`}</td>
                <td className="num mono">{u.idlePct === null ? '—' : `${fmt(u.idlePct, 1)}%`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>,
    )
  }
  return <div className="pcards">{sections}</div>
}

const seconds = (v: number) => (v < 60 ? `${fmt(v, 1)} s` : runtimeText(v))

/** R10-Q2: every job's sampling, evaluation and postprocess bars in its worker's lane; pointer or arrow keys read a job, click or Enter opens it. */
function Timeline({ timeline, selected, onSelect }: { timeline: Exclude<ReturnType<typeof runtimeTimeline>, { mode: 'empty' }>; selected: string | null; onSelect: (id: string) => void }) {
  const { t } = useCopy()
  const [active, setActive] = useState<string | null>(null)
  const lanes = timeline.mode === 'pool' ? timeline.lanes.length : timeline.laneCount
  const laneH = lanes > 40 ? 6 : lanes > 20 ? 8 : 12
  const height = 24 + lanes * laneH + 24
  const laneLabel = (i: number) => (timeline.mode === 'pool' ? `${t(STAGE_KEY[timeline.lanes[i].stage])} ${timeline.lanes[i].worker}` : `${i + 1}`)
  const current = timeline.rows.find((r) => r.id === active) ?? null
  // The arrow keys walk the jobs in the order they started.
  const order = useMemo(() => {
    const start = (r: (typeof timeline.rows)[number]) => Math.min(...STAGES.map((s) => r.stages[s]?.[0] ?? Infinity))
    return [...timeline.rows].sort((a, b) => start(a) - start(b)).map((r) => r.id)
  }, [timeline])
  const describe = (r: (typeof timeline.rows)[number]) => `${r.name} · ${t('ra.gen', { g: r.generation })} · ${STAGES.map((s) => `${t(STAGE_KEY[s])} ${seconds(r.seconds[s])}`).join(' · ')}`
  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    const i = active === null ? -1 : order.indexOf(active)
    const moves: Record<string, number> = { ArrowRight: i + 1, ArrowLeft: Math.max(0, i - 1), Home: 0, End: order.length - 1 }
    if (event.key === 'Escape') setActive(null)
    else if (event.key === 'Enter' && active) onSelect(active)
    else if (event.key in moves && order.length) setActive(order[Math.max(0, Math.min(order.length - 1, moves[event.key]))])
    else return
    event.preventDefault()
  }
  return (
    <Sized>
      {(width) => {
        const m = { l: 96, r: 18 }
        const x = scaleLinear(timeline.range[1] > timeline.range[0] ? timeline.range : [timeline.range[0], timeline.range[0] + 1], [m.l, Math.max(m.l + 1, width - m.r)])
        const bars = timeline.rows.flatMap((r) =>
          STAGES.flatMap((stage) => {
            const lane = r.lanes[stage]
            const span = r.stages[stage]
            if (lane === null || span === null) return []
            const w = Math.max(1, x(span[0] + Math.max(r.seconds[stage], timeline.minBarSeconds)) - x(span[0]))
            return [{ id: r.id, stage, x: x(span[0]), w, y: 12 + lane * laneH }]
          }),
        )
        return (
          <div className="xy" tabIndex={0} role="group" aria-label={t('rp.timelineLabel', { n: int(timeline.rows.length), lanes: int(lanes) })} aria-describedby="timeline-help" onKeyDown={onKey} onBlur={() => setActive(null)}>
            <svg
              className="chart"
              width={width}
              height={height}
              viewBox={`0 0 ${width} ${height}`}
              aria-hidden="true"
              onPointerMove={(event) => {
                const r = event.currentTarget.getBoundingClientRect()
                const px = event.clientX - r.left
                const py = event.clientY - r.top
                const hit = bars.find((b) => px >= b.x - 1 && px <= b.x + b.w + 1 && py >= b.y && py <= b.y + laneH - 1)
                setActive(hit?.id ?? null)
              }}
              onPointerLeave={() => setActive(null)}
              onClick={() => active && onSelect(active)}
            >
              {Array.from({ length: lanes }, (_, i) =>
                lanes <= 24 || i % Math.ceil(lanes / 24) === 0 ? (
                  <text key={i} x={m.l - 8} y={12 + i * laneH + laneH - 2} textAnchor="end" className="axis">{laneLabel(i)}</text>
                ) : null,
              )}
              {bars.map((b, k) => (
                <rect key={k} x={b.x} y={b.y} width={b.w} height={laneH - 2} fill={STAGE_FILL[b.stage]} opacity={active && active !== b.id ? 0.55 : 1} />
              ))}
              {bars.filter((b) => b.id === selected).map((b, k) => (
                <rect key={`s${k}`} x={b.x - 1.5} y={b.y - 1.5} width={b.w + 3} height={laneH + 1} fill="none" stroke="var(--signal-graphic)" strokeWidth={1.5} />
              ))}
              <line x1={m.l} x2={width - m.r} y1={height - 22} y2={height - 22} stroke="var(--stroke)" />
              {x.ticks(Math.max(2, Math.floor((width - m.l - m.r) / 90))).map((v) => (
                <text key={v} x={x(v)} y={height - 8} textAnchor="middle" className="axis">{runtimeText(v)}</text>
              ))}
            </svg>
            <span className="legend">
              {STAGES.map((s) => (
                <span key={s}><i className="sw" style={{ background: STAGE_FILL[s] }} />{t(STAGE_KEY[s])}</span>
              ))}
              <span><i className="sw is-ring" />{t('rp.selectedJob')}</span>
            </span>
            {current && <p className="small mono" translate="no">{describe(current)}</p>}
            <p id="timeline-help" className="sr-only">{t('rp.timelineHelp')}</p>
            <p className="sr-only" aria-live="polite">{current ? describe(current) : ''}</p>
          </div>
        )
      }}
    </Sized>
  )
}

// ---- R09: the eleven analysis plots, in pairs (OD-6) ----

function RateChart({ title, series }: { title: string; series: RateSeries }) {
  const { t } = useCopy()
  const all = [series.overall, ...series.byType.map((s) => s.points)].flat()
  if (series.overall.length === 0) return <Card title={title}><p className="small muted">{t('rp.noData')}</p></Card>
  const maxGen = Math.max(1, ...all.map((p) => p.generation))
  const lines: Series[] = [
    { key: 'all', label: t('rp.overall'), points: series.overall.map((p) => [p.generation, p.value]), width: 1.8 },
    ...series.byType.map((s, k) => ({ key: s.type, label: s.type, points: s.points.map((p) => [p.generation, p.value] as [number, number]), stroke: 'var(--muted)', dash: PATTERNS[(k + 1) % PATTERNS.length], width: 1.1 })),
  ]
  const gens = [...new Set(series.overall.map((p) => p.generation))]
  return (
    <Card title={title}>
      <XYChart
        label={title}
        x={[0, maxGen]}
        y={[0, 100]}
        series={lines}
        ends
        xFormat={(v) => int(v)}
        yFormat={(v) => `${int(v)}%`}
        xTitle={t('ra.genAxis')}
        stops={gens}
        read={(g) => lines.map((l) => `${l.label} ${fmt([...l.points].reverse().find(([x]) => x <= g)?.[1] ?? null, 0)}%`).join(' · ')}
      />
    </Card>
  )
}

/** A model × patch type grid; the cell holds the value, its shade the share of the largest. */
function Heatmap({ rows, cols, cells, format, label }: { rows: string[]; cols: string[]; cells: (number | null)[][]; format: (v: number) => string; label: string }) {
  const { t } = useCopy()
  const top = Math.max(1e-12, ...cells.flat().filter((v): v is number => v !== null))
  if (rows.length === 0 || cols.length === 0) return <p className="small muted">{t('rp.noData')}</p>
  return (
    <div className="tscroll">
      <table className="itable heat" aria-label={label}>
        <thead>
          <tr>
            <th />
            {cols.map((c) => <th key={c} className="num mono" translate="no">{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r}>
              <th className="mono wrap" translate="no">{r}</th>
              {cols.map((c, j) => {
                const v = cells[i][j]
                return (
                  <td key={c} className="num mono" style={v === null ? undefined : { background: `color-mix(in srgb, var(--ramp-1) ${Math.round((v / top) * 70)}%, transparent)` }}>
                    {v === null ? '—' : format(v)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function modelGrid(stats: ModelStats | null, rate: (c: Tally) => number | null) {
  if (!stats) return null
  const labels = modelLabels(stats.rows.map((r) => r.model))
  return { rows: labels, cols: stats.patchTypes, cells: stats.rows.map((r) => r.cells.map(rate)) }
}

/** Per patch type: the programs written against the tries that failed before them, or the improvement rate. */
function TypeBars({ stats, show }: { stats: PatchTypeStat[] | null; show: 'success' | 'improve' }) {
  const { t } = useCopy()
  if (!stats || stats.length === 0) return <p className="small muted">{t('rp.noData')}</p>
  const most = Math.max(1, ...stats.map((s) => s.successes + s.failures))
  return (
    <ol className="bars hbars">
      {stats.map((s) => (
        <li key={s.patchType}>
          <span className="mono faint" translate="no">{s.patchType}</span>
          <span className="bar" aria-hidden="true">
            {show === 'success' ? (
              <>
                <i style={{ width: `${((s.successes + s.failures) / most) * 100}%` }} />
                <b style={{ width: `${(s.successes / most) * 100}%` }} />
              </>
            ) : (
              <b style={{ width: `${(s.improvementRate ?? 0) * 100}%` }} />
            )}
          </span>
          <span className="mono small">
            {show === 'success' ? t('rp.successBar', { s: int(s.successes), f: int(s.failures), p: pct(s.successRate, 0) }) : pct(s.improvementRate, 1)}
          </span>
        </li>
      ))}
    </ol>
  )
}

function Pair({ title, model, type, note }: { title: string; model: ReactNode; type: ReactNode; note?: string }) {
  const { t } = useCopy()
  return (
    <Card title={title} note={note}>
      <h5>{t('rp.perModel')}</h5>
      {model}
      <h5>{t('rp.perType')}</h5>
      {type}
    </Card>
  )
}

type DiagnosticsProps = { programs: RunProgram[]; onSelect: (id: string) => void }

/**
 * R09-E1: a computation that fails leaves a notice in this section instead of taking down the page, as
 * the old page's "Error computing analysis statistics" banner did (VT:17536-17545). The notice is fixed
 * text, never the exception's message. It clears when the run's programs change.
 */
class DiagnosticsBoundary extends Component<{ programs: RunProgram[]; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  componentDidUpdate(prev: { programs: RunProgram[] }) {
    if (prev.programs !== this.props.programs && this.state.failed) this.setState({ failed: false })
  }
  render() {
    return this.state.failed ? <DiagnosticsFailed /> : this.props.children
  }
}

function DiagnosticsFailed() {
  const { t } = useCopy()
  return <Notice kind="error" label={t('icon.error')} title={t('rp.dxFailed')} subtitle={t('rp.dxFailedHint')} />
}

/** R09 under OD-6: the eleven analysis plots, two to a row where the canvas allows. */
export function DiagnosticsSection(props: DiagnosticsProps) {
  return (
    <DiagnosticsBoundary programs={props.programs}>
      <DiagnosticsCharts {...props} />
    </DiagnosticsBoundary>
  )
}

function DiagnosticsCharts({ programs, onSelect }: DiagnosticsProps) {
  const { t } = useCopy()
  const correct = useMemo(() => correctnessRates(programs), [programs])
  const improve = useMemo(() => improvementRates(programs), [programs])
  const bloat = useMemo(() => codeBloat(programs), [programs])
  const avg = useMemo(() => generationAvgCost(programs), [programs])
  const heat = useMemo(() => modelPatchCost(programs), [programs])
  const patches = useMemo(() => ({ llm: llmStats(programs), type: mutationStats(programs) }), [programs])
  const novelty = useMemo(() => ({ llm: noveltyLlmStats(programs), type: noveltyMutationStats(programs) }), [programs])
  const resample = useMemo(() => ({ llm: resampleLlmStats(programs), type: resampleMutationStats(programs) }), [programs])
  const rateGrid = (stats: ModelStats | null, which: 'success' | 'improve', label: string) => {
    const g = modelGrid(stats, (c) => (which === 'success' ? c.successRate : c.improvementRate))
    return g ? <Heatmap {...g} label={label} format={(v) => pct(v, 0)} /> : <p className="small muted">{t('rp.noData')}</p>
  }
  const maxGen = Math.max(1, ...programs.map(generationOf))
  const bloatGen = bloat.points.map((p) => p.generation)
  // Every model × patch type pair is drawn and read, as the old page drew them (VT:16450-16469); the four
  // with the highest mean are dashed and labelled. Models are shortened as in the cost heatmap.
  const comboModels = [...new Set(avg.combos.flatMap((c) => (c.model === null ? [] : [c.model])))]
  const shortModels = costLabels(comboModels)
  const comboName = (c: ComboCost) => `${c.model === null ? '—' : shortModels[comboModels.indexOf(c.model)]} / ${c.patchType ?? '—'}`
  return (
    <div className="pcards">
      <RateChart title={t('rp.dx.correctness')} series={correct} />
      <RateChart title={t('rp.dx.improvement')} series={improve} />
      <Card title={t('rp.dx.bloat')}>
        {bloat.points.length === 0 ? (
          <p className="small muted">{t('rp.noData')}</p>
        ) : (
          <XYChart
            label={t('rp.dx.bloat')}
            x={[0, maxGen]}
            y={[0, Math.max(1, ...bloat.points.map((p) => p.loc))]}
            series={[
              { key: 'loc', label: t('rp.loc'), points: bloat.points.map((p) => [p.generation, p.loc]), width: 1.6 },
              { key: 'lloc', label: t('rp.lloc'), points: bloat.points.map((p) => [p.generation, p.lloc]), stroke: 'var(--muted)', dash: '4 3' },
              ...(bloat.loc ? [{ key: 'trend', label: t('rp.trendLine'), points: [[0, bloat.loc.intercept], [maxGen, bloat.loc.intercept + bloat.loc.slope * maxGen]] as [number, number][], stroke: 'var(--stroke)', dash: '1.5 3', width: 1 }] : []),
            ]}
            ends
            xFormat={(v) => int(v)}
            yFormat={(v) => int(v)}
            xTitle={t('ra.genAxis')}
            stops={bloatGen}
            read={(g) => {
              const p = bloat.points.find((q) => q.generation === g)
              return p ? `${t('ra.gen', { g })} · ${t('rp.loc')} ${fmt(p.loc, 1)} · ${t('rp.lloc')} ${fmt(p.lloc, 1)} · n ${int(p.n)}` : null
            }}
          />
        )}
      </Card>
      <Card title={t('rp.dx.avgCost')} note={t('rp.avgCostNote')}>
        {avg.overall.length === 0 ? (
          <p className="small muted">{t('rp.noData')}</p>
        ) : (
          <XYChart
            label={t('rp.dx.avgCost')}
            height={Math.min(340, Math.max(168, 19 * (avg.combos.length + 3) + 24))}
            x={[0, maxGen]}
            y={[0, Math.max(1e-6, ...avg.overall.map((g) => g.runningMean ?? 0), ...avg.combos.flatMap((c) => c.points.map((p) => p.value ?? 0)))]}
            series={[
              { key: 'all', label: t('rp.overall'), points: avg.overall.flatMap((g) => (g.runningMean === null ? [] : [[g.generation, g.runningMean] as [number, number]])), width: 1.8 },
              ...avg.combos.map((c, k) => ({
                key: `${c.model}|${c.patchType}`,
                label: k < 4 ? comboName(c) : undefined,
                points: c.points.flatMap((p) => (p.value === null ? [] : [[p.generation, p.value] as [number, number]])),
                stroke: k < 4 ? 'var(--muted)' : 'var(--stroke)',
                dash: k < 4 ? PATTERNS[(k + 1) % PATTERNS.length] : undefined,
                width: k < 4 ? 1.1 : 0.8,
              })),
            ]}
            ends
            xFormat={(v) => int(v)}
            yFormat={(v) => money(v)}
            xTitle={t('ra.genAxis')}
            stops={avg.overall.map((g) => g.generation)}
            read={(g) => {
              const row = avg.overall.find((x) => x.generation === g)
              if (!row) return null
              const pairs = avg.combos.flatMap((c) => {
                const p = c.points.find((q) => q.generation === g)
                return p && p.n > 0 ? [`${comboName(c)} ${p.value === null ? '—' : money(p.value)}`] : []
              })
              return [t('ra.gen', { g }), `${t('rp.genMean')} ${row.mean === null ? '—' : money(row.mean)}`, `${t('rp.runningMean')} ${row.runningMean === null ? '—' : money(row.runningMean)}`, ...pairs]
            }}
          />
        )}
      </Card>
      <Pair title={t('rp.dx.patchSuccess')} model={rateGrid(patches.llm, 'success', t('rp.dx.patchSuccess'))} type={<TypeBars stats={patches.type} show="success" />} />
      <Pair title={t('rp.dx.patchImprove')} model={rateGrid(patches.llm, 'improve', t('rp.dx.patchImprove'))} type={<TypeBars stats={patches.type} show="improve" />} />
      <Pair title={t('rp.dx.novelty')} model={rateGrid(novelty.llm, 'success', t('rp.dx.novelty'))} type={<TypeBars stats={novelty.type} show="success" />} />
      <Pair title={t('rp.dx.resample')} model={rateGrid(resample.llm, 'success', t('rp.dx.resample'))} type={<TypeBars stats={resample.type} show="success" />} />
      <Card title={t('rp.dx.costHeat')} note={t('rp.costHeatNote')}>
        <Heatmap rows={costLabels(heat.rows)} cols={heat.cols.map((c) => c ?? '—')} cells={heat.cells} format={(v) => money(v)} label={t('rp.dx.costHeat')} />
      </Card>
      <Card title={t('rp.dx.offspring')}>
        <Offspring programs={programs} onSelect={onSelect} />
      </Card>
    </div>
  )
}
