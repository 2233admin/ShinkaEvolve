import { Delaunay, line, curveMonotoneX, curveStepAfter, scaleLinear, tickStep } from 'd3'
import { useMemo, useState, type KeyboardEvent, type MouseEvent, type PointerEvent } from 'react'
import { fmt, int } from '../format'
import { useCopy } from '../i18n/lang'
import type { TrialPoint } from '../types'
import { frontier } from './data'
import { useWidth } from './useWidth'

type Props = {
  points: TrialPoint[]
  /** The selected hypothesis: its trials are filled, the others hollow. */
  selected: string
  /** Where a trial's program opens, or null when its experiment has no analysis. */
  hrefOf: (p: TrialPoint) => string | null
  onOpen: (href: string) => void
}

const HEIGHT = 280
// Below this width the end labels leave the plot and sit under it, so the plot keeps its width.
const NARROW = 560

/**
 * Every trial of the direction in evaluation order (P07): the record frontier, the noise line drawn by
 * the chart itself, this hypothesis's trials filled and the others hollow, failures as crosses in
 * their own band. Its right edge is the overview's gauge: the orange segment is how far the best
 * record clears the noise line. Pointing at a trial, or moving to it with the arrow keys, shows it;
 * a click or Enter opens its program.
 */
export function TrialChart({ points, selected, hrefOf, onOpen }: Props) {
  const { t } = useCopy()
  const [box, width] = useWidth<HTMLDivElement>()
  const [active, setActive] = useState<number | null>(null)
  const narrow = width < NARROW
  const m = { t: 20, r: narrow ? 16 : 132, b: 40, l: 48 }
  const failY = HEIGHT - m.b - 10
  const plotBottom = failY - 22

  const steps = useMemo(() => frontier(points), [points])
  const last = steps.at(-1) ?? null
  const failed = points.filter((p) => p.score === null).length

  const geometry = useMemo(() => {
    if (width === 0 || points.length === 0) return null
    const x = scaleLinear([1, Math.max(2, points[points.length - 1].n)], [m.l, width - m.r])
    const values = [...points.flatMap((p) => (p.score === null ? [] : [p.score])), ...points.map((p) => p.bar)]
    const lo = Math.min(...values)
    const hi = Math.max(...values)
    const span = hi - lo || 1
    // The bottom snaps to a labelled tick so the lowest scores sit above a labelled line; the top stays at the data.
    const top = hi + span * 0.06
    const bottom = lo - span * 0.04
    const step = tickStep(bottom, top, 4)
    const y = scaleLinear([Math.floor(bottom / step) * step, top], [plotBottom, m.t])
    const at = points.map((p) => [x(p.n), p.score === null ? failY : y(p.score)] as [number, number])
    return { x, y, at, finder: Delaunay.from(at) }
  }, [points, width, m.l, m.r, m.t, plotBottom, failY])

  if (geometry === null) return <div className="tchart" ref={box} />
  const { x, y, at, finder } = geometry
  const ex = last ? x(last.n) : x(points[points.length - 1].n)
  const clears = last !== null && last.best > last.bar
  const gap = last ? last.best - last.bar : 0
  const gapText = `${gap >= 0 ? '+' : '−'}${Math.abs(gap).toFixed(2)}`
  const yTicks = y.ticks(4)
  const xTicks = x.ticks(Math.max(2, Math.floor((width - m.l - m.r) / 70))).filter(Number.isInteger)

  const point = active === null ? null : points[active]
  const href = point ? hrefOf(point) : null
  const describe = (p: TrialPoint) =>
    t('rep.pointLine', { n: int(p.n), ref: `${p.hypothesis} / ${p.experiment}`, gen: p.generation, score: p.score === null ? t('rep.pointFailed') : fmt(p.score) })

  // The trial under the pointer, if one is within reach; a tap is looked up from its own position, never from hover.
  const hit = (event: PointerEvent<SVGRectElement> | MouseEvent<SVGRectElement>) => {
    const bounds = event.currentTarget.ownerSVGElement!.getBoundingClientRect()
    const px = event.clientX - bounds.left
    const py = event.clientY - bounds.top
    const index = finder.find(px, py)
    const [cx, cy] = at[index]
    return Math.hypot(cx - px, cy - py) <= 14 ? index : null
  }
  const open = (index: number | null) => {
    if (index === null) return
    const target = hrefOf(points[index])
    if (target) onOpen(target)
  }
  const recordIndex = () => {
    let best = -1
    points.forEach((p, i) => {
      if (p.score !== null && (best === -1 || p.score > (points[best].score as number))) best = i
    })
    return best === -1 ? 0 : best
  }
  // ← → step through the trials in order, Home and End jump to the ends; the first key lands on the best record.
  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter') {
      if (active === null) return
      event.preventDefault()
      open(active)
      return
    }
    if (event.key === 'Escape') {
      setActive(null)
      return
    }
    const end = points.length - 1
    const from = active ?? recordIndex()
    const moves: Record<string, number> = { ArrowRight: from + 1, ArrowLeft: from - 1, Home: 0, End: end }
    if (!(event.key in moves)) return
    event.preventDefault()
    setActive(active === null ? from : Math.max(0, Math.min(end, moves[event.key])))
  }

  const labels = last && (
    <>
      <span className="lbl">{t('rep.record')} <b className="mono">{fmt(last.best)}</b></span>
      <span className={clears ? 'lbl sig' : 'lbl muted'}>{t(clears ? 'rep.clears' : 'rep.below')} <b className="mono">{gapText}</b></span>
      <span className="lbl muted">{t('rep.noise')} <b className="mono">{fmt(last.bar)}</b></span>
    </>
  )

  return (
    <div className="tchart" ref={box}>
      <div
        className="tchart-plot"
        tabIndex={0}
        role="group"
        aria-label={t('rep.chartLabel', { n: int(points.length), failed: int(failed) })}
        aria-describedby="tchart-help"
        onKeyDown={onKey}
        onFocus={(event) => {
          // Keyboard focus lands on the best record; a pointer keeps what it points at.
          if (event.currentTarget.matches(':focus-visible')) setActive((a) => a ?? recordIndex())
        }}
        onBlur={() => setActive(null)}
      >
        <svg width={width} height={HEIGHT} viewBox={`0 0 ${width} ${HEIGHT}`} aria-hidden="true">
          {yTicks.map((v) => (
            <g key={v}>
              <line x1={m.l} x2={ex} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
              <text x={m.l - 8} y={y(v) + 4} textAnchor="end" className="axis">{fmt(v)}</text>
            </g>
          ))}
          <text x={m.l} y={10} className="axis-title">{t('rep.scoreAxis')}</text>
          {points.map((p, i) =>
            p.score === null ? null : p.hypothesis === selected ? (
              <circle key={i} cx={at[i][0]} cy={at[i][1]} r={2} fill="var(--muted)" />
            ) : (
              <circle key={i} cx={at[i][0]} cy={at[i][1]} r={1.9} fill="none" stroke="var(--stroke)" />
            ),
          )}
          <line x1={m.l} x2={ex} y1={failY - 11} y2={failY - 11} stroke="var(--line)" />
          <text x={m.l - 8} y={failY + 4} textAnchor="end" className="axis">{t('rep.failed')}</text>
          {points.map((p, i) =>
            p.score === null ? <path key={i} d={`M${at[i][0] - 2.5},${failY - 2.5}l5,5m0,-5l-5,5`} stroke="var(--signal-graphic)" strokeWidth={1.2} /> : null,
          )}
          {steps.length > 0 && (
            <>
              <path d={line<(typeof steps)[number]>().curve(curveMonotoneX).x((s) => x(s.n)).y((s) => y(s.bar))(steps) ?? ''} fill="none" stroke="var(--muted)" strokeWidth={1.4} strokeDasharray="4 3" />
              <path d={line<(typeof steps)[number]>().curve(curveStepAfter).x((s) => x(s.n)).y((s) => y(s.best))(steps) ?? ''} fill="none" stroke="var(--text)" strokeWidth={1.6} />
            </>
          )}
          {last && (
            <>
              <line x1={ex} x2={ex} y1={y(last.bar)} y2={y(last.best)} stroke={clears ? 'var(--signal-graphic)' : 'var(--muted)'} strokeWidth={2.5} />
              <rect x={ex - 5} y={y(last.bar) - 1} width={10} height={2} fill="var(--muted)" />
              <path d={`M${ex},${y(last.best) - 6}l5.5,6-5.5,6-5.5-6z`} fill={clears ? 'var(--signal-graphic)' : 'var(--text)'} />
            </>
          )}
          {last && !narrow && (
            <>
              <text x={ex + 12} y={y(last.best) + 4} className="lbl">{t('rep.record')} {fmt(last.best)}</text>
              <text x={ex + 12} y={(y(last.best) + y(last.bar)) / 2 + 4} className={clears ? 'lbl sig' : 'lbl muted'}>{t(clears ? 'rep.clears' : 'rep.below')} {gapText}</text>
              <text x={ex + 12} y={y(last.bar) + 4} className="lbl muted">{t('rep.noise')} {fmt(last.bar)}</text>
            </>
          )}
          <line x1={m.l} x2={ex} y1={HEIGHT - m.b + 2} y2={HEIGHT - m.b + 2} stroke="var(--stroke)" />
          {xTicks.map((v) => (
            <g key={v}>
              <line x1={x(v)} x2={x(v)} y1={HEIGHT - m.b + 2} y2={HEIGHT - m.b + 6} stroke="var(--stroke)" />
              <text x={x(v)} y={HEIGHT - m.b + 19} textAnchor="middle" className="axis">{int(v)}</text>
            </g>
          ))}
          <text x={m.l} y={HEIGHT - 3} className="axis-title">{t('rep.programAxis', { n: int(points[points.length - 1].n) })}</text>
          {active !== null && <circle cx={at[active][0]} cy={at[active][1]} r={6} fill="none" stroke="var(--text)" strokeWidth={1.5} />}
          <rect
            x={0}
            y={0}
            width={width}
            height={HEIGHT}
            fill="transparent"
            className={href ? 'tchart-hit is-link' : 'tchart-hit'}
            onPointerMove={(event) => setActive(hit(event))}
            onPointerLeave={() => setActive(null)}
            onClick={(event) => open(hit(event))}
          />
        </svg>
        {point && (
          // The note sits beside the trial, on whichever side has more room.
          <div
            className="tchart-tip"
            style={{
              left: at[active!][0] + (at[active!][0] > width / 2 ? -12 : 12),
              top: Math.max(0, at[active!][1] - 56),
              transform: at[active!][0] > width / 2 ? 'translateX(-100%)' : undefined,
            }}
          >
            <span className="mono">{describe(point)}</span>
            {href && <span className="faint">{t('rep.pointOpen')}</span>}
          </div>
        )}
      </div>
      {narrow && <p className="tchart-labels">{labels}</p>}
      <p id="tchart-help" className="sr-only">{t('rep.chartHelp')}</p>
      <p className="sr-only" aria-live="polite">{point ? describe(point) : ''}</p>
    </div>
  )
}
