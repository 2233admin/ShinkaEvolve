import { useMemo, useState, type KeyboardEvent } from 'react'
import { fmt, int, pct } from '../format'
import { useCopy } from '../i18n/lang'
import { fertility, fitLine, generationOf, mutationPoints, nameOf, patchTypeOf, type MutationPoint, type RunProgram } from './data'
import { signed } from './Inspector'
import { ProgramButton } from './panes'

/**
 * R09-Q4: mutation size against the change in score, with failures in their own band. Pointing at a
 * point, or moving to it with ← → once the chart has focus, reads it out; a click or Enter opens it
 * in the inspector.
 */
function MutationChart({ points, onSelect }: { points: MutationPoint[]; onSelect: (id: string) => void }) {
  const { t } = useCopy()
  const [active, setActive] = useState<number | null>(null)
  const drawn = points.filter((p) => p.delta !== null)
  const fit = fitLine(drawn.map((p) => ({ x: p.size, y: p.delta as number })))
  const W = 360
  const H = 200
  const m = { l: 44, r: 10, t: 10, b: 46 }
  const maxSize = Math.max(1, ...points.map((p) => p.size))
  const deltas = drawn.map((p) => p.delta as number)
  const dlo = Math.min(0, ...deltas)
  const dhi = Math.max(0, ...deltas)
  const sx = (v: number) => m.l + (v / maxSize) * (W - m.l - m.r)
  const syv = (v: number) => m.t + (dhi > dlo ? 1 - (v - dlo) / (dhi - dlo) : 0.5) * (H - m.t - m.b - 18)
  const band = H - m.b + 4
  // Points in reading order, left to right, so the arrow keys walk along the size axis.
  const at = useMemo(
    () => points.map((p) => ({ p, x: sx(p.size), y: p.delta === null ? band : syv(p.delta) })).sort((a, b) => a.x - b.x || a.y - b.y),
    // The scales follow from the points.
    [points],
  )
  const describe = (p: MutationPoint) =>
    `${nameOf(p.program)} (${p.program.id.slice(0, 8)}) · ${p.delta === null ? t(p.failed === 'self' ? 'ra.failed' : 'ra.parentFailed') : `Δ ${signed(p.delta, 4)}`} · ${t('ra.linesChanged', { n: p.size })} (+${p.added} −${p.deleted} ~${p.modified}) · ${patchTypeOf(p.program) ?? '—'}`
  const hit = (event: { clientX: number; clientY: number; currentTarget: SVGSVGElement }) => {
    const box = event.currentTarget.getBoundingClientRect()
    const k = W / box.width
    const px = (event.clientX - box.left) * k
    const py = (event.clientY - box.top) * k
    let near: number | null = null
    at.forEach((a, i) => {
      if (Math.hypot(a.x - px, a.y - py) <= 8 && (near === null || Math.hypot(a.x - px, a.y - py) < Math.hypot(at[near].x - px, at[near].y - py))) near = i
    })
    return near
  }
  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') return setActive(null)
    if ((event.key === 'Enter' || event.key === ' ') && active !== null) {
      event.preventDefault()
      return onSelect(at[active].p.program.id)
    }
    const end = at.length - 1
    const moves: Record<string, number> = { ArrowRight: (active ?? -1) + 1, ArrowLeft: (active ?? 1) - 1, Home: 0, End: end }
    if (!(event.key in moves)) return
    event.preventDefault()
    setActive(Math.max(0, Math.min(end, moves[event.key])))
  }
  const current = active === null ? null : at[active]
  return (
    <div className="chart-box mchart" tabIndex={0} role="group" aria-label={t('ra.mutation')} aria-describedby="mchart-help" onKeyDown={onKey} onBlur={() => setActive(null)}>
      <svg
        className="chart"
        width="100%"
        viewBox={`0 0 ${W} ${H}`}
        aria-hidden="true"
        onPointerMove={(event) => setActive(hit(event))}
        onPointerLeave={() => setActive(null)}
        onClick={(event) => {
          const i = hit(event)
          if (i !== null) onSelect(at[i].p.program.id)
        }}
      >
        <line x1={m.l} x2={W - m.r} y1={syv(0)} y2={syv(0)} stroke="var(--grid)" />
        <text x={m.l - 6} y={syv(dhi) + 4} textAnchor="end" className="axis">{signed(dhi)}</text>
        <text x={m.l - 6} y={syv(0) + 4} textAnchor="end" className="axis">0</text>
        {dlo < 0 && <text x={m.l - 6} y={syv(dlo) + 4} textAnchor="end" className="axis">{fmt(dlo)}</text>}
        <line x1={m.l} x2={W - m.r} y1={band - 8} y2={band - 8} stroke="var(--line)" strokeDasharray="2 3" />
        <text x={m.l - 6} y={band + 3} textAnchor="end" className="axis">{t('ra.failedShort')}</text>
        {fit && <line x1={sx(0)} x2={sx(maxSize)} y1={syv(fit.intercept)} y2={syv(fit.intercept + fit.slope * maxSize)} stroke="var(--muted)" strokeWidth={1.2} strokeDasharray="4 3" />}
        {at.map(({ p, x, y }) =>
          p.delta === null ? (
            p.failed === 'self' ? (
              <path key={p.program.id} d={`M${x - 3},${y - 3}l6,6m0,-6l-6,6`} stroke="var(--signal-graphic)" strokeWidth={1.3} />
            ) : (
              <circle key={p.program.id} cx={x} cy={y} r={3} fill="none" stroke="var(--muted)" />
            )
          ) : (
            <circle key={p.program.id} cx={x} cy={y} r={3} fill={p.delta > 0 ? 'var(--text)' : 'var(--stroke)'} />
          ),
        )}
        {current && <circle cx={current.x} cy={current.y} r={6.5} fill="none" stroke="var(--text)" strokeWidth={1.5} strokeDasharray="2 2" />}
        <line x1={m.l} x2={W - m.r} y1={H - 26} y2={H - 26} stroke="var(--stroke)" />
        {[0, maxSize / 2, maxSize].map((v) => (
          <text key={v} x={sx(v)} y={H - 12} textAnchor="middle" className="axis">{int(v)}</text>
        ))}
        <text x={W - m.r} y={H - 1} textAnchor="end" className="axis-title">{t('ra.mutationAxis')}</text>
      </svg>
      {current && (
        <div className="lanes-tip mchart-tip" style={{ left: `${(current.x / W) * 100}%`, top: `${(current.y / H) * 100}%` }}>
          <span className="mono" translate="no">{describe(current.p)}</span>
          <span className="faint">{t('ra.pointOpen')}</span>
        </div>
      )}
      <p id="mchart-help" className="sr-only">{t('ra.mutationHelp')}</p>
      <p className="sr-only" aria-live="polite">{current ? describe(current.p) : ''}</p>
      <p className="small faint">{t('ra.mutationNote', { n: int(drawn.length), f: int(points.length - drawn.length) })}{fit ? ` ${t('ra.trend', { s: signed(fit.slope * 100, 4) })}` : ''}</p>
    </div>
  )
}

/** R09-Q11 and R09-Q4: the most fertile parents, and mutation size against the change in score; in the lineage panel and in Progress › 诊断 (OD-6). */
export function Offspring({ programs, onSelect }: { programs: RunProgram[]; onSelect: (id: string) => void }) {
  const { t } = useCopy()
  const { top, deadEnds } = useMemo(() => fertility(programs), [programs])
  const points = useMemo(() => mutationPoints(programs), [programs])
  const most = Math.max(1, ...top.map((x) => x.children))
  return (
    <>
      <h3>{t('ra.fertile')}</h3>
      {top.length === 0 ? (
        <p className="small muted">{t('ra.noFertile')}</p>
      ) : (
        <ol className="bars">
          {top.map((x) => (
            <li key={x.program.id}>
              <span className="mono faint">{t('ra.gen', { g: generationOf(x.program) })}</span>
              <ProgramButton p={x.program} onSelect={onSelect} />
              <span className="bar" aria-hidden="true">
                <i style={{ width: `${(x.children / most) * 100}%` }} />
                <b style={{ width: `${(x.improving / most) * 100}%` }} />
              </span>
              <span className="mono small">{t('ra.childrenOf', { c: x.children, i: x.improving, p: pct(x.children ? x.improving / x.children : 0) })}</span>
            </li>
          ))}
        </ol>
      )}
      <p className="small faint">{t('ra.deadEnds', { n: int(deadEnds) })}</p>
      <h3>{t('ra.mutation')}</h3>
      {points.length === 0 ? <p className="small muted">{t('ra.noMutation')}</p> : <MutationChart points={points} onSelect={onSelect} />}
    </>
  )
}
