import { curveMonotoneX, curveStepAfter, line, scaleLinear } from 'd3'
import { useMemo, useState, type KeyboardEvent, type MouseEvent, type PointerEvent, type ReactNode } from 'react'
import { DASHES, scoreOf, stackLabels } from '../compare/series'
import { Icon } from '../design/icons'
import { useWidth } from '../design/useWidth'
import { fmt, int, money, pct } from '../format'
import { useCopy } from '../i18n/lang'
import { runCost, runtimeText, type RunStats } from '../runs/data'
import { generationOf, islandOf, lanesOf, lineOf, type Prompt, type RunProgram } from './data'
import { costTotals, generationRows, modelRows, posteriorSeries, type GenerationRow } from './progressData'
import { DiagnosticsSection, MetricsSection, RuntimeSection, SpendSection } from './ProgressSections'
import { Scrubber } from './Scrubber'

// The approved geometry (hifi/build.mjs › progressView): a 64px gutter for each panel's y labels, 128px
// on the right for the noise bracket's labels, and the five panels' heights.
export const PROGRESS_M = { l: 64, r: 128 }
const H = { score: 116, cost: 60, failed: 56, programs: 50, posterior: 56 }
// Posterior lines take the hi-fi's three inks, then the comparison page's patterns.
const POST_INK: [string, number][] = [['var(--text)', 1.6], ['var(--muted)', 1.4], ['var(--stroke)', 1.2]]

type Scrub = { max: number; value: number; onChange: (generation: number) => void; count: number }

type Props = {
  programs: RunProgram[]
  byId: ReadonlyMap<string, RunProgram>
  /** The last generation shown (the scrubber); the axis always runs to the run's last. */
  upTo: number
  maxGen: number
  selected: RunProgram | null
  onSelect: (id: string) => void
  prompts: Prompt[]
  promptsError: string | null
  stats: RunStats
  /** The direction's noise bar at each generation of this run; empty for an ordinary run. */
  noise: ReadonlyMap<number, number>
  scrub: Scrub
}

type Panel = 'score' | 'cost' | 'failed' | 'programs' | 'posterior'

/**
 * Progress (6.3, design.md › 四个分析视图 › 进展): is the search still getting better. Five small
 * multiples share the generation axis and the scrubber; the per-model table follows, then the
 * per-metric charts, score against spend, runtime and the diagnostics in disclosures.
 */
export function Progress({ programs, byId, upTo, maxGen, selected, onSelect, prompts, promptsError, stats, noise, scrub }: Props) {
  const { t } = useCopy()
  const [box, width] = useWidth<HTMLDivElement>()
  const [active, setActive] = useState<{ g: number; y: number } | null>(null)
  const [byIsland, setByIsland] = useState(false)
  const [range, setRange] = useState({ lo: '', hi: '' })

  const promptsKnown = !stats.has_prompt_evo || promptsError === null
  const rows = useMemo(() => generationRows(programs, prompts, promptsKnown), [programs, prompts, promptsKnown])
  const totals = useMemo(() => costTotals(programs, stats.has_prompt_evo ? stats.prompt_evo_cost : 0, runCost(stats)), [programs, stats])
  const posteriors = useMemo(() => posteriorSeries(programs), [programs])
  const models = useMemo(() => modelRows(programs, posteriors), [programs, posteriors])
  const shown = rows.filter((r) => r.generation <= upTo)
  const zeroCost = totals.total === 0
  const runBest = rows[rows.length - 1]?.cumulative ?? null

  // The score axis spans every valid score and the noise line, rounded out to tick values, so nothing
  // is hidden until the inputs narrow it; an entered limit is used exactly (R01-C1). The whole run sets
  // the automatic range, so replay keeps it still.
  const extent = useMemo(() => {
    const scores = programs.map(scoreOf).filter((s): s is number => s !== null)
    return scores.length ? [Math.min(...scores), Math.max(...scores)] : [0, 1]
  }, [programs])
  const noiseValues = [...noise.values()]
  const raw: [number, number] = [Math.min(extent[0], ...noiseValues), Math.max(extent[1], ...noiseValues)]
  const auto = scaleLinear(raw[1] > raw[0] ? raw : [raw[0] - 1, raw[0] + 1], [0, 1]).nice().domain() as [number, number]
  const lo = range.lo.trim() === '' ? auto[0] : Number(range.lo)
  const hi = range.hi.trim() === '' ? auto[1] : Number(range.hi)
  const scoreRange: [number, number] = Number.isFinite(lo) && Number.isFinite(hi) && hi > lo ? [lo, hi] : auto

  const x = scaleLinear([0, Math.max(1, maxGen)], [PROGRESS_M.l, Math.max(PROGRESS_M.l + 1, width - PROGRESS_M.r)])
  const bw = Math.max(2, Math.min(14, (x(1) - x(0)) * 0.62))
  const gx = (g: number) => x(g).toFixed(1)
  const generationAt = (px: number) => Math.max(0, Math.min(upTo, Math.round(x.invert(px))))
  const ys = (height: number, domain: [number, number]) => scaleLinear(domain, [height - 6, 10]).nice()

  const yScore = scaleLinear(scoreRange, [H.score - 6, 10])
  const points = programs.filter((p) => generationOf(p) <= upTo && scoreOf(p) !== null)
  const inRange = points.filter((p) => {
    const s = scoreOf(p) as number
    return s >= scoreRange[0] && s <= scoreRange[1]
  })
  const outside = points.length - inRange.length
  const line0 = selected && generationOf(selected) <= upTo ? lineOf(byId, selected.id).filter((p) => generationOf(p) <= upTo && scoreOf(p) !== null) : []
  const ringed = new Set(line0.map((p) => p.id))
  const head = shown[shown.length - 1] ?? null
  const bestProgram = head?.cumulativeId ? (byId.get(head.cumulativeId) ?? null) : null
  const lastNoiseGen = [...noise.keys()].filter((g) => g <= upTo).pop()
  const bracket = head && head.cumulative !== null && lastNoiseGen !== undefined ? { best: head.cumulative, bar: noise.get(lastNoiseGen) as number } : null

  // Per-island running best (D-1: the islands as a facet of the one score chart).
  const islandLines = useMemo(() => {
    return lanesOf(programs).map((island) => {
      let best: number | null = null
      const pts: [number, number][] = []
      for (let g = 0; g <= maxGen; g += 1) {
        for (const p of programs) {
          if (generationOf(p) !== g || islandOf(p) !== island) continue
          const s = scoreOf(p)
          if (s !== null && (best === null || s > best)) best = s
        }
        if (best !== null) pts.push([g, best])
      }
      return { island, pts }
    })
  }, [programs, maxGen])

  const costOf = (r: GenerationRow): number | null => (zeroCost ? r.computeTime : r.modelCost === null || r.metaCost === null || r.promptCost === null ? null : r.modelCost + r.metaCost + r.promptCost)
  // R01-Q7: the spend so far at the end of each generation; it stops at the first generation with an unpriced part.
  const cumulativeCost: (number | null)[] = []
  for (const r of rows) {
    const c = costOf(r)
    const before = cumulativeCost.length ? cumulativeCost[cumulativeCost.length - 1] : 0
    cumulativeCost.push(c === null || before === null ? null : before + c)
  }
  const cumulativePairs = rows.flatMap((r, i): [number, number][] => (r.generation <= upTo && cumulativeCost[i] !== null ? [[r.generation, cumulativeCost[i] as number]] : []))
  const cumulativeEnd = cumulativePairs.length && cumulativePairs[cumulativePairs.length - 1][0] === upTo ? cumulativePairs[cumulativePairs.length - 1][1] : null
  const costMax = Math.max(1e-9, ...shown.map((r) => costOf(r) ?? 0))
  const yCost = ys(H.cost, [0, costMax])
  const cumMax = Math.max(1e-9, ...cumulativeCost.filter((v): v is number => v !== null))
  const yCum = scaleLinear([0, cumMax], [H.cost - 6, 10])
  const failedMax = Math.max(3, ...shown.map((r) => r.failedIds.length))
  const yFailed = ys(H.failed, [0, failedMax])
  // One × per failed program however many fail in a generation (design.md › 进展): the crosses shrink to their spacing.
  const crossHalf = Math.max(1.2, Math.min(3, ((H.failed - 16) / failedMax - 1) / 2))
  const yPrograms = ys(H.programs, [0, Math.max(1, ...rows.map((r) => r.programs))])
  const postMax = Math.max(1e-9, ...posteriors.rows.flatMap((r) => Object.values(r.means).filter((v): v is number => v !== null)))
  const yPost = ys(H.posterior, [0, postMax])
  const even = posteriors.models.length ? 1 / posteriors.models.length : null

  // R01-C2: a click selects the program under the pointer, else the generation's best (or, when every
  // program of the generation failed, its first failure).
  const pick = (g: number) => {
    const row = rows[g]
    return row?.bestId ?? row?.failedIds[0] ?? null
  }
  const nearest = (panel: Panel, px: number, py: number): string | null => {
    if (panel === 'score') {
      let id: string | null = null
      let d = 7
      for (const p of inRange) {
        const dist = Math.hypot(x(generationOf(p)) - px, yScore(scoreOf(p) as number) - py)
        if (dist < d) {
          d = dist
          id = p.id
        }
      }
      return id
    }
    if (panel === 'failed') {
      const g = generationAt(px)
      const k = Math.floor(yFailed.invert(py))
      return Math.abs(x(g) - px) < 6 ? (rows[g]?.failedIds[k] ?? null) : null
    }
    return null
  }
  const local = (event: PointerEvent<SVGSVGElement> | MouseEvent<SVGSVGElement>) => {
    const r = event.currentTarget.getBoundingClientRect()
    return [event.clientX - r.left, event.clientY - r.top, r.top - (box.current?.getBoundingClientRect().top ?? 0)] as const
  }
  const handlers = (panel: Panel) => ({
    onPointerMove: (event: PointerEvent<SVGSVGElement>) => {
      const [px, py, top] = local(event)
      if (px < PROGRESS_M.l - 6 || px > width - PROGRESS_M.r + 6) return setActive(null)
      setActive({ g: generationAt(px), y: top + py })
    },
    onPointerLeave: () => setActive(null),
    onClick: (event: MouseEvent<SVGSVGElement>) => {
      const [px, py] = local(event)
      if (px < PROGRESS_M.l - 6 || px > width - PROGRESS_M.r + 6) return
      const id = nearest(panel, px, py) ?? pick(generationAt(px))
      if (id) onSelect(id)
    },
  })
  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    const from = active?.g ?? (selected ? Math.min(upTo, generationOf(selected)) : upTo)
    const moves: Record<string, number> = { ArrowRight: from + 1, ArrowLeft: from - 1, Home: 0, End: upTo }
    if (event.key === 'Enter' && active) {
      const id = pick(active.g)
      if (id) onSelect(id)
    } else if (event.key === 'Escape') setActive(null)
    // The first arrow shows where the reading starts (the selected program's generation); Home and End jump at once.
    else if (event.key in moves) setActive({ g: Math.max(0, Math.min(upTo, active || event.key === 'Home' || event.key === 'End' ? moves[event.key] : from)), y: 8 })
    else return
    event.preventDefault()
  }

  const grid = (y: ReturnType<typeof ys>, format: (v: number) => string, count = 3) =>
    y.ticks(count).map((v) => (
      <g key={v}>
        <line x1={PROGRESS_M.l} x2={width - PROGRESS_M.r} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
        <text x={PROGRESS_M.l - 8} y={y(v) + 3.5} textAnchor="end" className="axis">{format(v)}</text>
      </g>
    ))
  const marks = (height: number) => (
    <>
      {selected && generationOf(selected) <= upTo && (
        <line x1={x(generationOf(selected))} x2={x(generationOf(selected))} y1={4} y2={height - 2} stroke="var(--text)" strokeWidth={1} strokeDasharray="2 3" />
      )}
      {active && <line x1={x(active.g)} x2={x(active.g)} y1={4} y2={height - 2} stroke="var(--stroke)" strokeWidth={1} />}
    </>
  )
  const panel = (kind: Panel, title: string, legend: ReactNode, height: number, body: ReactNode, aside?: ReactNode) => (
    <section className="pm" aria-label={title}>
      <div className="pmhead">
        <b>{title}</b>
        {legend && <span className="legend">{legend}</span>}
        {aside}
      </div>
      {width > 0 && (
        <svg className="chart" width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true" {...handlers(kind)}>
          {body}
          {marks(height)}
        </svg>
      )}
    </section>
  )

  const islandEnds = byIsland ? stackLabels(islandLines.map((l) => yScore(l.pts.filter(([g]) => g <= upTo).at(-1)?.[1] ?? scoreRange[0])), 12, 8, H.score - 4) : []
  const midX = (PROGRESS_M.l + width - PROGRESS_M.r) / 2
  const reading = active ? rows[active.g] : null
  const readingText = reading
    ? [
        t('ra.gen', { g: reading.generation }),
        `${t('rp.perGenBest')} ${fmt(reading.best)}`,
        `${t('rp.cumBest')} ${fmt(reading.cumulative)}`,
        `${t('rp.median')} ${fmt(reading.median)}`,
        ...(noise.has(reading.generation) ? [`${t('rp.noise')} ${fmt(noise.get(reading.generation))}`] : []),
        zeroCost
          ? `${t('rp.computeShort')} ${runtimeText(reading.computeTime)}`
          : `${t('rp.cost')} ${costOf(reading) === null ? t('ra.costUnknown') : money(costOf(reading) as number)}`,
        t('rp.readCounts', { n: int(reading.programs), f: int(reading.failedIds.length) }),
        ...posteriors.models.map((m) => `${m} ${fmt(posteriors.rows.find((r) => r.generation === reading.generation)?.means[m] ?? null)}`),
      ]
    : []

  return (
    <div className="lineage progress-view">
      <div className="vbar">
        <span className="legend">
          {zeroCost ? (
            <span>{t('rp.zeroCost', { time: runtimeText(rows.reduce((s, r) => s + (r.computeTime ?? 0), 0)), avg: runtimeText(rows.reduce((s, r) => s + (r.computeTime ?? 0), 0) / Math.max(1, programs.length)) })}</span>
          ) : (
            <CostLine totals={totals} />
          )}
          <span>{t('rp.validShare', { n: int(programs.filter((p) => scoreOf(p) !== null).length), total: int(programs.length), pct: pct(programs.length ? programs.filter((p) => scoreOf(p) !== null).length / programs.length : null, 1) })}</span>
        </span>
        <span className="vtools">
          {outside > 0 && <span className="small sig">{t('ra.outside', { n: int(outside) })}</span>}
          <label className="range">
            <span className="faint">{t('ra.scoreAxis')}</span>
            <input type="number" step="any" inputMode="decimal" value={range.lo} placeholder={fmt(auto[0])} aria-label={t('ra.scoreMin')} onChange={(e) => setRange((r) => ({ ...r, lo: e.target.value }))} />
            <span className="faint">–</span>
            <input type="number" step="any" inputMode="decimal" value={range.hi} placeholder={fmt(auto[1])} aria-label={t('ra.scoreMax')} onChange={(e) => setRange((r) => ({ ...r, hi: e.target.value }))} />
          </label>
          {(range.lo !== '' || range.hi !== '') && (
            <button type="button" className="btn is-small" onClick={() => setRange({ lo: '', hi: '' })}>{t('ra.scoreAuto')}</button>
          )}
        </span>
      </div>
      <div className="progress" ref={box}>
        <div
          className="pgroup"
          tabIndex={0}
          role="group"
          aria-label={t('rp.groupLabel', { g: int(upTo + 1), n: int(scrub.count) })}
          aria-describedby="progress-help"
          onKeyDown={onKey}
          onBlur={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setActive(null)
          }}
        >
          {panel(
            'score',
            t('rp.cumBest'),
            <>
              <span><i className="ln" />{t('rp.cumBest')}</span>
              {noise.size > 0 && <span><i className="ln is-dash" />{t('rp.noise')}</span>}
              <span><i className="ln is-thin" />{t('rp.median')}</span>
              <span><i className="dot" />{t('rp.perGenBest')}</span>
              <span><i className="dot is-hollow" />{t('rp.validPrograms')}</span>
            </>,
            H.score,
            <>
              {grid(yScore, (v) => fmt(v))}
              <defs>
                <clipPath id="pscore-clip">
                  <rect x={PROGRESS_M.l - 6} y={0} width={Math.max(0, width - PROGRESS_M.l - PROGRESS_M.r + 12)} height={H.score} />
                </clipPath>
              </defs>
              <g clipPath="url(#pscore-clip)">
                {inRange.map((p) => (
                  <circle key={p.id} cx={gx(generationOf(p))} cy={yScore(scoreOf(p) as number).toFixed(1)} r={1.7} fill="none" stroke="var(--stroke)" strokeWidth={0.8} />
                ))}
                <path d={line<GenerationRow>().defined((r) => r.median !== null).x((r) => x(r.generation)).y((r) => yScore(r.median as number))(shown) ?? ''} fill="none" stroke="var(--stroke)" strokeWidth={1} />
                {noise.size > 0 && (
                  <path
                    d={line<[number, number]>().curve(curveMonotoneX).x(([g]) => x(g)).y(([, v]) => yScore(v))([...noise].filter(([g]) => g <= upTo)) ?? ''}
                    fill="none"
                    stroke="var(--muted)"
                    strokeWidth={1.4}
                    strokeDasharray="4 3"
                  />
                )}
                {shown.map((r) => (r.best === null ? null : <circle key={r.generation} cx={gx(r.generation)} cy={yScore(r.best)} r={2.2} fill="var(--muted)" />))}
                {byIsland &&
                  islandLines.map((l, k) => (
                    <path
                      key={String(l.island)}
                      d={line<[number, number]>().curve(curveStepAfter).x(([g]) => x(g)).y(([, v]) => yScore(v))(l.pts.filter(([g]) => g <= upTo)) ?? ''}
                      fill="none"
                      stroke="var(--muted)"
                      strokeWidth={1.1}
                      strokeDasharray={[...DASHES, '8 3 1.5 3'][k % 4]}
                    />
                  ))}
                <path d={line<GenerationRow>().curve(curveStepAfter).defined((r) => r.cumulative !== null).x((r) => x(r.generation)).y((r) => yScore(r.cumulative as number))(shown) ?? ''} fill="none" stroke="var(--text)" strokeWidth={1.8} />
                {line0.map((p) => (
                  <circle key={p.id} cx={gx(generationOf(p))} cy={yScore(scoreOf(p) as number)} r={p.id === selected?.id ? 6 : 3.6} fill="none" stroke="var(--text)" strokeWidth={p.id === selected?.id ? 1.5 : 1} />
                ))}
                {bestProgram && !ringed.has(bestProgram.id) && (
                  <circle cx={gx(generationOf(bestProgram))} cy={yScore(scoreOf(bestProgram) as number)} r={8.5} fill="none" stroke="var(--text)" strokeWidth={1.5} />
                )}
              </g>
              {bestProgram && (
                <text
                  x={x(generationOf(bestProgram)) + (x(generationOf(bestProgram)) > midX ? -14 : 14)}
                  y={Math.max(12, yScore(scoreOf(bestProgram) as number) - 10)}
                  textAnchor={x(generationOf(bestProgram)) > midX ? 'end' : 'start'}
                  className="lbl halo"
                >
                  {t('ra.best')} {fmt(scoreOf(bestProgram))} · {t('ra.gen', { g: generationOf(bestProgram) })}
                </text>
              )}
              {bracket && (
                <>
                  <line x1={x(upTo)} x2={x(upTo)} y1={yScore(bracket.bar)} y2={yScore(bracket.best)} stroke={bracket.best > bracket.bar ? 'var(--signal-graphic)' : 'var(--muted)'} strokeWidth={2.5} />
                  <rect x={x(upTo) - 5} y={yScore(bracket.bar) - 1} width={10} height={2} fill="var(--muted)" />
                  <path d={`M${x(upTo)},${yScore(bracket.best) - 6}l5.5,6-5.5,6-5.5-6z`} fill={bracket.best > bracket.bar ? 'var(--signal-graphic)' : 'var(--text)'} />
                </>
              )}
              {bracket && !byIsland && (
                <>
                  <text x={x(upTo) + 12} y={yScore(bracket.best) + 4} className="lbl">{t('rep.record')} {fmt(bracket.best)}</text>
                  <text x={x(upTo) + 12} y={(yScore(bracket.best) + yScore(bracket.bar)) / 2 + 4} className={bracket.best > bracket.bar ? 'lbl sig' : 'lbl muted'}>
                    {t(bracket.best > bracket.bar ? 'rep.clears' : 'rep.below')} {bracket.best > bracket.bar ? '+' : ''}{fmt(bracket.best - bracket.bar)}
                  </text>
                  <text x={x(upTo) + 12} y={yScore(bracket.bar) + 4} className="lbl muted">{t('rep.noise')} {fmt(bracket.bar)}</text>
                </>
              )}
              {byIsland &&
                islandLines.map((l, k) => (
                  <text key={String(l.island)} x={width - PROGRESS_M.r + 10} y={islandEnds[k] + 3.5} className="axis">
                    {l.island === null ? t('ra.noIsland') : t('ra.island', { k: l.island })} {fmt(l.pts.filter(([g]) => g <= upTo).at(-1)?.[1] ?? null)}
                  </text>
                ))}
            </>,
            <button type="button" className="toggle" aria-pressed={byIsland} onClick={() => setByIsland((on) => !on)}>
              {t('rp.byIsland')}
            </button>,
          )}
          {panel(
            'cost',
            zeroCost ? t('rp.computeTitle') : t('rp.costSplit'),
            zeroCost ? (
              <span><i className="ln is-thin" />{t('rp.cumulative')}</span>
            ) : (
              <>
                <span><i className="sw" style={{ background: 'var(--muted)' }} />{t('rp.modelCost')}</span>
                <span><i className="sw" style={{ background: 'var(--stroke)' }} />{t('rp.costMeta')}</span>
                <span><i className="sw" style={{ background: 'var(--text)' }} />{t('rp.costPrompt')}</span>
                <span><i className="ln is-thin" />{t('rp.cumulative')}</span>
              </>
            ),
            H.cost,
            <>
              {grid(yCost, (v) => (zeroCost ? runtimeText(v) : money(v)), 2)}
              {shown.map((r) => {
                if (zeroCost) {
                  return r.computeTime === null ? (
                    <text key={r.generation} x={x(r.generation)} y={H.cost - 8} textAnchor="middle" className="axis sig">?</text>
                  ) : (
                    <rect key={r.generation} x={x(r.generation) - bw / 2} y={yCost(r.computeTime)} width={bw} height={Math.max(0, yCost(0) - yCost(r.computeTime))} fill="var(--muted)" />
                  )
                }
                if (r.modelCost === null || r.metaCost === null || r.promptCost === null) {
                  return <text key={r.generation} x={x(r.generation)} y={H.cost - 8} textAnchor="middle" className="axis sig">?</text>
                }
                let base = 0
                return (
                  <g key={r.generation}>
                    {([[r.modelCost, 'var(--muted)'], [r.metaCost, 'var(--stroke)'], [r.promptCost, 'var(--text)']] as const).map(([v, fill], k) => {
                      if (!v) return null
                      const top = base + v
                      const rect = <rect key={k} x={x(r.generation) - bw / 2} y={yCost(top)} width={bw} height={Math.max(1, yCost(base) - yCost(top) - 1)} fill={fill} />
                      base = top
                      return rect
                    })}
                  </g>
                )
              })}
              <path
                d={line<[number, number]>().x(([g]) => x(g)).y(([, v]) => yCum(v))(cumulativePairs) ?? ''}
                fill="none"
                stroke="var(--stroke)"
                strokeWidth={1}
                strokeDasharray="1.5 2"
              />
              {cumulativeEnd !== null && (
                <text x={width - PROGRESS_M.r + 10} y={yCum(cumulativeEnd) + 3.5} className="axis">
                  {t('rp.cumulative')} {zeroCost ? runtimeText(cumulativeEnd) : money(cumulativeEnd)}
                </text>
              )}
            </>,
          )}
          {panel(
            'failed',
            t('rp.failures'),
            <span><Icon name="cross" />{t('rp.oneCross')}</span>,
            H.failed,
            <>
              {grid(yFailed, (v) => int(v), 2)}
              {shown.map((r) =>
                r.failedIds.map((id, k) => (
                  <path
                    key={id}
                    d={`M${x(r.generation) - crossHalf},${yFailed(k + 0.5) - crossHalf}l${2 * crossHalf},${2 * crossHalf}m0,${-2 * crossHalf}l${-2 * crossHalf},${2 * crossHalf}`}
                    stroke="var(--signal-graphic)"
                    strokeWidth={crossHalf < 2 ? 1 : 1.3}
                  />
                )),
              )}
            </>,
          )}
          {panel(
            'programs',
            t('rp.throughput'),
            null,
            H.programs,
            <>
              {grid(yPrograms, (v) => int(v), 2)}
              {shown.map((r) => (r.programs ? <rect key={r.generation} x={x(r.generation) - bw / 2} y={yPrograms(r.programs)} width={bw} height={Math.max(0, yPrograms(0) - yPrograms(r.programs))} fill="var(--stroke)" /> : null))}
            </>,
          )}
          {posteriors.models.length > 0 &&
            panel(
              'posterior',
              t('rp.posteriorPanel'),
              <>
                {posteriors.models.map((m, k) => (
                  <span key={m} translate="no">
                    <svg width="16" height="6" aria-hidden="true">
                      <line x1={0} x2={16} y1={3} y2={3} stroke={POST_INK[k % 3][0]} strokeWidth={POST_INK[k % 3][1]} strokeDasharray={DASHES[Math.floor(k / 3) % DASHES.length]} />
                    </svg>
                    {m}
                  </span>
                ))}
                {even !== null && <span><i className="ln is-dash" />{t('rp.evenSplit', { v: fmt(even) })}</span>}
              </>,
              H.posterior,
              <>
                {grid(yPost, (v) => fmt(v), 2)}
                {even !== null && <line x1={PROGRESS_M.l} x2={width - PROGRESS_M.r} y1={yPost(even)} y2={yPost(even)} stroke="var(--muted)" strokeDasharray="4 3" />}
                {posteriors.models.map((m, k) => (
                  <path
                    key={m}
                    d={line<PosteriorRow>().defined((r) => r.means[m] !== null && r.means[m] !== undefined && r.generation <= upTo).x((r) => x(r.generation)).y((r) => yPost(r.means[m] as number))(posteriors.rows) ?? ''}
                    fill="none"
                    stroke={POST_INK[k % 3][0]}
                    strokeWidth={POST_INK[k % 3][1]}
                    strokeDasharray={DASHES[Math.floor(k / 3) % DASHES.length]}
                  />
                ))}
              </>,
            )}
          {active && reading && (
            <div className="lanes-tip ptip" style={{ left: x(active.g) + (x(active.g) > width / 2 ? -14 : 14), top: Math.max(4, active.y - 12), transform: x(active.g) > width / 2 ? 'translateX(-100%)' : undefined }}>
              {readingText.map((line, k) => (
                <span key={k} className={k === 0 ? undefined : 'mono'}>{line}</span>
              ))}
              <span className="faint">{reading.bestId || reading.failedIds.length ? t('rp.openBest') : t('rp.noValid')}</span>
            </div>
          )}
        </div>
        <p id="progress-help" className="sr-only">{t('rp.help')}</p>
        <p className="sr-only" aria-live="polite">{readingText.join(' · ')}</p>
        <div className="pscrub">
          <Scrubber max={scrub.max} value={scrub.value} onChange={scrub.onChange} count={scrub.count} left={PROGRESS_M.l} right={PROGRESS_M.r} />
        </div>
        <ModelTable models={models} runBest={runBest} hasPosteriors={posteriors.models.length > 0} />
        <Disclosure title={t('rp.metricsHead')}>
          <MetricsSection programs={programs} upTo={upTo} selected={selected} onSelect={onSelect} />
        </Disclosure>
        <Disclosure title={t('rp.spendHead')}>
          <SpendSection programs={programs} prompts={prompts} stats={stats} zeroCost={zeroCost} scoreRange={scoreRange} />
        </Disclosure>
        <Disclosure title={t('rp.runtimeHead')}>
          <RuntimeSection programs={programs} rows={rows} selected={selected} onSelect={onSelect} scoreRange={scoreRange} />
        </Disclosure>
        <Disclosure title={t('rp.diagnosticsHead')}>
          <DiagnosticsSection programs={programs} onSelect={onSelect} />
        </Disclosure>
      </div>
    </div>
  )
}

type PosteriorRow = { generation: number; means: Record<string, number | null> }

/** R01-Q1/Q3, L10-Q1: the run's cost by part, under the cost panel's title (unknown parts as —). */
function CostLine({ totals }: { totals: ReturnType<typeof costTotals> }) {
  const { t } = useCopy()
  const m = (v: number | null) => (v === null ? '—' : money(v))
  return (
    <>
      <span>
        {t('rp.costTotal', { v: m(totals.total) })}
      </span>
      <span>
        {t('rp.modelCost')} {t('rp.modelCostParts', { v: m(totals.api === null || totals.embed === null || totals.novelty === null ? null : totals.api + totals.embed + totals.novelty), api: m(totals.api), embed: m(totals.embed), novelty: m(totals.novelty) })}
      </span>
      <span>{t('rp.costMeta')} {m(totals.meta)}</span>
      <span>{t('rp.costPrompt')} {m(totals.prompt)}</span>
      {totals.unattributed !== null && totals.unattributed > 0 && <span>{t('rp.unattributed', { v: money(totals.unattributed) })}</span>}
      <span>{t('rp.perProgram', { v: m(totals.perProgram) })}</span>
    </>
  )
}

/** OD-4: each model's programs, success rate, best (◆ marks the run's best), mean, cost and latest posterior. */
function ModelTable({ models, runBest, hasPosteriors }: { models: ReturnType<typeof modelRows>; runBest: number | null; hasPosteriors: boolean }) {
  const { t } = useCopy()
  return (
    <section className="models" aria-labelledby="models-head">
      <h3 id="models-head">{t('rp.models')}</h3>
      {models.length === 0 ? (
        <p className="small muted">{t('rp.noModels')}</p>
      ) : (
        <div className="tscroll">
          <table className="itable mtable">
            <thead>
              <tr>
                <th>{t('rp.model')}</th>
                <th className="num">{t('rp.programs')}</th>
                <th className="num">{t('rp.success')}</th>
                <th className="num">{t('ra.best')}</th>
                <th className="num">{t('rp.mean')}</th>
                <th className="num">{t('rp.cost')}</th>
                <th className="num">{t('rp.posteriorLatest')}</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.model}>
                  <td className="wrap"><span className="mono" translate="no">{m.model}</span></td>
                  <td className="num mono">{int(m.programs)}</td>
                  <td className="num mono">{pct(m.success, 1)}</td>
                  <td className="num mono">
                    {m.best !== null && m.best === runBest && <i className="leadmark" aria-label={t('rp.runBest')}>◆</i>}
                    {fmt(m.best)}
                  </td>
                  <td className="num mono">{fmt(m.mean)}</td>
                  <td className="num mono">{m.cost === null ? '—' : money(m.cost)}</td>
                  <td className="num mono">{fmt(m.posteriorLatest)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {!hasPosteriors && <p className="small muted mnote">{t('rp.posteriorNA')}</p>}
    </section>
  )
}

/** A section below the table that opens on request; its charts are drawn only while it is open. */
function Disclosure({ title, children }: { title: string; children: ReactNode }) {
  const { t } = useCopy()
  const [open, setOpen] = useState(false)
  return (
    <details className="pdisc" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>
        <Icon name="chevron" />
        <b>{title}</b>
        <span className="faint">{t(open ? 'rp.collapse' : 'rp.expand')}</span>
      </summary>
      {open && <div className="pdisc-body">{children}</div>}
    </details>
  )
}
