import { Delaunay, scaleLinear, select, tickStep, ticks, zoom, zoomIdentity, type ZoomBehavior, type ZoomTransform } from 'd3'
import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent, type PointerEvent, type ReactNode } from 'react'
import { scoreOf } from '../compare/series'
import { Icon } from '../design/icons'
import { useSize } from '../design/useWidth'
import { fmt, int } from '../format'
import type { CopyKey } from '../i18n/copy'
import { useCopy } from '../i18n/lang'
import { bestOf, bestSoFar, fitLabels, generationOf, islandOf, lanesOf, lineOf, modelOf, nameOf, patchTypeOf, promptOrder, rampDomain, type MetaUpdate, type Prompt, type RunProgram } from './data'

// The global conclusions hold the meta-analysis and the prompt versions (design.md › 四个分析视图, OD-3).
export const PANEL_SECTIONS = ['conclusions', 'islands', 'path', 'offspring'] as const
export type PanelSection = (typeof PANEL_SECTIONS)[number]
export const PANEL_LABEL: Record<PanelSection, CopyKey> = {
  conclusions: 'ra.panel.conclusions',
  islands: 'ra.panel.islands',
  path: 'ra.panel.path',
  offspring: 'ra.panel.offspring',
}

// The approved lane geometry (hifi/build.mjs › lineageView): two tracks above the lanes, the island
// labels and score ticks in a 128px gutter, and a failure band at the bottom of every lane. Folding
// the island list (L01-C6) keeps only the island names and gives the axis the rest of the gutter.
const TOP = 64
const M = { l: 128, r: 28 }
const GUTTER_FOLDED = 72
const MIN_LANE = 112
const ZOOM_MAX = 8
// The width of one character of a 10.5px mono axis label.
const LABEL_CHAR = 6.4

/** A patch type's mark, as the restored tree drew them (L01-Q3): init ◆, full ●, diff ■, cross ✚, fix ▲. */
export function markPath(type: string | null, x: number, y: number): string | null {
  switch (type) {
    case 'init':
      return `M${x},${y - 4}L${x + 4},${y}L${x},${y + 4}L${x - 4},${y}Z`
    case 'diff':
      return `M${x - 2.8},${y - 2.8}h5.6v5.6h-5.6Z`
    case 'cross':
      return `M${x - 1.1},${y - 4}h2.2v2.9h2.9v2.2h-2.9v2.9h-2.2v-2.9h-2.9v-2.2h2.9Z`
    case 'fix':
      return `M${x},${y - 4}L${x + 3.8},${y + 3}H${x - 3.8}Z`
    default:
      return null
  }
}

/** A score's place on the ramp, in the theme's own ramp colours. */
const rampFill = (domain: [number, number] | null, s: number): string => {
  if (domain === null) return 'var(--ramp-1)'
  const [lo, hi] = domain
  const k = hi > lo ? Math.max(0, Math.min(1, (s - lo) / (hi - lo))) : 1
  return `color-mix(in srgb, var(--ramp-1) ${Math.round(k * 100)}%, var(--ramp-0))`
}

function Mark({ type, x, y, fill }: { type: string | null; x: number; y: number; fill: string }) {
  const d = markPath(type, x, y)
  return d ? <path d={d} style={{ fill }} /> : <circle cx={x} cy={y} r={3.2} style={{ fill }} />
}

type Point = { p: RunProgram; lane: number; x: number; y: number }

type Props = {
  programs: RunProgram[]
  byId: ReadonlyMap<string, RunProgram>
  /** The last generation shown (the scrubber). */
  upTo: number
  maxGen: number
  /** The program in the inspector. */
  selected: RunProgram | null
  onSelect: (id: string) => void
  prompts: Prompt[]
  promptsError: string | null
  hasPromptEvo: boolean
  bestPrompts: ReadonlySet<string>
  updates: MetaUpdate[]
  metaError: string | null
  /** The open panel section, if any. */
  panel: PanelSection | null
  onPanel: (section: PanelSection, extra?: { prompt?: string; meta?: number }) => void
  /** The open lineage panel, laid over the lanes below the tracks. */
  panelNode: ReactNode
  scrubber: ReactNode
}

/**
 * Lineage (6.2, L01, L07): every program in its island's lane, generations left to right, score up
 * within the lane and failures in a band below it. Same-island parent links are drawn; the best
 * program's ancestry is the signal line and the selected program's ancestry is drawn across lanes.
 */
export function Lineage({ programs, byId, upTo, maxGen, selected, onSelect, prompts, promptsError, hasPromptEvo, bestPrompts, updates, metaError, panel, onPanel, panelNode, scrubber }: Props) {
  const { t } = useCopy()
  const [box, size] = useSize<HTMLDivElement>()
  const svgRef = useRef<SVGSVGElement>(null)
  const zoomRef = useRef<ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const [view, setView] = useState<ZoomTransform>(zoomIdentity)
  const [active, setActive] = useState<string | null>(null)
  const [range, setRange] = useState({ lo: '', hi: '' })
  // The legend folds away to give the lanes its rows (L07-C2).
  const [legendOpen, setLegendOpen] = useState(true)
  const [islandsOpen, setIslandsOpen] = useState(true)

  const lanes = useMemo(() => lanesOf(programs), [programs])
  // The lane of programs without an island is not an island.
  const islandCount = lanes.filter((lane) => lane !== null).length
  const visible = useMemo(() => programs.filter((p) => generationOf(p) <= upTo), [programs, upTo])
  const best = useMemo(() => bestOf(visible), [visible])
  const ramp = useMemo(() => rampDomain(programs), [programs])
  const auto = useMemo((): [number, number] | null => {
    const scores = programs.map(scoreOf).filter((s): s is number => s !== null)
    if (scores.length === 0) return null
    const lo = Math.min(...scores)
    const hi = Math.max(...scores)
    if (lo === hi) return [lo - 0.5, hi + 0.5]
    // The axis bottom sits on a labelled tick; the top stays at the data (design.md › 图表语法).
    const step = tickStep(lo, hi, 2)
    return [Math.floor(lo / step) * step, hi]
  }, [programs])
  const domain = useMemo((): [number, number] | null => {
    if (auto === null) return null
    const lo = range.lo.trim() === '' ? auto[0] : Number(range.lo)
    const hi = range.hi.trim() === '' ? auto[1] : Number(range.hi)
    return Number.isFinite(lo) && Number.isFinite(hi) && hi > lo ? [lo, hi] : auto
  }, [auto, range])

  const { width, height: boxHeight } = size
  const laneH = Math.max(MIN_LANE, (boxHeight - TOP) / Math.max(1, lanes.length))
  const height = TOP + laneH * lanes.length
  const ml = islandsOpen ? M.l : GUTTER_FOLDED
  const x = scaleLinear([0, Math.max(1, maxGen)], [ml, Math.max(ml + 1, width - M.r)])
  const sy = scaleLinear(domain ?? [0, 1], [1, 0])
  const laneTop = (k: number) => TOP + k * laneH
  const yIn = (k: number, s: number) => laneTop(k) + 34 + sy(s) * (laneH - 70)
  const failY = (k: number) => laneTop(k) + laneH - 14
  const zx = (g: number) => view.applyX(x(g))
  const zy = (y: number) => view.applyY(y)
  const laneIndex = useMemo(() => new Map(lanes.map((lane, k) => [lane, k])), [lanes])

  // Every shown program's position on screen; scores outside an overridden score axis are left out.
  const { points, outside } = useMemo(() => {
    const out: Point[] = []
    let hidden = 0
    for (const p of visible) {
      const lane = laneIndex.get(islandOf(p)) ?? 0
      const s = scoreOf(p)
      if (s !== null && domain !== null && (s < domain[0] || s > domain[1])) {
        hidden += 1
        continue
      }
      out.push({ p, lane, x: zx(generationOf(p)), y: zy(s === null ? failY(lane) : yIn(lane, s)) })
    }
    return { points: out, outside: hidden }
    // The scales are rebuilt from these inputs on every render.
  }, [visible, laneIndex, domain, width, boxHeight, view, maxGen, lanes.length, islandsOpen])
  const at = useMemo(() => new Map(points.map((pt) => [pt.p.id, pt])), [points])
  const finder = useMemo(() => Delaunay.from(points, (pt) => pt.x, (pt) => pt.y), [points])
  const byLane = useMemo(() => {
    const m = new Map<number, Point[]>()
    for (const pt of points) m.set(pt.lane, [...(m.get(pt.lane) ?? []), pt])
    for (const list of m.values()) list.sort((a, b) => generationOf(a.p) - generationOf(b.p) || (a.p.timestamp ?? 0) - (b.p.timestamp ?? 0))
    return m
  }, [points])

  useEffect(() => {
    const svg = svgRef.current
    if (!svg || width === 0) return
    const behavior = zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, ZOOM_MAX])
      .extent([[0, 0], [width, height]])
      .translateExtent([[0, 0], [width, height]])
      .on('zoom', (event: { transform: ZoomTransform }) => setView(event.transform))
    zoomRef.current = behavior
    select(svg).call(behavior).on('dblclick.zoom', null)
    return () => {
      select(svg).on('.zoom', null)
    }
  }, [width, height])
  const zoomBy = (k: number) => {
    if (svgRef.current && zoomRef.current) zoomRef.current.scaleBy(select(svgRef.current), k)
  }
  const resetZoom = () => {
    if (svgRef.current && zoomRef.current) zoomRef.current.transform(select(svgRef.current), zoomIdentity)
  }

  const bestIds = new Set(best.map((p) => p.id))
  const chains = best.map((b) => lineOf(byId, b.id).filter((p) => at.has(p.id)))
  const chainIds = new Set(chains.flat().map((p) => p.id))
  const selectedLine = selected && at.has(selected.id) ? lineOf(byId, selected.id).filter((p) => at.has(p.id)) : []
  const pathOf = (line: RunProgram[]) => line.map((p, i) => `${i ? 'L' : 'M'}${at.get(p.id)!.x.toFixed(1)},${at.get(p.id)!.y.toFixed(1)}`).join('')

  // Same-island parent links only; a migration would cross every lane (it is drawn for the selection).
  let edges = ''
  for (const pt of points) {
    const parent = pt.p.parent_id ? at.get(pt.p.parent_id) : undefined
    if (parent && parent.lane === pt.lane) edges += `M${parent.x.toFixed(1)},${parent.y.toFixed(1)}L${pt.x.toFixed(1)},${pt.y.toFixed(1)}`
  }
  const climbs = lanes.map((lane, k) =>
    bestSoFar(visible.filter((p) => islandOf(p) === lane))
      .map((s, i, all) => {
        const y = zy(yIn(k, s.best))
        const next = i + 1 < all.length ? all[i + 1].generation : Math.min(upTo, maxGen)
        return `M${zx(s.generation).toFixed(1)},${y.toFixed(1)}H${zx(next).toFixed(1)}`
      })
      .join(''),
  )

  const describe = (p: RunProgram) => {
    const s = scoreOf(p)
    const island = islandOf(p)
    return t('ra.pointLine', {
      name: nameOf(p),
      id: p.id.slice(0, 8),
      g: generationOf(p),
      island: island === null ? t('ra.noIsland') : t('ra.island', { k: island }),
      score: s === null ? t('ra.failed') : fmt(s, 4),
      type: patchTypeOf(p) ?? '—',
      model: modelOf(p) ?? '—',
    })
  }
  const activePoint = active ? (at.get(active) ?? null) : null

  const hit = (event: PointerEvent<SVGRectElement> | MouseEvent<SVGRectElement>): string | null => {
    if (points.length === 0) return null
    const bounds = event.currentTarget.ownerSVGElement!.getBoundingClientRect()
    const px = event.clientX - bounds.left
    const py = event.clientY - bounds.top
    const pt = points[finder.find(px, py)]
    return Math.hypot(pt.x - px, pt.y - py) <= 12 ? pt.p.id : null
  }
  const start = () => (selected && at.has(selected.id) ? selected.id : (best[0]?.id ?? points[0]?.p.id ?? null))
  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === '+' || event.key === '=') return zoomBy(1.5)
    if (event.key === '-') return zoomBy(1 / 1.5)
    if (event.key === '0') return resetZoom()
    if (event.key === 'Escape') return setActive(null)
    const from = active ?? start()
    if (from === null) return
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      onSelect(from)
      return
    }
    const pt = at.get(from)
    if (!pt) return setActive(start())
    const list = byLane.get(pt.lane) ?? []
    const i = list.findIndex((q) => q.p.id === from)
    let next: Point | undefined
    if (event.key === 'ArrowRight') next = list[Math.min(list.length - 1, i + 1)]
    else if (event.key === 'ArrowLeft') next = list[Math.max(0, i - 1)]
    else if (event.key === 'Home') next = list[0]
    else if (event.key === 'End') next = list[list.length - 1]
    else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      const step = event.key === 'ArrowDown' ? 1 : -1
      for (let k = pt.lane + step; k >= 0 && k < lanes.length && !next; k += step) {
        const other = byLane.get(k) ?? []
        next = other.reduce<Point | undefined>((a, q) => (a === undefined || Math.abs(generationOf(q.p) - generationOf(pt.p)) < Math.abs(generationOf(a.p) - generationOf(pt.p)) ? q : a), undefined)
      }
    } else return
    event.preventDefault()
    if (next) setActive(next.p.id)
  }

  const orderedPrompts = promptOrder(prompts)
  const latestUpdate = updates.length - 1
  // Track labels that would cover a mark or another label give way; the best prompt and the latest update are placed first.
  const promptText = (q: Prompt, k: number) => `P${k} ${typeof q.fitness === 'number' ? fmt(q.fitness) : '—'}`
  const metaText = (u: MetaUpdate, k: number) => `#${k + 1} · ${int(u.processed)}`
  const promptGen = (q: Prompt) => (typeof q.program_generation === 'number' ? q.program_generation : 0)
  const promptLabels = fitLabels(
    orderedPrompts.map((q, k) => ({ x: zx(promptGen(q)), w: promptText(q, k).length * LABEL_CHAR, keep: bestPrompts.has(q.id), off: promptGen(q) > upTo })),
    ml - 2,
    width - 4,
  )
  const metaLabels = fitLabels(
    updates.map((u, k) => ({ x: zx(u.generation ?? 0), w: metaText(u, k).length * LABEL_CHAR, keep: k === latestUpdate, off: u.generation === null || u.generation > upTo })),
    ml - 2,
    width - 4,
  )
  // The best label reads towards the middle of the plot so that it never runs off either edge.
  const midX = (ml + width - M.r) / 2
  const tied = best.length > 1
  const head = best[0] ? at.get(best[0].id) : undefined
  const failedIn = (lane: number | null) => visible.filter((p) => islandOf(p) === lane && scoreOf(p) === null).length
  const countIn = (lane: number | null) => visible.filter((p) => islandOf(p) === lane).length
  const laneTicks = domain ? ticks(domain[0], domain[1], 2) : []
  const plotW = Math.max(0, width - ml - M.r)
  const labelX = islandsOpen ? 16 : 10
  const failedTotal = visible.filter((p) => scoreOf(p) === null).length

  return (
    <div className="lineage">
      <div className="vbar">
        <button type="button" className="toggle" aria-expanded={legendOpen} aria-controls="lanes-legend lanes-shapes" onClick={() => setLegendOpen((open) => !open)}>
          {t('ra.legend')}
        </button>
        <button type="button" className="toggle" aria-expanded={islandsOpen} aria-controls="lanes-islands" onClick={() => setIslandsOpen((open) => !open)}>
          {t('ra.islandList', { n: islandCount })}
        </button>
        {legendOpen && (
          <div className="legend" id="lanes-legend">
            <span>
              <span className="faint">{t('ra.scoreRamp')}</span>
              <span className="ramp" aria-hidden="true">
                {[0, 25, 50, 75, 100].map((k) => (
                  <i key={k} style={{ background: `color-mix(in srgb, var(--ramp-1) ${k}%, var(--ramp-0))` }} />
                ))}
              </span>
              {ramp && (
                <span className="mono faint">
                  <span className="op">≤</span>
                  {fmt(ramp[0])}–{fmt(ramp[1])}
                </span>
              )}
            </span>
            <span><i className="ln is-signal" />{t('ra.bestPath')}</span>
            <span><i className="ln is-text" />{t('ra.selectedLine')}</span>
            <span><Icon name="cross" />{t('ra.failedMark')}</span>
            <span><i className="ln is-grid" />{t('ra.sameIsland')}</span>
            <span><i className="ln is-thin" />{t('ra.laneBest')}</span>
          </div>
        )}
      </div>
      <div className="vbar is-sub">
        {legendOpen && (
          <span className="legend shapes" id="lanes-shapes" translate="no">
            <span className="faint">{t('ra.patch')}</span>
            {(['init', 'full', 'diff', 'cross', 'fix'] as const).map((type) => (
              <span key={type}>
                <svg width="10" height="10" viewBox="-5 -5 10 10" aria-hidden="true">
                  <Mark type={type} x={0} y={0} fill="var(--muted)" />
                </svg>
                {type}
              </span>
            ))}
          </span>
        )}
        {best.length === 0 && visible.length > 0 && <span className="small faint">{t('ra.noBest')}</span>}
        {outside > 0 && <span className="small sig">{t('ra.outside', { n: int(outside) })}</span>}
        <span className="vtools">
          <span className="seg is-small" role="group" aria-label={t('ra.panels')}>
            {PANEL_SECTIONS.map((s) => (
              <button key={s} type="button" aria-pressed={panel === s} onClick={() => onPanel(s)}>
                {t(PANEL_LABEL[s])}
              </button>
            ))}
          </span>
          <label className="range">
            <span className="faint">{t('ra.scoreAxis')}</span>
            <input type="number" step="any" inputMode="decimal" value={range.lo} placeholder={auto ? fmt(auto[0]) : ''} aria-label={t('ra.scoreMin')} onChange={(e) => setRange((r) => ({ ...r, lo: e.target.value }))} />
            <span className="faint">–</span>
            <input type="number" step="any" inputMode="decimal" value={range.hi} placeholder={auto ? fmt(auto[1]) : ''} aria-label={t('ra.scoreMax')} onChange={(e) => setRange((r) => ({ ...r, hi: e.target.value }))} />
          </label>
          {(range.lo !== '' || range.hi !== '') && (
            <button type="button" className="btn is-small" onClick={() => setRange({ lo: '', hi: '' })}>{t('ra.scoreAuto')}</button>
          )}
          <span className="seg is-small" role="group" aria-label={t('ra.zoom')}>
            <button type="button" aria-label={t('ra.zoomOut')} title={t('ra.zoomOut')} disabled={view.k <= 1} onClick={() => zoomBy(1 / 1.5)}>−</button>
            <button type="button" aria-label={t('ra.zoomIn')} title={t('ra.zoomIn')} disabled={view.k >= ZOOM_MAX} onClick={() => zoomBy(1.5)}>+</button>
            <button type="button" disabled={view.k === 1 && view.x === 0 && view.y === 0} onClick={resetZoom}>{t('ra.zoomReset')}</button>
          </span>
        </span>
      </div>
      <div className="lanes-box">
        <div className="lanes-scroll" ref={box}>
        {width > 0 && (
          <div
            className="lanes-plot"
            tabIndex={0}
            role="group"
            aria-label={t('ra.lanesLabel', { n: int(visible.length), f: int(failedTotal), islands: islandCount })}
            aria-describedby="lanes-help"
            onKeyDown={onKey}
            onFocus={(event) => {
              if (event.target === event.currentTarget && event.currentTarget.matches(':focus-visible')) setActive((a) => a ?? start())
            }}
            onBlur={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setActive(null)
            }}
          >
            <svg ref={svgRef} className="chart lanes" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
              <defs>
                <clipPath id="lanes-plot-clip">
                  <rect x={ml - 10} y={TOP} width={plotW + 20} height={height - TOP} />
                </clipPath>
                <clipPath id="lanes-gutter-clip">
                  <rect x={0} y={TOP} width={ml - 2} height={height - TOP} />
                </clipPath>
                <clipPath id="lanes-track-clip">
                  <rect x={ml - 10} y={0} width={plotW + 20 + M.r} height={TOP} />
                </clipPath>
              </defs>
              <g aria-hidden="true">
                <text x={labelX + 4} y={24} className="axis">{t(islandsOpen ? 'ra.promptTrack' : 'ra.promptTrackShort')}</text>
                <text x={labelX + 4} y={50} className="axis">{t(islandsOpen ? 'ra.metaTrack' : 'ra.metaTrackShort')}</text>
                <line x1={ml} x2={width - M.r} y1={20} y2={20} stroke="var(--line)" />
                <line x1={ml} x2={width - M.r} y1={46} y2={46} stroke="var(--line)" />
                {!hasPromptEvo && <text x={ml + 8} y={16} className="axis">{t('ra.noPrompts')}</text>}
                {promptsError !== null && <text x={ml + 8} y={16} className="axis sig">{t('ra.promptsFailed', { error: promptsError })}</text>}
                {metaError !== null ? (
                  <text x={ml + 8} y={42} className="axis sig">{t('ra.metaFailed')} {metaError}</text>
                ) : updates.length === 0 ? (
                  <text x={ml + 8} y={42} className="axis">{t('ra.noMetaTrack')}</text>
                ) : null}
              </g>
              <g id="lanes-islands" clipPath="url(#lanes-gutter-clip)" aria-hidden="true">
                {lanes.map((lane, k) => {
                  const f = failedIn(lane)
                  return (
                    <g key={k}>
                      <text x={labelX} y={zy(laneTop(k) + 26)} className="lane">{lane === null ? t('ra.noIsland') : t('ra.island', { k: lane })}</text>
                      {islandsOpen && <text x={labelX} y={zy(laneTop(k) + 44)} className="axis">{t('ra.programsN', { n: int(countIn(lane)) })}</text>}
                      {islandsOpen && <text x={labelX} y={zy(laneTop(k) + 60)} className={f ? 'axis sig' : 'axis'}>{t('ra.failedN', { n: int(f) })}</text>}
                      {laneTicks.map((v) => (
                        <text key={v} x={ml - 6} y={zy(yIn(k, v)) + 3.5} textAnchor="end" className="axis">{fmt(v)}</text>
                      ))}
                      <text x={ml - 6} y={zy(failY(k)) + 3.5} textAnchor="end" className="axis">{t('ra.failedShort')}</text>
                    </g>
                  )
                })}
              </g>
              <g aria-hidden="true">
                {lanes.map((_, k) => (
                  <line key={k} x1={0} x2={width} y1={Math.max(TOP, zy(laneTop(k)))} y2={Math.max(TOP, zy(laneTop(k)))} stroke="var(--line)" />
                ))}
              </g>
              <g clipPath="url(#lanes-plot-clip)" aria-hidden="true">
                {lanes.map((_, k) => (
                  <line key={k} x1={ml} x2={width - M.r} y1={zy(failY(k) - 8)} y2={zy(failY(k) - 8)} stroke="var(--line)" strokeDasharray="2 3" />
                ))}
                <path d={edges} fill="none" stroke="var(--grid)" strokeWidth={0.8} />
                {climbs.map((d, k) => (d ? <path key={k} d={d} fill="none" stroke="var(--stroke)" strokeWidth={1} /> : null))}
                {points.map((pt) =>
                  scoreOf(pt.p) === null ? (
                    <path key={pt.p.id} d={`M${(pt.x - 3).toFixed(1)},${(pt.y - 3).toFixed(1)}l6,6m0,-6l-6,6`} stroke="var(--signal-graphic)" strokeWidth={1.3} />
                  ) : chainIds.has(pt.p.id) ? null : (
                    <Mark key={pt.p.id} type={patchTypeOf(pt.p)} x={pt.x} y={pt.y} fill={rampFill(ramp, scoreOf(pt.p) as number)} />
                  ),
                )}
                {selectedLine.length > 1 && <path d={pathOf(selectedLine)} fill="none" stroke="var(--text)" strokeWidth={1.4} strokeLinejoin="round" />}
                {selectedLine.slice(0, -1).map((p) => (chainIds.has(p.id) ? null : <circle key={p.id} cx={at.get(p.id)!.x} cy={at.get(p.id)!.y} r={4} fill="none" stroke="var(--text)" strokeWidth={1.2} />))}
                {chains.map((line, i) => (line.length > 1 ? <path key={i} d={pathOf(line)} fill="none" stroke="var(--signal-graphic)" strokeWidth={2} strokeLinejoin="round" /> : null))}
                {[...chainIds].map((id) =>
                  bestIds.has(id) ? null : <circle key={id} cx={at.get(id)!.x} cy={at.get(id)!.y} r={4} fill="var(--bg)" stroke="var(--signal-graphic)" strokeWidth={2} />,
                )}
                {best.map((b) => {
                  const pt = at.get(b.id)
                  return pt ? (
                    <g key={b.id}>
                      <circle cx={pt.x} cy={pt.y} r={5.5} fill="var(--signal-graphic)" />
                      <circle cx={pt.x} cy={pt.y} r={9.5} fill="none" stroke="var(--text)" strokeWidth={1.5} />
                    </g>
                  ) : null
                })}
                {head && (
                  <text x={head.x > midX ? head.x + 10 : head.x - 10} y={head.y - 16} textAnchor={head.x > midX ? 'end' : 'start'} className="lbl sig halo">
                    {t('ra.bestLabel', { score: fmt(scoreOf(best[0])), g: generationOf(best[0]) })}
                    {tied ? ` · ${t('ra.tied', { n: best.length })}` : ''}
                  </text>
                )}
                {selected && at.has(selected.id) && (
                  <rect x={at.get(selected.id)!.x - 8} y={at.get(selected.id)!.y - 8} width={16} height={16} rx={3} fill="none" stroke="var(--text)" strokeWidth={1.5} />
                )}
                {activePoint && <circle cx={activePoint.x} cy={activePoint.y} r={7} fill="none" stroke="var(--text)" strokeWidth={1.5} strokeDasharray="2 2" />}
              </g>
              <rect
                x={ml - 10}
                y={TOP}
                width={plotW + 20}
                height={height - TOP}
                fill="transparent"
                className="lanes-hit"
                onPointerMove={(event) => setActive(hit(event))}
                onPointerLeave={() => setActive(null)}
                onClick={(event) => {
                  const id = hit(event)
                  if (id) onSelect(id)
                }}
              />
              <g clipPath="url(#lanes-track-clip)">
                {orderedPrompts.map((q, k) => {
                  const g = promptGen(q)
                  if (g > upTo) return null
                  const px = zx(g)
                  const isBest = bestPrompts.has(q.id)
                  const label = t('ra.promptMark', { k, fitness: typeof q.fitness === 'number' ? fmt(q.fitness) : '—', g })
                  const open = () => onPanel('conclusions', { prompt: q.id })
                  const side = promptLabels[k]
                  const w = promptText(q, k).length * LABEL_CHAR
                  return (
                    <g key={q.id} className="mark" role="button" tabIndex={0} aria-label={label} onClick={open} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), open())}>
                      <title>{label}</title>
                      <rect x={side === 'left' ? px - 8 - w : px - 7} y={10} width={side ? w + 15 : 14} height={16} fill="transparent" />
                      <path d={`M${px},14l5,8h-10z`} fill={isBest ? 'var(--text)' : 'var(--bg)'} stroke="var(--text)" strokeWidth={1.2} />
                      {side && (
                        <text x={side === 'right' ? px + 8 : px - 8} y={24} textAnchor={side === 'right' ? 'start' : 'end'} className={isBest ? 'axis is-strong' : 'axis'}>
                          {promptText(q, k)}
                        </text>
                      )}
                    </g>
                  )
                })}
                {updates.map((u, k) => {
                  if (u.generation === null || u.generation > upTo) return null
                  const px = zx(u.generation)
                  const label = t('ra.metaMark', { k: k + 1, n: int(u.processed), g: u.generation })
                  const open = () => onPanel('conclusions', { meta: u.processed })
                  const side = metaLabels[k]
                  const w = metaText(u, k).length * LABEL_CHAR
                  return (
                    <g key={u.processed} className="mark" role="button" tabIndex={0} aria-label={label} onClick={open} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), open())}>
                      <title>{label}</title>
                      <rect x={side === 'left' ? px - 8 - w : px - 6} y={36} width={side ? w + 14 : 12} height={16} fill="transparent" />
                      <rect x={px - 4} y={42} width={8} height={8} fill={k === latestUpdate ? 'var(--text)' : 'var(--bg)'} stroke="var(--text)" strokeWidth={1.2} />
                      {side && (
                        <text x={side === 'right' ? px + 8 : px - 8} y={50} textAnchor={side === 'right' ? 'start' : 'end'} className={k === latestUpdate ? 'axis is-strong' : 'axis'}>
                          {metaText(u, k)}
                        </text>
                      )}
                    </g>
                  )
                })}
              </g>
            </svg>
            {activePoint && (
              <div
                className="lanes-tip"
                style={{
                  left: activePoint.x + (activePoint.x > width / 2 ? -14 : 14),
                  top: Math.max(4, activePoint.y - 52),
                  transform: activePoint.x > width / 2 ? 'translateX(-100%)' : undefined,
                }}
              >
                <span className="mono" translate="no">{describe(activePoint.p)}</span>
                <span className="faint">{t('ra.pointOpen')}</span>
              </div>
            )}
          </div>
        )}
        </div>
        {panelNode}
        <p id="lanes-help" className="sr-only">{t('ra.lanesHelp')}</p>
        <p className="sr-only" aria-live="polite">{activePoint ? describe(activePoint.p) : ''}</p>
      </div>
      {scrubber}
    </div>
  )
}
