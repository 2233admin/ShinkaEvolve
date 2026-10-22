import { hierarchy, select, tree, zoom, zoomIdentity, type ZoomBehavior, type ZoomTransform } from 'd3'
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { useWidth } from '../design/useWidth'
import { fmt, int } from '../format'
import { useCopy } from '../i18n/lang'
import type { Prompt } from './data'
import { markPath } from './Lineage'

type Node = { prompt: Prompt | null; k: number; children: Node[] }

const ROW = 40
const COL = 112
const PAD = { l: 18, t: 22, b: 18 }
// The restored prompt tree zoomed between 0.3 and 4 times (VT:17734-17736).
const ZOOM: [number, number] = [0.3, 4]

/** A prompt version's mark at the prompt tree's size: the patch-type shapes of the program lanes, larger. */
function PromptMark({ type, fill, stroke }: { type: string | null; fill: string; stroke: string }) {
  const d = markPath(type, 0, 0)
  return d ? <path d={d} transform="scale(1.6)" style={{ fill }} stroke={stroke} strokeWidth={0.8} /> : <circle r={5.2} style={{ fill }} stroke={stroke} strokeWidth={1.2} />
}

type Props = {
  /** Prompt versions in the order they were made; P{k} is a version's place in it. */
  prompts: Prompt[]
  best: ReadonlySet<string>
  open: string | null
  onOpen: (id: string) => void
}

/**
 * L02-Q1, L02-C1, L02-C3: how the system prompts evolved, as the restored page drew it — a
 * parent→child tree with the patch type as the shape, fitness as the fill and the best version
 * marked; zoom and pan with the wheel and drag, the buttons, or + − 0 on the keyboard.
 */
export function PromptTree({ prompts, best, open, onOpen }: Props) {
  const { t } = useCopy()
  const [box, width] = useWidth<HTMLDivElement>()
  const svgRef = useRef<SVGSVGElement>(null)
  const zoomRef = useRef<ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const [view, setView] = useState<ZoomTransform>(zoomIdentity)

  const layout = useMemo(() => {
    const ids = new Set(prompts.map((q) => q.id))
    const nodes = new Map(prompts.map((q, k): [string, Node] => [q.id, { prompt: q, k, children: [] }]))
    const root: Node = { prompt: null, k: -1, children: [] }
    for (const q of prompts) {
      const parent = q.parent_id && ids.has(q.parent_id) && q.parent_id !== q.id ? nodes.get(q.parent_id)! : root
      parent.children.push(nodes.get(q.id)!)
    }
    const h = tree<Node>().nodeSize([ROW, COL])(hierarchy(root, (n) => n.children))
    const shown = h.descendants().filter((d) => d.data.prompt !== null)
    const xs = shown.map((d) => d.x)
    const top = Math.min(0, ...xs)
    return {
      nodes: shown.map((d) => ({ node: d.data, x: PAD.l + (d.depth - 1) * COL, y: PAD.t + d.x - top })),
      links: shown.filter((d) => d.parent && d.parent.data.prompt !== null).map((d) => ({ from: d.parent!, to: d })),
      top,
      height: PAD.t + Math.max(...xs, 0) - top + PAD.b,
    }
  }, [prompts])
  const fitness = prompts.map((q) => q.fitness).filter((f): f is number => typeof f === 'number')
  const lo = Math.min(...fitness)
  const hi = Math.max(...fitness)
  const fill = (f: number | null | undefined) =>
    typeof f !== 'number' ? 'var(--bg)' : `color-mix(in srgb, var(--ramp-1) ${hi > lo ? Math.round(((f - lo) / (hi - lo)) * 100) : 100}%, var(--ramp-0))`
  const height = Math.max(140, Math.min(360, layout.height))
  const pos = (n: { x: number; depth: number }) => ({ x: PAD.l + (n.depth - 1) * COL, y: PAD.t + n.x - layout.top })

  useEffect(() => {
    const svg = svgRef.current
    if (!svg || width === 0) return
    const behavior = zoom<SVGSVGElement, unknown>()
      .scaleExtent(ZOOM)
      .on('zoom', (event: { transform: ZoomTransform }) => setView(event.transform))
    zoomRef.current = behavior
    select(svg).call(behavior).on('dblclick.zoom', null)
    return () => {
      select(svg).on('.zoom', null)
    }
  }, [width])
  const zoomBy = (k: number) => svgRef.current && zoomRef.current?.scaleBy(select(svgRef.current), k)
  const reset = () => svgRef.current && zoomRef.current?.transform(select(svgRef.current), zoomIdentity)
  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === '+' || event.key === '=') zoomBy(1.4)
    else if (event.key === '-') zoomBy(1 / 1.4)
    else if (event.key === '0') reset()
  }

  return (
    <div className="ptree" ref={box} onKeyDown={onKey}>
      <div className="ptree-tools">
        <span className="legend">
          <span className="faint">{t('ra.fitness')}</span>
          <span className="ramp" aria-hidden="true">
            {[0, 25, 50, 75, 100].map((k) => (
              <i key={k} style={{ background: `color-mix(in srgb, var(--ramp-1) ${k}%, var(--ramp-0))` }} />
            ))}
          </span>
          {fitness.length > 0 && <span className="mono faint">{fmt(lo)}–{fmt(hi)}</span>}
          <span><i className="ring" />{t('ra.bestPrompt')}</span>
        </span>
        <span className="seg is-small" role="group" aria-label={t('ra.zoom')}>
          <button type="button" aria-label={t('ra.zoomOut')} title={t('ra.zoomOut')} disabled={view.k <= ZOOM[0]} onClick={() => zoomBy(1 / 1.4)}>−</button>
          <button type="button" aria-label={t('ra.zoomIn')} title={t('ra.zoomIn')} disabled={view.k >= ZOOM[1]} onClick={() => zoomBy(1.4)}>+</button>
          <button type="button" disabled={view.k === 1 && view.x === 0 && view.y === 0} onClick={reset}>{t('ra.zoomReset')}</button>
        </span>
      </div>
      {width > 0 && (
        <svg ref={svgRef} className="chart ptree-svg" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="group" aria-label={t('ra.promptTree', { n: prompts.length })}>
          <g transform={view.toString()}>
            {layout.links.map(({ from, to }) => {
              const a = pos(from)
              const b = pos(to)
              const mid = (a.x + b.x) / 2
              return <path key={to.data.prompt!.id} d={`M${a.x},${a.y}C${mid},${a.y} ${mid},${b.y} ${b.x},${b.y}`} fill="none" stroke="var(--stroke)" strokeWidth={1.2} />
            })}
            {layout.nodes.map(({ node, x, y }) => {
              const q = node.prompt!
              const isBest = best.has(q.id)
              const label = t('ra.promptNode', {
                k: node.k,
                g: q.generation ?? '—',
                type: q.patch_type ?? '—',
                fitness: typeof q.fitness === 'number' ? fmt(q.fitness, 3) : '—',
                n: int(q.program_count ?? 0),
              })
              const choose = () => onOpen(q.id)
              return (
                <g
                  key={q.id}
                  className="mark"
                  transform={`translate(${x},${y})`}
                  role="button"
                  tabIndex={0}
                  aria-label={isBest ? `${label} · ${t('ra.bestPrompt')}` : label}
                  aria-pressed={q.id === open}
                  onClick={choose}
                  onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), choose())}
                >
                  <title>{label}</title>
                  <rect x={-10} y={-12} width={COL - 8} height={26} fill="transparent" />
                  {q.id === open && <rect x={-9} y={-9} width={18} height={18} rx={3} fill="none" stroke="var(--text)" strokeWidth={1.5} />}
                  {isBest && <circle r={10.5} fill="none" stroke="var(--signal-graphic)" strokeWidth={1.6} />}
                  <PromptMark type={q.patch_type ?? null} fill={fill(q.fitness)} stroke="var(--text)" />
                  <text x={14} y={-1} className={isBest ? 'lbl sig' : 'lbl'}>P{node.k}</text>
                  <text x={14} y={11} className="axis">
                    G{q.generation ?? '—'} {q.patch_type ?? '—'} · {typeof q.fitness === 'number' ? fmt(q.fitness, 3) : '—'} ({int(q.program_count ?? 0)})
                  </text>
                </g>
              )
            })}
          </g>
        </svg>
      )}
    </div>
  )
}
