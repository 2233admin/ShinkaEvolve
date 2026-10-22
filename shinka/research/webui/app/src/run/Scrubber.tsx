import { scaleLinear } from 'd3'
import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { Icon } from '../design/icons'
import { useWidth } from '../design/useWidth'
import { int } from '../format'
import { useCopy } from '../i18n/lang'
import { useMatches } from '../shell/useMatches'

type Props = {
  /** The last generation of the run. */
  max: number
  /** The last generation shown. */
  value: number
  onChange: (generation: number) => void
  /** How many programs are shown up to `value`. */
  count: number
  /** The plot's left and right margins, so the track lines up with the generation axis above it. */
  left: number
  right: number
}

// One generation per step: replay is a discrete, user-started stepping, never an interpolation (design.md › 动效).
const STEP_MS = 450

/**
 * The generation scrubber Lineage and Progress share (OD-8): drag or step one generation; replay runs
 * only when pressed and stops at the last generation. With reduced motion, each press of replay steps
 * one generation.
 */
export function Scrubber({ max, value, onChange, count, left, right }: Props) {
  const { t } = useCopy()
  const [box, width] = useWidth<HTMLDivElement>()
  const reduced = useMatches('(prefers-reduced-motion: reduce)')
  const [playing, setPlaying] = useState(false)
  const latest = useRef({ value, onChange })
  latest.current = { value, onChange }
  const x = scaleLinear([0, Math.max(1, max)], [left, Math.max(left + 1, width - right)])
  const clamp = (g: number) => Math.max(0, Math.min(max, Math.round(g)))

  useEffect(() => {
    if (!playing) return
    const timer = window.setInterval(() => {
      const { value: v, onChange: set } = latest.current
      if (v >= max) setPlaying(false)
      else set(v + 1)
    }, STEP_MS)
    return () => window.clearInterval(timer)
  }, [playing, max])
  useEffect(() => {
    if (value >= max) setPlaying(false)
  }, [value, max])

  const replay = () => {
    if (playing) return setPlaying(false)
    const from = value >= max ? 0 : value
    if (reduced) return onChange(value >= max ? 0 : value + 1)
    onChange(from)
    setPlaying(true)
  }
  const fromPointer = (event: PointerEvent<SVGSVGElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    onChange(clamp(x.invert(event.clientX - bounds.left)))
  }
  const onKey = (event: KeyboardEvent<SVGSVGElement>) => {
    const moves: Record<string, number> = { ArrowRight: value + 1, ArrowUp: value + 1, ArrowLeft: value - 1, ArrowDown: value - 1, PageUp: value + 5, PageDown: value - 5, Home: 0, End: max }
    if (!(event.key in moves)) return
    event.preventDefault()
    setPlaying(false)
    onChange(clamp(moves[event.key]))
  }
  const ticks = x.ticks(Math.max(2, Math.floor((width - left - right) / 60))).filter(Number.isInteger)

  return (
    <div className="scrub" ref={box}>
      <div className="scrubctl">
        <button type="button" className="btn is-small" disabled={value <= 0} onClick={() => onChange(clamp(value - 1))}>
          <Icon name="prev" />
          {t('ra.prevGen')}
        </button>
        <button type="button" className="btn is-small" disabled={value >= max} onClick={() => onChange(clamp(value + 1))}>
          {t('ra.nextGen')}
          <Icon name="next" />
        </button>
        <button type="button" className="btn is-small" aria-pressed={playing} onClick={replay}>
          <Icon name={playing ? 'pause' : 'play'} />
          {playing ? t('ra.stop') : t('ra.replay')}
        </button>
        <span className="mono faint scrub-count" aria-live="polite">{t('ra.scrub', { g: value, n: int(count) })}</span>
      </div>
      {width > 0 && (
        <svg
          width={width}
          height={40}
          viewBox={`0 0 ${width} 40`}
          className="scrub-track"
          role="slider"
          tabIndex={0}
          aria-label={t('ra.genSlider')}
          aria-valuemin={0}
          aria-valuemax={max}
          aria-valuenow={value}
          aria-valuetext={t('ra.genValue', { g: value })}
          onKeyDown={onKey}
          onPointerDown={(event) => {
            setPlaying(false)
            event.currentTarget.setPointerCapture(event.pointerId)
            fromPointer(event)
          }}
          onPointerMove={(event) => {
            if (event.currentTarget.hasPointerCapture(event.pointerId)) fromPointer(event)
          }}
        >
          <text x={20} y={31} className="axis">{t('ra.genAxis')}</text>
          <line x1={x(0)} x2={x(max)} y1={10} y2={10} stroke="var(--stroke)" strokeWidth={2} />
          <line x1={x(0)} x2={x(value)} y1={10} y2={10} stroke="var(--text)" strokeWidth={2} />
          <rect x={x(value) - 6} y={3} width={12} height={14} rx={3} fill="var(--text)" className="scrub-thumb" />
          {ticks.map((g) => (
            <g key={g}>
              <line x1={x(g)} x2={x(g)} y1={16} y2={20} stroke="var(--stroke)" />
              <text x={x(g)} y={33} textAnchor="middle" className="axis">{g}</text>
            </g>
          ))}
        </svg>
      )}
    </div>
  )
}
