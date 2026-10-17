import { useEffect, useRef, useState } from 'react'
import { STATUS } from './format'
import type { Hypothesis } from './types'

type Span = { h: Hypothesis; start: number; end: number }

// 每个有试验的假设一条轨道，区间是它所有实验第一次到最后一次试验。
function spansOf(hyps: Hypothesis[]): Span[] {
  const out: Span[] = []
  for (const h of hyps) {
    const t = h.experiments.flatMap((e) => (e.first_trial && e.last_trial ? [Date.parse(e.first_trial), Date.parse(e.last_trial)] : []))
    if (t.length) out.push({ h, start: Math.min(...t), end: Math.max(...t) })
  }
  return out
}

const MIN = 60_000
const LABEL_PX = 200 // 轨道左侧标签列的宽度（Astro 固定）
const iso = (ms: number) => new Date(ms).toISOString()

// 列宽 = zoom × 单位像素（分钟、小时 60，天 120）。挑最细的刻度，使整条时间线在不横向滚动的前提下列宽不小于 0.6 倍。
const UNITS = [
  { interval: 'minute', ms: MIN, px: 60 },
  { interval: 'hour', ms: 60 * MIN, px: 60 },
  { interval: 'day', ms: 24 * 60 * MIN, px: 120 },
] as const

// Astro 按整刻度数列：起止对齐到刻度边界，否则最后一个不满刻度会被丢掉。
function fit(lo: number, hi: number, width: number) {
  const avail = Math.max(width - LABEL_PX, 240)
  for (const u of UNITS) {
    const start = Math.floor(lo / u.ms) * u.ms
    const end = Math.max(Math.ceil(hi / u.ms) * u.ms, start + u.ms)
    const cols = (end - start) / u.ms
    if (cols * u.px * 0.6 <= avail || u.interval === 'day')
      return { interval: u.interval, start, end, zoom: Math.min(4, Math.max(0.3, avail / (cols * u.px))) }
  }
  throw new Error('unreachable')
}

export function Timeline({ hyps, selected, onSelect }: { hyps: Hypothesis[]; selected: string; onSelect: (id: string) => void }) {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(900)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const spans = spansOf(hyps)
  if (spans.length === 0) return <p className="muted">还没有试验，画不出时间线——用 <code className="mono">shinka_research run</code> 跑实验。</p>

  const lo = Math.min(...spans.map((s) => s.start))
  const hi = Math.max(...spans.map((s) => s.end))
  const pad = Math.max((hi - lo) * 0.03, 5 * MIN)
  const { interval, zoom, start: t0, end: t1 } = fit(lo - pad, hi + pad, width)

  return (
    <div className="timeline-wrap" ref={ref}>
      <rux-timeline key={`${interval}-${zoom.toFixed(2)}`} start={iso(t0)} end={iso(t1)} interval={interval} zoom={zoom} timezone="UTC" hideJDay>
        {spans.map(({ h, start, end }) => (
          <rux-track key={h.id}>
            <div slot="label" className="track-label">
              <span className="mono">{h.id}</span>
            </div>
            <rux-time-region
              start={iso(start)}
              end={iso(Math.max(end, start + MIN))}
              status={STATUS[h.status].rux}
              selected={h.id === selected}
              onClick={() => onSelect(h.id)}
            >
              {h.title}
            </rux-time-region>
          </rux-track>
        ))}
      </rux-timeline>
    </div>
  )
}
