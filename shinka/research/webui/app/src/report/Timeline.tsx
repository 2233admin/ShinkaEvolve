import { scaleUtc, utcFormat } from 'd3'
import type { KeyboardEvent, ReactNode } from 'react'
import { untagged } from '../demo'
import { Glyph } from '../design/icons'
import { STATUS, when } from '../format'
import { Rich, useCopy } from '../i18n/lang'
import { PARTS } from '../Term'
import type { Hypothesis, SnapshotEvent } from '../types'
import { marks, spans, type MarkKind, type Span } from './data'
import { useWidth } from '../design/useWidth'

const ROW = 40
const DAY = 864e5

const MARK: Record<MarkKind, ReactNode> = {
  skeptic: <path d="M0,-5l5,5-5,5-5-5z" fill="var(--bg)" stroke="var(--text)" strokeWidth={1.3} />,
  verdict: <path d="M0,-5l5,5-5,5-5-5z" fill="var(--text)" />,
  confirm: (
    <>
      <rect x={-4.5} y={-1} width={9} height={6.5} rx={1} fill="var(--bg)" stroke="var(--text)" strokeWidth={1.3} />
      <path d="M-2.5,-1v-1.7a2.5,2.5 0 0 1 4.9-.7" fill="none" stroke="var(--text)" strokeWidth={1.3} />
    </>
  ),
}

type Props = { hypotheses: Hypothesis[]; events: SnapshotEvent[]; selected: string; onSelect: (id: string) => void }

/**
 * When each hypothesis ran (P08): one track from its first to its last trial, with its skeptic checks,
 * verdict and lockbox confirmation on the track. The axis picks days or hours to fit the span; a track
 * selects its hypothesis.
 */
export function Timeline({ hypotheses, events, selected, onSelect }: Props) {
  // The measured box stays mounted while the direction has no trials, so tracks that arrive with a
  // later snapshot are drawn at once.
  const [box, width] = useWidth<HTMLDivElement>()
  const tracks = spans(hypotheses)
  return (
    <div className="tline" ref={box}>
      {tracks.length === 0 ? (
        <p className="muted small"><Rich k="dir.noTimeline" parts={PARTS} /></p>
      ) : (
        width > 0 && <Tracks tracks={tracks} hypotheses={hypotheses} events={events} selected={selected} onSelect={onSelect} width={width} />
      )}
    </div>
  )
}

function Tracks({ tracks, hypotheses, events, selected, onSelect, width }: Props & { tracks: Span[]; width: number }) {
  const { t } = useCopy()
  const onTrack = marks(hypotheses, events)
  const label = width < 560 ? 132 : 176
  const times = [...tracks.flatMap((s) => [s.start, s.end]), ...onTrack.map((m) => m.time)]
  const lo = Math.min(...times)
  const hi = Math.max(...times)
  // Whole days when the research spans days; otherwise a padded window in hours.
  const days = hi - lo >= 2 * DAY
  const domain = days ? [Math.floor(lo / DAY) * DAY, Math.ceil(hi / DAY) * DAY + DAY] : [lo - Math.max((hi - lo) * 0.05, 15 * 6e4), hi + Math.max((hi - lo) * 0.05, 15 * 6e4)]
  const x = scaleUtc(domain.map((v) => new Date(v)), [label, Math.max(label + 120, width - 16)])
  // About one label per 48px; in days never finer than one a day.
  const fit = Math.max(2, Math.floor((width - label - 16) / 48))
  const ticks = x.ticks(days ? Math.min(fit, Math.round((domain[1] - domain[0]) / DAY)) : fit)
  // In hours a midnight tick carries the new date, so a span across days still reads.
  const tickText = (tick: Date) => utcFormat(days || (tick.getUTCHours() === 0 && tick.getUTCMinutes() === 0) ? '%m-%d' : '%H:%M')(tick)
  const height = tracks.length * ROW + 34
  const spanText = (a: number, b: number) => {
    const [ta, tb] = [when(new Date(a).toISOString()), when(new Date(b).toISOString())]
    return ta.slice(0, 10) === tb.slice(0, 10) ? `${ta.slice(5)}–${tb.slice(11)}` : `${ta.slice(5)} – ${tb.slice(5)}`
  }
  const axisTitle = days ? t('rep.dateAxis') : t('rep.timeAxis', { day: when(new Date(lo).toISOString()).slice(0, 10) })
  const pick = (event: KeyboardEvent<SVGGElement>, id: string) => {
    if (event.key !== 'Enter' && event.key !== ' ') return
    event.preventDefault()
    onSelect(id)
  }

  return (
    <>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="group" aria-label={t('rep.timeline')}>
        {ticks.map((tick) => (
          <g key={tick.getTime()} aria-hidden="true">
            <line x1={x(tick)} x2={x(tick)} y1={0} y2={height - 24} stroke="var(--grid)" />
            <text x={x(tick)} y={height - 8} textAnchor="middle" className="axis">{tickText(tick)}</text>
          </g>
        ))}
        {tracks.map((s, k) => {
          const h = s.hypothesis
          const y = k * ROW + 20
          const isSelected = h.id === selected
          const x1 = x(new Date(s.start))
          const x2 = Math.max(x(new Date(s.end)), x1 + 2)
          const span = spanText(s.start, s.end)
          // The label starts under the track; near the right edge it ends there instead (10.5px mono is 6.3px a character).
          const late = x1 - 3 + span.length * 6.3 > width
          return (
            <g
              key={h.id}
              className="tl-track"
              role="button"
              tabIndex={0}
              aria-pressed={isSelected}
              aria-label={t('rep.trackLabel', { id: h.id, status: t(STATUS[h.status].key), span, title: untagged(h.title) })}
              onClick={() => onSelect(h.id)}
              onKeyDown={(event) => pick(event, h.id)}
            >
              <rect x={0} y={y - ROW / 2} width={width} height={ROW} fill="transparent" />
              {/* Trailing spaces keep the drawn texts separate words when read together (ID, status, span), so the
                  accessible name, which starts with them, matches what is seen. SVG does not draw them. */}
              <text x={4} y={y + 4} className="axis mono">{`${h.id} `}</text>
              <g transform={`translate(42,${y - 7})`}><Glyph kind={h.status} /></g>
              <text x={62} y={y + 4} className="tl">{`${t(STATUS[h.status].key)} `}</text>
              <line x1={x1} x2={x2} y1={y} y2={y} stroke={isSelected ? 'var(--text)' : 'var(--muted)'} strokeWidth={6} strokeLinecap="round" />
              <text x={late ? Math.min(width, x2 + 3) : x1 - 3} y={y + 18} textAnchor={late ? 'end' : undefined} className="axis halo">{span}</text>
              {onTrack.filter((m) => m.hypothesis === h.id).map((m, i) => (
                <g key={i} transform={`translate(${x(new Date(m.time))},${y})`}>{MARK[m.kind]}</g>
              ))}
            </g>
          )
        })}
        <text x={0} y={height - 8} className="axis-title" aria-hidden="true">{axisTitle}</text>
      </svg>
      <p className="tline-key small faint">
        <span><svg width="12" height="12" viewBox="-6 -6 12 12" aria-hidden="true">{MARK.skeptic}</svg>{t('evt.skeptic')}</span>
        <span><svg width="12" height="12" viewBox="-6 -6 12 12" aria-hidden="true">{MARK.verdict}</svg>{t('evt.verdict')}</span>
        <span><svg width="12" height="12" viewBox="-6 -6 12 12" aria-hidden="true">{MARK.confirm}</svg>{t('evt.confirm')}</span>
      </p>
    </>
  )
}
