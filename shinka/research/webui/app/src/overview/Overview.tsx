import { Fragment, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from 'react'
import { untagged } from '../demo'
import { EventMark, Glyph, Icon } from '../design/icons'
import { directionState, STATE_LOOK, type DirectionState } from '../direction'
import { fmt, int, money, pct, STATUS, when } from '../format'
import { COPY, type CopyKey } from '../i18n/copy'
import { Rich, useCopy } from '../i18n/lang'
import { routeHref } from '../shell/route'
import { useMatches } from '../shell/useMatches'
import { PARTS } from '../Term'
import type { DirectionEvidence, DirectionSummary, Status, WorkspaceEvent, WorkspaceSnapshot } from '../types'

// Gauges and their header axis share one inset, wide enough for a centred end label.
const EV_PAD = 14

// Statuses read best-first: outcomes, then work in progress, then what has not started.
const STATUS_ORDER: Status[] = ['confirmed', 'supported', 'unconfirmed', 'refuted', 'inconclusive', 'challenged', 'running', 'selected', 'proposed', 'dropped']

type Scale = { lo: number; hi: number; ticks: number[] }

/** One evidence scale for every row: round ticks (1, 2 or 5 × 10ⁿ) covering every best record and noise bar. */
function evidenceScale(rows: DirectionSummary[]): Scale | null {
  const values = rows.flatMap((d) => [d.evidence.best_score, d.evidence.noise_bar]).filter((v): v is number => v !== null)
  if (values.length === 0) return null
  let lo = Math.min(...values)
  let hi = Math.max(...values)
  if (lo === hi) [lo, hi] = [lo - 0.1, hi + 0.1]
  const raw = (hi - lo) / 4
  const power = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? raw
  const first = Math.floor(lo / step)
  const last = Math.ceil(hi / step)
  const ticks = Array.from({ length: last - first + 1 }, (_, k) => Number(((first + k) * step).toPrecision(12)))
  return { lo: ticks[0], hi: ticks[ticks.length - 1], ticks }
}

const at = (scale: Scale, width: number) => (v: number) => EV_PAD + ((v - scale.lo) / (scale.hi - scale.lo)) * (width - 2 * EV_PAD)

function Gauge({ evidence, scale, width }: { evidence: DirectionEvidence; scale: Scale | null; width: number }) {
  const { t } = useCopy()
  const { best_score: best, noise_bar: bar } = evidence
  if (best === null || bar === null || scale === null) return <div className="gauge is-empty">{t('home.noTrials')}</div>
  const x = at(scale, width)
  const above = best > bar
  const ink = above ? 'var(--signal-graphic)' : 'var(--muted)'
  return (
    <div className="gauge">
      <svg width={width} height="20" viewBox={`0 0 ${width} 20`} role="img" aria-label={`${t('home.best')} ${fmt(best)} · ${t('home.noise')} ${fmt(bar)}`}>
        <line x1={EV_PAD} x2={width - EV_PAD} y1="10" y2="10" stroke="var(--line-strong)" />
        {scale.ticks.map((tick) => (
          <line key={tick} x1={x(tick)} x2={x(tick)} y1="7" y2="13" stroke="var(--stroke)" />
        ))}
        <line x1={x(Math.min(bar, best))} x2={x(Math.max(bar, best))} y1="10" y2="10" stroke={ink} strokeWidth="2.5" strokeDasharray={above ? undefined : '2 2'} />
        <rect x={x(bar) - 1} y="2" width="2" height="16" fill="var(--muted)" />
        <path d={`M${x(best)},4l5.5,6-5.5,6-5.5-6z`} fill={above ? 'var(--signal-graphic)' : 'var(--bg)'} stroke={above ? 'var(--signal-graphic)' : 'var(--text)'} strokeWidth="1.3" />
      </svg>
      <div className="gread">
        <span>{t('home.best')} <b className="mono">{fmt(best)}</b></span>
        <span>{t('home.noise')} <b className="mono">{fmt(bar)}</b></span>
      </div>
    </div>
  )
}

/** Tick labels for the shared scale: every tick when labels are 44px apart, else every other one. */
function GaugeAxis({ scale, width }: { scale: Scale; width: number }) {
  const x = at(scale, width)
  const every = (width - 2 * EV_PAD) / (scale.ticks.length - 1) >= 44 ? 1 : 2
  return (
    <svg width={width} height="14" viewBox={`0 0 ${width} 14`} aria-hidden="true">
      {scale.ticks.filter((_, i) => i % every === 0).map((tick) => (
        <text key={tick} x={x(tick)} y="11" textAnchor="middle" className="axis">{fmt(tick)}</text>
      ))}
    </svg>
  )
}

function Hypotheses({ counts }: { counts: Partial<Record<Status, number>> }) {
  const { t } = useCopy()
  return (
    <div className="hyps">
      {STATUS_ORDER.filter((s) => counts[s]).map((s) => (
        <span key={s}>
          <Glyph kind={s} />
          <em>{t(STATUS[s].key)}</em>
          <b className="mono">{counts[s]}</b>
        </span>
      ))}
    </div>
  )
}

function Trials({ trials, evaluated, failed }: { trials: number; evaluated: number; failed: number }) {
  const { t } = useCopy()
  return (
    <div className="trials">
      <b className="mono">{int(trials)}</b>
      {evaluated > 0 && (
        <>
          <span className="faint">{t('home.programs')} <span className="mono">{int(evaluated)}</span></span>
          <span className="faint">{t('home.failed')} <span className="mono">{int(failed)}</span></span>
        </>
      )}
    </div>
  )
}

/** Spend against budget; over budget takes the signal, an unknown cost makes the spend a lower bound. */
function Budget({ spend, budget, unknown, width }: { spend: number; budget: number | null; unknown: number; width: number }) {
  const { t } = useCopy()
  const atLeast = unknown > 0 ? '≥ ' : ''
  const note = unknown > 0 && <span className="faint small">{t('meta.unknownCost', { n: unknown })}</span>
  if (budget === null) {
    return (
      <div className="budget">
        <span className="mono">{atLeast}{money(spend)}</span>
        <span className="faint">{t('home.noBudget')}</span>
        {note}
      </div>
    )
  }
  const share = budget > 0 ? spend / budget : spend > 0 ? Infinity : 0
  const over = spend > budget
  return (
    <div className="budget">
      <span className="mono">
        {atLeast}{money(spend)}<span className="faint"> / {money(budget)}</span>
      </span>
      <span className="bline">
        <svg width={width} height="6" viewBox={`0 0 ${width} 6`} aria-hidden="true">
          <rect width={width} height="2" y="2" fill="var(--line-strong)" />
          <rect width={(Math.min(1, share) * width).toFixed(1)} height="2" y="2" fill={over ? 'var(--signal-graphic)' : 'var(--muted)'} />
          {over && <rect x={width - 3} y="0" width="3" height="6" fill="var(--signal-graphic)" />}
        </svg>
        <span className={over ? 'mono sig' : 'mono faint'}>{Number.isFinite(share) ? `${atLeast}${Math.round(share * 100)}%` : '—'}</span>
      </span>
      {note}
    </div>
  )
}

function Lockbox({ opened }: { opened: boolean | null }) {
  const { t } = useCopy()
  if (opened === null) return <span className="lock faint">—</span>
  return opened ? (
    <span className="lock is-open"><Icon name="unlock" /><span>{t('home.opened')}</span></span>
  ) : (
    <span className="lock"><Icon name="lock" /><span>{t('home.sealed')}</span></span>
  )
}

type SortKey = 'activity' | 'evidence' | 'trials' | 'spend' | 'title'

const clearance = (d: DirectionSummary) =>
  d.evidence.best_score === null || d.evidence.noise_bar === null ? -Infinity : d.evidence.best_score - d.evidence.noise_bar

const SORTS: Record<SortKey, { label: CopyKey; order: (a: DirectionSummary, b: DirectionSummary) => number }> = {
  activity: { label: 'home.sortActivity', order: (a, b) => (b.evidence.last_activity ?? '').localeCompare(a.evidence.last_activity ?? '') },
  evidence: { label: 'home.sortEvidence', order: (a, b) => clearance(b) - clearance(a) },
  trials: { label: 'home.sortTrials', order: (a, b) => b.trials - a.trials },
  spend: { label: 'home.sortSpend', order: (a, b) => b.spend_usd - a.spend_usd },
  title: { label: 'home.sortTitle', order: (a, b) => untagged(a.direction).localeCompare(untagged(b.direction)) },
}

const rowId = (id: string) => `dir-${id}`

/** Arrow keys move between the rows' links; Enter opens the focused direction. */
function moveBetweenRows(event: KeyboardEvent<HTMLDivElement>) {
  if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
  const links = [...event.currentTarget.querySelectorAll<HTMLAnchorElement>('a.dlink')]
  const index = links.indexOf(document.activeElement as HTMLAnchorElement)
  if (index === -1) return
  event.preventDefault()
  links[Math.max(0, Math.min(links.length - 1, index + (event.key === 'ArrowDown' ? 1 : -1)))].focus()
}

function eventWord(event: string, t: (key: CopyKey) => string): string {
  const key = `evt.${event}`
  return key in COPY ? t(key as CopyKey) : event
}

// Below the table the feed is a short glance (the hi-fi 1024 frame draws six); beside it the column scrolls through every event.
const BELOW_EVENTS = 6

function Feed({ events, titles, below }: { events: WorkspaceEvent[]; titles: ReadonlyMap<string, string>; below: boolean }) {
  const { t } = useCopy()
  return (
    <aside className={below ? 'feed is-below' : 'feed'} aria-label={t('home.activity')}>
      <header>
        <b>{t('home.activity')}</b>
        <span className="faint">{t('home.utc')} · {t('home.allDirections')}</span>
      </header>
      {events.length === 0 && <p className="faint feed-empty">{t('home.noActivity')}</p>}
      {(below ? events.slice(0, BELOW_EVENTS) : events).map((e, index) => {
        const title = titles.get(e.direction) ?? e.direction
        return (
          <article className="ev" key={index}>
            <EventMark event={e.event} />
            <div className="evbody">
              <div className="evtitle">
                <b className={e.event === 'run-failed' ? 'sig' : undefined}>{eventWord(e.event, t)}</b>
                <a className="trunc faint" href={routeHref({ view: 'direction', direction: e.direction, hypothesis: e.hypothesis || null })} title={title} translate="no">
                  {title}
                </a>
              </div>
              <div className="evref mono faint">
                <span translate="no">{[e.hypothesis, e.experiment].filter(Boolean).join(' / ')}</span>
                <time dateTime={e.time}>{when(e.time)}</time>
              </div>
              {e.withheld ? (
                <p className="log is-withheld">{t('event.withheld')}</p>
              ) : e.note ? (
                <code className="log" translate="no" title={e.note}>{e.note}</code>
              ) : null}
            </div>
          </article>
        )
      })}
    </aside>
  )
}

/**
 * The research directions overview (award-redesign 4.1): one sentence, one sortable table, one activity column.
 * `freshness` is the shell's last-updated control; phones, whose top bar has no room for it, show it under the sentence.
 */
export function Overview({ workspace, freshness }: { workspace: WorkspaceSnapshot; freshness: ReactNode }) {
  const { lang, t } = useCopy()
  const phone = useMatches('(max-width: 767px)')
  const compact = useMatches('(max-width: 1279px)')
  const [sort, setSort] = useState<SortKey>('activity')
  const readable = workspace.directions.filter((d): d is DirectionSummary => d.error === null)
  const broken = workspace.directions.filter((d) => d.error !== null)
  const titles = new Map(readable.map((d) => [d.id, untagged(d.direction)]))
  const gaugeWidth = phone ? 286 : compact ? 200 : 176
  const scale = evidenceScale(readable)
  const states = new Map<string, DirectionState>(readable.map((d) => [d.id, directionState(d)]))
  const rows = [...readable].sort(SORTS[sort].order)

  if (workspace.directions.length === 0) {
    return (
      <section className="home-empty" aria-labelledby="home-title">
        <h1 className="dsent" id="home-title">{t('home.emptyTitle')}</h1>
        <p className="dsub">{t('home.emptyHint')}</p>
        {/* Lines break only between arguments, so a flag such as --direction never splits at its hyphens. */}
        <code className="cmd">
          {t('home.initCommand').split(' ').map((arg, index) => (
            <Fragment key={index}>
              {index > 0 && ' '}
              <span className="arg">{arg}</span>
            </Fragment>
          ))}
        </code>
        <p className="dsub">{t('home.emptyNav')}</p>
        {phone && <div className="dupdated">{freshness}</div>}
      </section>
    )
  }

  const sum = (f: (d: DirectionSummary) => number) => readable.reduce((s, d) => s + f(d), 0)
  const totals = {
    trials: sum((d) => d.trials),
    evaluated: sum((d) => d.evidence.evaluated_programs),
    failed: sum((d) => d.evidence.failed_programs),
    spend: sum((d) => d.spend_usd),
    unknown: sum((d) => d.unknown_cost_runs),
    budgeted: readable.some((d) => d.budget_usd !== null),
    budget: sum((d) => d.budget_usd ?? 0),
    opened: readable.filter((d) => d.lockbox_opened === true).length,
  }
  const statusSum: Partial<Record<Status, number>> = {}
  for (const d of readable) {
    for (const [s, n] of Object.entries(d.evidence.hypothesis_status) as [Status, number][]) statusSum[s] = (statusSum[s] ?? 0) + n
  }
  // The sentence links each category to its first row; an empty category is plain text.
  const first = (match: (d: DirectionSummary) => boolean) => rows.find(match)
  const category = (match: (d: DirectionSummary) => boolean, signal = false) => {
    const target = first(match)
    return (text: string) => (target ? (
      <a href={`#${encodeURIComponent(rowId(target.id))}`} className={signal ? 'is-signal' : undefined}>{text}</a>
    ) : (
      <span>{text}</span>
    ))
  }
  const isConfirmed = (d: DirectionSummary) => (d.evidence.hypothesis_status.confirmed ?? 0) > 0
  const isState = (state: DirectionState) => (d: DirectionSummary) => states.get(d.id) === state
  const withTrials = readable.filter((d) => d.evidence.best_score !== null && d.evidence.noise_bar !== null)
  const above = withTrials.filter((d) => clearance(d) > 0).length
  const spendText = `${totals.unknown > 0 ? '≥ ' : ''}${money(totals.spend)}`
  const presentStates = [...new Set(readable.map((d) => states.get(d.id)!))]
  const presentStatuses = STATUS_ORDER.filter((s) => statusSum[s])
  const style = { '--gw': `${gaugeWidth}px`, '--hw': lang === 'en' ? '196px' : '168px' } as CSSProperties

  return (
    <div className={compact ? 'home is-stacked' : 'home'} style={style}>
      <section className="home-main" aria-labelledby="home-title">
        <h1 className="dsent" id="home-title">
          <Rich
            k="home.sentence"
            vars={{
              n: readable.length,
              conf: readable.filter(isConfirmed).length,
              act: readable.filter(isState('active')).length,
              over: readable.filter(isState('over-budget')).length,
              fresh: readable.filter(isState('new')).length,
            }}
            parts={{
              conf: category(isConfirmed),
              act: category(isState('active')),
              over: category(isState('over-budget'), true),
              fresh: category(isState('new')),
            }}
          />
        </h1>
        <p className="dsub">
          <Rich
            k="home.sub"
            vars={{
              trials: int(totals.trials),
              evaluated: int(totals.evaluated),
              failed: int(totals.failed),
              rate: pct(totals.evaluated ? totals.failed / totals.evaluated : 0, 1),
              spend: spendText,
              budget: totals.budgeted ? money(totals.budget) : '—',
              above,
              withTrials: withTrials.length,
            }}
            parts={PARTS}
          />
        </p>
        {phone && <div className="dupdated">{freshness}</div>}
        <div className="tablebar">
          <span className="faint">{t('home.utc')}</span>
          <label className="select">
            <span>{t('home.sort')}</span>
            <select value={sort} onChange={(event) => setSort(event.target.value as SortKey)}>
              {(Object.keys(SORTS) as SortKey[]).map((key) => (
                <option key={key} value={key}>{t(SORTS[key].label)}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="dtable" role="table" aria-label={t('home.table')} onKeyDown={moveBetweenRows}>
          <div className="drow dhead" role="row">
            <span role="columnheader">{t('home.colDirection')}</span>
            <span role="columnheader">{t('home.colHypotheses')}</span>
            <span role="columnheader" className="evhead">
              {t('home.colEvidence')}
              {scale && <GaugeAxis scale={scale} width={gaugeWidth} />}
            </span>
            <span role="columnheader" className="num">{PARTS.trials(t('home.colTrials'))}</span>
            <span role="columnheader">{t('home.colBudget')}</span>
            <span role="columnheader">{PARTS.sealed(t('home.colLockbox'))}</span>
          </div>
          {rows.map((d) => {
            const look = STATE_LOOK[states.get(d.id)!]
            return (
              <div className="drow" role="row" id={rowId(d.id)} key={d.id}>
                <div className="dtitle" role="cell">
                  <a className="dlink" href={routeHref({ view: 'direction', direction: d.id, hypothesis: null })} translate="no">
                    {untagged(d.direction)}
                  </a>
                  <span className="dstate">
                    <Glyph kind={look.glyph} signal={look.signal} />
                    <span className={look.signal ? 'sig' : undefined}>{t(look.word)}</span>
                  </span>
                  <span className="dfolder mono faint">
                    <span className="trunc" translate="no">{d.id}</span>
                    <span>{t('home.last')} {d.evidence.last_activity ? when(d.evidence.last_activity) : '—'}</span>
                  </span>
                </div>
                <div role="cell"><Hypotheses counts={d.evidence.hypothesis_status} /></div>
                <div role="cell" className="dgauge"><Gauge evidence={d.evidence} scale={scale} width={gaugeWidth} /></div>
                <div role="cell"><Trials trials={d.trials} evaluated={d.evidence.evaluated_programs} failed={d.evidence.failed_programs} /></div>
                <div role="cell"><Budget spend={d.spend_usd} budget={d.budget_usd} unknown={d.unknown_cost_runs} width={phone ? 120 : 72} /></div>
                <div role="cell"><Lockbox opened={d.lockbox_opened} /></div>
              </div>
            )
          })}
          {broken.map((d) => {
            const why = t('ws.readError', { error: d.error ?? '' })
            return (
              <div className="drow is-error" role="row" key={d.id}>
                <div className="dtitle" role="cell">
                  <span className="dname">
                    <Glyph kind="error" signal />
                    <b className="mono" translate="no">{d.id}</b>
                  </span>
                  <span className="dwhy sig">{why}</span>
                </div>
                <div role="cell" className="faint derror">{t('home.unreadable')}</div>
              </div>
            )
          })}
          {readable.length > 0 && (
            <div className="drow dtotal" role="row">
              <div className="dtitle" role="cell"><b>{t('home.total', { n: readable.length })}</b></div>
              <div role="cell"><Hypotheses counts={statusSum} /></div>
              <div role="cell" className="dgauge" />
              <div role="cell"><Trials trials={totals.trials} evaluated={totals.evaluated} failed={totals.failed} /></div>
              <div role="cell"><Budget spend={totals.spend} budget={totals.budgeted ? totals.budget : null} unknown={totals.unknown} width={phone ? 120 : 72} /></div>
              <div role="cell"><span className="lock is-open"><Icon name="unlock" /><span>{t('home.openedN', { n: totals.opened })}</span></span></div>
            </div>
          )}
        </div>
        {totals.unknown > 0 && <p className="keynote">≥ {t('home.lowerBound')}</p>}
        <div className="key">
          <span className="faint">{t('home.keyDirection')}</span>
          {presentStates.map((s) => (
            <span key={s}>
              <Glyph kind={STATE_LOOK[s].glyph} signal={STATE_LOOK[s].signal} />
              {t(STATE_LOOK[s].word)}
            </span>
          ))}
        </div>
        <div className="key">
          <span className="faint">{t('home.keyHypothesis')}</span>
          {presentStatuses.map((s) => (
            <span key={s}>
              <Glyph kind={s} />
              {t(STATUS[s].key)}
            </span>
          ))}
        </div>
        {scale && <p className="keynote">{t('home.keyScale', { ticks: scale.ticks.map((tick) => fmt(tick)).join(' · ') })}</p>}
        {compact && <Feed events={workspace.events} titles={titles} below />}
      </section>
      {!compact && <Feed events={workspace.events} titles={titles} below={false} />}
    </div>
  )
}
