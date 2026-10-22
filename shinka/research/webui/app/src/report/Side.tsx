import { Fragment } from 'react'
import { Glyph, Icon } from '../design/icons'
import { fmt, int, money, num, pct, STATUS, when } from '../format'
import { COPY, type CopyKey } from '../i18n/copy'
import { Rich, useCopy } from '../i18n/lang'
import { PARTS } from '../Term'
import type { Evidence, Hypothesis, TrialPoint } from '../types'
import { experimentProgram, programHref } from './data'

/**
 * The verdict and its reason (P09-Q4). Before one: the hypothesis's own in-sample progress read
 * against the noise line now (P09-Q3), or the next command when nothing has run (P09-Q6).
 */
export function Verdict({ h, points }: { h: Hypothesis; points: TrialPoint[] }) {
  const { t } = useCopy()
  if (h.verdict) {
    return (
      <>
        <div className="vline">
          <Glyph kind={h.verdict.label} />
          <b>{t(STATUS[h.verdict.label].key)}</b>
          <span className="faint mono">{when(h.verdict.created)} UTC</span>
        </div>
        <pre className="log is-block" translate="no">{h.verdict.reason}</pre>
      </>
    )
  }
  const mine = points.filter((p) => p.hypothesis === h.id)
  if (mine.length === 0) {
    return (
      <p className="muted small">
        <Rich k={h.experiments.length === 0 ? 'rep.nextScaffold' : 'rep.nextRun'} parts={PARTS} />
      </p>
    )
  }
  const scores = mine.flatMap((p) => (p.score === null ? [] : [p.score]))
  const best = scores.length ? Math.max(...scores) : null
  const bar = points[points.length - 1].bar
  const reading = best === null ? 'rep.readingFailed' : best > bar ? 'rep.readingAbove' : 'rep.readingBelow'
  return (
    <>
      <div className="vline">
        <Glyph kind={h.status} />
        <b>{t(STATUS[h.status].key)}</b>
        <span className="faint">{t('rep.noVerdict')}</span>
      </div>
      <dl className="kv">
        <dt className="faint"><Rich k="rep.preTrials" parts={PARTS} /></dt>
        <dd className="mono">{t('rep.preTrialsFailed', { n: int(mine.length), failed: int(mine.length - scores.length) })}</dd>
        <dt className="faint"><Rich k="rep.preBest" parts={PARTS} /></dt>
        <dd className="mono">{fmt(best)}</dd>
        <dt className="faint"><Rich k="rep.preNoise" parts={PARTS} /></dt>
        <dd className="mono">{fmt(bar)}</dd>
      </dl>
      <p className="small">{t(reading)}</p>
      <p className="muted small">{t('rep.validationLater')}</p>
    </>
  )
}

// The confirmation's figures in the approved order; any other recorded field follows under its own label.
const LOCKBOX_ROWS: [string, CopyKey, (v: unknown) => string][] = [
  ['lockbox_sharpe', 'rep.lbSharpe', (v) => fmt(v)],
  ['lockbox_buy_hold_sharpe', 'rep.lbBuyHold', (v) => fmt(v)],
  ['lockbox_annual_return', 'rep.lbReturn', (v) => pct(v, 1)],
  ['lockbox_max_drawdown', 'rep.lbDrawdown', (v) => pct(v, 1)],
  ['dsr', 'rep.lbDsr', (v) => fmt(v, 3)],
  ['lockbox_positive_fraction', 'rep.lbPositive', (v) => pct(v)],
  ['n_trials', 'rep.lbTrials', (v) => (num(v) === null ? '—' : int(num(v)!))],
]
const SHOWN_ELSEWHERE = new Set(['passed', 'experiment', 'program_id', 'lockbox_bars', 'lockbox_start'])

/** The one lockbox opening (P09-Q5): pass or fail, its figures, the bars it covered and the program it judged. */
export function LockboxResults({ h, data }: { h: Hypothesis; data: Evidence }) {
  const { t } = useCopy()
  const known = new Set(LOCKBOX_ROWS.map(([key]) => key))
  const extra = Object.keys(data).filter((k) => !known.has(k) && !SHOWN_ELSEWHERE.has(k) && data[k] !== undefined && data[k] !== null && typeof data[k] !== 'object')
  const program = typeof data.program_id === 'string' ? data.program_id : null
  const experiment = h.experiments.find((e) => e.id === data.experiment)
  const bars = num(data.lockbox_bars)
  return (
    <>
      <div className="vline">
        <Glyph kind={h.status} />
        <b>{t(STATUS[h.status].key)}</b>
        <span className={data.passed === false ? 'sig' : 'faint'}>{t(data.passed === false ? 'rep.lbFailed' : 'rep.lbPassed')}</span>
      </div>
      <dl className="kv is-grid">
        {LOCKBOX_ROWS.filter(([key]) => data[key] !== undefined && data[key] !== null).map(([key, label, show]) => (
          <Fragment key={key}>
            <dt className="faint">{t(label)}</dt>
            <dd className="mono">{show(data[key])}</dd>
          </Fragment>
        ))}
        {extra.map((key) => {
          const label = `confirm.${key}`
          const value = data[key]
          // A recorded category such as a lockbox failure has its own words.
          const word = `confirm.${key}.${String(value)}`
          const worded = word in COPY
          return (
            <Fragment key={key}>
              <dt className="faint">{label in COPY ? t(label as CopyKey) : <span translate="no">{key}</span>}</dt>
              <dd className={worded ? undefined : 'mono'}>
                {worded ? t(word as CopyKey) : typeof value === 'number' ? (Number.isInteger(value) ? int(value) : fmt(value, 3)) : typeof value === 'boolean' ? t(value ? 'confirm.yes' : 'confirm.no') : <span translate="no">{String(value)}</span>}
              </dd>
            </Fragment>
          )
        })}
      </dl>
      {bars !== null && <p className="muted small">{t('rep.lbBars', { n: int(bars), start: String(data.lockbox_start ?? '').slice(0, 10) })}</p>}
      {program && (
        <p className="muted small">
          {t('rep.lbProgram')}{' '}
          {experiment?.analysis_url ? (
            <a className="mono" href={programHref(experiment.analysis_url, program)} translate="no">{`${experiment.id} · ${program.slice(0, 8)}`}</a>
          ) : (
            <span className="mono" translate="no">{program.slice(0, 8)}</span>
          )}
        </p>
      )}
    </>
  )
}

/**
 * Each experiment's span, counts, best score and cost (P09-Q2), and the way into its analysis: the
 * link opens, on its code, the program the verdict rests on, or the experiment's best before a check.
 */
export function Experiments({ h, points }: { h: Hypothesis; points: TrialPoint[] }) {
  const { t } = useCopy()
  if (h.experiments.length === 0) return <p className="muted small">{t('rep.noExperiments')}</p>
  return (
    <div className="xlist">
      {h.experiments.map((e) => {
        const program = experimentProgram(h, e, points)
        const href = e.analysis_url && (program ? programHref(e.analysis_url, program) : e.analysis_url)
        return (
          <article className="xrow" key={e.id}>
            <b className="mono" translate="no">{e.id}</b>
            <span className="faint mono">{e.first_trial && e.last_trial ? `${when(e.first_trial)} → ${when(e.last_trial)} UTC` : t('rep.noTrialsYet')}</span>
            <span><b className="mono">{int(e.n_evaluated)}</b> {t('rep.evaluated')}</span>
            <span><b className="mono">{int(e.n_probed)}</b> {t('rep.probed')}</span>
            <span>{t('rep.best')} <b className="mono">{fmt(e.best_is_score)}</b></span>
            {e.spend_known ? <b className="mono">{money(e.spend_usd)}</b> : <span title={t('rep.costUnknown')}><b className="mono">—</b> {t('rep.costUnknown')}</span>}
            {href && (
              <a className="btn" href={href}>
                {t('rep.openAnalysis')}
                <Icon name="arrow" />
              </a>
            )}
          </article>
        )
      })}
    </div>
  )
}
