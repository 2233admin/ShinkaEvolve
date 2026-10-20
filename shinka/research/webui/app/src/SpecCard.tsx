import { InlineNotification, Link, Tile } from '@carbon/react'
import { COPY, type CopyKey } from './i18n/copy'
import { Rich, useCopy } from './i18n/lang'
import { Markdown } from './Markdown'
import { STATUS, checksOf, fmt, money, num, pct, spendOf, trialsOf, when } from './format'
import { PARTS } from './Term'
import type { Evidence, Hypothesis, TrialPoint } from './types'

function Spec({ rows }: { rows: [React.ReactNode, React.ReactNode][] }) {
  return (
    <dl className="spec">
      {rows.map(([k, v], i) => (
        <div key={i}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  )
}

function Confirmation({ data }: { data: Evidence }) {
  const { t } = useCopy()
  const keys = Object.keys(data).filter((k) => data[k] !== undefined && typeof data[k] !== 'object')
  return (
    <section>
      <h3>
        <Rich k="spec.lockboxResult" parts={PARTS} />
      </h3>
      <Spec
        rows={keys.map((k) => {
          const key = `confirm.${k}`
          const v = data[k]
          const shown = typeof v === 'boolean' ? t(v ? 'confirm.yes' : 'confirm.no')
            : typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(3))
            : <span translate="no">{typeof v === 'string' ? v : JSON.stringify(v)}</span>
          return [key in COPY ? t(key as CopyKey) : <span translate="no">{k}</span>, <span className="mono">{shown}</span>]
        })}
      />
    </section>
  )
}

function Ledger({ h }: { h: Hypothesis }) {
  const { t } = useCopy()
  const v = h.verdict!
  const ev = v.evidence
  const th = ev.thresholds
  const noise = num(ev.expected_max_sharpe_annual)
  return (
    <section>
      <h3>{t('spec.ledger')}</h3>
      <ul className="ledger">
        {checksOf(ev, th, t('spec.pboNone')).map((c) => (
          <li key={c.key}>
            <rux-status status={c.pass === null ? 'off' : c.pass ? 'normal' : 'critical'} />
            <span className="ledger-label">
              {c.key === 'oos' && <Rich k="spec.checkOos" parts={PARTS} />}
              {c.key === 'hold' && <Rich k="spec.checkHold" parts={PARTS} />}
              {c.key === 'dsr' && <Rich k="spec.checkDsr" vars={{ v: th.dsr_supported }} parts={PARTS} />}
              {c.key === 'sub' && <Rich k="spec.checkSub" vars={{ v: pct(th.min_positive_fraction) }} parts={PARTS} />}
              {c.key === 'pbo' && <Rich k="spec.checkPbo" vars={{ v: th.max_pbo }} parts={PARTS} />}
            </span>
            <span className="mono ledger-value">{c.value}</span>
          </li>
        ))}
      </ul>
      <Spec
        rows={[
          [<Rich k="spec.isSharpe" parts={PARTS} />, <span className="mono">{fmt(ev.is_sharpe)}</span>],
          [<Rich k="spec.noise" parts={PARTS} />, <span className="mono">{fmt(noise)}</span>],
          [<Rich k="spec.trials" parts={PARTS} />, <span className="mono">{fmt(ev.n_trials, 0)}</span>],
        ]}
      />
      <p className="reason" translate="no">{v.reason}</p>
    </section>
  )
}

// 还没有试验的两种情形：没搭实验，或搭了没跑。
function NextStep({ h }: { h: Hypothesis }) {
  const { t } = useCopy()
  return (
    <section>
      <h3>{t('spec.next')}</h3>
      <p className="reason">
        <Rich k={h.experiments.length === 0 ? 'spec.nextScaffold' : 'spec.nextRun'} parts={PARTS} />
      </p>
    </section>
  )
}

// 有试验、没结论：只看这个假设自己的样本内进度，对着此刻的噪声线读。
function Progress({ pts, bar }: { pts: TrialPoint[]; bar: number | null }) {
  const { t } = useCopy()
  const scores = pts.flatMap((p) => (p.score === null ? [] : [p.score]))
  const best = scores.length ? Math.max(...scores) : null
  let reading = t('spec.readingFailed')
  if (best !== null && bar !== null) reading = best > bar ? t('spec.readingAbove') : t('spec.readingBelow')
  return (
    <section>
      <h3>{t('spec.progress')}</h3>
      <Spec
        rows={[
          [<Rich k="spec.trials" parts={PARTS} />, <span className="mono">{t('spec.trialsFailed', { n: pts.length, failed: pts.length - scores.length })}</span>],
          [<Rich k="spec.best" parts={PARTS} />, <span className="mono">{fmt(best)}</span>],
          [<Rich k="spec.noiseNow" parts={PARTS} />, <span className="mono">{fmt(bar)}</span>],
        ]}
      />
      <p className="reason">{reading}</p>
    </section>
  )
}

function Experiments({ h }: { h: Hypothesis }) {
  const { t } = useCopy()
  const exps = h.experiments.filter((e) => e.analysis_url)
  if (exps.length === 0) return null
  return (
    <section>
      <h3>{t('spec.analysis')}</h3>
      <ul className="trees">
        {exps.map((e) => (
          <li key={e.id}>
            <span className="mono" translate="no">{e.id}</span>
            <Link href={e.analysis_url!}>{t('spec.openAnalysis')}</Link>
          </li>
        ))}
      </ul>
    </section>
  )
}

type Props = { h: Hypothesis; points: TrialPoint[] }

export function SpecCard({ h, points }: Props) {
  const { t } = useCopy()
  const st = STATUS[h.status]
  const mine = points.filter((p) => p.hypothesis === h.id)
  const bar = points.length ? points[points.length - 1].bar : null
  return (
    <Tile className="spec-card">
      <h2 className="spec-title" translate="no">{h.title}</h2>
      <Spec
        rows={[
          [t('spec.id'), <span className="mono">{h.id}</span>],
          [
            t('spec.status'),
            <span className="status-cell">
              <rux-status status={st.rux} />
              {t(st.key)}
            </span>,
          ],
          [t('spec.created'), <span className="mono">{when(h.created)}</span>],
          [t('spec.experiments'), <span className="mono">{h.experiments.length}</span>],
          [<Rich k="spec.trials" parts={PARTS} />, <span className="mono">{trialsOf(h)}</span>],
          [t('spec.spend'), <span className="mono">{money(spendOf(h))}</span>],
        ]}
      />
      {h.body.trim() && <div translate="no"><Markdown text={h.body} /></div>}
      <Experiments h={h} />
      {h.verdict ? (
        <Ledger h={h} />
      ) : mine.length === 0 ? (
        <NextStep h={h} />
      ) : (
        <>
          <Progress pts={mine} bar={bar} />
          <InlineNotification
            kind="info"
            statusIconDescription={t('icon.info')}
            lowContrast
            hideCloseButton
            title={t('spec.noVerdict')}
            subtitle={t('spec.noVerdictHint')}
          />
        </>
      )}
      {h.confirmation && <Confirmation data={h.confirmation} />}
    </Tile>
  )
}
