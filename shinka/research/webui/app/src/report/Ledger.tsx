import type { ReactNode } from 'react'
import { Icon } from '../design/icons'
import { checksOf, fmt, int, num, pct } from '../format'
import { Rich, useCopy } from '../i18n/lang'
import { PARTS } from '../Term'
import type { Hypothesis } from '../types'

function Outcome({ pass }: { pass: boolean | null }) {
  const { t } = useCopy()
  if (pass === null) return <span className="faint">—</span>
  return (
    <span className={pass ? 'lpass' : 'lpass sig'}>
      <Icon name={pass ? 'check' : 'cross'} />
      {t(pass ? 'rep.pass' : 'rep.fail')}
    </span>
  )
}

/**
 * The skeptic's five rules with the thresholds it judged by (P09-Q4), then the reference figures it
 * reported. Validation figures exist only once there is a verdict.
 */
export function Ledger({ h }: { h: Hypothesis }) {
  const { t } = useCopy()
  if (h.verdict === null) return <p className="muted small">{t('rep.ledgerPending')}</p>
  const ev = h.verdict.evidence
  const th = ev.thresholds
  const [oos, hold, dsr, sub, pbo] = checksOf(ev, th, t('rep.pboNone'))
  const rules: [ReactNode, string, string, boolean | null][] = [
    [<Rich k="rep.checkOos" parts={PARTS} />, oos.value, '> 0.00', oos.pass],
    [<Rich k="rep.checkHold" parts={PARTS} />, hold.value, t('rep.aboveBaseline'), hold.pass],
    [<Rich k="rep.checkDsr" parts={PARTS} />, dsr.value, `≥ ${fmt(th.dsr_supported, 3)}`, dsr.pass],
    [<Rich k="rep.checkSub" parts={PARTS} />, sub.value, `≥ ${pct(th.min_positive_fraction)}`, sub.pass],
    [<Rich k="rep.checkPbo" parts={PARTS} />, pbo.value, `≤ ${fmt(th.max_pbo)}`, pbo.pass],
  ]
  const trials = num(ev.n_trials)
  const references: [ReactNode, string][] = [
    [<Rich k="rep.refIs" parts={PARTS} />, fmt(ev.is_sharpe)],
    [<Rich k="rep.refNoise" parts={PARTS} />, fmt(ev.expected_max_sharpe_annual)],
    [<Rich k="rep.refTrials" parts={PARTS} />, trials === null ? '—' : int(trials)],
    [<Rich k="rep.refSeed" parts={PARTS} />, `${fmt(ev.oos_sharpe)} / ${fmt(ev.baseline_seed_oos_sharpe)}`],
    [<Rich k="rep.refDecay" parts={PARTS} />, fmt(ev.decay)],
  ]
  return (
    <div className="ltable" role="table" aria-label={t('rep.ledger')}>
      <div className="lrow lhead" role="row">
        <span role="columnheader">{t('rep.check')}</span>
        <span role="columnheader" className="num">{t('rep.result')}</span>
        <span role="columnheader" className="num">{t('rep.threshold')}</span>
        <span role="columnheader">{t('rep.outcome')}</span>
      </div>
      {rules.map(([label, value, threshold, pass], i) => (
        <div className="lrow" role="row" key={i}>
          <span role="cell">{label}</span>
          <span role="cell" className="num mono">{value}</span>
          <span role="cell" className="num mono faint">{threshold}</span>
          <span role="cell"><Outcome pass={pass} /></span>
        </div>
      ))}
      {references.map(([label, value], i) => (
        <div className="lrow is-ref" role="row" key={`r${i}`}>
          <span role="cell">{label}</span>
          <span role="cell" className="num mono">{value}</span>
          <span role="cell" className="num mono faint">—</span>
          <span role="cell" className="faint">{t('rep.reference')}</span>
        </div>
      ))}
    </div>
  )
}
