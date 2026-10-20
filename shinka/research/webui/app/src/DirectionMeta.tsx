import { Glyph, Icon } from './design/icons'
import { directionState, STATE_LOOK } from './direction'
import { pct, usd, when } from './format'
import { Rich, useCopy } from './i18n/lang'
import { PARTS } from './Term'
import type { DirectionSummary, Snapshot } from './types'

// P03-Q1: the direction's state, trials, spend against budget (a note from 80%, the signal once over)
// and lockbox, each with its word.
export function DirectionMeta({ snap, entry }: { snap: Snapshot; entry: DirectionSummary | null }) {
  const { t } = useCopy()
  const look = entry ? STATE_LOOK[directionState(entry)] : null
  const budget = snap.budget_usd
  const ratio = budget ? snap.spend_usd / budget : null
  const spend = `${snap.unknown_cost_runs > 0 ? '≥ ' : ''}${usd(snap.spend_usd)}`
  const lockbox = snap.lockbox === null ? t('lockbox.unset') : snap.lockbox.opened ? t('lockbox.opened') : t('lockbox.sealed')
  return (
    <div className="dmeta">
      {look && (
        <span className="dmeta-item">
          <Glyph kind={look.glyph} signal={look.signal} />
          <span className={look.signal ? 'sig' : undefined}>{t(look.word)}</span>
        </span>
      )}
      <span className="dmeta-item faint">
        {t('meta.created')} <span className="mono">{when(snap.created)} UTC</span>
      </span>
      <span className="dmeta-item">
        <Rich k="meta.trials" vars={{ n: snap.trials }} parts={PARTS} />
        <span className="faint">· {t('meta.evaluated', { n: snap.trial_points.length })}</span>
      </span>
      <span className="dmeta-item">
        <span className="mono">{budget === null ? spend : `${spend} / ${usd(budget)}`}</span>
        {budget === null && <span className="faint">{t('meta.noBudget')}</span>}
        {ratio !== null && ratio >= 0.8 && ratio <= 1 && (
          <>
            <Glyph kind="error" />
            <span>{t('meta.budgetUsed', { pct: pct(ratio) })}</span>
          </>
        )}
        {snap.unknown_cost_runs > 0 && <span className="faint">· {t('meta.unknownCost', { n: snap.unknown_cost_runs })}</span>}
      </span>
      <span className="dmeta-item">
        <Icon name={snap.lockbox !== null && !snap.lockbox.opened ? 'lock' : 'unlock'} />
        <Rich k="meta.lockbox" vars={{ state: lockbox }} parts={PARTS} />
      </span>
    </div>
  )
}
