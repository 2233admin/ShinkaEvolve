import { EventStream } from './EventStream'
import { HypothesisTable } from './HypothesisTable'
import { Rich, useCopy } from './i18n/lang'
import { SpecCard } from './SpecCard'
import { PARTS } from './Term'
import { Timeline } from './Timeline'
import { TrialChart } from './TrialChart'
import type { Hypothesis, Snapshot } from './types'

type Props = { snap: Snapshot; current: Hypothesis; light: boolean; onSelect: (id: string) => void }

export function ResearchTab({ snap, current, light, onSelect }: Props) {
  const { t } = useCopy()
  return (
    <div className="grid">
      <section className="area-table">
        <h2 className="section-title">{t('dir.hypotheses')}</h2>
        <div className="table-wrap">
          <HypothesisTable hyps={snap.hypotheses} selected={current.id} onSelect={onSelect} />
        </div>
      </section>
      <section className="area-chart">
        <h2 className="section-title">{t('dir.trials')}</h2>
        {snap.trial_points.length === 0 ? (
          <p className="muted">
            {snap.probes > 0
              ? <Rich k="dir.noProgramsProbes" vars={{ n: snap.probes }} parts={PARTS} />
              : <Rich k="dir.noTrials" parts={PARTS} />}
          </p>
        ) : (
          <TrialChart points={snap.trial_points} probes={snap.probes} selected={current.id} light={light} />
        )}
      </section>
      <section className="area-timeline">
        <h2 className="section-title">{t('dir.timeline')}</h2>
        <Timeline hyps={snap.hypotheses} selected={current.id} onSelect={onSelect} />
      </section>
      <aside className="area-side">
        <SpecCard h={current} points={snap.trial_points} />
      </aside>
      <section className="area-events">
        <h2 className="section-title">{t('dir.events')}</h2>
        <EventStream events={snap.events} />
      </section>
    </div>
  )
}
