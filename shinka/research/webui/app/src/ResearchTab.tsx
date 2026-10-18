import { EventStream } from './EventStream'
import { HypothesisTable } from './HypothesisTable'
import { SpecCard } from './SpecCard'
import { Timeline } from './Timeline'
import { TrialChart } from './TrialChart'
import type { Hypothesis, Snapshot } from './types'

type Props = { snap: Snapshot; current: Hypothesis; light: boolean; onSelect: (id: string) => void }

export function ResearchTab({ snap, current, light, onSelect }: Props) {
  return (
    <div className="grid">
      <section className="area-table">
        <h2 className="section-title">假设</h2>
        <div className="table-wrap">
          <HypothesisTable hyps={snap.hypotheses} selected={current.id} onSelect={onSelect} />
        </div>
      </section>
      <section className="area-chart">
        <h2 className="section-title">试验</h2>
        {snap.trial_points.length === 0 ? (
          <p className="muted">
            {snap.probes > 0 ? `还没有进化出的程序（只有 ${snap.probes} 次试探，试探不画点）` : '还没有试验'}——用{' '}
            <code className="mono">shinka_research run</code> 跑实验。
          </p>
        ) : (
          <TrialChart points={snap.trial_points} probes={snap.probes} selected={current.id} light={light} />
        )}
      </section>
      <section className="area-timeline">
        <h2 className="section-title">实验时间线</h2>
        <Timeline hyps={snap.hypotheses} selected={current.id} onSelect={onSelect} />
      </section>
      <aside className="area-side">
        <SpecCard h={current} points={snap.trial_points} />
      </aside>
      <section className="area-events">
        <h2 className="section-title">事件</h2>
        <EventStream events={snap.events} />
      </section>
    </div>
  )
}
