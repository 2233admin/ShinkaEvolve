import { Fragment } from 'react'
import { untagged } from '../demo'
import { Icon } from '../design/icons'
import { int, money, spendOf, trialsOf, when } from '../format'
import type { CopyKey } from '../i18n/copy'
import { Rich, useCopy } from '../i18n/lang'
import { Markdown } from '../Markdown'
import { PARTS } from '../Term'
import type { Hypothesis, Snapshot, TrialPoint } from '../types'
import { bodySection, programHref } from './data'
import { Events } from './Events'
import { HypothesisTable } from './HypothesisTable'
import { Ledger } from './Ledger'
import { useRemembered } from './remember'
import { Experiments, LockboxResults, Verdict } from './Side'
import { Timeline } from './Timeline'
import { TrialChart } from './TrialChart'

const DEFINITIONS: [string, CopyKey][] = [
  ['Claim', 'rep.claim'],
  ['Mechanism', 'rep.mechanism'],
  ['Falsification', 'rep.falsification'],
]

/**
 * The selected hypothesis as the notebook states it (P09-Q1). When the body has Claim, Mechanism and
 * Falsification sections they lead and the whole body folds under them; otherwise the body is shown
 * as written.
 */
function Claim({ h, direction }: { h: Hypothesis; direction: string }) {
  const { t } = useCopy()
  const [open, setOpen] = useRemembered(`${direction}/${h.id}/body`, false)
  const definitions = DEFINITIONS.map(([heading, label]) => [label, bodySection(h.body, heading)] as const).filter(([, text]) => text)
  const unknownCost = h.experiments.some((e) => !e.spend_known)
  const body = h.body.trim() && (
    <div translate="no">
      <Markdown text={h.body} />
    </div>
  )
  return (
    <>
      <h2 id="claim-title">
        {t('rep.hypothesis')} <span className="mono" translate="no">{h.id}</span>
      </h2>
      <p className="claim" translate="no">{untagged(h.title)}</p>
      {definitions.length > 0 && (
        <dl className="defs">
          {definitions.map(([label, text]) => (
            <Fragment key={label}>
              <dt>{t(label)}</dt>
              <dd translate="no">{text}</dd>
            </Fragment>
          ))}
        </dl>
      )}
      <p className="hmeta">
        <span>{t('meta.created')} <span className="mono">{when(h.created)} UTC</span></span>
        <span>{t('rep.experimentsN', { n: h.experiments.length })}</span>
        <span><Rich k="meta.trials" vars={{ n: int(trialsOf(h)) }} parts={PARTS} /></span>
        <span className="mono" title={unknownCost ? t('rep.costUnknown') : undefined}>{`${unknownCost ? '≥ ' : ''}${money(spendOf(h))}`}</span>
      </p>
      {body &&
        (definitions.length > 0 ? (
          <details className="fulltext" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
            <summary>{t('rep.fullText')}</summary>
            {body}
          </details>
        ) : (
          <div className="fulltext">{body}</div>
        ))}
    </>
  )
}

type Props = {
  snap: Snapshot
  /** The direction's folder ID: what the report remembers its sort and open sections under. */
  direction: string
  current: Hypothesis
  onSelect: (id: string) => void
  /** Opens an in-app link: an experiment's analysis or one of its programs. */
  onOpen: (href: string) => void
}

/**
 * The direction report (award-redesign 4.2) in the order the argument runs: hypotheses, then the
 * selected one's claim, its trials against the noise line, the check ledger and the verdict with the
 * lockbox confirmation and experiments, then the timeline and the events. From 1440px the verdict
 * column sits beside the argument.
 */
export function Report({ snap, direction, current, onSelect, onOpen }: Props) {
  const { t } = useCopy()
  const analysis = new Map(snap.hypotheses.flatMap((h) => h.experiments.map((e) => [`${h.id}/${e.id}`, e.analysis_url] as const)))
  const hrefOf = (p: TrialPoint) => {
    const url = analysis.get(`${p.hypothesis}/${p.experiment}`)
    return url ? programHref(url, p.program) : null
  }
  const failed = snap.trial_points.filter((p) => p.score === null).length
  return (
    <>
      <section className="rsec" aria-labelledby="hyp-title">
        <h2 id="hyp-title">{t('rep.hypotheses')}</h2>
        <HypothesisTable hypotheses={snap.hypotheses} direction={direction} selected={current.id} onSelect={onSelect} />
      </section>
      <div className="rcols">
        <section className="r-claim" aria-labelledby="claim-title">
          <Claim key={current.id} h={current} direction={direction} />
        </section>
        <section className="r-chart" aria-labelledby="chart-title">
          <div className="shead">
            <h2 id="chart-title">{t('rep.trialsAndNoise')}</h2>
            {snap.trial_points.length > 0 && (
              <p className="legend">
                <span><i className="ln" aria-hidden="true" />{t('rep.record')}</span>
                <span><i className="ln is-dash" aria-hidden="true" />{t('rep.noise')}</span>
                <span><i className="dot" aria-hidden="true" />{t('rep.thisHyp')}</span>
                <span><i className="dot is-hollow" aria-hidden="true" />{t('rep.otherHyp')}</span>
                <span>
                  <i className="is-signal" aria-hidden="true"><Icon name="cross" /></i>
                  {t('rep.failedN', { n: int(failed) })}
                </span>
              </p>
            )}
          </div>
          {snap.trial_points.length === 0 ? (
            <p className="muted small">
              {snap.probes > 0 ? <Rich k="dir.noProgramsProbes" vars={{ n: snap.probes }} parts={PARTS} /> : <Rich k="dir.noTrials" parts={PARTS} />}
            </p>
          ) : (
            <>
              <TrialChart points={snap.trial_points} selected={current.id} hrefOf={hrefOf} onOpen={onOpen} />
              {snap.probes > 0 && <p className="faint small">{t('rep.probesNote', { n: int(snap.probes) })}</p>}
            </>
          )}
        </section>
        <section className="r-ledger" aria-labelledby="ledger-title">
          <h2 id="ledger-title">
            {t('rep.ledger')} · <span className="mono" translate="no">{current.id}</span>
          </h2>
          <Ledger h={current} />
        </section>
        <aside className="r-side">
          <section aria-labelledby="verdict-title">
            <h2 id="verdict-title">{t('rep.verdict')}</h2>
            <Verdict h={current} points={snap.trial_points} />
          </section>
          {current.confirmation && (
            <section aria-labelledby="lockbox-title">
              <h2 id="lockbox-title">{t('rep.lockboxResults')}</h2>
              <LockboxResults h={current} data={current.confirmation} />
            </section>
          )}
          <section aria-labelledby="exp-title">
            <h2 id="exp-title">{t('rep.experiments')}</h2>
            <Experiments h={current} points={snap.trial_points} />
          </section>
        </aside>
        <section className="r-timeline" aria-labelledby="timeline-title">
          <h2 id="timeline-title">{t('rep.timeline')}</h2>
          <Timeline hypotheses={snap.hypotheses} events={snap.events} selected={current.id} onSelect={onSelect} />
        </section>
        <section className="r-events" aria-labelledby="events-title">
          <h2 id="events-title">{t('rep.events')}</h2>
          <Events events={snap.events} direction={direction} />
        </section>
      </div>
    </>
  )
}
