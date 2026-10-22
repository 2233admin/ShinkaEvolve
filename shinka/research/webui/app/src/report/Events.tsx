import { EventMark } from '../design/icons'
import { when } from '../format'
import { COPY, type CopyKey } from '../i18n/copy'
import { useCopy } from '../i18n/lang'
import type { SnapshotEvent } from '../types'
import { useRemembered } from './remember'

/**
 * What happened in the direction, newest first (P10): time, hypothesis and experiment, the event as
 * shape and word, and its note as written. Skeptic and verdict notes stay hidden until the verdict.
 * The filter keeps the rows whose text contains it, as the event log always could (P10-Q2); it is
 * remembered per direction for the rest of the visit.
 */
export function Events({ events, direction }: { events: SnapshotEvent[]; direction: string }) {
  const { t } = useCopy()
  const [query, setQuery] = useRemembered(`${direction}/events`, '')
  if (events.length === 0) return <p className="muted small">{t('dir.noEvents')}</p>
  const word = (e: SnapshotEvent) => {
    const key = `evt.${e.event}`
    return key in COPY ? t(key as CopyKey) : e.event
  }
  const needle = query.trim().toLowerCase()
  const shown = needle
    ? events.filter((e) => [when(e.time), e.hypothesis, e.experiment, e.event, word(e), e.withheld ? '' : (e.note ?? '')].join(' ').toLowerCase().includes(needle))
    : events
  return (
    <>
      <div className="efilter">
        <input type="search" value={query} placeholder={t('rep.eventsFilter')} aria-label={t('rep.eventsFilter')} onChange={(event) => setQuery(event.currentTarget.value)} />
        <span className="faint small" role="status">{needle ? t('rep.eventsShown', { n: shown.length, total: events.length }) : ''}</span>
      </div>
      {shown.length === 0 ? (
        <p className="muted small">{t('rep.eventsNone')}</p>
      ) : (
        <div className="estream" role="table" aria-label={t('rep.events')}>
          {shown.map((e, index) => (
            <div className="erow" role="row" key={index}>
              <span role="cell" className="mono faint"><time dateTime={e.time}>{when(e.time)}</time></span>
              <span role="cell" className="mono faint" translate="no">{[e.hypothesis, e.experiment].filter(Boolean).join(' / ')}</span>
              <span role="cell" className="eword">
                <EventMark event={e.event} />
                <b className={e.event === 'run-failed' ? 'sig' : undefined}>{word(e)}</b>
              </span>
              <span role="cell">
                {e.withheld ? <span className="log is-line is-withheld">{t('event.withheld')}</span> : e.note && <code className="log is-line" translate="no">{e.note}</code>}
              </span>
            </div>
          ))}
        </div>
      )}
    </>
  )
}
