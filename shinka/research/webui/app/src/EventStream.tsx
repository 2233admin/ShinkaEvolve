import { EVENT_STATUS } from './format'
import { useCopy } from './i18n/lang'
import type { SnapshotEvent } from './types'

export function EventStream({ events }: { events: SnapshotEvent[] }) {
  const { t } = useCopy()
  if (events.length === 0) return <p className="muted">{t('dir.noEvents')}</p>
  const data = events.map((e) => ({
    timestamp: new Date(e.time),
    status: EVENT_STATUS[e.event] ?? ('off' as const),
    message: `${e.hypothesis}/${e.experiment} · ${e.event} · ${e.withheld ? t('event.withheld') : e.note}`,
  }))
  return <rux-log data={data} />
}
