import { EVENT_STATUS } from './format'
import type { SnapshotEvent } from './types'

export function EventStream({ events }: { events: SnapshotEvent[] }) {
  if (events.length === 0) return <p className="muted">还没有事件。</p>
  const data = events.map((e) => ({
    timestamp: new Date(e.time),
    status: EVENT_STATUS[e.event] ?? ('off' as const),
    message: `${e.hypothesis}/${e.experiment} · ${e.event} · ${e.note}`,
  }))
  return <rux-log data={data} />
}
