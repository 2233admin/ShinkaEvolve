import { useEffect, useState } from 'react'
import type { RunEntry } from '../types'
import type { RunStats } from './data'

// As the legacy list did (D-P01a): every listed run, four requests at a time.
const BATCH = 4

/**
 * A run's statistics as last read, and whether the latest read failed: `stats` null with `failed`
 * means it was never read; `stats` with `failed` means the numbers shown are from an earlier read.
 * A run absent from the map is still loading.
 */
export type StatsEntry = { stats: RunStats | null; failed: boolean }
export type StatsMap = ReadonlyMap<string, StatsEntry>

/**
 * `/get_database_stats` for every listed run, read again whenever the shell lists the runs again, so
 * the numbers follow its automatic refresh and its pause. Numbers already shown stay until new ones
 * arrive, and a failed read says so on its row instead of passing old numbers off as current.
 */
export function useRunStats(runs: RunEntry[] | null): StatsMap {
  const [stats, setStats] = useState<StatsMap>(() => new Map())
  useEffect(() => {
    if (runs === null) return
    const controller = new AbortController()
    const read = async (r: RunEntry): Promise<[string, RunStats | null]> => {
      try {
        const res = await fetch(`/get_database_stats?db_path=${encodeURIComponent(r.path)}`, { cache: 'no-store', signal: controller.signal })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return [r.path, (await res.json()) as RunStats]
      } catch (error) {
        if (controller.signal.aborted) throw error
        return [r.path, null]
      }
    }
    void (async () => {
      for (let i = 0; i < runs.length; i += BATCH) {
        const results = await Promise.all(runs.slice(i, i + BATCH).map(read))
        if (controller.signal.aborted) return
        setStats((previous) => {
          const next = new Map(previous)
          for (const [path, value] of results) {
            next.set(path, value === null ? { stats: previous.get(path)?.stats ?? null, failed: true } : { stats: value, failed: false })
          }
          return next
        })
      }
    })().catch(() => {
      /* Aborted by a newer listing. */
    })
    return () => controller.abort()
  }, [runs])
  return stats
}
