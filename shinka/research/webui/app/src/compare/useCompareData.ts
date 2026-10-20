import { useEffect, useState } from 'react'
import type { RunStats } from '../runs/data'
import type { RunEntry } from '../types'
import type { ProgramSummary, PromptVersion, RunData } from './series'

/** The selected runs' data in the order of the selection, the latest read failure, and the selection it was read for. */
type Read = { key: string; runs: RunData[] | null; error: string | null }

async function readJson<T>(url: string, signal: AbortSignal): Promise<T> {
  const res = await fetch(url, { cache: 'no-store', signal })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return (await res.json()) as T
}

/**
 * Every selected run's programs, statistics and, when it evolves prompts, its prompt versions (the
 * legacy compare page's reads), read again whenever the shell lists the runs again, so the comparison
 * follows the automatic refresh and its pause. A read shows only once every run has arrived; a failed
 * read keeps the data shown before and says why.
 */
export function useCompareData(paths: string[], listing: RunEntry[] | null): { runs: RunData[] | null; error: string | null } {
  const key = JSON.stringify(paths)
  const [read, setRead] = useState<Read>({ key, runs: null, error: null })
  useEffect(() => {
    if (listing === null || paths.length < 2) return
    const controller = new AbortController()
    const names = new Map(listing.map((r) => [r.path, r.name]))
    const one = async (path: string): Promise<RunData> => {
      const q = `db_path=${encodeURIComponent(path)}`
      try {
        const [programs, stats] = await Promise.all([
          readJson<ProgramSummary[]>(`/get_programs_summary?${q}`, controller.signal),
          readJson<RunStats>(`/get_database_stats?${q}`, controller.signal),
        ])
        const prompts = stats.has_prompt_evo ? await readJson<PromptVersion[]>(`/get_system_prompts?${q}`, controller.signal) : []
        return { programs, stats, prompts }
      } catch (error) {
        if (controller.signal.aborted) throw error
        throw new Error(`${names.get(path) ?? path}: ${error instanceof Error ? error.message : String(error)}`)
      }
    }
    Promise.all(paths.map(one)).then(
      (runs) => setRead({ key, runs, error: null }),
      (error: unknown) => {
        if (controller.signal.aborted) return
        const message = error instanceof Error ? error.message : String(error)
        setRead((previous) => ({ key, runs: previous.key === key ? previous.runs : null, error: message }))
      },
    )
    return () => controller.abort()
    // `paths` is compared by its key; a new listing is a new read.
  }, [listing, key])
  return read.key === key ? read : { runs: null, error: null }
}
