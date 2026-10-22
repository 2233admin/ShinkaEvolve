import { useCallback, useEffect, useState } from 'react'
import type { RunEntry, Snapshot, WorkspaceSnapshot } from '../types'

// OD-17: the data refreshes on its own and keeps the current selection and scroll position. The run
// list refreshes as often as its legacy page did (P01-C5); every other page every 30 seconds.
export const REFRESH_MS = 30_000
export const RUNS_REFRESH_MS = 10_000

export type ShellData = {
  workspace: WorkspaceSnapshot | null
  /** The open direction's snapshot; null until it has loaded, and null when the direction does not exist. */
  snap: Snapshot | null
  /** The open direction does not exist in the workspace (the snapshot answered 404); the shell itself is fine. */
  missing: boolean
  /** Every run database; null until listed, and kept from the last good listing when one fails. */
  runs: RunEntry[] | null
  /** The first listing attempt has finished, whatever its outcome. */
  listed: boolean
  /** Why the latest listing failed (P01-E1); the runs above are still from the last good one. */
  runsError: string | null
  /** When the workspace (and open direction) last loaded. */
  updated: Date | null
  /** Why the latest load failed; the data above is still from `updated`. */
  error: string | null
  refresh: () => void
  /** Automatic refresh is on. A pause lasts for this visit only (the legacy toggle reset on every load); a click on Updated still refreshes. */
  live: boolean
  setLive: (live: boolean) => void
  /** How often the automatic refresh runs, in milliseconds. */
  every: number
}

async function getJson<T>(url: string, signal: AbortSignal): Promise<T> {
  const res = await fetch(url, { cache: 'no-store', signal })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<T>
}

/** Workspace summary, run list and the open direction's snapshot, refreshed together every `every` ms while visible. */
export function useShellData(direction: string | null, every = REFRESH_MS): ShellData {
  const [loaded, setLoaded] = useState<{
    workspace: WorkspaceSnapshot | null
    snap: Snapshot | null
    snapFor: string | null
    missing: boolean
    updated: Date | null
    error: string | null
  }>({ workspace: null, snap: null, snapFor: null, missing: false, updated: null, error: null })
  const [runs, setRuns] = useState<RunEntry[] | null>(null)
  const [runsError, setRunsError] = useState<string | null>(null)
  const [listed, setListed] = useState(false)
  const [round, setRound] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    const { signal } = controller
    Promise.all([
      getJson<WorkspaceSnapshot>('/api/workspace', signal),
      // An unknown direction is an answer about the page, not a failed connection: the rest of the shell loads.
      direction === null
        ? null
        : getJson<Snapshot>(`/api/snapshot?direction=${encodeURIComponent(direction)}`, signal).catch((e: unknown) => {
            if (e instanceof Error && e.message === 'HTTP 404') return 'missing' as const
            throw e
          }),
    ]).then(
      ([workspace, snap]) =>
        setLoaded({ workspace, snap: snap === 'missing' ? null : snap, snapFor: direction, missing: snap === 'missing', updated: new Date(), error: null }),
      (e: unknown) => {
        // A failure says nothing about whether the direction exists, so an earlier 404 does not outlive it.
        if (!signal.aborted) setLoaded((s) => ({ ...s, missing: false, error: e instanceof Error ? e.message : String(e) }))
      },
    )
    // The research pages never wait for the run list.
    getJson<RunEntry[]>('/list_databases', signal)
      .then(
        (listing) => {
          setRuns(listing)
          setRunsError(null)
        },
        (e: unknown) => {
          if (!signal.aborted) setRunsError(e instanceof Error ? e.message : String(e))
        },
      )
      .finally(() => {
        if (!signal.aborted) setListed(true)
      })
    return () => controller.abort()
  }, [direction, round])

  const [live, setLive] = useState(true)
  useEffect(() => {
    if (!live) return
    const again = () => {
      if (document.visibilityState === 'visible') setRound((r) => r + 1)
    }
    const timer = window.setInterval(again, every)
    document.addEventListener('visibilitychange', again)
    window.addEventListener('online', again)
    return () => {
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', again)
      window.removeEventListener('online', again)
    }
  }, [live, every])
  const resume = useCallback((next: boolean) => {
    setLive(next)
    // Resuming catches up at once instead of waiting for the next tick.
    if (next) setRound((r) => r + 1)
  }, [])

  const refresh = useCallback(() => setRound((r) => r + 1), [])
  const { workspace, snap, snapFor, missing, updated, error } = loaded
  const here = snapFor === direction
  return { workspace, snap: here ? snap : null, missing: here && missing, runs, listed, runsError, updated, error, refresh, live, setLive: resume, every }
}
