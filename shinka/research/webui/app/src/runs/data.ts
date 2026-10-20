/**
 * The run list's derived data (award-redesign 5.1). Plain functions with type-only imports, so node can
 * test them directly.
 */
import type { RunEntry } from '../types'

/** `/get_database_stats` for one run, as the run list reads it. */
export type RunStats = {
  program_count: number
  correct_count: number
  best_score: number | null
  max_generation: number | null
  gens_since_improvement: number | null
  /** The programs' recorded cost; null when one of them has no price. */
  total_cost: number | null
  /** Prompt evolution's recorded cost; null when a prompt has no price. */
  prompt_evo_cost: number | null
  prompt_count: number
  has_prompt_evo: boolean
  total_runtime_seconds: number | null
  /** Unix seconds of the newest program; null for a run without programs. */
  last_update: number | null
}

/**
 * A run's statistics as last read, and whether the latest read failed: `stats` null with `failed`
 * means it was never read; `stats` with `failed` means the numbers are from an earlier read.
 */
export type StatsEntry = { stats: RunStats | null; failed: boolean }

export type RunKind = 'ordinary' | 'experiment'
export type RunSort = 'updated' | 'name' | 'score' | 'cost'
export const RUN_SORTS: RunSort[] = ['updated', 'name', 'score', 'cost']

/**
 * A research experiment is a run the workspace serves (client paths are `workspace/…` or `runs/…`,
 * docs/research/workspace.md) under one of its directions; every other run is an ordinary run.
 */
export const runKind = (r: RunEntry, directions: ReadonlySet<string>): RunKind =>
  r.path.startsWith('workspace/') && directions.has(runGroup(r)) ? 'experiment' : 'ordinary'

/** What a run is listed under: its task, which for a research experiment is its direction's folder ID. */
export const runGroup = (r: RunEntry): string => r.task || r.name.split('/')[0]

/** The run's name inside its group: the experiment's path or the run directory. */
export function runLabel(r: RunEntry): string {
  const name = r.name.replace(/\/results$/, '')
  const group = `${runGroup(r)}/`
  return name.startsWith(group) ? name.slice(group.length) : name
}

/** A run's cost as every total counts it (OD-9): its programs plus prompt evolution; null when either has an unpriced entry. */
export const runCost = (s: RunStats): number | null => (s.total_cost === null || s.prompt_evo_cost === null ? null : s.total_cost + s.prompt_evo_cost)

/** A run matches a search when its task or name contains every word of it, in any case. */
export function matchesRun(r: RunEntry, query: string): boolean {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  const text = `${r.task} ${r.name}`.toLowerCase()
  return words.every((w) => text.includes(w))
}

const VALUE: Record<RunSort, (r: RunEntry, s: RunStats | undefined) => number | string | null> = {
  updated: (_r, s) => s?.last_update ?? null,
  name: (r) => runLabel(r),
  score: (_r, s) => s?.best_score ?? null,
  cost: (_r, s) => (s ? runCost(s) : null),
}

/** Names read A to Z; every number reads largest first; a run without the value sorts last. */
export function compareRuns(sort: RunSort, stats: ReadonlyMap<string, RunStats>) {
  return (a: RunEntry, b: RunEntry): number => {
    const [va, vb] = [VALUE[sort](a, stats.get(a.path)), VALUE[sort](b, stats.get(b.path))]
    if (va === null || vb === null) return va === vb ? a.name.localeCompare(b.name) : va === null ? 1 : -1
    if (typeof va === 'string' || typeof vb === 'string') return String(va).localeCompare(String(vb))
    return vb - va || a.name.localeCompare(b.name)
  }
}

export type RunGroup = { key: string; kind: RunKind; runs: RunEntry[] }

/** Runs in their groups: each group in the order of its first run under the sort, runs sorted inside it. */
export function groupRuns(runs: RunEntry[], sort: RunSort, stats: ReadonlyMap<string, RunStats>, directions: ReadonlySet<string>): RunGroup[] {
  const order = compareRuns(sort, stats)
  const groups = new Map<string, RunGroup>()
  for (const r of [...runs].sort(order)) {
    const key = runGroup(r)
    const group = groups.get(key) ?? { key, kind: runKind(r, directions), runs: [] }
    group.runs.push(r)
    groups.set(key, group)
  }
  return [...groups.values()]
}

export type RunTotals = {
  runs: number
  programs: number
  valid: number
  failed: number
  prompts: number
  /** Across evaluators, so only a reference: scores are comparable within one evaluator. */
  peak: number | null
  /** The recorded spend of the runs read; a lower bound when `partial` is above zero. */
  cost: number
  /**
   * Runs the cost covers only in part, each counted once: not read yet, unpriced, or kept from an
   * earlier read because the latest failed. A run's records only accumulate, so an earlier read is at
   * most what the run holds now.
   */
  partial: number
  runtime: number
}

/** Totals over the listed runs, from each run's latest statistics that arrived. */
export function runTotals(runs: RunEntry[], stats: ReadonlyMap<string, StatsEntry>): RunTotals {
  const t: RunTotals = { runs: runs.length, programs: 0, valid: 0, failed: 0, prompts: 0, peak: null, cost: 0, partial: 0, runtime: 0 }
  for (const r of runs) {
    const entry = stats.get(r.path)
    const s = entry?.stats
    if (!entry || !s) {
      t.partial += 1
      continue
    }
    t.programs += s.program_count
    t.valid += s.correct_count
    t.failed += s.program_count - s.correct_count
    t.prompts += s.prompt_count
    if (s.best_score !== null && (t.peak === null || s.best_score > t.peak)) t.peak = s.best_score
    // An unpriced part leaves the run's cost unknown; its recorded part still counts toward the lower bound.
    if (entry.failed || runCost(s) === null) t.partial += 1
    t.cost += (s.total_cost ?? 0) + (s.prompt_evo_cost ?? 0)
    t.runtime += s.total_runtime_seconds ?? 0
  }
  return t
}

/** A run's length the way the list shows it: minutes under an hour, otherwise hours and minutes. */
export function runtimeText(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return '—'
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  const minutes = Math.round(seconds / 60)
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`
}
