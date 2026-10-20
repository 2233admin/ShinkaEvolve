/**
 * The compare page's derived data (award-redesign 5.2, P02). The curves follow the legacy compare page
 * (analysis/compare.html) point for point; costs follow rundb.run_cost (OD-9). Plain functions with
 * type-only imports, so node can test them directly.
 */
import type { RunStats } from '../runs/data'

export type Metric = 'best_score' | 'program_count' | 'correct_count'
export const METRICS: Metric[] = ['best_score', 'program_count', 'correct_count']

/** The URL's `metric`; anything else is the default, the cumulative best. */
export const metricOf = (value: string): Metric => METRICS.find((m) => m === value) ?? 'best_score'

/** One program as `/get_programs_summary` returns it, with the fields the comparison reads. */
export type ProgramSummary = {
  id: string
  generation?: number | null
  timestamp?: number | null
  correct?: boolean | number | null
  combined_score?: number | null
  metadata?: Record<string, unknown> | null
}

/** One prompt version as `/get_system_prompts` returns it. */
export type PromptVersion = {
  timestamp?: number | null
  program_generation?: number | null
  metadata?: { llm?: { cost?: unknown } | null } | null
}

export type RunData = { programs: ProgramSummary[]; stats: RunStats; prompts: PromptVersion[] }

export type Point = { x: number; y: number }

/**
 * An absent or null cost field is a recorded zero (the runner writes a field only for the steps that
 * ran); a present one must be an amount, otherwise the cost is unknown.
 */
export const recordedCost = (value: unknown): number | null =>
  value === undefined || value === null ? 0 : typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null

/** Island copies repeat their source's metadata: they are neither charges nor programs of their own. */
export const isIslandCopy = (p: ProgramSummary): boolean => Boolean(p.metadata?._is_island_copy || p.metadata?._spawned_island)

/** A program's cost under rundb.run_cost; an unpriced headless call has no known cost. */
export function programCost(p: ProgramSummary): number | null {
  const md = p.metadata ?? {}
  if (isIslandCopy(p)) return 0
  const llm = (md.llm_result ?? {}) as Record<string, unknown>
  const model = String(md.model_name || md.model || llm.model_name || llm.model || '')
  if (model.startsWith('headless/') && (md.headless_pricing_unknown === true || md.headless_pricing_unknown === 1)) return null
  const values = ['api_costs', 'embed_cost', 'novelty_cost', 'meta_cost'].map((key) => recordedCost(md[key]))
  return values.some((v) => v === null) ? null : values.reduce((sum: number, v) => sum + (v as number), 0)
}

/** A valid score: a correct program's finite combined_score. */
export const scoreOf = (p: ProgramSummary): number | null =>
  p.correct && typeof p.combined_score === 'number' && Number.isFinite(p.combined_score) ? p.combined_score : null

type Tally = { best: number | null; programs: number; correct: number }

const tally = (): Tally => ({ best: null, programs: 0, correct: 0 })

function count(t: Tally, p: ProgramSummary) {
  t.programs += 1
  if (p.correct) t.correct += 1
  const score = scoreOf(p)
  if (score !== null) t.best = t.best === null ? score : Math.max(t.best, score)
}

const valueOf = (t: Tally, metric: Metric): number | null =>
  metric === 'best_score' ? t.best : metric === 'program_count' ? t.programs : t.correct

/** P02-Q1: the metric at the end of every generation, in generation order; a program without one is generation 0. */
export function byGeneration(programs: ProgramSummary[], metric: Metric): Point[] {
  const generations = new Map<number, ProgramSummary[]>()
  for (const p of programs) {
    const g = p.generation || 0
    const list = generations.get(g)
    if (list) list.push(p)
    else generations.set(g, [p])
  }
  const t = tally()
  const points: Point[] = []
  for (const g of [...generations.keys()].sort((a, b) => a - b)) {
    for (const p of generations.get(g)!) count(t, p)
    const y = valueOf(t, metric)
    if (y !== null) points.push({ x: g, y })
  }
  return points
}

/** Above this many points the spend curve keeps every k-th point and the last (P02-Q2). */
export const MAX_POINTS = 200

/**
 * P02-Q2: the metric against cumulative spend. Programs and prompt versions are charged in one clock
 * order, Unix time when every one of them has a time and generation otherwise; a prompt charged after
 * the last program still moves the curve's end to the run's total. From the first charge without a
 * price the spend is unknown, so the curve ends there.
 */
export function bySpend(run: Pick<RunData, 'programs' | 'prompts'>, metric: Metric): Point[] {
  const { programs, prompts } = run
  const byTime = programs.every((p) => Number.isFinite(p.timestamp)) && prompts.every((v) => Number.isFinite(v.timestamp))
  const charges = [
    ...programs.map((p) => ({ time: byTime ? (p.timestamp as number) : p.generation || 0, cost: programCost(p), program: p as ProgramSummary | null })),
    ...prompts.map((v) => ({ time: byTime ? v.timestamp || 0 : v.program_generation || 0, cost: recordedCost(v.metadata?.llm?.cost), program: null })),
  ].sort((a, b) => a.time - b.time)
  const t = tally()
  let spent: number | null = 0
  let value: number | null = null
  const points: { x: number | null; y: number | null }[] = []
  for (const { cost, program } of charges) {
    spent = spent === null || cost === null ? null : spent + cost
    if (!program) continue
    count(t, program)
    value = valueOf(t, metric)
    points.push({ x: spent, y: value })
  }
  if (points.length && points[points.length - 1].x !== spent) points.push({ x: spent, y: value })
  const step = Math.ceil(points.length / MAX_POINTS)
  const sampled = points.length > MAX_POINTS ? points.filter((_, i) => i % step === 0 || i === points.length - 1) : points
  return sampled.flatMap((p) => (p.x === null || p.y === null ? [] : [{ x: p.x, y: p.y }]))
}

/** A step curve's value at x: its last point at or before x; null before the curve starts or after it ends. */
export function valueAt(points: Point[], at: number): number | null {
  if (points.length === 0 || at < points[0].x || at > points[points.length - 1].x) return null
  let value = points[0].y
  for (const p of points) {
    if (p.x > at) break
    value = p.y
  }
  return value
}

export type BoardEntry = { run: number; program: ProgramSummary; score: number; cost: number | null }

/** A run's best programs that enter the leaderboard (hi-fi extract.mjs › compare `top`). */
export const PER_RUN = 4

/**
 * The best valid programs across the runs, best first, as the hi-fi ranks them: each run's best
 * `PER_RUN`, then the best `n` of those, so one run cannot fill the board.
 */
export function crossBoard(runs: ProgramSummary[][], n = 8): BoardEntry[] {
  const best = (a: BoardEntry, b: BoardEntry) => b.score - a.score
  return runs
    .flatMap((programs, run) =>
      programs
        .flatMap((program) => {
          const score = scoreOf(program)
          return score === null ? [] : [{ run, program, score, cost: programCost(program) }]
        })
        .sort(best)
        .slice(0, PER_RUN),
    )
    .sort(best)
    .slice(0, n)
}

/**
 * Which runs lead a row, judged at the shown precision: the best shown value leads, runs tied there lead
 * together, and a row where every run shows the same value has no lead. A run without the value never
 * leads; where `unknownMayLead`, an unknown value could be the better one, so its row has no lead.
 */
export function leads(values: (number | null)[], better: 'max' | 'min', format: (v: number) => string, unknownMayLead = false): boolean[] {
  const known = values.filter((v): v is number => v !== null)
  if (known.length === 0 || (unknownMayLead && known.length < values.length)) return values.map(() => false)
  const target = format(better === 'max' ? Math.max(...known) : Math.min(...known))
  const shown = values.map((v) => (v === null ? null : format(v)))
  const tied = new Set(shown).size === 1
  return shown.map((s) => !tied && s === target)
}

/**
 * Scores compare only within a task (the run list's caveat): the runs' places in the selection by
 * group, groups in the order of their first run, so A, B, A is [[0, 2], [1]].
 */
export function byGroup(keys: string[]): number[][] {
  const groups = new Map<string, number[]>()
  keys.forEach((key, i) => {
    const list = groups.get(key)
    if (list) list.push(i)
    else groups.set(key, [i])
  })
  return [...groups.values()]
}

/**
 * A run-level row's own scale: its values with room on both sides. Counts, rates and costs cannot
 * fall below zero, so their scale stops there; a score can be negative and keeps its range.
 */
export function stripDomain(values: number[]): [number, number] {
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  const pad = (hi - lo) * 0.3 || Math.abs(hi) * 0.1 || 1
  return [lo >= 0 ? Math.max(0, lo - pad) : lo - pad, hi + pad]
}

/** Each group's leaderboard, its entries naming runs by their place in the whole selection. */
export function groupBoards(runs: ProgramSummary[][], groups: number[][], n = 8): BoardEntry[][] {
  return groups.map((group) => crossBoard(group.map((i) => runs[i]), n).map((e) => ({ ...e, run: group[e.run] })))
}

/**
 * A row's leads judged within each group, by run in the selection's order: true leads, false trails,
 * null where the run is alone in its group and nothing is judged.
 */
export function groupLeads(values: (number | null)[], groups: number[][], better: 'max' | 'min', format: (v: number) => string, unknownMayLead = false): (boolean | null)[] {
  const out = values.map((): boolean | null => null)
  for (const group of groups) {
    if (group.length < 2) continue
    const lead = leads(group.map((i) => values[i]), better, format, unknownMayLead)
    group.forEach((i, k) => {
      out[i] = lead[k]
    })
  }
  return out
}

/** Line patterns: solid, dashed, dotted (the hi-fi's three). */
export const DASHES = ['', '6 4', '1.5 3']
export const MARK_COUNT = 6

/** A run's shape (0–5), line pattern, and whether both are drawn in the muted ink. */
export type SeriesStyle = { mark: number; dash: string; muted: boolean }

/**
 * Run k's shape and line: six shapes by three patterns, so eighteen runs read apart before a style
 * repeats; the first three are the hi-fi's circle–solid, square–dashed, triangle–dotted.
 */
export const seriesStyle = (k: number): SeriesStyle => {
  const mark = k % MARK_COUNT
  return { mark, dash: DASHES[(k + Math.floor(k / MARK_COUNT)) % DASHES.length], muted: mark === 2 || mark === 5 }
}

/** End-label positions: in the order of their curves' ends, at least `gap` apart, kept within [top, bottom]. */
export function stackLabels(ys: number[], gap: number, top: number, bottom: number): number[] {
  const order = ys.map((y, i) => [y, i] as const).sort((a, b) => a[0] - b[0])
  const at = new Array<number>(ys.length)
  let last = -Infinity
  for (const [y, i] of order) {
    at[i] = Math.max(y, last + gap, top)
    last = at[i]
  }
  // Labels pushed past the bottom move back up, keeping their gaps.
  let next = Infinity
  for (let k = order.length - 1; k >= 0; k -= 1) {
    const i = order[k][1]
    at[i] = Math.min(at[i], bottom, next - gap)
    next = at[i]
  }
  return at
}
