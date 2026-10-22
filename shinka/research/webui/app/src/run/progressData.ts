/**
 * The progress view's derived data (award-redesign 6.3, design.md › 四个分析视图 › 进展): one row per
 * generation for the small multiples, the noise line of an experiment run, model posteriors and the
 * per-model table, the per-metric series (L04) and the score against spend and time (L10, R01).
 * Plain functions, so node can test them directly. The hi-fi derives the same rows in
 * `hifi/extract.mjs:181-228`; every program row counts here, island copies included, while
 * `programCost` keeps copies free.
 */
import { isCorrect, isIslandCopy, programCost, recordedCost, scoreOf } from '../compare/series.ts'
import type { TrialPoint } from '../types.ts'
import { evaluationOrder, generationOf, modelOf, type Prompt, type RunProgram } from './data.ts'

const finite = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const sum = (values: number[]): number => values.reduce((a, b) => a + b, 0)
/** A total of parts that may be unknown: unknown as soon as one part is. */
const total = (values: (number | null)[]): number | null => (values.some((v) => v === null) ? null : sum(values as number[]))

function median(values: number[]): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

/** A prompt version's recorded cost (OD-9), as the comparison page charges it. */
export const promptCost = (q: Prompt): number | null => recordedCost((q.metadata?.llm as Record<string, unknown> | undefined)?.cost)

/** Seconds a program's evaluation took, as the runner records it; none when it was not recorded. */
export const computeTimeOf = (p: RunProgram): number | null => {
  const v = finite(p.metadata?.compute_time)
  return v !== null && v >= 0 ? v : null
}

export type GenerationRow = {
  generation: number
  /** Every program recorded at this generation, failures and island copies included. */
  programs: number
  failedIds: string[]
  /** The best valid score of the generation and its program (the first in evaluation order on a tie). */
  best: number | null
  bestId: string | null
  /** The best valid score so far and the program that first reached it. */
  cumulative: number | null
  cumulativeId: string | null
  median: number | null
  /** Model calls: the programs' costs less meta-analysis (OD-9); null when a program is unpriced. */
  modelCost: number | null
  metaCost: number | null
  /** The prompt versions that took effect at this generation; null when they could not be read or priced. */
  promptCost: number | null
  /** Seconds of evaluation recorded at this generation; null when a program has none. */
  computeTime: number | null
}

/**
 * One row for every generation from 0 to the last, empty ones included so the panels share one axis.
 * A failed program never counts towards best, cumulative or median, whatever score it recorded.
 */
export function generationRows(programs: RunProgram[], prompts: Prompt[], promptsKnown: boolean): GenerationRow[] {
  const last = programs.reduce((m, p) => Math.max(m, generationOf(p)), 0)
  const byGeneration = new Map<number, RunProgram[]>()
  for (const p of evaluationOrder(programs)) {
    const g = generationOf(p)
    byGeneration.set(g, [...(byGeneration.get(g) ?? []), p])
  }
  const rows: GenerationRow[] = []
  let cumulative: number | null = null
  let cumulativeId: string | null = null
  for (let g = 0; g <= last; g += 1) {
    const group = byGeneration.get(g) ?? []
    let best: number | null = null
    let bestId: string | null = null
    const valid: number[] = []
    for (const p of group) {
      const s = scoreOf(p)
      if (s === null) continue
      valid.push(s)
      if (best === null || s > best) {
        best = s
        bestId = p.id
      }
    }
    if (best !== null && (cumulative === null || best > cumulative)) {
      cumulative = best
      cumulativeId = bestId
    }
    const costs = group.map(programCost)
    const meta = total(group.map((p) => (isIslandCopy(p) ? 0 : recordedCost(p.metadata?.meta_cost))))
    const all = total(costs)
    rows.push({
      generation: g,
      programs: group.length,
      failedIds: group.filter((p) => !isCorrect(p)).map((p) => p.id),
      best,
      bestId,
      cumulative,
      cumulativeId,
      median: median(valid),
      modelCost: all === null || meta === null ? null : Math.max(0, all - meta),
      metaCost: all === null ? null : meta,
      // A prompt version without the generation it took effect at is in the run's total but in no generation (hi-fi extract.mjs:182).
      promptCost: promptsKnown ? total(prompts.filter((q) => finite(q.program_generation) === g).map(promptCost)) : null,
      computeTime: total(group.filter((p) => !isIslandCopy(p)).map(computeTimeOf)),
    })
  }
  return rows
}

/**
 * The direction's noise line along this run (design.md › 图表语法): the trial points of this run's
 * programs, and at each generation the bar of its last trial. An ordinary run has no trial points.
 */
export function noiseByGeneration(points: TrialPoint[], ids: ReadonlySet<string>): Map<number, number> {
  const out = new Map<number, number>()
  for (const t of [...points].sort((a, b) => a.n - b.n)) if (ids.has(t.program) && Number.isFinite(t.bar)) out.set(t.generation, t.bar)
  return new Map([...out].sort((a, b) => a[0] - b[0]))
}

export type CostTotals = {
  /** The run's cost as every total counts it (runCost): programs plus prompt evolution. */
  total: number | null
  api: number | null
  embed: number | null
  novelty: number | null
  meta: number | null
  prompt: number | null
  /** Recorded spend no program carries, such as proposals discarded before evaluation. */
  unattributed: number | null
  /** The cost per program that is not an island copy. */
  perProgram: number | null
}

/** R01-Q1/Q3, L10-Q1: the run's cost by part, each unknown when one of its entries is unpriced. */
export function costTotals(programs: RunProgram[], promptEvo: number | null, runTotal: number | null): CostTotals {
  const charged = programs.filter((p) => !isIslandCopy(p))
  const part = (key: string) => total(charged.map((p) => (programCost(p) === null ? null : recordedCost(p.metadata?.[key]))))
  const programsTotal = total(charged.map(programCost))
  const attributed = programsTotal === null || promptEvo === null ? null : programsTotal + promptEvo
  const rest = runTotal === null || attributed === null ? null : runTotal - attributed
  return {
    total: runTotal,
    api: part('api_costs'),
    embed: part('embed_cost'),
    novelty: part('novelty_cost'),
    meta: part('meta_cost'),
    prompt: promptEvo,
    unattributed: rest !== null && rest > 1e-9 ? rest : rest === null ? null : 0,
    perProgram: runTotal === null || charged.length === 0 ? null : runTotal / charged.length,
  }
}

export type PosteriorSeries = { models: string[]; rows: { generation: number; means: Record<string, number | null> }[] }

const nonEmpty = (v: unknown): v is Record<string, unknown> => Boolean(v) && typeof v === 'object' && Object.keys(v as object).length > 0

/**
 * Model posteriors as the restored Ensembling view reads them (hi-fi extract.mjs:206-220): the
 * programs whose LLM result carries `model_posteriors`, else those with `metadata.model_posteriors`
 * when none does; each generation averages its programs, model by model.
 */
export function posteriorSeries(programs: RunProgram[]): PosteriorSeries {
  const ofResult = (p: RunProgram) => (p.metadata?.llm_result as Record<string, unknown> | undefined)?.model_posteriors
  const fromResult = programs.filter((p) => nonEmpty(ofResult(p)))
  const holders = fromResult.length ? fromResult : programs.filter((p) => nonEmpty(p.metadata?.model_posteriors))
  const of = (p: RunProgram): Record<string, unknown> => (nonEmpty(ofResult(p)) ? (ofResult(p) as Record<string, unknown>) : (p.metadata?.model_posteriors as Record<string, unknown>) ?? {})
  const models = [...new Set(holders.flatMap((p) => Object.keys(of(p))))].sort()
  const generations = [...new Set(holders.map(generationOf))].sort((a, b) => a - b)
  return {
    models,
    rows: generations.map((generation) => {
      const group = holders.filter((p) => generationOf(p) === generation)
      return {
        generation,
        means: Object.fromEntries(
          models.map((m) => {
            const values = group.map((p) => finite(of(p)[m])).filter((v): v is number => v !== null)
            return [m, values.length ? sum(values) / values.length : null]
          }),
        ),
      }
    }),
  }
}

export type ModelRow = {
  model: string
  programs: number
  /** Valid programs over all of the model's programs. */
  success: number | null
  best: number | null
  mean: number | null
  cost: number | null
  posteriorMean: number | null
  posteriorLatest: number | null
}

/** OD-4: each model's programs, success rate, best and mean score, cost and posteriors. */
export function modelRows(programs: RunProgram[], posteriors: PosteriorSeries): ModelRow[] {
  const named = [...new Set([...programs.map(modelOf).filter((m): m is string => m !== null), ...posteriors.models])].sort()
  return named
    .map((model) => {
      const group = programs.filter((p) => modelOf(p) === model)
      const valid = group.map(scoreOf).filter((s): s is number => s !== null)
      const posts = posteriors.rows.map((r) => r.means[model] ?? null).filter((v): v is number => v !== null)
      return {
        model,
        programs: group.length,
        success: group.length ? valid.length / group.length : null,
        best: valid.length ? Math.max(...valid) : null,
        mean: valid.length ? sum(valid) / valid.length : null,
        cost: group.length ? total(group.map(programCost)) : null,
        posteriorMean: posts.length ? sum(posts) / posts.length : null,
        posteriorLatest: posts.length ? posts[posts.length - 1] : null,
      }
    })
    .filter((r) => r.programs > 0 || (r.posteriorMean ?? 0) > 0)
}

/** L04: the numeric public metrics valid programs report; private metrics never enter. */
export function metricKeys(programs: RunProgram[]): string[] {
  const keys = new Set<string>()
  for (const p of programs) {
    if (!isCorrect(p)) continue
    for (const [k, v] of Object.entries(p.public_metrics ?? {})) if (finite(v) !== null) keys.add(k)
  }
  return [...keys].sort()
}

export type MetricPoint = { id: string; generation: number; value: number }
export type MetricStat = { generation: number; max: number; min: number; cumulativeMax: number; cumulativeMin: number }

/**
 * L04-Q1/Q2 as the restored Metrics view drew them (viz_tree.html:13199-13265): each valid program's
 * value, and per generation the highest and lowest value with their running extremes.
 */
export function metricSeries(programs: RunProgram[], key: string): { points: MetricPoint[]; stats: MetricStat[] } {
  const points = evaluationOrder(programs)
    .filter(isCorrect)
    .flatMap((p) => {
      const value = finite(p.public_metrics?.[key])
      return value === null ? [] : [{ id: p.id, generation: generationOf(p), value }]
    })
  const stats: MetricStat[] = []
  let cumulativeMax = -Infinity
  let cumulativeMin = Infinity
  for (const generation of [...new Set(points.map((p) => p.generation))].sort((a, b) => a - b)) {
    const values = points.filter((p) => p.generation === generation).map((p) => p.value)
    const max = Math.max(...values)
    const min = Math.min(...values)
    cumulativeMax = Math.max(cumulativeMax, max)
    cumulativeMin = Math.min(cumulativeMin, min)
    stats.push({ generation, max, min, cumulativeMax, cumulativeMin })
  }
  return { points, stats }
}

export type TimePoint = { x: number; best: number; id: string }

/**
 * L10-Q4: the best valid score against the evaluation time spent so far, for runs whose spend is all
 * zero; in evaluation order, and it stops before the first program whose time was not recorded
 * (L10-E1). `complete` says whether every program was counted.
 */
export function byComputeTime(programs: RunProgram[]): { points: TimePoint[]; complete: boolean } {
  const points: TimePoint[] = []
  let spent = 0
  let best: number | null = null
  let id = ''
  for (const p of evaluationOrder(programs)) {
    if (isIslandCopy(p)) continue
    const t = computeTimeOf(p)
    if (t === null) return { points, complete: false }
    spent += t
    const s = scoreOf(p)
    if (s !== null && (best === null || s > best)) {
      best = s
      id = p.id
    }
    if (best !== null) points.push({ x: spent, best, id })
  }
  return { points, complete: true }
}

/**
 * R01-Q6: the best valid score against wall-clock time since the run's first program, as the restored
 * Meta view measured it (viz_tree.html:7306-7345); programs without a timestamp are left out.
 */
export function byWallTime(programs: RunProgram[]): { points: TimePoint[]; scores: { x: number; score: number; id: string }[] } {
  const timed = programs.filter((p) => finite(p.timestamp) !== null).sort((a, b) => (a.timestamp as number) - (b.timestamp as number))
  if (timed.length === 0) return { points: [], scores: [] }
  const start = timed[0].timestamp as number
  const points: TimePoint[] = []
  const scores: { x: number; score: number; id: string }[] = []
  let best: number | null = null
  let id = ''
  for (const p of timed) {
    const s = scoreOf(p)
    if (s === null) continue
    const x = (p.timestamp as number) - start
    scores.push({ x, score: s, id: p.id })
    if (best === null || s > best) {
      best = s
      id = p.id
    }
    points.push({ x, best, id })
  }
  return { points, scores }
}
