/**
 * The progress view's generation trends (award-redesign 6.3), ported from the research copy of the
 * legacy tree page (analysis/viz_tree.html): the running correctness and improvement rates, overall
 * and by patch type, and the lines of code per generation with their linear trend. Plain functions,
 * so node can test them directly. An empty series is the one "no data" signal, as the legacy plots'
 * empty message was.
 */
import { isCorrect } from '../compare/series.ts'
import { generationOf, patchTypeOf, type RunProgram } from './data.ts'

/** The patch types the legacy rate plots draw a line for, in their drawing order (VT:14394, 16077-16083). */
export const PATCH_TYPES = ['diff', 'full', 'cross', 'init', 'fix'] as const

/** A running rate after one counted program: the percentage so far (0–100) and its denominator. */
export type RatePoint = { generation: number; value: number; n: number }
export type TypeSeries = { type: string; points: RatePoint[] }
/** The overall line and one line per patch type that has a point, in `PATCH_TYPES` order. */
export type RateSeries = { overall: RatePoint[]; byType: TypeSeries[] }

type Step = { generation: number; type: string | null; hit: boolean; weight: number }

/**
 * The legacy plots' walk order (VT:14384-14392, 14460-14468): by generation, ties by
 * `metadata.timestamp` (absent or falsy as 0), remaining ties in input order. The difference is taken
 * as the legacy did, so numeric strings compare as numbers and anything else ties.
 */
function walkOrder(programs: RunProgram[]): RunProgram[] {
  return [...programs].sort((a, b) => {
    const ga = generationOf(a)
    const gb = generationOf(b)
    if (ga !== gb) return ga - gb
    const d = Number(a.metadata?.timestamp || 0) - Number(b.metadata?.timestamp || 0)
    return Number.isNaN(d) ? 0 : d
  })
}

/** Cumulative rates from the run's start, one point per step: no window and no smoothing. */
function running(steps: Step[]): RateSeries {
  const overall: RatePoint[] = []
  const tallies: Record<string, { hits: number; n: number; points: RatePoint[] }> = Object.fromEntries(PATCH_TYPES.map((t) => [t, { hits: 0, n: 0, points: [] }]))
  let hits = 0
  let n = 0
  for (const s of steps) {
    n += s.weight
    if (s.hit) hits++
    overall.push({ generation: s.generation, value: (hits / n) * 100, n })
    if (s.type === null || !(PATCH_TYPES as readonly string[]).includes(s.type)) continue
    const t = tallies[s.type]
    t.n += s.weight
    if (s.hit) t.hits++
    t.points.push({ generation: s.generation, value: (t.hits / t.n) * 100, n: t.n })
  }
  const byType = PATCH_TYPES.flatMap((type): TypeSeries[] => {
    const points = tallies[type].points
    return points.length > 0 ? [{ type, points }] : []
  })
  return { overall, byType }
}

/**
 * The running correctness rate as computeCorrectnessRateStats (VT:14378-14451) and
 * renderCorrectnessRatePlot (VT:16054-16117) drew it: every program in walk order adds one to the
 * denominator and, when it passed, one to the count; the point after it sits at its generation, so a
 * generation holds one point per program. A patch type outside `PATCH_TYPES` counts only overall.
 * Empty when there are no programs.
 */
export function correctnessRates(programs: RunProgram[]): RateSeries {
  return running(walkOrder(programs).map((p) => ({ generation: generationOf(p), type: patchTypeOf(p), hit: isCorrect(p), weight: 1 })))
}

const score = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

/** Attempts a program took (`metadata.patch_attempt`, 1-indexed); 1 when absent or not a positive number. */
const attemptsOf = (p: RunProgram): number => {
  const a = p.metadata?.patch_attempt
  return typeof a === 'number' && Number.isFinite(a) && a > 0 ? a : 1
}

/**
 * The running improvement rate as computeImprovementRateStats (VT:14453-14548) and
 * renderImprovementRatePlot (VT:16119-16182) drew it: a program counts when its parent is in the run
 * and both carry a score; it adds its patch attempts to the denominator and one to the count when it
 * passed and beat its parent's recorded score (the parent's own correctness is not asked). Points per
 * program in walk order, as for correctness. Empty when no program counts.
 */
export function improvementRates(programs: RunProgram[]): RateSeries {
  // As the legacy Map: a repeated id resolves to its last program.
  const byId = new Map(programs.map((p) => [p.id, p]))
  const steps = walkOrder(programs).flatMap((p): Step[] => {
    const parent = p.parent_id ? byId.get(p.parent_id) : undefined
    if (!parent) return []
    const own = score(p.combined_score)
    const theirs = score(parent.combined_score)
    if (own === null || theirs === null) return []
    return [{ generation: generationOf(p), type: patchTypeOf(p), hit: isCorrect(p) && own > theirs, weight: attemptsOf(p) }]
  })
  return running(steps)
}

/** One generation's mean lines of code and logical lines of code over the `n` programs that report them. */
export type BloatPoint = { generation: number; loc: number; lloc: number; n: number }
/** A least-squares line, y = slope · generation + intercept. */
export type Trend = { slope: number; intercept: number }
export type CodeBloat = { points: BloatPoint[]; loc: Trend | null; lloc: Trend | null }

/**
 * The least-squares line through the points (linearRegression, VT:14551-14562); none for fewer than
 * two distinct x, where the legacy got NaN and drew no line.
 */
export function linearFit(xs: number[], ys: number[]): Trend | null {
  const n = xs.length
  const sumX = xs.reduce((a, b) => a + b, 0)
  const sumY = ys.reduce((a, b) => a + b, 0)
  const sumXY = xs.reduce((s, x, i) => s + x * ys[i], 0)
  const sumXX = xs.reduce((s, x) => s + x * x, 0)
  const denominator = n * sumXX - sumX * sumX
  if (n === 0 || denominator === 0) return null
  const slope = (n * sumXY - sumX * sumY) / denominator
  return { slope, intercept: (sumY - slope * sumX) / n }
}

/**
 * Lines of code across generations as computeCodeBloatStats (VT:14283-14322) and renderCodeBloatPlot
 * (VT:15848-15932) drew them: programs whose `metadata.code_analysis_metrics` has a numeric
 * `lines_of_code`; logical lines fall back to it when absent or zero. Each generation averages its
 * programs, generations ascending, and each measure gets a linear trend through the generation means.
 * Empty points when no program reports code metrics.
 */
export function codeBloat(programs: RunProgram[]): CodeBloat {
  const groups = new Map<number, { loc: number; lloc: number; n: number }>()
  for (const p of programs) {
    const m = p.metadata?.code_analysis_metrics
    if (!m || typeof m !== 'object') continue
    const metrics = m as Record<string, unknown>
    const loc = score(metrics.lines_of_code)
    if (loc === null) continue
    const lloc = score(metrics.logical_lines_of_code) || loc
    const g = generationOf(p)
    const group = groups.get(g) ?? { loc: 0, lloc: 0, n: 0 }
    group.loc += loc
    group.lloc += lloc
    group.n++
    groups.set(g, group)
  }
  const points = [...groups.entries()]
    .sort(([a], [b]) => a - b)
    .map(([generation, g]): BloatPoint => ({ generation, loc: g.loc / g.n, lloc: g.lloc / g.n, n: g.n }))
  const xs = points.map((d) => d.generation)
  return {
    points,
    loc: linearFit(xs, points.map((d) => d.loc)),
    lloc: linearFit(xs, points.map((d) => d.lloc)),
  }
}
