/**
 * The progress view's cost charts (award-redesign 6.3): cost by model × patch type and the running
 * average cost across generations, ported from the legacy tree page (analysis/viz_tree.html,
 * `computeCostStats` VT:14564-14792 as its two cost renderers read it). Every program's cost is
 * `programCost`: island copies are free but still counted, and an unpriced program makes every total
 * it enters unknown (null) rather than smaller. Plain functions, so node can test them directly.
 */
import { programCost } from '../compare/series.ts'
import { generationOf, modelOf, patchTypeOf, type RunProgram } from './data.ts'

/** The legacy page's key for a missing model or patch type; it never leaves this module. */
const MISSING = 'N/A'

/** A running cost total: null once any part of it is unknown. */
const addCost = (total: number | null, cost: number | null): number | null => (total === null || cost === null ? null : total + cost)

type Tally = { total: number | null; count: number }

type Keyed = { model: string; patch: string; generation: number; cost: number | null }

/** Each program's model, patch type, generation and cost, with the legacy page's missing-value key (VT:14576-14605). */
const keyed = (programs: RunProgram[]): Keyed[] =>
  programs.map((p) => ({ model: modelOf(p) ?? MISSING, patch: patchTypeOf(p) ?? MISSING, generation: generationOf(p), cost: programCost(p) }))

const shown = (key: string): string | null => (key === MISSING ? null : key)

/**
 * Average cost per program by model (rows) × patch type (cols). `cells[i][j]` is the mean cost of
 * the programs of `rows[i]` with patch `cols[j]`: null where none exist (`counts[i][j]` is 0) or where
 * one of them is unpriced (`counts` > 0). A null col is the programs without a patch type.
 */
export type ModelPatchCost = { rows: string[]; cols: (string | null)[]; cells: (number | null)[][]; counts: number[][] }

/**
 * The model × patch type heatmap: `computeCostStats` VT:14582-14614 and 14681-14699 (rows and cols
 * in code-unit order, mean = total / count, count includes failed, free and unpriced programs),
 * reduced as `renderModelPatchCostHeatmap` VT:16314-16352 shows it: the `init` column and the row of
 * programs without a model are dropped; the column of programs without a patch type stays. No
 * programs gives no rows and no cols (the legacy page's "no cost data" state, VT:16297-16302).
 */
export function modelPatchCost(programs: RunProgram[]): ModelPatchCost {
  const rows = keyed(programs)
  const tallies = new Map<string, Tally>()
  for (const r of rows) {
    const key = `${r.model}||${r.patch}`
    const t = tallies.get(key) ?? { total: 0, count: 0 }
    tallies.set(key, { total: addCost(t.total, r.cost), count: t.count + 1 })
  }
  const models = [...new Set(rows.map((r) => r.model))].sort().filter((m) => m !== MISSING)
  const patches = [...new Set(rows.map((r) => r.patch))].sort().filter((pt) => pt !== 'init')
  return {
    rows: models,
    cols: patches.map(shown),
    cells: models.map((m) =>
      patches.map((pt) => {
        const t = tallies.get(`${m}||${pt}`)
        return t && t.total !== null ? t.total / t.count : null
      }),
    ),
    counts: models.map((m) => patches.map((pt) => tallies.get(`${m}||${pt}`)?.count ?? 0)),
  }
}

/**
 * The heatmap's row labels (`renderModelPatchCostHeatmap` VT:16322-16347): with more than one model,
 * the longest common prefix ending in '-', '_' or '/' is cut from every name, unless that leaves a
 * label empty or starting with a digit or '.', in which case the names stay whole. The legacy cut
 * of labels over 30 characters to '…' is left to the view.
 */
export function modelLabels(models: string[]): string[] {
  let prefix = ''
  if (models.length > 1) {
    const sorted = [...models].sort()
    const first = sorted[0]
    const last = sorted[sorted.length - 1]
    for (let i = 0; i < first.length && first[i] === last[i]; i++) {
      if (first[i] === '-' || first[i] === '_' || first[i] === '/') prefix = first.slice(0, i + 1)
    }
    if (models.some((m) => /^$|^[\d.]/.test(m.slice(prefix.length)))) prefix = ''
  }
  return models.map((m) => m.slice(prefix.length) || m)
}

/**
 * One generation of the whole run: its programs (`n`) and their mean cost, and the running mean over
 * every program up to and including it (`runningN`). A mean is null once an unpriced program enters it;
 * a running mean stays null from then on.
 */
export type GenerationCost = { generation: number; n: number; mean: number | null; runningN: number; runningMean: number | null }

/**
 * One model × patch type's running mean at every generation of the run: null before its first
 * program and from its first unpriced program on; `n` is its programs so far. `final` is the last
 * point's value. A null model or patch type is the programs without one.
 */
export type ComboCost = {
  model: string | null
  patchType: string | null
  points: { generation: number; value: number | null; n: number }[]
  final: number | null
}

export type GenerationAvgCost = { overall: GenerationCost[]; combos: ComboCost[] }

/**
 * The average cost across generations (`renderGenerationAvgCostPlot` VT:16405-16492 over
 * `computeCostStats` VT:14650-14672, 14716-14729 and 14761-14779). `overall` is every generation in
 * order with its own mean (`generationAvgCosts`, which the legacy page only tested for emptiness) and
 * the running mean it plotted as "Overall". `combos` are the model × patch type running means aligned
 * to all generations, `init` excluded, highest final mean first, unknown finals last, ties in the
 * order of their `model||patch` key by `localeCompare`. No programs gives no generations and no combos
 * (the legacy page's "no generation-level cost data" state, VT:16409-16412).
 */
export function generationAvgCost(programs: RunProgram[]): GenerationAvgCost {
  const rows = keyed(programs)
  const byGen = new Map<number, Tally>()
  const byCombo = new Map<string, { model: string; patch: string; gens: Map<number, Tally> }>()
  for (const r of rows) {
    const g = byGen.get(r.generation) ?? { total: 0, count: 0 }
    byGen.set(r.generation, { total: addCost(g.total, r.cost), count: g.count + 1 })
    const key = `${r.model}||${r.patch}`
    const combo = byCombo.get(key) ?? { model: r.model, patch: r.patch, gens: new Map<number, Tally>() }
    const c = combo.gens.get(r.generation) ?? { total: 0, count: 0 }
    combo.gens.set(r.generation, { total: addCost(c.total, r.cost), count: c.count + 1 })
    byCombo.set(key, combo)
  }
  const generations = [...byGen.keys()].sort((a, b) => a - b)

  let total: number | null = 0
  let count = 0
  const overall = generations.map((generation): GenerationCost => {
    const g = byGen.get(generation) as Tally
    total = addCost(total, g.total)
    count += g.count
    return { generation, n: g.count, mean: g.total === null ? null : g.total / g.count, runningN: count, runningMean: total === null ? null : total / count }
  })

  const combos = [...byCombo.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .filter(([, c]) => c.patch !== 'init')
    .map(([, c]): ComboCost => {
      let sum: number | null = 0
      let n = 0
      const points = generations.map((generation) => {
        const t = c.gens.get(generation)
        if (t) {
          sum = addCost(sum, t.total)
          n += t.count
        }
        return { generation, value: n > 0 && sum !== null ? sum / n : null, n }
      })
      return { model: shown(c.model), patchType: shown(c.patch), points, final: points.at(-1)?.value ?? null }
    })
    .sort((a, b) => (a.final === null ? (b.final === null ? 0 : 1) : b.final === null ? -1 : b.final - a.final))
  return { overall, combos }
}
