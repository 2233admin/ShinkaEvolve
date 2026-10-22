/**
 * The progress view's success and improvement statistics by model and by patch type (award-redesign
 * 6.3), ported from the analysis tab of the research tree page (analysis/viz_tree.html, "VT"). Every
 * program with a parent is one successful proposal; the attempt counter the runner recorded says how
 * many tries it took, and a child improved when its score is above its parent's. Groups keep the order
 * in which the programs first name them, as the legacy charts did. Plain functions, so node can test
 * them directly.
 */
import { isCorrect } from '../compare/series.ts'
import { modelOf, patchTypeOf, type RunProgram } from './data.ts'

/** The counts behind one heatmap cell or one patch type's bars, and the rates the charts plot. */
export type Tally = {
  /** Tries the group's programs took (legacy `totalAttempts`). */
  attempts: number
  /** Programs in the group: each exists, so each is a success (legacy `successes`). */
  successes: number
  /** Programs scoring above their parent (legacy `improvements`). */
  improved: number
  /** Programs that passed evaluation (legacy `correctPrograms`). */
  correct: number
  /** Programs that passed evaluation and scored above their parent (legacy `improvementsFromCorrect`). */
  improvedCorrect: number
  /** successes / attempts, the performance heatmaps' value; null without attempts (the legacy drew 0%). */
  successRate: number | null
  /** improvedCorrect / attempts, the improvement heatmap's and bars' value; null without attempts (the legacy drew 0%). */
  improvementRate: number | null
}

/** One model × patch type cell; a pair no program had has zero counts and null rates. */
export type ModelCell = Tally & { patchType: string }

/** One model's row: one cell per column of the matrix's `patchTypes`, in that order. */
export type ModelRow = { model: string; cells: ModelCell[] }

/** A model × patch type matrix: rows in first-seen model order, columns in first-seen patch type order across the rows. */
export type ModelStats = { patchTypes: string[]; rows: ModelRow[] }

/** One patch type's bars: successes stacked on the tries that failed before them, and its improvement rate. */
export type PatchTypeStat = Tally & {
  patchType: string
  /** Tries that failed before each program was written: attempts − 1 per program (legacy `failures`). */
  failures: number
}

/** A program without a recorded model or patch type is grouped here, as the legacy did. */
export const UNKNOWN = 'unknown'

type Meta = Record<string, unknown>

/** The legacy's `metadata.x || 1` for an attempt counter, reading only positive amounts. */
const counter = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : 1)

/** The legacy's `!x || x <= 0` filter: only a positive counter marks a novelty or resample attempt. */
const positive = (v: unknown): boolean => typeof v === 'number' && Number.isFinite(v) && v > 0

const recorded = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

/** The legacy's `node.combined_score > parent.combined_score`, between recorded scores; correctness is not required. */
function improves(child: RunProgram, parent: RunProgram | undefined): boolean {
  const c = recorded(child.combined_score)
  const p = parent ? recorded(parent.combined_score) : null
  return c !== null && p !== null && c > p
}

const rate = (part: number, attempts: number): number | null => (attempts > 0 ? part / attempts : null)

type Counts = Omit<Tally, 'successRate' | 'improvementRate'>
const counts = (): Counts => ({ attempts: 0, successes: 0, improved: 0, correct: 0, improvedCorrect: 0 })
const withRates = (c: Counts): Tally => ({ ...c, successRate: rate(c.successes, c.attempts), improvementRate: rate(c.improvedCorrect, c.attempts) })

function count(c: Counts, attempts: number, improved: boolean, correct: boolean) {
  c.attempts += attempts
  c.successes++
  if (improved) c.improved++
  if (correct) {
    c.correct++
    if (improved) c.improvedCorrect++
  }
}

/** Which children a variant counts and how many tries each took; null skips the program. */
type Attempts = (md: Meta) => number | null

const patchAttempts: Attempts = (md) => counter(md.patch_attempt)
const noveltyAttempts: Attempts = (md) => (positive(md.novelty_attempt) ? counter(md.novelty_attempt) : null)
const resampleAttempts: Attempts = (md) => (positive(md.resample_attempt) ? counter(md.resample_attempt) : null)

function byModel(programs: RunProgram[], attemptsOf: Attempts): ModelStats | null {
  if (programs.length === 0) return null
  const byId = new Map(programs.map((p) => [p.id, p]))
  const rows = new Map<string, Map<string, Counts>>()
  for (const p of programs) {
    if (!p.parent_id) continue
    const attempts = attemptsOf(p.metadata ?? {})
    if (attempts === null) continue
    const parent = byId.get(p.parent_id)
    if (!parent) continue
    const model = modelOf(p) ?? UNKNOWN
    const patchType = patchTypeOf(p) ?? UNKNOWN
    let cells = rows.get(model)
    if (!cells) rows.set(model, (cells = new Map()))
    let cell = cells.get(patchType)
    if (!cell) cells.set(patchType, (cell = counts()))
    count(cell, attempts, improves(p, parent), isCorrect(p))
  }
  const patchTypes = [...new Set([...rows.values()].flatMap((cells) => [...cells.keys()]))]
  return {
    patchTypes,
    rows: [...rows].map(([model, cells]) => ({
      model,
      cells: patchTypes.map((patchType) => ({ patchType, ...withRates(cells.get(patchType) ?? counts()) })),
    })),
  }
}

function byPatchType(programs: RunProgram[], attemptsOf: Attempts): PatchTypeStat[] | null {
  if (programs.length === 0) return null
  const byId = new Map(programs.map((p) => [p.id, p]))
  const stats = new Map<string, Counts & { failures: number }>()
  for (const p of programs) {
    if (!p.parent_id) continue
    const attempts = attemptsOf(p.metadata ?? {})
    if (attempts === null) continue
    const patchType = patchTypeOf(p) ?? UNKNOWN
    let stat = stats.get(patchType)
    if (!stat) stats.set(patchType, (stat = { ...counts(), failures: 0 }))
    count(stat, attempts, improves(p, byId.get(p.parent_id)), isCorrect(p))
    stat.failures += attempts - 1
  }
  return [...stats].map(([patchType, { failures, ...c }]) => ({ patchType, ...withRates(c), failures }))
}

/**
 * Success and improvement by model and patch type over every child program (VT:13805-13884
 * `computeLLMStats`), as the performance heatmap (VT:14857-14986, `successRate`) and the improvement
 * heatmap (VT:14988-15117, `improvementRate`) draw it. A program counts when its parent is among the
 * programs; its tries are `patch_attempt`, else 1. Null for no programs (the legacy's "no data" state);
 * no rows when no program's parent is present (its "no model data" state).
 */
export const llmStats = (programs: RunProgram[]): ModelStats | null => byModel(programs, patchAttempts)

/**
 * Success, failed tries and improvement by patch type over every child program (VT:13886-13943
 * `computeMutationStats`), as the stacked success/failure bars (VT:15119-15172) and the improvement-rate
 * bars (VT:15174-15228, `improvementRate`) draw it. A child counts even when its parent is not among
 * the programs (it then never improves); its tries are `patch_attempt`, else 1. Null for no programs;
 * an empty list when no program has a parent (the legacy's "no patch type data" state).
 */
export const mutationStats = (programs: RunProgram[]): PatchTypeStat[] | null => byPatchType(programs, patchAttempts)

/**
 * `llmStats` over the novelty attempts only (VT:13945-14020 `computeNoveltyLLMStats`, drawn as a
 * success-rate heatmap by VT:15230-15354): children with a positive `novelty_attempt`, which is also
 * their tries. Null for no programs; no rows when none qualifies.
 */
export const noveltyLlmStats = (programs: RunProgram[]): ModelStats | null => byModel(programs, noveltyAttempts)

/**
 * `mutationStats` over the novelty attempts only (VT:14022-14080 `computeNoveltyMutationStats`, drawn
 * as stacked success/failure bars by VT:15483-15539): tries are `novelty_attempt`. Null for no
 * programs; an empty list when none qualifies.
 */
export const noveltyMutationStats = (programs: RunProgram[]): PatchTypeStat[] | null => byPatchType(programs, noveltyAttempts)

/**
 * `llmStats` over the resample attempts only (VT:14082-14157 `computeResampleLLMStats`, drawn as a
 * success-rate heatmap by VT:15601-15725): children with a positive `resample_attempt`, which is also
 * their tries. Null for no programs; no rows when none qualifies.
 */
export const resampleLlmStats = (programs: RunProgram[]): ModelStats | null => byModel(programs, resampleAttempts)

/**
 * `mutationStats` over the resample attempts only (VT:14159-14218 `computeResampleMutationStats`, drawn
 * as stacked success/failure bars by VT:15727-15783): tries are `resample_attempt`. Null for no
 * programs; an empty list when none qualifies.
 */
export const resampleMutationStats = (programs: RunProgram[]): PatchTypeStat[] | null => byPatchType(programs, resampleAttempts)

/**
 * The model heatmaps' row labels (VT:14887-14918, repeated by every model heatmap): the models' names
 * without the prefix all of them share, unless dropping it would leave some name shorter than three
 * characters or starting with a character that is not a letter or digit, in which case the full names.
 * Shortening long labels is left to the view.
 */
export function modelLabels(models: string[]): string[] {
  let prefix = ''
  if (models.length > 1) {
    const sorted = [...models].sort()
    const first = sorted[0]
    const last = sorted[sorted.length - 1]
    let i = 0
    while (i < first.length && first[i] === last[i]) i++
    prefix = first.slice(0, i)
  }
  if (models.some((m) => m.length - prefix.length < 3 || /^[^a-zA-Z0-9]/.test(m.slice(prefix.length)))) prefix = ''
  return models.map((m) => m.slice(prefix.length))
}
