/**
 * The direction report's derived data (award-redesign 4.2). Plain functions with type-only imports,
 * so node can test them directly.
 */
import { bestIs, oosOf, STATUS_ORDER, trialsOf } from '../format.ts'
import { parseRoute, routeHref } from '../shell/route.ts'
import type { Experiment, Hypothesis, SnapshotEvent, TrialPoint } from '../types'

/** The text under a `## Heading` of a hypothesis body, blockquote markers dropped, as one line; '' when absent. */
export function bodySection(body: string, heading: string): string {
  const lines = body.split('\n')
  const want = `## ${heading}`.toLowerCase()
  const start = lines.findIndex((line) => line.replace(/^>\s*/, '').trim().toLowerCase() === want)
  if (start === -1) return ''
  const out: string[] = []
  for (const line of lines.slice(start + 1)) {
    if (/^>?\s*#/.test(line)) break
    if (line.trim()) out.push(line.replace(/^>\s*/, '').trim())
  }
  return out.join(' ')
}

/**
 * The program an experiment's link opens: the one its skeptic check judged, or the one the verdict
 * names for it, so the link shows what the verdict rests on; before any check, its highest in-sample
 * score, where a failed program never counts.
 */
export function experimentProgram(h: Hypothesis, experiment: Experiment, points: TrialPoint[]): string | null {
  const verdict = h.verdict?.evidence
  const judged = experiment.skeptic?.program_id ?? (verdict?.experiment === experiment.id ? verdict.program_id : null)
  if (typeof judged === 'string') return judged
  let best: TrialPoint | null = null
  for (const p of points) {
    if (p.hypothesis !== h.id || p.experiment !== experiment.id || p.score === null) continue
    if (best === null || p.score > (best.score as number)) best = p
  }
  return best?.program ?? null
}

/** An experiment's analysis with one program open on its code, keeping the research context that Back returns to. */
export function programHref(analysisUrl: string, program: string): string {
  const route = parseRoute(new URL(analysisUrl, 'http://shell').search)
  if (route.view !== 'run') return analysisUrl
  return routeHref({ ...route, program, pane: 'code', panels: route.panels.filter(([key]) => key !== 'right_tab') })
}

export type Step = { n: number; best: number; bar: number }

/** After each trial, the best score so far (failures never raise it) and that trial's noise line; none before the first score. */
export function frontier(points: TrialPoint[]): Step[] {
  const out: Step[] = []
  let best = -Infinity
  for (const p of points) {
    if (p.score !== null && p.score > best) best = p.score
    if (best > -Infinity) out.push({ n: p.n, best, bar: p.bar })
  }
  return out
}

export type Span = { hypothesis: Hypothesis; start: number; end: number }

/** One span per hypothesis with trials: its experiments' first to last trial. */
export function spans(hypotheses: Hypothesis[]): Span[] {
  const out: Span[] = []
  for (const h of hypotheses) {
    const times = h.experiments.flatMap((e) => (e.first_trial && e.last_trial ? [Date.parse(e.first_trial), Date.parse(e.last_trial)] : []))
    if (times.length) out.push({ hypothesis: h, start: Math.min(...times), end: Math.max(...times) })
  }
  return out
}

export type MarkKind = 'skeptic' | 'verdict' | 'confirm'
export type Mark = { hypothesis: string; time: number; kind: MarkKind }

/** The checks, verdicts and lockbox confirmations to draw on each hypothesis's track. */
export function marks(hypotheses: Hypothesis[], events: SnapshotEvent[]): Mark[] {
  // The verdict comes from the hypothesis, which keeps it however old; the event list is the latest 50.
  const out: Mark[] = hypotheses.flatMap((h) => (h.verdict ? [{ hypothesis: h.id, time: Date.parse(h.verdict.created), kind: 'verdict' as const }] : []))
  for (const e of events) {
    if (e.event === 'skeptic' || e.event === 'confirm') out.push({ hypothesis: e.hypothesis, time: Date.parse(e.time), kind: e.event })
  }
  return out
}

export type SortKey = 'id' | 'title' | 'status' | 'experiments' | 'trials' | 'best' | 'oos' | 'verdict'
export type Sort = { key: SortKey; descending: boolean }

const VALUE: Record<SortKey, (h: Hypothesis) => number | string | null> = {
  id: (h) => h.id,
  title: (h) => h.title,
  status: (h) => STATUS_ORDER.indexOf(h.status),
  experiments: (h) => h.experiments.length,
  trials: trialsOf,
  best: bestIs,
  // Validation figures exist only after a verdict, so before it they sort as missing.
  oos: oosOf,
  verdict: (h) => h.verdict?.created ?? null,
}

/** The hypotheses in the chosen order; a missing value sorts last whichever way the column runs. */
export function sortHypotheses(hypotheses: Hypothesis[], sort: Sort | null): Hypothesis[] {
  if (sort === null) return hypotheses
  const value = VALUE[sort.key]
  const sign = sort.descending ? -1 : 1
  return [...hypotheses].sort((a, b) => {
    const x = value(a)
    const y = value(b)
    if (x === null || y === null) return x === y ? 0 : x === null ? 1 : -1
    return sign * (typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y)))
  })
}
