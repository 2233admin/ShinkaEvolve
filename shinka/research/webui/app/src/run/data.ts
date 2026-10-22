/**
 * The run analysis page's derived data (award-redesign 6.1, 6.2): the best programs and their
 * ancestry, the island lanes, meta-analysis files, prompt versions and the lineage panel's tables.
 * Plain functions, so node can test them directly.
 */
import { isCorrect, scoreOf, type ProgramSummary } from '../compare/series.ts'

export type Migration = { generation?: number | null; from?: number | null; to?: number | null; timestamp?: number | null }

/** One program as `/get_programs_summary` returns it. */
export type RunProgram = ProgramSummary & {
  parent_id?: string | null
  island_idx?: number | null
  complexity?: number | null
  children_count?: number | null
  language?: string | null
  text_feedback?: string | null
  public_metrics?: Record<string, unknown> | null
  private_metrics?: Record<string, unknown> | null
  private_metrics_hidden?: boolean | null
  top_k_inspiration_ids?: string[] | null
  archive_inspiration_ids?: string[] | null
  migration_history?: Migration[] | null
  in_archive?: boolean | number | null
}

/** `/get_program_details` adds the source, its diff against the parent and the prompt version used. */
export type ProgramDetail = RunProgram & { code?: string | null; code_diff?: string | null; system_prompt_id?: string | null }

/** `/get_meta_files` and `/get_meta_content` (the server's absolute paths are not used). */
export type MetaFile = { processed_count: number; filename: string }
export type MetaContent = MetaFile & { content: string }

/** One prompt version as `/get_system_prompts` returns it. */
export type Prompt = {
  id: string
  name?: string | null
  description?: string | null
  parent_id?: string | null
  generation?: number | null
  program_generation?: number | null
  patch_type?: string | null
  timestamp?: number | null
  program_count?: number | null
  correct_program_count?: number | null
  fitness?: number | null
  total_improvement?: number | null
  in_archive?: boolean | number | null
  program_ids?: string[] | null
  prompt_text?: string | null
  metadata?: Record<string, unknown> | null
}

const finite = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const text = (v: unknown): string | null => (typeof v === 'string' && v !== '' ? v : null)

export const generationOf = (p: RunProgram): number => finite(p.generation) ?? 0
export const islandOf = (p: RunProgram): number | null => finite(p.island_idx)
export const patchTypeOf = (p: RunProgram): string | null => text(p.metadata?.patch_type)
/** A program's name: its patch name, else its short id. */
export const nameOf = (p: RunProgram): string => text(p.metadata?.patch_name) ?? p.id.slice(0, 8)

/**
 * What the inspector says about private metrics: they are never shown (run-analysis-ui spec), only
 * whether the run held some back. A response that carries values anyway reads as hidden.
 */
export const privateMetrics = (p: RunProgram): 'hidden' | 'none' =>
  p.private_metrics_hidden === true || Object.keys(p.private_metrics ?? {}).length > 0 ? 'hidden' : 'none'

/**
 * Why a failed evaluation failed, for the inspector's header (S-14): the evaluator's feedback, else the
 * last line it wrote to stderr; nothing for a valid program.
 */
export function failureReason(p: RunProgram): string | null {
  if (scoreOf(p) !== null) return null
  const feedback = text(p.text_feedback)?.trim()
  if (feedback) return feedback
  const stderr = text(p.metadata?.stderr_log)?.trim().split('\n').filter((line) => line.trim()).at(-1)
  return stderr?.trim() ?? null
}

/** The model that wrote a program, wherever the runner recorded it. */
export function modelOf(p: RunProgram): string | null {
  const md = p.metadata ?? {}
  const llm = (md.llm_result ?? {}) as Record<string, unknown>
  return text(md.model_name) ?? text(md.model) ?? text(llm.model_name) ?? text(llm.model)
}

/** Evaluation order, the order `processed_count` counts in: generation, then time (hi-fi extract.mjs:252). */
export function evaluationOrder(programs: RunProgram[]): RunProgram[] {
  return [...programs].sort((a, b) => generationOf(a) - generationOf(b) || (finite(a.timestamp) ?? 0) - (finite(b.timestamp) ?? 0))
}

/**
 * The run's best programs: every valid program at the highest valid score, in evaluation order. A
 * failed evaluation is never best, and ties are all kept rather than cut to the first (D-L01c).
 */
export function bestOf(programs: RunProgram[]): RunProgram[] {
  let top: number | null = null
  for (const p of programs) {
    const s = scoreOf(p)
    if (s !== null && (top === null || s > top)) top = s
  }
  return top === null ? [] : evaluationOrder(programs).filter((p) => scoreOf(p) === top)
}

/** A program's ancestry from its root to itself; it stops at a parent the run does not hold and at a loop. */
export function lineOf(byId: ReadonlyMap<string, RunProgram>, id: string): RunProgram[] {
  const line: RunProgram[] = []
  const seen = new Set<string>()
  for (let p = byId.get(id); p && !seen.has(p.id); p = p.parent_id ? byId.get(p.parent_id) : undefined) {
    seen.add(p.id)
    line.unshift(p)
  }
  return line
}

/** The initial program a diff compares with: the lineage's own root when it is from generation 0, else the run's first. */
export function initialOf(line: RunProgram[], programs: RunProgram[]): RunProgram | null {
  if (line.length > 0 && generationOf(line[0]) === 0) return line[0]
  return evaluationOrder(programs).find((p) => generationOf(p) === 0) ?? null
}

/** The score ramp's domain: the 5th percentile of the valid scores to the best (design.md › 图表语法). */
export function rampDomain(programs: RunProgram[]): [number, number] | null {
  const scores = programs.map(scoreOf).filter((s): s is number => s !== null).sort((a, b) => a - b)
  if (scores.length === 0) return null
  const at = (scores.length - 1) * 0.05
  const lo = scores[Math.floor(at)] + (scores[Math.ceil(at)] - scores[Math.floor(at)]) * (at - Math.floor(at))
  return [lo, scores[scores.length - 1]]
}

/** Islands in lane order; programs without an island share a last lane (null). */
export function lanesOf(programs: RunProgram[]): (number | null)[] {
  const islands = [...new Set(programs.map(islandOf))]
  return islands.sort((a, b) => (a === null ? 1 : b === null ? -1 : a - b))
}

/** L07-Q2: each island's population after every generation, counting every program it holds, failures included. */
export function population(programs: RunProgram[]): { island: number | null; counts: number[] }[] {
  const last = programs.reduce((m, p) => Math.max(m, generationOf(p)), 0)
  return lanesOf(programs).map((island) => {
    const counts = new Array<number>(last + 1).fill(0)
    for (const p of programs) if (islandOf(p) === island) counts[generationOf(p)] += 1
    for (let g = 1; g <= last; g += 1) counts[g] += counts[g - 1]
    return { island, counts }
  })
}

/** The best valid score so far, at each generation where it rose (L07-Q1: an island's progression). */
export function bestSoFar(programs: RunProgram[]): { generation: number; best: number }[] {
  const steps: { generation: number; best: number }[] = []
  for (const p of evaluationOrder(programs)) {
    const s = scoreOf(p)
    if (s === null || (steps.length > 0 && s <= steps[steps.length - 1].best)) continue
    const g = generationOf(p)
    if (steps.length > 0 && steps[steps.length - 1].generation === g) steps[steps.length - 1].best = s
    else steps.push({ generation: g, best: s })
  }
  return steps
}

export type MetaUpdate = { processed: number; file: string; generation: number | null }

/**
 * Meta-analysis updates in order, each placed on the generation axis at the generation of the program
 * whose evaluation completed that count (hi-fi extract.mjs:254).
 */
export function metaUpdates(files: MetaFile[], programs: RunProgram[]): MetaUpdate[] {
  const order = evaluationOrder(programs)
  return [...files]
    .sort((a, b) => a.processed_count - b.processed_count)
    .map((f) => {
      const at = order[Math.min(f.processed_count, order.length) - 1]
      return { processed: f.processed_count, file: f.filename, generation: at ? generationOf(at) : null }
    })
}

export const META_SECTIONS = {
  programs: 'INDIVIDUAL PROGRAM SUMMARIES',
  insights: 'GLOBAL INSIGHTS SCRATCHPAD',
  recommendations: 'META RECOMMENDATIONS',
} as const

/** One section of a meta-analysis file, as markdown without its heading; empty when the file has none. */
export function metaSection(content: string, heading: string): string {
  const lines = content.split('\n')
  const start = lines.findIndex((line) => line.trim() === `# ${heading}`)
  if (start === -1) return ''
  const end = lines.findIndex((line, i) => i > start && /^# /.test(line))
  return lines.slice(start + 1, end === -1 ? undefined : end).join('\n').trim()
}

const NAME_LINE = /^\*\*Program Name:/
// MetaSummarizer appends this line to every summary (shinka/core/summarizer.py, step 1).
const IDENTIFIER = /^\*\*Program Identifier:\*\*\s*Generation (\d+) - Patch Name (.*) - Correct Program: (\w+)\s*$/

/**
 * The program's own summaries in a meta file's individual section. Each summary written by the
 * runner ends with its identifier line (generation and patch name); the demo names the short id in
 * the name line instead. Every summary that matches is returned, never only the first.
 */
export function programSummaries(content: string, p: RunProgram): string[] {
  const lines = metaSection(content, META_SECTIONS.programs).split('\n')
  const found: string[] = []
  if (lines.some((line) => IDENTIFIER.test(line.trim()))) {
    let from = 0
    lines.forEach((line, i) => {
      const id = IDENTIFIER.exec(line.trim())
      if (!id) return
      const chunk = lines.slice(from, i + 1)
      from = i + 1
      const name = chunk.findIndex((l) => NAME_LINE.test(l))
      if (Number(id[1]) === generationOf(p) && id[2] === nameOf(p)) found.push(chunk.slice(Math.max(0, name)).join('\n').trim())
    })
    return found
  }
  const tag = `(${p.id.slice(0, 8)},`
  lines.forEach((line, i) => {
    if (!NAME_LINE.test(line) || !line.includes(tag)) return
    const end = lines.findIndex((l, j) => j > i && NAME_LINE.test(l))
    found.push(lines.slice(i, end === -1 ? undefined : end).join('\n').trim())
  })
  return found
}

/** Prompt versions in the order they were made: generation, then time. */
export function promptOrder(prompts: Prompt[]): Prompt[] {
  return [...prompts].sort((a, b) => (finite(a.generation) ?? 0) - (finite(b.generation) ?? 0) || (finite(a.timestamp) ?? 0) - (finite(b.timestamp) ?? 0))
}

/** The best prompt versions: the highest fitness among those that produced a program (L02-Q1); ties are all kept. */
export function bestPrompts(prompts: Prompt[]): Set<string> {
  const eligible = prompts.filter((q) => (finite(q.program_count) ?? 0) >= 1 && finite(q.fitness) !== null)
  const top = Math.max(...eligible.map((q) => q.fitness as number))
  return new Set(eligible.filter((q) => q.fitness === top).map((q) => q.id))
}

export type PromptRow = { program: RunProgram; delta: number | null; beat: number | null }

/**
 * The programs a prompt version produced (L02-Q4), in evaluation order: Δ against the parent's valid
 * score and Beat%, the share of the run's other valid programs it scores above (VT:18094-18115).
 * A failed program, or one whose parent failed, has no Δ.
 */
export function promptPrograms(prompt: Prompt, programs: RunProgram[]): PromptRow[] {
  const byId = new Map(programs.map((p) => [p.id, p]))
  const ids = new Set(prompt.program_ids ?? [])
  const valid = programs.map(scoreOf).filter((s): s is number => s !== null).sort((a, b) => a - b)
  const beat = (s: number) => {
    let below = 0
    while (below < valid.length && valid[below] < s) below += 1
    return valid.length <= 1 ? 100 : (below / (valid.length - 1)) * 100
  }
  return evaluationOrder(programs.filter((p) => ids.has(p.id))).map((p) => {
    const s = scoreOf(p)
    const parent = p.parent_id ? byId.get(p.parent_id) : undefined
    const ps = parent ? scoreOf(parent) : null
    return { program: p, delta: s !== null && ps !== null ? s - ps : null, beat: s === null ? null : beat(s) }
  })
}

const mean = (xs: number[]): number | null => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null)

/** A prompt version's programs at a glance: valid, improved on their parent, average Δ and average Beat%. */
export function promptTally(rows: PromptRow[]) {
  return {
    correct: rows.filter((r) => isCorrect(r.program)).length,
    improved: rows.filter((r) => r.delta !== null && r.delta > 0).length,
    delta: mean(rows.flatMap((r) => (r.delta === null ? [] : [r.delta]))),
    beat: mean(rows.flatMap((r) => (r.beat === null ? [] : [r.beat]))),
  }
}

export type MigrationRow = { program: RunProgram; generation: number | null; from: number | null; to: number | null }

/** Every recorded migration between islands (L07-Q4), by generation. */
export function migrations(programs: RunProgram[]): MigrationRow[] {
  const rows = programs.flatMap((p) =>
    (p.migration_history ?? []).map((m) => ({ program: p, generation: finite(m.generation), from: finite(m.from), to: finite(m.to) })),
  )
  return rows.sort((a, b) => (a.generation ?? Infinity) - (b.generation ?? Infinity) || generationOf(a.program) - generationOf(b.program))
}

export type Parent = { program: RunProgram; children: number; improving: number }

/**
 * R09-Q11: the most fertile parents, with how many of their children beat them (a failed child never
 * does), and the valid programs that have no child yet.
 */
export function fertility(programs: RunProgram[], n = 10): { top: Parent[]; deadEnds: number } {
  const children = new Map<string, RunProgram[]>()
  for (const p of programs) if (p.parent_id) children.set(p.parent_id, [...(children.get(p.parent_id) ?? []), p])
  const parents = programs.flatMap((p): Parent[] => {
    const kids = children.get(p.id) ?? []
    if (kids.length === 0) return []
    const own = scoreOf(p)
    const improving = kids.filter((k) => {
      const s = scoreOf(k)
      return s !== null && (own === null || s > own)
    }).length
    return [{ program: p, children: kids.length, improving }]
  })
  parents.sort((a, b) => b.children - a.children || b.improving - a.improving || generationOf(a.program) - generationOf(b.program))
  const deadEnds = programs.filter((p) => scoreOf(p) !== null && !children.has(p.id)).length
  return { top: parents.slice(0, n), deadEnds }
}

export type MutationPoint = { program: RunProgram; size: number; added: number; deleted: number; modified: number; delta: number | null; failed: 'self' | 'parent' | null }

/**
 * R09-Q4: each child's mutation size (lines added, deleted and modified, from the runner's
 * `diff_summary`) against its change in score. A failed child or a child of a failed parent has no
 * change in score and is marked instead of drawn at 0.
 */
export function mutationPoints(programs: RunProgram[]): MutationPoint[] {
  const byId = new Map(programs.map((p) => [p.id, p]))
  return programs.flatMap((p): MutationPoint[] => {
    const parent = p.parent_id ? byId.get(p.parent_id) : undefined
    const summary = p.metadata?.diff_summary
    if (!parent || typeof summary !== 'object' || summary === null) return []
    const lines = summary as Record<string, unknown>
    const [added, deleted, modified] = ['added', 'deleted', 'modified'].map((k) => finite(lines[k]) ?? 0)
    const s = scoreOf(p)
    const ps = scoreOf(parent)
    const delta = s !== null && ps !== null ? s - ps : null
    return [{ program: p, size: added + deleted + modified, added, deleted, modified, delta, failed: s === null ? 'self' : ps === null ? 'parent' : null }]
  })
}

/** The least-squares line through the points; none for fewer than two distinct x values. */
export function fitLine(points: { x: number; y: number }[]): { slope: number; intercept: number } | null {
  const mx = mean(points.map((p) => p.x))
  const my = mean(points.map((p) => p.y))
  if (mx === null || my === null) return null
  const sxx = points.reduce((s, p) => s + (p.x - mx) ** 2, 0)
  if (sxx === 0) return null
  const slope = points.reduce((s, p) => s + (p.x - mx) * (p.y - my), 0) / sxx
  return { slope, intercept: my - slope * mx }
}

export type LabelSide = 'right' | 'left' | null

/**
 * Where each track label goes: right of its mark, else left of it, else nowhere. A label never covers a
 * drawn mark or another label and stays between `lo` and `hi`; the kept labels (the best prompt, the
 * latest meta update) are placed first, so only marks and each other can take their room. Items that are
 * `off` are not drawn.
 */
export function fitLabels(items: { x: number; w: number; keep: boolean; off: boolean }[], lo: number, hi: number): LabelSide[] {
  const marks = items.filter((it) => !it.off).map(({ x }) => [x - 6, x + 6])
  const placed: [number, number][] = []
  const side: LabelSide[] = items.map(() => null)
  const order = items.map((_, i) => i).sort((a, b) => Number(items[b].keep) - Number(items[a].keep) || items[a].x - items[b].x)
  for (const i of order) {
    const { x, w, off } = items[i]
    if (off) continue
    const spans: [LabelSide, number, number][] = [['right', x + 8, x + 8 + w], ['left', x - 8 - w, x - 8]]
    const pick = spans.find(([, a, b]) => a >= lo && b <= hi && !marks.some(([c, d]) => a < d && b > c) && !placed.some(([c, d]) => a < d + 6 && b + 6 > c))
    if (!pick) continue
    placed.push([pick[1], pick[2]])
    side[i] = pick[0]
  }
  return side
}

const EXTENSIONS: Record<string, string> = { python: 'py', javascript: 'js', typescript: 'ts', cuda: 'cu', cpp: 'cpp', c: 'c', rust: 'rs', julia: 'jl', swift: 'swift', json: 'json' }

/** The code download's file name, as the restored page wrote it: `<patch_name>_gen<N>.<ext>` (R05-X1). */
export function codeFileName(p: RunProgram): string {
  const language = text(p.language)?.toLowerCase() ?? 'python'
  const ext = Object.hasOwn(EXTENSIONS, language) ? EXTENSIONS[language] : language.replace(/[^a-z0-9]/g, '') || 'txt'
  return `${nameOf(p).replace(/[\\/:*?"<>|\s]+/g, '_')}_gen${generationOf(p)}.${ext}`
}

/**
 * Highlighted HTML split into lines: a span that crosses a line break is closed at the end of the
 * line and reopened on the next, so every line stands alone in its own row.
 */
export function splitHighlighted(html: string): string[] {
  const out: string[] = []
  const open: string[] = []
  for (const line of html.split('\n')) {
    let row = open.join('')
    for (const m of line.matchAll(/<span[^>]*>|<\/span>/g)) {
      if (m[0] === '</span>') open.pop()
      else open.push(m[0])
    }
    row += line + '</span>'.repeat(open.length)
    out.push(row)
  }
  return out
}

export type DiffKind = 'add' | 'del' | 'hunk' | 'head' | 'same'

/**
 * A unified diff's lines by kind (R06): file headers, hunk headers, added, removed and unchanged. A
 * line starting `---` or `+++` inside a hunk is a removed or added line, not a header.
 */
export function diffRows(unified: string): { kind: DiffKind; text: string }[] {
  let inHunk = false
  return unified.replace(/\n$/, '').split('\n').map((line) => {
    if (line.startsWith('Index:') || line.startsWith('diff ')) inHunk = false
    if (line.startsWith('@@')) inHunk = true
    const kind: DiffKind = line.startsWith('@@')
      ? 'hunk'
      : !inHunk && (line.startsWith('+++') || line.startsWith('---') || line.startsWith('Index:') || line.startsWith('diff ') || /^=+$/.test(line))
        ? 'head'
        : line.startsWith('+')
          ? 'add'
          : line.startsWith('-')
            ? 'del'
            : 'same'
    return { kind, text: line }
  })
}
