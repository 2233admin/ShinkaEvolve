// Builds the shared content fixture for the direction previews from a running
// `shinka_research ui <demo workspace> --runs-root <demo runs>` (read-only HTTP).
// The data is the labelled demo workspace (DEMO · 演示数据，非真实研究), never a real run.
// Usage: node extract-fixture.mjs http://127.0.0.1:8772 > writes ./fixture.json
import { writeFileSync } from 'node:fs'

const base = process.argv[2] ?? 'http://127.0.0.1:8772'
const get = async (path) => {
  const response = await fetch(base + path)
  if (!response.ok) throw new Error(`${path} -> HTTP ${response.status}`)
  return response.json()
}
const query = (params) => new URLSearchParams(params).toString()
const round = (value, digits = 4) => (value === null || value === undefined || !Number.isFinite(value) ? null : Number(value.toFixed(digits)))
const DEMO_TAG = /DEMO\s*·\s*演示数据，非真实研究\s*[—–:·-]?\s*/g
const clean = (text) => String(text ?? '').replace(DEMO_TAG, '').trim()
const COSTS = ['api_costs', 'embed_cost', 'novelty_cost', 'meta_cost']
const programCost = (program) => {
  const values = COSTS.map((key) => program.metadata?.[key])
  return values.every((value) => typeof value === 'number') ? values.reduce((a, b) => a + b, 0) : null
}
const median = (values) => {
  if (!values.length) return null
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

const workspace = await get('/api/workspace')
const databases = await get('/list_databases')
const directions = []
const events = []
for (const entry of workspace.directions) {
  if (entry.error) {
    directions.push({ id: entry.id, title: clean(entry.direction), error: entry.error })
    continue
  }
  const snapshot = await get('/api/snapshot?' + query({ direction: entry.id }))
  const points = snapshot.trial_points
  const scored = points.filter((point) => point.score !== null)
  const best = scored.length ? Math.max(...scored.map((point) => point.score)) : null
  const records = points.filter((point) => point.record).map((point) => [point.n, round(point.score)])
  const bars = points.filter((point, index) => index === points.length - 1 || index % Math.max(1, Math.floor(points.length / 60)) === 0)
    .map((point) => [point.n, round(point.bar)])
  const failed = points.filter((point) => point.score === null).map((point) => point.n)
  const times = [...snapshot.events.map((event) => event.time), ...points.map((point) => point.time)].sort()
  for (const event of snapshot.events) {
    events.push({ time: event.time, direction: entry.id, hypothesis: event.hypothesis, experiment: event.experiment, event: event.event, note: clean(event.note).slice(0, 140) })
  }
  const hypotheses = snapshot.hypotheses.map((hypothesis) => ({ id: hypothesis.id, title: clean(hypothesis.title), status: hypothesis.status, experiments: hypothesis.experiments.length }))
  const terminal = new Set(['supported', 'refuted', 'confirmed', 'unconfirmed', 'dropped', 'inconclusive'])
  const state = snapshot.trials === 0 ? 'new'
    : snapshot.budget_usd !== null && snapshot.spend_usd > snapshot.budget_usd ? 'over-budget'
    : snapshot.lockbox?.opened && hypotheses.every((hypothesis) => terminal.has(hypothesis.status)) ? 'concluded'
    : 'active'
  directions.push({
    id: entry.id,
    title: clean(snapshot.direction),
    created: snapshot.created,
    state,
    hypotheses,
    trials: snapshot.trials,
    probes: snapshot.probes,
    spend: round(snapshot.spend_usd, 2),
    budget: snapshot.budget_usd,
    lockbox: snapshot.lockbox === null ? null : snapshot.lockbox.opened ? 'opened' : 'sealed',
    best: round(best),
    bar: points.length ? round(points.at(-1).bar) : null,
    failed: failed.length,
    lastActivity: times.at(-1) ?? null,
    records,
    bars,
    failures: failed.slice(0, 400),
  })
}
events.sort((a, b) => b.time.localeCompare(a.time))

// Every program stays authoritative for counts, lineage, progress and the inspector.
// Only the solution-space drawing skips programs without a 2D projection (x/y = null).
const programsOf = (path) => get('/get_programs?' + query({ db_path: path }))
const projected = (program) => Array.isArray(program.embedding_pca_2d) && program.embedding_pca_2d.length === 2
const byDirection = new Map()
for (const row of databases) {
  const match = /^workspace\/([^/]+)\/experiments\/([^/]+)\/([^/]+)\/results\//.exec(row.path)
  if (!match) continue
  const stats = await get('/get_database_stats?' + query({ db_path: row.path }))
  const list = byDirection.get(match[1]) ?? []
  list.push({ path: row.path, hypothesis: match[2], experiment: match[3], programs: stats.program_count ?? 0 })
  byDirection.set(match[1], list)
}

function compact(programs) {
  const index = new Map(programs.map((program, i) => [program.id, i]))
  return programs.map((program) => [
    projected(program) ? round(program.embedding_pca_2d[0]) : null, projected(program) ? round(program.embedding_pca_2d[1]) : null,
    program.correct ? round(program.combined_score) : null,
    program.generation, program.island_idx ?? 0,
    program.parent_id && index.has(program.parent_id) ? index.get(program.parent_id) : -1,
  ])
}
function lineage(rows, bestIndex) {
  const path = []
  for (let i = bestIndex; i !== -1 && path.length < 200; i = rows[i][5]) path.push(i)
  return path.reverse()
}
const maps = {}
for (const [direction, list] of byDirection) {
  const main = [...list].sort((a, b) => b.programs - a.programs)[0]
  if (!main || !main.programs) continue
  const rows = compact(await programsOf(main.path))
  let bestIndex = -1
  rows.forEach((row, i) => { if (row[2] !== null && (bestIndex === -1 || row[2] > rows[bestIndex][2])) bestIndex = i })
  maps[direction] = { hypothesis: main.hypothesis, experiment: main.experiment, programs: rows, best: bestIndex, path: lineage(rows, bestIndex) }
}

const target = byDirection.get('d1-momentum-regimes').find((item) => item.hypothesis === 'h001' && item.experiment === 'e01')
const full = await programsOf(target.path)
const rows = compact(full)
let bestIndex = -1
rows.forEach((row, i) => { if (row[2] !== null && (bestIndex === -1 || row[2] > rows[bestIndex][2])) bestIndex = i })
const bestProgram = full[bestIndex]
const details = await get('/get_program_details?' + query({ db_path: target.path, id: bestProgram.id }))
const parent = full.find((program) => program.id === bestProgram.parent_id)
const generations = [...new Set(full.map((program) => program.generation))].sort((a, b) => a - b)
let running = -Infinity
const progress = generations.map((generation) => {
  const cohort = full.filter((program) => program.generation === generation)
  const scores = cohort.filter((program) => program.correct && Number.isFinite(program.combined_score)).map((program) => program.combined_score)
  const best = scores.length ? Math.max(...scores) : null
  if (best !== null) running = Math.max(running, best)
  const costs = cohort.map(programCost)
  return {
    generation,
    best: round(best),
    cumulative: Number.isFinite(running) ? round(running) : null,
    median: round(median(scores)),
    failed: cohort.length - scores.length,
    cost: costs.every((cost) => cost !== null) ? round(costs.reduce((a, b) => a + b, 0), 4) : null,
  }
})
const models = [...new Set(full.map((program) => program.metadata?.model_name).filter(Boolean))].sort()
const analysis = {
  direction: 'd1-momentum-regimes', hypothesis: 'h001', experiment: 'e01', database: target.path,
  stats: { programs: full.length, generations: generations.length, islands: new Set(full.map((program) => program.island_idx)).size, failed: full.filter((program) => !program.correct).length, models },
  programs: rows.map((row, i) => [...row, full[i].metadata?.patch_type ?? null, full[i].metadata?.model_name ?? null]),
  best: {
    index: bestIndex, id: bestProgram.id, generation: bestProgram.generation, island: bestProgram.island_idx,
    score: round(bestProgram.combined_score), parentScore: parent?.correct ? round(parent.combined_score) : null,
    name: clean(details.metadata?.patch_name ?? bestProgram.metadata?.patch_name ?? bestProgram.id),
    description: clean(details.metadata?.patch_description ?? '').slice(0, 220),
    model: details.metadata?.model_name ?? null, patch: details.metadata?.patch_type ?? null,
    cost: round(programCost(details), 4),
    code: String(details.code ?? '').split('\n').slice(0, 60),
    diff: String(details.code_diff ?? '').split('\n').slice(0, 60),
  },
  path: lineage(rows, bestIndex),
  progress,
}

const fixture = {
  schema: 'shinka-research.direction-preview-fixture.v1',
  source: 'demo-20261019 · DEMO · 演示数据，非真实研究',
  workspace: { name: workspace.name, libraryPages: workspace.library?.pages?.length ?? 0, runs: databases.length },
  directions,
  events,
  maps,
  analysis,
}
writeFileSync(new URL('./fixture.json', import.meta.url), JSON.stringify(fixture) + '\n')
console.log(JSON.stringify({ directions: directions.map((d) => [d.id, d.state, d.trials, d.best, d.bar]), maps: Object.keys(maps), programs: full.length, best: analysis.best.score }))
