// Builds hifi/fixture.json for the high-fidelity review frames from a running
// `shinka_research ui <demo workspace> --runs-root <demo runs>` (read-only HTTP GET only).
// The data is the labelled demo workspace (DEMO · 演示数据，非真实研究), never a real run.
// Usage: node extract.mjs http://127.0.0.1:8772
import { writeFileSync } from 'node:fs'

const base = process.argv[2] ?? 'http://127.0.0.1:8772'
async function get(path) {
  const response = await fetch(base + path)
  if (!response.ok) throw new Error(`${path} -> HTTP ${response.status}`)
  return response.json()
}
const query = (params) => new URLSearchParams(params).toString()
const round = (value, digits = 4) => (value === null || value === undefined || !Number.isFinite(value) ? null : Number(value.toFixed(digits)))
// The demo generator tags every text as "DEMO · 演示数据，非真实研究", sometimes in brackets
// ("[DEMO · …] evaluating") or followed by punctuation ("DEMO · …. In-sample"); frames show the
// demo label once in their own chrome, so the tag and its wrapper are removed from the content.
const DEMO_TAG = /\[?DEMO\s*·\s*演示数据，非真实研究\]?\s*[—–:：·.。,，-]?\s*/g
const clean = (text) => String(text ?? '').replace(DEMO_TAG, '').trim()
const COSTS = ['api_costs', 'embed_cost', 'novelty_cost', 'meta_cost']
// A program's cost is unknown when it records none of the cost fields; a recorded zero stays zero.
function programCost(program) {
  const values = COSTS.map((key) => program.metadata?.[key]).filter((value) => typeof value === 'number')
  return values.length ? values.reduce((a, b) => a + b, 0) : null
}
const median = (values) => {
  const sorted = values.filter((v) => v !== null).sort((a, b) => a - b)
  if (!sorted.length) return null
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}
const score = (program) => (program.correct && Number.isFinite(program.combined_score) ? program.combined_score : null)

// The report's check ledger shows only these skeptic fields; anything else (such as the absolute
// `best_program` path) stays out so the fixture is the same on every machine.
const CHECK_FIELDS = ['experiment', 'program_id', 'is_score', 'is_sharpe', 'n_trials', 'trial_sharpe_var_per_period', 'expected_max_sharpe_annual', 'dsr', 'pbo', 'oos_sharpe', 'oos_positive_fraction', 'oos_bars', 'decay', 'baseline_seed_oos_sharpe', 'baseline_buy_hold_oos_sharpe', 'thresholds', 'rule_label', 'rule_reason']
const pickChecks = (record) => (record ? Object.fromEntries(CHECK_FIELDS.filter((key) => key in record).map((key) => [key, record[key]])) : null)
// ---------- research directions with the evidence summary the workspace API will carry ----------
const workspace = await get('/api/workspace')
const directions = []
const events = []
let report = null
for (const entry of workspace.directions) {
  if (entry.error) { directions.push({ id: entry.id, error: clean(entry.error) }); continue }
  const snap = await get('/api/snapshot?' + query({ direction: entry.id }))
  const valid = snap.trial_points.filter((p) => p.score !== null && p.score !== undefined)
  const counts = {}
  for (const h of snap.hypotheses) counts[h.status] = (counts[h.status] ?? 0) + 1
  const times = [...snap.events.map((e) => e.time), ...snap.trial_points.map((p) => p.time)].filter(Boolean).sort()
  const over = snap.budget_usd !== null && snap.spend_usd > snap.budget_usd
  const concluded = snap.hypotheses.length > 0 && snap.hypotheses.every((h) => ['confirmed', 'unconfirmed', 'refuted', 'dropped', 'supported', 'inconclusive'].includes(h.status)) && snap.lockbox?.opened
  directions.push({
    id: entry.id,
    title: clean(snap.direction),
    created: snap.created,
    state: over ? 'over-budget' : snap.trial_points.length === 0 ? 'new' : concluded ? 'concluded' : 'active',
    trials: snap.trials,
    spend: round(snap.spend_usd, 4),
    budget: snap.budget_usd,
    lockbox: snap.lockbox?.opened ? 'opened' : snap.lockbox ? 'sealed' : 'none',
    evidence: {
      hypothesisStatus: counts,
      best: valid.length ? round(Math.max(...valid.map((p) => p.score))) : null,
      bar: snap.trial_points.length ? round(snap.trial_points.at(-1).bar) : null,
      failed: snap.trial_points.length - valid.length,
      lastActivity: times.at(-1) ?? null,
    },
    hypotheses: snap.hypotheses.map((h) => ({ id: h.id, title: clean(h.title), status: h.status })),
    records: snap.trial_points.filter((p) => p.record).map((p) => [p.n, round(p.score)]),
    bars: snap.trial_points.filter((_, i) => i % 5 === 0 || i === snap.trial_points.length - 1).map((p) => [p.n, round(p.bar)]),
  })
  for (const e of snap.events) events.push({ direction: entry.id, time: e.time, event: e.event, hypothesis: e.hypothesis ?? null, experiment: e.experiment ?? null, note: clean(e.note) })
  if (entry.id === 'd2-overnight-gap') {
    report = {
      id: entry.id,
      title: clean(snap.direction),
      created: snap.created,
      trials: snap.trials,
      spend: round(snap.spend_usd, 4),
      budget: snap.budget_usd,
      lockbox: snap.lockbox,
      hypotheses: snap.hypotheses.map((h) => ({
        id: h.id,
        title: clean(h.title),
        status: h.status,
        created: h.created,
        body: clean(h.body),
        experiments: h.experiments.map((x) => ({ id: x.id, created: x.created, evaluated: x.n_evaluated, probed: x.n_probed, best: round(x.best_is_score), spend: round(x.spend_usd, 4), first: x.first_trial, last: x.last_trial, skeptic: pickChecks(x.skeptic) })),
        verdict: h.verdict ? { label: h.verdict.label, reason: clean(h.verdict.reason), evidence: pickChecks(h.verdict.evidence) } : null,
      })),
      points: snap.trial_points.map((p) => ({ n: p.n, time: p.time, hypothesis: p.hypothesis, experiment: p.experiment, generation: p.generation, score: round(p.score), record: p.record, bar: round(p.bar) })),
      probes: snap.probes ?? [],
    }
  }
}
events.sort((a, b) => b.time.localeCompare(a.time))

// ---------- all runs ----------
const databases = await get('/list_databases')
const runs = []
const programsByPath = new Map()
for (const db of databases) {
  let programs
  try { programs = await get('/get_programs?' + query({ db_path: db.path })) } catch (error) { runs.push({ name: db.name, task: db.task ?? null, path: db.path, error: String(error.message) }); continue }
  programsByPath.set(db.path, programs)
  const costs = programs.map(programCost)
  const scores = programs.map(score).filter((v) => v !== null)
  runs.push({
    name: db.name,
    task: db.task ?? null,
    path: db.path,
    kind: db.path.startsWith('runs/') ? 'ordinary' : 'experiment',
    programs: programs.length,
    failed: programs.filter((p) => !p.correct).length,
    best: scores.length ? round(Math.max(...scores)) : null,
    cost: programs.length && costs.every((c) => c !== null) ? round(costs.reduce((a, b) => a + b, 0), 4) : programs.length ? null : 0,
    generations: programs.length ? Math.max(...programs.map((p) => p.generation)) + 1 : 0,
    islands: new Set(programs.map((p) => p.island_idx)).size,
    updated: programs.map((p) => p.timestamp).filter(Boolean).sort().at(-1) ?? null,
  })
}

// ---------- one experiment for the analysis frames ----------
const target = databases.find((d) => d.name.startsWith('d1-momentum-regimes') && d.path.includes('h001/e01'))
const full = programsByPath.get(target.path)
const index = new Map(full.map((p, i) => [p.id, i]))
const rows = full.map((p) => [
  round(p.embedding_pca_2d?.[0] ?? null, 3), round(p.embedding_pca_2d?.[1] ?? null, 3),
  round(score(p)), p.generation, p.island_idx, p.parent_id ? index.get(p.parent_id) ?? -1 : -1,
  clean(p.metadata?.patch_name ?? ''), clean(p.metadata?.model_name ?? ''), round(programCost(p), 5),
  p.embedding_cluster_id ?? null, clean(p.metadata?.patch_type ?? ''),
])
let bestIndex = -1
rows.forEach((row, i) => { if (row[2] !== null && (bestIndex === -1 || row[2] > rows[bestIndex][2])) bestIndex = i })
const path = []
for (let i = bestIndex; i !== -1 && path.length < 200; i = rows[i][5]) path.unshift(i)
const generations = [...new Set(full.map((p) => p.generation))].sort((a, b) => a - b)
let running = -Infinity
const progress = generations.map((generation) => {
  const group = full.filter((p) => p.generation === generation)
  const valid = group.map(score).filter((v) => v !== null)
  if (valid.length) running = Math.max(running, ...valid)
  const costs = group.map(programCost)
  return {
    generation,
    programs: group.length,
    failed: group.filter((p) => !p.correct).length,
    best: valid.length ? round(Math.max(...valid)) : null,
    cumulative: Number.isFinite(running) ? round(running) : null,
    median: round(median(valid)),
    cost: costs.every((c) => c !== null) ? round(costs.reduce((a, b) => a + b, 0), 4) : null,
    metaCost: round(group.reduce((a, p) => a + (p.metadata?.meta_cost ?? 0), 0), 4),
  }
})
const models = [...new Set(full.map((p) => p.metadata?.model_name).filter(Boolean))].sort()
const modelStats = models.map((model) => {
  const group = full.filter((p) => p.metadata?.model_name === model)
  const valid = group.map(score).filter((v) => v !== null)
  const costs = group.map(programCost)
  return { model: clean(model), programs: group.length, success: round(valid.length / group.length, 3), best: valid.length ? round(Math.max(...valid)) : null, mean: valid.length ? round(valid.reduce((a, b) => a + b, 0) / valid.length) : null, cost: costs.every((c) => c !== null) ? round(costs.reduce((a, b) => a + b, 0), 4) : null }
})
const best = full[bestIndex]
const parent = best.parent_id ? full[index.get(best.parent_id)] : null
const metricKeys = Object.keys(best.public_metrics ?? {}).filter((k) => typeof best.public_metrics[k] === 'number').slice(0, 8)
const analysis = {
  run: target.name,
  direction: 'd1-momentum-regimes', hypothesis: 'h001', experiment: 'e01',
  stats: { programs: full.length, failed: full.filter((p) => !p.correct).length, generations: generations.length, islands: new Set(full.map((p) => p.island_idx)).size, models: models.length },
  rows, bestIndex, path, progress, modelStats, metricKeys,
  metrics: full.map((p) => (p.correct ? metricKeys.map((k) => round(p.public_metrics?.[k] ?? null)) : null)),
  best: {
    id: best.id, name: clean(best.metadata?.patch_name ?? best.id.slice(0, 8)), generation: best.generation, island: best.island_idx,
    score: round(score(best)), parentScore: parent ? round(score(parent)) : null, model: clean(best.metadata?.model_name ?? ''),
    cost: round(programCost(best), 5), patchType: clean(best.metadata?.patch_type ?? ''),
    description: clean(best.metadata?.patch_description ?? ''), code: clean(best.code).split('\n'), diff: clean(best.code_diff ?? '').split('\n'),
    feedback: clean(best.text_feedback ?? ''), stdout: clean(best.metadata?.stdout_log ?? ''), thought: clean(best.metadata?.thought ?? ''),
    metrics: Object.fromEntries(metricKeys.map((k) => [k, round(best.public_metrics[k])])),
  },
  failedExample: (() => { const p = full.find((x) => !x.correct); return p ? { id: p.id, name: clean(p.metadata?.patch_name ?? p.id.slice(0, 8)), generation: p.generation, island: p.island_idx, stderr: clean(p.metadata?.stderr_log ?? ''), feedback: clean(p.text_feedback ?? '') } : null })(),
}

// ---------- prompt evolution and meta-analysis of the same run (OD-3), read-only HTTP ----------
// Meta updates are numbered by processed programs; each is placed on the generation axis at the
// generation of that program in evaluation order. File paths from the server stay out.
const evaluationOrder = [...full].sort((a, b) => a.generation - b.generation || (a.timestamp ?? 0) - (b.timestamp ?? 0))
const metaFiles = await get('/get_meta_files?' + query({ db_path: target.path }))
const metaUpdates = metaFiles.map((m) => ({ processed: m.processed_count, file: m.filename, generation: evaluationOrder[Math.min(m.processed_count, evaluationOrder.length) - 1]?.generation ?? null }))
const metaSection = (text, heading) => {
  const lines = text.split('\n')
  const start = lines.findIndex((line) => line.trim() === `# ${heading}`)
  if (start === -1) return []
  const out = []
  for (const line of lines.slice(start + 1)) { if (/^# /.test(line)) break; if (line.trim()) out.push(clean(line)) }
  return out
}
const latestMeta = metaUpdates.length ? await get('/get_meta_content?' + query({ db_path: target.path, processed_count: metaUpdates.at(-1).processed })) : null
analysis.meta = {
  updates: metaUpdates,
  latest: latestMeta ? { processed: latestMeta.processed_count, file: latestMeta.filename, insights: metaSection(latestMeta.content, 'GLOBAL INSIGHTS SCRATCHPAD'), recommendations: metaSection(latestMeta.content, 'META RECOMMENDATIONS'), chars: latestMeta.content.length } : null,
}
const prompts = await get('/get_system_prompts?' + query({ db_path: target.path }))
const promptName = new Map(prompts.map((p) => [p.id, clean(p.name)]))
analysis.prompts = prompts.map((p) => ({ name: clean(p.name), generation: p.program_generation, parent: p.parent_id ? promptName.get(p.parent_id) ?? null : null, patchType: clean(p.patch_type), description: clean(p.description), programs: p.program_count, correct: p.correct_program_count, fitness: round(p.fitness) })).sort((a, b) => a.generation - b.generation)

const fixture = { schema: 'shinka-research.hifi-fixture.v1', source: 'demo-20261019 · DEMO · 演示数据，非真实研究', workspace: { name: workspace.name, library: workspace.library ? { pages: workspace.library.pages?.length ?? null, pending: workspace.library.pending?.length ?? null } : null }, directions, events, report, runs, analysis }
writeFileSync(new URL('./fixture.json', import.meta.url), JSON.stringify(fixture) + '\n')
console.log(JSON.stringify({ directions: directions.map((d) => [d.id, d.state, d.trials, d.evidence?.best, d.evidence?.bar]), runs: runs.length, programs: full.length, best: analysis.best.score, metricKeys, models }))
