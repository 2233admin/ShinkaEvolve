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
// Same rule as rundb.run_cost: an absent or null field is a recorded zero (the runner writes a field
// only for the steps that ran, meta_cost only where a meta update charged the program); a present
// value must be a finite number ≥ 0. A malformed value or an unpriced headless call is unknown, and
// so is every total over it. Island copies repeat their source's metadata and are not charges.
const amount = (value) => (value === undefined || value === null ? 0 : typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null)
const isCopy = (program) => Boolean(program.metadata?._is_island_copy || program.metadata?._spawned_island)
function programCost(program) {
  const m = program.metadata ?? {}
  if (isCopy(program)) return 0
  const model = String([m.model_name, m.model, m.llm_result?.model_name, m.llm_result?.model].find(Boolean) ?? '')
  if (model.startsWith('headless/') && (m.headless_pricing_unknown === true || m.headless_pricing_unknown === 1)) return null
  const values = COSTS.map((key) => amount(m[key]))
  return values.includes(null) ? null : values.reduce((a, b) => a + b, 0)
}
const promptCost = (prompt) => amount(prompt.metadata?.llm?.cost)
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
// The analysis run's noise line by generation: the direction's noise bar at the last trial point of
// that run in each generation, so Progress can show the same evidence as home and the report.
let analysisNoise = []
const CONFIRMATION_FIELDS = ['experiment', 'program_id', 'lockbox_start', 'lockbox_bars', 'lockbox_sharpe', 'lockbox_annual_return', 'lockbox_max_drawdown', 'lockbox_buy_hold_sharpe', 'n_trials', 'dsr', 'lockbox_positive_fraction', 'passed']
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
    // trials is the notebook's count (finished experiments plus probes, the DSR basis); evaluated is
    // every program in the run databases, which also covers a run still in progress.
    trials: snap.trials,
    evaluated: snap.trial_points.length,
    probes: snap.probes,
    // The server's spend is the recorded sum; it is a lower bound while some run's cost is unknown.
    spend: round(snap.spend_usd, 4),
    unknownCostRuns: snap.unknown_cost_runs,
    spendKnown: snap.unknown_cost_runs === 0,
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
  if (entry.id === 'd1-momentum-regimes') {
    const byGeneration = new Map()
    for (const p of snap.trial_points) if (p.hypothesis === 'h001' && p.experiment === 'e01') byGeneration.set(p.generation, round(p.bar))
    analysisNoise = [...byGeneration].sort((a, b) => a[0] - b[0])
  }
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
        verdict: h.verdict ? { label: h.verdict.label, reason: clean(h.verdict.reason), created: h.verdict.created ?? null, evidence: pickChecks(h.verdict.evidence) } : null,
        confirmation: h.confirmation ? Object.fromEntries(CONFIRMATION_FIELDS.filter((key) => key in h.confirmation).map((key) => [key, h.confirmation[key]])) : null,
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
const promptsByPath = new Map()
for (const db of databases) {
  let programs, stats, prompts
  try {
    programs = await get('/get_programs?' + query({ db_path: db.path }))
    stats = await get('/get_database_stats?' + query({ db_path: db.path }))
    prompts = stats.has_prompt_evo ? await get('/get_system_prompts?' + query({ db_path: db.path })) : []
  } catch (error) { runs.push({ name: db.name, task: db.task ?? null, path: db.path, error: String(error.message) }); continue }
  programsByPath.set(db.path, programs)
  promptsByPath.set(db.path, prompts)
  const scores = programs.map(score).filter((v) => v !== null)
  // One spend definition everywhere (OD-9): the server's run_cost, recorded program costs plus
  // prompt evolution, the same total the research notebook and the budget use. Unknown stays null.
  runs.push({
    name: db.name,
    task: db.task ?? null,
    path: db.path,
    kind: db.path.startsWith('runs/') ? 'ordinary' : 'experiment',
    programs: programs.length,
    valid: stats.correct_count ?? scores.length,
    failed: programs.filter((p) => !p.correct).length,
    best: scores.length ? round(Math.max(...scores)) : null,
    cost: typeof stats.total_cost === 'number' && typeof stats.prompt_evo_cost === 'number' ? round(stats.total_cost + stats.prompt_evo_cost, 4) : null,
    promptCost: typeof stats.prompt_evo_cost === 'number' ? round(stats.prompt_evo_cost, 4) : null,
    prompts: stats.prompt_count ?? 0,
    stall: stats.gens_since_improvement ?? null,
    runtime: stats.total_runtime_seconds ?? null,
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
// Prompt-evolution spend sits in prompts.sqlite (metadata.llm.cost) and is placed at the generation
// the prompt version took effect; programs carry model calls and meta-analysis cost (OD-9 split).
// A generation's prompt spend is unknown when a prompt version that took effect there has a
// malformed llm.cost.
const promptCostAt = (promptList, generation) => {
  const costs = promptList.filter((p) => p.program_generation === generation).map(promptCost)
  return costs.includes(null) ? null : costs.reduce((a, b) => a + b, 0)
}
const targetPrompts = promptsByPath.get(target.path) ?? []
let running = -Infinity
const progress = generations.map((generation) => {
  const group = full.filter((p) => p.generation === generation)
  const valid = group.map(score).filter((v) => v !== null)
  if (valid.length) running = Math.max(running, ...valid)
  const costs = group.map(programCost)
  const known = costs.every((c) => c !== null)
  const metaCost = known ? group.reduce((a, p) => a + (isCopy(p) ? 0 : amount(p.metadata?.meta_cost)), 0) : null
  return {
    generation,
    programs: group.length,
    failed: group.filter((p) => !p.correct).length,
    best: valid.length ? round(Math.max(...valid)) : null,
    cumulative: Number.isFinite(running) ? round(running) : null,
    median: round(median(valid)),
    modelCost: known ? round(costs.reduce((a, b) => a + b, 0) - metaCost, 5) : null,
    metaCost: metaCost === null ? null : round(metaCost, 5),
    promptCost: promptCostAt(targetPrompts, generation) === null ? null : round(promptCostAt(targetPrompts, generation), 5),
  }
})
// Model posteriors read as the original Ensembling view reads them (viz_tree.html:11358-11460): programs with a
// non-empty llm_result.model_posteriors, or metadata.model_posteriors when no program has one; the models are
// every key of those maps; each generation averages its programs; a model's average is the mean of its
// per-generation means and its latest value the last of them. OD-4: trajectory in Progress, average in Contest.
const nonEmpty = (map) => Boolean(map) && typeof map === 'object' && Object.keys(map).length > 0
const fromResult = full.filter((p) => nonEmpty(p.metadata?.llm_result?.model_posteriors))
const withPosteriors = fromResult.length ? fromResult : full.filter((p) => nonEmpty(p.metadata?.model_posteriors))
const posteriorOf = (p) => p.metadata?.llm_result?.model_posteriors || p.metadata?.model_posteriors || {}
const mean = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : null)
const posteriorModels = [...new Set(withPosteriors.flatMap((p) => Object.keys(posteriorOf(p))))].sort()
const posteriorMeans = [...new Set(withPosteriors.map((p) => p.generation))].sort((a, b) => a - b).map((generation) => {
  const group = withPosteriors.filter((p) => p.generation === generation)
  return [generation, Object.fromEntries(posteriorModels.map((m) => [m, mean(group.map((p) => posteriorOf(p)[m]).filter((v) => v !== undefined && v !== null))]))]
})
const posteriors = posteriorMeans.map(([generation, means]) => [generation, Object.fromEntries(posteriorModels.map((m) => [clean(m), round(means[m])]))])
const models = [...new Set([...full.map((p) => p.metadata?.model_name).filter(Boolean), ...posteriorModels])].sort()
const modelStats = models.map((model) => {
  const group = full.filter((p) => p.metadata?.model_name === model)
  const valid = group.map(score).filter((v) => v !== null)
  const costs = group.map(programCost)
  const posts = posteriorMeans.map(([, means]) => means[model] ?? null).filter((v) => v !== null)
  return { model: clean(model), programs: group.length, success: group.length ? round(valid.length / group.length, 3) : null, best: valid.length ? round(Math.max(...valid)) : null, mean: valid.length ? round(valid.reduce((a, b) => a + b, 0) / valid.length) : null, cost: group.length && costs.every((c) => c !== null) ? round(costs.reduce((a, b) => a + b, 0), 4) : null, posteriorMean: round(mean(posts)), posteriorLatest: round(posts.at(-1) ?? null) }
}).filter((s) => s.programs > 0 || s.posteriorMean > 0)
const best = full[bestIndex]
const parent = best.parent_id ? full[index.get(best.parent_id)] : null
const metricKeys = Object.keys(best.public_metrics ?? {}).filter((k) => typeof best.public_metrics[k] === 'number').slice(0, 8)
const analysis = {
  run: target.name,
  direction: 'd1-momentum-regimes', hypothesis: 'h001', experiment: 'e01',
  stats: { programs: full.length, failed: full.filter((p) => !p.correct).length, generations: generations.length, islands: new Set(full.map((p) => p.island_idx)).size, models: models.length },
  rows, bestIndex, path, progress, modelStats, metricKeys, posteriors,
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
// The selected program's own summary (OD-3: the inspector's meta tab) is its block under
// "# INDIVIDUAL PROGRAM SUMMARIES" in the newest meta file that mentions it.
const programSummary = async (program) => {
  const tag = `(${program.id.slice(0, 8)},`
  for (const update of [...metaUpdates].reverse()) {
    const content = (await get('/get_meta_content?' + query({ db_path: target.path, processed_count: update.processed }))).content
    const lines = content.split('\n')
    const start = lines.findIndex((line) => line.startsWith('**Program Name:') && line.includes(tag))
    if (start === -1) continue
    const out = [clean(lines[start])]
    for (const line of lines.slice(start + 1)) { if (line.startsWith('**Program Name:') || /^# /.test(line)) break; if (line.trim()) out.push(clean(line)) }
    return { file: update.file, processed: update.processed, lines: out }
  }
  return null
}
analysis.best.metaSummary = await programSummary(best)
analysis.noise = analysisNoise
// Short program ids in the same order as rows, so tables can name programs that share a patch name.
analysis.ids = full.map((p) => p.id.slice(0, 8))
if (analysis.ids.length !== analysis.rows.length) throw new Error('fixture check failed: one id per program row')
const promptName = new Map(targetPrompts.map((p) => [p.id, clean(p.name)]))
analysis.prompts = targetPrompts.map((p) => ({ name: clean(p.name), generation: p.program_generation, parent: p.parent_id ? promptName.get(p.parent_id) ?? null : null, patchType: clean(p.patch_type), description: clean(p.description), programs: p.program_count, correct: p.correct_program_count, fitness: round(p.fitness), cost: round(promptCost(p), 5) })).sort((a, b) => a.generation - b.generation)

// ---------- multi-run comparison (OD-5: the compare page reuses the contest components) ----------
// For the three runs selected on the runs frame: cumulative best by generation and by cumulative
// spend (programs plus prompt evolution, OD-9), and the valid programs for a cross-run leaderboard.
const COMPARE = ['h001/e01', 'h002/e01', 'h003/e02'].map((key) => databases.find((d) => d.name.startsWith('d1-momentum-regimes') && d.path.includes(key)))
const compare = COMPARE.map((db) => {
  const programs = programsByPath.get(db.path)
  const promptList = promptsByPath.get(db.path) ?? []
  const gens = [...new Set(programs.map((p) => p.generation))].sort((a, b) => a - b)
  let bestSoFar = -Infinity
  // Once any program or prompt version lacks a recorded cost, the cumulative spend from that
  // generation on is unknown (null), never a cheaper-looking curve.
  let spent = 0
  const curve = gens.map((generation) => {
    const group = programs.filter((p) => p.generation === generation)
    for (const s of group.map(score)) if (s !== null) bestSoFar = Math.max(bestSoFar, s)
    const costs = group.map(programCost)
    const promptSpend = promptCostAt(promptList, generation)
    if (spent !== null) spent = costs.every((c) => c !== null) && promptSpend !== null ? spent + costs.reduce((a, b) => a + b, 0) + promptSpend : null
    return [generation, Number.isFinite(bestSoFar) ? round(bestSoFar) : null, spent === null ? null : round(spent, 4)]
  })
  const top = programs.filter((p) => score(p) !== null).sort((a, b) => score(b) - score(a)).slice(0, 4).map((p) => ({ id: p.id.slice(0, 8), name: clean(p.metadata?.patch_name ?? ''), generation: p.generation, score: round(score(p)), cost: round(programCost(p), 5) }))
  return { name: db.name, curve, top }
})

// ---------- one cost definition, checked (OD-9; unknown is never zero) ----------
const near = (a, b, tolerance) => Math.abs(a - b) <= tolerance
const assert = (ok, message) => { if (!ok) throw new Error(`fixture check failed: ${message}`) }
const analysisRun = runs.find((r) => r.name === target.name)
const split = progress.map((g) => (g.modelCost === null || g.metaCost === null || g.promptCost === null ? null : g.modelCost + g.metaCost + g.promptCost))
assert(analysisRun.cost === null ? split.some((v) => v === null) : split.every((v) => v !== null) && near(split.reduce((a, b) => a + b, 0), analysisRun.cost, 0.001), `analysis run generation costs sum to the run cost (${analysisRun.cost})`)
for (const d of directions.filter((x) => !x.error)) {
  const own = runs.filter((r) => r.name.startsWith(`${d.id}/`))
  assert(d.unknownCostRuns === own.filter((r) => r.cost === null).length, `${d.id}: the snapshot's unknown-cost runs match the run statistics`)
  if (d.spendKnown) assert(near(own.reduce((a, r) => a + r.cost, 0), d.spend, 0.001), `${d.id}: its runs equal the notebook spend (${d.spend})`)
}
for (const c of compare) {
  const run = runs.find((r) => r.name === c.name)
  assert(run.cost === null ? c.curve.at(-1)[2] === null : near(c.curve.at(-1)[2], run.cost, 0.001), `${c.name}: cumulative spend ends at the run cost (${run.cost})`)
}
const fixture = { schema: 'shinka-research.hifi-fixture.v1', source: 'demo-20261019 · DEMO · 演示数据，非真实研究', workspace: { name: workspace.name, library: workspace.library ? { pages: workspace.library.pages?.length ?? null, pending: workspace.library.pending?.length ?? null } : null }, directions, events, report, runs, analysis, compare }
writeFileSync(new URL('./fixture.json', import.meta.url), JSON.stringify(fixture) + '\n')
console.log(JSON.stringify({ directions: directions.map((d) => [d.id, d.state, d.trials, d.evaluated, d.evidence?.failed, d.spend, d.spendKnown]), runs: runs.map((r) => [r.name, r.cost, r.valid, r.stall, r.prompts]), programs: full.length, best: analysis.best.score, metaSummary: analysis.best.metaSummary?.file ?? null, noise: analysis.noise.length, compare: compare.map((c) => [c.name, c.curve.length, c.curve.at(-1)]) }))
