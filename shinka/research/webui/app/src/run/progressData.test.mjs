// The progress view's derived data (award-redesign 6.3); node strips the TypeScript types.
import assert from 'node:assert/strict'
import test from 'node:test'
import { byComputeTime, costTotals, generationRows, metricKeys, modelRows, noiseByGeneration, posteriorSeries } from './progressData.ts'

const p = (id, generation, score, metadata = {}, extra = {}) => ({
  id,
  generation,
  timestamp: generation * 10,
  correct: score !== null,
  // A failed evaluation can still carry a raw score; it must never count as one.
  combined_score: score ?? 99,
  metadata,
  ...extra,
})

test('a failed program never becomes a generation’s best, the running best or part of its median', () => {
  const rows = generationRows([p('a', 0, 0.2), p('f', 1, null), p('b', 1, 0.1), p('c', 1, 0.3), p('d', 2, 0.25)], [], true)
  assert.deepEqual(
    rows.map((r) => [r.generation, r.programs, r.failedIds, r.best, r.bestId, r.cumulative, r.cumulativeId, r.median]),
    [
      [0, 1, [], 0.2, 'a', 0.2, 'a', 0.2],
      [1, 3, ['f'], 0.3, 'c', 0.3, 'c', 0.2],
      [2, 1, [], 0.25, 'd', 0.3, 'c', 0.25],
    ],
  )
})

test('a generation’s cost splits into model calls, meta-analysis and prompt evolution, and an unpriced part is unknown, not smaller', () => {
  const programs = [
    p('a', 0, 0.1, { api_costs: 0.3, meta_cost: 0.1 }),
    p('copy', 0, 0.1, { api_costs: 0.3, meta_cost: 0.1, _is_island_copy: true }),
    p('b', 1, 0.2, { api_costs: 'n/a' }),
  ]
  // A prompt version that does not say when it took effect is in no generation.
  const prompts = [{ id: 'q0', program_generation: 0, metadata: { llm: { cost: 0.05 } } }, { id: 'q1', metadata: { llm: { cost: 0.4 } } }]
  const [g0, g1] = generationRows(programs, prompts, true)
  assert.deepEqual([g0.modelCost?.toFixed(2), g0.metaCost?.toFixed(2), g0.promptCost], ['0.30', '0.10', 0.05])
  assert.deepEqual([g1.modelCost, g1.metaCost, g1.promptCost], [null, null, 0])
  // Prompt versions that could not be read leave every generation's prompt spend unknown.
  assert.equal(generationRows(programs, prompts, false)[0].promptCost, null)
})

test('the run’s cost parts add up to its total, and spend no program carries shows as unattributed', () => {
  const programs = [p('a', 0, 0.1, { api_costs: 0.5, embed_cost: 0.1, novelty_cost: 0.2, meta_cost: 0.2 }), p('copy', 1, 0.1, { api_costs: 0.5, _is_island_copy: true })]
  const t = costTotals(programs, 0.3, 1.5)
  assert.deepEqual([t.api, t.embed, t.novelty, t.meta, t.prompt, t.total, t.perProgram], [0.5, 0.1, 0.2, 0.2, 0.3, 1.5, 1.5])
  assert.equal(t.unattributed?.toFixed(2), '0.20')
  // An unknown run total leaves the remainder and the cost per program unknown, never a smaller figure.
  const unknown = costTotals(programs, null, null)
  assert.deepEqual([unknown.total, unknown.unattributed, unknown.perProgram], [null, null, null])
})

test('the noise line follows this run’s trials only, taking each generation’s last bar', () => {
  const points = [
    { n: 1, generation: 0, program: 'a', bar: 0.1 },
    { n: 2, generation: 0, program: 'other-run', bar: 0.9 },
    { n: 3, generation: 0, program: 'b', bar: 0.15 },
    { n: 4, generation: 2, program: 'c', bar: 0.2 },
  ]
  assert.deepEqual([...noiseByGeneration(points, new Set(['a', 'b', 'c']))], [[0, 0.15], [2, 0.2]])
})

test('posteriors come from the LLM result when any program has them, averaged per generation, and the model table reads them', () => {
  const programs = [
    p('a', 0, 0.2, { model_name: 'm1', llm_result: { model_posteriors: { m1: 0.6, m2: 0.4 } }, model_posteriors: { m1: 0.1 } }),
    p('b', 0, null, { model_name: 'm2', llm_result: { model_posteriors: { m1: 0.4, m2: 0.6 } } }),
    p('c', 1, 0.5, { model_name: 'm1', model_posteriors: { m1: 0.9, m2: 0.1 } }),
  ]
  const series = posteriorSeries(programs)
  assert.deepEqual(series, { models: ['m1', 'm2'], rows: [{ generation: 0, means: { m1: 0.5, m2: 0.5 } }] })
  const [m1, m2] = modelRows(programs, series)
  assert.deepEqual([m1.model, m1.programs, m1.success, m1.best, m1.mean, m1.posteriorLatest], ['m1', 2, 1, 0.5, 0.35, 0.5])
  assert.deepEqual([m2.model, m2.programs, m2.success, m2.best], ['m2', 1, 0, null])
})

test('only public metrics of valid programs are offered, and time-based progress stops at the first unrecorded time', () => {
  const programs = [
    p('a', 0, 0.2, { compute_time: 10 }, { public_metrics: { sharpe: 1.2, label: 'x' }, private_metrics: { holdout: 0.9 } }),
    p('f', 0, null, { compute_time: 5 }, { public_metrics: { failed_only: 1 } }),
    p('b', 1, 0.4, {}, { public_metrics: { sharpe: 1.4 } }),
  ]
  assert.deepEqual(metricKeys(programs), ['sharpe'])
  assert.deepEqual(byComputeTime(programs), { points: [{ x: 10, best: 0.2, id: 'a' }, { x: 15, best: 0.2, id: 'a' }], complete: false })
})
