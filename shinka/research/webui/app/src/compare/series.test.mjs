// The compare page's curves, costs and leads (award-redesign 5.2); node strips the TypeScript types.
import assert from 'node:assert/strict'
import test from 'node:test'
import { byGeneration, byGroup, bySpend, crossBoard, groupBoards, groupLeads, leads, MAX_POINTS, metricOf, programCost, scoreOf, seriesStyle, stackLabels, stripDomain, valueAt } from './series.ts'

const program = (id, generation, score, extra = {}) => ({
  id,
  generation,
  timestamp: 'timestamp' in extra ? extra.timestamp : generation * 10,
  correct: score !== null,
  combined_score: score,
  metadata: { api_costs: extra.cost ?? 0.1, ...extra.metadata },
})
const prompt = (timestamp, generation, cost) => ({ timestamp, program_generation: generation, metadata: { llm: { cost } } })
const xy = (points) => points.map((p) => [p.x, Math.round(p.y * 1e6) / 1e6])

test('by generation each metric accumulates, failures never raise the best, and generations sort as numbers', () => {
  const programs = [program('a', 10, 0.5), program('b', 9, 0.2), program('c', 9, null), program('d', null, 0.1), program('e', 10, 0.4)]
  assert.deepEqual(xy(byGeneration(programs, 'best_score')), [[0, 0.1], [9, 0.2], [10, 0.5]])
  assert.deepEqual(xy(byGeneration(programs, 'program_count')), [[0, 1], [9, 3], [10, 5]])
  assert.deepEqual(xy(byGeneration(programs, 'correct_count')), [[0, 1], [9, 2], [10, 4]])
  // Until a program is valid the best has no value, so its curve starts later.
  assert.deepEqual(xy(byGeneration([program('f', 0, null), program('g', 1, 0.3)], 'best_score')), [[1, 0.3]])
})

test('only true, 1 and "true" pass an evaluation; a "false" string scores and counts as a failure', () => {
  const as = (correct) => ({ id: String(correct), generation: 0, correct, combined_score: 0.9 })
  assert.deepEqual([true, 1, 'true'].map((c) => scoreOf(as(c))), [0.9, 0.9, 0.9])
  assert.deepEqual([false, 0, 'false', '1', null, undefined].map((c) => scoreOf(as(c))), [null, null, null, null, null, null])
  assert.deepEqual(xy(byGeneration([as(1), as('false'), as(true)], 'correct_count')), [[0, 2]])
})

test('by spend programs and prompts charge in time order and a late prompt moves the end to the total', () => {
  const run = {
    programs: [program('a', 0, 0.2, { timestamp: 100, cost: 1 }), program('b', 1, 0.6, { timestamp: 300, cost: 2 })],
    prompts: [prompt(200, 0, 0.5), prompt(400, 1, 0.25)],
  }
  assert.deepEqual(xy(bySpend(run, 'best_score')), [[1, 0.2], [3.5, 0.6], [3.75, 0.6]])
})

test('without a time on every charge the clock is the generation', () => {
  const run = {
    programs: [program('a', 2, 0.2, { timestamp: null, cost: 1 }), program('b', 1, 0.6, { timestamp: 50, cost: 2 })],
    prompts: [prompt(10, 2, 0.5)],
  }
  assert.deepEqual(xy(bySpend(run, 'program_count')), [[2, 1], [3, 2], [3.5, 2]])
})

test('paid work without a program moves the spend curve end to the run total', () => {
  const programs = [program('a', 0, 0.2, { timestamp: 1, cost: 1 }), program('b', 1, 0.6, { timestamp: 2, cost: 2 })]
  // The server's total holds a discarded surplus proposal no program carries.
  const stats = { total_cost: 3.5, prompt_evo_cost: 0 }
  assert.deepEqual(xy(bySpend({ programs, prompts: [], stats }, 'best_score')), [[1, 0.2], [3, 0.6], [3.5, 0.6]])
  // An unknown total, or the charges' own unknown spend, leaves the end where the charges put it.
  assert.deepEqual(xy(bySpend({ programs, prompts: [], stats: { total_cost: null, prompt_evo_cost: 0 } }, 'best_score')), [[1, 0.2], [3, 0.6]])
  const headless = { metadata: { model_name: 'headless/agent', headless_pricing_unknown: true } }
  assert.deepEqual(xy(bySpend({ programs: [programs[0], program('h', 1, 0.6, { timestamp: 2, ...headless })], prompts: [], stats }, 'best_score')), [[1, 0.2]])
})

test('the spend curve ends at the first charge without a price', () => {
  const headless = { metadata: { model_name: 'headless/agent', headless_pricing_unknown: true } }
  const run = {
    programs: [program('a', 0, 0.2, { timestamp: 1, cost: 1 }), program('b', 1, 0.6, { timestamp: 2, ...headless }), program('c', 2, 0.9, { timestamp: 3 })],
    prompts: [],
  }
  assert.deepEqual(xy(bySpend(run, 'best_score')), [[1, 0.2]])
  // An unpriced prompt ends it the same way.
  assert.deepEqual(xy(bySpend({ programs: [program('a', 0, 0.2, { timestamp: 1, cost: 1 })], prompts: [prompt(0, 0, 'n/a')] }, 'best_score')), [])
})

test('a long spend curve keeps every k-th point and its true end', () => {
  const programs = Array.from({ length: 450 }, (_, i) => program(`p${i}`, i, i / 1000, { timestamp: i, cost: 1 }))
  const points = bySpend({ programs, prompts: [] }, 'program_count')
  assert.ok(points.length <= MAX_POINTS)
  assert.deepEqual(xy(points.slice(0, 2)), [[1, 1], [4, 4]])
  assert.deepEqual(xy(points.slice(-1)), [[450, 450]])
})

test('a program costs its recorded fields; copies are free and an unpriced headless call is unknown', () => {
  assert.equal(programCost({ id: 'a', metadata: { api_costs: 0.5, embed_cost: null, meta_cost: 0.25 } }), 0.75)
  assert.equal(programCost({ id: 'b', metadata: { api_costs: 3, _is_island_copy: true } }), 0)
  assert.equal(programCost({ id: 'c', metadata: { api_costs: 0, llm_result: { model_name: 'headless/x' }, headless_pricing_unknown: 1 } }), null)
  assert.equal(programCost({ id: 'd', metadata: { api_costs: -1 } }), null)
})

test('the leaderboard ranks every valid program across runs, island copies included, and leaves failures out', () => {
  const runs = [
    [program('a', 1, 0.9), program('copy', 2, 0.92, { metadata: { _spawned_island: true } }), program('x', 3, null)],
    [program('b', 1, 0.95), program('c', 2, 0.1)],
  ]
  assert.deepEqual(crossBoard(runs, 3).map((e) => [e.run, e.program.id, e.score, e.cost]), [[1, 'b', 0.95, 0.1], [0, 'copy', 0.92, 0], [0, 'a', 0.9, 0.1]])
})

test('one run cannot fill the leaderboard: each run brings at most its best four', () => {
  const strong = Array.from({ length: 6 }, (_, i) => program(`s${i}`, i, 0.9 - i / 100))
  const weak = Array.from({ length: 6 }, (_, i) => program(`w${i}`, i, 0.5 - i / 100))
  const board = crossBoard([strong, weak])
  assert.deepEqual(board.map((e) => e.program.id), ['s0', 's1', 's2', 's3', 'w0', 'w1', 'w2', 'w3'])
})

test('a row leads at the shown precision, ties share it, and an unknown that could be better blocks it', () => {
  const two = (v) => v.toFixed(2)
  assert.deepEqual(leads([1.03, 1.1, 0.88], 'max', two), [false, true, false])
  assert.deepEqual(leads([0.041, 0.039, 0.05], 'min', two), [true, true, false])
  assert.deepEqual(leads([3, 3], 'min', two), [false, false])
  assert.deepEqual(leads([null, 0.8], 'max', two), [false, true])
  assert.deepEqual(leads([9.31, null], 'min', two, true), [false, false])
})

test('eighteen runs get eighteen styles and the first three are the hi-fi ones', () => {
  const styles = Array.from({ length: 18 }, (_, k) => seriesStyle(k))
  assert.equal(new Set(styles.map((s) => `${s.mark}|${s.dash}`)).size, 18)
  assert.deepEqual(styles.slice(0, 3).map((s) => [s.mark, s.dash, s.muted]), [[0, '', false], [1, '6 4', false], [2, '1.5 3', true]])
})

test('end labels keep their order and gap, and stay inside the plot', () => {
  assert.deepEqual(stackLabels([50, 52, 10], 14, 0, 200), [50, 64, 10])
  assert.deepEqual(stackLabels([195, 198], 14, 0, 200), [186, 200])
})

test('an unknown metric in the URL reads as the cumulative best', () => {
  assert.equal(metricOf('correct_count'), 'correct_count')
  assert.equal(metricOf('nope'), 'best_score')
})

test('a reading takes the step in force and has no value outside the curve', () => {
  const points = [{ x: 2, y: 0.3 }, { x: 5, y: 0.6 }, { x: 9, y: 0.7 }]
  assert.deepEqual([1, 2, 4, 5, 8, 9, 10].map((at) => valueAt(points, at)), [null, 0.3, 0.3, 0.6, 0.6, 0.7, null])
})

test('runs of different tasks are ranked and led only within their task, named by their place in the selection', () => {
  // A, B, A: the second task's run sits between the first task's two.
  const groups = byGroup(['bin', 'circle', 'bin'])
  assert.deepEqual(groups, [[0, 2], [1]])
  const runs = [[program('a', 1, 0.5)], [program('c', 1, 2.4)], [program('b', 1, 0.9)]]
  assert.deepEqual(groupBoards(runs, groups).map((board) => board.map((e) => [e.run, e.program.id])), [[[2, 'b'], [0, 'a']], [[1, 'c']]])
  // The circle run's 2.4 does not beat the bin runs, and a task's lone run is not judged.
  assert.deepEqual(groupLeads([0.5, 2.4, 0.9], groups, 'max', (v) => v.toFixed(2)), [false, null, true])
})

test('a row scale holds negative scores and stops at zero for amounts', () => {
  // A research run's best can be below zero (the controlled BTC run scored −0.17).
  const [lo, hi] = stripDomain([-0.17, -0.05])
  assert.ok(lo < -0.17 && hi > -0.05, `[${lo}, ${hi}]`)
  assert.deepEqual(stripDomain([0.01, 2]).map((v) => Math.round(v * 1000) / 1000), [0, 2.597])
  assert.deepEqual(stripDomain([5, 5]), [4.5, 5.5])
})
