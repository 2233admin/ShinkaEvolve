// The run list's rules: what a reader can trust in the table depends on these, not on layout
// (node strips the TypeScript types).
import assert from 'node:assert/strict'
import test from 'node:test'
import { compareRuns, groupRuns, matchesRun, runCost, runKind, runLabel, runTotals, runtimeText } from './data.ts'

const exp = (dir, h, e) => ({ path: `workspace/${dir}/experiments/${h}/${e}/results/programs.sqlite`, name: `${dir}/experiments/${h}/${e}/results`, task: dir })
const ordinary = (task, run) => ({ path: `runs/${task}/${run}/programs.sqlite`, name: `${task}/${run}`, task })
const stats = (o) => ({ program_count: 10, correct_count: 8, best_score: 0.5, max_generation: 3, gens_since_improvement: 1, total_cost: 1, prompt_evo_cost: 0.5, prompt_count: 2, has_prompt_evo: true, total_runtime_seconds: 60, last_update: 100, ...o })

test('a workspace run is an experiment only under one of its directions', () => {
  const dirs = new Set(['d1'])
  assert.equal(runKind(exp('d1', 'h001', 'e01'), dirs), 'experiment')
  assert.equal(runKind({ path: 'workspace/loose/run/programs.sqlite', name: 'loose/run', task: 'loose' }, dirs), 'ordinary')
  assert.equal(runKind(ordinary('d1', 'run_1'), dirs), 'ordinary')
  assert.equal(runLabel(exp('d1', 'h001', 'e01')), 'experiments/h001/e01')
  assert.equal(runLabel(ordinary('pack', 'run_1')), 'run_1')
})

test('a run costs its programs plus prompt evolution, and an unpriced part leaves it unknown', () => {
  assert.equal(runCost(stats({})), 1.5)
  assert.equal(runCost(stats({ total_cost: null })), null)
  assert.equal(runCost(stats({ prompt_evo_cost: null })), null)
})

test('a search matches task and name word by word, in any case', () => {
  const r = ordinary('circle_packing', 'run_20261015')
  assert.ok(matchesRun(r, 'CIRCLE 2026'))
  assert.ok(!matchesRun(r, 'circle bin'))
  assert.ok(matchesRun(r, '  '))
})

test('numbers sort largest first and a run without the value goes last; groups follow their first run', () => {
  const a = ordinary('t1', 'a')
  const b = ordinary('t2', 'b')
  const c = ordinary('t1', 'c')
  const s = new Map([[a.path, stats({ best_score: 0.2, last_update: 300 })], [b.path, stats({ best_score: 0.9, last_update: 200 })]])
  assert.deepEqual([a, b, c].sort(compareRuns('score', s)).map((r) => r.path), [b.path, a.path, c.path])
  assert.deepEqual([c, b, a].sort(compareRuns('name', s)).map((r) => runLabel(r)), ['a', 'b', 'c'])
  const groups = groupRuns([a, b, c], 'updated', s, new Set())
  assert.deepEqual(groups.map((g) => [g.key, g.runs.map((r) => runLabel(r))]), [['t1', ['a', 'c']], ['t2', ['b']]])
})

const read = (o) => ({ stats: stats(o), failed: false })

test('totals count what has arrived and turn the cost into a lower bound for the rest', () => {
  const a = ordinary('t', 'a')
  const b = ordinary('t', 'b')
  const c = ordinary('t', 'c')
  const s = new Map([[a.path, read({ best_score: 0.4 })], [b.path, read({ total_cost: null, best_score: 0.7, correct_count: 10 })]])
  const t = runTotals([a, b, c], s)
  // The unpriced run's prompt cost (0.5) still counts toward the lower bound.
  assert.deepEqual([t.runs, t.programs, t.valid, t.failed, t.cost, t.partial, t.peak], [3, 20, 18, 2, 2, 2, 0.7])
})

test('a run whose latest read failed keeps its earlier numbers in the totals and makes the cost a lower bound, counted once', () => {
  const a = ordinary('t', 'a')
  const b = ordinary('t', 'b')
  const c = ordinary('t', 'c')
  const s = new Map([
    [a.path, { stats: stats({}), failed: true }],
    // Kept from an earlier read and unpriced: still one run with a partial cost.
    [b.path, { stats: stats({ total_cost: null }), failed: true }],
    [c.path, { stats: null, failed: true }],
  ])
  const t = runTotals([a, b, c], s)
  assert.deepEqual([t.programs, t.cost, t.partial], [20, 2, 3])
})

test('a runtime reads in minutes under an hour and in hours and minutes above', () => {
  assert.equal(runtimeText(null), '—')
  assert.equal(runtimeText(1790), '30m')
  assert.equal(runtimeText(7914), '2h 12m')
})
