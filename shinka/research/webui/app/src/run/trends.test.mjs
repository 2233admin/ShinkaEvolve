// The progress view's generation trends (award-redesign 6.3); node strips the TypeScript types.
import assert from 'node:assert/strict'
import test from 'node:test'
import { codeBloat, correctnessRates, improvementRates } from './trends.ts'

const prog = (id, generation, { correct = true, score = 1, parent = null, type, at, attempt, metrics } = {}) => ({
  id,
  generation,
  correct,
  combined_score: score,
  parent_id: parent,
  metadata: { patch_type: type, timestamp: at, patch_attempt: attempt, code_analysis_metrics: metrics },
})
const point = (generation, value, n) => ({ generation, value, n })

test('the correctness rate runs from the start of the run, one point per program in generation then timestamp order', () => {
  const programs = [
    prog('a', 1, { type: 'diff', at: 5 }),
    prog('b', 0, { type: 'init' }),
    prog('c', 1, { type: 'full', at: 2, correct: false }),
    // Ties a on generation and timestamp: input order decides.
    prog('d', 1, { type: 'diff', at: 5, correct: 0 }),
    // A numeric-string timestamp compares as a number: before 5 in generation 2, after 3.
    prog('e', 2, { type: 'mystery', at: 5 }),
    prog('f', 2, { type: 'diff', at: '4', correct: 'false' }),
    prog('g', 2, { at: 3, correct: 1 }),
  ]
  const { overall, byType } = correctnessRates(programs)
  // Walk: b, c, a, d, g, f, e.
  assert.deepEqual(overall, [
    point(0, 100, 1),
    point(1, 50, 2),
    point(1, (2 / 3) * 100, 3),
    point(1, 50, 4),
    point(2, 60, 5),
    point(2, 50, 6),
    point(2, (4 / 7) * 100, 7),
  ])
  // Only the five legacy types get a line, in the legacy order, and only when they have a point.
  assert.deepEqual(byType, [
    { type: 'diff', points: [point(1, 100, 1), point(1, 50, 2), point(2, (1 / 3) * 100, 3)] },
    { type: 'full', points: [point(1, 0, 1)] },
    { type: 'init', points: [point(0, 100, 1)] },
  ])
})

test('the improvement rate counts programs whose parent and both scores are known, over their patch attempts', () => {
  const programs = [
    prog('r', 0, { score: 1, type: 'init' }),
    prog('x', 1, { parent: 'r', score: 2, type: 'diff', attempt: 3 }),
    prog('y', 1, { parent: 'r', score: 0.5, type: 'full', attempt: 0 }),
    // A failed program counts as an attempt, never as an improvement, whatever it scored.
    prog('q', 1, { parent: 'r', score: 5, type: 'diff', correct: false }),
    // Beating a failed parent's recorded score still counts.
    prog('z', 2, { parent: 'q', score: 6, type: 'diff' }),
    prog('orphan', 2, { parent: 'gone', score: 9, type: 'diff' }),
    prog('unscored', 2, { parent: 'r', score: null, correct: false, type: 'diff' }),
    prog('childOfUnscored', 3, { parent: 'unscored', score: 9, type: 'diff' }),
  ]
  const { overall, byType } = improvementRates(programs)
  assert.deepEqual(overall, [point(1, (1 / 3) * 100, 3), point(1, 25, 4), point(1, 20, 5), point(2, (2 / 6) * 100, 6)])
  assert.deepEqual(byType, [
    { type: 'diff', points: [point(1, (1 / 3) * 100, 3), point(1, 25, 4), point(2, 40, 5)] },
    { type: 'full', points: [point(1, 0, 1)] },
  ])
})

test('no program, or none with a scored parent, gives empty rate series', () => {
  assert.deepEqual(correctnessRates([]), { overall: [], byType: [] })
  assert.deepEqual(improvementRates([]), { overall: [], byType: [] })
  assert.deepEqual(improvementRates([prog('r', 0), prog('x', 1, { parent: 'gone' })]), { overall: [], byType: [] })
})

test('lines of code average per generation, logical lines fall back to lines, and each gets a least-squares trend', () => {
  const { points, loc, lloc } = codeBloat([
    prog('a', 2, { metrics: { lines_of_code: 100, logical_lines_of_code: 80 } }),
    prog('b', 0, { metrics: { lines_of_code: 40 } }),
    prog('c', 2, { metrics: { lines_of_code: 50, logical_lines_of_code: 0 } }),
    // Without a line count, or without metrics at all, a program is left out; so is its generation.
    prog('d', 1, { metrics: { logical_lines_of_code: 999 } }),
    prog('e', 1),
  ])
  assert.deepEqual(points, [
    { generation: 0, loc: 40, lloc: 40, n: 1 },
    { generation: 2, loc: 75, lloc: 65, n: 2 },
  ])
  assert.deepEqual(loc, { slope: 17.5, intercept: 40 })
  assert.deepEqual(lloc, { slope: 12.5, intercept: 40 })
})

test('a single generation has points but no trend; a run without code metrics has neither', () => {
  const single = codeBloat([prog('a', 3, { metrics: { lines_of_code: 10 } }), prog('b', 3, { metrics: { lines_of_code: 20 } })])
  assert.deepEqual(single, { points: [{ generation: 3, loc: 15, lloc: 15, n: 2 }], loc: null, lloc: null })
  assert.deepEqual(codeBloat([prog('a', 0), prog('b', 1)]), { points: [], loc: null, lloc: null })
  assert.deepEqual(codeBloat([]), { points: [], loc: null, lloc: null })
})
