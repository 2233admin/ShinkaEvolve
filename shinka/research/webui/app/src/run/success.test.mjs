// The progress view's success and improvement statistics (award-redesign 6.3); node strips the TypeScript types.
import assert from 'node:assert/strict'
import test from 'node:test'
import {
  llmStats,
  modelLabels,
  mutationStats,
  noveltyLlmStats,
  noveltyMutationStats,
  resampleLlmStats,
  resampleMutationStats,
} from './success.ts'

const prog = (id, parent, score, metadata = {}, correct = true) => ({ id, parent_id: parent, correct, combined_score: score, metadata })

const root = prog('r', null, 0.5, { model_name: 'zeta', patch_type: 'init' })
const run = [
  root,
  prog('a', 'r', 0.6, { model_name: 'zeta', patch_type: 'diff', patch_attempt: 2 }),
  // Scores above its parent but failed evaluation: improved, not a correct improvement.
  prog('b', 'r', 0.7, { model_name: 'zeta', patch_type: 'full' }, false),
  prog('c', 'r', 0.4, { model_name: 'alpha', patch_type: 'diff' }),
  // Its parent is not among the programs.
  prog('d', 'gone', 0.9, { model_name: 'zeta', patch_type: 'diff', patch_attempt: 3 }),
  prog('e', 'r', 0.5, {}),
]

const cell = (patchType, attempts, successes, improved, correct, improvedCorrect) => ({
  patchType,
  attempts,
  successes,
  improved,
  correct,
  improvedCorrect,
  successRate: attempts > 0 ? successes / attempts : null,
  improvementRate: attempts > 0 ? improvedCorrect / attempts : null,
})

test('the model matrix counts children whose parent is present, by model and patch type in first-seen order, with an empty cell for a pair no program had', () => {
  assert.deepEqual(llmStats(run), {
    patchTypes: ['diff', 'full', 'unknown'],
    rows: [
      { model: 'zeta', cells: [cell('diff', 2, 1, 1, 1, 1), cell('full', 1, 1, 1, 0, 0), cell('unknown', 0, 0, 0, 0, 0)] },
      { model: 'alpha', cells: [cell('diff', 1, 1, 0, 1, 0), cell('full', 0, 0, 0, 0, 0), cell('unknown', 0, 0, 0, 0, 0)] },
      { model: 'unknown', cells: [cell('diff', 0, 0, 0, 0, 0), cell('full', 0, 0, 0, 0, 0), cell('unknown', 1, 1, 0, 1, 0)] },
    ],
  })
  const zeta = llmStats(run).rows[0].cells
  assert.equal(zeta[0].successRate, 0.5)
  assert.equal(zeta[0].improvementRate, 0.5)
  assert.equal(zeta[2].successRate, null)
})

test('the patch type bars count a child whose parent is missing, which never improves, and stack the tries that failed before each program', () => {
  assert.deepEqual(mutationStats(run), [
    { ...cell('diff', 6, 3, 1, 3, 1), failures: 3 },
    { ...cell('full', 1, 1, 1, 0, 0), failures: 0 },
    { ...cell('unknown', 1, 1, 0, 1, 0), failures: 0 },
  ])
})

test('an improvement needs both scores recorded; a missing score is not zero', () => {
  const programs = [
    prog('p', null, null),
    prog('q', null, -1),
    prog('x', 'p', 0.1, { patch_type: 'diff' }),
    prog('y', 'q', null, { patch_type: 'diff' }),
    prog('z', 'q', -0.5, { patch_type: 'diff' }),
  ]
  const [diff] = mutationStats(programs)
  assert.equal(diff.improved, 1)
  assert.equal(diff.improvedCorrect, 1)
})

test('only a positive attempt counter is read; anything else counts as one try', () => {
  const programs = [root, prog('a', 'r', 0.1, { patch_type: 'diff', patch_attempt: -2 }), prog('b', 'r', 0.1, { patch_type: 'diff', patch_attempt: '4' })]
  const [diff] = mutationStats(programs)
  assert.equal(diff.attempts, 2)
  assert.equal(diff.failures, 0)
})

test('the novelty statistics keep children with a positive novelty attempt, which is also their tries', () => {
  const programs = [
    root,
    prog('a', 'r', 0.9, { model_name: 'm', patch_type: 'diff', novelty_attempt: 3, patch_attempt: 5 }),
    prog('b', 'r', 0.9, { model_name: 'm', patch_type: 'diff', novelty_attempt: 0 }),
    prog('c', 'r', 0.9, { model_name: 'm', patch_type: 'diff', novelty_attempt: -1 }),
    prog('d', 'r', 0.9, { model_name: 'm', patch_type: 'diff', novelty_attempt: '2' }),
    prog('e', 'r', 0.9, { model_name: 'm', patch_type: 'full' }),
    prog('f', 'gone', 0.9, { model_name: 'm', patch_type: 'full', novelty_attempt: 1 }),
  ]
  assert.deepEqual(noveltyLlmStats(programs), { patchTypes: ['diff'], rows: [{ model: 'm', cells: [cell('diff', 3, 1, 1, 1, 1)] }] })
  assert.deepEqual(noveltyMutationStats(programs), [
    { ...cell('diff', 3, 1, 1, 1, 1), failures: 2 },
    { ...cell('full', 1, 1, 0, 1, 0), failures: 0 },
  ])
})

test('the resample statistics keep children with a positive resample attempt, which is also their tries', () => {
  const programs = [
    root,
    prog('a', 'r', 0.2, { model_name: 'm', patch_type: 'fix', resample_attempt: 2, patch_attempt: 4, novelty_attempt: 1 }, false),
    prog('b', 'r', 0.9, { model_name: 'm', patch_type: 'fix', novelty_attempt: 2 }),
  ]
  assert.deepEqual(resampleLlmStats(programs), { patchTypes: ['fix'], rows: [{ model: 'm', cells: [cell('fix', 2, 1, 0, 0, 0)] }] })
  assert.deepEqual(resampleMutationStats(programs), [{ ...cell('fix', 2, 1, 0, 0, 0), failures: 1 }])
})

test('no programs is no data; programs without qualifying children are no groups', () => {
  for (const stats of [llmStats, mutationStats, noveltyLlmStats, noveltyMutationStats, resampleLlmStats, resampleMutationStats]) {
    assert.equal(stats([]), null)
  }
  for (const stats of [llmStats, noveltyLlmStats, resampleLlmStats]) assert.deepEqual(stats([root]), { patchTypes: [], rows: [] })
  for (const stats of [mutationStats, noveltyMutationStats, resampleMutationStats]) assert.deepEqual(stats([root]), [])
  assert.deepEqual(noveltyLlmStats(run), { patchTypes: [], rows: [] })
  assert.deepEqual(resampleMutationStats(run), [])
})

test('model labels drop the shared prefix unless a name would be left short or start with punctuation', () => {
  assert.deepEqual(modelLabels(['openai/gpt-4o', 'openai/o3-mini']), ['gpt-4o', 'o3-mini'])
  assert.deepEqual(modelLabels(['gpt-4o', 'gpt-4o-mini']), ['gpt-4o', 'gpt-4o-mini'])
  assert.deepEqual(modelLabels(['model-x1', 'model_x2']), ['model-x1', 'model_x2'])
  assert.deepEqual(modelLabels(['openai/gpt-4o']), ['openai/gpt-4o'])
  // The prefix is the one the alphabetically first and last names share.
  assert.deepEqual(modelLabels(['ab-xyz', 'ac-zzz', 'ab-qrs']), ['b-xyz', 'c-zzz', 'b-qrs'])
})
