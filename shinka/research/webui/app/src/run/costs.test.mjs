// The progress view's cost charts (award-redesign 6.3); node strips the TypeScript types.
import assert from 'node:assert/strict'
import test from 'node:test'
import { generationAvgCost, modelLabels, modelPatchCost } from './costs.ts'

// A program whose whole cost is its API cost; `metadata` adds island and pricing flags.
const p = (id, generation, model, patch, cost, metadata = {}) => ({
  id,
  generation,
  correct: true,
  combined_score: 1,
  metadata: { model_name: model, patch_type: patch, api_costs: cost, ...metadata },
})
const copy = { _is_island_copy: true }

test('the heatmap’s cell is the mean cost of its programs: an island copy counts at no cost, an unpriced program leaves the cell unknown', () => {
  const programs = [
    p('a1', 1, 'prov/gpt-a', 'diff', 0.25),
    p('a2', 1, 'prov/gpt-a', 'diff', 0.5),
    p('a3', 2, 'prov/gpt-a', 'diff', 0.5, copy),
    p('a4', 2, 'prov/gpt-a', 'full', 0.25),
    p('a5', 3, 'prov/gpt-a', undefined, 1),
    p('h1', 1, 'headless/claude', 'full', 0.5),
    p('h2', 2, 'headless/claude', 'full', 0, { headless_pricing_unknown: true }),
    p('b1', 0, 'prov/gpt-b', 'init', 1),
    p('n1', 1, undefined, 'diff', 0.25),
  ]
  assert.deepEqual(modelPatchCost(programs), {
    // The init column and the row without a model are dropped; a model with only init keeps its row.
    rows: ['headless/claude', 'prov/gpt-a', 'prov/gpt-b'],
    // Programs without a patch type keep their column, where the legacy key 'N/A' sorted.
    cols: [null, 'diff', 'full'],
    cells: [
      [null, null, null],
      [1, 0.25, 0.25],
      [null, null, null],
    ],
    counts: [
      [0, 0, 2],
      [1, 3, 1],
      [0, 0, 0],
    ],
  })
})

test('the heatmap’s rows and columns are in code-unit order, as the legacy page sorted them', () => {
  const { rows, cols } = modelPatchCost([p('1', 0, 'alpha', 'diff', 0), p('2', 0, 'Zeta', 'Full', 0)])
  assert.deepEqual(rows, ['Zeta', 'alpha'])
  assert.deepEqual(cols, ['Full', 'diff'])
})

test('no programs give an empty heatmap and no generations', () => {
  assert.deepEqual(modelPatchCost([]), { rows: [], cols: [], cells: [], counts: [] })
  assert.deepEqual(generationAvgCost([]), { overall: [], combos: [] })
})

test('model labels lose a shared prefix ending in a separator, unless a label would then be empty or start with a digit or a dot', () => {
  assert.deepEqual(modelLabels(['anthropic/claude-opus', 'anthropic/claude-sonnet']), ['opus', 'sonnet'])
  assert.deepEqual(modelLabels(['openai/gpt-4o', 'openai/gpt-5']), ['openai/gpt-4o', 'openai/gpt-5'])
  assert.deepEqual(modelLabels(['openai/o1', 'openai/o1-mini']), ['o1', 'o1-mini'])
  assert.deepEqual(modelLabels(['a/x', 'b/y']), ['a/x', 'b/y'])
  assert.deepEqual(modelLabels(['openai/gpt-5']), ['openai/gpt-5'])
})

test('each generation has its own mean and the running mean so far; an unpriced program makes the running mean unknown from then on', () => {
  const programs = [
    p('i', 0, 'a', 'init', 1),
    p('d1', 1, 'a', 'diff', 0.5),
    p('d1c', 1, 'a', 'diff', 0.5, copy),
    p('d2', 2, 'a', 'diff', Number.NaN),
    p('f1', 2, 'b', 'full', 0.25),
    p('d3', 3, 'a', 'diff', 0.5),
    p('f2', 3, 'b', 'full', 0.75),
    p('g1', 3, 'a', 'full', 0.25),
  ]
  const { overall, combos } = generationAvgCost(programs)
  assert.deepEqual(overall, [
    { generation: 0, n: 1, mean: 1, runningN: 1, runningMean: 1 },
    { generation: 1, n: 2, mean: 0.25, runningN: 3, runningMean: 0.5 },
    { generation: 2, n: 2, mean: null, runningN: 5, runningMean: null },
    { generation: 3, n: 3, mean: 0.5, runningN: 8, runningMean: null },
  ])
  // init is not a combo; the highest final mean comes first and an unknown final comes last.
  assert.deepEqual(combos, [
    {
      model: 'b',
      patchType: 'full',
      points: [
        { generation: 0, value: null, n: 0 },
        { generation: 1, value: null, n: 0 },
        { generation: 2, value: 0.25, n: 1 },
        { generation: 3, value: 0.5, n: 2 },
      ],
      final: 0.5,
    },
    {
      model: 'a',
      patchType: 'full',
      points: [
        { generation: 0, value: null, n: 0 },
        { generation: 1, value: null, n: 0 },
        { generation: 2, value: null, n: 0 },
        { generation: 3, value: 0.25, n: 1 },
      ],
      final: 0.25,
    },
    {
      model: 'a',
      patchType: 'diff',
      points: [
        { generation: 0, value: null, n: 0 },
        { generation: 1, value: 0.25, n: 2 },
        { generation: 2, value: null, n: 3 },
        { generation: 3, value: null, n: 4 },
      ],
      final: null,
    },
  ])
})

test('combos with the same final mean keep their key order, programs without a model or patch type form their own combo, and a program without a generation is generation 0', () => {
  const { overall, combos } = generationAvgCost([
    p('y', 1, 'y', 'diff', 0.5),
    p('x', 1, 'x', 'diff', 0.5),
    p('n', null, undefined, undefined, 0.5),
  ])
  assert.deepEqual(overall.map((g) => [g.generation, g.n]), [[0, 1], [1, 2]])
  assert.deepEqual(
    combos.map((c) => [c.model, c.patchType, c.final]),
    [
      [null, null, 0.5],
      ['x', 'diff', 0.5],
      ['y', 'diff', 0.5],
    ],
  )
})
