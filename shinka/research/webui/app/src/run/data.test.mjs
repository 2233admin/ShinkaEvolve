// The run analysis page's derived data (award-redesign 6.1, 6.2); node strips the TypeScript types.
import assert from 'node:assert/strict'
import test from 'node:test'
import {
  bestOf,
  bestPrompts,
  bestSoFar,
  codeFileName,
  diffRows,
  failureReason,
  fertility,
  fitLabels,
  initialOf,
  lineOf,
  metaSection,
  metaUpdates,
  migrations,
  mutationPoints,
  population,
  privateMetrics,
  programSummaries,
  promptPrograms,
  promptTally,
  rampDomain,
  splitHighlighted,
} from './data.ts'

const p = (id, generation, score, extra = {}) => ({
  id,
  generation,
  timestamp: generation * 10 + (extra.at ?? 0),
  correct: score !== null,
  // A failed evaluation can still carry a raw score; it must never count as one.
  combined_score: score ?? 9,
  parent_id: extra.parent ?? null,
  island_idx: extra.island ?? 0,
  metadata: { patch_name: extra.name ?? `patch_${id}`, ...extra.metadata },
  migration_history: extra.migrations ?? [],
})

test('a failed program’s reason is the runner’s failure reason, else its feedback, else the last line it wrote to stderr; a valid one has none', () => {
  const failed = (extra) => ({ id: 'f', correct: false, combined_score: 0, ...extra })
  assert.equal(failureReason(failed({ text_feedback: 'feedback', metadata: { failure_reason: ' Patch generation failed before evaluation ', stderr_log: 'other' } })), 'Patch generation failed before evaluation')
  assert.equal(failureReason(failed({ text_feedback: ' look-ahead detected ', metadata: { stderr_log: 'Traceback\nValueError: x\n' } })), 'look-ahead detected')
  assert.equal(failureReason(failed({ metadata: { stderr_log: 'Traceback (most recent call last)\n  File "a.py"\nValueError: bad shape\n\n' } })), 'ValueError: bad shape')
  assert.equal(failureReason(failed({})), null)
  assert.equal(failureReason({ id: 'v', correct: true, combined_score: 0.5, text_feedback: 'fine' }), null)
})

test('a track label never covers a mark or another label, tied best ones included, and flips left at the edge', () => {
  // Two tied best prompts 30px apart: the first has no room that leaves the second's mark clear, so it goes unlabelled.
  const items = [
    { x: 100, w: 40, keep: true, off: false },
    { x: 130, w: 40, keep: true, off: false },
    { x: 200, w: 40, keep: false, off: true },
    { x: 290, w: 40, keep: false, off: false },
  ]
  const sides = fitLabels(items, 90, 300)
  assert.deepEqual(sides, [null, 'right', null, 'left'])
  const spans = sides.flatMap((s, i) => (s === null ? [] : [s === 'right' ? [items[i].x + 8, items[i].x + 8 + items[i].w] : [items[i].x - 8 - items[i].w, items[i].x - 8]]))
  const marks = items.filter((it) => !it.off).map((it) => [it.x - 6, it.x + 6])
  for (const [a, b] of spans) {
    assert.ok(a >= 90 && b <= 300)
    assert.ok(!marks.some(([c, d]) => a < d && b > c))
  }
  for (const [i, [a, b]] of spans.entries()) for (const [c, d] of spans.slice(i + 1)) assert.ok(b <= c || d <= a)
})

test('private metrics are only ever reported as hidden, even when a response carries their values', () => {
  assert.equal(privateMetrics({ id: 'a', private_metrics: {}, private_metrics_hidden: true }), 'hidden')
  assert.equal(privateMetrics({ id: 'b', private_metrics: { held_out: 1.5 } }), 'hidden')
  assert.equal(privateMetrics({ id: 'c', private_metrics: {} }), 'none')
  assert.equal(privateMetrics({ id: 'd' }), 'none')
})

const ids = (programs) => programs.map((x) => x.id)

test('every program tied at the best valid score is best, and a failed one never is', () => {
  const programs = [p('a', 0, 0.5), p('b', 1, 0.9), p('c', 2, null), p('d', 2, 0.9, { metadata: { _is_island_copy: true } })]
  assert.deepEqual(ids(bestOf(programs)), ['b', 'd'])
  assert.deepEqual(bestOf([p('x', 0, null)]), [])
})

test('an ancestry runs from the root to the program and stops at a missing parent or a loop', () => {
  const programs = [p('r', 0, 0.1), p('a', 1, 0.2, { parent: 'r' }), p('b', 2, 0.3, { parent: 'a' }), p('o', 3, 0.1, { parent: 'gone' }), p('l1', 1, 0.1, { parent: 'l2' }), p('l2', 1, 0.1, { parent: 'l1' })]
  const byId = new Map(programs.map((x) => [x.id, x]))
  assert.deepEqual(ids(lineOf(byId, 'b')), ['r', 'a', 'b'])
  assert.deepEqual(ids(lineOf(byId, 'o')), ['o'])
  assert.deepEqual(ids(lineOf(byId, 'l1')), ['l2', 'l1'])
  // The diff against the initial program compares with the lineage's own root, else the run's first.
  assert.equal(initialOf(lineOf(byId, 'b'), programs).id, 'r')
  assert.equal(initialOf(lineOf(byId, 'o'), programs).id, 'r')
})

test('the score ramp runs from the 5th percentile of valid scores to the best', () => {
  const programs = Array.from({ length: 21 }, (_, i) => p(`p${i}`, i, i / 20)).concat([p('f', 0, null)])
  assert.deepEqual(rampDomain(programs).map((v) => Math.round(v * 1000) / 1000), [0.05, 1])
  assert.equal(rampDomain([p('f', 0, null)]), null)
})

test('an island’s best so far steps only where it rises', () => {
  const programs = [p('a', 0, 0.3), p('b', 1, 0.2), p('c', 2, 0.5), p('d', 2, 0.6, { at: 1 }), p('e', 3, null), p('f', 4, 0.6)]
  assert.deepEqual(bestSoFar(programs), [{ generation: 0, best: 0.3 }, { generation: 2, best: 0.6 }])
})

test('a meta update sits at the generation of the program that completed its count', () => {
  const programs = [p('a', 0, 0.1), p('b', 0, 0.2, { at: 1 }), p('c', 1, 0.3), p('d', 3, null)]
  const files = [{ processed_count: 9, filename: 'meta_9.txt' }, { processed_count: 2, filename: 'meta_2.txt' }]
  assert.deepEqual(metaUpdates(files, programs), [
    { processed: 2, file: 'meta_2.txt', generation: 0 },
    { processed: 9, file: 'meta_9.txt', generation: 3 },
  ])
})

const RUNNER = `# INDIVIDUAL PROGRAM SUMMARIES

The following are summaries of individual programs evaluated since the last meta update:

**Program Name: Momentum with brake**
- **Implementation**: Adds a drawdown brake.
**Program Identifier:** Generation 3 - Patch Name add_brake - with - dashes - Correct Program: True

**Program Name: Momentum with brake again**
- **Implementation**: Same patch name in the same generation.
**Program Identifier:** Generation 3 - Patch Name add_brake - with - dashes - Correct Program: False

**Program Name: Other**
**Program Identifier:** Generation 4 - Patch Name add_brake - with - dashes - Correct Program: True

# GLOBAL INSIGHTS SCRATCHPAD

Insight one.

# META RECOMMENDATIONS

1. Do this.
`

test('a program’s summaries are read from the runner’s identifier lines, every match kept', () => {
  const found = programSummaries(RUNNER, p('x', 3, 0.4, { name: 'add_brake - with - dashes' }))
  assert.equal(found.length, 2)
  assert.ok(found[0].startsWith('**Program Name: Momentum with brake**'))
  assert.ok(!found[0].includes('The following are summaries'))
  assert.ok(found[1].endsWith('Correct Program: False'))
  assert.deepEqual(programSummaries(RUNNER, p('y', 5, 0.4, { name: 'add_brake - with - dashes' })), [])
  assert.equal(metaSection(RUNNER, 'GLOBAL INSIGHTS SCRATCHPAD'), 'Insight one.')
  assert.equal(metaSection(RUNNER, 'META RECOMMENDATIONS'), '1. Do this.')
  assert.equal(metaSection(RUNNER, 'NOT THERE'), '')
})

test('a demo summary is found by the short id in its name line', () => {
  const demo = '# INDIVIDUAL PROGRAM SUMMARIES\n\n**Program Name: add_x (ec4f4e0f, gen 32, island 3)**\n- **Performance**: 1.03\n\n**Program Name: add_y (12345678, gen 2, island 0)**\n- **Performance**: 0.5\n'
  assert.deepEqual(programSummaries(demo, p('ec4f4e0f-c465', 32, 1.03)), ['**Program Name: add_x (ec4f4e0f, gen 32, island 3)**\n- **Performance**: 1.03'])
})

test('the best prompt is the fittest that produced a program, ties kept', () => {
  const prompts = [
    { id: 'p0', fitness: 0.6, program_count: 10 },
    { id: 'p1', fitness: 0.9, program_count: 0 },
    { id: 'p2', fitness: 0.7, program_count: 3 },
    { id: 'p3', fitness: 0.7, program_count: 1 },
  ]
  assert.deepEqual([...bestPrompts(prompts)], ['p2', 'p3'])
  assert.deepEqual([...bestPrompts([])], [])
})

test('a prompt’s programs carry Δ against a valid parent and Beat% among the valid programs', () => {
  const programs = [p('r', 0, 0.1), p('a', 1, 0.4, { parent: 'r' }), p('b', 1, 0.2, { parent: 'r' }), p('c', 2, null, { parent: 'a' }), p('d', 2, 0.3, { parent: 'c' })]
  const rows = promptPrograms({ id: 'q', program_ids: ['a', 'c', 'd'] }, programs)
  assert.deepEqual(rows.map((r) => [r.program.id, r.delta === null ? null : Math.round(r.delta * 100) / 100, r.beat === null ? null : Math.round(r.beat)]), [
    ['a', 0.3, 100],
    ['c', null, null],
    ['d', null, 67],
  ])
  const tally = promptTally(rows)
  assert.deepEqual([tally.correct, tally.improved, Math.round(tally.beat)], [2, 1, 83])
})

test('migrations list every move by generation', () => {
  const programs = [p('a', 5, 0.1, { migrations: [{ generation: 20, from: 1, to: 2 }] }), p('b', 3, 0.2, { migrations: [{ generation: 10, from: 0, to: 3 }, { generation: 30, from: 3, to: 1 }] })]
  assert.deepEqual(migrations(programs).map((m) => [m.program.id, m.generation, m.from, m.to]), [['b', 10, 0, 3], ['a', 20, 1, 2], ['b', 30, 3, 1]])
})

test('an island’s population accumulates by generation and counts failures', () => {
  const programs = [p('a', 0, 0.1, { island: 0 }), p('b', 2, null, { island: 0 }), p('c', 1, 0.2, { island: 1 }), p('d', 2, 0.3, { island: 1 })]
  assert.deepEqual(population(programs), [{ island: 0, counts: [1, 1, 2] }, { island: 1, counts: [0, 1, 2] }])
})

test('the most fertile parents count only valid children that beat them; valid leaves are dead ends', () => {
  const programs = [p('r', 0, 0.5), p('a', 1, 0.6, { parent: 'r' }), p('b', 1, 0.4, { parent: 'r' }), p('c', 1, null, { parent: 'r' }), p('d', 2, 0.7, { parent: 'a' })]
  const { top, deadEnds } = fertility(programs)
  assert.deepEqual(top.map((x) => [x.program.id, x.children, x.improving]), [['r', 3, 1], ['a', 1, 1]])
  assert.equal(deadEnds, 2)
})

test('mutation size sums the diff summary and a failure on either side has no change in score', () => {
  const programs = [
    p('r', 0, 0.5),
    p('a', 1, 0.6, { parent: 'r', metadata: { diff_summary: { added: 2, deleted: 1, modified: 3 } } }),
    p('b', 1, null, { parent: 'r', metadata: { diff_summary: { added: 1 } } }),
    p('c', 2, 0.7, { parent: 'b', metadata: { diff_summary: { modified: 4 } } }),
    p('d', 2, 0.7, { parent: 'a' }),
  ]
  assert.deepEqual(mutationPoints(programs).map((m) => [m.program.id, m.size, m.delta === null ? null : Math.round(m.delta * 10) / 10, m.failed]), [
    ['a', 6, 0.1, null],
    ['b', 1, null, 'self'],
    ['c', 4, null, 'parent'],
  ])
})

test('highlighted lines reopen a span that crosses a line break', () => {
  assert.deepEqual(splitHighlighted('<span class="s">"""a\nb"""</span> x\ny'), ['<span class="s">"""a</span>', '<span class="s">b"""</span> x', 'y'])
})

test('a diff line that starts with --- inside a hunk is a removed line, not a header', () => {
  const rows = diffRows('--- a/main.py\n+++ b/main.py\n@@ -1,2 +1,2 @@\n x\n---- y\n+++z\n')
  assert.deepEqual(rows.map((r) => r.kind), ['head', 'head', 'hunk', 'same', 'del', 'add'])
})

test('the code download is named after the patch and generation with the language’s extension', () => {
  assert.equal(codeFileName({ ...p('a', 7, 0.1, { name: 'add brake/v2' }), language: 'python' }), 'add_brake_v2_gen7.py')
  assert.equal(codeFileName({ ...p('a', 0, 0.1), language: 'cuda' }), 'patch_a_gen0.cu')
  assert.equal(codeFileName({ id: 'abcdef123', generation: 2, metadata: {}, language: 'constructor' }), 'abcdef12_gen2.constructor')
})
