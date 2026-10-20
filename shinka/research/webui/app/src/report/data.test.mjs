// The direction report's derived data: what the reader sees depends on these rules, not on layout
// (node strips the TypeScript types).
import assert from 'node:assert/strict'
import test from 'node:test'
import { parseRoute } from '../shell/route.ts'
import { bodySection, experimentProgram, frontier, marks, programHref, sortHypotheses } from './data.ts'

const point = (n, hypothesis, experiment, score, bar = 0.5, program = `p${n}`) =>
  ({ n, time: '', hypothesis, experiment, generation: 0, program, score, record: false, bar })

test('a body section is read whether or not it is quoted, and stops at the next heading', () => {
  const body = '# h001\n\n> ## Claim\n> Gaps continue\n> into the close.\n\n## Mechanism\nSlow digestion.\n'
  assert.equal(bodySection(body, 'Claim'), 'Gaps continue into the close.')
  assert.equal(bodySection(body, 'mechanism'), 'Slow digestion.')
  assert.equal(bodySection(body, 'Falsification'), '')
})

test("an experiment opens the program its check judged, else its best scored trial of its own", () => {
  const points = [point(1, 'h001', 'e01', 0.4), point(2, 'h001', 'e01', null), point(3, 'h002', 'e01', 0.9), point(4, 'h001', 'e02', 0.8), point(5, 'h001', 'e01', 0.6)]
  const h = (verdict) => ({ id: 'h001', verdict })
  assert.equal(experimentProgram(h(null), { id: 'e01' }, points), 'p5')
  assert.equal(experimentProgram(h(null), { id: 'e01' }, [point(1, 'h001', 'e01', null)]), null)
  // The skeptic's program wins over a better trial, and the verdict's program counts only for the experiment it names.
  assert.equal(experimentProgram(h(null), { id: 'e01', skeptic: { program_id: 'p1' } }, points), 'p1')
  const verdict = { evidence: { experiment: 'e02', program_id: 'judged' } }
  assert.equal(experimentProgram(h(verdict), { id: 'e02', skeptic: null }, points), 'judged')
  assert.equal(experimentProgram(h(verdict), { id: 'e01', skeptic: null }, points), 'p5')
})

test('a program link opens that program on its code and keeps the research to return to', () => {
  const analysis = '/?view=run&run=workspace%2Fd%2B1%2Fexperiments%2Fh001%2Fe01%2Fresults%2Fprograms.sqlite&direction=d%2B1&hypothesis=h001&right_tab=agent-info'
  const route = parseRoute(new URL(programHref(analysis, 'p 7'), 'http://shell').search)
  assert.equal(route.view, 'run')
  assert.equal(route.run, 'workspace/d+1/experiments/h001/e01/results/programs.sqlite')
  assert.equal(route.program, 'p 7')
  assert.deepEqual([route.direction, route.hypothesis], ['d+1', 'h001'])
  assert.equal(route.pane, 'code')
  assert.deepEqual(route.panels, [])
})

test('the record frontier starts at the first score and failures never raise it', () => {
  const steps = frontier([point(1, 'h', 'e', null, 0.1), point(2, 'h', 'e', 0.3, 0.2), point(3, 'h', 'e', null, 0.25), point(4, 'h', 'e', 0.2, 0.3), point(5, 'h', 'e', 0.7, 0.35)])
  assert.deepEqual(steps.map((s) => [s.n, s.best, s.bar]), [[2, 0.3, 0.2], [3, 0.3, 0.25], [4, 0.3, 0.3], [5, 0.7, 0.35]])
})

test('verdicts come from the hypotheses and checks and confirmations from the events', () => {
  const hyps = [{ id: 'h001', verdict: { created: '2026-01-02T00:00:00+00:00' } }, { id: 'h002', verdict: null }]
  const events = [
    { time: '2026-01-01T00:00:00+00:00', hypothesis: 'h001', event: 'skeptic' },
    { time: '2026-01-02T00:00:00+00:00', hypothesis: 'h001', event: 'verdict' },
    { time: '2026-01-03T00:00:00+00:00', hypothesis: 'h001', event: 'confirm' },
    { time: '2026-01-01T00:00:00+00:00', hypothesis: 'h002', event: 'run' },
  ]
  assert.deepEqual(marks(hyps, events).map((m) => `${m.hypothesis}:${m.kind}`).sort(), ['h001:confirm', 'h001:skeptic', 'h001:verdict'])
})

test('sorting keeps hypotheses without a value last in both directions', () => {
  const h = (id, oos) => ({ id, title: id, status: 'running', experiments: [], verdict: oos === undefined ? null : { created: '', evidence: { oos_sharpe: oos } } })
  const hyps = [h('a'), h('b', 0.2), h('c', -0.1)]
  assert.deepEqual(sortHypotheses(hyps, { key: 'oos', descending: false }).map((x) => x.id), ['c', 'b', 'a'])
  assert.deepEqual(sortHypotheses(hyps, { key: 'oos', descending: true }).map((x) => x.id), ['b', 'c', 'a'])
})
