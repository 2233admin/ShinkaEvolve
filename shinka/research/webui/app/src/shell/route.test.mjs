// The workbench URL contract: every page survives the URL, and pages a restored analysis page still
// draws survive the trip through that page and back (node strips the TypeScript types).
import assert from 'node:assert/strict'
import test from 'node:test'
import { framePath, HostError, projectedHref } from '../legacyFrame.ts'
import { ALL_RUNS, parseRoute, routeHref, runRoute } from './route.ts'

// A direction folder name that must not be decoded twice or split.
const ODD = 'workspace/中文 space + % literal%2F # &/experiments/h001/e01/results/programs.sqlite'
const B = 'runs/task/run_b/programs.sqlite'
const at = (href) => parseRoute(new URL(href, 'http://shell').search)
const inside = (path) => new URL(path, 'http://shell')
const failsWith = (key) => (error) => error instanceof HostError && error.key === key

const opened = {
  ...runRoute(ODD, { direction: '中文 space + % literal%2F # &', hypothesis: 'h001' }),
  tab: 'progress',
  program: 'p-1',
  pane: 'diff',
  gen: '12',
  panels: [['left_tab', 'embeddings'], ['right_tab', 'code'], ['meta_section', 'individual-programs']],
}

test('every page and its state survive the URL', () => {
  for (const route of [
    { view: 'directions' },
    { view: 'direction', direction: '中文 space + % literal%2F # &', hypothesis: 'h002' },
    { view: 'runs', q: 'gap & drift', kind: 'experiment', sort: 'cost', select: [ODD, B] },
    { view: 'compare', runs: [ODD, B] },
    opened,
    { view: 'library', page: 'wiki/concept-lockbox' },
  ]) {
    assert.deepEqual(at(routeHref(route)), route)
  }
})

test('a missing or unknown part opens a page that exists', () => {
  assert.deepEqual(at('/'), { view: 'directions' })
  assert.deepEqual(at('/?view=nowhere'), { view: 'directions' })
  assert.deepEqual(at('/?view=direction'), { view: 'directions' })
  assert.deepEqual(at('/?view=run'), ALL_RUNS)
})

test('the frame gets the restored page, with the run list names as labels', () => {
  const names = new Map([[ODD, 'odd/e01'], [B, 'task/run_b']])
  const compare = inside(framePath({ view: 'compare', runs: [ODD, B] }, names))
  assert.equal(compare.pathname, '/compare.html')
  assert.deepEqual(JSON.parse(compare.searchParams.get('selection')), [
    { path: ODD, label: 'odd/e01' },
    { path: B, label: 'task/run_b' },
  ])
  assert.throws(() => framePath({ view: 'compare', runs: [ODD] }, names), failsWith('an.compareSelection'))
  const run = inside(framePath(opened, names))
  assert.equal(run.pathname, '/viz_tree.html')
  assert.equal(run.searchParams.get('db_path'), ODD)
  assert.equal(run.searchParams.get('selected_node'), 'p-1')
  assert.equal(run.searchParams.get('left_tab'), 'embeddings')
  // The restored page has no native views: those parameters wait in the public URL for task 6.1.
  assert.equal(run.searchParams.get('tab'), null)
})

test('each hosted page survives the trip through its restored page', () => {
  for (const route of [{ view: 'compare', runs: [ODD, B] }, opened]) {
    assert.equal(projectedHref(inside(framePath(route, new Map())), route), routeHref(route))
  }
})

test("the code pane opens the restored page's code tab and comes back as the code pane", () => {
  const code = { ...opened, pane: 'code', panels: [['left_tab', 'embeddings'], ['right_tab', 'meta']] }
  const run = inside(framePath(code, new Map()))
  // The code tab wins over a right tab the URL also carries.
  assert.deepEqual(run.searchParams.getAll('right_tab'), ['agent-code'])
  assert.equal(projectedHref(run, code), routeHref({ ...code, panels: [['left_tab', 'embeddings']] }))
  // Choosing another tab in the frame ends the code pane; any other pane belongs to the native views.
  const other = inside(`/viz_tree.html?${new URLSearchParams({ db_path: ODD, right_tab: 'node-details' })}`)
  assert.equal(at(projectedHref(other, code)).pane, null)
  assert.equal(at(projectedHref(other, opened)).pane, 'diff')
})

test('where the frame goes becomes the public URL', () => {
  const frame = (path, params) => inside(`${path}?${new URLSearchParams(params)}`)
  // The same run: research context and native state stay, the frame's own state follows it.
  assert.deepEqual(at(projectedHref(frame('/viz_tree.html', { db_path: ODD, selected_node: 'p-2', left_tab: 'clusters', stray: 'x' }), opened)), {
    ...opened,
    program: 'p-2',
    panels: [['left_tab', 'clusters']],
  })
  // Another run, say from the compare page's rows: the research context does not belong to it.
  assert.deepEqual(at(projectedHref(frame('/viz_tree.html', { db_path: B }), opened)), runRoute(B))
  // The run list is native now: a frame that lands on the old one has left what the host draws.
  assert.throws(() => projectedHref(frame('/analysis/index.html', { q: 'gap' }), opened), failsWith('an.leftWorkspace'))
  assert.throws(() => projectedHref(frame('/compare.html', { selection: JSON.stringify([{ path: B, label: 'b' }]) }), opened), failsWith('an.compareSelection'))
  assert.throws(() => projectedHref(inside('/elsewhere.html'), opened), failsWith('an.leftWorkspace'))
})
