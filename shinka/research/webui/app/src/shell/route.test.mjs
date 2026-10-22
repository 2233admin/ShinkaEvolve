// The workbench URL contract: every page survives the URL, and a run's analysis survives the trip
// through the restored analysis page and back (node strips the TypeScript types).
import assert from 'node:assert/strict'
import test from 'node:test'
import { framePath, HostError, projectedHref } from '../legacyFrame.ts'
import { ALL_RUNS, analysisView, parseRoute, routeHref, runRoute, viewRoute, viewTab } from './route.ts'

// A direction folder name that must not be decoded twice or split.
const ODD = 'workspace/中文 space + % literal%2F # &/experiments/h001/e01/results/programs.sqlite'
const B = 'runs/task/run_b/programs.sqlite'
const at = (href) => parseRoute(new URL(href, 'http://shell').search)
const inside = (path) => new URL(path, 'http://shell')
const frame = (params) => inside(`/viz_tree.html?${new URLSearchParams(params)}`)
const failsWith = (key) => (error) => error instanceof HostError && error.key === key

const opened = {
  ...runRoute(ODD, { direction: '中文 space + % literal%2F # &', hypothesis: 'h001' }),
  tab: 'lineage',
  program: 'p-1',
  pane: 'diff',
  gen: '12',
  panel: 'conclusions',
  prompt: 'q-3',
  meta: '140',
}
// The same analysis on the restored page, which keeps its own panels in the URL.
const hosted = { ...opened, tab: 'legacy', panels: [['left_tab', 'embeddings'], ['meta_section', 'individual-programs']] }

test('every page and its state survive the URL', () => {
  for (const route of [
    { view: 'directions' },
    { view: 'direction', direction: '中文 space + % literal%2F # &', hypothesis: 'h002' },
    { view: 'runs', q: 'gap & drift', kind: 'experiment', sort: 'cost', select: [ODD, B] },
    { view: 'compare', runs: [ODD, B], metric: 'correct_count' },
    opened,
    hosted,
    { view: 'library', page: 'wiki/concept-lockbox' },
  ]) {
    assert.deepEqual(at(routeHref(route)), route)
  }
})

test('the default compare metric stays out of the URL', () => {
  const href = routeHref({ view: 'compare', runs: [B, ODD], metric: '' })
  assert.equal(new URL(href, 'http://shell').searchParams.has('metric'), false)
  assert.deepEqual(at(href), { view: 'compare', runs: [B, ODD], metric: '' })
})

test('a missing or unknown part opens a page that exists', () => {
  assert.deepEqual(at('/'), { view: 'directions' })
  assert.deepEqual(at('/?view=nowhere'), { view: 'directions' })
  assert.deepEqual(at('/?view=direction'), { view: 'directions' })
  assert.deepEqual(at('/?view=run'), ALL_RUNS)
})

test('a run opens Lineage and Progress natively, and solution space and contest on the restored page', () => {
  assert.equal(analysisView(runRoute(B)), 'lineage')
  assert.equal(analysisView({ ...runRoute(B), tab: 'lineage' }), 'lineage')
  assert.equal(analysisView({ ...runRoute(B), tab: 'progress' }), 'progress')
  assert.equal(analysisView({ ...runRoute(B), tab: 'nowhere' }), 'lineage')
  assert.equal(analysisView({ ...runRoute(B), tab: 'legacy' }), 'legacy')
  for (const tab of ['space', 'contest']) assert.equal(analysisView({ ...runRoute(B), tab }), 'legacy')
  // A link written before the native views names only the restored page's panels: it keeps opening there.
  assert.equal(analysisView({ ...runRoute(B), panels: [['left_tab', 'clusters']] }), 'legacy')
})

test('the current tab is the view the URL names; the restored page alone is the run bar’s link', () => {
  for (const tab of ['lineage', 'progress', 'space', 'contest', 'legacy']) assert.equal(viewTab({ ...runRoute(B), tab }), tab)
  assert.equal(viewTab(runRoute(B)), 'lineage')
  assert.equal(viewTab({ ...runRoute(B), tab: 'nowhere' }), 'lineage')
  assert.equal(viewTab({ ...runRoute(B), panels: [['left_tab', 'clusters-view']] }), 'legacy')
})

test('solution space and contest open the restored page on their own panel and stay that tab inside it', () => {
  const space = viewRoute(opened, 'space')
  const contest = viewRoute({ ...space, panels: [['left_tab', 'clusters-view']] }, 'contest')
  // Moving between them drops the other view's panel, so each opens on its own.
  assert.deepEqual(contest.panels, [])
  for (const [route, panel] of [[space, 'clusters-view'], [contest, 'table-view']]) {
    const run = inside(framePath(route))
    assert.equal(run.searchParams.get('selected_node'), 'p-1')
    assert.equal(run.searchParams.get('left_tab'), panel)
    // The first projection writes the panel into the public URL; after that the trip changes nothing.
    const once = projectedHref(run, route)
    assert.deepEqual(at(once), { ...route, panels: [['left_tab', panel]] })
    assert.equal(projectedHref(inside(framePath(at(once))), at(once)), once)
  }
  // A left panel the URL already names wins over the view's own.
  assert.equal(inside(framePath({ ...space, panels: [['left_tab', 'maps-view']] })).searchParams.get('left_tab'), 'maps-view')
  // Any panel the reader opens inside keeps the tab the page was opened as, with the program chosen there.
  assert.deepEqual(at(projectedHref(frame({ db_path: ODD, selected_node: 'p-2', left_tab: 'tree-view' }), contest)), {
    ...contest,
    program: 'p-2',
    panels: [['left_tab', 'tree-view']],
  })
  // The run bar's link keeps the panel and becomes the plain restored page; another run drops the view.
  assert.equal(viewRoute({ ...space, panels: [['left_tab', 'clusters-view']] }, 'legacy').tab, 'legacy')
  assert.equal(at(projectedHref(frame({ db_path: B }), space)).tab, 'legacy')
})

test('switching views keeps the selection, and a native view drops the restored page’s panels', () => {
  assert.deepEqual(viewRoute(hosted, 'lineage'), { ...hosted, tab: 'lineage', panels: [] })
  assert.deepEqual(viewRoute(opened, 'legacy'), { ...opened, tab: 'legacy' })
})

test('the frame gets the restored run page with the program and the pane’s tab', () => {
  const run = inside(framePath(hosted))
  assert.equal(run.pathname, '/viz_tree.html')
  assert.equal(run.searchParams.get('db_path'), ODD)
  assert.equal(run.searchParams.get('selected_node'), 'p-1')
  assert.equal(run.searchParams.get('left_tab'), 'embeddings')
  assert.deepEqual(run.searchParams.getAll('right_tab'), ['code-diff'])
  // The restored page has no native views or panel: those parameters wait in the public URL.
  for (const key of ['tab', 'gen', 'panel', 'prompt', 'meta']) assert.equal(run.searchParams.get(key), null)
})

test('a hosted run survives the trip through its restored page', () => {
  assert.equal(projectedHref(inside(framePath(hosted)), hosted), routeHref(hosted))
})

test("each inspector pane opens the restored page's tab for it and comes back as that pane", () => {
  const tabs = { code: 'agent-code', diff: 'code-diff', evaluation: 'log-output', llm: 'llm-result', meta: 'meta-analysis' }
  for (const [pane, tab] of Object.entries(tabs)) {
    const route = { ...hosted, pane, panels: [['left_tab', 'embeddings'], ['right_tab', 'throughput']] }
    const run = inside(framePath(route))
    // The pane's tab wins over a right tab the URL also carries.
    assert.deepEqual(run.searchParams.getAll('right_tab'), [tab])
    assert.equal(projectedHref(run, route), routeHref({ ...route, panels: [['left_tab', 'embeddings']] }))
  }
  // A pane name the inspector does not have opens no tab.
  assert.deepEqual(inside(framePath({ ...hosted, pane: 'constructor' })).searchParams.getAll('right_tab'), [])
  // A tab the inspector does not have ends the pane and stays the restored page's own.
  const details = projectedHref(frame({ db_path: ODD, right_tab: 'node-details' }), hosted)
  assert.equal(at(details).pane, null)
  assert.deepEqual(at(details).panels, [['right_tab', 'node-details']])
  // A frame that has not chosen a tab keeps the pane.
  assert.equal(at(projectedHref(frame({ db_path: ODD }), hosted)).pane, 'diff')
})

test('where the frame goes becomes the public URL', () => {
  // The same run: research context and native state stay, the frame's own state follows it.
  assert.deepEqual(at(projectedHref(frame({ db_path: ODD, selected_node: 'p-2', left_tab: 'clusters', stray: 'x' }), opened)), {
    ...opened,
    tab: 'legacy',
    program: 'p-2',
    panels: [['left_tab', 'clusters']],
  })
  // Another run the restored page links to: the research context and native state do not belong to it.
  assert.deepEqual(at(projectedHref(frame({ db_path: B }), opened)), { ...runRoute(B), tab: 'legacy' })
  // The run list and the compare page are native now: a frame that lands on the old ones has left what the host draws.
  assert.throws(() => projectedHref(inside(`/analysis/index.html?${new URLSearchParams({ q: 'gap' })}`), opened), failsWith('an.leftWorkspace'))
  assert.throws(() => projectedHref(inside(`/compare.html?${new URLSearchParams({ selection: JSON.stringify([{ path: B, label: 'b' }]) })}`), opened), failsWith('an.leftWorkspace'))
  assert.throws(() => projectedHref(inside('/elsewhere.html'), opened), failsWith('an.leftWorkspace'))
})
