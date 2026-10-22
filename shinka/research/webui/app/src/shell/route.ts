/**
 * One page and its state, exactly as the URL says (award-redesign design.md › 页面与路由). Pages a
 * restored analysis page still draws use these same public parameters; legacyFrame.ts converts
 * them for the frame. Plain functions with no imports, so node can test them directly.
 */
export type Route =
  | { view: 'directions' }
  | { view: 'direction'; direction: string; hypothesis: string | null }
  /** `kind` is ordinary, experiment or '' for every run; `sort` '' is the list's default order. */
  | { view: 'runs'; q: string; kind: string; sort: string; select: string[] }
  /** `metric` is program_count or correct_count; '' is the cumulative best. */
  | { view: 'compare'; runs: string[]; metric: string }
  | {
      view: 'run'
      run: string
      /** The analysis view: lineage, progress, space or contest, or `legacy` for the restored page (see `analysisView` and `viewTab`). */
      tab: string | null
      program: string | null
      /** The inspector pane: code, diff, evaluation, llm or meta; absent is code. */
      pane: string | null
      /** The last generation the views show (the generation scrubber); absent is the newest. */
      gen: string | null
      /** The lineage panel's open section: conclusions, prompts, migrations, path or offspring; absent is closed. */
      panel: string | null
      /** The prompt version open in the lineage panel (L02-S1). */
      prompt: string | null
      /** The meta-analysis update the conclusions show, by its processed count (R03-C1); absent is the latest. */
      meta: string | null
      /** The restored analysis page's own panels (left_tab, right_tab, meta_section). */
      panels: [string, string][]
      /** The research an experiment's analysis returns to; absent for an ordinary run. */
      direction: string | null
      hypothesis: string | null
    }
  | { view: 'library'; page: string | null }

export type RunRoute = Extract<Route, { view: 'run' }>

export const RUN_PANELS = ['left_tab', 'right_tab', 'meta_section']

/**
 * The analysis views in their tab order. Lineage and Progress are drawn natively; solution space
 * and contest stay on the restored analysis page (owner decision A, design.md › 迁移与删除).
 */
export const ANALYSIS_VIEWS = ['lineage', 'progress', 'space', 'contest'] as const
export const NATIVE_VIEWS = ['lineage', 'progress'] as const
export const HOSTED_VIEWS = ['space', 'contest'] as const
export type AnalysisView = (typeof NATIVE_VIEWS)[number] | 'legacy'
export type ViewTab = (typeof ANALYSIS_VIEWS)[number] | 'legacy'

/** The hosted view a tab names (the view a restored page was opened as), or null. */
export const hostedView = (tab: string | null) => HOSTED_VIEWS.find((view) => view === tab) ?? null

/**
 * Which page a run's URL opens. A native view draws natively; `legacy`, a hosted view and a URL that
 * only names the restored page's own panels open the restored page; anything else is Lineage.
 */
export function analysisView(route: RunRoute): AnalysisView {
  const { tab } = route
  if (tab === 'legacy' || hostedView(tab)) return 'legacy'
  const native = NATIVE_VIEWS.find((view) => view === tab)
  if (native) return native
  return tab === null && route.panels.length > 0 ? 'legacy' : 'lineage'
}

/** The tab a run's URL marks as current: a hosted view keeps its own tab on the restored page. */
export function viewTab(route: RunRoute): ViewTab {
  const view = analysisView(route)
  return view === 'legacy' ? (hostedView(route.tab) ?? 'legacy') : view
}

/** The same run and selection in another view: every view but `legacy` drops the restored page's own panels. */
export const viewRoute = (route: RunRoute, tab: string): RunRoute => ({ ...route, tab, panels: tab === 'legacy' ? route.panels : [] })

export const ALL_RUNS: Extract<Route, { view: 'runs' }> = { view: 'runs', q: '', kind: '', sort: '', select: [] }

export function parseRoute(search: string): Route {
  const p = new URLSearchParams(search)
  const one = (key: string) => p.get(key) || null
  switch (p.get('view')) {
    case 'direction': {
      const direction = one('direction')
      if (direction) return { view: 'direction', direction, hypothesis: one('hypothesis') }
      break
    }
    case 'runs':
      return { view: 'runs', q: p.get('q') ?? '', kind: p.get('kind') ?? '', sort: p.get('sort') ?? '', select: p.getAll('select') }
    case 'compare':
      return { view: 'compare', runs: p.getAll('runs'), metric: p.get('metric') ?? '' }
    case 'run': {
      const run = one('run')
      if (!run) return ALL_RUNS
      return {
        view: 'run',
        run,
        tab: one('tab'),
        program: one('program'),
        pane: one('pane'),
        gen: one('gen'),
        panel: one('panel'),
        prompt: one('prompt'),
        meta: one('meta'),
        panels: RUN_PANELS.flatMap((key) => p.getAll(key).map((value): [string, string] => [key, value])),
        direction: one('direction'),
        hypothesis: one('hypothesis'),
      }
    }
    case 'library':
      return { view: 'library', page: one('page') }
  }
  return { view: 'directions' }
}

export function routeHref(route: Route): string {
  const p = new URLSearchParams({ view: route.view })
  const put = (key: string, value: string | null) => {
    if (value) p.append(key, value)
  }
  switch (route.view) {
    case 'direction':
      put('direction', route.direction)
      put('hypothesis', route.hypothesis)
      break
    case 'runs':
      put('q', route.q)
      put('kind', route.kind)
      put('sort', route.sort)
      for (const run of route.select) p.append('select', run)
      break
    case 'compare':
      for (const run of route.runs) p.append('runs', run)
      put('metric', route.metric)
      break
    case 'run':
      put('run', route.run)
      put('tab', route.tab)
      put('program', route.program)
      put('pane', route.pane)
      put('gen', route.gen)
      put('panel', route.panel)
      put('prompt', route.prompt)
      put('meta', route.meta)
      put('direction', route.direction)
      put('hypothesis', route.hypothesis)
      for (const [key, value] of route.panels) p.append(key, value)
      break
    case 'library':
      put('page', route.page)
      break
  }
  return `/?${p}`
}

/** A run's analysis with nothing chosen yet: the run list, palette and research links open this. */
export function runRoute(run: string, context: { direction: string; hypothesis: string | null } | null = null): RunRoute {
  return {
    view: 'run',
    run,
    tab: null,
    program: null,
    pane: null,
    gen: null,
    panel: null,
    prompt: null,
    meta: null,
    panels: [],
    direction: context?.direction ?? null,
    hypothesis: context?.hypothesis ?? null,
  }
}
