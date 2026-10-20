/**
 * One page and its state, exactly as the URL says (award-redesign design.md › 页面与路由). Pages a
 * restored analysis page still draws use these same public parameters; legacyFrame.ts converts
 * them for the frame. Plain functions with no imports, so node can test them directly.
 */
export type Route =
  | { view: 'directions' }
  | { view: 'direction'; direction: string; hypothesis: string | null }
  | { view: 'runs'; q: string; sort: string; select: string[] }
  | { view: 'compare'; runs: string[] }
  | {
      view: 'run'
      run: string
      /** lineage | progress | space | contest, the inspector pane and the generation: drawn from task 6.1, kept until then. */
      tab: string | null
      program: string | null
      pane: string | null
      gen: string | null
      /** The restored analysis page's own panels (left_tab, right_tab, meta_section) until task 6.1 replaces it. */
      panels: [string, string][]
      /** The research an experiment's analysis returns to; absent for an ordinary run. */
      direction: string | null
      hypothesis: string | null
    }
  | { view: 'library'; page: string | null }

export type RunRoute = Extract<Route, { view: 'run' }>

export const RUN_PANELS = ['left_tab', 'right_tab', 'meta_section']

export const ALL_RUNS: Route = { view: 'runs', q: '', sort: '', select: [] }

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
      return { view: 'runs', q: p.get('q') ?? '', sort: p.get('sort') ?? '', select: p.getAll('select') }
    case 'compare':
      return { view: 'compare', runs: p.getAll('runs') }
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
      put('sort', route.sort)
      for (const run of route.select) p.append('select', run)
      break
    case 'compare':
      for (const run of route.runs) p.append('runs', run)
      break
    case 'run':
      put('run', route.run)
      put('tab', route.tab)
      put('program', route.program)
      put('pane', route.pane)
      put('gen', route.gen)
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
    panels: [],
    direction: context?.direction ?? null,
    hypothesis: context?.hypothesis ?? null,
  }
}
