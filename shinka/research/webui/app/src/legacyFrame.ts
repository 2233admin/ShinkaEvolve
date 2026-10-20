import type { CopyKey } from './i18n/copy'
import { ALL_RUNS, RUN_PANELS, routeHref, runRoute, type Route } from './shell/route.ts'

/** Routes still drawn by a restored analysis page inside the frame (tasks 5.1, 5.2 and 6.x replace them). */
export type HostedRoute = Extract<Route, { view: 'runs' | 'compare' | 'run' }>

/** A failure the host explains in the reader's language; anything else shows its own message. */
export class HostError extends Error {
  readonly key: CopyKey
  constructor(key: CopyKey) {
    super(key)
    this.key = key
  }
}

const PAGES: Record<string, HostedRoute['view']> = {
  '/analysis/index.html': 'runs',
  '/compare.html': 'compare',
  '/viz_tree.html': 'run',
}

type Selection = { path: string; label: string }[]

function isSelection(value: unknown): value is Selection {
  return Array.isArray(value) && value.every((item) =>
    typeof item === 'object' && item !== null && typeof item.path === 'string' && typeof item.label === 'string')
}

/** The compare page's selection: at least two runs, each with its path and label. */
function selection(text: string | null): Selection {
  const value: unknown = JSON.parse(text ?? 'null')
  if (!isSelection(value) || value.length < 2) throw new HostError('an.compareSelection')
  return value
}

/** The restored page that draws a hosted route; selections carry the run list's names as labels. */
export function framePath(route: HostedRoute, names: ReadonlyMap<string, string>): string {
  const labelled = (paths: string[]) => JSON.stringify(paths.map((path) => ({ path, label: names.get(path) ?? path })))
  const p = new URLSearchParams()
  switch (route.view) {
    case 'runs':
      if (route.q) p.set('q', route.q)
      if (route.sort) p.set('sort', route.sort)
      if (route.select.length) p.set('selected', labelled(route.select))
      return `/analysis/index.html?${p}`
    case 'compare':
      if (route.runs.length < 2) throw new HostError('an.compareSelection')
      p.set('selection', labelled(route.runs))
      return `/compare.html?${p}`
    case 'run':
      p.set('db_path', route.run)
      if (route.program) p.set('selected_node', route.program)
      for (const [key, value] of route.panels) p.append(key, value)
      return `/viz_tree.html?${p}`
  }
}

/**
 * The public URL of where the restored page is now. While the frame shows the run it was opened
 * for, the research context and the native view state stay; on another run they do not apply.
 */
export function projectedHref(inside: URL, route: HostedRoute): string {
  const q = inside.searchParams
  switch (PAGES[inside.pathname]) {
    case 'runs': {
      let select: string[] = []
      try {
        const value: unknown = JSON.parse(q.get('selected') ?? '[]')
        if (isSelection(value)) select = value.map((item) => item.path)
      } catch {
        /* The run list ignores an unreadable selection too. */
      }
      return routeHref({ view: 'runs', q: q.get('q') ?? '', sort: q.get('sort') ?? '', select })
    }
    case 'compare':
      return routeHref({ view: 'compare', runs: selection(q.get('selection')).map((item) => item.path) })
    case 'run': {
      const run = q.get('db_path')
      if (!run) return routeHref(ALL_RUNS)
      const kept = route.view === 'run' && route.run === run ? route : runRoute(run)
      return routeHref({
        ...kept,
        program: q.get('selected_node') || null,
        panels: RUN_PANELS.flatMap((key) => q.getAll(key).map((value): [string, string] => [key, value])),
      })
    }
    default:
      throw new HostError('an.leftWorkspace')
  }
}
