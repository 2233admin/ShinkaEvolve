import type { CopyKey } from './i18n/copy'
import { ALL_RUNS, RUN_PANELS, routeHref, runRoute, type Route } from './shell/route.ts'

/** The route still drawn by the restored analysis page inside the frame (tasks 6.x replace it). */
export type HostedRoute = Extract<Route, { view: 'run' }>

/** A failure the host explains in the reader's language; anything else shows its own message. */
export class HostError extends Error {
  readonly key: CopyKey
  constructor(key: CopyKey) {
    super(key)
    this.key = key
  }
}

const RUN_PAGE = '/viz_tree.html'

/** The restored run page's code tab: the public URL's `pane=code`. */
const CODE_TAB = 'agent-code'

/** The restored page that draws a hosted route. */
export function framePath(route: HostedRoute): string {
  const p = new URLSearchParams()
  p.set('db_path', route.run)
  if (route.program) p.set('selected_node', route.program)
  // The inspector's code pane (task 6.1) is the restored page's code tab; it wins over a right_tab.
  const code = route.pane === 'code'
  for (const [key, value] of route.panels) if (!(code && key === 'right_tab')) p.append(key, value)
  if (code) p.append('right_tab', CODE_TAB)
  return `${RUN_PAGE}?${p}`
}

/**
 * The public URL of where the restored page is now. While the frame shows the run it was opened
 * for, the research context and the native view state stay; on another run they do not apply.
 */
export function projectedHref(inside: URL, route: HostedRoute): string {
  if (inside.pathname !== RUN_PAGE) throw new HostError('an.leftWorkspace')
  const q = inside.searchParams
  const run = q.get('db_path')
  if (!run) return routeHref(ALL_RUNS)
  const kept = route.run === run ? route : runRoute(run)
  const panels = RUN_PANELS.flatMap((key) => q.getAll(key).map((value): [string, string] => [key, value]))
  const code = panels.some(([key, value]) => key === 'right_tab' && value === CODE_TAB)
  return routeHref({
    ...kept,
    program: q.get('selected_node') || null,
    // The frame's own tab decides: its code tab is pane=code, any other tab ends a code pane.
    pane: code ? 'code' : kept.pane === 'code' ? null : kept.pane,
    panels: code ? panels.filter(([key]) => key !== 'right_tab') : panels,
  })
}
