import type { CopyKey } from './i18n/copy'
import { ALL_RUNS, RUN_PANELS, routeHref, runRoute, type Route } from './shell/route.ts'

/** The route the restored analysis page draws inside the frame (`tab=legacy`, until task 7.1 removes it). */
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

/** The inspector's panes (task 6.1) and the restored page's right tabs that show the same thing. */
const PANE_TABS = new Map([
  ['code', 'agent-code'],
  ['diff', 'code-diff'],
  ['evaluation', 'log-output'],
  ['llm', 'llm-result'],
  ['meta', 'meta-analysis'],
])

const paneOf = (tab: string): string | null => [...PANE_TABS].find(([, value]) => value === tab)?.[0] ?? null

/** The restored page that draws a hosted route: the program stays selected and the pane opens its tab there. */
export function framePath(route: HostedRoute): string {
  const p = new URLSearchParams()
  p.set('db_path', route.run)
  if (route.program) p.set('selected_node', route.program)
  // The inspector's pane is a tab of the restored page; it wins over a right_tab the URL also carries.
  const tab = route.pane === null ? null : (PANE_TABS.get(route.pane) ?? null)
  for (const [key, value] of route.panels) if (!(tab && key === 'right_tab')) p.append(key, value)
  if (tab) p.append('right_tab', tab)
  return `${RUN_PAGE}?${p}`
}

/**
 * The public URL of where the restored page is now. While the frame shows the run it was opened
 * for, the research context and the native views' state stay; on another run they do not apply.
 */
export function projectedHref(inside: URL, route: HostedRoute): string {
  if (inside.pathname !== RUN_PAGE) throw new HostError('an.leftWorkspace')
  const q = inside.searchParams
  const run = q.get('db_path')
  if (!run) return routeHref(ALL_RUNS)
  const kept = route.run === run ? route : runRoute(run)
  const panels = RUN_PANELS.flatMap((key) => q.getAll(key).map((value): [string, string] => [key, value]))
  const right = q.getAll('right_tab').at(-1) ?? null
  const pane = right === null ? null : paneOf(right)
  return routeHref({
    ...kept,
    tab: 'legacy',
    program: q.get('selected_node') || null,
    // The frame's right tab decides: a tab the inspector also has is that pane, any other ends the pane.
    pane: right === null ? kept.pane : pane,
    panels: pane === null ? panels : panels.filter(([key]) => key !== 'right_tab'),
  })
}
