import type { CopyKey } from './i18n/copy'
import { ALL_RUNS, hostedView, RUN_PANELS, routeHref, runRoute, type Route } from './shell/route.ts'

/** The route the restored analysis page draws inside the frame: `tab=legacy`, `space` or `contest`. */
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

/**
 * The restored page's left panel each hosted view opens on when the URL names none: the PCA scatter
 * for solution space (L06), the ranked program table for contest (OD-2's leaderboard).
 */
const VIEW_PANEL = { space: 'clusters-view', contest: 'table-view' } as const

const paneOf = (tab: string): string | null => [...PANE_TABS].find(([, value]) => value === tab)?.[0] ?? null

/**
 * The restored page that draws a hosted route: the program stays selected, the pane opens its tab
 * there, and a hosted view without a left panel in the URL opens on its own.
 */
export function framePath(route: HostedRoute): string {
  const p = new URLSearchParams()
  p.set('db_path', route.run)
  if (route.program) p.set('selected_node', route.program)
  // The inspector's pane is a tab of the restored page; it wins over a right_tab the URL also carries.
  const tab = route.pane === null ? null : (PANE_TABS.get(route.pane) ?? null)
  for (const [key, value] of route.panels) if (!(tab && key === 'right_tab')) p.append(key, value)
  if (tab) p.append('right_tab', tab)
  const view = hostedView(route.tab)
  if (view && !p.has('left_tab')) p.set('left_tab', VIEW_PANEL[view])
  return `${RUN_PAGE}?${p}`
}

/**
 * The public URL of where the restored page is now. While the frame shows the run it was opened
 * for, the research context, the native views' state and the hosted view it was opened as stay
 * (whatever panel the reader moves to inside it); on another run they do not apply.
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
    tab: hostedView(kept.tab) ?? 'legacy',
    program: q.get('selected_node') || null,
    // The frame's right tab decides: a tab the inspector also has is that pane, any other ends the pane.
    pane: right === null ? kept.pane : pane,
    panels: pane === null ? panels : panels.filter(([key]) => key !== 'right_tab'),
  })
}
