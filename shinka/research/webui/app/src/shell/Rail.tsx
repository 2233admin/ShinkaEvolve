import { untagged } from '../demo'
import { Glyph, Icon, Mark, type IconName } from '../design/icons'
import { directionState, STATE_LOOK } from '../direction'
import type { CopyKey } from '../i18n/copy'
import { useCopy } from '../i18n/lang'
import { ALL_RUNS, routeHref, type Route } from './route'
import type { ShellData } from './useShellData'

type Section = 'research' | 'runs' | 'library'

const NAV: { section: Section; icon: IconName; label: CopyKey; short: CopyKey; route: Route }[] = [
  { section: 'research', icon: 'directions', label: 'nav.research', short: 'nav.researchShort', route: { view: 'directions' } },
  { section: 'runs', icon: 'runs', label: 'nav.runs', short: 'nav.runsShort', route: ALL_RUNS },
  { section: 'library', icon: 'library', label: 'nav.library', short: 'nav.libraryShort', route: { view: 'library', page: null } },
]

/** The nav entry a page sits under; an experiment's analysis stays under its research. */
function sectionOf(route: Route): Section {
  switch (route.view) {
    case 'directions':
    case 'direction':
      return 'research'
    case 'run':
      return route.direction ? 'research' : 'runs'
    case 'runs':
    case 'compare':
      return 'runs'
    case 'library':
      return 'library'
  }
}

type NavProps = { route: Route; data: ShellData; className: string }

/** The three sections with their counts; `page` marks the entry when it is exactly this page. */
function NavLinks({ route, data, className }: NavProps) {
  const { t } = useCopy()
  const section = sectionOf(route)
  const lib = data.workspace?.library
  const counts: Record<Section, number | null> = {
    research: data.workspace ? data.workspace.directions.length : null,
    runs: data.runs ? data.runs.length : null,
    library: data.workspace ? (lib?.pages.length ?? 0) : null,
  }
  const pending = lib?.pending.length ?? 0
  return NAV.map((item) => (
    <a
      key={item.section}
      className={item.section === section ? `${className} is-on` : className}
      href={routeHref(item.route)}
      aria-current={route.view === item.route.view ? 'page' : undefined}
    >
      <Icon name={item.icon} />
      <span className="nav-label">{t(item.label)}</span>
      <span className="nav-short">{t(item.short)}</span>
      <span className="count mono">{counts[item.section] ?? '—'}</span>
      {item.section === 'library' && pending > 0 && (
        <span className="pending mono" title={t('nav.pending', { n: pending })} aria-label={t('nav.pending', { n: pending })}>
          {pending}
        </span>
      )}
    </a>
  ))
}

type RailProps = { route: Route; data: ShellData; narrow: boolean; demo: boolean; onSearch: () => void }

export function Rail({ route, data, narrow, demo, onSearch }: RailProps) {
  const { t } = useCopy()
  const open = route.view === 'direction' || route.view === 'run' ? route.direction : null
  const directions = data.workspace?.directions ?? []
  return (
    <aside className={narrow ? 'rail is-narrow' : 'rail'}>
      <a className="brand" href={routeHref({ view: 'directions' })}>
        <Mark />
        <span>Shinka Research</span>
      </a>
      {data.workspace && (
        <div className="ws">
          <span className="trunc mono" translate="no">{data.workspace.name}</span>
          {demo && <span className="demo" title={t('shell.demo')}>DEMO</span>}
        </div>
      )}
      <button type="button" className="search" onClick={onSearch} aria-keyshortcuts="Control+K">
        <Icon name="search" />
        <span>{t('shell.search')}</span>
        <kbd>Ctrl K</kbd>
      </button>
      <nav aria-label={t('nav.label')}>
        <NavLinks route={route} data={data} className="nav" />
      </nav>
      {directions.length > 0 && (
        <section className="raildirs" aria-labelledby="rail-directions">
          <h2 className="railhead" id="rail-directions">{t('nav.directions')}</h2>
          <ul>
            {directions.map((d) => {
              if (d.error !== null) {
                const why = t('ws.readError', { error: d.error })
                return (
                  <li key={d.id}>
                    <span className="railitem is-error" title={why}>
                      <Glyph kind="error" label={why} />
                      <span className="trunc" translate="no">{d.id}</span>
                    </span>
                  </li>
                )
              }
              const look = STATE_LOOK[directionState(d)]
              return (
                <li key={d.id}>
                  <a
                    className={d.id === open ? 'railitem is-on' : 'railitem'}
                    href={routeHref({ view: 'direction', direction: d.id, hypothesis: null })}
                    aria-current={route.view === 'direction' && d.id === open ? 'page' : undefined}
                    title={untagged(d.direction)}
                  >
                    <Glyph kind={look.glyph} label={t(look.word)} signal={look.signal} />
                    <span className="trunc" translate="no">{untagged(d.direction)}</span>
                  </a>
                </li>
              )
            })}
          </ul>
        </section>
      )}
      {demo && (
        <div className="railfoot">
          <span className="demo" title={t('shell.demo')}>DEMO</span>
          <span className="railfoot-note">{t('shell.demo')}</span>
        </div>
      )}
    </aside>
  )
}

/** Phone navigation: the same three sections as a bottom tab bar. */
export function TabBar({ route, data }: { route: Route; data: ShellData }) {
  const { t } = useCopy()
  return (
    <nav className="tabbar" aria-label={t('nav.label')}>
      <NavLinks route={route} data={data} className="tab" />
    </nav>
  )
}
