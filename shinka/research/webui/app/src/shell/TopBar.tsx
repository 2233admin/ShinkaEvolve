import { useState } from 'react'
import { Glyph, Icon, Mark } from '../design/icons'
import type { ThemeName } from '../design/theme'
import { when } from '../format'
import { useCopy, type Lang } from '../i18n/lang'
import { Modal } from './Modal'
import { routeHref } from './route'
import type { ShellData } from './useShellData'

export type Crumb = {
  label: string
  href?: string
  /** Research data (a direction title, a run name) is shown as written. */
  data?: boolean
}

type ChoiceProps<T extends string> = {
  label: string
  value: T
  options: { value: T; text: string; lang?: string }[]
  onChange: (value: T) => void
}

/** A two-way segmented switch: one pressed button per option. */
function Choice<T extends string>({ label, value, options, onChange }: ChoiceProps<T>) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          lang={o.lang}
          translate={o.lang ? 'no' : undefined}
          aria-pressed={o.value === value}
          onClick={() => onChange(o.value)}
        >
          {o.text}
        </button>
      ))}
    </div>
  )
}

/**
 * OD-17: when the data last loaded; a click refreshes now. A failed refresh says so and keeps the
 * data; a paused refresh says so with its shape and word.
 */
export function Updated({ data }: { data: ShellData }) {
  const { t } = useCopy()
  if (data.updated === null) {
    return data.error === null ? (
      <span className="updated">{t('state.loading')}</span>
    ) : (
      <button type="button" className="updated is-stale" onClick={data.refresh}>
        <Glyph kind="error" signal />
        <span className="sig">{t('shell.stale')}</span>
        <span>{t('action.refresh')}</span>
      </button>
    )
  }
  const time = when(data.updated.toISOString())
  const stale = data.error !== null
  const hint = stale ? t('shell.staleHint', { time }) : data.live ? t('shell.updatedHint', { s: data.every / 1000 }) : t('shell.pausedHint')
  return (
    <button type="button" className={stale ? 'updated is-stale' : 'updated'} onClick={data.refresh} title={hint}>
      {stale && <Glyph kind="error" signal />}
      {stale && <span className="sig">{t('shell.stale')}</span>}
      {!data.live && (
        <span className="paused">
          <Icon name="pause" />
          {t('shell.paused')}
        </span>
      )}
      <span>{t('shell.updated')}</span>
      <span className="mono">{time} UTC</span>
    </button>
  )
}

/** Pauses or resumes the automatic refresh of every page the shell draws itself (P01-C5 under OD-17). */
function LiveToggle({ data }: { data: ShellData }) {
  const { t } = useCopy()
  const label = t(data.live ? 'shell.pause' : 'shell.resume')
  return (
    <button type="button" className="live-toggle" aria-label={label} title={label} aria-pressed={!data.live} onClick={() => data.setLive(!data.live)}>
      <Icon name={data.live ? 'pause' : 'play'} />
    </button>
  )
}

type Props = {
  crumbs: Crumb[]
  data: ShellData
  /** Demo data: on phones, where the rail is hidden, the top bar carries the DEMO mark. */
  demo: boolean
  /** False while a restored analysis page with its own refresh switch fills the page (until task 6.1). */
  liveToggle: boolean
  theme: ThemeName
  onTheme: (theme: ThemeName) => void
  onLang: (lang: Lang) => void
  onSearch: () => void
}

export function TopBar({ crumbs, data, demo, liveToggle, theme, onTheme, onLang, onSearch }: Props) {
  const { lang, t } = useCopy()
  const [settings, setSettings] = useState(false)
  const tools = (
    <>
      <span className="updated-group">
        <Updated data={data} />
        {liveToggle && data.updated !== null && <LiveToggle data={data} />}
      </span>
      <Choice
        label={t('theme.label')}
        value={theme}
        options={[{ value: 'dark', text: t('theme.dark') }, { value: 'light', text: t('theme.light') }]}
        onChange={onTheme}
      />
      <Choice
        label={t('lang.label')}
        value={lang}
        options={[{ value: 'zh', text: '中', lang: 'zh-CN' }, { value: 'en', text: 'EN', lang: 'en' }]}
        onChange={onLang}
      />
    </>
  )
  return (
    <header className="top">
      <a className="top-mark" href={routeHref({ view: 'directions' })} aria-label="Shinka Research">
        <Mark />
      </a>
      <nav className="crumbs" aria-label={t('crumb.label')}>
        <ol>
          {crumbs.map((crumb, index) => (
            <li key={index}>
              {index === crumbs.length - 1 || crumb.href === undefined ? (
                <span className="is-here trunc" aria-current={index === crumbs.length - 1 ? 'page' : undefined} translate={crumb.data ? 'no' : undefined}>
                  {crumb.label}
                </span>
              ) : (
                <a className="trunc" href={crumb.href} translate={crumb.data ? 'no' : undefined}>
                  {crumb.label}
                </a>
              )}
            </li>
          ))}
        </ol>
      </nav>
      {demo && <span className="demo top-demo" title={t('shell.demo')}>DEMO</span>}
      <div className="tools">{tools}</div>
      <button type="button" className="top-button top-search" aria-label={t('shell.search')} aria-keyshortcuts="Control+K" onClick={onSearch}>
        <Icon name="search" />
      </button>
      <button type="button" className="top-button top-settings" aria-haspopup="dialog" onClick={() => setSettings(true)}>
        {t('shell.settings')}
      </button>
      <Modal open={settings} onClose={() => setSettings(false)} label={t('shell.settings')} className="sheet">
        <div className="sheet-body">
          <div className="sheet-head">
            <h2>{t('shell.settings')}</h2>
            <button type="button" className="top-button" onClick={() => setSettings(false)}>
              {t('shell.close')}
            </button>
          </div>
          <div className="sheet-tools">{tools}</div>
        </div>
      </Modal>
    </header>
  )
}
