import { useEffect, useState } from 'react'
import { Toggle } from '@carbon/react'
import { usd } from './format'
import { useCopy, type Lang } from './i18n/lang'
import type { Snapshot } from './types'

type Props = {
  snap: Snapshot | null
  light: boolean
  onTheme: (light: boolean) => void
  onLang: (lang: Lang) => void
}

function useNarrow(): boolean {
  const q = '(max-width: 900px)'
  const [narrow, setNarrow] = useState(() => window.matchMedia(q).matches)
  useEffect(() => {
    const m = window.matchMedia(q)
    const on = () => setNarrow(m.matches)
    m.addEventListener('change', on)
    return () => m.removeEventListener('change', on)
  }, [])
  return narrow
}

export function TopBar({ snap, light, onTheme, onLang }: Props) {
  const { lang, t } = useCopy()
  let budget: { sub: string; status: 'normal' | 'caution' | 'critical' } | null = null
  if (snap) {
    const { spend_usd: s, budget_usd: b } = snap
    const ratio = b ? s / b : 0
    budget = {
      sub: b === null ? usd(s) : `${usd(s)} / ${usd(b)}`,
      status: ratio >= 1 ? 'critical' : ratio >= 0.8 ? 'caution' : 'normal',
    }
  }
  const narrow = useNarrow()
  const lockbox = snap?.lockbox
  const lock = !snap
    ? null
    : lockbox == null
      ? { sub: t('lockbox.unset'), icon: 'lock-open', status: 'off' as const }
      : lockbox.opened
        ? { sub: t('lockbox.opened'), icon: 'lock-open', status: 'standby' as const }
        : { sub: t('lockbox.sealed'), icon: 'lock', status: 'serious' as const }

  const icons =
    snap && budget && lock ? (
      <>
        <rux-monitoring-icon icon="assessment" label={t('top.trials')} sublabel={String(snap.trials)} status="normal" title={t('term.trials')} />
        <rux-monitoring-icon icon="attach-money" label={t('top.budget')} sublabel={budget.sub} status={budget.status} />
        <rux-monitoring-icon icon={lock.icon} label={t('top.lockbox')} sublabel={lock.sub} status={lock.status} title={t('term.sealed')} />
      </>
    ) : null

  // 窄屏放不下，状态图标挪到顶栏下面单独一行。
  return (
    <>
      <rux-global-status-bar app-domain="SHINKA" app-name="RESEARCH" menu-icon="" compact={narrow || undefined}>
        <div slot="right-side" className="bar-right">
          {!narrow && icons}
          <div className="lang-switch" role="group" aria-label={t('lang.label')}>
            <button type="button" lang="zh-CN" translate="no" aria-pressed={lang === 'zh'} onClick={() => onLang('zh')}>中</button>
            <button type="button" lang="en" translate="no" aria-pressed={lang === 'en'} onClick={() => onLang('en')}>EN</button>
          </div>
          <Toggle
            id="theme"
            labelText={t('theme.label')}
            hideLabel
            labelA={t('theme.dark')}
            labelB={t('theme.light')}
            toggled={light}
            onToggle={onTheme}
            size="sm"
          />
        </div>
      </rux-global-status-bar>
      {narrow && icons && <div className="status-strip">{icons}</div>}
      <div className="stripes" aria-hidden="true" />
    </>
  )
}
