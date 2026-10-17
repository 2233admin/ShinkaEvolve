import { useEffect, useState } from 'react'
import { Toggle } from '@carbon/react'
import { usd } from './format'
import { TERMS } from './terms'
import type { Snapshot } from './types'

type Props = { snap: Snapshot | null; light: boolean; onTheme: (light: boolean) => void }

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

export function TopBar({ snap, light, onTheme }: Props) {
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
      ? { sub: '未设', icon: 'lock-open', status: 'off' as const }
      : lockbox.opened
        ? { sub: '已开封', icon: 'lock-open', status: 'standby' as const }
        : { sub: '封存', icon: 'lock', status: 'serious' as const }

  const icons =
    snap && budget && lock ? (
      <>
        <rux-monitoring-icon icon="assessment" label="试验数" sublabel={String(snap.trials)} status="normal" title={TERMS.trials} />
        <rux-monitoring-icon icon="attach-money" label="预算" sublabel={budget.sub} status={budget.status} />
        <rux-monitoring-icon icon={lock.icon} label="封存段" sublabel={lock.sub} status={lock.status} title={TERMS.sealed} />
      </>
    ) : null

  // 窄屏放不下，状态图标挪到顶栏下面单独一行。
  return (
    <>
      <rux-global-status-bar app-domain="SHINKA" app-name="RESEARCH" menu-icon="" compact={narrow || undefined}>
        <div slot="right-side" className="bar-right">
          {!narrow && icons}
          <Toggle id="theme" labelText="主题" hideLabel labelA="暗" labelB="亮" toggled={light} onToggle={onTheme} size="sm" />
        </div>
      </rux-global-status-bar>
      {narrow && icons && <div className="status-strip">{icons}</div>}
      <div className="stripes" aria-hidden="true" />
    </>
  )
}
