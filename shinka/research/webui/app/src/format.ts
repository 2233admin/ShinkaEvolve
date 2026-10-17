import type { Evidence, Hypothesis, Status, Thresholds } from './types'

export type RuxStatus = 'normal' | 'caution' | 'critical' | 'standby' | 'off'

export const STATUS: Record<Status, { label: string; rux: RuxStatus }> = {
  proposed: { label: '待跑', rux: 'off' },
  selected: { label: '已选', rux: 'standby' },
  running: { label: '运行中', rux: 'standby' },
  challenged: { label: '待判定', rux: 'caution' },
  supported: { label: '成立', rux: 'normal' },
  refuted: { label: '否定', rux: 'critical' },
  inconclusive: { label: '未定', rux: 'caution' },
  confirmed: { label: '已确认', rux: 'normal' },
  unconfirmed: { label: '未确认', rux: 'critical' },
  dropped: { label: '放弃', rux: 'off' },
}

export const EVENT_STATUS: Record<string, RuxStatus> = {
  verdict: 'normal',
  skeptic: 'caution',
  run: 'standby',
  'run-failed': 'critical',
  confirm: 'normal',
}

export const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

export const fmt = (v: unknown, digits = 2): string => {
  const n = num(v)
  return n === null ? '—' : n.toFixed(digits)
}

export const pct = (v: unknown): string => {
  const n = num(v)
  return n === null ? '—' : `${(n * 100).toFixed(0)}%`
}

export const usd = (v: number): string => `$${v.toFixed(2)}`

export const when = (iso: string): string => iso.slice(0, 16).replace('T', ' ')

export const trialsOf = (h: Hypothesis): number => h.experiments.reduce((s, e) => s + e.n_evaluated + e.n_probed, 0)

export const spendOf = (h: Hypothesis): number => h.experiments.reduce((s, e) => s + e.spend_usd, 0)

export const bestIs = (h: Hypothesis): number | null => {
  const xs = h.experiments.map((e) => e.best_is_score).filter((x): x is number => x !== null)
  return xs.length ? Math.max(...xs) : null
}

export const oosOf = (h: Hypothesis): number | null => (h.verdict ? num(h.verdict.evidence.oos_sharpe) : null)

export type Check = { key: string; label: string; pass: boolean | null; value: string }

// 五条规则与 skeptic.decide() 一一对应，阈值用判定时记下的那组。
export function checksOf(ev: Evidence, th: Thresholds): Check[] {
  const oos = num(ev.oos_sharpe)
  const hold = num(ev.baseline_buy_hold_oos_sharpe)
  const dsr = num(ev.dsr)
  const frac = num(ev.oos_positive_fraction)
  const pbo = num(ev.pbo)
  return [
    { key: 'oos', label: '验证段 Sharpe > 0', pass: oos === null ? null : oos > 0, value: fmt(oos) },
    {
      key: 'hold',
      label: '跑赢买入持有',
      pass: oos === null || hold === null ? null : oos > hold,
      value: `${fmt(oos)} vs ${fmt(hold)}`,
    },
    { key: 'dsr', label: `DSR ≥ ${th.dsr_supported}`, pass: dsr === null ? null : dsr >= th.dsr_supported, value: fmt(dsr, 3) },
    {
      key: 'sub',
      label: `子区间为正 ≥ ${pct(th.min_positive_fraction)}`,
      pass: frac === null ? null : frac >= th.min_positive_fraction,
      value: pct(frac),
    },
    {
      key: 'pbo',
      label: `PBO ≤ ${th.max_pbo}`,
      pass: pbo === null ? true : pbo <= th.max_pbo,
      value: pbo === null ? '无法计算，视为通过' : fmt(pbo),
    },
  ]
}
