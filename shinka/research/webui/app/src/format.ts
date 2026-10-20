import type { CopyKey } from './i18n/copy'
import type { Evidence, Hypothesis, Status, Thresholds } from './types'

export type RuxStatus = 'normal' | 'caution' | 'critical' | 'standby' | 'off'

export const STATUS: Record<Status, { key: CopyKey; rux: RuxStatus }> = {
  proposed: { key: 'status.proposed', rux: 'off' },
  selected: { key: 'status.selected', rux: 'standby' },
  running: { key: 'status.running', rux: 'standby' },
  challenged: { key: 'status.challenged', rux: 'caution' },
  supported: { key: 'status.supported', rux: 'normal' },
  refuted: { key: 'status.refuted', rux: 'critical' },
  inconclusive: { key: 'status.inconclusive', rux: 'caution' },
  confirmed: { key: 'status.confirmed', rux: 'normal' },
  unconfirmed: { key: 'status.unconfirmed', rux: 'critical' },
  dropped: { key: 'status.dropped', rux: 'off' },
}

export const EVENT_STATUS: Record<string, RuxStatus> = {
  verdict: 'normal',
  skeptic: 'caution',
  run: 'standby',
  'run-failed': 'critical',
  confirm: 'normal',
}

export const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

// design.md › 格式: scores 2 decimals, DSR 3; negatives take the minus sign (U+2212), and a value
// that rounds to zero shows as zero, never as −0.00. Missing is —.
export const fmt = (v: unknown, digits = 2): string => {
  const n = num(v)
  if (n === null) return '—'
  return (Math.abs(n) < 0.5 * 10 ** -digits ? 0 : n).toFixed(digits).replace(/^-/, '−')
}

export const pct = (v: unknown, digits = 0): string => {
  const n = num(v)
  return n === null ? '—' : `${(n * 100).toFixed(digits)}%`
}

/** Whole counts with thousands separators. */
export const int = (v: number): string => Math.round(v).toLocaleString('en-US')

/** $0.00; a recorded amount under half a cent shows as <$0.01, never as $0.00. */
export const money = (v: number): string =>
  v === 0 ? '$0.00' : Math.abs(v) < 0.005 ? '<$0.01' : `${v < 0 ? '−' : ''}$${Math.abs(v).toFixed(2)}`

export const when = (iso: string): string => iso.slice(0, 16).replace('T', ' ')

export const trialsOf = (h: Hypothesis): number => h.experiments.reduce((s, e) => s + e.n_evaluated + e.n_probed, 0)

export const spendOf = (h: Hypothesis): number => h.experiments.reduce((s, e) => s + e.spend_usd, 0)

export const bestIs = (h: Hypothesis): number | null => {
  const xs = h.experiments.map((e) => e.best_is_score).filter((x): x is number => x !== null)
  return xs.length ? Math.max(...xs) : null
}

export const oosOf = (h: Hypothesis): number | null => (h.verdict ? num(h.verdict.evidence.oos_sharpe) : null)

export type Check = { key: string; pass: boolean | null; value: string }

// 五条规则与 skeptic.decide() 一一对应，阈值用判定时记下的那组。pboNone 是“无法计算”时显示的话。
export function checksOf(ev: Evidence, th: Thresholds, pboNone: string): Check[] {
  const oos = num(ev.oos_sharpe)
  const hold = num(ev.baseline_buy_hold_oos_sharpe)
  const dsr = num(ev.dsr)
  const frac = num(ev.oos_positive_fraction)
  const pbo = num(ev.pbo)
  return [
    { key: 'oos', pass: oos === null ? null : oos > 0, value: fmt(oos) },
    { key: 'hold', pass: oos === null || hold === null ? null : oos > hold, value: `${fmt(oos)} vs ${fmt(hold)}` },
    { key: 'dsr', pass: dsr === null ? null : dsr >= th.dsr_supported, value: fmt(dsr, 3) },
    { key: 'sub', pass: frac === null ? null : frac >= th.min_positive_fraction, value: pct(frac) },
    { key: 'pbo', pass: pbo === null ? true : pbo <= th.max_pbo, value: pbo === null ? pboNone : fmt(pbo) },
  ]
}
