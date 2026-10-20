import type { DetailedHTMLProps, HTMLAttributes } from 'react'

type RuxStatusValue = 'normal' | 'caution' | 'serious' | 'critical' | 'standby' | 'off'
type El<P = object> = DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement> & P

export interface RuxLogEntry {
  timestamp: Date
  status: RuxStatusValue
  message: string
}

declare module 'react' {
  namespace JSX {
    interface IntrinsicElements {
      'rux-status': El<{ status: RuxStatusValue }>
      'rux-log': El<{ data: RuxLogEntry[] }>
      'rux-timeline': El<{ start: string; end: string; interval?: 'month' | 'week' | 'hour' | 'day' | 'minute'; timezone?: string; hideJDay?: boolean; zoom?: number; showGrid?: boolean; rulerPosition?: 'top' | 'bottom' | 'both' }>
      'rux-track': El
      'rux-time-region': El<{ start: string; end: string; status?: RuxStatusValue; selected?: boolean; hideTimestamp?: boolean }>
    }
  }
}
