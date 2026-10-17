import type { DetailedHTMLProps, HTMLAttributes } from 'react'

type RuxStatusValue = 'normal' | 'caution' | 'critical' | 'standby' | 'off'
type El<P = object> = DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement> & P

export interface RuxLogEntry {
  timestamp: Date
  status: RuxStatusValue
  message: string
}

declare module 'react' {
  namespace JSX {
    interface IntrinsicElements {
      'rux-global-status-bar': El<{ 'app-domain'?: string; 'app-name'?: string; 'app-version'?: string; 'menu-icon'?: string; compact?: boolean }>
      'rux-monitoring-icon': El<{ icon: string; label?: string; sublabel?: string; status?: RuxStatusValue; title?: string }>
      'rux-status': El<{ status: RuxStatusValue }>
      'rux-log': El<{ data: RuxLogEntry[] }>
    }
  }
}
