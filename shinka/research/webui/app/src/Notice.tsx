import type { ReactNode } from 'react'

type Props = { kind: 'error' | 'info'; label: string; title: string; subtitle?: string; children?: ReactNode }

// Carbon's InformationFilled and ErrorFilled at 20px, the icons the notice showed before Carbon was removed.
const ICONS = {
  info: {
    box: '0 0 32 32',
    d: 'M16,2A14,14,0,1,0,30,16,14,14,0,0,0,16,2Zm0,6a1.5,1.5,0,1,1-1.5,1.5A1.5,1.5,0,0,1,16,8Zm4,16.125H12v-2.25h2.875v-5.75H13v-2.25h4.125v8H20Z',
  },
  error: {
    box: '0 0 20 20',
    d: 'M10,1c-5,0-9,4-9,9s4,9,9,9s9-4,9-9S15,1,10,1z M13.5,14.5l-8-8l1-1l8,8L13.5,14.5z',
  },
}

/** A one-line notice (design-system Notice): what is wrong or missing, then what to do. No close button, no action. */
export function Notice({ kind, label, title, subtitle, children }: Props) {
  const icon = ICONS[kind]
  return (
    <div className={`notice notice--${kind}`} role="status">
      <div className="notice-details">
        <svg className="notice-icon" width="20" height="20" viewBox={icon.box} role="img" aria-label={label}>
          <path d={icon.d} />
        </svg>
        <div className="notice-text">
          <div className="notice-title">{title}</div>
          {subtitle && <div className="notice-sub">{subtitle}</div>}
          {children}
        </div>
      </div>
    </div>
  )
}
