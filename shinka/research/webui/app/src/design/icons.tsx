import type { ReactNode } from 'react'

// Shapes from the approved high-fidelity package (hifi/build.mjs › glyphs). Direction states are
// squares and hypothesis states circles, so the two never share a shape; a glyph always sits next to
// its word, never alone.

const CUT = { stroke: 'var(--bg)' }

const GLYPHS = {
  supported: <circle cx="7" cy="7" r="5.5" fill="currentColor" />,
  confirmed: (
    <>
      <circle cx="7" cy="7" r="5.5" fill="currentColor" />
      <path d="M4.4 7.2l1.8 1.8 3.4-3.6" style={CUT} strokeWidth="1.5" fill="none" />
    </>
  ),
  refuted: (
    <>
      <circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M3.6 10.4l6.8-6.8" stroke="currentColor" strokeWidth="1.4" />
    </>
  ),
  inconclusive: (
    <>
      <circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M7 2a5 5 0 0 1 0 10z" fill="currentColor" />
    </>
  ),
  running: (
    <>
      <circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeDasharray="2.2 1.9" />
      <circle cx="7" cy="7" r="1.8" fill="currentColor" />
    </>
  ),
  challenged: (
    <>
      <circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M7 4v3.4" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="7" cy="9.6" r=".9" fill="currentColor" />
    </>
  ),
  proposed: <circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" strokeWidth="1.3" strokeDasharray="1 2.1" />,
  selected: <circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" strokeWidth="1.3" />,
  unconfirmed: (
    <>
      <circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M4.8 4.8l4.4 4.4m0-4.4l-4.4 4.4" stroke="currentColor" strokeWidth="1.3" />
    </>
  ),
  dropped: <circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" strokeWidth="1.2" strokeDasharray="3 2" />,
  'dir-active': (
    <>
      <rect x="2" y="2" width="10" height="10" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <rect x="5" y="5" width="4" height="4" fill="currentColor" />
    </>
  ),
  'dir-concluded': (
    <>
      <rect x="1.5" y="1.5" width="11" height="11" rx="1.5" fill="currentColor" />
      <path d="M4.3 7.2l1.8 1.8 3.6-3.8" style={CUT} strokeWidth="1.5" fill="none" />
    </>
  ),
  'dir-new': (
    <rect x="2" y="2" width="10" height="10" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.3" strokeDasharray="1.6 1.6" />
  ),
  'dir-over': (
    <>
      <rect x="2" y="2" width="10" height="10" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M4 10L10 4" stroke="currentColor" strokeWidth="1.4" />
      <path d="M10 4v3M10 4H7" stroke="currentColor" strokeWidth="1.4" fill="none" />
    </>
  ),
  error: (
    <>
      <path d="M7 1.6l5.8 10.2H1.2z" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
      <path d="M7 5.4v3" stroke="currentColor" strokeWidth="1.4" />
      <circle cx="7" cy="10" r=".8" fill="currentColor" />
    </>
  ),
  // Nothing to show yet (hifi `empty`): rounder corners and longer dashes than a new direction's square.
  empty: <rect x="2" y="2" width="10" height="10" rx="2" fill="none" stroke="currentColor" strokeWidth="1.3" strokeDasharray="2 1.6" />,
} satisfies Record<string, ReactNode>

export type GlyphKind = keyof typeof GLYPHS

/** A 14px state shape. Decorative unless labelled; `signal` paints it in the one signal orange. */
export function Glyph({ kind, label, signal = false }: { kind: GlyphKind; label?: string; signal?: boolean }) {
  return (
    <svg
      className={signal ? 'glyph is-signal' : 'glyph'}
      width="14"
      height="14"
      viewBox="0 0 14 14"
      role={label ? 'img' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      {GLYPHS[kind]}
    </svg>
  )
}

// Nav icons are drawn on a 16px grid, the small ones on 14px, exactly as approved.
const ICONS = {
  search: {
    box: 14,
    body: (
      <>
        <circle cx="6" cy="6" r="4.2" fill="none" stroke="currentColor" strokeWidth="1.4" />
        <path d="M9.2 9.2l3.3 3.3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      </>
    ),
  },
  directions: {
    box: 16,
    body: (
      <>
        <path d="M3 13.5V8.5m0 0c0-2.8 2-4 4.8-4M3 8.5c0 2.4 2.2 3.6 4.8 3.6M7.8 4.5h5.2M7.8 12.1H13" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
        <circle cx="3" cy="13.5" r="1.3" fill="currentColor" />
      </>
    ),
  },
  runs: {
    box: 16,
    body: (
      <>
        <rect x="2" y="3" width="12" height="2.6" rx=".6" fill="none" stroke="currentColor" strokeWidth="1.3" />
        <rect x="2" y="6.8" width="12" height="2.6" rx=".6" fill="none" stroke="currentColor" strokeWidth="1.3" />
        <rect x="2" y="10.6" width="12" height="2.6" rx=".6" fill="none" stroke="currentColor" strokeWidth="1.3" />
      </>
    ),
  },
  library: {
    box: 16,
    body: (
      <path d="M2.5 3.2c2-.6 3.8-.4 5.5.8 1.7-1.2 3.5-1.4 5.5-.8v9.6c-2-.6-3.8-.4-5.5.8-1.7-1.2-3.5-1.4-5.5-.8zM8 4v9.6" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
    ),
  },
  lock: {
    box: 14,
    body: (
      <>
        <rect x="2.5" y="6" width="9" height="6.5" rx="1.2" fill="none" stroke="currentColor" strokeWidth="1.3" />
        <path d="M4.5 6V4.3a2.5 2.5 0 0 1 5 0V6" fill="none" stroke="currentColor" strokeWidth="1.3" />
      </>
    ),
  },
  unlock: {
    box: 14,
    body: (
      <>
        <rect x="2.5" y="6" width="9" height="6.5" rx="1.2" fill="none" stroke="currentColor" strokeWidth="1.3" />
        <path d="M4.5 6V4.3a2.5 2.5 0 0 1 4.9-.7" fill="none" stroke="currentColor" strokeWidth="1.3" />
      </>
    ),
  },
  check: {
    box: 12,
    body: <path d="M2.5 6.2l2.4 2.4 4.6-5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />,
  },
  cross: {
    box: 12,
    body: <path d="M3 3l6 6m0-6l-6 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />,
  },
  arrow: {
    box: 12,
    body: <path d="M2 6h7.5M6.5 3l3 3-3 3" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />,
  },
  pause: {
    box: 12,
    body: <path d="M4 2.5v7M8 2.5v7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />,
  },
  play: {
    box: 12,
    body: <path d="M3.5 2.3l6 3.7-6 3.7z" fill="currentColor" />,
  },
  chevron: {
    box: 12,
    body: <path d="M3.5 4.5l2.5 2.5 2.5-2.5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />,
  },
  prev: {
    box: 12,
    body: <path d="M7.5 2.5L4 6l3.5 3.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />,
  },
  next: {
    box: 12,
    body: <path d="M4.5 2.5L8 6l-3.5 3.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />,
  },
  download: {
    box: 14,
    body: <path d="M7 2v7m-3-3l3 3 3-3M2.5 11.5h9" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />,
  },
} satisfies Record<string, { box: number; body: ReactNode }>

export type IconName = keyof typeof ICONS

/** A decorative line icon at its drawn size; the control around it carries the name. */
export function Icon({ name }: { name: IconName }) {
  const { box, body } = ICONS[name]
  return (
    <svg className="icon" width={box} height={box} viewBox={`0 0 ${box} ${box}`} aria-hidden="true">
      {body}
    </svg>
  )
}

/** The Shinka Research mark: an evolving branch. */
export function Mark() {
  return (
    <svg className="mark" width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      <path d="M4 14.5V10m0 0c0-3 2.2-4 5-4m-5 4c0 2.6 2.4 4 5 4M9 6V3.5M9 6c2.8 0 5 1 5 4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="9" cy="3.5" r="1.6" fill="currentColor" />
      <circle cx="14" cy="10" r="1.6" fill="currentColor" />
      <circle cx="9" cy="14" r="1.6" fill="currentColor" />
    </svg>
  )
}

// Event marks: the shape goes with the event word, never colour alone; a failed run is the signal.
const EVENT_SHAPES: Record<string, ReactNode> = {
  'run-failed': <path d="M3 3l6 6m0-6l-6 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />,
  verdict: <path d="M6 1.5l4.5 4.5L6 10.5 1.5 6z" fill="currentColor" />,
  skeptic: <path d="M6 1.8l4.2 4.2L6 10.2 1.8 6z" fill="none" stroke="currentColor" strokeWidth="1.3" />,
  run: <rect x="2.5" y="2.5" width="7" height="7" fill="currentColor" />,
  scaffold: <path d="M6 2v8M2 6h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />,
  probe: <path d="M6 2.2l4 7H2z" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />,
}

export function EventMark({ event }: { event: string }) {
  if (event === 'confirm' || event === 'lockbox') {
    return (
      <span className="evmark">
        <Icon name="unlock" />
      </span>
    )
  }
  return (
    <span className={event === 'run-failed' ? 'evmark is-signal' : 'evmark'}>
      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
        {EVENT_SHAPES[event] ?? EVENT_SHAPES.run}
      </svg>
    </span>
  )
}
