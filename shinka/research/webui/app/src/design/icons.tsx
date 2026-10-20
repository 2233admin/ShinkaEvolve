import type { ReactNode } from 'react'

// Shapes from the approved high-fidelity package (hifi/build.mjs › glyphs). Direction states are
// squares and hypothesis states circles, so the two never share a shape; a glyph always sits next to
// its word, never alone.

const CUT = { stroke: 'var(--bg)' }

const GLYPHS = {
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
