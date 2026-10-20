import type { GlyphKind } from './design/icons'
import type { CopyKey } from './i18n/copy'
import type { DirectionSummary, Status } from './types'

export type DirectionState = 'active' | 'concluded' | 'new' | 'over-budget'

// Statuses that end a hypothesis; a direction concludes once every hypothesis has one and the
// lockbox is open. Supported is not among them: confirm turns every supported hypothesis into
// confirmed or unconfirmed, so one still supported behind an open lockbox means that
// confirmation is unfinished.
const SETTLED: ReadonlySet<Status> = new Set<Status>(['refuted', 'inconclusive', 'confirmed', 'unconfirmed', 'dropped'])

/** The overview's state of a direction: the approved high-fidelity package's rule, except that a supported hypothesis still awaits the lockbox. */
export function directionState(d: DirectionSummary): DirectionState {
  if (d.budget_usd !== null && d.spend_usd > d.budget_usd) return 'over-budget'
  if (d.evidence.evaluated_programs === 0) return 'new'
  const statuses = Object.keys(d.evidence.hypothesis_status) as Status[]
  const settled = d.hypotheses > 0 && d.lockbox_opened === true && statuses.every((s) => SETTLED.has(s))
  return settled ? 'concluded' : 'active'
}

/** Shape, word and colour of each state; only over-budget takes the signal orange. */
export const STATE_LOOK: Record<DirectionState, { glyph: GlyphKind; word: CopyKey; signal: boolean }> = {
  active: { glyph: 'dir-active', word: 'dstate.active', signal: false },
  concluded: { glyph: 'dir-concluded', word: 'dstate.concluded', signal: false },
  new: { glyph: 'dir-new', word: 'dstate.new', signal: false },
  'over-budget': { glyph: 'dir-over', word: 'dstate.over', signal: true },
}
