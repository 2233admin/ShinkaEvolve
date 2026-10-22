// The overview's direction state, as a reader takes it at a glance (node strips the TypeScript types).
import assert from 'node:assert/strict'
import test from 'node:test'
import { directionState } from './direction.ts'

const direction = (statuses, opened) => ({
  id: 'd',
  direction: 'd',
  created: '',
  hypotheses: Object.values(statuses).reduce((a, b) => a + b, 0),
  trials: 10,
  spend_usd: 0,
  unknown_cost_runs: 0,
  budget_usd: null,
  lockbox_opened: opened,
  evidence: { hypothesis_status: statuses, best_score: null, noise_bar: null, evaluated_programs: 10, failed_programs: 0, last_activity: null },
  error: null,
})

test('a direction concludes once its lockbox is open and no hypothesis still awaits it', () => {
  assert.equal(directionState(direction({ confirmed: 1, refuted: 1 }, true)), 'concluded')
  // An open lockbox with a hypothesis still supported: its confirmation did not finish.
  assert.equal(directionState(direction({ supported: 1, refuted: 1 }, true)), 'active')
  assert.equal(directionState(direction({ refuted: 2 }, false)), 'active')
})
