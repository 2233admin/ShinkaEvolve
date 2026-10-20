// design.md › 格式 as every number on screen follows it (node strips the TypeScript types).
import assert from 'node:assert/strict'
import test from 'node:test'
import { fmt, pct } from './format.ts'

test('a negative number takes the minus sign and one that rounds to zero shows as zero', () => {
  assert.equal(fmt(-1.234), '−1.23')
  assert.equal(fmt(-0.001), '0.00')
  // A lockbox return or drawdown below zero.
  assert.equal(pct(-0.0512, 1), '−5.1%')
  assert.equal(pct(-0.0004, 1), '0.0%')
  assert.equal(pct(null), '—')
})
