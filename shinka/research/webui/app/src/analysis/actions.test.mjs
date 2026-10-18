import assert from 'node:assert/strict'
import test from 'node:test'
import { parseAction, plotURL } from './actions.js'

const allowed = new Set(['selectNodeById', 'refreshChartAxes', 'open'])
const input = { value: '-0.25', checked: true }

test('saved node link preserves event control and one exact selection', () => {
  assert.deepEqual(parseAction("event.preventDefault(); selectNodeById('fixture-a2');", allowed, input, {}), [
    { name: '$event', args: ['preventDefault'] }, { name: 'selectNodeById', args: ['fixture-a2'] },
  ])
  assert.deepEqual(parseAction('refreshChartAxes()', allowed, input, {}), [{ name: 'refreshChartAxes', args: [] }])
})

test('executable arguments and suffixes are rejected as a whole', () => {
  for (const action of [
    "selectNodeById('a'); alert(1)", "setTimeout('alert(1)', 0)",
    'refreshChartAxes(() => alert(1))', 'selectNodeById(window.document.cookie)',
    "selectNodeById('a'+location.href)", "selectNodeById.constructor('alert(1)')()",
  ]) assert.equal(parseAction(action, allowed, input, {}), null, action)
})

test('plot action is parsed once and cannot open another origin or script URL', () => {
  assert.deepEqual(parseAction("window.open('/plot_file/runs/test/gen_0/results/plots/a.png', '_blank')", allowed, input, {}), [
    { name: 'open', args: ['/plot_file/runs/test/gen_0/results/plots/a.png', '_blank'] },
  ])
  const origin = 'http://127.0.0.1:8769'
  assert.equal(plotURL('/plot_file/runs/a.png', origin)?.pathname, '/plot_file/runs/a.png')
  for (const url of ['https://example.com/plot_file/a.png', '//example.com/plot_file/a.png', 'javascript:alert(1)', '/library/raw/a.png', '/plot_file/../library/raw/a.png']) {
    assert.equal(plotURL(url, origin), null, url)
  }
})
