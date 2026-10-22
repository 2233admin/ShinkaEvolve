// Library and notebook text is untrusted: markup shows as text and links stay links
// (node strips the TypeScript types).
import assert from 'node:assert/strict'
import test from 'node:test'
import { renderMarkdown } from './renderMarkdown.ts'
import { parseRoute } from './shell/route.ts'

test('markup in a page is shown as text and a script link is not a link', () => {
  const html = renderMarkdown('<svg onload=alert(1)><script>alert(2)</script>\n\n[x](javascript:alert(3))')
  assert.doesNotMatch(html, /<svg|<script/i)
  assert.doesNotMatch(html, /href="javascript:/i)
  assert.match(html, /&lt;svg onload=alert\(1\)&gt;/)
})

test('a [[page]] link opens that library page in the app; any other link opens a new tab', () => {
  const html = renderMarkdown('See [[concept lockbox|the lockbox]] and https://example.org')
  const href = html.match(/<a href="([^"]+)" data-wiki/)[1].replaceAll('&amp;', '&')
  assert.deepEqual(parseRoute(new URL(href, 'http://shell').search), { view: 'library', page: 'wiki/concept lockbox' })
  assert.match(html, />the lockbox<\/a>/)
  assert.match(html, /href="https:\/\/example\.org" target="_blank" rel="noopener noreferrer"/)
})
