// Writes hifi/review-manifest.json: SHA-256 of every reviewed artifact plus the per-frame checks the
// capture harness recorded in captures/checks.json (from index.html?frame=<id>&check=1).
// The owner's hi-fi review (tasks.md 1.2) binds to these hashes.
// Usage: node manifest.mjs   (after node build.mjs and a capture of every frame)
import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync } from 'node:fs'

const here = (path) => new URL(path, import.meta.url)
const sha = (path) => createHash('sha256').update(readFileSync(here(path))).digest('hex')
const checks = JSON.parse(readFileSync(here('./captures/checks.json'), 'utf8'))
const failures = (c) => c.overflowX || c.clipped.length || c.badTrunc.length || c.badScroll.length || c.svgOut.length || c.overlaps.length || c.lowContrast.length
const manifest = {
  schema: 'shinka-research.hifi-review-manifest.v1',
  change: 'award-redesign',
  data: 'demo-20261019 · DEMO · 演示数据，非真实研究',
  sources: Object.fromEntries(['build.mjs', 'extract.mjs', 'manifest.mjs', 'fixture.json', 'index.html'].map((f) => [f, sha(`./${f}`)])),
  browser: checks.browser,
  frames: checks.frames.map((f) => ({ id: f.id, width: f.width, height: f.height, png: `captures/${f.id}.png`, sha256: sha(`./captures/${f.id}.png`), passed: !failures(f.checks), checks: f.checks })),
}
manifest.allPassed = manifest.frames.every((f) => f.passed)
writeFileSync(here('./review-manifest.json'), JSON.stringify(manifest, null, 2) + '\n')
console.log(JSON.stringify({ frames: manifest.frames.length, allPassed: manifest.allPassed, indexHtml: manifest.sources['index.html'] }))
