// Renders the three direction candidates for the award-redesign change from the shared
// fixture (./fixture.json, demo data) into a self-contained ./index.html.
// Usage: node build.mjs   (needs the app's installed d3 and fonts; no network)
import { readFileSync, writeFileSync } from 'node:fs'
import * as d3 from '../../../../app/node_modules/d3/src/index.js'

const here = (path) => new URL(path, import.meta.url)
const fixture = JSON.parse(readFileSync(here('./fixture.json'), 'utf8'))
const font = (path) => readFileSync(here(`../../../../app/node_modules/${path}`)).toString('base64')
const INTER = font('@fontsource-variable/inter/files/inter-latin-wght-normal.woff2')
const PLEX400 = font('@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2')
const PLEX500 = font('@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-500-normal.woff2')

const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c])
const int = (n) => (n ?? 0).toLocaleString('en-US')
const num = (n, d = 2) => (n === null || n === undefined ? '—' : (Math.abs(n) < 0.5 * 10 ** -d ? 0 : Number(n)).toFixed(d))
const money = (n) => (n === null || n === undefined ? '未设' : `$${Number(n).toFixed(2)}`)
const pad = (n) => String(n).padStart(2, '0')
// Times are UTC: YYYY-MM-DD HH:MM for instants and YYYY-MM-DD for days (design-system/README.md:10).
const clock = (iso) => (iso ? new Date(iso).toISOString().slice(0, 16).replace('T', ' ') : '—')
const day = (iso) => (iso ? new Date(iso).toISOString().slice(0, 10) : '—')
const STATUS = { proposed: '待跑', selected: '已选', running: '运行中', challenged: '待判定', supported: '成立', refuted: '否定', inconclusive: '未定', confirmed: '已确认', unconfirmed: '未确认', dropped: '放弃' }
const STATE = { active: '进行中', concluded: '已结论', new: '新建', 'over-budget': '超预算' }
const EVENT = { scaffold: '搭建实验', run: '运行', 'run-failed': '运行失败', skeptic: '质疑检验', verdict: '判词', probe: '试探', confirm: '封存段确认', lockbox: '封存段', hypothesis: '提出假设', note: '备注' }
const eventLabel = (event) => EVENT[event] ?? event

const dirs = fixture.directions.filter((d) => !d.error)
const total = {
  trials: d3.sum(dirs, (d) => d.trials),
  spend: d3.sum(dirs, (d) => d.spend),
  budget: d3.sum(dirs, (d) => d.budget ?? 0),
  above: dirs.filter((d) => d.best !== null && d.bar !== null && d.best > d.bar).length,
  withTrials: dirs.filter((d) => d.trials > 0).length,
  challenged: d3.sum(dirs, (d) => d.hypotheses.filter((h) => h.status === 'challenged').length),
  failed: d3.sum(dirs, (d) => d.failed),
  overBudget: dirs.filter((d) => d.state === 'over-budget').length,
  confirmed: dirs.filter((d) => d.hypotheses.some((h) => h.status === 'confirmed')).length,
  running: d3.sum(dirs, (d) => d.hypotheses.filter((h) => h.status === 'running').length),
}
const failRate = total.trials ? total.failed / total.trials : 0
// One evidence scale for every direction, so best records and noise bars compare across rows.
const evidenceDomain = (() => {
  const [lo, hi] = d3.extent(dirs.flatMap((d) => [d.best, d.bar]).filter((v) => v !== null))
  const margin = (hi - lo) * 0.12 || 0.1
  return [lo - margin, hi + margin]
})()
const evidenceMajors = d3.ticks(...evidenceDomain, 8).filter((t) => Math.round(t * 10) % 5 === 0)
const shortTitle = (d) => d.title.replace(/[（(].*?[)）]/g, '').trim()
const statusCounts = (d) => d3.rollups(d.hypotheses, (v) => v.length, (h) => h.status)
const A = fixture.analysis
const rows = A.programs // [x, y, score|null, generation, island, parentIndex, patch, model]
const pathSet = new Set(A.path)

// ---------- shared geometry ----------
// The noise line is sampled on its own grid; keep it inside the trial range and end it exactly at
// the direction's current noise bar, so the chart label and every other readout show one number.
function recordSeries(d) {
  if (!d.records.length) return { steps: [], bars: d.bars }
  const steps = [...d.records]
  steps.push([d.trials, steps.at(-1)[1]])
  const bars = d.bars.filter((p) => p[0] < d.trials)
  if (d.bar !== null) bars.push([d.trials, d.bar])
  return { steps, bars }
}
function trialChart(d, { w, h, m = { t: 8, r: 8, b: 18, l: 8 }, ink, muted, signal, grid, axis = false, labels = false, font = 'inherit', xDomain = null, yDomain = null }) {
  const { steps, bars } = recordSeries(d)
  if (!steps.length) return ''
  const values = [...steps.map((p) => p[1]), ...bars.map((p) => p[1])].filter((v) => v !== null)
  const extent = d3.extent(values)
  const span = (extent[1] - extent[0]) || 1
  const x = d3.scaleLinear(xDomain ?? [1, d.trials], [m.l, w - m.r])
  const y = d3.scaleLinear(yDomain ?? [extent[0] - span * 0.12, extent[1] + span * 0.18], [h - m.b, m.t])
  const step = d3.line().curve(d3.curveStepAfter).x((p) => x(p[0])).y((p) => y(p[1]))(steps)
  const barLine = d3.line().curve(d3.curveMonotoneX).x((p) => x(p[0])).y((p) => y(p[1]))(bars)
  const ticks = x.ticks(4).filter((t) => Number.isInteger(t))
  const last = steps.at(-1)
  const lastBar = bars.at(-1)
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true" style="display:block;overflow:visible;font-family:${font}">
    ${grid ? y.ticks(3).map((t) => `<line x1="${m.l}" x2="${w - m.r}" y1="${y(t)}" y2="${y(t)}" stroke="${grid}" stroke-width="1"/>${yDomain ? `<text x="${m.l - 4}" y="${y(t) + 3}" font-size="9.5" text-anchor="end" fill="${muted}">${num(t)}</text>` : ''}`).join('') : ''}
    <path d="${barLine}" fill="none" stroke="${muted}" stroke-width="1.25" stroke-dasharray="3 3"/>
    <path d="${step}" fill="none" stroke="${d.best > d.bar ? signal : ink}" stroke-width="1.6" stroke-linejoin="round"/>
    <circle cx="${x(last[0])}" cy="${y(last[1])}" r="2.6" fill="${d.best > d.bar ? signal : ink}"/>
    ${axis ? ticks.map((t) => `<text x="${x(t)}" y="${h - 3}" font-size="10" text-anchor="middle" fill="${muted}">${int(t)}</text>`).join('') : ''}
    ${labels ? `<text x="${x(last[0]) - 6}" y="${y(last[1]) - 7}" font-size="11" text-anchor="end" fill="${d.best > d.bar ? signal : ink}">最佳 ${num(last[1])}</text>
    <text x="${x(lastBar[0]) - 6}" y="${y(lastBar[1]) + 14}" font-size="11" text-anchor="end" fill="${muted}">噪声线 ${num(lastBar[1])}</text>` : ''}
  </svg>`
}
function hierarchy(size) {
  const nodes = [{ id: 'root', parent: null }, ...rows.map((r, i) => ({ id: String(i), parent: r[5] === -1 ? 'root' : String(r[5]), row: r, i }))]
  const root = d3.stratify().id((n) => n.id).parentId((n) => n.parent)(nodes)
  root.sort((a, b) => (a.data.row?.[4] ?? 0) - (b.data.row?.[4] ?? 0) || (a.data.row?.[3] ?? 0) - (b.data.row?.[3] ?? 0))
  d3.cluster().size(size).separation((a, b) => (a.parent === b.parent ? 1 : 1.25))(root)
  return root
}
const scoreExtent = d3.extent(rows.filter((r) => r[2] !== null), (r) => r[2])
const maxGen = d3.max(rows, (r) => r[3])

// Score field over the embedding projection (Nadaraya-Watson), the density of every program and
// the density of programs with a valid score. The scored coastline is first filled at the lowest
// score level, then the score bands are drawn inside it, so no scored ground shows the hatch;
// explored ground without valid scores (failures only) keeps the hatch and is never coloured with
// a score borrowed from elsewhere.
function fields(points, gx, gy, extentX, extentY, bw) {
  const score = new Float64Array(gx * gy)
  const density = new Float64Array(gx * gy)
  const scoredDensity = new Float64Array(gx * gy)
  for (let j = 0; j < gy; j++) {
    for (let i = 0; i < gx; i++) {
      const cx = extentX[0] + ((i + 0.5) / gx) * (extentX[1] - extentX[0])
      const cy = extentY[1] - ((j + 0.5) / gy) * (extentY[1] - extentY[0])
      let ws = 0, wv = 0, dens = 0
      for (const p of points) {
        const w = Math.exp(-((p[0] - cx) ** 2 + (p[1] - cy) ** 2) / (2 * bw * bw))
        dens += w
        if (p[2] !== null) { ws += w; wv += w * p[2] }
      }
      score[j * gx + i] = ws > 1e-12 ? wv / ws : 0
      density[j * gx + i] = dens
      scoredDensity[j * gx + i] = ws
    }
  }
  const inside = Array.from(score).filter((_, k) => scoredDensity[k] > 0.6).sort(d3.ascending)
  const levels = [...new Set(d3.range(9).map((k) => +d3.quantile(inside, 0.06 + k * 0.105).toFixed(5)))]
  return { score, density, scoredDensity, levels }
}
const atlasRamp = d3.scaleSequential(d3.interpolateRgbBasis(['#12304a', '#174a63', '#1d6475', '#2a8285', '#4ea499', '#8ccab3', '#dcebd0']))
let atlasId = 0
function atlasMap(map, w, h, { focus = 3, label = true, dense = false }) {
  if (!map) return ''
  const id = `atlas-${++atlasId}`
  const all = map.programs
  const has = (p) => p && p[0] !== null && p[1] !== null
  const pts = all.filter(has)
  if (!pts.length) return ''
  const ex = d3.extent(pts, (p) => p[0]), ey = d3.extent(pts, (p) => p[1])
  const padX = (ex[1] - ex[0]) * 0.18, padY = (ey[1] - ey[0]) * 0.2
  const fitX = [ex[0] - padX, ex[1] + padX], fitY = [ey[0] - padY, ey[1] + padY]
  const scale = Math.min(w / (fitX[1] - fitX[0]), h / (fitY[1] - fitY[0]))
  // The fields cover the whole canvas, so coastlines never meet a grid edge inside the view.
  const cx0 = (fitX[0] + fitX[1]) / 2, cy0 = (fitY[0] + fitY[1]) / 2
  const X = [cx0 - w / scale / 2, cx0 + w / scale / 2], Y = [cy0 - h / scale / 2, cy0 + h / scale / 2]
  const px = (v) => (v - X[0]) * scale
  const py = (v) => (Y[1] - v) * scale
  const gx = dense ? 220 : 150
  const gy = Math.max(8, Math.round(gx * (h / w)))
  const { score, density, scoredDensity, levels } = fields(pts, gx, gy, X, Y, 0.06 * Math.max(fitX[1] - fitX[0], fitY[1] - fitY[0]))
  const cellX = w / gx, cellY = h / gy
  const geo = d3.geoPath(d3.geoTransform({ point(x, y) { this.stream.point(x * cellX, y * cellY) } }))
  const explored = geo(d3.contours().size([gx, gy]).thresholds([0.6])(Array.from(density))[0])
  const coast = geo(d3.contours().size([gx, gy]).thresholds([0.6])(Array.from(scoredDensity))[0])
  const shelf = geo(d3.contours().size([gx, gy]).thresholds([0.25])(Array.from(density))[0])
  atlasRamp.domain([levels[0], levels.at(-1)])
  const bands = d3.contours().size([gx, gy]).thresholds(levels)(Array.from(score))
  const fills = bands.map((c, k) => `<path d="${geo(c)}" fill="${atlasRamp(c.value)}" stroke="#0b1622" stroke-opacity="${k % 3 === 2 ? 0.6 : 0.25}" stroke-width="${k % 3 === 2 ? 0.9 : 0.5}"/>`).join('')
  const best = has(all[map.best]) ? all[map.best] : null
  const chain = (start) => { const out = []; for (let i = start; i !== -1 && out.length < 200; i = all[i][5]) out.push(i); return out.reverse() }
  const routes = [...all.keys()].filter((i) => all[i][2] !== null && i !== map.best).sort((a, b) => all[b][2] - all[a][2]).slice(0, focus).map(chain)
  const line = (indices) => d3.line().curve(d3.curveCatmullRom.alpha(0.5))(indices.filter((i) => has(all[i])).map((i) => [px(all[i][0]), py(all[i][1])]))
  const trail = map.path.filter((i) => has(all[i])).map((i) => [px(all[i][0]), py(all[i][1])])
  const dots = pts.map((p) => (p[2] === null
    ? `<path d="M${(px(p[0]) - 2.2).toFixed(1)},${(py(p[1]) - 2.2).toFixed(1)}l4.4,4.4m0,-4.4l-4.4,4.4" stroke="#e2725b" stroke-width="1.1"/>`
    : `<circle cx="${px(p[0]).toFixed(1)}" cy="${py(p[1]).toFixed(1)}" r="${dense ? 1.6 : 1.2}" fill="#e7eef2" fill-opacity="0.6"/>`)).join('')
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true" style="display:block">
    <defs><clipPath id="${id}"><path d="${coast}"/></clipPath>
      <pattern id="${id}-sea" width="6" height="6" patternUnits="userSpaceOnUse"><path d="M0 6L6 0" stroke="#122a3b" stroke-width="0.6"/></pattern>
      <pattern id="${id}-failed" width="5" height="5" patternUnits="userSpaceOnUse"><path d="M0 5L5 0" stroke="#e2725b" stroke-opacity="0.55" stroke-width="0.8"/></pattern></defs>
    <rect width="${w}" height="${h}" fill="#0b1622"/>
    ${d3.range(0, w, 32).map((x) => `<line x1="${x}" x2="${x}" y1="0" y2="${h}" stroke="#14293a" stroke-width="0.6"/>`).join('')}
    ${d3.range(0, h, 32).map((y) => `<line x1="0" x2="${w}" y1="${y}" y2="${y}" stroke="#14293a" stroke-width="0.6"/>`).join('')}
    <path d="${shelf}" fill="url(#${id}-sea)" stroke="#1d3a4f" stroke-width="0.8" stroke-dasharray="1 3"/>
    <path d="${explored}" fill="url(#${id}-failed)" stroke="#e2725b" stroke-opacity="0.5" stroke-width="0.8" stroke-dasharray="2 2"/>
    <path d="${coast}" fill="${atlasRamp(levels[0])}"/>
    <g clip-path="url(#${id})">${fills}</g>
    <path d="${coast}" fill="none" stroke="#a9cbd6" stroke-opacity="0.7" stroke-width="1"/>
    ${routes.map((r) => `<path d="${line(r)}" fill="none" stroke="#cfe2e6" stroke-opacity="0.45" stroke-width="1" stroke-dasharray="3 3"/>`).join('')}
    ${dots}
    <path d="${d3.line().curve(d3.curveCatmullRom.alpha(0.5))(trail)}" fill="none" stroke="#0b1622" stroke-width="4.5" stroke-linecap="round" stroke-opacity="0.8"/>
    <path d="${d3.line().curve(d3.curveCatmullRom.alpha(0.5))(trail)}" fill="none" stroke="#e9b949" stroke-width="1.8" stroke-linecap="round"/>
    ${trail.slice(0, -1).map(([x, y]) => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2" fill="#0b1622" stroke="#e9b949" stroke-width="1.2"/>`).join('')}
    ${best ? `<path d="M${px(best[0])},${py(best[1]) - 7}l6,10.4h-12z" fill="#e9b949" stroke="#0b1622" stroke-width="1.2"/>` : ''}
    ${best && label ? `<g><line x1="${px(best[0]) + 8}" y1="${py(best[1]) - 4}" x2="${px(best[0]) + 30}" y2="${py(best[1]) - 22}" stroke="#e9b949" stroke-width="0.8"/>
      <text x="${px(best[0]) + 33}" y="${py(best[1]) - 24}" class="c-spot">实验最佳 ${num(best[2])}</text></g>` : ''}
  </svg>`
}

// ---------- tiny python highlighter ----------
const TOKEN = /\b(def|return|import|from|for|in|if|elif|else|while|class|lambda|None|True|False|and|or|not|with|as|try|except|raise|pass|yield)\b|\b(\d+(?:\.\d+)?)\b/g
// One pass over the escaped code so inserted markup is never re-tokenised.
function highlight(line, palette) {
  const commentAt = line.indexOf('#')
  const code = commentAt >= 0 ? line.slice(0, commentAt) : line
  const comment = commentAt >= 0 ? line.slice(commentAt) : ''
  const parts = code.split(/('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")/)
  const html = parts.map((part, k) => (k % 2
    ? `<span style="color:${palette.string}">${esc(part)}</span>`
    : esc(part).replace(TOKEN, (match, keyword, number) => (keyword
      ? `<span style="color:${palette.keyword};font-weight:500">${keyword}</span>`
      : `<span style="color:${palette.number}">${number}</span>`)))).join('')
  return html + (comment ? `<span style="color:${palette.comment}">${esc(comment)}</span>` : '')
}
const codeLines = A.best.code.filter((line, i, all) => i < 34 || line.trim()).slice(0, 34)

// ======================= A · 暗场仪器 =======================
const A_GLYPH = {
  supported: '<circle cx="7" cy="7" r="5.5" fill="currentColor"/>',
  confirmed: '<circle cx="7" cy="7" r="5.5" fill="currentColor"/><path d="M4.4 7.2l1.8 1.8 3.4-3.6" stroke="#0a0a0b" stroke-width="1.5" fill="none"/>',
  refuted: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M3.6 10.4l6.8-6.8" stroke="currentColor" stroke-width="1.4"/>',
  inconclusive: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M7 2a5 5 0 0 1 0 10z" fill="currentColor"/>',
  running: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.4" stroke-dasharray="2.2 1.9"/><circle cx="7" cy="7" r="1.8" fill="currentColor"/>',
  challenged: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M7 4v3.4" stroke="currentColor" stroke-width="1.5"/><circle cx="7" cy="9.6" r=".9" fill="currentColor"/>',
  proposed: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.3" stroke-dasharray="1 2.1"/>',
  selected: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.3"/>',
  unconfirmed: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M4.8 4.8l4.4 4.4m0-4.4l-4.4 4.4" stroke="currentColor" stroke-width="1.3"/>',
  dropped: '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="1.2" opacity=".5"/>',
}
const aGlyph = (status, cls = '') => `<svg class="a-glyph ${cls}" width="14" height="14" viewBox="0 0 14 14" role="img" aria-label="${esc(STATUS[status] ?? status)}">${A_GLYPH[status] ?? A_GLYPH.selected}</svg>`
const aStateGlyph = (d) => aGlyph({ active: 'running', concluded: 'confirmed', new: 'proposed', 'over-budget': 'challenged' }[d.state], d.state === 'over-budget' ? 'is-signal' : '')
const LOCK = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><rect x="2.5" y="6" width="9" height="6.5" rx="1.2" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M4.5 6V4.3a2.5 2.5 0 0 1 5 0V6" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>'
const UNLOCK = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><rect x="2.5" y="6" width="9" height="6.5" rx="1.2" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M4.5 6V4.3a2.5 2.5 0 0 1 4.9-.7" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>'
const MARK = '<svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><path d="M4 14.5V10m0 0c0-3 2.2-4 5-4m-5 4c0 2.6 2.4 4 5 4M9 6V3.5M9 6c2.8 0 5 1 5 4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><circle cx="9" cy="3.5" r="1.6" fill="currentColor"/><circle cx="14" cy="10" r="1.6" fill="currentColor"/><circle cx="9" cy="14" r="1.6" fill="currentColor"/></svg>'

function aGauge(d, width = 168) {
  if (d.best === null || d.bar === null) return `<div class="a-gauge is-empty"><span>尚无试验</span></div>`
  const x = d3.scaleLinear(evidenceDomain, [4, width - 4])
  const above = d.best > d.bar
  return `<div class="a-gauge"><svg width="${width}" height="22" viewBox="0 0 ${width} 22" aria-hidden="true">
      <line x1="0" x2="${width}" y1="11" y2="11" stroke="var(--line2)" stroke-width="1"/>
      ${x.ticks(8).map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="${evidenceMajors.includes(t) ? 7 : 9}" y2="${evidenceMajors.includes(t) ? 15 : 13}" stroke="var(--faint)" stroke-width="1"/>`).join('')}
      <line x1="${x(Math.min(d.bar, d.best))}" x2="${x(Math.max(d.bar, d.best))}" y1="11" y2="11" stroke="${above ? 'var(--signal)' : 'var(--muted)'}" stroke-width="2" ${above ? '' : 'stroke-dasharray="2 2"'}/>
      <rect x="${x(d.bar) - 1}" y="3" width="2" height="16" fill="var(--text)" opacity=".55"/>
      <path d="M${x(d.best)},5l5,6l-5,6l-5,-6z" fill="${above ? 'var(--signal)' : 'none'}" stroke="${above ? 'var(--signal)' : 'var(--text)'}" stroke-width="1.3"/>
    </svg><div class="a-gauge-read"><span>最佳 <b>${num(d.best)}</b></span><span>噪声 <b>${num(d.bar)}</b></span></div></div>`
}
function aBudget(d) {
  if (d.budget === null) return `<div class="a-budget"><span class="a-mono">${money(d.spend)}</span><span class="a-faint">未设预算</span></div>`
  const ratio = d.spend / d.budget
  const over = ratio > 1
  return `<div class="a-budget"><span class="a-mono">${money(d.spend)} <span class="a-faint">/ ${money(d.budget)}</span></span>
    <svg width="104" height="6" viewBox="0 0 104 6" aria-hidden="true"><rect width="104" height="2" y="2" fill="var(--line2)"/>
    <rect width="${Math.min(1, ratio) * 104}" height="2" y="2" fill="${over ? 'var(--signal)' : 'var(--text)'}" opacity="${over ? 1 : 0.7}"/>
    ${over ? '<rect x="101" y="0" width="3" height="6" fill="var(--signal)"/>' : ''}</svg>
    <span class="${over ? 'a-signal' : 'a-faint'} a-mono">${Math.round(ratio * 100)}%</span></div>`
}
const aHyps = (d, labelled = false) => `<div class="a-hyps">${statusCounts(d).map(([s, n]) => `<span>${aGlyph(s)}${labelled ? `<em>${esc(STATUS[s])}</em>` : ''}<b>${n}</b></span>`).join('')}</div>`
const aGlyphKey = () => {
  const present = [...new Set(dirs.flatMap((d) => statusCounts(d).map(([s]) => s)))]
  return `<div class="a-glyphkey"><span class="a-faint">假设状态</span>${present.map((s) => `<span>${aGlyph(s)}${esc(STATUS[s])}</span>`).join('')}<span class="a-faint">证据刻度各行相同，长刻线为 ${evidenceMajors.map((t) => num(t)).join(' 与 ')}</span></div>`
}
const aLock = (d) => (d.lockbox === 'sealed' ? `<span class="a-lock" title="封存段未开">${LOCK}<span>封存</span></span>` : d.lockbox === 'opened' ? `<span class="a-lock is-open" title="封存段已开">${UNLOCK}<span>已开</span></span>` : '<span class="a-lock a-faint">—</span>')
const aLockLabel = (d) => `<span class="a-lock ${d.lockbox === 'opened' ? 'is-open' : ''}">${d.lockbox === 'sealed' ? `${LOCK}<span>封存段未开</span>` : d.lockbox === 'opened' ? `${UNLOCK}<span>封存段已开</span>` : '<span>封存段 —</span>'}</span>`

function aRail(active, collapsed = false) {
  const nav = [['研究方向', dirs.length], ['全部运行', fixture.workspace.runs], ['资料库', fixture.workspace.libraryPages]]
  if (collapsed) {
    return `<aside class="a-rail is-collapsed"><div class="a-brand">${MARK}</div>
      ${nav.map(([label], k) => `<div class="a-navicon ${label === active ? 'is-active' : ''}" title="${label}">${['◆', '▤', '❐'][k]}</div>`).join('')}</aside>`
  }
  return `<aside class="a-rail"><div class="a-brand">${MARK}<span>Shinka Research</span><span class="a-chev">⌄</span></div>
    <div class="a-search"><span>搜索方向、程序…</span><kbd>Ctrl K</kbd></div>
    <nav>${nav.map(([label, n]) => `<div class="a-nav ${label === active ? 'is-active' : ''}"><span>${label}</span><span class="a-mono a-faint">${n}</span></div>`).join('')}</nav>
    <div class="a-railhead">方向</div>
    ${dirs.map((d) => `<div class="a-railitem">${aStateGlyph(d)}<span>${esc(shortTitle(d))}</span></div>`).join('')}
    <div class="a-railfoot"><span class="a-demo">DEMO</span><span>演示数据，非真实研究</span></div></aside>`
}
const aEventGlyph = (event) => (event === 'run-failed' ? '<span class="a-ev is-signal">✕</span>' : event === 'verdict' ? '<span class="a-ev">◆</span>' : event === 'skeptic' ? '<span class="a-ev">◇</span>' : event === 'confirm' || event === 'lockbox' ? `<span class="a-ev">${UNLOCK}</span>` : '<span class="a-ev is-dim">●</span>')

function aHome() {
  return `<div class="a-app">${aRail('研究方向')}
  <main class="a-main">
    <header class="a-top"><div class="a-crumb"><span class="a-faint">工作区</span><span class="a-faint">/</span><span>研究方向</span></div>
      <div class="a-tools"><span class="a-btn">排序 · 最近活动</span><span class="a-btn">全部状态</span></div></header>
    <section class="a-content">
      <div class="a-titlerow"><h1>研究方向</h1><p class="a-sub"><span class="a-mono">${dirs.length}</span> 个方向 · <span class="a-mono">${int(total.trials)}</span> 次试验 · 花费 <span class="a-mono">${money(total.spend)}</span> / 预算 <span class="a-mono">${money(total.budget)}</span> · 时间为 UTC</p></div>
      <div class="a-kpis">
        <div><span>高于噪声线</span><b class="a-mono">${total.above}<i>/${total.withTrials}</i></b><em>有试验的方向</em></div>
        <div><span>待判定假设</span><b class="a-mono">${total.challenged}</b><em>质疑后尚无判词</em></div>
        <div><span>评估失败</span><b class="a-mono">${(failRate * 100).toFixed(1)}<i>%</i></b><em>${int(total.failed)} 次失败试验</em></div>
        <div class="is-signal"><span>预算告警</span><b class="a-mono">${total.overBudget}</b><em>超出预算的方向</em></div>
      </div>
      <div class="a-table" role="table">
        <div class="a-thead" role="row"><span>方向</span><span>假设</span><span>证据 · 最佳 vs 噪声线</span><span>试验</span><span>预算</span><span>封存段</span></div>
        ${dirs.map((d, k) => `<div class="a-row ${k === 0 ? 'is-focus' : ''}" role="row">
          <div class="a-cell-title">${aStateGlyph(d)}<div><b>${esc(shortTitle(d))}</b><span class="a-mono a-faint">${esc(d.id)}</span><span class="a-meta"><span class="a-demo">DEMO</span>${STATE[d.state]} · ${clock(d.lastActivity)}</span></div></div>
          ${aHyps(d)}
          ${aGauge(d)}
          <div class="a-trials"><b class="a-mono">${int(d.trials)}</b>${trialChart(d, { w: 64, h: 22, m: { t: 3, r: 2, b: 3, l: 2 }, ink: 'var(--text)', muted: 'var(--faint)', signal: 'var(--signal)' })}</div>
          ${aBudget(d)}
          ${aLock(d)}
        </div>`).join('')}
      </div>
      ${aGlyphKey()}
    </section>
  </main>
  <aside class="a-feed"><header><b>动态</b><span class="a-faint">全部方向 · UTC</span></header>
    ${fixture.events.slice(0, 9).map((e) => `<div class="a-event">${aEventGlyph(e.event)}<div><div class="a-evtitle"><b>${esc(eventLabel(e.event))}</b><span class="a-mono a-faint">${esc(e.direction.split('-')[0])} · ${esc(e.hypothesis)}${e.experiment ? '/' + esc(e.experiment) : ''}</span></div><time class="a-mono">${clock(e.time)}</time><p>${esc(e.note || '—')}</p></div></div>`).join('')}
  </aside></div>`
}

function aAnalysis() {
  const root = hierarchy([2 * Math.PI, 1])
  const R0 = 26, R1 = 262
  const radius = (node) => (node.data.row ? R0 + (node.data.row[3] / maxGen) * (R1 - R0) : 0)
  const point = (node) => [Math.sin(node.x) * radius(node), -Math.cos(node.x) * radius(node)]
  const shade = d3.scaleLinear(scoreExtent, ['#3b3b43', '#f1f1f3']).clamp(true)
  const link = d3.linkRadial().angle((n) => n.x).radius(radius)
  const links = root.links().filter((l) => l.source.data.row)
  const best = root.descendants().find((n) => n.data.i === A.best.index)
  const islands = d3.groups(root.children, (n) => n.data.row[4]).map(([island, kids]) => {
    const angles = kids.flatMap((k) => k.leaves().map((l) => l.x))
    return { island, a0: d3.min(angles) - 0.02, a1: d3.max(angles) + 0.02 }
  })
  const arc = d3.arc().innerRadius(R1 + 14).outerRadius(R1 + 16)
  const cx = 470, cy = 330
  const prog = A.progress
  const px = d3.scaleLinear([0, maxGen], [56, 900])
  const py = d3.scaleLinear(d3.extent(prog.flatMap((p) => [p.cumulative, p.median]).filter((v) => v !== null)), [118, 22]).nice()
  const cum = d3.line().curve(d3.curveStepAfter).defined((p) => p.cumulative !== null).x((p) => px(p.generation)).y((p) => py(p.cumulative))(prog)
  const med = d3.line().curve(d3.curveMonotoneX).defined((p) => p.median !== null).x((p) => px(p.generation)).y((p) => py(p.median))(prog)
  const lift = A.best.parentScore === null ? null : A.best.score - A.best.parentScore
  return `<div class="a-app">${aRail('全部运行', true)}
  <main class="a-main a-analysis">
    <header class="a-top"><div class="a-crumb"><span class="a-faint">${esc(shortTitle(dirs[0]))}</span><span class="a-faint">/</span><span class="a-mono">h001 · e01</span></div>
      <div class="a-seg" role="tablist"><span class="is-on">谱系</span><span>进展</span><span>解空间</span><span>对比</span></div>
      <div class="a-tools"><span class="a-mono a-faint">${A.stats.programs} 程序 · ${A.stats.generations} 代 · ${A.stats.islands} 岛 · ${A.stats.failed} 失败</span></div></header>
    <section class="a-canvas">
      <svg class="a-tree" width="944" height="660" viewBox="0 0 944 660" role="img" aria-label="进化谱系，${A.stats.programs} 个程序，橙色为最佳程序的祖先链">
        <g transform="translate(${cx},${cy})">
          ${[0.25, 0.5, 0.75, 1].map((f) => `<circle r="${R0 + f * (R1 - R0)}" fill="none" stroke="var(--line)" stroke-width="1"/>
            <text x="4" y="${-(R0 + f * (R1 - R0)) - 4}" class="a-axis">第 ${Math.round(f * maxGen)} 代</text>`).join('')}
          ${islands.map((s) => `<path d="${arc({ startAngle: s.a0, endAngle: s.a1 })}" fill="var(--muted)"/>
            <text transform="rotate(${((s.a0 + s.a1) / 2) * 180 / Math.PI}) translate(0,${-(R1 + 26)})" text-anchor="middle" class="a-axis">岛 ${s.island}</text>`).join('')}
          <g fill="none" stroke="#33333a" stroke-width="0.8">${links.map((l) => `<path d="${link(l)}"/>`).join('')}</g>
          <g fill="none" stroke="var(--signal)" stroke-width="1.8">${links.filter((l) => pathSet.has(l.target.data.i) && pathSet.has(l.source.data.i)).map((l) => `<path d="${link(l)}"/>`).join('')}</g>
          ${root.descendants().filter((n) => n.data.row).map((n) => {
            const [x, y] = point(n)
            const r = n.data.row
            if (r[2] === null) return `<path d="M${(x - 2).toFixed(1)},${(y - 2).toFixed(1)}l4,4m0,-4l-4,4" stroke="var(--muted)" stroke-width="1"/>`
            return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${pathSet.has(n.data.i) ? 2.6 : 1.9}" fill="${pathSet.has(n.data.i) ? 'var(--signal)' : shade(r[2])}"/>`
          }).join('')}
          ${(() => { const [x, y] = point(best); return `<circle cx="${x}" cy="${y}" r="7" fill="none" stroke="var(--signal)" stroke-width="1.4"/><circle cx="${x}" cy="${y}" r="11" fill="none" stroke="var(--signal)" stroke-opacity=".35"/>` })()}
        </g>
        <g transform="translate(36,36)" class="a-legend"><text y="0" class="a-legtitle">得分</text>
          <defs><linearGradient id="a-ramp"><stop offset="0" stop-color="#3b3b43"/><stop offset="1" stop-color="#f1f1f3"/></linearGradient></defs>
          <rect y="10" width="120" height="6" fill="url(#a-ramp)"/><text y="32" class="a-axis">${num(scoreExtent[0])}</text><text x="120" y="32" text-anchor="end" class="a-axis">${num(scoreExtent[1])}</text>
          <line x1="0" x2="18" y1="52" y2="52" stroke="var(--signal)" stroke-width="2"/><text x="26" y="56" class="a-axis">最佳程序的祖先链</text>
          <path d="M2,68l6,6m0,-6l-6,6" stroke="var(--muted)"/><text x="26" y="75" class="a-axis">评估失败（不计入最佳）</text></g>
      </svg>
      <div class="a-strip"><div class="a-striphead"><b>进展</b><span class="a-faint">每代累计最佳 · 中位</span><span class="a-key"><i class="is-ink"></i>累计最佳<i class="is-dash"></i>中位</span></div>
        <svg width="944" height="132" viewBox="0 0 944 132" aria-hidden="true">
          ${py.ticks(3).map((t) => `<line x1="56" x2="900" y1="${py(t)}" y2="${py(t)}" stroke="var(--line)"/><text x="48" y="${py(t) + 4}" text-anchor="end" class="a-axis">${num(t)}</text>`).join('')}
          <path d="${med}" fill="none" stroke="var(--faint)" stroke-dasharray="3 3" stroke-width="1.3"/>
          <path d="${cum}" fill="none" stroke="var(--text)" stroke-width="1.6"/>
          <line x1="${px(A.best.generation)}" x2="${px(A.best.generation)}" y1="16" y2="118" stroke="var(--signal)" stroke-width="1"/>
          ${px.ticks(6).map((t) => `<text x="${px(t)}" y="130" text-anchor="middle" class="a-axis">${t}</text>`).join('')}
        </svg></div>
    </section>
  </main>
  <aside class="a-inspector">
    <div class="a-insphead"><span class="a-mono a-faint">第 ${A.best.generation} 代 · 岛 ${A.best.island} · ${esc(A.best.patch ?? '—')}</span>
      <h2>${esc(A.best.name)}</h2>
      <div class="a-score"><b class="a-mono">${num(A.best.score)}</b>${lift === null ? '' : `<span class="a-mono a-signal">+${num(lift)} 相对父程序</span>`}</div>
      <div class="a-chips"><span>${esc(A.best.model ?? '—')}</span><span class="a-mono">${A.best.cost === null ? '费用未知' : money(A.best.cost)}</span><span class="a-mono">${esc(A.best.id.slice(0, 8))}</span></div></div>
    <div class="a-tabs" role="tablist"><span class="is-on">代码</span><span>差异</span><span>评估</span><span>LLM</span><span>草稿</span></div>
    <pre class="a-code">${codeLines.map((line, i) => `<span class="ln">${i + 1}</span>${highlight(line, { keyword: '#ffffff', string: '#a8a8b2', number: '#d6d6dc', comment: '#5e5e67' })}`).join('\n')}</pre>
  </aside></div>`
}

function aMobile() {
  return `<div class="a-phone"><header class="a-ptop">${MARK}<b>研究方向</b><span class="a-faint">⌕</span></header>
    <div class="a-pscroll"><div class="a-psum"><h1>研究方向</h1><p><span class="a-mono">${dirs.length}</span> 个方向 · <span class="a-mono">${int(total.trials)}</span> 次试验 · <span class="a-signal">${total.overBudget} 个超预算</span></p><p>证据刻度各方向相同</p></div>
    ${dirs.map((d) => `<div class="a-pcard">
      <div class="a-cell-title">${aStateGlyph(d)}<div><b>${esc(shortTitle(d))}</b><span class="a-meta"><span class="a-demo">DEMO</span>${STATE[d.state]} · ${int(d.trials)} 次试验</span></div></div>
      ${aHyps(d, true)}${aGauge(d, 300)}<div class="a-prow">${aBudget(d)}${aLockLabel(d)}</div></div>`).join('')}</div>
    <nav class="a-ptabs"><span class="is-on">方向</span><span>运行</span><span>资料库</span></nav></div>`
}

const A_CSS = `
[data-direction-id="dark-instrument"]{--bg:#0a0a0b;--rail:#0e0e10;--raised:#16161a;--hover:#1c1c21;--line:rgba(255,255,255,.07);--line2:rgba(255,255,255,.13);--text:#ededef;--muted:#8e8e96;--faint:#5e5e67;--signal:#f2622e;}
[data-direction-id="dark-instrument"] .frame{background:var(--bg);color:var(--text);font:450 13px/20px InterV,"PingFang SC","Microsoft YaHei","Noto Sans SC",sans-serif;font-feature-settings:"cv11","ss01";}
[data-direction-id="dark-instrument"] .a-mono{font-family:PlexMono,"JetBrains Mono",Consolas,monospace;font-variant-numeric:tabular-nums;letter-spacing:0}
[data-direction-id="dark-instrument"] .a-faint{color:var(--faint)} [data-direction-id="dark-instrument"] .a-signal{color:var(--signal)}
[data-direction-id="dark-instrument"] .a-app{display:flex;height:100%}
[data-direction-id="dark-instrument"] .a-rail{width:232px;flex:none;background:var(--rail);border-right:1px solid var(--line);padding:12px 10px;display:flex;flex-direction:column;gap:2px}
[data-direction-id="dark-instrument"] .a-rail.is-collapsed{width:56px;align-items:center;gap:6px}
[data-direction-id="dark-instrument"] .a-brand{display:flex;align-items:center;gap:9px;height:32px;padding:0 8px;font-weight:590;font-size:14px;color:var(--text)}
[data-direction-id="dark-instrument"] .a-chev{color:var(--faint);margin-left:auto}
[data-direction-id="dark-instrument"] .a-search{display:flex;justify-content:space-between;align-items:center;height:30px;margin:8px 0 10px;padding:0 10px;border:1px solid var(--line2);border-radius:6px;color:var(--faint);font-size:12.5px}
[data-direction-id="dark-instrument"] kbd{font:500 11px PlexMono,monospace;color:var(--muted);border:1px solid var(--line2);border-radius:4px;padding:0 4px}
[data-direction-id="dark-instrument"] .a-nav{display:flex;justify-content:space-between;height:30px;align-items:center;padding:0 10px;border-radius:6px;color:var(--muted)}
[data-direction-id="dark-instrument"] .a-nav.is-active{background:var(--hover);color:var(--text)}
[data-direction-id="dark-instrument"] .a-navicon{width:34px;height:34px;display:grid;place-items:center;border-radius:6px;color:var(--faint)}
[data-direction-id="dark-instrument"] .a-navicon.is-active{background:var(--hover);color:var(--text)}
[data-direction-id="dark-instrument"] .a-railhead{margin:18px 10px 6px;font-size:12px;color:var(--faint)}
[data-direction-id="dark-instrument"] .a-railitem{display:flex;gap:9px;align-items:center;height:30px;padding:0 10px;color:var(--muted);white-space:nowrap;overflow:hidden}
[data-direction-id="dark-instrument"] .a-railitem span{overflow:hidden;text-overflow:ellipsis}
[data-direction-id="dark-instrument"] .a-railfoot{margin-top:auto;display:flex;gap:8px;align-items:center;padding:10px;color:var(--faint);font-size:12px}
[data-direction-id="dark-instrument"] .a-demo{font:500 10px/16px PlexMono,monospace;letter-spacing:.04em;color:var(--muted);border:1px solid var(--line2);border-radius:3px;padding:0 4px}
[data-direction-id="dark-instrument"] .a-glyph{flex:none;color:var(--text)} [data-direction-id="dark-instrument"] .a-glyph.is-signal{color:var(--signal)}
[data-direction-id="dark-instrument"] .a-main{flex:1;min-width:0;display:flex;flex-direction:column}
[data-direction-id="dark-instrument"] .a-top{height:52px;flex:none;display:flex;align-items:center;gap:20px;padding:0 24px;border-bottom:1px solid var(--line)}
[data-direction-id="dark-instrument"] .a-crumb{display:flex;gap:8px;align-items:center;white-space:nowrap;min-width:0;max-width:360px;overflow:hidden}
[data-direction-id="dark-instrument"] .a-crumb span{overflow:hidden;text-overflow:ellipsis}
[data-direction-id="dark-instrument"] .a-tools{margin-left:auto;display:flex;gap:8px;align-items:center;white-space:nowrap;font-size:12px}
[data-direction-id="dark-instrument"] .a-btn{height:28px;padding:0 10px;display:inline-flex;align-items:center;border:1px solid var(--line2);border-radius:6px;color:var(--muted);font-size:12.5px}
[data-direction-id="dark-instrument"] .a-content{padding:28px 32px;overflow:hidden}
[data-direction-id="dark-instrument"] .a-titlerow{display:flex;align-items:baseline;gap:16px}
[data-direction-id="dark-instrument"] h1{margin:0;font-size:26px;line-height:34px;font-weight:590;letter-spacing:-.01em}
[data-direction-id="dark-instrument"] .a-sub{margin:0;color:var(--muted)}
[data-direction-id="dark-instrument"] .a-kpis{display:grid;grid-template-columns:repeat(4,1fr);margin:26px 0 28px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
[data-direction-id="dark-instrument"] .a-kpis div{padding:14px 18px 16px;display:grid;gap:4px;border-left:1px solid var(--line)}
[data-direction-id="dark-instrument"] .a-kpis div:first-child{border-left:0;padding-left:0}
[data-direction-id="dark-instrument"] .a-kpis span{color:var(--muted);font-size:12.5px}
[data-direction-id="dark-instrument"] .a-kpis b{font-size:24px;line-height:30px;font-weight:500}
[data-direction-id="dark-instrument"] .a-kpis i{font-style:normal;color:var(--faint);font-size:15px}
[data-direction-id="dark-instrument"] .a-kpis em{font-style:normal;color:var(--faint);font-size:12px}
[data-direction-id="dark-instrument"] .a-kpis .is-signal b{color:var(--signal)}
[data-direction-id="dark-instrument"] .a-thead,[data-direction-id="dark-instrument"] .a-row{display:grid;grid-template-columns:224px 100px 168px 88px 124px 1fr;align-items:center;column-gap:14px;padding-left:12px}
[data-direction-id="dark-instrument"] .a-thead{height:34px;color:var(--faint);font-size:12px;border-bottom:1px solid var(--line);white-space:nowrap}
[data-direction-id="dark-instrument"] .a-row{min-height:84px;border-bottom:1px solid var(--line);position:relative}
[data-direction-id="dark-instrument"] .a-row.is-focus{background:rgba(255,255,255,.035);box-shadow:inset 2px 0 0 var(--signal)}
[data-direction-id="dark-instrument"] .a-cell-title{display:flex;gap:10px;align-items:flex-start}
[data-direction-id="dark-instrument"] .a-cell-title .a-glyph{margin-top:3px}
[data-direction-id="dark-instrument"] .a-cell-title div{display:grid;gap:1px;min-width:0}
[data-direction-id="dark-instrument"] .a-cell-title b{font-weight:510;font-size:13.5px;line-height:20px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;line-break:strict;word-break:normal}
[data-direction-id="dark-instrument"] .a-cell-title .a-mono{font-size:11.5px}
[data-direction-id="dark-instrument"] .a-meta{display:flex;gap:6px;align-items:center;color:var(--faint);font-size:12px;margin-top:2px}
[data-direction-id="dark-instrument"] .a-hyps{display:flex;flex-wrap:wrap;gap:6px 10px}
[data-direction-id="dark-instrument"] .a-hyps span{display:inline-flex;gap:4px;align-items:center;color:var(--muted)} [data-direction-id="dark-instrument"] .a-hyps b{font:500 12px PlexMono,monospace;color:var(--text)} [data-direction-id="dark-instrument"] .a-hyps em{font-style:normal;font-size:12px}
[data-direction-id="dark-instrument"] .a-glyphkey{display:flex;gap:16px;align-items:center;padding:12px 0 0 12px;font-size:12px;color:var(--muted)} [data-direction-id="dark-instrument"] .a-glyphkey span{display:inline-flex;gap:5px;align-items:center}
[data-direction-id="dark-instrument"] .a-gauge{display:grid;gap:3px} [data-direction-id="dark-instrument"] .a-gauge.is-empty{color:var(--faint);font-size:12px}
[data-direction-id="dark-instrument"] .a-gauge-read{display:flex;gap:12px;font-size:11.5px;color:var(--faint)} [data-direction-id="dark-instrument"] .a-gauge-read b{font:500 12px PlexMono,monospace;color:var(--text)}
[data-direction-id="dark-instrument"] .a-trials{display:grid;gap:4px} [data-direction-id="dark-instrument"] .a-trials b{font-weight:500;font-size:13px}
[data-direction-id="dark-instrument"] .a-budget{display:grid;gap:4px;font-size:12px}
[data-direction-id="dark-instrument"] .a-lock{display:inline-flex;gap:6px;align-items:center;color:var(--muted);font-size:12px} [data-direction-id="dark-instrument"] .a-lock.is-open{color:var(--text)}
[data-direction-id="dark-instrument"] .a-feed{width:300px;flex:none;border-left:1px solid var(--line);background:var(--rail);padding:0 16px;overflow:hidden}
[data-direction-id="dark-instrument"] .a-feed header{height:52px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);margin:0 -16px 6px;padding:0 16px}
[data-direction-id="dark-instrument"] .a-event{display:grid;grid-template-columns:14px 1fr;gap:10px;padding:11px 0;border-bottom:1px solid var(--line)}
[data-direction-id="dark-instrument"] .a-ev{color:var(--muted);font-size:11px;line-height:20px} [data-direction-id="dark-instrument"] .a-ev.is-signal{color:var(--signal)} [data-direction-id="dark-instrument"] .a-ev.is-dim{color:var(--faint);font-size:8px}
[data-direction-id="dark-instrument"] .a-evtitle{display:flex;gap:8px;align-items:baseline;white-space:nowrap} [data-direction-id="dark-instrument"] .a-evtitle b{font-weight:510} [data-direction-id="dark-instrument"] .a-evtitle time{margin-left:auto}
[data-direction-id="dark-instrument"] .a-event p{margin:2px 0 0;color:var(--muted);font-size:12px;line-height:18px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
[data-direction-id="dark-instrument"] .a-event time{display:block;font-size:11px;color:var(--faint);line-height:18px}
[data-direction-id="dark-instrument"] .a-seg{display:flex;gap:2px;padding:2px;border:1px solid var(--line2);border-radius:7px}
[data-direction-id="dark-instrument"] .a-seg span{padding:3px 12px;border-radius:5px;color:var(--muted);font-size:12.5px;white-space:nowrap} [data-direction-id="dark-instrument"] .a-seg .is-on{background:var(--hover);color:var(--text)}
[data-direction-id="dark-instrument"] .a-canvas{flex:1;position:relative;overflow:hidden}
[data-direction-id="dark-instrument"] .a-axis{font:400 10.5px PlexMono,monospace;fill:var(--faint)} [data-direction-id="dark-instrument"] .a-legtitle{font-size:12px;fill:var(--muted)}
[data-direction-id="dark-instrument"] .a-strip{position:absolute;left:0;right:0;bottom:0;border-top:1px solid var(--line);background:var(--rail)}
[data-direction-id="dark-instrument"] .a-striphead{display:flex;gap:12px;align-items:center;padding:12px 24px 0}
[data-direction-id="dark-instrument"] .a-key{margin-left:auto;display:flex;gap:8px;align-items:center;color:var(--faint);font-size:12px}
[data-direction-id="dark-instrument"] .a-key i{display:inline-block;width:16px;height:0;border-top:1.6px solid var(--text)} [data-direction-id="dark-instrument"] .a-key i.is-dash{border-top:1.4px dashed var(--faint)}
[data-direction-id="dark-instrument"] .a-inspector{width:440px;flex:none;border-left:1px solid var(--line);background:var(--rail);display:flex;flex-direction:column;min-height:0}
[data-direction-id="dark-instrument"] .a-insphead{padding:18px 20px 14px;display:grid;gap:6px;border-bottom:1px solid var(--line)}
[data-direction-id="dark-instrument"] .a-insphead h2{margin:0;font-size:16px;line-height:22px;font-weight:590}
[data-direction-id="dark-instrument"] .a-score{display:flex;gap:12px;align-items:baseline} [data-direction-id="dark-instrument"] .a-score b{font-size:26px;font-weight:500} [data-direction-id="dark-instrument"] .a-score span{font-size:12px}
[data-direction-id="dark-instrument"] .a-chips{display:flex;gap:6px;flex-wrap:wrap} [data-direction-id="dark-instrument"] .a-chips span{border:1px solid var(--line2);border-radius:5px;padding:1px 7px;color:var(--muted);font-size:12px}
[data-direction-id="dark-instrument"] .a-tabs{display:flex;gap:18px;padding:0 20px;height:40px;align-items:stretch;border-bottom:1px solid var(--line)}
[data-direction-id="dark-instrument"] .a-tabs span{display:flex;align-items:center;color:var(--muted)} [data-direction-id="dark-instrument"] .a-tabs .is-on{color:var(--text);box-shadow:inset 0 -2px 0 var(--signal)}
[data-direction-id="dark-instrument"] .a-code{margin:0;padding:14px 16px;font:400 12px/20px PlexMono,"JetBrains Mono",monospace;color:#c8c8cf;white-space:pre;overflow:hidden;flex:1}
[data-direction-id="dark-instrument"] .a-code .ln{display:inline-block;width:26px;margin-right:14px;text-align:right;color:#45454d}
[data-direction-id="dark-instrument"] .a-phone{height:100%;display:flex;flex-direction:column;background:var(--bg)} [data-direction-id="dark-instrument"] .a-ptop,[data-direction-id="dark-instrument"] .a-ptabs{flex:none} [data-direction-id="dark-instrument"] .a-pscroll{flex:1;min-height:0;overflow-y:auto;scrollbar-width:none}
[data-direction-id="dark-instrument"] .a-ptop{height:52px;display:flex;align-items:center;gap:10px;padding:0 16px;border-bottom:1px solid var(--line)} [data-direction-id="dark-instrument"] .a-ptop span{margin-left:auto;font-size:18px}
[data-direction-id="dark-instrument"] .a-psum{padding:18px 16px 8px} [data-direction-id="dark-instrument"] .a-psum h1{font-size:22px;line-height:30px} [data-direction-id="dark-instrument"] .a-psum p{margin:4px 0 0;color:var(--muted)}
[data-direction-id="dark-instrument"] .a-pcard{margin:0 16px;padding:14px 0;border-bottom:1px solid var(--line);display:grid;gap:10px} [data-direction-id="dark-instrument"] .a-prow{display:flex;justify-content:space-between;align-items:flex-end;gap:12px}
[data-direction-id="dark-instrument"] .a-ptabs{margin-top:auto;height:72px;padding-bottom:16px;display:flex;justify-content:space-around;align-items:center;border-top:1px solid var(--line);background:var(--rail);color:var(--muted)} [data-direction-id="dark-instrument"] .a-ptabs .is-on{color:var(--text);font-weight:510}
`

// ======================= B · 研究报告 =======================
const B_STATUS_COLOR = { supported: '#2e9e5b', confirmed: '#17803d', refuted: '#e4573d', inconclusive: '#c88d00', running: '#3b6fd8', challenged: '#8e5be8', proposed: '#9a9aa2', selected: '#9a9aa2', unconfirmed: '#b3471f', dropped: '#b4b4bb' }
const bChip = (status, n) => `<span class="b-chip"><i style="background:${B_STATUS_COLOR[status]}"></i>${STATUS[status]}${n > 1 ? ` ${n}` : ''}</span>`
// Shared scales for the small multiples, so heights and trial counts compare across directions.
const facetX = [1, d3.max(dirs, (d) => d.trials)]
const facetY = (() => {
  const values = dirs.flatMap((d) => { const { steps, bars } = recordSeries(d); return [...steps, ...bars].map((p) => p[1]) }).filter((v) => v !== null)
  const [lo, hi] = d3.extent(values)
  const span = (hi - lo) || 1
  return [lo - span * 0.08, hi + span * 0.14]
})()
function bHeadline() {
  const confirmed = dirs.filter((d) => d.hypotheses.some((h) => h.status === 'confirmed')).length
  const phrase = (cls, text) => `<em class="${cls}">${text}</em>`
  return `${dirs.length} 个研究方向里，${phrase('b-ok', `${confirmed} 个已通过封存段确认`)}，${phrase('b-warn', `${total.overBudget} 个超出预算`)}，<span class="b-keep">${total.running} 个假设仍在运行</span>。`
}
function bFacet(d, w = 276) {
  const chart = d.trials ? trialChart(d, { w, h: 148, m: { t: 14, r: 6, b: 22, l: 30 }, ink: '#1b1b1f', muted: '#8a8a94', signal: '#0e8a7a', grid: '#efeff1', axis: true, labels: true, font: 'PlexMono,monospace', xDomain: facetX, yDomain: facetY })
    : `<div class="b-empty">尚无试验<br><span>${d.hypotheses.length} 个假设已提出，等待对话发起实验</span></div>`
  return `<article class="b-facet"><h3>${esc(shortTitle(d))}</h3>
    <div class="b-facetmeta"><span class="b-mono">${esc(d.id)}</span><span class="b-tag">演示</span><span class="b-state b-state-${d.state}">${STATE[d.state]}</span></div>
    <div class="b-chips">${statusCounts(d).map(([s, n]) => bChip(s, n)).join('')}</div>
    <figure>${chart}</figure>
    <dl class="b-stats"><div><dt>试验</dt><dd>${int(d.trials)}</dd></div><div><dt>花费</dt><dd class="${d.state === 'over-budget' ? 'b-over' : ''}">${money(d.spend)}<span> / ${money(d.budget)}</span></dd></div><div><dt>封存段</dt><dd>${d.lockbox === 'sealed' ? '未开' : d.lockbox === 'opened' ? '已开' : '—'}</dd></div></dl></article>`
}
function bTimeline(width) {
  const times = fixture.events.map((e) => new Date(e.time))
  const x = d3.scaleUtc(d3.extent(times), [150, width - 30]).nice()
  const lanes = dirs.map((d) => d.id)
  const color = { verdict: '#1b1b1f', skeptic: '#8e5be8', run: '#3b6fd8', 'run-failed': '#e4573d', scaffold: '#9a9aa2', confirm: '#17803d', lockbox: '#17803d', probe: '#c88d00' }
  return `<svg width="${width}" height="${lanes.length * 26 + 28}" viewBox="0 0 ${width} ${lanes.length * 26 + 28}" role="img" aria-label="各方向最近活动时间线" style="display:block;font-family:PlexMono,monospace">
    ${lanes.map((id, k) => `<line x1="150" x2="${width - 10}" y1="${k * 26 + 13}" y2="${k * 26 + 13}" stroke="#efeff1"/><text x="0" y="${k * 26 + 17}" font-size="11.5" fill="#6a6a73">${esc(id)}</text>`).join('')}
    ${fixture.events.map((e) => `<circle cx="${x(new Date(e.time)).toFixed(1)}" cy="${lanes.indexOf(e.direction) * 26 + 13}" r="${e.event === 'verdict' ? 5 : 3.6}" fill="${color[e.event] ?? '#9a9aa2'}" fill-opacity=".9" stroke="#fff" stroke-width="1.2"/>`).join('')}
    ${x.ticks(d3.utcDay.every(4)).map((t) => `<text x="${x(t)}" y="${lanes.length * 26 + 22}" font-size="11" text-anchor="${x(t) > width - 70 ? 'end' : 'middle'}" fill="#8a8a94">${day(t.toISOString())}</text>`).join('')}
  </svg>`
}
function bHome() {
  return `<div class="b-page"><header class="b-top"><b class="b-word">Shinka Research</b><nav><span class="is-on">研究方向</span><span>全部运行</span><span>资料库</span></nav><span class="b-search">搜索方向、程序… <kbd>Ctrl K</kbd></span></header>
  <main class="b-wrap"><p class="b-kicker">工作区 · ${day(fixture.events[0]?.time ?? dirs[0].created)} · DEMO 演示数据，非真实研究</p>
    <h1 class="b-headline">${bHeadline()}</h1>
    <p class="b-lede">累计 <b>${int(total.trials)}</b> 次试验，花费 <b>${money(total.spend)}</b> / 预算 <b>${money(total.budget)}</b>。“${esc(shortTitle(dirs[0]))}”的最佳样本内记录 <b>${num(dirs[0].best)}</b>${dirs[0].best > dirs[0].bar ? '高于' : '低于'}噪声线 <b>${num(dirs[0].bar)}</b>。</p>
    <section class="b-facets">${dirs.map((d) => bFacet(d)).join('')}</section>
    <section class="b-activity"><h2>活动时间线 <span>每个点是一次记录在案的研究事件，大点为判词 · 时间为 UTC</span></h2>${bTimeline(1200)}</section></main></div>`
}
function bAnalysis() {
  const W = 760, H = 470
  const root = hierarchy([H - 40, 1])
  const X = d3.scaleLinear([0, maxGen], [70, W - 20])
  const yOf = (n) => n.x + 20
  const blues = d3.scaleSequential([scoreExtent[0], scoreExtent[1]], (t) => d3.interpolateBlues(0.3 + t * 0.7))
  const link = d3.linkHorizontal().x((n) => (n.data.row ? X(n.data.row[3]) : 40)).y(yOf)
  const links = root.links().filter((l) => l.source.data.row)
  const bands = d3.groups(root.children, (n) => n.data.row[4]).map(([island, kids]) => ({ island, y0: d3.min(kids.flatMap((k) => k.leaves().map(yOf))) - 6, y1: d3.max(kids.flatMap((k) => k.leaves().map(yOf))) + 6 }))
  const prog = A.progress
  const small = (title, w, h, body) => `<figure class="b-small"><figcaption>${title}</figcaption><svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="display:block;font-family:PlexMono,monospace">${body}</svg></figure>`
  const sx = d3.scaleLinear([0, maxGen], [28, 232])
  const sy = d3.scaleLinear(d3.extent(prog.flatMap((p) => [p.cumulative, p.median]).filter((v) => v !== null)), [92, 10]).nice()
  const costs = prog.filter((p) => p.cost !== null)
  const cy = d3.scaleLinear([0, d3.max(costs, (p) => p.cost) || 1], [92, 10]).nice()
  const fy = d3.scaleLinear([0, d3.max(prog, (p) => p.failed) || 1], [92, 10]).nice()
  const lift = A.best.parentScore === null ? null : A.best.score - A.best.parentScore
  return `<div class="b-page"><header class="b-top"><b class="b-word">Shinka Research</b><nav><span>研究方向</span><span class="is-on">全部运行</span><span>资料库</span></nav><span class="b-search">搜索方向、程序… <kbd>Ctrl K</kbd></span></header>
  <main class="b-wrap b-an"><p class="b-kicker">${esc(shortTitle(dirs[0]))} › h001 › e01 · 演示数据</p>
    <h1 class="b-h1">实验 e01 的进化谱系</h1>
    <nav class="b-views"><span class="is-on">谱系</span><span>进展</span><span>解空间</span><span>对比</span></nav>
    <div class="b-grid"><div>
      <figure class="b-fig"><svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="谱系：横轴为代数，纵向按岛屿分带" style="display:block;font-family:PlexMono,monospace">
        ${bands.map((b, k) => `<rect x="0" y="${b.y0}" width="${W}" height="${b.y1 - b.y0}" fill="${k % 2 ? '#fafafb' : '#f4f5f7'}"/><text x="6" y="${b.y0 + 16}" font-size="11" fill="#6a6a73">岛 ${b.island}</text>`).join('')}
        ${X.ticks(6).map((t) => `<line x1="${X(t)}" x2="${X(t)}" y1="0" y2="${H - 18}" stroke="#e7e7ea" stroke-dasharray="2 3"/><text x="${X(t)}" y="${H - 4}" font-size="11" text-anchor="${X(t) > W - 40 ? 'end' : X(t) < 40 ? 'start' : 'middle'}" fill="#8a8a94">第 ${t} 代</text>`).join('')}
        <g fill="none" stroke="#c9ccd3" stroke-width=".7">${links.map((l) => `<path d="${link(l)}"/>`).join('')}</g>
        <g fill="none" stroke="#f28e2b" stroke-width="2">${links.filter((l) => pathSet.has(l.target.data.i) && pathSet.has(l.source.data.i)).map((l) => `<path d="${link(l)}"/>`).join('')}</g>
        ${root.descendants().filter((n) => n.data.row).map((n) => {
          const x = X(n.data.row[3]), y = yOf(n), s = n.data.row[2]
          return s === null ? `<path d="M${x - 2.2},${y - 2.2}l4.4,4.4m0,-4.4l-4.4,4.4" stroke="#e4573d" stroke-width="1.1"/>` : `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${pathSet.has(n.data.i) ? 3.2 : 2.2}" fill="${pathSet.has(n.data.i) ? '#f28e2b' : blues(s)}" stroke="#fff" stroke-width=".6"/>`
        }).join('')}
        ${(() => { const b = root.descendants().find((n) => n.data.i === A.best.index); const x = X(b.data.row[3]), y = yOf(b); return `<circle cx="${x}" cy="${y}" r="8" fill="none" stroke="#f28e2b" stroke-width="1.5"/><text x="${x - 12}" y="${y - 12}" font-size="12" text-anchor="end" fill="#1b1b1f" font-weight="500">最佳 ${num(A.best.score)}</text>` })()}
      </svg><figcaption><b>图 1</b> · ${A.stats.programs} 个程序按代排列，颜色越深得分越高；橙线是最佳程序的祖先链，红叉为评估失败，不计入最佳。</figcaption></figure>
      <div class="b-smalls">
        ${small('累计最佳与中位得分', 240, 104, `${sy.ticks(3).map((t) => `<line x1="28" x2="232" y1="${sy(t)}" y2="${sy(t)}" stroke="#efeff1"/><text x="24" y="${sy(t) + 3}" font-size="9.5" text-anchor="end" fill="#8a8a94">${num(t, 1)}</text>`).join('')}
          <path d="${d3.line().curve(d3.curveMonotoneX).defined((p) => p.median !== null).x((p) => sx(p.generation)).y((p) => sy(p.median))(prog)}" fill="none" stroke="#9a9aa2" stroke-dasharray="3 2"/>
          <path d="${d3.line().curve(d3.curveStepAfter).defined((p) => p.cumulative !== null).x((p) => sx(p.generation)).y((p) => sy(p.cumulative))(prog)}" fill="none" stroke="#3b6fd8" stroke-width="1.6"/>`)}
        ${small('每代费用（美元）', 240, 104, `${cy.ticks(3).map((t) => `<line x1="28" x2="232" y1="${cy(t)}" y2="${cy(t)}" stroke="#efeff1"/><text x="24" y="${cy(t) + 3}" font-size="9.5" text-anchor="end" fill="#8a8a94">${num(t, 2)}</text>`).join('')}
          ${costs.map((p) => `<rect x="${sx(p.generation) - 2}" y="${cy(p.cost)}" width="4.5" height="${92 - cy(p.cost)}" fill="#6cc5b0"/>`).join('')}`)}
        ${small('每代评估失败数', 240, 104, `${fy.ticks(3).map((t) => `<line x1="28" x2="232" y1="${fy(t)}" y2="${fy(t)}" stroke="#efeff1"/><text x="24" y="${fy(t) + 3}" font-size="9.5" text-anchor="end" fill="#8a8a94">${t}</text>`).join('')}
          ${prog.map((p) => `<rect x="${sx(p.generation) - 2}" y="${fy(p.failed)}" width="4.5" height="${92 - fy(p.failed)}" fill="#e4573d" fill-opacity=".75"/>`).join('')}`)}
      </div></div>
      <aside class="b-side"><p class="b-kicker">所选程序</p><h2>${esc(A.best.name)}</h2>
        <dl class="b-kv"><div><dt>得分</dt><dd>${num(A.best.score)}${lift === null ? '' : ` <span class="b-up">+${num(lift)}</span>`}</dd></div><div><dt>代 · 岛</dt><dd>${A.best.generation} · ${A.best.island}</dd></div><div><dt>补丁</dt><dd>${esc(A.best.patch ?? '—')}</dd></div><div><dt>模型</dt><dd>${esc(A.best.model ?? '—')}</dd></div><div><dt>费用</dt><dd>${A.best.cost === null ? '未知' : money(A.best.cost)}</dd></div></dl>
        <pre class="b-code">${codeLines.slice(0, 19).map((line, i) => `<span class="ln">${i + 1}</span>${highlight(line, { keyword: '#c4282a', string: '#0e7b3c', number: '#1f5bd6', comment: '#8a8a94' })}`).join('\n')}</pre>
        <p class="b-more">查看完整代码、差异、评估与 LLM 记录 →</p></aside></div></main></div>`
}
function bMobile() {
  return `<div class="b-page b-phone"><header class="b-top"><b class="b-word">Shinka Research</b><span class="b-burger">☰</span></header>
    <main class="b-wrap"><p class="b-kicker">工作区 · DEMO 演示数据</p><h1 class="b-headline">${bHeadline()}</h1>
    <section class="b-facets">${dirs.map((d) => bFacet(d, 358)).join('')}</section></main></div>`
}
const B_CSS = `
[data-direction-id="research-paper"] .frame{background:#fff;color:#1b1b1f;font:400 15px/1.7 InterV,"PingFang SC","Microsoft YaHei","Noto Sans SC",sans-serif;font-feature-settings:"cv11"}
[data-direction-id="research-paper"] .b-mono,[data-direction-id="research-paper"] dd,[data-direction-id="research-paper"] kbd{font-family:PlexMono,"JetBrains Mono",monospace;font-variant-numeric:tabular-nums}
[data-direction-id="research-paper"] .b-page{height:100%;overflow:hidden}
[data-direction-id="research-paper"] .b-top{height:60px;display:flex;align-items:center;gap:36px;padding:0 120px;border-bottom:1px solid #e7e7ea}
[data-direction-id="research-paper"] .b-word{font-size:15px;font-weight:650;letter-spacing:-.01em}
[data-direction-id="research-paper"] .b-top nav{display:flex;gap:24px;color:#6a6a73;font-size:14px} [data-direction-id="research-paper"] .b-top nav .is-on{color:#1b1b1f;box-shadow:0 2px 0 #1b1b1f;padding-bottom:2px}
[data-direction-id="research-paper"] .b-search{margin-left:auto;display:flex;gap:10px;align-items:center;height:34px;padding:0 10px 0 12px;width:250px;justify-content:space-between;background:#f4f4f5;border-radius:8px;color:#8a8a94;font-size:13px}
[data-direction-id="research-paper"] kbd{font-size:11px;color:#6a6a73;background:#fff;border:1px solid #e1e1e5;border-radius:4px;padding:0 5px}
[data-direction-id="research-paper"] .b-wrap{width:1200px;margin:0 auto;padding-top:34px}
[data-direction-id="research-paper"] .b-kicker{margin:0 0 10px;font:500 12px/1.4 PlexMono,monospace;color:#6a6a73;letter-spacing:.02em}
[data-direction-id="research-paper"] .b-headline{margin:0;max-width:900px;font-size:36px;line-height:1.32;font-weight:650;letter-spacing:0;line-break:strict;word-break:normal;text-wrap:balance}
[data-direction-id="research-paper"] .b-headline em,[data-direction-id="research-paper"] .b-keep{white-space:nowrap}
[data-direction-id="research-paper"] .b-headline em{font-style:normal;background-repeat:no-repeat;background-size:100% 3px;background-position:0 92%}
[data-direction-id="research-paper"] .b-ok{background-image:linear-gradient(#2e9e5b,#2e9e5b)} [data-direction-id="research-paper"] .b-warn{background-image:linear-gradient(#e4573d,#e4573d)}
[data-direction-id="research-paper"] .b-lede{margin:14px 0 30px;max-width:780px;color:#4a4a52;font-size:16px;line-height:1.75} [data-direction-id="research-paper"] .b-lede b{font-family:PlexMono,monospace;font-weight:500;color:#1b1b1f}
[data-direction-id="research-paper"] .b-facets{display:grid;grid-template-columns:repeat(4,276px);gap:32px;padding-top:22px;border-top:1px solid #1b1b1f}
[data-direction-id="research-paper"] .b-facet h3{margin:0;font-size:16px;line-height:1.45;font-weight:650;min-height:46px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
[data-direction-id="research-paper"] .b-facetmeta{display:flex;gap:8px;align-items:center;margin:6px 0 8px;font-size:12px;color:#6a6a73}
[data-direction-id="research-paper"] .b-tag{border:1px solid #d4d4d8;border-radius:3px;padding:0 4px;font-size:11px;line-height:16px}
[data-direction-id="research-paper"] .b-state{margin-left:auto;font-weight:600} [data-direction-id="research-paper"] .b-state-over-budget{color:#e4573d} [data-direction-id="research-paper"] .b-state-concluded{color:#17803d} [data-direction-id="research-paper"] .b-state-active{color:#3b6fd8} [data-direction-id="research-paper"] .b-state-new{color:#8a8a94}
[data-direction-id="research-paper"] .b-chips{display:flex;flex-wrap:wrap;gap:4px 12px;min-height:24px;font-size:12.5px;color:#4a4a52}
[data-direction-id="research-paper"] .b-chip{display:inline-flex;align-items:center;gap:5px} [data-direction-id="research-paper"] .b-chip i{width:8px;height:8px;border-radius:50%}
[data-direction-id="research-paper"] figure{margin:10px 0 8px}
[data-direction-id="research-paper"] .b-empty{height:148px;display:grid;place-content:center;text-align:center;color:#8a8a94;font-size:14px;line-height:1.6;background:repeating-linear-gradient(135deg,#fafafa 0 6px,#f4f4f5 6px 7px)} [data-direction-id="research-paper"] .b-empty span{font-size:12.5px}
[data-direction-id="research-paper"] .b-stats{display:grid;grid-template-columns:auto 1fr auto;gap:12px;margin:0;padding-top:8px;border-top:1px solid #e7e7ea}
[data-direction-id="research-paper"] dt{font-size:11.5px;color:#8a8a94} [data-direction-id="research-paper"] dd{margin:0;font-size:13.5px} [data-direction-id="research-paper"] dd span{color:#8a8a94} [data-direction-id="research-paper"] .b-over{color:#e4573d}
[data-direction-id="research-paper"] .b-activity{margin-top:30px} [data-direction-id="research-paper"] .b-activity h2{font-size:15px;font-weight:650;margin:0 0 12px} [data-direction-id="research-paper"] .b-activity h2 span{font-weight:400;color:#8a8a94;font-size:13px;margin-left:10px}
[data-direction-id="research-paper"] .b-an .b-kicker{margin-bottom:6px} [data-direction-id="research-paper"] .b-h1{margin:0;font-size:28px;line-height:1.3;font-weight:650}
[data-direction-id="research-paper"] .b-views{display:flex;gap:26px;margin:16px 0 18px;border-bottom:1px solid #e7e7ea;font-size:14px;color:#6a6a73} [data-direction-id="research-paper"] .b-views span{padding-bottom:9px} [data-direction-id="research-paper"] .b-views .is-on{color:#1b1b1f;box-shadow:inset 0 -2px 0 #1b1b1f}
[data-direction-id="research-paper"] .b-grid{display:grid;grid-template-columns:760px minmax(0,1fr);gap:40px}
[data-direction-id="research-paper"] .b-fig{margin:0} [data-direction-id="research-paper"] figcaption{font-size:13px;line-height:1.6;color:#4a4a52;margin-top:8px}
[data-direction-id="research-paper"] .b-smalls{display:grid;grid-template-columns:repeat(3,240px);gap:20px;margin-top:6px} [data-direction-id="research-paper"] .b-small{margin:0} [data-direction-id="research-paper"] .b-small figcaption{margin:0 0 2px;font-size:12.5px;font-weight:600;color:#1b1b1f}
[data-direction-id="research-paper"] .b-side{border-left:1px solid #e7e7ea;padding-left:24px;min-width:0} [data-direction-id="research-paper"] .b-side h2{margin:0 0 10px;font-size:18px;line-height:1.4;font-weight:650}
[data-direction-id="research-paper"] .b-kv{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 16px;margin:0 0 14px} [data-direction-id="research-paper"] .b-kv dd{overflow:hidden;text-overflow:ellipsis;white-space:nowrap} [data-direction-id="research-paper"] .b-up{color:#17803d}
[data-direction-id="research-paper"] .b-code{margin:0;padding:12px 14px;background:#f7f7f8;border-radius:8px;font:400 12px/19px PlexMono,"JetBrains Mono",monospace;color:#2a2a30;white-space:pre;overflow:hidden;max-height:440px}
[data-direction-id="research-paper"] .b-code .ln{display:inline-block;width:22px;margin-right:12px;text-align:right;color:#b4b4bb}
[data-direction-id="research-paper"] .b-more{margin:10px 0 0;color:#0e8a7a;font-size:13.5px;font-weight:500}
[data-direction-id="research-paper"] .b-phone{display:flex;flex-direction:column} [data-direction-id="research-paper"] .b-phone .b-top{flex:none;padding:0 16px;height:52px} [data-direction-id="research-paper"] .b-burger{margin-left:auto;font-size:18px}
[data-direction-id="research-paper"] .b-phone .b-wrap{width:auto;flex:1;min-height:0;overflow-y:auto;scrollbar-width:none;padding:22px 16px 28px} [data-direction-id="research-paper"] .b-phone .b-headline{font-size:24px;line-height:1.42}
[data-direction-id="research-paper"] .b-phone .b-facets{grid-template-columns:1fr;gap:26px;margin-top:22px}
`

// ======================= C · 适应度地形 =======================
function cScale(d, width) {
  if (d.budget === null) return `<div class="c-scale"><span>花费 ${money(d.spend)} · 未设预算</span></div>`
  const ratio = d.spend / d.budget
  const w = Math.min(1.25, ratio) / 1.25 * width
  return `<div class="c-scale"><svg width="${width}" height="16" viewBox="0 0 ${width} 16" aria-hidden="true">
    ${d3.range(0, 1.0001, 0.25).map((t) => `<line x1="${(t / 1.25) * width}" x2="${(t / 1.25) * width}" y1="3" y2="11" stroke="#5d7686"/>`).join('')}
    <rect x="0" y="6" width="${(1 / 1.25) * width}" height="2" fill="#2b4558"/>
    <rect x="0" y="6" width="${w}" height="2" fill="${ratio > 1 ? '#e2725b' : '#e9b949'}"/>
    ${ratio > 1 ? `<path d="M${(1 / 1.25) * width},2v12" stroke="#e2725b" stroke-width="1.5"/>` : ''}</svg>
    <span>${money(d.spend)} / ${money(d.budget)}${ratio > 1 ? ` <b class="c-coral">超支 ${Math.round(ratio * 100 - 100)}%</b>` : ''}</span></div>`
}
const C_SYMBOL = { supported: '▲', confirmed: '★', refuted: '✕', inconclusive: '△', running: '◉', challenged: '◇', proposed: '○', selected: '○', unconfirmed: '✕', dropped: '·' }
const cEvidence = (d) => (d.best === null || d.bar === null ? '' : `<span class="c-evid ${d.best > d.bar ? 'is-above' : ''}">方向最佳 <b>${num(d.best)}</b> · 噪声线 <b>${num(d.bar)}</b></span>`)
function cSheet(d, k, w, h, mapH) {
  const map = fixture.maps[d.id]
  const body = map ? atlasMap(map, w - 2, mapH, {}) : `<div class="c-unsurveyed" style="height:${mapH}px"><span>未勘测</span><em>${d.hypotheses.length} 个假设已提出 · 尚无试验</em></div>`
  return `<article class="c-sheet" style="width:${w}px${h ? `;height:${h}px` : ''}">
    <header><span class="c-sheetno">SHEET ${pad(k + 1)}</span><b>${esc(shortTitle(d))}</b>${cEvidence(d)}<span class="c-trials">${int(d.trials)} 次试验</span></header>
    <div class="c-map">${body}${map ? `<span class="c-coord">${esc(d.id)} · ${esc(map.hypothesis)}/${esc(map.experiment)}</span>` : ''}</div>
    <footer>${cEvidence(d)}${cScale(d, 120)}<span class="c-legend">${statusCounts(d).map(([s, n]) => `<span title="${esc(STATUS[s])}"><i class="c-sym-${s}">${C_SYMBOL[s]}</i>${STATUS[s]} ${n}</span>`).join('')}</span>
      <span class="c-seal ${d.lockbox === 'opened' ? 'is-open' : ''}">${d.lockbox === 'sealed' ? '封存段 · 未开' : d.lockbox === 'opened' ? '封存段 · 已开' : '封存段 · 无'}</span><span class="c-demo">DEMO</span></footer></article>`
}
function cHome() {
  return `<div class="c-page"><header class="c-top"><b>SHINKA RESEARCH</b><span class="c-sub">研究地图集</span><nav><span class="is-on">方向</span><span>运行</span><span>资料</span></nav><span class="c-read">${dirs.length} SHEETS · ${int(total.trials)} TRIALS · ${money(total.spend)}</span></header>
    <main class="c-sheets">${dirs.map((d, k) => cSheet(d, k, 686, 388, 300)).join('')}</main></div>`
}
function cAnalysis() {
  const map = { programs: rows.map((r) => r.slice(0, 6)), best: A.best.index, path: A.path, hypothesis: 'h001', experiment: 'e01' }
  const prog = A.progress
  const px = d3.scaleLinear([0, maxGen], [16, 940])
  const py = d3.scaleLinear(d3.extent(prog.map((p) => p.cumulative).filter((v) => v !== null)), [96, 18]).nice()
  const area = d3.area().curve(d3.curveMonotoneX).defined((p) => p.cumulative !== null).x((p) => px(p.generation)).y0(96).y1((p) => py(p.cumulative))(prog)
  const lift = A.best.parentScore === null ? null : A.best.score - A.best.parentScore
  return `<div class="c-page"><header class="c-top"><b>SHINKA RESEARCH</b><span class="c-sub">${esc(shortTitle(dirs[0]))} · h001/e01</span><nav><span>谱系</span><span>进展</span><span class="is-on">解空间</span><span>对比</span></nav><span class="c-read">${A.stats.programs} PROGRAMS · ${A.stats.generations} GEN · ${A.stats.islands} ISLANDS</span></header>
    <main class="c-full"><div class="c-mapwrap">${atlasMap(map, 990, 640, { dense: true, focus: 5 })}</div>
      <aside class="c-legendbox"><b>图例</b><span><i class="c-swatch"></i>得分等值面（仅有效得分覆盖处）</span><span><i class="c-hatch"></i>已勘测·有效得分不足</span><span><i class="c-dot"></i>程序</span><span><i class="c-gold"></i>最佳谱系</span><span><i class="c-x">✕</i>评估失败</span><em>嵌入投影，距离不代表真实相似度</em></aside>
      <aside class="c-panel"><span class="c-sheetno">第 ${A.best.generation} 代 · 岛 ${A.best.island} · ${esc(A.best.patch ?? '—')}</span><h2>${esc(A.best.name)}</h2>
        <div class="c-elev"><b>${num(A.best.score)}</b>${lift === null ? '' : `<span>▲ ${num(lift)} 高于父程序</span>`}</div>
        <div class="c-tabs"><span class="is-on">代码</span><span>差异</span><span>评估</span><span>LLM</span></div>
        <pre class="c-code">${codeLines.slice(0, 20).map((line, i) => `<span class="ln">${i + 1}</span>${highlight(line, { keyword: '#e9b949', string: '#8ccab3', number: '#e7eef2', comment: '#5d7686' })}`).join('\n')}</pre></aside>
      <section class="c-profile"><header><b>高程剖面</b><span>每代累计最佳得分 · 金线为所选程序所在代</span></header>
        <svg width="956" height="104" viewBox="0 0 956 104" aria-hidden="true"><defs><linearGradient id="c-prof" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#4ba39b" stop-opacity=".9"/><stop offset="1" stop-color="#10283a" stop-opacity=".2"/></linearGradient></defs>
          ${py.ticks(3).map((t) => `<line x1="16" x2="940" y1="${py(t)}" y2="${py(t)}" stroke="#1f3a4d"/><text x="944" y="${py(t) + 3}" class="c-axis">${num(t, 2)}</text>`).join('')}
          <path d="${area}" fill="url(#c-prof)" stroke="#8ccab3" stroke-width="1.2"/>
          <line x1="${px(A.best.generation)}" x2="${px(A.best.generation)}" y1="8" y2="96" stroke="#e9b949" stroke-width="1.2"/>
          ${px.ticks(8).map((t) => `<text x="${px(t)}" y="104" text-anchor="middle" class="c-axis">G${t}</text>`).join('')}</svg></section></main></div>`
}
function cMobile() {
  return `<div class="c-page c-phone"><header class="c-top"><b>SHINKA</b><span class="c-sub">研究地图集</span><span class="c-menu">☰</span></header>
    <main class="c-stack">${dirs.map((d, k) => cSheet(d, k, 358, null, 182)).join('')}</main></div>`
}
const C_CSS = `
[data-direction-id="fitness-atlas"] .frame{background:#0b1622;color:#e7eef2;font:400 13px/1.6 "Bahnschrift","Barlow Semi Condensed","PingFang SC","Microsoft YaHei",sans-serif}
[data-direction-id="fitness-atlas"] .c-page{height:100%;display:flex;flex-direction:column;overflow:hidden}
[data-direction-id="fitness-atlas"] .c-top{height:56px;flex:none;display:flex;align-items:center;gap:22px;padding:0 24px;border-bottom:1px solid #1f3a4d;font-stretch:condensed}
[data-direction-id="fitness-atlas"] .c-top b{letter-spacing:.22em;font-weight:600;font-size:14px}
[data-direction-id="fitness-atlas"] .c-sub{color:#8fa6b5;font-size:13px}
[data-direction-id="fitness-atlas"] .c-top nav{display:flex;gap:4px;margin-left:12px} [data-direction-id="fitness-atlas"] .c-top nav span{padding:4px 12px;color:#8fa6b5;border:1px solid transparent} [data-direction-id="fitness-atlas"] .c-top nav .is-on{color:#e9b949;border-color:#3e6075}
[data-direction-id="fitness-atlas"] .c-read{margin-left:auto;letter-spacing:.14em;font-size:12px;color:#8fa6b5}
[data-direction-id="fitness-atlas"] .c-sheets{display:grid;grid-template-columns:686px 686px;gap:20px;padding:24px}
[data-direction-id="fitness-atlas"] .c-sheet{border:1px solid #3e6075;outline:1px solid #1f3a4d;outline-offset:3px;display:flex;flex-direction:column;background:#0b1622}
[data-direction-id="fitness-atlas"] .c-sheet header{height:34px;flex:none;display:flex;align-items:center;gap:12px;padding:0 12px;border-bottom:1px solid #2b4558}
[data-direction-id="fitness-atlas"] .c-sheetno{letter-spacing:.18em;font-size:11px;color:#e9b949;white-space:nowrap}
[data-direction-id="fitness-atlas"] .c-sheet header b{font-weight:600;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
[data-direction-id="fitness-atlas"] .c-trials{margin-left:auto;color:#8fa6b5;font-size:12px;letter-spacing:.04em;white-space:nowrap}
[data-direction-id="fitness-atlas"] .c-evid{color:#8fa6b5;font-size:12px;white-space:nowrap} [data-direction-id="fitness-atlas"] .c-evid b{color:#e7eef2;font-weight:600} [data-direction-id="fitness-atlas"] .c-evid.is-above b:first-of-type{color:#e9b949}
[data-direction-id="fitness-atlas"] .c-sheet header .c-evid{margin-left:auto} [data-direction-id="fitness-atlas"] .c-sheet header .c-evid + .c-trials{margin-left:0} [data-direction-id="fitness-atlas"] .c-sheet footer .c-evid{display:none}
[data-direction-id="fitness-atlas"] .c-map{position:relative;flex:1;overflow:hidden}
[data-direction-id="fitness-atlas"] .c-coord{position:absolute;left:10px;bottom:8px;font-size:11px;letter-spacing:.08em;color:#8fa6b5;background:#0b1622cc;padding:0 6px}
[data-direction-id="fitness-atlas"] .c-unsurveyed{display:grid;place-content:center;text-align:center;gap:6px;background-image:linear-gradient(#163042 1px,transparent 1px),linear-gradient(90deg,#163042 1px,transparent 1px);background-size:24px 24px}
[data-direction-id="fitness-atlas"] .c-unsurveyed span{letter-spacing:.4em;font-size:22px;color:#5d7686} [data-direction-id="fitness-atlas"] .c-unsurveyed em{font-style:normal;color:#8fa6b5;font-size:12.5px}
[data-direction-id="fitness-atlas"] .c-sheet footer{height:44px;flex:none;display:flex;align-items:center;gap:16px;padding:0 12px;border-top:1px solid #2b4558;font-size:12px;color:#8fa6b5}
[data-direction-id="fitness-atlas"] .c-scale{display:flex;align-items:center;gap:8px;white-space:nowrap} [data-direction-id="fitness-atlas"] .c-coral{color:#e2725b;font-weight:600}
[data-direction-id="fitness-atlas"] .c-legend{display:flex;gap:10px;white-space:nowrap;overflow:hidden} [data-direction-id="fitness-atlas"] .c-legend i{font-style:normal;margin-right:3px;color:#e9b949}
[data-direction-id="fitness-atlas"] .c-sym-refuted,[data-direction-id="fitness-atlas"] .c-sym-unconfirmed{color:#e2725b!important} [data-direction-id="fitness-atlas"] .c-sym-running{color:#8ccab3!important} [data-direction-id="fitness-atlas"] .c-sym-proposed,[data-direction-id="fitness-atlas"] .c-sym-challenged{color:#8fa6b5!important}
[data-direction-id="fitness-atlas"] .c-seal{margin-left:auto;border:1px solid #3e6075;padding:1px 8px;letter-spacing:.06em;white-space:nowrap} [data-direction-id="fitness-atlas"] .c-seal.is-open{border-color:#e9b949;color:#e9b949}
[data-direction-id="fitness-atlas"] .c-demo{letter-spacing:.16em;font-size:10px;color:#5d7686}
[data-direction-id="fitness-atlas"] .c-maplabel{font:600 10.5px Bahnschrift,"Barlow Semi Condensed",sans-serif;letter-spacing:.14em;fill:#cfe2e6;paint-order:stroke;stroke:#0b1622;stroke-width:3px}
[data-direction-id="fitness-atlas"] .c-spot{font:600 12px Bahnschrift,"Microsoft YaHei",sans-serif;fill:#e9b949;paint-order:stroke;stroke:#0b1622;stroke-width:3px}
[data-direction-id="fitness-atlas"] .c-full{position:relative;flex:1;overflow:hidden;background:#0b1622}
[data-direction-id="fitness-atlas"] .c-mapwrap{position:absolute;left:12px;top:44px}
[data-direction-id="fitness-atlas"] .c-legendbox{position:absolute;left:24px;top:14px;right:448px;display:flex;align-items:center;gap:18px;font-size:12px;color:#cfe2e6;white-space:nowrap}
[data-direction-id="fitness-atlas"] .c-legendbox b{letter-spacing:.2em;font-size:11px;color:#e9b949} [data-direction-id="fitness-atlas"] .c-legendbox span{display:inline-flex;align-items:center;gap:6px} [data-direction-id="fitness-atlas"] .c-legendbox em{margin-left:auto;font-style:normal;color:#8fa6b5;font-size:11.5px}
[data-direction-id="fitness-atlas"] .c-swatch{width:22px;height:10px;background:linear-gradient(90deg,#144259,#4ba39b,#dcebd0)} [data-direction-id="fitness-atlas"] .c-hatch{width:22px;height:10px;box-sizing:border-box;border:1px dashed rgba(226,114,91,.6);background:repeating-linear-gradient(135deg,rgba(226,114,91,.6) 0 1px,transparent 1px 4px)} [data-direction-id="fitness-atlas"] .c-dot{width:5px;height:5px;border-radius:50%;background:#e7eef2;margin:0 8px} [data-direction-id="fitness-atlas"] .c-gold{width:22px;height:2px;background:#e9b949} [data-direction-id="fitness-atlas"] .c-x{font-style:normal;color:#e2725b;width:22px;text-align:center}
[data-direction-id="fitness-atlas"] .c-panel{position:absolute;right:24px;top:24px;width:400px;bottom:24px;background:#0b1622;border:1px solid #3e6075;outline:1px solid #1f3a4d;outline-offset:3px;padding:16px;display:flex;flex-direction:column;gap:8px}
[data-direction-id="fitness-atlas"] .c-panel h2{margin:0;font-size:17px;font-weight:600;line-height:1.35}
[data-direction-id="fitness-atlas"] .c-elev{display:flex;align-items:baseline;gap:12px} [data-direction-id="fitness-atlas"] .c-elev b{font-size:30px;font-weight:600;color:#e9b949;letter-spacing:.02em} [data-direction-id="fitness-atlas"] .c-elev span{color:#8ccab3;font-size:12.5px}
[data-direction-id="fitness-atlas"] .c-tabs{display:flex;gap:18px;border-bottom:1px solid #2b4558;padding-bottom:6px;color:#8fa6b5} [data-direction-id="fitness-atlas"] .c-tabs .is-on{color:#e7eef2;box-shadow:0 7px 0 -5px #e9b949}
[data-direction-id="fitness-atlas"] .c-code{margin:0;font:400 12px/19px PlexMono,"JetBrains Mono",monospace;color:#cfe2e6;white-space:pre;overflow:hidden;flex:1}
[data-direction-id="fitness-atlas"] .c-code .ln{display:inline-block;width:22px;margin-right:12px;text-align:right;color:#3e6075}
[data-direction-id="fitness-atlas"] .c-profile{position:absolute;left:24px;bottom:24px;width:972px;padding:10px 8px 6px;background:#0b1622;border:1px solid #3e6075}
[data-direction-id="fitness-atlas"] .c-profile header{display:flex;gap:12px;align-items:baseline;padding:0 8px 4px} [data-direction-id="fitness-atlas"] .c-profile b{letter-spacing:.14em;font-size:12px;color:#e9b949} [data-direction-id="fitness-atlas"] .c-profile span{color:#8fa6b5;font-size:12px}
[data-direction-id="fitness-atlas"] .c-axis{font:400 10px Bahnschrift,sans-serif;fill:#5d7686;letter-spacing:.06em}
[data-direction-id="fitness-atlas"] .c-phone .c-top{height:52px;gap:12px;padding:0 16px} [data-direction-id="fitness-atlas"] .c-menu{margin-left:auto;font-size:18px}
[data-direction-id="fitness-atlas"] .c-stack{display:grid;align-content:start;gap:18px;padding:16px;flex:1;min-height:0;overflow-y:auto;scrollbar-width:none} [data-direction-id="fitness-atlas"] .c-phone .c-map{flex:none} [data-direction-id="fitness-atlas"] .c-phone .c-sheet footer{height:auto;flex-wrap:wrap;gap:6px 10px;padding:8px 12px} [data-direction-id="fitness-atlas"] .c-phone .c-legend{order:3;flex-basis:100%;flex-wrap:wrap;row-gap:2px;white-space:normal} [data-direction-id="fitness-atlas"] .c-phone .c-sheet header .c-evid{display:none} [data-direction-id="fitness-atlas"] .c-phone .c-sheet footer .c-evid{display:inline;order:-1;flex-basis:100%}
[data-direction-id="fitness-atlas"] .c-phone .c-sheet header{height:auto;min-height:34px;padding:7px 12px;align-items:flex-start} [data-direction-id="fitness-atlas"] .c-phone .c-sheet header b{flex:1;min-width:0;white-space:normal;line-height:1.4} [data-direction-id="fitness-atlas"] .c-phone .c-sheet header .c-sheetno,[data-direction-id="fitness-atlas"] .c-phone .c-sheet header .c-trials{line-height:19.6px}
`

// ======================= shell =======================
const DIRECTIONS = [
  {
    id: 'dark-instrument', name: 'A · 暗场仪器', home: aHome, analysis: aAnalysis, mobile: aMobile, css: A_CSS, analysisView: '谱系',
    thesis: '研究进程是一台精密仪器：一屏读完全部方向的状态与证据，键盘直达任何程序。',
    signature: '证据刻度：每个方向把最佳记录和噪声线画在同一把刻度上，越过噪声线的一段与谱系里的最佳祖先链用同一种信号橙。',
    axes: { luminance: 'dark (neutral graphite #0a0a0b)', typeFamily: 'neo-grotesk variable UI sans + plex mono figures', color: 'graphite neutrals + one signal orange; status by glyph shape', layout: 'three-pane app shell: nav rail, content, activity or inspector', density: 'dense', era: 'contemporary product software (2020s)', material: 'layered dark planes with hairline rules' },
    axesZh: { 明度: '深色（中性石墨 #0a0a0b）', 字体: '新怪诞无衬线 UI 字体 + Plex 等宽数字', 色彩: '石墨中性色 + 唯一信号橙；状态用图形区分', 版式: '三栏应用：导航栏、内容、动态或检查器', 密度: '紧凑', 时代: '当代产品软件（2020 年代）', 材质: '分层深色平面 + 细线分隔' },
  },
  {
    id: 'research-paper', name: 'B · 研究报告', home: bHome, analysis: bAnalysis, mobile: bMobile, css: B_CSS, analysisView: '谱系',
    thesis: '每个方向是一篇可核验的研究报告：标题句直接陈述现状，图在前，数字跟着图走。',
    signature: '数据句标题：首页第一句话由实时数据生成，彩色下划线把“已确认”“超预算”连到下面对应的小图。',
    axes: { luminance: 'light (paper white #ffffff)', typeFamily: 'humanist sans reading text + monospace figures and kickers', color: 'white ground with observable-style categorical status palette', layout: 'editorial column with small multiples and a sidenote column', density: 'airy, reading-first', era: 'contemporary data journalism and computational notebooks', material: 'flat paper with hairline figure rules' },
    axesZh: { 明度: '浅色（纸白 #ffffff）', 字体: '人文无衬线正文 + 等宽数字与眉题', 色彩: '白底 + Observable 式分类状态色', 版式: '编辑式单栏 + 小倍数图 + 旁注栏', 密度: '疏朗，阅读优先', 时代: '当代数据新闻与计算笔记本', 材质: '平面纸张 + 细线图框' },
  },
  {
    id: 'fitness-atlas', name: 'C · 适应度地形', home: cHome, analysis: cAnalysis, mobile: cMobile, css: C_CSS, analysisView: '解空间',
    thesis: '进化是在适应度地形上的勘探：每个方向是一幅图幅，程序是测点，最佳谱系是登顶路线。',
    signature: '等值地形：只在有效得分测点覆盖的范围内插值画等高线，有效得分不足的已勘测区画成珊瑚色斜线，不借用远处得分；金色路线标出最佳程序的祖先链，空方向显示为“未勘测”图幅。',
    axes: { luminance: 'dark (deep navy #0b1622)', typeFamily: 'condensed DIN-style grotesk with letter-spaced map labels', color: 'hypsometric teal ramp + gold route + coral failures', layout: 'full-bleed map sheets with floating legend and panels', density: 'balanced', era: 'survey cartography and nautical charts (20th century)', material: 'engraved print with double-rule sheet frames' },
    axesZh: { 明度: '深色（深海军蓝 #0b1622）', 字体: '窄体 DIN 式无衬线 + 字距拉开的地图标注', 色彩: '分层设色青绿色阶 + 金色路线 + 珊瑚色失败', 版式: '满幅图幅 + 浮动图例与面板', 密度: '均衡', 时代: '测绘制图与航海图（20 世纪）', 材质: '雕版印刷感 + 双线图框' },
  },
]
const frame = (d, kind, w, h, html) => `<div class="frame" data-frame="${d.id}-${kind}" style="width:${w}px;height:${h}px">${html}</div>`
const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shinka Research · 方向样稿（非最终产品）</title>
<style>
@font-face{font-family:InterV;src:url(data:font/woff2;base64,${INTER}) format("woff2");font-weight:100 900;font-display:block}
@font-face{font-family:PlexMono;src:url(data:font/woff2;base64,${PLEX400}) format("woff2");font-weight:400;font-display:block}
@font-face{font-family:PlexMono;src:url(data:font/woff2;base64,${PLEX500}) format("woff2");font-weight:500;font-display:block}
*{box-sizing:border-box} body{margin:0;background:#e4e4e7;color:#18181b;font:400 14px/1.6 system-ui,"PingFang SC","Microsoft YaHei",sans-serif}
.shell-head{padding:28px 32px 8px;max-width:1500px}
.shell-head h1{margin:0;font-size:22px} .shell-head p{margin:6px 0 0;color:#52525b;max-width:900px}
.stage{margin:24px 32px 48px;padding:20px;background:#f4f4f5;border:1px solid #d4d4d8}
.stage h2{margin:0;font-size:18px} .stage .thesis{margin:6px 0 2px;font-size:15px} .stage .signature{margin:0 0 10px;color:#3f3f46}
.axes{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px;margin:0 0 16px;font-size:12px}
.axes div{background:#fff;border:1px solid #e4e4e7;padding:6px 8px} .axes b{display:block;font-size:11px;color:#71717a;font-weight:600}
.frames{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) 196px;gap:16px;align-items:start}
.slot{position:relative;overflow:hidden;background:#000} .slot .frame{position:absolute;left:0;top:0;transform-origin:0 0;overflow:hidden}
.slot-label{font-size:12px;color:#52525b;margin:6px 0 0}
@media (max-width:900px){.axes{grid-template-columns:repeat(2,minmax(0,1fr))}.frames{grid-template-columns:1fr}.stage{margin:16px 12px}.shell-head{padding:20px 16px 4px}}
body.capture{background:transparent} body.capture .shell-head,body.capture .slot-label,body.capture .stage>h2,body.capture .thesis,body.capture .signature,body.capture .axes{display:none}
body.capture .stage{margin:0;padding:0;border:0;background:none} body.capture .frames{display:block} body.capture .slot{display:none;position:static;overflow:visible} body.capture .slot.is-target{display:block}
body.capture .slot .frame{position:static;transform:none!important} body.capture .stage:not(.has-target){display:none}
${DIRECTIONS.map((d) => d.css).join('\n')}
</style></head><body data-direction-preview="award-redesign">
<header class="shell-head"><h1>Shinka Research 方向样稿 · direction preview（不是最终产品）</h1>
<p>三个方向使用同一份演示数据（DEMO · 演示数据，非真实研究）、同一组状态与相同视口：首页 1440×900、实验分析 1440×900、手机首页 390×844（框内可滚动查看全部方向）。请按方向选择，而不是按单个细节选择；选定后再细化设计系统。</p></header>
${DIRECTIONS.map((d) => `<section class="stage" data-direction-id="${d.id}"><h2>${d.name}</h2><p class="thesis">${d.thesis}</p><p class="signature"><b>标志：</b>${d.signature}</p>
<div class="axes">${Object.entries(d.axesZh).map(([k, v]) => `<div><b>${k}</b>${esc(v)}</div>`).join('')}</div>
<div class="frames">
<div><div class="slot" data-w="1440" data-h="900">${frame(d, 'home', 1440, 900, d.home())}</div><p class="slot-label">首页 · 研究方向总览</p></div>
<div><div class="slot" data-w="1440" data-h="900">${frame(d, 'analysis', 1440, 900, d.analysis())}</div><p class="slot-label">实验分析 · ${d.analysisView}与程序检查器</p></div>
<div><div class="slot" data-w="390" data-h="844">${frame(d, 'mobile', 390, 844, d.mobile())}</div><p class="slot-label">手机首页 · 390×844 · 框内可滚动</p></div>
</div></section>`).join('\n')}
<script>
const target = new URLSearchParams(location.search).get('frame')
if (target) {
  document.body.classList.add('capture')
  const node = document.querySelector('[data-frame="' + CSS.escape(target) + '"]')
  node.closest('.slot').classList.add('is-target')
  node.closest('.stage').classList.add('has-target')
} else {
  const fit = () => document.querySelectorAll('.slot').forEach((slot) => {
    const w = +slot.dataset.w, h = +slot.dataset.h, s = slot.clientWidth / w
    slot.style.height = h * s + 'px'
    slot.firstElementChild.style.transform = 'scale(' + s + ')'
  })
  addEventListener('resize', fit); fit()
}
</script></body></html>
`
writeFileSync(here('./index.html'), html)
console.log(JSON.stringify({ bytes: html.length, directions: DIRECTIONS.map((d) => d.id) }))
