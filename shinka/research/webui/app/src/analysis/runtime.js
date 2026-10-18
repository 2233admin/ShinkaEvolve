import Chart from 'chart.js/auto'
import * as d3 from 'd3'
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'
import * as Diff from 'diff'
import JSZip from 'jszip'
import Plotly from 'plotly.js-dist-min'
import DOMPurify from 'dompurify'
import { parseAction, plotURL } from './actions.js'
import { translatedLabel } from './accessibility.js'
import '@fontsource-variable/inter'
import '@fontsource/ibm-plex-mono/400.css'
import 'virtual:tokens.css'
import './workbench.css'

hljs.registerLanguage('python', python)
const markdown = new MarkdownIt({ html: false, linkify: true })
Object.assign(window, { Chart, d3, hljs, Diff, JSZip, Plotly,
  // The recovered renderer only consumes these two Markdown operations.
  marked: { parse: (text) => markdown.render(String(text)), parseInline: (text) => markdown.renderInline(String(text)) },
})
window.shinkaHydratePrograms = async (programs, dbPath) => {
  await Promise.all(programs.filter((program) => program.code === undefined).map(async (program) => {
    const query = new URLSearchParams({ db_path: dbPath, id: program.id })
    const response = await fetch(`/get_program_details?${query}`)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    Object.assign(program, await response.json())
  }))
}

// Dynamic templates retain a finite, non-eval action grammar. Raw experiment
// HTML never gets event handlers, script URLs or access to arbitrary globals.
const actions = new Set(window.ShinkaActions)
DOMPurify.addHook('uponSanitizeAttribute', (node, data) => {
  if (/^on/i.test(data.attrName)) {
    if (['onclick', 'onchange', 'oninput'].includes(data.attrName)
      && parseAction(data.attrValue, actions, node, new Event('validation'))) {
      node.setAttribute(`data-wb-${data.attrName.slice(2)}`, data.attrValue)
    }
    data.keepAttr = false
  }
  if (data.attrName === 'style' && /url\s*\(|expression\s*\(/i.test(data.attrValue)) data.keepAttr = false
  if (['src', 'href', 'xlink:href'].includes(data.attrName) && /^(?:https?:)?\/\//i.test(data.attrValue) && node.tagName.toLowerCase() !== 'a') data.keepAttr = false
})
window.ShinkaHTML = (value) => {
  const html = String(value ?? '')
  // HTML parsers drop a bare <td>/<tr>. Keep the fragment's table context
  // while sanitizing, then return only the sanitized children.
  const tag = html.match(/^\s*<(td|th|tr|thead|tbody|tfoot)\b/i)?.[1].toLowerCase()
  const wrappers = { td: ['<table><tbody><tr>', '</tr></tbody></table>', 'tr'], th: ['<table><tbody><tr>', '</tr></tbody></table>', 'tr'],
    tr: ['<table><tbody>', '</tbody></table>', 'tbody'], thead: ['<table>', '</table>', 'table'],
    tbody: ['<table>', '</table>', 'table'], tfoot: ['<table>', '</table>', 'table'] }
  const wrapper = wrappers[tag]
  const clean = DOMPurify.sanitize(wrapper ? wrapper[0] + html + wrapper[1] : html, {
    USE_PROFILES: { html: true, svg: true }, FORBID_TAGS: ['style', 'iframe', 'object', 'embed', 'form'],
    FORBID_ATTR: ['srcset', 'ping', 'formaction'],
  })
  return wrapper ? new DOMParser().parseFromString(clean, 'text/html').querySelector(wrapper[2]).innerHTML : clean
}
// D3's dynamic HTML uses the same boundary as compiled DOM assignments.
const d3HTML = d3.selection.prototype.html
d3.selection.prototype.html = function (value) {
  if (!arguments.length) return d3HTML.call(this)
  return d3HTML.call(this, typeof value === 'function'
    ? function (...args) { return window.ShinkaHTML(value.apply(this, args)) }
    : window.ShinkaHTML(value))
}
for (const type of ['click', 'change', 'input']) {
  document.addEventListener(type, (event) => {
    const element = event.target.closest?.(`[data-wb-${type}]`)
    if (!element) return
    for (const command of parseAction(element.getAttribute(`data-wb-${type}`), actions, element, event) ?? []) {
      if (command.name === '$event') event[command.args[0]]()
      else if (command.name === 'open') {
        const url = plotURL(command.args[0], location.origin)
        if (url) window.open(url, '_blank', 'noopener')
      } else window[command.name]?.apply(element, command.args)
    }
  }, true)
}

function token(name) { return getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim() }
Chart.register({
  id: 'research-theme',
  beforeUpdate(chart) {
    chart.options.color = token('muted')
    chart.options.font = { family: token('font-sans'), size: 12 }
    for (const series of chart.data.datasets) {
      if (typeof series.researchSeries !== 'number') continue
      series.borderColor = token(['ink', 'muted', 'line-strong'][series.researchSeries % 3])
      series.backgroundColor = series.borderColor
      series.borderDash = [[], [6, 4], [2, 3], [8, 3, 2, 3], [1, 2]][Math.floor(series.researchSeries / 3) % 5]
    }
    for (const axis of Object.values(chart.options.scales ?? {})) {
      axis.grid.color = token('line')
      axis.ticks.color = token('muted')
      axis.title.color = token('muted')
      if (axis.title.text) axis.title.text = translatedLabel(axis.title.text)
    }
    if (chart.options.plugins.legend) {
      chart.options.plugins.legend.labels.color = token('ink')
    }
  },
})
const originalPlot = Plotly.newPlot.bind(Plotly)
Plotly.newPlot = (target, data, layout = {}, config = {}) => originalPlot(target, data, {
  ...layout, paper_bgcolor: token('base'), plot_bgcolor: token('base'),
  font: { ...layout.font, family: token('font-sans'), color: token('ink'), size: 12 },
  xaxis: { ...layout.xaxis, gridcolor: token('line'), zerolinecolor: token('line-strong') },
  yaxis: { ...layout.yaxis, gridcolor: token('line'), zerolinecolor: token('line-strong') },
}, { ...config, responsive: true, displaylogo: false })
window.shinkaSetTheme = (theme) => {
  document.documentElement.dataset.theme = theme
  document.body?.classList.toggle('dark', theme !== 'light')
  Chart.defaults.color = token('muted')
  Chart.defaults.font.family = token('font-sans')
  Object.values(Chart.instances).forEach((chart) => chart.update('none'))
  document.querySelectorAll('.js-plotly-plot').forEach((plot) => Plotly.relayout(plot, {
    paper_bgcolor: token('base'), plot_bgcolor: token('base'), 'font.color': token('ink'),
    'xaxis.gridcolor': token('line'), 'yaxis.gridcolor': token('line'),
  }))
}
const theme = window.parent.document.documentElement.dataset.theme ?? localStorage.getItem('shinka-theme') ?? 'dark'
document.documentElement.dataset.theme = theme

document.addEventListener('DOMContentLoaded', () => {
  window.shinkaSetTheme(theme)
  // Tabs preserve their saved data-tab IDs and click pipeline. Only semantics
  // and keyboard behavior are added; the recovered engine remains state owner.
  for (const selector of ['.left-tab', '.tab', '.meta-sub-tab']) {
    const tabs = [...document.querySelectorAll(selector)]
    if (!tabs.length) continue
    tabs[0].parentElement.setAttribute('role', 'tablist')
    function sync() {
      tabs.forEach((tab) => {
        const active = tab.classList.contains('active')
        tab.setAttribute('role', 'tab')
        tab.setAttribute('aria-selected', String(active))
        tab.tabIndex = active ? 0 : -1
        const panelId = tab.dataset.tab || 'meta-content'
        tab.id ||= `tab-${tab.dataset.tab || tab.dataset.section}`
        tab.setAttribute('aria-controls', panelId)
        const panel = document.getElementById(panelId)
        panel?.setAttribute('role', 'tabpanel')
        if (active) panel?.setAttribute('aria-labelledby', tab.id)
      })
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener('keydown', (event) => {
        const next = { ArrowRight: (index + 1) % tabs.length, ArrowLeft: (index + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1 }[event.key]
        if (next !== undefined) {
          event.preventDefault(); tabs[next].click(); tabs[next].focus(); tabs[next].scrollIntoView({ block: 'nearest', inline: 'nearest' })
        } else if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); tab.click() }
      })
    })
    new MutationObserver(sync).observe(tabs[0].parentElement, { subtree: true, attributes: true, attributeFilter: ['class'] })
    sync()
  }
  document.querySelectorAll('input:not([type="checkbox"]):not([type="range"]), select').forEach((control) => {
    if (!control.labels?.length && !control.hasAttribute('aria-label')) control.setAttribute('aria-label', control.placeholder || control.title || control.id.replaceAll('-', ' '))
  })
  document.documentElement.lang = localStorage.getItem('shinka-lang') || 'zh'
})
