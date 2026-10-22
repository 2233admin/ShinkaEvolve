import english from '../../../analysis/i18n_en.json'
import chinese from '../../../analysis/i18n_zh.json'

// English unless Chinese was chosen (owner, 2026-10-22), as in the shell (i18n/lang.tsx).
let language = localStorage.getItem('shinka-lang') === 'zh' ? 'zh' : 'en'
const dictionaries = { en: english, zh: chinese }
const keys = new Map()
for (const dictionary of Object.values(dictionaries)) {
  for (const [key, text] of Object.entries(dictionary)) keys.set(text.toLowerCase(), key)
}
export function translatedLabel(value) {
  const text = String(value)
  const match = text.match(/^(\s*)(.*?)([:：]?)(\s*)$/s)
  const key = keys.get(match[2].toLowerCase())
  return key ? match[1] + dictionaries[language][key] + match[3] + match[4] : text
}
const wired = new WeakSet()
const selectable = '#tree g.node, .prompt-node, .ens-card, #prompt-description-toggle, .path-flow-node, .path-curve-dot, .path-diff-dot, .path-diff-toggle'
function enhance(root) {
  if (!(root instanceof Element)) return
  for (const label of [root, ...root.querySelectorAll('h1,h2,h3,h4,strong,label,button,option,th')]) {
    if (!label.matches('h1,h2,h3,h4,strong,label,button,option,th') || label.closest('pre,code,textarea')) continue
    for (const node of label.childNodes) {
      if (node.nodeType !== Node.TEXT_NODE) continue
      const translated = translatedLabel(node.textContent)
      if (translated !== node.textContent) node.textContent = translated
    }
  }
  for (const item of [root, ...root.querySelectorAll(selectable)]) {
    if (!item.matches(selectable) || wired.has(item)) continue
    wired.add(item)
    const bound = item.__data__
    const data = bound?.data ?? bound
    if (data?.isUnifiedRoot) {
      item.setAttribute('role', 'img')
      item.setAttribute('aria-label', '所有程序的共同根')
      continue
    }
    item.tabIndex = 0
    item.setAttribute('role', 'button')
    const name = item.textContent.trim().replace(/\s+/g, ' ')
    item.setAttribute('aria-label', item.matches('.prompt-node')
      ? `提示 ${data?.name || data?.id || name}，适应度 ${data?.fitness ?? '—'}`
      : data?.id
        ? `程序 ${data.id}，代数 ${data.generation}，得分 ${data.correct ? data.combined_score ?? '—' : '评估失败'}`
        : item.dataset.nodeId ? `程序 ${item.dataset.nodeId}，${name}` : name || `路径步骤 ${Number(item.dataset.idx) + 1}`)
    item.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        item.dispatchEvent(new MouseEvent('click', { bubbles: true }))
      }
    })
  }
}
window.shinkaSetLanguage = (value) => {
  language = value === 'zh' ? 'zh' : 'en'
  document.documentElement.lang = language
  enhance(document.body)
}
document.addEventListener('DOMContentLoaded', () => {
  enhance(document.body)
  new MutationObserver((records) => {
    for (const record of records) {
      for (const node of record.addedNodes) enhance(node instanceof Element ? node : node.parentElement)
    }
  }).observe(document.body, { childList: true, subtree: true })
})
