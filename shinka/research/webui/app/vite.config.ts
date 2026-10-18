import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

const TOKENS = fileURLToPath(new URL('../design-system/tokens.json', import.meta.url))

type Token = { name: string; value: string | Record<string, string> }
type Tokens = {
  name: string
  color: { themes: { id: string }[]; tokens: Token[] }
  type: { families: Record<string, string> }
  [family: string]: unknown
}

// 把 design-system/tokens.json 编译成 CSS，形状同 design-pipeline 的 compileTokensCss：
// 第一个主题挂在 :root 与 [data-theme]，其余主题只挂 [data-theme]，然后是长度类 token 与字体栈。
function tokensCss(t: Tokens): string {
  const first = t.color.themes[0].id
  const out = [`/* ${t.name} — generated from tokens.json */`]
  t.color.themes.forEach((theme, i) => {
    out.push(`${i === 0 ? `:root, [data-theme="${theme.id}"]` : `[data-theme="${theme.id}"]`} {`)
    for (const k of t.color.tokens) {
      const v = String(typeof k.value === 'string' ? k.value : (k.value[theme.id] ?? k.value[first]))
      const alias = v.match(/^\{([^{}]+)\}$/)
      out.push(`  --${k.name}: ${alias ? `var(--${alias[1]})` : v};`)
    }
    out.push('}')
  })
  out.push(':root {')
  for (const [key, fam] of Object.entries(t)) {
    if (['name', 'version', 'color', 'type', 'contrast', 'meta', 'shadow'].includes(key)) continue
    for (const k of (fam as { tokens: Token[] }).tokens) out.push(`  --${k.name}: ${k.value};`)
  }
  for (const [key, stack] of Object.entries(t.type.families)) out.push(`  --font-${key}: ${stack};`)
  out.push('}')
  return out.join('\n') + '\n'
}

const VIRTUAL = 'virtual:tokens.css'

export function designTokens(): Plugin {
  return {
    name: 'design-tokens',
    resolveId: (id) => (id === VIRTUAL ? `\0${VIRTUAL}` : null),
    load(id) {
      if (id !== `\0${VIRTUAL}`) return null
      this.addWatchFile(TOKENS)
      return tokensCss(JSON.parse(readFileSync(TOKENS, 'utf8')) as Tokens)
    },
  }
}

export default defineConfig({
  plugins: [react(), designTokens()],
  build: { outDir: '../dist', emptyOutDir: true },
  server: { proxy: { '/api': 'http://127.0.0.1:8765' } },
})
