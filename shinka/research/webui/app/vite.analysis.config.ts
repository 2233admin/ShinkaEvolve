import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { defineConfig, type Plugin } from 'vite'
import ts from 'typescript'

// Preserve the recovered analysis engine, but compile every HTML insertion
// through the same sanitizer. This boundary is limited to the owned documents.
function safeScript(source: string): string {
  const result = ts.transpileModule(source, {
    reportDiagnostics: true,
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
    transformers: { before: [(context) => {
      const visit: ts.Visitor = (node) => {
        if (ts.isBinaryExpression(node)
          && [ts.SyntaxKind.EqualsToken, ts.SyntaxKind.PlusEqualsToken].includes(node.operatorToken.kind)
          && ts.isPropertyAccessExpression(node.left) && ['innerHTML', 'outerHTML'].includes(node.left.name.text)) {
          const value = ts.visitNode(node.right, visit) as ts.Expression
          const html = node.operatorToken.kind === ts.SyntaxKind.PlusEqualsToken
            ? ts.factory.createBinaryExpression(node.left, ts.SyntaxKind.PlusToken, value) : value
          return ts.factory.updateBinaryExpression(node, node.left, ts.factory.createToken(ts.SyntaxKind.EqualsToken),
            ts.factory.createCallExpression(ts.factory.createIdentifier('ShinkaHTML'), undefined, [html]))
        }
        if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression)
          && node.expression.name.text === 'insertAdjacentHTML' && node.arguments.length === 2) {
          return ts.factory.updateCallExpression(node, node.expression, node.typeArguments,
            [node.arguments[0], ts.factory.createCallExpression(ts.factory.createIdentifier('ShinkaHTML'), undefined, [node.arguments[1]])])
        }
        return ts.visitEachChild(node, visit, context)
      }
      return (root) => ts.visitNode(root, visit) as ts.SourceFile
    }] },
  })
  const errors = result.diagnostics?.filter(item => item.category === ts.DiagnosticCategory.Error) ?? []
  if (errors.length) throw new Error(errors.map(item => ts.flattenDiagnosticMessageText(item.messageText, '\n')).join('\n'))
  return result.outputText
}

function workbenchDocuments(): Plugin {
  return {
    name: 'owned-analysis-documents',
    generateBundle() {
      for (const name of ['viz_tree.html']) {
        const source = readFileSync(new URL(`../analysis/${name}`, import.meta.url), 'utf8')
        const declared = new Set([...source.matchAll(/function\s+([\w$]+)\s*\(/g)].map((m) => m[1]))
        const actions = [...new Set([...source.matchAll(/on(?:click|change|input)="([^"]*)"/g)]
          .flatMap((match) => [...match[1].matchAll(/(?:^|;)\s*(?:window\.)?([\w$]+)\s*\(/g)].map((m) => m[1])))]
          .filter((name) => declared.has(name) || name === 'open')
        let html = source
          .replace(/<link\b[^>]*href="(?:https?:[^\"]*|favicon\.png)"[^>]*>/g, '')
          .replace(/<script\b[^>]*src="https?:[^\"]*"[^>]*>\s*<\/script>/g, '')
          .replace(/<script>([\s\S]*?)<\/script>/g, (_, script: string) => `<script>${safeScript(script)}</script>`)
          .replace(/<img\b[^>]*class="logo"[^>]*>/g, '')
        html = html.replace('</head>', `<link rel="stylesheet" href="/analysis/runtime.css">\n<script>window.ShinkaActions=${JSON.stringify(actions)}</script>\n<script src="/analysis/runtime.iife.js"></script>\n</head>`)
        this.emitFile({ type: 'asset', fileName: name, source: html })
      }
      for (const name of ['i18n_en.json', 'i18n_zh.json']) {
        this.emitFile({ type: 'asset', fileName: name, source: readFileSync(new URL(`../analysis/${name}`, import.meta.url)) })
      }
      this.emitFile({
        type: 'asset', fileName: 'THIRD_PARTY_NOTICES.txt',
        source: readFileSync(new URL('../analysis/licenses/THIRD_PARTY_NOTICES.amended.txt', import.meta.url)),
      })
    },
  }
}

export default defineConfig({
  plugins: [workbenchDocuments()],
  build: {
    outDir: '../dist/analysis', emptyOutDir: true,
    lib: {
      entry: fileURLToPath(new URL('./src/analysis/runtime.js', import.meta.url)),
      name: 'ShinkaAnalysis', formats: ['iife'], fileName: 'runtime', cssFileName: 'runtime',
    },
    rollupOptions: { output: { inlineDynamicImports: true } },
  },
})
