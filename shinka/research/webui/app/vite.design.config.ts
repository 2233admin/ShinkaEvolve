import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const here = (path: string) => fileURLToPath(new URL(path, import.meta.url))

// The design-system preview (design-system/preview.html): served from the webui folder so the page lives in the
// design-system package while it renders the app's own components. Dev server only; nothing is built or shipped.
// React is resolved from the app, because the webui folder itself has no node_modules.
export default defineConfig({
  root: here('..'),
  plugins: [react()],
  cacheDir: here('node_modules/.vite-design'),
  resolve: { alias: { react: here('node_modules/react'), 'react-dom': here('node_modules/react-dom') } },
  server: { open: '/design-system/preview.html' },
})
