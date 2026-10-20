import { createRoot } from 'react-dom/client'
import { defineCustomElements } from '@astrouxds/astro-web-components/loader'
import '@fontsource-variable/inter'
import '@fontsource/ibm-plex-mono/400.css'
import '@fontsource/ibm-plex-mono/500.css'
import '@carbon/styles/css/styles.css'
import '@astrouxds/astro-web-components/dist/astro-web-components/astro-web-components.css'
import '../../design-system/components/bundle.css'
import './design/tokens.css'
import './app.css'
import App from './App'

defineCustomElements()
createRoot(document.getElementById('root')!).render(<App />)
