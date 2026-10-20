import { createContext, Fragment, useCallback, useContext, useEffect, useLayoutEffect, useState, type ReactNode } from 'react'
import { COPY, type CopyKey } from './copy'

export type Lang = 'zh' | 'en'

// Shared with the recovered analysis pages (analysis/accessibility.js and viz_tree.html read the same key).
export const LANG_KEY = 'shinka-lang'

/** The remembered language; Chinese is the default and anything unreadable falls back to it. */
export function storedLang(): Lang {
  try {
    return localStorage.getItem(LANG_KEY) === 'en' ? 'en' : 'zh'
  } catch {
    return 'zh'
  }
}

/** Language state for the app root: applied to <html lang>, remembered, and kept in step with other frames. */
export function useLangState(): [Lang, (lang: Lang) => void] {
  const [lang, setLang] = useState<Lang>(storedLang)
  useLayoutEffect(() => {
    document.documentElement.lang = lang === 'en' ? 'en' : 'zh-CN'
    try {
      localStorage.setItem(LANG_KEY, lang)
    } catch {
      /* Without storage the choice lasts for this page only. */
    }
  }, [lang])
  // The analysis page's own 中/EN button writes the same key from its frame.
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === LANG_KEY) setLang(e.newValue === 'en' ? 'en' : 'zh')
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [])
  return [lang, setLang]
}

const LangContext = createContext<Lang>('zh')
export const LangProvider = LangContext.Provider

type Vars = Record<string, string | number>

function fill(text: string, vars?: Vars): string {
  return vars ? text.replace(/\{(\w+)\}/g, (m, k: string) => (k in vars ? String(vars[k]) : m)) : text
}

/** Plain text for a key in the current language, with {name} placeholders filled. `t` changes only with the language. */
export function useCopy(): { lang: Lang; t: (key: CopyKey, vars?: Vars) => string } {
  const lang = useContext(LangContext)
  const t = useCallback((key: CopyKey, vars?: Vars) => fill(COPY[key][lang], vars), [lang])
  return { lang, t }
}

type Parts = Record<string, (text: string) => ReactNode>

/** A sentence with marked spans: <tag>text</tag> is rendered by parts[tag]; the rest is plain text. */
export function Rich({ k, vars, parts }: { k: CopyKey; vars?: Vars; parts: Parts }) {
  const lang = useContext(LangContext)
  const text = fill(COPY[k][lang], vars)
  const out: ReactNode[] = []
  let last = 0
  for (const m of text.matchAll(/<(\w+)>(.*?)<\/\1>/g)) {
    if (m.index > last) out.push(text.slice(last, m.index))
    const render = parts[m[1]]
    out.push(<Fragment key={m.index}>{render ? render(m[2]) : m[2]}</Fragment>)
    last = m.index + m[0].length
  }
  if (last < text.length) out.push(text.slice(last))
  return <>{out}</>
}
