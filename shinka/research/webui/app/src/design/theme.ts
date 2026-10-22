import { useLayoutEffect, useState } from 'react'

export type ThemeName = 'dark' | 'light'

// Shared with the recovered analysis runtime (analysis/runtime.js reads the same key).
export const THEME_KEY = 'shinka-theme'

/** The remembered theme; dark is the default and anything unreadable falls back to it. */
export function storedTheme(): ThemeName {
  try {
    return localStorage.getItem(THEME_KEY) === 'light' ? 'light' : 'dark'
  } catch {
    return 'dark'
  }
}

function applyTheme(theme: ThemeName): void {
  document.documentElement.dataset.theme = theme
  try {
    localStorage.setItem(THEME_KEY, theme)
  } catch {
    /* Without storage the choice lasts for this page only. */
  }
}

/** Current theme plus a setter that applies it to <html> and remembers it. */
export function useTheme(): [ThemeName, (theme: ThemeName) => void] {
  const [theme, setTheme] = useState<ThemeName>(storedTheme)
  // Layout effect: applied before children read token values in their passive effects.
  useLayoutEffect(() => applyTheme(theme), [theme])
  return [theme, setTheme]
}
