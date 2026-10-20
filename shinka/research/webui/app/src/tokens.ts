import { useEffect, useState } from 'react'

export type Tokens = Record<'ink' | 'muted' | 'base' | 'line' | 'lineStrong' | 'accent', string>

// Opaque canonical tokens only: mix() below parses #rrggbb, and --line/--line-strong are translucent.
const NAMES: Record<keyof Tokens, string> = {
  ink: '--text', muted: '--muted', base: '--bg', line: '--grid', lineStrong: '--stroke', accent: '--signal-graphic',
}

function read(): Tokens {
  const cs = getComputedStyle(document.documentElement)
  return Object.fromEntries(
    Object.entries(NAMES).map(([k, v]) => [k, cs.getPropertyValue(v).trim()]),
  ) as Tokens
}

// 画布类组件（Carbon Charts）吃不了 CSS 变量，运行时从 design/tokens.css 读成色值；
// 主题切换时重读。useTheme 在 layout effect 里改主题，这里的 passive effect 一定在其后。
export function useTokens(light: boolean): Tokens | null {
  const [t, setT] = useState<Tokens | null>(null)
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setT(read())
  }, [light])
  return t
}

const hex = (c: string): [number, number, number] => {
  const h = c.replace('#', '')
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number]
}

// 把 a 往 b 混 t（0..1），用来得到「压暗」的点色。
export function mix(a: string, b: string, t: number): string {
  const [x, y] = [hex(a), hex(b)]
  return `#${x.map((v, i) => Math.round(v + (y[i] - v) * t).toString(16).padStart(2, '0')).join('')}`
}
