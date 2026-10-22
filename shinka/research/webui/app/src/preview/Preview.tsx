/**
 * The design-system preview (award-redesign 7.2, `design-system/preview.html`): the tokens of both themes beside each
 * other, then the app's own components on demo data, so what is previewed is what ships. `pnpm design` serves it;
 * it is never part of the production build. The theme and language switches act on the whole page.
 */
import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Definition } from '../Definition'
import { Notice } from '../Notice'
import { useTheme } from '../design/theme'
import { LangProvider, useCopy, useLangState } from '../i18n/lang'
import { Report } from '../report/Report'
import { bestOf, bestPrompts, generationOf, metaUpdates, type Prompt, type RunProgram } from '../run/data'
import { Lineage } from '../run/Lineage'
import { Progress } from '../run/Progress'
import { noiseByGeneration } from '../run/progressData'
import { ViewTabs } from '../run/RunPage'
import { Scrubber } from '../run/Scrubber'
import type { RunStats } from '../runs/data'
import { runRoute, type ViewTab } from '../shell/route'
import type { Snapshot } from '../types'
import fixture from './fixture.json'

const data = fixture as unknown as { source: string; run: string; snap: Snapshot; programs: RunProgram[]; stats: RunStats; prompts: Prompt[] }

const COLORS = ['--bg', '--rail', '--raised', '--hover', '--selected', '--line', '--line-strong', '--text', '--muted', '--faint', '--signal', '--signal-graphic', '--stroke', '--grid', '--notice-error-bg', '--ramp-0', '--ramp-1']
// Text-bearing tokens and the grounds they sit on: the contrast the design rules promise (design.md › 设计系统).
const TEXTS = ['--text', '--muted', '--faint', '--signal']
const GROUNDS = ['--bg', '--raised', '--hover']
const TYPES = ['--type-headline', '--type-title', '--type-section', '--type-body', '--type-small']
const SPACES = ['--space-1', '--space-2', '--space-3', '--space-4', '--space-5', '--space-6', '--space-8', '--space-10']

function rgb(value: string): [number, number, number] | null {
  const hex = /^#([0-9a-f]{6})$/i.exec(value.trim())
  if (hex) return [0, 2, 4].map((i) => parseInt(hex[1].slice(i, i + 2), 16)) as [number, number, number]
  return null
}

function contrast(a: string, b: string): string {
  const [x, y] = [rgb(a), rgb(b)]
  if (!x || !y) return '—'
  const lum = (c: [number, number, number]) => {
    const [r, g, bl] = c.map((v) => {
      const s = v / 255
      return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
    })
    return 0.2126 * r + 0.7152 * g + 0.0722 * bl
  }
  const [hi, lo] = [lum(x), lum(y)].sort((m, n) => n - m)
  return `${((hi + 0.05) / (lo + 0.05)).toFixed(2)}:1`
}

/** One theme's tokens, read back from the cascade so the table always shows what tokens.css defines. */
function ThemeTokens({ theme }: { theme: 'dark' | 'light' }) {
  const box = useRef<HTMLDivElement>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  useLayoutEffect(() => {
    const style = getComputedStyle(box.current!)
    setValues(Object.fromEntries([...COLORS, ...TYPES, ...SPACES, '--radius-control', '--radius-float', '--focus-ring'].map((k) => [k, style.getPropertyValue(k).trim()])))
  }, [theme])
  return (
    <div ref={box} data-theme={theme} className="pv-theme">
      <h3>{theme}</h3>
      <table className="pv-table">
        <tbody>
          {COLORS.map((k) => (
            <tr key={k}>
              <td><span className="pv-swatch" style={{ background: `var(${k})` }} /></td>
              <td className="mono">{k}</td>
              <td className="mono faint">{values[k]}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h4>contrast</h4>
      <table className="pv-table">
        <thead>
          <tr><th />{GROUNDS.map((g) => <th key={g} className="mono">{g}</th>)}</tr>
        </thead>
        <tbody>
          {TEXTS.map((text) => (
            <tr key={text}>
              <td className="mono" style={{ color: `var(${text})` }}>{text}</td>
              {GROUNDS.map((g) => <td key={g} className="mono" style={{ background: `var(${g})`, color: `var(${text})` }}>{contrast(values[text] ?? '', values[g] ?? '')}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      <h4>type</h4>
      {TYPES.map((k) => <p key={k} style={{ font: `var(${k})` }}>{k.slice(7)} · 研究方向 Directions 1.0213</p>)}
      <h4>space · radius · focus</h4>
      <div className="pv-row">{SPACES.map((k) => <span key={k} className="pv-space" style={{ width: `var(${k})` }} title={`${k} ${values[k]}`} />)}</div>
      <div className="pv-row">
        <span className="pv-box" style={{ borderRadius: 'var(--radius-control)' }}>control</span>
        <span className="pv-box" style={{ borderRadius: 'var(--radius-float)' }}>float</span>
        <span className="pv-box" style={{ outline: 'var(--focus-ring)', outlineOffset: 2 }}>focus</span>
      </div>
    </div>
  )
}

function Body({ onLang }: { onLang: (lang: 'zh' | 'en') => void }) {
  const { lang, t } = useCopy()
  const [theme, setTheme] = useTheme()
  const { snap, programs, stats, prompts } = data
  const [current, setCurrent] = useState(snap.hypotheses[0].id)
  const byId = useMemo(() => new Map(programs.map((p) => [p.id, p])), [programs])
  const best = useMemo(() => bestOf(programs), [programs])
  const maxGen = programs.reduce((m, p) => Math.max(m, generationOf(p)), 0)
  const [upTo, setUpTo] = useState(maxGen)
  const [selected, setSelected] = useState<string | null>(best[0]?.id ?? null)
  const shown = selected ? byId.get(selected) ?? null : null
  const noise = useMemo(() => noiseByGeneration(snap.trial_points, new Set(programs.map((p) => p.id))), [snap, programs])
  const count = programs.filter((p) => generationOf(p) <= upTo).length
  const route = runRoute(data.run)
  const hypothesis = snap.hypotheses.find((h) => h.id === current) ?? snap.hypotheses[0]
  return (
    <main className="pv">
      <header className="pv-head">
        <h1>Shinka Research · design system</h1>
        <p className="muted">{data.source} · {data.run}</p>
        <div className="pv-row">
          <button type="button" className="pv-btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>{theme === 'dark' ? t('pal.toLight') : t('pal.toDark')}</button>
          <button type="button" className="pv-btn" onClick={() => onLang(lang === 'zh' ? 'en' : 'zh')}>{lang === 'zh' ? t('pal.toEn') : t('pal.toZh')}</button>
        </div>
      </header>

      <section className="pv-sec">
        <h2>Tokens</h2>
        <div className="pv-themes">
          <ThemeTokens theme="dark" />
          <ThemeTokens theme="light" />
        </div>
      </section>

      <section className="pv-sec">
        <h2>Component states</h2>
        <h3>View tabs: each tab current</h3>
        {(['lineage', 'progress', 'space', 'contest'] satisfies ViewTab[]).map((tab) => (
          <div key={tab} className="top pv-tabs"><ViewTabs route={{ ...route, tab }} /></div>
        ))}
        <h3>Notices</h3>
        <div className="pv-stack">
          <Notice kind="info" label={t('icon.info')} title={t('empty.hypotheses')} subtitle={t('empty.hypothesesHint')} />
          <Notice kind="error" label={t('icon.error')} title={t('error.read')} subtitle={t('error.readHint', { error: 'HTTP 500' })} />
          <p className="muted">{t('dir.noEvents')}</p>
        </div>
        <h3>Term</h3>
        <p>
          <Definition definition={t('term.dsr')}>DSR</Definition> · <Definition definition={t('term.noise')}>{t('rep.noise')}</Definition>
        </p>
      </section>

      <section className="pv-sec">
        <h2>Report: hypothesis table, trials against the noise line, ledger, verdict, timeline, events</h2>
        <div className="page report-page">
          <Report snap={snap} direction="preview" current={hypothesis} onSelect={setCurrent} onOpen={() => undefined} />
        </div>
      </section>

      <section className="pv-sec">
        <h2>Lineage and progress (generation ≤ {maxGen})</h2>
        <div className="page analysis-page pv-canvas">
          <div className="analysis">
            <section className="canvas" aria-label={t('ra.view.lineage')}>
              <Lineage
                programs={programs}
                byId={byId}
                upTo={upTo}
                maxGen={maxGen}
                selected={shown}
                onSelect={setSelected}
                prompts={prompts}
                promptsError={null}
                hasPromptEvo={stats.has_prompt_evo}
                bestPrompts={bestPrompts(prompts)}
                updates={metaUpdates([], programs)}
                metaError={null}
                panel={null}
                onPanel={() => undefined}
                panelNode={null}
                scrubber={<Scrubber max={maxGen} value={upTo} onChange={setUpTo} count={count} left={128} right={28} />}
              />
            </section>
          </div>
        </div>
        <div className="page analysis-page pv-canvas">
          <div className="analysis">
            <section className="canvas" aria-label={t('ra.view.progress')}>
              <Progress
                programs={programs}
                byId={byId}
                upTo={upTo}
                maxGen={maxGen}
                selected={shown}
                onSelect={setSelected}
                prompts={prompts}
                promptsError={null}
                stats={stats}
                noise={noise}
                scrub={{ max: maxGen, value: upTo, onChange: setUpTo, count }}
              />
            </section>
          </div>
        </div>
      </section>
    </main>
  )
}

export function Preview() {
  const [lang, setLang] = useLangState()
  return (
    <LangProvider value={lang}>
      <Body onLang={setLang} />
    </LangProvider>
  )
}
