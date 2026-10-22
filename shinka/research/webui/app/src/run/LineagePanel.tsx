import { createTwoFilesPatch } from 'diff'
import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import { Dialog, Modal, ModalOverlay } from 'react-aria-components'
import { DASHES, isCorrect, scoreOf, stackLabels } from '../compare/series'
import { Icon } from '../design/icons'
import { fmt, int, money, pct, when } from '../format'
import { useCopy } from '../i18n/lang'
import { Markdown } from '../Markdown'
import { Notice } from '../Notice'
import {
  diffRows,
  fertility,
  fitLine,
  generationOf,
  lineOf,
  META_SECTIONS,
  metaSection,
  migrations,
  mutationPoints,
  nameOf,
  population,
  patchTypeOf,
  promptOrder,
  promptPrograms,
  promptTally,
  type MetaContent,
  type MetaUpdate,
  type MutationPoint,
  type Prompt,
  type PromptRow,
  type ProgramDetail,
  type RunProgram,
} from './data'
import { breakable, DiffView } from './panes'
import { shownDelta, signed } from './Inspector'
import { PANEL_LABEL, type PanelSection } from './Lineage'
import { PromptTree } from './PromptTree'
import { runQuery, saveText, shownText, useJson } from './useRunData'

const WIDTH_KEY = 'shinka-lineage-panel'
const MIN_WIDTH = 320
// The comparison page's three patterns plus dash-dot, so the usual four islands each get their own line.
const ISLAND_DASHES = [...DASHES, '8 3 1.5 3']

function storedWidth(): number {
  try {
    const v = Number(localStorage.getItem(WIDTH_KEY))
    return Number.isFinite(v) && v >= MIN_WIDTH ? v : 404
  } catch {
    return 404
  }
}

const isoOf = (seconds: unknown): string | null => (typeof seconds === 'number' && Number.isFinite(seconds) ? new Date(seconds * 1000).toISOString() : null)

function ProgramButton({ p, onSelect }: { p: RunProgram; onSelect: (id: string) => void }) {
  return (
    <button type="button" className="ilink" title={`${nameOf(p)} · ${p.id}`} onClick={() => onSelect(p.id)}>
      <span className="mono pname" translate="no">{breakable(nameOf(p))}</span>
      <span className="mono faint">{p.id.slice(0, 8)}</span>
    </button>
  )
}

function Busy({ error, title }: { error: string; title: string }) {
  const { t } = useCopy()
  return <Notice kind="error" label={t('icon.error')} title={title} subtitle={error === 'HTTP 503' ? t('ra.busy') : error} />
}

type Sort = { key: 'gen' | 'score' | 'delta' | 'beat'; desc: boolean }

/** L02-Q4: the programs a prompt version produced, sortable by generation, score, Δ and Beat%. */
function PromptPrograms({ rows, onSelect, selected }: { rows: PromptRow[]; onSelect: (id: string) => void; selected: string | null }) {
  const { t } = useCopy()
  const [sort, setSort] = useState<Sort>({ key: 'gen', desc: false })
  const value = (r: PromptRow): number | null =>
    sort.key === 'gen' ? generationOf(r.program) : sort.key === 'score' ? scoreOf(r.program) : sort.key === 'delta' ? r.delta : r.beat
  // A row without the value sorts last in both directions.
  const sorted = [...rows].sort((a, b) => {
    const va = value(a)
    const vb = value(b)
    if (va === null || vb === null) return va === null ? (vb === null ? 0 : 1) : -1
    return sort.desc ? vb - va : va - vb
  })
  const header = (key: Sort['key'], label: string) => {
    const on = sort.key === key
    return (
      <th scope="col" className="num" aria-sort={on ? (sort.desc ? 'descending' : 'ascending') : 'none'}>
        <button type="button" className="sortbtn" onClick={() => setSort({ key, desc: on ? !sort.desc : key !== 'gen' })}>
          {label}
          <span aria-hidden="true">{on ? (sort.desc ? ' ↓' : ' ↑') : ''}</span>
        </button>
      </th>
    )
  }
  return (
    <div className="tscroll">
    <table className="itable is-click">
      <thead>
        <tr>
          {header('gen', t('ra.col.gen'))}
          <th scope="col">{t('ra.col.program')}</th>
          {header('score', t('ra.col.score'))}
          {header('delta', 'Δ')}
          {header('beat', 'Beat%')}
          <th scope="col" className="num">{t('ra.col.valid')}</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((r) => {
          const s = scoreOf(r.program)
          return (
            <tr key={r.program.id} className={r.program.id === selected ? 'is-selected' : undefined}>
              <td className="num mono">{generationOf(r.program)}</td>
              <td><ProgramButton p={r.program} onSelect={onSelect} /></td>
              <td className="num mono">{s === null ? <span className="sig">{t('ra.failedShort')}</span> : fmt(s, 4)}</td>
              <td className="num mono">{r.delta === null ? '—' : signed(r.delta, 4)}</td>
              <td className="num mono">{r.beat === null ? '—' : `${fmt(r.beat, 1)}%`}</td>
              <td className="num">{isCorrect(r.program) ? <Icon name="check" /> : <span className="sig"><Icon name="cross" /></span>}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
    </div>
  )
}

/** L02-Q3: a prompt version's full text and its line diff against the parent, in a dialog. */
function PromptText({ prompt, parent, label }: { prompt: Prompt; parent: Prompt | null; label: string }) {
  const { t } = useCopy()
  const [open, setOpen] = useState(false)
  const diff = useMemo(
    () => (parent && open ? diffRows(createTwoFilesPatch(parent.name ?? 'parent', prompt.name ?? 'prompt', parent.prompt_text ?? '', prompt.prompt_text ?? '')) : []),
    [parent, prompt, open],
  )
  return (
    <>
      <button type="button" className="btn is-small" onClick={() => setOpen(true)}>{t('ra.promptTextOpen')}</button>
      <ModalOverlay className="dialog-overlay" isOpen={open} onOpenChange={setOpen} isDismissable>
        <Modal className="dialog">
          <Dialog className="dialog-body" aria-label={label}>
            <header className="dialog-head">
              <h2 className="mono" translate="no">{label}</h2>
              <button type="button" className="btn is-small" onClick={() => setOpen(false)}>{t('ra.close')}</button>
            </header>
            <h3>{t('ra.promptText')}</h3>
            <pre className="ilog is-tall" translate="no">{prompt.prompt_text ?? ''}</pre>
            <h3>{t('ra.promptDiff')}</h3>
            {parent === null ? (
              <p className="small muted">{t('ra.noPromptParent')}</p>
            ) : (
              <div className="code is-diff is-wrap" tabIndex={0} role="region" aria-label={t('ra.promptDiff')} translate="no">
                {diff.map((r, i) => (
                  <div className={`dl is-${r.kind}`} key={i}>{r.text || ' '}</div>
                ))}
              </div>
            )}
          </Dialog>
        </Modal>
      </ModalOverlay>
    </>
  )
}

type PromptsProps = {
  prompts: Prompt[]
  promptsError: string | null
  hasPromptEvo: boolean
  bestPrompts: ReadonlySet<string>
  programs: RunProgram[]
  promptId: string | null
  onPrompt: (id: string | null) => void
  onSelect: (id: string) => void
  selected: string | null
}

/**
 * L02: the prompt tree (parent → child, shape for the patch type, fill for fitness, the best ringed),
 * the same versions as a table, and one version open in detail.
 */
function PromptsSection({ prompts, promptsError, hasPromptEvo, bestPrompts, programs, promptId, onPrompt, onSelect, selected }: PromptsProps) {
  const { t } = useCopy()
  const ordered = promptOrder(prompts)
  const index = new Map(ordered.map((q, k) => [q.id, k]))
  const open = promptId ? (ordered.find((q) => q.id === promptId) ?? null) : null
  const rows = useMemo(() => (open ? promptPrograms(open, programs) : []), [open, programs])
  const detail = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (promptId) detail.current?.scrollIntoView({ block: 'nearest' })
  }, [promptId])
  if (!hasPromptEvo) return <p className="small muted">{t('ra.noPromptEvo')}</p>
  if (promptsError !== null) return <Busy error={promptsError} title={t('ra.promptsFailedTitle')} />
  if (ordered.length === 0) return <p className="small muted">{t('ra.noPromptEvo')}</p>
  const parentLabel = (q: Prompt) => (q.parent_id && index.has(q.parent_id) ? `P${index.get(q.parent_id)}` : '—')
  return (
    <>
      <p className="small faint">{t('ra.promptsNote', { n: ordered.length })}</p>
      <PromptTree prompts={ordered} best={bestPrompts} open={promptId} onOpen={(id) => onPrompt(id === promptId ? null : id)} />
      <div className="tscroll">
      <table className="itable is-click">
        <thead>
          <tr>
            <th scope="col">P</th>
            <th scope="col" className="num">{t('ra.col.gen')}</th>
            <th scope="col">{t('ra.col.parent')}</th>
            <th scope="col">{t('ra.col.type')}</th>
            <th scope="col" className="num">{t('ra.col.programs')}</th>
            <th scope="col" className="num">{t('ra.col.fitness')}</th>
            <th scope="col">{t('ra.col.description')}</th>
          </tr>
        </thead>
        <tbody>
          {ordered.map((q, k) => (
            <tr key={q.id} className={q.id === promptId ? 'is-selected' : undefined}>
              <td>
                <button type="button" className="ilink mono" aria-pressed={q.id === promptId} onClick={() => onPrompt(q.id === promptId ? null : q.id)}>
                  P{k}{bestPrompts.has(q.id) ? ` · ${t('ra.best')}` : ''}
                </button>
              </td>
              <td className="num mono">{q.program_generation ?? '—'}</td>
              <td className="mono faint">{parentLabel(q)}</td>
              <td className="mono" translate="no">{q.patch_type ?? '—'}</td>
              <td className="num mono">{int(q.program_count ?? 0)}</td>
              <td className="num mono">{typeof q.fitness === 'number' ? fmt(q.fitness) : '—'}</td>
              <td className="wrap"><span className="muted" translate="no">{q.description ?? '—'}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
      <div ref={detail}>
        {open && <PromptDetail prompt={open} k={index.get(open.id) ?? 0} parent={open.parent_id ? (ordered.find((q) => q.id === open.parent_id) ?? null) : null} parentLabel={parentLabel(open)} best={bestPrompts.has(open.id)} rows={rows} onSelect={onSelect} selected={selected} />}
      </div>
    </>
  )
}

function PromptDetail({ prompt, k, parent, parentLabel, best, rows, onSelect, selected }: { prompt: Prompt; k: number; parent: Prompt | null; parentLabel: string; best: boolean; rows: PromptRow[]; onSelect: (id: string) => void; selected: string | null }) {
  const { t } = useCopy()
  const tally = promptTally(rows)
  const llm = (prompt.metadata?.llm ?? null) as { model_name?: unknown; cost?: unknown } | null
  const created = isoOf(prompt.timestamp)
  const label = `P${k} ${prompt.name ?? ''}`.trim()
  return (
    <section className="pdetail" aria-label={label}>
      <header className="pdetail-head">
        {best && <span className="bestmark">{t('ra.best')}</span>}
        <h4 className="mono trunc" translate="no">{label}</h4>
        <PromptText prompt={prompt} parent={parent} label={label} />
      </header>
      {prompt.description && <p className="small" translate="no">{prompt.description}</p>}
      <dl className="kv">
        {(
          [
            [t('ra.col.type'), prompt.patch_type ?? '—'],
            [t('ra.promptGen'), prompt.generation ?? '—'],
            [t('ra.programGen'), prompt.program_generation ?? '—'],
            [t('ra.col.parent'), parentLabel],
            [t('ra.archive'), prompt.in_archive ? t('ra.inArchive') : t('ra.notInArchive')],
            [t('ra.storedFitness'), typeof prompt.fitness === 'number' ? fmt(prompt.fitness, 4) : '—'],
            [t('ra.liveFitness'), tally.beat === null ? '—' : fmt(tally.beat / 100, 4)],
            [t('ra.col.programs'), t('ra.correctOf', { c: int(tally.correct), n: int(rows.length) })],
            [t('ra.improvedOf'), t('ra.improvedCount', { i: int(tally.improved), c: int(tally.correct) })],
            [t('ra.totalImprovement'), typeof prompt.total_improvement === 'number' ? fmt(prompt.total_improvement, 4) : '—'],
            [t('ra.model'), typeof llm?.model_name === 'string' ? llm.model_name : '—'],
            [t('ra.cost'), typeof llm?.cost === 'number' ? money(llm.cost) : '—'],
            [t('ra.created'), created ? `${when(created)} UTC` : '—'],
          ] as [string, ReactNode][]
        ).map(([key, v]) => (
          <div key={key}>
            <dt className="faint">{key}</dt>
            <dd className="mono">{v}</dd>
          </div>
        ))}
      </dl>
      <h3>{t('ra.promptPrograms', { n: int(rows.length) })}</h3>
      <p className="small faint">{t('ra.promptAverages', { d: tally.delta === null ? '—' : signed(tally.delta, 4), b: tally.beat === null ? '—' : `${fmt(tally.beat, 1)}%` })}</p>
      {rows.length === 0 ? <p className="small muted">{t('ra.noPromptPrograms')}</p> : <PromptPrograms rows={rows} onSelect={onSelect} selected={selected} />}
    </section>
  )
}

type ConclusionsProps = { run: string; updates: MetaUpdate[]; metaError: string | null; meta: number | null; onMeta: (processed: number) => void; again: unknown }

/** R03 under OD-3: the meta-analysis update's recommendations and global insights; any update can be opened, and what is shown downloads as plain text. */
function Conclusions({ run, updates, metaError, meta, onMeta, again }: ConclusionsProps) {
  const { t } = useCopy()
  const body = useRef<HTMLDivElement>(null)
  const k = meta === null ? updates.length - 1 : updates.findIndex((u) => u.processed === meta)
  const update = updates[k === -1 ? updates.length - 1 : k] ?? null
  const content = useJson<MetaContent>(update ? `/get_meta_content?${runQuery(run, { processed_count: String(update.processed) })}` : null, again)
  if (metaError !== null) return <Busy error={metaError} title={t('ra.metaFailed')} />
  if (update === null) return <p className="small muted">{t('ra.noMeta')}</p>
  const n = updates.indexOf(update) + 1
  const recs = content.data ? metaSection(content.data.content, META_SECTIONS.recommendations) : ''
  const insights = content.data ? metaSection(content.data.content, META_SECTIONS.insights) : ''
  return (
    <>
      <div className="seg is-small is-wrap" role="group" aria-label={t('ra.metaUpdates')}>
        {updates.map((u, i) => (
          <button key={u.processed} type="button" aria-pressed={u === update} onClick={() => onMeta(u.processed)}>
            #{i + 1} · {int(u.processed)}
          </button>
        ))}
      </div>
      {content.data && (
        <button type="button" className="btn is-small dltext" onClick={() => body.current && saveText(shownText(body.current), `${update.file.replace(/\.txt$/, '')}_conclusions.txt`)}>
          <Icon name="download" />
          {t('ra.downloadText')}
        </button>
      )}
      <div className="conclusions" ref={body}>
        <div className="metahead">
          <b>{t('ra.updateN', { k: n })}</b>
          <span className="mono faint">{t('ra.metaUpdate', { file: update.file, n: int(update.processed) })}{update.generation !== null ? ` · ${t('ra.gen', { g: update.generation })}` : ''}</span>
        </div>
        {content.loading ? (
          <p className="small muted">{t('state.loading')}</p>
        ) : content.error ? (
          <Busy error={content.error} title={t('ra.metaFailed')} />
        ) : (
          <>
            <h3>{t('ra.recs')}</h3>
            {recs ? <Markdown text={recs} /> : <p className="small muted">{t('ra.noSection')}</p>}
            <h3>{t('ra.insights')}</h3>
            {insights ? <Markdown text={insights} /> : <p className="small muted">{t('ra.noSection')}</p>}
          </>
        )}
      </div>
    </>
  )
}

/** L07-Q2: each island's population over the generations, one line per island labelled at its end. */
function Population({ programs }: { programs: RunProgram[] }) {
  const { t } = useCopy()
  const series = useMemo(() => population(programs), [programs])
  const last = Math.max(0, ...series.map((s) => s.counts.length - 1))
  const most = Math.max(1, ...series.map((s) => s.counts[s.counts.length - 1] ?? 0))
  const W = 360
  const H = 150
  const m = { l: 34, r: 72, t: 8, b: 24 }
  const x = (g: number) => m.l + (last > 0 ? (g / last) * (W - m.l - m.r) : 0)
  const y = (n: number) => m.t + (1 - n / most) * (H - m.t - m.b)
  const ends = stackLabels(series.map((s) => y(s.counts[s.counts.length - 1] ?? 0)), 12, m.t, H - m.b)
  return (
    <div className="chart-box">
    <svg className="chart" width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={t('ra.populationLabel', { list: series.map((s) => `${s.island === null ? t('ra.noIsland') : t('ra.island', { k: s.island })} ${s.counts[s.counts.length - 1] ?? 0}`).join(', ') })}>
      {[0, most].map((v) => (
        <g key={v}>
          <line x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} stroke="var(--grid)" />
          <text x={m.l - 6} y={y(v) + 3.5} textAnchor="end" className="axis">{int(v)}</text>
        </g>
      ))}
      {series.map((s, k) => (
        <g key={String(s.island)}>
          <path d={s.counts.map((n, g) => `${g ? 'L' : 'M'}${x(g).toFixed(1)},${y(n).toFixed(1)}`).join('')} fill="none" stroke="var(--text)" strokeWidth={1.4} strokeDasharray={ISLAND_DASHES[k % ISLAND_DASHES.length]} />
          <text x={W - m.r + 6} y={ends[k] + 3.5} className="axis">{s.island === null ? t('ra.noIsland') : t('ra.island', { k: s.island })} · {int(s.counts[s.counts.length - 1] ?? 0)}</text>
        </g>
      ))}
      {[0, last].map((g) => (
        <text key={g} x={x(g)} y={H - 8} textAnchor="middle" className="axis">{g}</text>
      ))}
      <text x={(m.l + W - m.r) / 2} y={H - 8} textAnchor="middle" className="axis-title">{t('ra.genAxis')}</text>
    </svg>
    </div>
  )
}

/** L07: how each island's population grew (Q2) and every migration between islands with the program that moved (Q4). */
function Islands({ programs, onSelect }: { programs: RunProgram[]; onSelect: (id: string) => void }) {
  const { t } = useCopy()
  const rows = useMemo(() => migrations(programs), [programs])
  return (
    <>
      <h3>{t('ra.population')}</h3>
      <Population programs={programs} />
      <p className="small faint">{t('ra.populationNote')}</p>
      <h3>{t('ra.migrationsHead')}</h3>
      {rows.length === 0 ? (
        <p className="small muted">{t('ra.noMigrations')}</p>
      ) : (
        <>
          <p className="small faint">{t('ra.migrationsNote', { n: int(rows.length) })}</p>
          <div className="tscroll">
            <table className="itable">
              <thead>
                <tr>
                  <th scope="col" className="num">{t('ra.col.gen')}</th>
                  <th scope="col">{t('ra.col.program')}</th>
                  <th scope="col" className="num">{t('ra.col.score')}</th>
                  <th scope="col">{t('ra.col.move')}</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((m, i) => {
                  const s = scoreOf(m.program)
                  return (
                    <tr key={i}>
                      <td className="num mono">{m.generation ?? '—'}</td>
                      <td><ProgramButton p={m.program} onSelect={onSelect} /></td>
                      <td className="num mono">{s === null ? <span className="sig">{t('ra.failedShort')}</span> : fmt(s, 4)}</td>
                      <td className="mono">{t('ra.move', { from: m.from ?? '—', to: m.to ?? '—' })}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </>
  )
}

/** One step's change against its parent, read when the step is opened (L09-Q3). */
function StepDiff({ run, id, again }: { run: string; id: string; again: unknown }) {
  const { t } = useCopy()
  const detail = useJson<ProgramDetail>(`/get_program_details?${runQuery(run, { id })}`, again)
  if (detail.loading) return <p className="small muted">{t('ra.loadingDiff')}</p>
  if (detail.error) return <Busy error={detail.error} title={t('ra.diffFailed')} />
  const diff = detail.data?.code_diff ?? null
  return diff ? <DiffView unified={diff} wrap={false} changesOnly /> : <p className="small muted">{t('ra.noParentDiff')}</p>
}

type PathProps = { run: string; best: RunProgram[]; byId: ReadonlyMap<string, RunProgram>; selected: string | null; onSelect: (id: string) => void; again: unknown }

/**
 * L09: the chain of edits from the root to the best program, its score climb, and what each step
 * changed. Every step's diff opens and closes on its own; the last step starts open, the middle ones
 * closed. A step, a curve dot or a name selects the program.
 */
function BestPath({ run, best, byId, selected, onSelect, again }: PathProps) {
  const { t } = useCopy()
  const [which, setWhich] = useState(0)
  const target = best[Math.min(which, best.length - 1)] ?? null
  const line = useMemo(() => (target ? lineOf(byId, target.id) : []), [byId, target])
  const [open, setOpen] = useState<ReadonlySet<string>>(new Set())
  useEffect(() => {
    setOpen(new Set(target && line.length > 1 ? [target.id] : []))
  }, [target, line.length])
  if (target === null) return <p className="small muted">{t('ra.noPath')}</p>
  const scores = line.map(scoreOf)
  const valid = scores.filter((s): s is number => s !== null)
  const lo = Math.min(...valid)
  const hi = Math.max(...valid)
  const W = 360
  const H = 72
  const px = (i: number) => 14 + (line.length > 1 ? (i / (line.length - 1)) * (W - 28) : (W - 28) / 2)
  const py = (s: number) => 8 + (hi > lo ? (1 - (s - lo) / (hi - lo)) * (H - 28) : (H - 28) / 2)
  const toggle = (id: string) => setOpen((o) => {
    const next = new Set(o)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    return next
  })
  return (
    <>
      {best.length > 1 && (
        <div className="seg is-small is-wrap" role="group" aria-label={t('ra.tiedBest')}>
          {best.map((b, i) => (
            <button key={b.id} type="button" aria-pressed={i === which} onClick={() => setWhich(i)}>
              {b.id.slice(0, 8)}
            </button>
          ))}
        </div>
      )}
      <p className="small faint">{t('ra.pathSteps', { n: line.length })}</p>
      <div className="chart-box">
      <svg className="chart pathcurve" width="100%" viewBox={`0 0 ${W} ${H}`} role="group" aria-label={t('ra.pathCurve')}>
        <line x1={14} x2={W - 14} y1={H - 12} y2={H - 12} stroke="var(--line)" strokeDasharray="2 3" />
        <path
          d={line.flatMap((_, i) => (scores[i] === null ? [] : [`${i === 0 || scores[i - 1] === null ? 'M' : 'L'}${px(i).toFixed(1)},${py(scores[i] as number).toFixed(1)}`])).join('')}
          fill="none"
          stroke="var(--signal-graphic)"
          strokeWidth={2}
        />
        {line.map((p, i) => {
          const label = `${nameOf(p)} · ${t('ra.gen', { g: generationOf(p) })} · ${scores[i] === null ? t('ra.failed') : fmt(scores[i], 4)}`
          return (
            <g key={p.id} className="mark" role="button" tabIndex={0} aria-label={label} onClick={() => onSelect(p.id)} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), onSelect(p.id))}>
              <title>{label}</title>
              <rect x={px(i) - 8} y={0} width={16} height={H} fill="transparent" />
              {scores[i] === null ? (
                <path d={`M${px(i) - 3},${H - 15}l6,6m0,-6l-6,6`} stroke="var(--signal-graphic)" strokeWidth={1.3} />
              ) : (
                <circle cx={px(i)} cy={py(scores[i] as number)} r={p.id === selected ? 5 : 3.5} fill={i === line.length - 1 ? 'var(--signal-graphic)' : 'var(--bg)'} stroke="var(--signal-graphic)" strokeWidth={2} />
              )}
            </g>
          )
        })}
      </svg>
      </div>
      <ol className="steps">
        {line.map((p, i) => {
          const s = scores[i]
          const before = i > 0 ? scores[i - 1] : null
          const shown = open.has(p.id)
          return (
            <li key={p.id} className={p.id === selected ? 'is-selected' : undefined}>
              <span className="mono faint">{t('ra.gen', { g: generationOf(p) })}</span>
              <ProgramButton p={p} onSelect={onSelect} />
              <span className="mono small step-meta">
                <span className="faint" translate="no">{patchTypeOf(p) ?? '—'}</span>
                {' · '}
                {s === null ? <span className="sig">{t('ra.failedShort')}</span> : fmt(s, 4)}
                {' · '}
                {s !== null && before !== null ? shownDelta(s, before, 4) : '—'}
              </span>
              {i > 0 && (
                <button type="button" className="btn is-small" aria-expanded={shown} onClick={() => toggle(p.id)}>
                  {t(shown ? 'ra.hideDiff' : 'ra.stepDiff')}
                </button>
              )}
              {shown && (
                <div className="step-diff">
                  <StepDiff run={run} id={p.id} again={again} />
                </div>
              )}
            </li>
          )
        })}
      </ol>
    </>
  )
}

/**
 * R09-Q4: mutation size against the change in score, with failures in their own band. Pointing at a
 * point, or moving to it with ← → once the chart has focus, reads it out; a click or Enter opens it
 * in the inspector.
 */
function MutationChart({ points, onSelect }: { points: MutationPoint[]; onSelect: (id: string) => void }) {
  const { t } = useCopy()
  const [active, setActive] = useState<number | null>(null)
  const drawn = points.filter((p) => p.delta !== null)
  const fit = fitLine(drawn.map((p) => ({ x: p.size, y: p.delta as number })))
  const W = 360
  const H = 200
  const m = { l: 44, r: 10, t: 10, b: 46 }
  const maxSize = Math.max(1, ...points.map((p) => p.size))
  const deltas = drawn.map((p) => p.delta as number)
  const dlo = Math.min(0, ...deltas)
  const dhi = Math.max(0, ...deltas)
  const sx = (v: number) => m.l + (v / maxSize) * (W - m.l - m.r)
  const syv = (v: number) => m.t + (dhi > dlo ? 1 - (v - dlo) / (dhi - dlo) : 0.5) * (H - m.t - m.b - 18)
  const band = H - m.b + 4
  // Points in reading order, left to right, so the arrow keys walk along the size axis.
  const at = useMemo(
    () => points.map((p) => ({ p, x: sx(p.size), y: p.delta === null ? band : syv(p.delta) })).sort((a, b) => a.x - b.x || a.y - b.y),
    // The scales follow from the points.
    [points],
  )
  const describe = (p: MutationPoint) =>
    `${nameOf(p.program)} (${p.program.id.slice(0, 8)}) · ${p.delta === null ? t(p.failed === 'self' ? 'ra.failed' : 'ra.parentFailed') : `Δ ${signed(p.delta, 4)}`} · ${t('ra.linesChanged', { n: p.size })} (+${p.added} −${p.deleted} ~${p.modified}) · ${patchTypeOf(p.program) ?? '—'}`
  const hit = (event: { clientX: number; clientY: number; currentTarget: SVGSVGElement }) => {
    const box = event.currentTarget.getBoundingClientRect()
    const k = W / box.width
    const px = (event.clientX - box.left) * k
    const py = (event.clientY - box.top) * k
    let near: number | null = null
    at.forEach((a, i) => {
      if (Math.hypot(a.x - px, a.y - py) <= 8 && (near === null || Math.hypot(a.x - px, a.y - py) < Math.hypot(at[near].x - px, at[near].y - py))) near = i
    })
    return near
  }
  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') return setActive(null)
    if ((event.key === 'Enter' || event.key === ' ') && active !== null) {
      event.preventDefault()
      return onSelect(at[active].p.program.id)
    }
    const end = at.length - 1
    const moves: Record<string, number> = { ArrowRight: (active ?? -1) + 1, ArrowLeft: (active ?? 1) - 1, Home: 0, End: end }
    if (!(event.key in moves)) return
    event.preventDefault()
    setActive(Math.max(0, Math.min(end, moves[event.key])))
  }
  const current = active === null ? null : at[active]
  return (
    <div className="chart-box mchart" tabIndex={0} role="group" aria-label={t('ra.mutation')} aria-describedby="mchart-help" onKeyDown={onKey} onBlur={() => setActive(null)}>
      <svg
        className="chart"
        width="100%"
        viewBox={`0 0 ${W} ${H}`}
        aria-hidden="true"
        onPointerMove={(event) => setActive(hit(event))}
        onPointerLeave={() => setActive(null)}
        onClick={(event) => {
          const i = hit(event)
          if (i !== null) onSelect(at[i].p.program.id)
        }}
      >
        <line x1={m.l} x2={W - m.r} y1={syv(0)} y2={syv(0)} stroke="var(--grid)" />
        <text x={m.l - 6} y={syv(dhi) + 4} textAnchor="end" className="axis">{signed(dhi)}</text>
        <text x={m.l - 6} y={syv(0) + 4} textAnchor="end" className="axis">0</text>
        {dlo < 0 && <text x={m.l - 6} y={syv(dlo) + 4} textAnchor="end" className="axis">{fmt(dlo)}</text>}
        <line x1={m.l} x2={W - m.r} y1={band - 8} y2={band - 8} stroke="var(--line)" strokeDasharray="2 3" />
        <text x={m.l - 6} y={band + 3} textAnchor="end" className="axis">{t('ra.failedShort')}</text>
        {fit && <line x1={sx(0)} x2={sx(maxSize)} y1={syv(fit.intercept)} y2={syv(fit.intercept + fit.slope * maxSize)} stroke="var(--muted)" strokeWidth={1.2} strokeDasharray="4 3" />}
        {at.map(({ p, x, y }) =>
          p.delta === null ? (
            p.failed === 'self' ? (
              <path key={p.program.id} d={`M${x - 3},${y - 3}l6,6m0,-6l-6,6`} stroke="var(--signal-graphic)" strokeWidth={1.3} />
            ) : (
              <circle key={p.program.id} cx={x} cy={y} r={3} fill="none" stroke="var(--muted)" />
            )
          ) : (
            <circle key={p.program.id} cx={x} cy={y} r={3} fill={p.delta > 0 ? 'var(--text)' : 'var(--stroke)'} />
          ),
        )}
        {current && <circle cx={current.x} cy={current.y} r={6.5} fill="none" stroke="var(--text)" strokeWidth={1.5} strokeDasharray="2 2" />}
        <line x1={m.l} x2={W - m.r} y1={H - 26} y2={H - 26} stroke="var(--stroke)" />
        {[0, maxSize / 2, maxSize].map((v) => (
          <text key={v} x={sx(v)} y={H - 12} textAnchor="middle" className="axis">{int(v)}</text>
        ))}
        <text x={W - m.r} y={H - 1} textAnchor="end" className="axis-title">{t('ra.mutationAxis')}</text>
      </svg>
      {current && (
        <div className="lanes-tip mchart-tip" style={{ left: `${(current.x / W) * 100}%`, top: `${(current.y / H) * 100}%` }}>
          <span className="mono" translate="no">{describe(current.p)}</span>
          <span className="faint">{t('ra.pointOpen')}</span>
        </div>
      )}
      <p id="mchart-help" className="sr-only">{t('ra.mutationHelp')}</p>
      <p className="sr-only" aria-live="polite">{current ? describe(current.p) : ''}</p>
      <p className="small faint">{t('ra.mutationNote', { n: int(drawn.length), f: int(points.length - drawn.length) })}{fit ? ` ${t('ra.trend', { s: signed(fit.slope * 100, 4) })}` : ''}</p>
    </div>
  )
}

/** R09-Q11 and R09-Q4: the most fertile parents, and mutation size against the change in score. */
function Offspring({ programs, onSelect }: { programs: RunProgram[]; onSelect: (id: string) => void }) {
  const { t } = useCopy()
  const { top, deadEnds } = useMemo(() => fertility(programs), [programs])
  const points = useMemo(() => mutationPoints(programs), [programs])
  const most = Math.max(1, ...top.map((x) => x.children))
  return (
    <>
      <h3>{t('ra.fertile')}</h3>
      {top.length === 0 ? (
        <p className="small muted">{t('ra.noFertile')}</p>
      ) : (
        <ol className="bars">
          {top.map((x) => (
            <li key={x.program.id}>
              <span className="mono faint">{t('ra.gen', { g: generationOf(x.program) })}</span>
              <ProgramButton p={x.program} onSelect={onSelect} />
              <span className="bar" aria-hidden="true">
                <i style={{ width: `${(x.children / most) * 100}%` }} />
                <b style={{ width: `${(x.improving / most) * 100}%` }} />
              </span>
              <span className="mono small">{t('ra.childrenOf', { c: x.children, i: x.improving, p: pct(x.children ? x.improving / x.children : 0) })}</span>
            </li>
          ))}
        </ol>
      )}
      <p className="small faint">{t('ra.deadEnds', { n: int(deadEnds) })}</p>
      <h3>{t('ra.mutation')}</h3>
      {points.length === 0 ? <p className="small muted">{t('ra.noMutation')}</p> : <MutationChart points={points} onSelect={onSelect} />}
    </>
  )
}

type Props = {
  section: PanelSection
  onClose: () => void
  run: string
  programs: RunProgram[]
  byId: ReadonlyMap<string, RunProgram>
  best: RunProgram[]
  prompts: Prompt[]
  promptsError: string | null
  hasPromptEvo: boolean
  bestPrompts: ReadonlySet<string>
  promptId: string | null
  onPrompt: (id: string | null) => void
  updates: MetaUpdate[]
  metaError: string | null
  meta: number | null
  onMeta: (processed: number) => void
  selected: string | null
  onSelect: (id: string) => void
  again: unknown
  /** The widest the panel may grow inside the canvas. */
  maxWidth: number
}

/** The lineage panel (OD-3, L02, L07, L09, R09-Q4/Q11): resizable, one section open at a time. */
export function LineagePanel({ section, onClose, maxWidth, ...props }: Props) {
  const { t } = useCopy()
  const [width, setWidth] = useState(storedWidth)
  const shown = Math.max(MIN_WIDTH, Math.min(width, maxWidth))
  useEffect(() => {
    try {
      localStorage.setItem(WIDTH_KEY, String(Math.round(width)))
    } catch {
      /* Without storage the width lasts for this page only. */
    }
  }, [width])
  let body: ReactNode
  switch (section) {
    case 'conclusions':
      body = (
        <>
          <Conclusions run={props.run} updates={props.updates} metaError={props.metaError} meta={props.meta} onMeta={props.onMeta} again={props.again} />
          <h3 className="psec">{t('ra.promptsHead')}</h3>
          <PromptsSection prompts={props.prompts} promptsError={props.promptsError} hasPromptEvo={props.hasPromptEvo} bestPrompts={props.bestPrompts} programs={props.programs} promptId={props.promptId} onPrompt={props.onPrompt} onSelect={props.onSelect} selected={props.selected} />
        </>
      )
      break
    case 'islands':
      body = <Islands programs={props.programs} onSelect={props.onSelect} />
      break
    case 'path':
      body = <BestPath run={props.run} best={props.best} byId={props.byId} selected={props.selected} onSelect={props.onSelect} again={props.again} />
      break
    case 'offspring':
      body = <Offspring programs={props.programs} onSelect={props.onSelect} />
      break
  }
  return (
    <aside className="lpanel" style={{ width: shown }} aria-label={t(PANEL_LABEL[section])}>
      <div
        className="lpanel-grip"
        role="separator"
        aria-orientation="vertical"
        aria-label={t('ra.resize')}
        aria-valuemin={MIN_WIDTH}
        aria-valuemax={Math.round(maxWidth)}
        aria-valuenow={Math.round(shown)}
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === 'ArrowLeft') setWidth(shown + 24)
          else if (event.key === 'ArrowRight') setWidth(shown - 24)
          else return
          event.preventDefault()
        }}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId)
        }}
        onPointerMove={(event) => {
          if (!event.currentTarget.hasPointerCapture(event.pointerId)) return
          const right = event.currentTarget.parentElement!.getBoundingClientRect().right
          setWidth(Math.max(MIN_WIDTH, Math.min(maxWidth, right - event.clientX)))
        }}
      />
      <header className="lpanel-head">
        <b>{t(PANEL_LABEL[section])}</b>
        <button type="button" className="btn is-small" onClick={onClose}>{t('ra.close')}</button>
      </header>
      <div className="lpanel-body">{body}</div>
    </aside>
  )
}
