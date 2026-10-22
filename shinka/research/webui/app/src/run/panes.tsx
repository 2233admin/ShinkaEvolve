import { createTwoFilesPatch } from 'diff'
import { useEffect, useMemo, useState, type ReactNode, type RefObject } from 'react'
import { isCorrect, recordedCost, scoreOf } from '../compare/series'
import { Markdown } from '../Markdown'
import { Notice } from '../Notice'
import { fmt, money, when } from '../format'
import { useCopy } from '../i18n/lang'
import { diffRows, initialOf, lineOf, nameOf, patchTypeOf, privateMetrics, type DiffKind, type MetaUpdate, type ProgramDetail, type RunProgram } from './data'
import { highlightLines } from './highlight'
import type { Summary } from './Inspector'
import { runQuery, useJson, type Fetched } from './useRunData'

type Plot = { filename: string; path: string; type?: string; ext?: string }

const isoOf = (seconds: unknown): string | null => (typeof seconds === 'number' && Number.isFinite(seconds) ? new Date(seconds * 1000).toISOString() : null)

/** A metric value as recorded: numbers to four places, anything else as its JSON. */
const valueText = (v: unknown): string => (typeof v === 'number' ? (Number.isInteger(v) ? String(v) : fmt(v, 4)) : typeof v === 'string' ? v : JSON.stringify(v))

/** A program or file name that may break after its underscores, so it wraps instead of being cut. */
export function breakable(name: string): ReactNode[] {
  return name.split(/(?<=_)/).flatMap((part, i) => (i === 0 ? [part] : [<wbr key={i} />, part]))
}

function Failed({ title, error }: { title: string; error: string }) {
  const { t } = useCopy()
  return (
    <div className="ipad">
      <Notice kind="error" label={t('icon.error')} title={title} subtitle={error === 'HTTP 503' ? t('ra.busy') : error} />
    </div>
  )
}

function Toggle({ on, onChange, children }: { on: boolean; onChange: (on: boolean) => void; children: ReactNode }) {
  return (
    <button type="button" className="toggle" aria-pressed={on} onClick={() => onChange(!on)}>
      {children}
    </button>
  )
}

/** Source with a line-number gutter; long lines scroll sideways unless wrapped. */
function CodeView({ lines, wrap }: { lines: string[]; wrap: boolean }) {
  const { t } = useCopy()
  return (
    <div className={wrap ? 'code is-wrap' : 'code'} tabIndex={0} role="region" aria-label={t('ra.codeLabel')} translate="no">
      {lines.map((html, i) => (
        <div className="cl" key={i}>
          <span className="ln" aria-hidden="true">{i + 1}</span>
          {/* highlight.js escapes the source; nothing else reaches this HTML. */}
          <span className="lc" dangerouslySetInnerHTML={{ __html: html || ' ' }} />
        </div>
      ))}
    </div>
  )
}

/** A unified diff by line kind; added and removed lines carry their kind for screen readers too. */
export function DiffView({ unified, wrap, changesOnly }: { unified: string; wrap: boolean; changesOnly: boolean }) {
  const { t } = useCopy()
  const rows = useMemo(() => diffRows(unified), [unified])
  const shown = changesOnly ? rows.filter((r) => r.kind !== 'same') : rows
  const SIGN: Record<DiffKind, string> = { add: t('ra.added'), del: t('ra.removed'), hunk: '', head: '', same: '' }
  return (
    <div className={wrap ? 'code is-diff is-wrap' : 'code is-diff'} tabIndex={0} role="region" aria-label={t('ra.codeLabel')} translate="no">
      {shown.map((r, i) => (
        <div className={`dl is-${r.kind}`} key={i}>
          {SIGN[r.kind] && <span className="sr-only">{SIGN[r.kind]} </span>}
          {r.text || ' '}
        </div>
      ))}
    </div>
  )
}

type CodeProps = { program: RunProgram; detail: Fetched<ProgramDetail>; file: string; wrap: boolean; onWrap: (wrap: boolean) => void }

/** R05: the program's source, highlighted, with copy and download. */
export function CodePane({ detail, file, wrap, onWrap }: CodeProps) {
  const { t } = useCopy()
  const code = detail.data?.code ?? null
  const language = detail.data?.language ?? null
  const lines = useMemo(() => (code ? highlightLines(code, language) : []), [code, language])
  const [copied, setCopied] = useState<'ok' | 'fail' | null>(null)
  useEffect(() => {
    if (copied === null) return
    const timer = window.setTimeout(() => setCopied(null), 1600)
    return () => window.clearTimeout(timer)
  }, [copied])
  if (detail.loading) return <p className="ipad muted">{t('ra.loadingCode')}</p>
  if (detail.error) return <Failed title={t('ra.codeFailed')} error={detail.error} />
  if (!code) return <p className="ipad muted">{t('ra.noCode')}</p>
  const copy = () => navigator.clipboard.writeText(code).then(() => setCopied('ok'), () => setCopied('fail'))
  return (
    <>
      <div className="codebar">
        <span className="codebar-file mono faint">
          <span translate="no">{breakable(file)}</span> <span>· {t('ra.lines', { n: lines.length })}</span>
        </span>
        <span className="codebar-tools">
          <button type="button" className="toggle" onClick={copy}>
            {copied === 'ok' ? t('ra.copied') : copied === 'fail' ? t('ra.copyFailed') : t('ra.copy')}
          </button>
          <Toggle on={wrap} onChange={onWrap}>{t('ra.wrap')}</Toggle>
        </span>
        <span className="sr-only" aria-live="polite">{copied === 'ok' ? t('ra.copied') : copied === 'fail' ? t('ra.copyFailed') : ''}</span>
      </div>
      <CodeView lines={lines} wrap={wrap} />
    </>
  )
}

type DiffProps = {
  run: string
  program: RunProgram
  detail: Fetched<ProgramDetail>
  programs: RunProgram[]
  byId: ReadonlyMap<string, RunProgram>
  wrap: boolean
  onWrap: (wrap: boolean) => void
  again: unknown
}

/** R06: what changed against the parent (the stored diff) or against the initial program (computed here). */
export function DiffPane({ run, program, detail, programs, byId, wrap, onWrap, again }: DiffProps) {
  const { t } = useCopy()
  const [base, setBase] = useState<'parent' | 'initial'>('parent')
  const [changesOnly, setChangesOnly] = useState(false)
  const initial = useMemo(() => initialOf(lineOf(byId, program.id), programs), [byId, program.id, programs])
  const wantInitial = base === 'initial' && initial !== null && initial.id !== program.id
  const initialDetail = useJson<ProgramDetail>(wantInitial ? `/get_program_details?${runQuery(run, { id: initial.id })}` : null, again)
  const code = detail.data?.code ?? null
  const initialCode = initialDetail.data?.code ?? null
  const vsInitial = useMemo(
    () => (wantInitial && code !== null && initialCode !== null ? createTwoFilesPatch('initial', 'current', initialCode, code) : null),
    [wantInitial, code, initialCode],
  )

  let body: ReactNode
  if (detail.loading || (wantInitial && initialDetail.loading)) body = <p className="ipad muted">{t('ra.loadingDiff')}</p>
  else if (detail.error) body = <Failed title={t('ra.diffFailed')} error={detail.error} />
  else if (wantInitial && initialDetail.error) body = <Failed title={t('ra.diffFailed')} error={initialDetail.error} />
  else if (base === 'parent') {
    const stored = detail.data?.code_diff ?? null
    body = stored ? <DiffView unified={stored} wrap={wrap} changesOnly={changesOnly} /> : <p className="ipad muted">{t('ra.noParentDiff')}</p>
  } else if (initial === null) body = <p className="ipad muted">{t('ra.noInitial')}</p>
  else if (initial.id === program.id) body = <p className="ipad muted">{t('ra.isInitial')}</p>
  else if (code === null || initialCode === null) body = <p className="ipad muted">{t('ra.noCode')}</p>
  else if (code === initialCode) body = <p className="ipad muted">{t('ra.sameAsInitial')}</p>
  else body = <DiffView unified={vsInitial ?? ''} wrap={wrap} changesOnly={changesOnly} />

  return (
    <>
      <div className="codebar">
        <span className="seg is-small" role="group" aria-label={t('ra.diffBase')}>
          <button type="button" aria-pressed={base === 'parent'} onClick={() => setBase('parent')}>{t('ra.vsParentDiff')}</button>
          <button type="button" aria-pressed={base === 'initial'} onClick={() => setBase('initial')}>{t('ra.vsInitial')}</button>
        </span>
        <span className="codebar-tools">
          <Toggle on={changesOnly} onChange={setChangesOnly}>{t('ra.changesOnly')}</Toggle>
          <Toggle on={wrap} onChange={onWrap}>{t('ra.wrap')}</Toggle>
        </span>
      </div>
      {base === 'initial' && initial !== null && initial.id !== program.id && (
        <p className="ipad small faint">{t('ra.initialIs', { name: nameOf(initial), id: initial.id.slice(0, 8) })}</p>
      )}
      {body}
    </>
  )
}

function KV({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="kv">
      {rows.map(([k, v]) => (
        <div key={k}>
          <dt className="faint">{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  )
}

function Metrics({ values }: { values: [string, unknown][] }) {
  return (
    <dl className="metrics">
      {values.map(([k, v]) => (
        <div key={k}>
          <dt className="mono faint" translate="no">{k}</dt>
          <dd className="mono">{valueText(v)}</dd>
        </div>
      ))}
    </dl>
  )
}

function Log({ text }: { text: string }) {
  return (
    <pre className="ilog" tabIndex={0} translate="no">
      {text}
    </pre>
  )
}

/** A program id as a button that selects it. */
function ProgramLink({ id, byId, onSelect }: { id: string; byId: ReadonlyMap<string, RunProgram>; onSelect: (id: string) => void }) {
  const p = byId.get(id)
  if (!p) return <span className="mono faint">{id.slice(0, 8)}</span>
  return (
    <button type="button" className="ilink mono" title={nameOf(p)} onClick={() => onSelect(id)}>
      {id.slice(0, 8)}
    </button>
  )
}

type EvalProps = { run: string; program: RunProgram; byId: ReadonlyMap<string, RunProgram>; onSelect: (id: string) => void; again: unknown }

/** R04 and R07: what the evaluator recorded — metrics, feedback, output, images — and the program's record. */
export function EvalPane({ run, program, byId, onSelect, again }: EvalProps) {
  const { t } = useCopy()
  const plots = useJson<Plot[]>(`/get_plots?${runQuery(run, { generation: String(program.generation ?? 0), program_id: program.id })}`, again)
  const md = program.metadata ?? {}
  const pub = Object.entries(program.public_metrics ?? {})
  const privateNote = privateMetrics(program)
  const stdout = typeof md.stdout_log === 'string' && md.stdout_log ? md.stdout_log : null
  const stderr = typeof md.stderr_log === 'string' && md.stderr_log ? md.stderr_log : null
  const analysis = md.code_analysis_metrics && typeof md.code_analysis_metrics === 'object' ? Object.entries(md.code_analysis_metrics as Record<string, unknown>) : []
  const moves = program.migration_history ?? []
  const time = isoOf(program.timestamp)
  return (
    <div className="ibody">
      <h3>{t('ra.metrics')}</h3>
      <Metrics values={[['combined_score', isCorrect(program) ? program.combined_score : t('ra.failed')], ...pub]} />
      {pub.length === 0 && <p className="small muted">{t('ra.noPublic')}</p>}
      <h3>{t('ra.private')}</h3>
      <p className="small muted">{privateNote === 'hidden' ? t('ra.privateHidden') : t('ra.noPrivate')}</p>
      <h3>{t('ra.feedback')}</h3>
      {program.text_feedback ? <Log text={program.text_feedback} /> : <p className="small muted">{t('ra.noFeedback')}</p>}
      {stdout === null && stderr === null && <p className="small muted">{t('ra.noLog')}</p>}
      {stdout !== null && (
        <>
          <h3>{t('ra.stdout')}</h3>
          <Log text={stdout} />
        </>
      )}
      {stderr !== null && (
        <>
          <h3>{t('ra.stderr')}</h3>
          <Log text={stderr} />
        </>
      )}
      <h3>{t('ra.plots')}</h3>
      {plots.loading ? (
        <p className="small muted">{t('state.loading')}</p>
      ) : plots.error ? (
        <p className="small sig">{t('ra.plotsFailed', { error: plots.error })}</p>
      ) : (plots.data ?? []).length === 0 ? (
        <p className="small muted">{t('ra.noPlots')}</p>
      ) : (
        <div className="plots">
          {(plots.data ?? []).map((plot) => {
            const url = `/plot_file/${encodeURIComponent(plot.path)}`
            return (
              <a key={plot.path} href={url} target="_blank" rel="noreferrer" title={t('ra.plotOpen', { file: plot.filename })}>
                <img src={url} alt={plot.filename} loading="lazy" />
                <span className="mono faint small">{plot.filename}</span>
              </a>
            )
          })}
        </div>
      )}
      <h3>{t('ra.codeAnalysis')}</h3>
      {analysis.length > 0 ? (
        <dl className="kv">
          {analysis.map(([k, v]) => (
            <div key={k}>
              <dt className="faint mono" translate="no">{k}</dt>
              <dd className="mono">{typeof v === 'number' ? fmt(v, 2) : valueText(v)}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="small muted">{t('ra.noAnalysis')}</p>
      )}
      <h3>{t('ra.general')}</h3>
      <KV
        rows={[
          [t('ra.id'), <span className="mono" translate="no">{program.id}</span>],
          [t('ra.parentId'), program.parent_id ? <ProgramLink id={program.parent_id} byId={byId} onSelect={onSelect} /> : '—'],
          [t('ra.language'), <span className="mono">{program.language ?? '—'}</span>],
          [t('ra.time'), <span className="mono">{time ? `${when(time)} UTC` : '—'}</span>],
          [t('ra.complexity'), <span className="mono">{typeof program.complexity === 'number' ? fmt(program.complexity, 2) : '—'}</span>],
          [t('ra.children'), <span className="mono">{program.children_count ?? '—'}</span>],
          [t('ra.archive'), program.in_archive ? t('ra.inArchive') : t('ra.notInArchive')],
        ]}
      />
      <h3>{t('ra.migrationHistory')}</h3>
      {moves.length === 0 ? (
        <p className="small muted">{t('ra.noMigration')}</p>
      ) : (
        <table className="itable">
          <thead>
            <tr>
              <th scope="col" className="num">{t('ra.col.gen')}</th>
              <th scope="col">{t('ra.col.move')}</th>
              <th scope="col">{t('ra.time')}</th>
            </tr>
          </thead>
          <tbody>
            {moves.map((m, i) => {
              const at = isoOf(m.timestamp)
              return (
                <tr key={i}>
                  <td className="num mono">{m.generation ?? '—'}</td>
                  <td className="mono">{t('ra.move', { from: m.from ?? '—', to: m.to ?? '—' })}</td>
                  <td className="mono">{at ? when(at) : '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}

// R08: these LLM fields are long and stay folded until opened.
const LONG_LLM = new Set(['content', 'msg', 'system_msg', 'new_msg_history', 'thought'])
// R04: metadata fields the other sections already show.
const SHOWN = new Set(['patch_type', 'patch_name', 'patch_description', 'model_name', 'llm_result', 'thought', 'stdout_log', 'stderr_log', 'code_analysis_metrics', 'api_costs', 'embed_cost', 'novelty_cost', 'meta_cost'])
// R04: the meta-analysis notes and the novelty judge's explanation stay folded.
const FOLDED = new Set(['meta_recommendations', 'meta_summary', 'meta_scratch_pad', 'novelty_explanation'])

function Folded({ text }: { text: string }) {
  const { t } = useCopy()
  return (
    <details className="fold">
      <summary>{t('ra.expand')}</summary>
      <pre className="ilog" translate="no">{text}</pre>
    </details>
  )
}

/** One recorded value: short text inline, long text and nested values folded, objects as JSON (R04-C1, R08). */
function Value({ value, fold }: { value: unknown; fold: boolean }) {
  const json = typeof value === 'object' && value !== null
  const text = json ? JSON.stringify(value, null, 2) : String(value)
  if (fold || text.length > 80 || json) return <Folded text={text} />
  return <span className="mono" translate="no">{text}</span>
}

type LlmProps = { program: RunProgram; byId: ReadonlyMap<string, RunProgram>; onSelect: (id: string) => void }

/** R08 and the rest of R04: the mutation's model, cost, patch, inspirations, reasoning and the full LLM record. */
export function LlmPane({ program, byId, onSelect }: LlmProps) {
  const { t } = useCopy()
  const md = program.metadata ?? {}
  const llm = md.llm_result
  const thought = typeof md.thought === 'string' && md.thought ? md.thought : null
  const part = (key: string) => {
    const v = recordedCost(md[key])
    return v === null ? '—' : money(v)
  }
  const links = (ids: string[] | null | undefined) =>
    ids && ids.length > 0 ? (
      <span className="ilinks">{ids.map((id) => <ProgramLink key={id} id={id} byId={byId} onSelect={onSelect} />)}</span>
    ) : (
      t('ra.none')
    )
  const other = Object.entries(md).filter(([k]) => !SHOWN.has(k))
  return (
    <div className="ibody">
      <KV
        rows={[
          [t('ra.model'), <span className="mono" translate="no">{typeof md.model_name === 'string' ? md.model_name : '—'}</span>],
          [t('ra.cost'), <span className="mono">{t('ra.costParts', { api: part('api_costs'), embed: part('embed_cost'), novelty: part('novelty_cost'), meta: part('meta_cost') })}</span>],
          [t('ra.patch'), <span className="mono" translate="no">{[patchTypeOf(program) ?? '—', nameOf(program)].join(' · ')}</span>],
          [t('ra.patchDescription'), typeof md.patch_description === 'string' && md.patch_description ? md.patch_description : '—'],
          [t('ra.topK'), links(program.top_k_inspiration_ids)],
          [t('ra.archiveInsp'), links(program.archive_inspiration_ids)],
        ]}
      />
      <h3>{t('ra.reasoning')}</h3>
      {thought ? <Markdown text={thought} /> : <p className="small muted">{t('ra.noReasoning')}</p>}
      <h3>{t('ra.llmResult')}</h3>
      {llm === null || llm === undefined ? (
        <p className="small muted">{t('ra.noLlm')}</p>
      ) : typeof llm === 'object' && !Array.isArray(llm) ? (
        <dl className="kv is-wide">
          {Object.entries(llm as Record<string, unknown>).map(([k, v]) => (
            <div key={k}>
              <dt className="faint mono" translate="no">{k}</dt>
              <dd><Value value={v} fold={LONG_LLM.has(k)} /></dd>
            </div>
          ))}
        </dl>
      ) : (
        <pre className="ilog" translate="no">{JSON.stringify(llm, null, 2)}</pre>
      )}
      <h3>{t('ra.otherMeta')}</h3>
      {other.length === 0 ? (
        <p className="small muted">{t('ra.noOtherMeta')}</p>
      ) : (
        <dl className="kv is-wide">
          {other.map(([k, v]) => (
            <div key={k}>
              <dt className="faint mono" translate="no">{k}</dt>
              <dd><Value value={v} fold={FOLDED.has(k)} /></dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  )
}

type MetaProps = { updates: MetaUpdate[]; metaError: string | null; summary: Summary | null; bodyRef: RefObject<HTMLDivElement | null> }

/** OD-3: the program's own meta-analysis summary; what it shows downloads as plain text (OD-12). */
export function MetaPane({ updates, metaError, summary, bodyRef }: MetaProps) {
  const { t } = useCopy()
  if (metaError !== null) return <Failed title={t('ra.metaFailed')} error={metaError} />
  if (updates.length === 0) return <p className="ipad muted">{t('ra.noMeta')}</p>
  if (summary === null || summary.state === 'loading') return <p className="ipad muted">{t('ra.loadingMeta')}</p>
  if (summary.state === 'error') return <Failed title={t('ra.metaFailed')} error={summary.error ?? ''} />
  if (summary.file === null) return <p className="ipad muted">{t('ra.noSummary', { n: summary.checked })}</p>
  return (
    <div className="ibody is-meta" ref={bodyRef}>
      <div className="metahead">
        <b>{t('ra.programSummary')}</b>
        <span className="mono faint">{t('ra.summaryFrom', { file: summary.file.filename, n: summary.file.processed_count })}</span>
      </div>
      {summary.entries.map((entry, i) => (
        <div className="metalist" key={i}>
          <Markdown text={entry} />
        </div>
      ))}
    </div>
  )
}

/** A program's validity in words, for tables and labels. */
export const scoreText = (p: RunProgram, failed: string): string => {
  const s = scoreOf(p)
  return s === null ? failed : fmt(s, 4)
}
