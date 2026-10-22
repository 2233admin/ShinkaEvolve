import { useEffect, useRef, useState } from 'react'
import { Tab, TabList, TabPanel, Tabs } from 'react-aria-components'
import { programCost, scoreOf } from '../compare/series'
import { Icon } from '../design/icons'
import { fmt, money } from '../format'
import type { CopyKey } from '../i18n/copy'
import { useCopy } from '../i18n/lang'
import { codeFileName, failureReason, generationOf, islandOf, modelOf, nameOf, programSummaries, type MetaContent, type MetaUpdate, type ProgramDetail, type RunProgram } from './data'
import { CodePane, DiffPane, EvalPane, LlmPane, MetaPane } from './panes'
import { fetchJson, runQuery, saveText, shownText, useJson } from './useRunData'

export const PANES = ['code', 'diff', 'evaluation', 'llm', 'meta'] as const
export type Pane = (typeof PANES)[number]
/** The URL's `pane`; anything else is the code. */
export const paneOf = (value: string | null): Pane => PANES.find((p) => p === value) ?? 'code'

const PANE_LABEL: Record<Pane, CopyKey> = {
  code: 'ra.pane.code',
  diff: 'ra.pane.diff',
  evaluation: 'ra.pane.evaluation',
  llm: 'ra.pane.llm',
  meta: 'ra.pane.meta',
}

/** A change in score with its sign: +0.13, −0.04, 0.00. */
export const signed = (v: number, digits = 2): string => (fmt(v, digits) !== fmt(0, digits) && v > 0 ? `+${fmt(v, digits)}` : fmt(v, digits))

/** The change between two shown values, taken at the shown precision so it agrees with both (design.md › 字体与数字). */
export const shownDelta = (a: number, b: number, digits = 2): string => {
  const k = 10 ** digits
  return signed(Math.round(a * k) / k - Math.round(b * k) / k, digits)
}

export type Summary = {
  key: string
  state: 'loading' | 'done' | 'error'
  /** The newest file that holds the program's summary, and the summaries found there. */
  file: MetaContent | null
  entries: string[]
  /** How many files were read before the search ended. */
  checked: number
  error: string | null
}

/**
 * OD-3: the selected program's own summary is its entry in the newest meta-analysis file that has
 * one; the files are read newest first and each only once.
 */
function useMetaSummary(run: string, program: RunProgram | null, updates: MetaUpdate[], again: unknown): Summary | null {
  const key = program ? `${run}|${program.id}|${updates.map((u) => u.processed).join(',')}` : ''
  const [found, setFound] = useState<Summary | null>(null)
  useEffect(() => {
    if (program === null || updates.length === 0) return
    let live = true
    const newestFirst = [...updates].reverse()
    setFound((f) => (f?.key === key ? f : { key, state: 'loading', file: null, entries: [], checked: 0, error: null }))
    ;(async (): Promise<Summary> => {
      let checked = 0
      for (const u of newestFirst) {
        const content = await fetchJson<MetaContent>(`/get_meta_content?${runQuery(run, { processed_count: String(u.processed) })}`)
        checked += 1
        const entries = programSummaries(content.content, program)
        if (entries.length > 0) return { key, state: 'done', file: content, entries, checked, error: null }
      }
      return { key, state: 'done', file: null, entries: [], checked, error: null }
    })().then(
      (result) => live && setFound(result),
      (error: unknown) => live && setFound({ key, state: 'error', file: null, entries: [], checked: 0, error: error instanceof Error ? error.message : String(error) }),
    )
    return () => {
      live = false
    }
    // `key` names the run, the program and the files; `again` retries after a refresh.
  }, [key, again])
  return found?.key === key ? found : null
}

type Props = {
  run: string
  /** The program shown: the selected one, else the run's best. */
  program: RunProgram | null
  /** A selected id the run does not hold. */
  missing: string | null
  programs: RunProgram[]
  byId: ReadonlyMap<string, RunProgram>
  best: ReadonlySet<string>
  pane: Pane
  onPane: (pane: Pane) => void
  onSelect: (id: string) => void
  updates: MetaUpdate[]
  metaError: string | null
  again: unknown
}

/** The resident program inspector (6.1, OD-11): selecting a program anywhere only updates it. */
export function Inspector({ run, program, missing, programs, byId, best, pane, onPane, onSelect, updates, metaError, again }: Props) {
  const { t } = useCopy()
  const [wrap, setWrap] = useState(false)
  const detail = useJson<ProgramDetail>(program ? `/get_program_details?${runQuery(run, { id: program.id })}` : null, again)
  const summary = useMetaSummary(run, program, updates, again)
  const metaBody = useRef<HTMLDivElement>(null)

  if (program === null) {
    return (
      <aside className="inspector" aria-label={t('ra.inspector')}>
        <p className="ipad muted">{missing ? t('ra.missing', { id: missing }) : t('ra.pick')}</p>
      </aside>
    )
  }

  const score = scoreOf(program)
  const parent = program.parent_id ? (byId.get(program.parent_id) ?? null) : null
  const parentScore = parent ? scoreOf(parent) : null
  const cost = programCost(program)
  const island = islandOf(program)
  const model = modelOf(program)
  const code = detail.data?.code ?? null
  const file = codeFileName(detail.data ?? program)
  const found = summary?.file ? summary : null
  const reason = failureReason(program)

  let download = null
  if (pane === 'code' && code) {
    download = (
      <button type="button" className="dlbtn" title={t('ra.downloadCode', { file })} aria-label={t('ra.downloadCode', { file })} onClick={() => saveText(code, file)}>
        <Icon name="download" />
      </button>
    )
  } else if (pane === 'meta' && found?.file) {
    // The file holds exactly the text the pane shows: its heading, its source and the program's summaries.
    const name = `${found.file.filename.replace(/\.txt$/, '')}_${program.id.slice(0, 8)}.txt`
    const label = t('ra.downloadMeta', { file: name })
    download = (
      <button type="button" className="dlbtn" title={label} aria-label={label} onClick={() => metaBody.current && saveText(shownText(metaBody.current), name)}>
        <Icon name="download" />
      </button>
    )
  }

  return (
    <aside className="inspector" aria-label={t('ra.inspector')}>
      {missing && <p className="ipad small sig">{t('ra.missing', { id: missing })}</p>}
      <header className="ihead">
        <div className="iname">
          {best.has(program.id) && <span className="bestmark">{t('ra.best')}</span>}
          <h2 className="mono trunc" translate="no" title={nameOf(program)}>{nameOf(program)}</h2>
          <span className="faint mono" title={program.id}>{program.id.slice(0, 8)}</span>
        </div>
        <div className="iscore">
          {score === null ? (
            <b className="is-failed sig"><Icon name="cross" />{t('ra.failed')}</b>
          ) : (
            <b className="mono" title={String(program.combined_score)}>{fmt(score)}</b>
          )}
          <span className="faint">
            {parent === null ? (
              t('ra.noParent')
            ) : (
              <>
                {t('ra.parent')}{' '}
                <button type="button" className="ilink mono" title={t('ra.openParent', { id: parent.id.slice(0, 8) })} onClick={() => onSelect(parent.id)}>
                  {parentScore === null ? t('ra.failedShort') : fmt(parentScore)}
                </button>
                {score !== null && parentScore !== null && (
                  <>
                    {' · '}{t('ra.vsParent')} <span className="mono">{shownDelta(score, parentScore)}</span>
                  </>
                )}
              </>
            )}
          </span>
        </div>
        {reason !== null && (
          <p className="ireason" title={reason} translate="no">
            {reason}
          </p>
        )}
        <div className="ichips">
          <span>{t('ra.gen', { g: generationOf(program) })}</span>
          <span>{island === null ? t('ra.noIsland') : t('ra.island', { k: island })}</span>
          {model && <span className="mono" translate="no">{model}</span>}
          <span className="mono">{cost === null ? t('ra.costUnknown') : money(cost)}</span>
        </div>
      </header>
      <Tabs className="itabs-root" selectedKey={pane} onSelectionChange={(key) => onPane(paneOf(String(key)))}>
        <div className="itabs">
          <TabList className="itablist" aria-label={t('ra.panes')}>
            {PANES.map((k) => (
              <Tab key={k} id={k} className="itab">
                {t(PANE_LABEL[k])}
              </Tab>
            ))}
          </TabList>
          {download}
        </div>
        <TabPanel id="code" className="ipanel is-code">
          <CodePane program={program} detail={detail} file={file} wrap={wrap} onWrap={setWrap} />
        </TabPanel>
        <TabPanel id="diff" className="ipanel is-code">
          <DiffPane run={run} program={program} detail={detail} programs={programs} byId={byId} wrap={wrap} onWrap={setWrap} again={again} />
        </TabPanel>
        <TabPanel id="evaluation" className="ipanel">
          <EvalPane run={run} program={program} byId={byId} onSelect={onSelect} again={again} />
        </TabPanel>
        <TabPanel id="llm" className="ipanel">
          <LlmPane program={program} byId={byId} onSelect={onSelect} />
        </TabPanel>
        <TabPanel id="meta" className="ipanel">
          <MetaPane updates={updates} metaError={metaError} summary={summary} bodyRef={metaBody} />
        </TabPanel>
      </Tabs>
    </aside>
  )
}
