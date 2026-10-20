import { DefinitionTooltip } from '@carbon/react'
import type { KeyboardEvent, ReactNode } from 'react'
import { untagged } from '../demo'
import { Glyph } from '../design/icons'
import { bestIs, fmt, int, oosOf, STATUS, trialsOf, when } from '../format'
import { useCopy } from '../i18n/lang'
import type { CopyKey } from '../i18n/copy'
import type { Hypothesis } from '../types'
import { sortHypotheses, type Sort, type SortKey } from './data'
import { useRemembered } from './remember'

const COLUMNS: { key: SortKey; label: CopyKey; num?: boolean }[] = [
  { key: 'id', label: 'rep.colId' },
  { key: 'title', label: 'rep.colTitle' },
  { key: 'status', label: 'rep.colOutcome' },
  { key: 'experiments', label: 'rep.colExperiments', num: true },
  { key: 'trials', label: 'rep.colTrials', num: true },
  { key: 'best', label: 'rep.colBest', num: true },
  { key: 'oos', label: 'rep.colOos', num: true },
  { key: 'verdict', label: 'rep.colVerdictAt', num: true },
]

/** Arrow keys move between the rows' buttons. */
function moveBetweenRows(event: KeyboardEvent<HTMLDivElement>) {
  if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
  const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('button.hpick')]
  const index = buttons.indexOf(document.activeElement as HTMLButtonElement)
  if (index === -1) return
  event.preventDefault()
  buttons[Math.max(0, Math.min(buttons.length - 1, index + (event.key === 'ArrowDown' ? 1 : -1)))].focus()
}

type Props = {
  hypotheses: Hypothesis[]
  /** The direction's folder ID: the sort is remembered per direction for the rest of the visit. */
  direction: string
  selected: string
  onSelect: (id: string) => void
}

/**
 * Every hypothesis of the direction (P06): outcome, experiments, trials, best in-sample score, the
 * validation Sharpe once there is a verdict, and when the verdict came. A row selects its hypothesis;
 * every column sorts, and a missing value stays last.
 */
export function HypothesisTable({ hypotheses, direction, selected, onSelect }: Props) {
  const { t } = useCopy()
  const [sort, setSort] = useRemembered<Sort | null>(`${direction}/sort`, null)
  const rows = sortHypotheses(hypotheses, sort)
  // Ascending, then descending, then the notebook's own order.
  const cycle = (key: SortKey) => setSort(sort?.key !== key ? { key, descending: false } : sort.descending ? null : { key, descending: true })

  const header = (key: SortKey, label: string): ReactNode => {
    const state = sort?.key === key ? (sort.descending ? 'descending' : 'ascending') : 'none'
    const next = state === 'none' ? 'table.sortAsc' : state === 'ascending' ? 'table.sortDesc' : 'table.sortNone'
    return (
      <button type="button" className="hsort" onClick={() => cycle(key)} aria-label={t(next, { header: label })}>
        {label}
        <span className="hsort-mark" aria-hidden="true">{state === 'ascending' ? '↑' : state === 'descending' ? '↓' : ''}</span>
      </button>
    )
  }

  return (
    <div className="hscroll">
      <div className="htable" role="table" aria-label={t('rep.hypotheses')} onKeyDown={moveBetweenRows}>
        <div className="hrow hhead" role="row">
          {COLUMNS.map((c) => (
            <span
              key={c.key}
              role="columnheader"
              className={c.num ? 'num' : undefined}
              aria-sort={sort?.key === c.key ? (sort.descending ? 'descending' : 'ascending') : undefined}
            >
              {header(c.key, t(c.label))}
            </span>
          ))}
        </div>
        {rows.map((h) => {
          const oos = oosOf(h)
          const isSelected = h.id === selected
          return (
            <div className={isSelected ? 'hrow is-selected' : 'hrow'} role="row" key={h.id}>
              <span role="cell" className="mono" translate="no">{h.id}</span>
              <span role="cell">
                <button type="button" className="hpick" aria-current={isSelected ? 'true' : undefined} onClick={() => onSelect(h.id)} translate="no">
                  {untagged(h.title)}
                </button>
              </span>
              <span role="cell" className="hstat">
                <Glyph kind={h.status} />
                {t(STATUS[h.status].key)}
              </span>
              <span role="cell" className="num mono">{h.experiments.length}</span>
              <span role="cell" className="num mono">{int(trialsOf(h))}</span>
              <span role="cell" className="num mono">{fmt(bestIs(h))}</span>
              <span role="cell" className="num mono">
                {h.verdict ? (
                  fmt(oos)
                ) : (
                  // autoAlign positions the definition against the viewport, so the table's scroll box never clips it.
                  <DefinitionTooltip definition={t('term.tbd')} openOnHover autoAlign>
                    —
                  </DefinitionTooltip>
                )}
              </span>
              <span role="cell" className="num mono faint">{h.verdict ? when(h.verdict.created) : '—'}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
