import {
  DataTable, DefinitionTooltip, Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@carbon/react'
import { STATUS, bestIs, fmt, oosOf, trialsOf } from './format'
import { Rich, useCopy } from './i18n/lang'
import { PARTS } from './Term'
import type { Hypothesis } from './types'

type Props = { hyps: Hypothesis[]; selected: string; onSelect: (id: string) => void }

export function HypothesisTable({ hyps, selected, onSelect }: Props) {
  const { lang, t } = useCopy()
  const headers = [
    { key: 'id', header: t('hyp.colId'), isSortable: true },
    { key: 'title', header: t('hyp.colTitle'), isSortable: true },
    { key: 'status', header: t('hyp.colStatus'), isSortable: true },
    { key: 'exps', header: t('hyp.colExperiments'), isSortable: true },
    { key: 'trials', header: <Rich k="hyp.colTrials" parts={PARTS} />, isSortable: false },
    { key: 'best', header: <Rich k="hyp.colBest" parts={PARTS} />, isSortable: false },
    { key: 'oos', header: <Rich k="hyp.colOos" parts={PARTS} />, isSortable: false },
  ]
  // 空值（—）排在最后，不论升降序都不影响其余行的比较。
  const sortRow = (a: unknown, b: unknown, { sortDirection }: { sortDirection: string }) => {
    const d = sortDirection === 'DESC' ? -1 : 1
    if (typeof a === 'number' && typeof b === 'number') return (a - b) * d
    return String(a).localeCompare(String(b), lang) * d
  }
  const byId = new Map(hyps.map((h) => [h.id, h]))
  const rows = hyps.map((h) => ({
    id: h.id,
    title: h.title,
    status: t(STATUS[h.status].key),
    exps: h.experiments.length,
    trials: trialsOf(h),
    best: bestIs(h) ?? -Infinity,
    oos: oosOf(h) ?? -Infinity,
  }))
  return (
    <DataTable rows={rows} headers={headers} sortRow={sortRow} size="lg">
      {({ rows: rs, headers: hs, getTableProps, getHeaderProps, getRowProps }) => (
        <Table {...getTableProps()} aria-label={t('hyp.table')}>
          <TableHead>
            <TableRow>
              {hs.map((h) => {
                const { key, ...p } = getHeaderProps({ header: h, isSortable: h.isSortable })
                return (
                  <TableHeader
                    key={key}
                    {...p}
                    // Carbon's assistive text for the sort button describes the action the click takes.
                    translateWithId={(_id, s) => {
                      const header = String(s?.header ?? '')
                      if (!s?.isSortHeader || s.sortDirection === s.sortStates.NONE) return t('table.sortAsc', { header })
                      return s.sortDirection === s.sortStates.ASC ? t('table.sortDesc', { header }) : t('table.sortNone', { header })
                    }}
                  >
                    {h.header}
                  </TableHeader>
                )
              })}
            </TableRow>
          </TableHead>
          <TableBody>
            {rs.map((r) => {
              const h = byId.get(r.id)!
              const { key, ...p } = getRowProps({ row: r })
              const oos = oosOf(h)
              const best = bestIs(h)
              return (
                <TableRow
                  key={key}
                  {...p}
                  className={`hrow${r.id === selected ? ' is-selected' : ''}`}
                  aria-selected={r.id === selected}
                  tabIndex={0}
                  onClick={() => onSelect(r.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      onSelect(r.id)
                    }
                  }}
                >
                  <TableCell className="mono">{h.id}</TableCell>
                  <TableCell className="title-cell" translate="no">{h.title}</TableCell>
                  <TableCell>
                    <span className="status-cell">
                      <rux-status status={STATUS[h.status].rux} />
                      {t(STATUS[h.status].key)}
                    </span>
                  </TableCell>
                  <TableCell className="mono num">{h.experiments.length}</TableCell>
                  <TableCell className="mono num">{trialsOf(h)}</TableCell>
                  <TableCell className="mono num">{fmt(best)}</TableCell>
                  <TableCell className="mono num">
                    {oos === null ? (
                      <DefinitionTooltip definition={t('term.tbd')} openOnHover>
                        —
                      </DefinitionTooltip>
                    ) : (
                      fmt(oos)
                    )}
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      )}
    </DataTable>
  )
}
