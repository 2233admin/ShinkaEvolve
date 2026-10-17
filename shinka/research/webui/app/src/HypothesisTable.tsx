import {
  DataTable, DefinitionTooltip, Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@carbon/react'
import { STATUS, bestIs, fmt, oosOf, trialsOf } from './format'
import { Term } from './Term'
import { TERMS } from './terms'
import type { Hypothesis } from './types'

type Props = { hyps: Hypothesis[]; selected: string; onSelect: (id: string) => void }

const headers = [
  { key: 'id', header: '编号', isSortable: true },
  { key: 'title', header: '假设', isSortable: true },
  { key: 'status', header: '状态', isSortable: true },
  { key: 'exps', header: '实验数', isSortable: true },
  { key: 'trials', header: <Term k="trials">试验数</Term>, isSortable: false },
  {
    key: 'best',
    header: (
      <>
        最好<Term k="insample">样本内</Term>分数
      </>
    ),
    isSortable: false,
  },
  { key: 'oos', header: <><Term k="validation">验证段</Term> <Term k="sharpe">Sharpe</Term></>, isSortable: false },
]

// 空值（—）排在最后，不论升降序都不影响其余行的比较。
const sortRow = (a: unknown, b: unknown, { sortDirection }: { sortDirection: string }) => {
  const d = sortDirection === 'DESC' ? -1 : 1
  if (typeof a === 'number' && typeof b === 'number') return (a - b) * d
  return String(a).localeCompare(String(b), 'zh') * d
}

export function HypothesisTable({ hyps, selected, onSelect }: Props) {
  const byId = new Map(hyps.map((h) => [h.id, h]))
  const rows = hyps.map((h) => ({
    id: h.id,
    title: h.title,
    status: STATUS[h.status].label,
    exps: h.experiments.length,
    trials: trialsOf(h),
    best: bestIs(h) ?? -Infinity,
    oos: oosOf(h) ?? -Infinity,
  }))

  return (
    <DataTable rows={rows} headers={headers} sortRow={sortRow} size="lg">
      {({ rows: rs, headers: hs, getTableProps, getHeaderProps, getRowProps }) => (
        <Table {...getTableProps()} aria-label="假设列表">
          <TableHead>
            <TableRow>
              {hs.map((h) => {
                const { key, ...p } = getHeaderProps({ header: h, isSortable: h.isSortable })
                return (
                  <TableHeader key={key} {...p}>
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
                  <TableCell className="title-cell">{h.title}</TableCell>
                  <TableCell>
                    <span className="status-cell">
                      <rux-status status={STATUS[h.status].rux} />
                      {STATUS[h.status].label}
                    </span>
                  </TableCell>
                  <TableCell className="mono num">{h.experiments.length}</TableCell>
                  <TableCell className="mono num">{trialsOf(h)}</TableCell>
                  <TableCell className="mono num">{fmt(best)}</TableCell>
                  <TableCell className="mono num">
                    {oos === null ? (
                      <DefinitionTooltip definition={TERMS.tbd} openOnHover>
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
