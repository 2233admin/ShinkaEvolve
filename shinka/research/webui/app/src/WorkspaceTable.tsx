import { Link, Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@carbon/react'
import { usd } from './format'
import { Rich, useCopy } from './i18n/lang'
import { routeHref } from './shell/route'
import type { DirectionEntry } from './types'

export function WorkspaceTable({ directions }: { directions: DirectionEntry[] }) {
  const { t } = useCopy()
  if (directions.length === 0) {
    return (
      <p className="muted">
        <Rich k="ws.empty" parts={{ cmd: () => <code className="mono">{t('ws.initCommand')}</code> }} />
      </p>
    )
  }

  return (
    <div className="table-wrap workspace-table">
      <Table size="lg" aria-label={t('ws.table')}>
        <TableHead>
          <TableRow>
            <TableHeader scope="col">{t('ws.colDirection')}</TableHeader>
            <TableHeader scope="col">{t('ws.colHypotheses')}</TableHeader>
            <TableHeader scope="col">{t('ws.colTrials')}</TableHeader>
            <TableHeader scope="col">{t('ws.colSpend')}</TableHeader>
            <TableHeader scope="col">{t('ws.colLockbox')}</TableHeader>
          </TableRow>
        </TableHead>
        <TableBody>
          {directions.map((entry) => (
            <TableRow key={entry.id}>
              <TableCell className="title-cell">
                <span translate="no">
                  {entry.error === null ? (
                    <Link
                      href={routeHref({ view: 'direction', direction: entry.id, hypothesis: null })}
                      aria-label={t('ws.linkLabel', { direction: entry.direction, id: entry.id })}
                    >
                      {entry.direction}
                    </Link>
                  ) : (
                    entry.direction
                  )}
                </span>
                <code className="mono muted workspace-folder" translate="no">{entry.id}</code>
              </TableCell>
              {entry.error === null ? (
                <>
                  <TableCell className="mono num">{entry.hypotheses}</TableCell>
                  <TableCell className="mono num">{entry.trials}</TableCell>
                  <TableCell className="mono num">
                    {usd(entry.spend_usd)} / {entry.budget_usd === null ? t('lockbox.unset') : usd(entry.budget_usd)}
                  </TableCell>
                  <TableCell>
                    {entry.lockbox_opened === null
                      ? t('lockbox.unset')
                      : entry.lockbox_opened ? t('lockbox.opened') : t('lockbox.sealed')}
                  </TableCell>
                </>
              ) : (
                <TableCell colSpan={4} className="workspace-error">{t('ws.readError', { error: entry.error })}</TableCell>
              )}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
