import { Link, Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@carbon/react'
import { usd } from './format'
import type { DirectionEntry } from './types'

export function WorkspaceTable({ directions }: { directions: DirectionEntry[] }) {
  if (directions.length === 0) {
    return (
      <p className="muted">
        还没有研究方向——用 <code className="mono">shinka_research init &lt;workspace&gt;/&lt;direction&gt; --direction "研究方向"</code> 初始化。
      </p>
    )
  }

  return (
    <div className="table-wrap workspace-table">
      <Table size="lg" aria-label="研究方向列表">
        <TableHead>
          <TableRow>
            <TableHeader scope="col">研究方向</TableHeader>
            <TableHeader scope="col">假设数</TableHeader>
            <TableHeader scope="col">试验数</TableHeader>
            <TableHeader scope="col">花费 / 预算</TableHeader>
            <TableHeader scope="col">封存段</TableHeader>
          </TableRow>
        </TableHead>
        <TableBody>
          {directions.map((entry) => (
            <TableRow key={entry.id}>
              <TableCell className="title-cell">
                {entry.error === null ? (
                  <Link
                    href={`/?direction=${encodeURIComponent(entry.id)}`}
                    aria-label={`${entry.direction}（目录：${entry.id}）`}
                  >
                    {entry.direction}
                  </Link>
                ) : (
                  entry.direction
                )}
                <code className="mono muted workspace-folder">{entry.id}</code>
              </TableCell>
              {entry.error === null ? (
                <>
                  <TableCell className="mono num">{entry.hypotheses}</TableCell>
                  <TableCell className="mono num">{entry.trials}</TableCell>
                  <TableCell className="mono num">
                    {usd(entry.spend_usd)} / {entry.budget_usd === null ? '未设' : usd(entry.budget_usd)}
                  </TableCell>
                  <TableCell>
                    {entry.lockbox_opened === null ? '未设' : entry.lockbox_opened ? '已开封' : '封存'}
                  </TableCell>
                </>
              ) : (
                <TableCell colSpan={4} className="workspace-error">读取失败：{entry.error}</TableCell>
              )}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
