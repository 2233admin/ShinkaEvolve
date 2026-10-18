import { Link, Tag } from '@carbon/react'

export type WorkspaceView = 'research' | 'runs' | 'library'

export function WorkspaceNav({ view, pending }: { view: WorkspaceView; pending: number }) {
  return (
    <nav className="workspace-nav" aria-label="工作区导航">
      <Link href="/" aria-current={view === 'research' ? 'page' : undefined}>研究方向</Link>
      <Link href="/?view=runs" aria-current={view === 'runs' ? 'page' : undefined}>全部运行</Link>
      <Link href="/?view=library" aria-current={view === 'library' ? 'page' : undefined}>
        资料库{pending > 0 && <Tag size="sm" type="gray" title={`${pending} 份资料待整理`}>{pending}</Tag>}
      </Link>
    </nav>
  )
}
