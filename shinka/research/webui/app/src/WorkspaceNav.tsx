import { Link, Tag } from '@carbon/react'
import { useCopy } from './i18n/lang'

export type WorkspaceView = 'research' | 'runs' | 'library'

export function WorkspaceNav({ view, pending }: { view: WorkspaceView; pending: number }) {
  const { t } = useCopy()
  return (
    <nav className="workspace-nav" aria-label={t('nav.label')}>
      <Link href="/" aria-current={view === 'research' ? 'page' : undefined}>{t('nav.research')}</Link>
      <Link href="/?view=runs" aria-current={view === 'runs' ? 'page' : undefined}>{t('nav.runs')}</Link>
      <Link href="/?view=library" aria-current={view === 'library' ? 'page' : undefined}>
        {t('nav.library')}
        {pending > 0 && <Tag size="sm" type="gray" title={t('nav.pending', { n: pending })}>{pending}</Tag>}
      </Link>
    </nav>
  )
}
