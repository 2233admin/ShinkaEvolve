import { useEffect, useMemo, useState, type KeyboardEvent } from 'react'
import type { CopyKey } from '../i18n/copy'
import { useCopy } from '../i18n/lang'
import { Modal } from './Modal'

export type PaletteItem = {
  key: string
  group: CopyKey
  label: string
  /** A second line that is also searched: a folder, path or task. */
  detail?: string
  /** Research data (titles, paths) is shown as written. */
  data?: boolean
  act: () => void
}

// One long group (hundreds of runs) must not bury the others; typing narrows it.
const PER_GROUP = 8

function Body({ items, onClose }: { items: PaletteItem[]; onClose: () => void }) {
  const { t } = useCopy()
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const shown = useMemo(() => {
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean)
    const taken = new Map<CopyKey, number>()
    return items.filter((item) => {
      const text = `${item.label} ${item.detail ?? ''}`.toLowerCase()
      if (!terms.every((term) => text.includes(term))) return false
      const n = taken.get(item.group) ?? 0
      taken.set(item.group, n + 1)
      return n < PER_GROUP
    })
  }, [items, query])
  const current = Math.min(active, shown.length - 1)
  useEffect(() => {
    document.getElementById(`palette-${current}`)?.scrollIntoView({ block: 'nearest' })
  }, [current])

  function choose(item: PaletteItem | undefined) {
    if (!item) return
    onClose()
    item.act()
  }
  function onKey(event: KeyboardEvent<HTMLInputElement>) {
    const step = event.key === 'ArrowDown' ? 1 : event.key === 'ArrowUp' ? -1 : 0
    if (step !== 0 && shown.length > 0) {
      event.preventDefault()
      setActive((current + step + shown.length) % shown.length)
    } else if (event.key === 'Enter') {
      event.preventDefault()
      choose(shown[current])
    }
  }

  const groups = [...new Set(shown.map((item) => item.group))]
  return (
    <div className="palette-body">
      <input
        className="palette-input"
        type="text"
        role="combobox"
        aria-expanded="true"
        aria-controls="palette-list"
        aria-autocomplete="list"
        aria-activedescendant={current >= 0 ? `palette-${current}` : undefined}
        aria-label={t('pal.label')}
        placeholder={t('pal.input')}
        value={query}
        onChange={(event) => {
          setQuery(event.target.value)
          setActive(0)
        }}
        onKeyDown={onKey}
        autoFocus
      />
      <div id="palette-list" className="palette-list" role="listbox" aria-label={t('pal.label')}>
        {shown.length === 0 && <p className="palette-empty">{t('pal.empty')}</p>}
        {groups.map((group) => (
          <div key={group} role="group" aria-labelledby={`palette-group-${group}`}>
            <div className="palette-group" id={`palette-group-${group}`} role="presentation">{t(group)}</div>
            {shown.map((item, index) =>
              item.group !== group ? null : (
                <div
                  key={item.key}
                  id={`palette-${index}`}
                  role="option"
                  aria-selected={index === current}
                  className={index === current ? 'palette-item is-active' : 'palette-item'}
                  onMouseMove={() => setActive(index)}
                  onClick={() => choose(item)}
                >
                  <span className="trunc" translate={item.data ? 'no' : undefined}>{item.label}</span>
                  {item.detail && <span className="palette-detail trunc mono" translate="no">{item.detail}</span>}
                </div>
              ),
            )}
          </div>
        ))}
      </div>
      <p className="palette-keys">{t('pal.keys')}</p>
    </div>
  )
}

/** Ctrl K: jump to any page, direction, run, program of the open run or library page, or switch theme and language. */
export function Palette({ open, onClose, items }: { open: boolean; onClose: () => void; items: PaletteItem[] }) {
  const { t } = useCopy()
  return (
    <Modal open={open} onClose={onClose} label={t('pal.label')} className="palette">
      <Body items={items} onClose={onClose} />
    </Modal>
  )
}
