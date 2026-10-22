import { useEffect, useRef, type ReactNode } from 'react'

type Props = { open: boolean; onClose: () => void; label: string; className: string; children: ReactNode }

/**
 * A native modal dialog: the browser makes the page behind it inert, closes it on Esc and returns
 * focus to where it was. A click on the backdrop closes it too. Content mounts only while open, so
 * each opening starts fresh.
 */
export function Modal({ open, onClose, label, className, children }: Props) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    if (open && !dialog.open) dialog.showModal()
    if (!open && dialog.open) dialog.close()
  }, [open])
  return (
    <dialog
      ref={ref}
      className={className}
      aria-label={label}
      onClose={onClose}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      {open && children}
    </dialog>
  )
}
