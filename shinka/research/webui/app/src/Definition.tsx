import { useState, type ReactNode } from 'react'
import { Button, Tooltip, TooltipTrigger } from 'react-aria-components'

/**
 * A word the page explains in place (design-system Term). react-aria opens the definition on hover and keyboard
 * focus and closes it on Escape, blur and press, in a portal placed against the viewport, so a scrolling table
 * never clips it. It never opens on touch, so a tap opens it here. The inline-block box around the button keeps
 * the line's height and baseline wherever the term sits, as Carbon's popover container did.
 */
export function Definition({ definition, children }: { definition: string; children: ReactNode }) {
  const [open, setOpen] = useState(false)
  return (
    <span className="term-box">
      <TooltipTrigger delay={0} closeDelay={0} isOpen={open} onOpenChange={setOpen}>
        <Button className="term" onPress={(e) => e.pointerType === 'touch' && setOpen(true)}>
          {children}
        </Button>
        <Tooltip className="term-pop" placement="bottom" offset={8}>
          {definition}
        </Tooltip>
      </TooltipTrigger>
    </span>
  )
}
