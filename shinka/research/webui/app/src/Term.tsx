import { DefinitionTooltip } from '@carbon/react'
import { TERMS, type TermKey } from './terms'

export function Term({ k, children }: { k: TermKey; children: string }) {
  return (
    <DefinitionTooltip definition={TERMS[k]} openOnHover>
      {children}
    </DefinitionTooltip>
  )
}
