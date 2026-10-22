import type { ReactNode } from 'react'
import { Definition } from './Definition'
import type { CopyKey } from './i18n/copy'
import { useCopy } from './i18n/lang'

export type TermKey = { [K in CopyKey]: K extends `term.${infer T}` ? T : never }[CopyKey]

export function Term({ k, children }: { k: TermKey; children: ReactNode }) {
  const { t } = useCopy()
  return (
    <Definition definition={t(`term.${k}`)}>{children}</Definition>
  )
}

const term = (k: TermKey) => (text: string) => <Term k={k}>{text}</Term>

/** Renderers for the marked spans used across the copy table: glossary terms and commands. */
export const PARTS = {
  sharpe: term('sharpe'),
  insample: term('insample'),
  validation: term('validation'),
  validation2: term('validation'),
  sealed: term('sealed'),
  trials: term('trials'),
  noise: term('noise'),
  dsr: term('dsr'),
  pbo: term('pbo'),
  subperiod: term('subperiod'),
  buyhold: term('buyhold'),
  cmd: (text: string) => <code className="mono">{text}</code>,
  cmd2: (text: string) => <code className="mono">{text}</code>,
  code: (text: string) => <code>{text}</code>,
}
