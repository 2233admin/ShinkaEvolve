export type Status =
  | 'proposed'
  | 'selected'
  | 'running'
  | 'challenged'
  | 'supported'
  | 'refuted'
  | 'inconclusive'
  | 'confirmed'
  | 'unconfirmed'
  | 'dropped'

export type Evidence = Record<string, number | string | null | undefined>

export interface Experiment {
  id: string
  created: string
  n_evaluated: number
  n_probed: number
  best_is_score: number | null
  spend_usd: number
  skeptic?: Evidence | null
}

export interface Verdict {
  label: Status
  reason: string
  evidence: Evidence
  created: string
}

export interface Hypothesis {
  id: string
  title: string
  status: Status
  created: string
  body: string
  experiments: Experiment[]
  verdict: Verdict | null
  confirmation: Evidence | null
}

export interface SnapshotEvent {
  time: string
  hypothesis: string
  experiment: string
  event: string
  note: string
}

export interface Snapshot {
  direction: string
  created: string
  budget_usd: number | null
  spend_usd: number
  trials: number
  lockbox: { opened: boolean; start: string; n_total: number; n_dev: number } | null
  hypotheses: Hypothesis[]
  events: SnapshotEvent[]
  library: { wiki_pages: number; pending: number }
}
