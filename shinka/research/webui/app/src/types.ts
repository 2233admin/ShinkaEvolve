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

// skeptic 判定时实际用的阈值（cli.cmd_skeptic 写入），台账照它比较。
export interface Thresholds {
  dsr_supported: number
  min_positive_fraction: number
  max_pbo: number
}

export interface Experiment {
  id: string
  created: string
  n_evaluated: number
  n_probed: number
  best_is_score: number | null
  spend_usd: number
  first_trial: string | null
  last_trial: string | null
  tree_db: string | null
  skeptic?: Evidence | null
}

export interface Verdict {
  label: Status
  reason: string
  evidence: Evidence & { thresholds: Thresholds }
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

// 整个方向按时间排的每一次试验；score 为空 = 程序失败，record = 到这一次为止的新最好，bar = 此刻的噪声线。
export interface TrialPoint {
  n: number
  time: string
  hypothesis: string
  experiment: string
  generation: number
  score: number | null
  record: boolean
  bar: number
}

export interface LibraryPage {
  name: string
  text: string
}

export interface Library {
  index: string
  log: string
  pending: string[]
  pages: LibraryPage[]
}

export interface Snapshot {
  direction: string
  created: string
  budget_usd: number | null
  spend_usd: number
  trials: number
  lockbox: { opened: boolean; start: string; n_total: number; n_dev: number } | null
  hypotheses: Hypothesis[]
  trial_points: TrialPoint[]
  // 试探（probe）不留程序记录，不画点，但计入噪声线的试验数。
  probes: number
  events: SnapshotEvent[]
  tree_url: string | null
  library: Library | null
}

export interface DirectionSummary {
  id: string
  direction: string
  created: string
  hypotheses: number
  trials: number
  spend_usd: number
  budget_usd: number | null
  lockbox_opened: boolean | null
  error: null
}

export interface DirectionError {
  id: string
  direction: string
  error: string
}

export type DirectionEntry = DirectionSummary | DirectionError

export interface WorkspaceSnapshot {
  name: string
  directions: DirectionEntry[]
  library: Library | null
}
