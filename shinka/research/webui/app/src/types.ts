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
  /** False when a call had no price; spend_usd then omits it. */
  spend_known: boolean
  first_trial: string | null
  last_trial: string | null
  analysis_url: string | null
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
  /** null while `withheld`: skeptic and verdict notes quote validation figures until the verdict. */
  note: string | null
  withheld: boolean
}

// 整个方向按时间排的每一次试验；score 为空 = 程序失败，record = 到这一次为止的新最好，bar = 此刻的噪声线。
export interface TrialPoint {
  n: number
  time: string
  hypothesis: string
  experiment: string
  generation: number
  /** Id of the evaluated program in its experiment's run database. */
  program: string
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
  /** Runs whose cost is not an amount; above 0, spend_usd is a lower bound. */
  unknown_cost_runs: number
  trials: number
  lockbox: { opened: boolean; start: string; n_total: number; n_dev: number } | null
  hypotheses: Hypothesis[]
  trial_points: TrialPoint[]
  // 试探（probe）不留程序记录，不画点，但计入噪声线的试验数。
  probes: number
  events: SnapshotEvent[]
  library: Library | null
}

/** The overview's evidence, the same numbers the direction page derives from trial_points. */
export interface DirectionEvidence {
  hypothesis_status: Partial<Record<Status, number>>
  best_score: number | null
  noise_bar: number | null
  evaluated_programs: number
  failed_programs: number
  last_activity: string | null
}

export interface DirectionSummary {
  id: string
  direction: string
  created: string
  hypotheses: number
  trials: number
  spend_usd: number
  unknown_cost_runs: number
  budget_usd: number | null
  lockbox_opened: boolean | null
  evidence: DirectionEvidence
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

/** One run database from /list_databases: its client path, display name and task group. */
export interface RunEntry {
  path: string
  name: string
  task: string
}
