/**
 * The progress view's runtime timeline and throughput numbers (award-redesign 6.3), ported from the
 * research tree's runtime timeline and throughput tab (analysis/viz_tree.html, cited as VT:line). Data
 * only: the parent draws. Times are seconds since `RuntimeRows.origin`, the earliest pipeline start of
 * the run's runtime rows (the left edge of the legacy timeline); add `origin` back for wall-clock time.
 */
import { isCorrect, isIslandCopy, scoreOf } from '../compare/series.ts'
import { generationOf, modelOf, nameOf, type RunProgram } from './data.ts'

export type Stage = 'sampling' | 'evaluation' | 'postprocess'
export const STAGES: readonly Stage[] = ['sampling', 'evaluation', 'postprocess']

/** [start, end] in seconds since the origin. */
export type Span = [number, number]

export type Attempts = { novelty: number | null; resample: number | null; patch: number | null }

/** One evaluated job with every stage timestamp recorded. */
export type RuntimeRow = {
  id: string
  generation: number
  isIslandCopy: boolean
  correct: boolean
  /** The valid score (`scoreOf`); only the copy priority reads it. */
  score: number | null
  name: string
  model: string | null
  attempts: Attempts
  /** `timeline_lane_mode === 'pool_slots'`: the runner recorded worker slots. */
  poolSlots: boolean
  workers: Record<Stage, number | null>
  workerCapacities: Record<Stage, number | null>
  /** The pipeline runs from its start to the postprocess finish. In pool rows the stage spans are reconciled. */
  stages: Record<'pipeline' | Stage, Span>
  /** Each stage's recorded duration, max(0, finish − start), taken before any reconciliation. */
  seconds: Record<Stage, number>
}

/** `origin` is epoch seconds, null when no program has a runtime row. */
export type RuntimeRows = { origin: number | null; rows: RuntimeRow[] }

/** VT:16517-16521 (parseRuntimeNumber): a number, or a string read with parseFloat; '' and null are unknown. */
function runtimeNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === '') return null
  const parsed = typeof value === 'number' ? value : parseFloat(String(value))
  return Number.isFinite(parsed) ? parsed : null
}

/**
 * VT:16523-16529 (getRuntimeTimelineRowPriority): which of a job's rows the timeline keeps. A non-copy
 * outranks a copy, then a correct row, then a scored one. Legacy counted any finite combined_score and
 * any truthy `correct`; here a score is a valid score and correct is `isCorrect` (contract D-L01b).
 */
export function runtimeRowPriority(row: Pick<RuntimeRow, 'isIslandCopy' | 'correct' | 'score'>): number {
  let priority = 0
  if (!row.isIslandCopy) priority += 10
  if (row.correct) priority += 2
  if (row.score !== null) priority += 1
  return priority
}

const shift = ([start, end]: Span, origin: number): Span => [start - origin, end - origin]

/**
 * VT:16531-16611 (getBaseRuntimeTimelineRows): one row per program with all seven stage timestamps
 * (metadata `pipeline_started_at` and `{sampling,evaluation,postprocess}_{started,finished}_at`), one
 * per job: rows sharing `source_job_id` (else their id) keep the highest priority, an equal priority the
 * earlier program timestamp (a missing one counts as 0), else the first seen; rows keep first-seen order.
 */
export function runtimeRows(programs: readonly RunProgram[]): RuntimeRows {
  const kept = new Map<unknown, { row: RuntimeRow; priority: number; timestamp: number }>()
  for (const p of programs) {
    const md = p.metadata ?? {}
    const at = (key: string) => runtimeNumber(md[key])
    const pipelineStart = at('pipeline_started_at')
    const samplingStart = at('sampling_started_at')
    const samplingEnd = at('sampling_finished_at')
    const evaluationStart = at('evaluation_started_at')
    const evaluationEnd = at('evaluation_finished_at')
    const postprocessStart = at('postprocess_started_at')
    const postprocessEnd = at('postprocess_finished_at')
    if (
      pipelineStart === null || samplingStart === null || samplingEnd === null || evaluationStart === null ||
      evaluationEnd === null || postprocessStart === null || postprocessEnd === null
    ) continue
    const row: RuntimeRow = {
      id: p.id,
      generation: generationOf(p),
      isIslandCopy: isIslandCopy(p),
      correct: isCorrect(p),
      score: scoreOf(p),
      name: nameOf(p),
      model: modelOf(p),
      attempts: { novelty: at('novelty_attempt'), resample: at('resample_attempt'), patch: at('patch_attempt') },
      poolSlots: md.timeline_lane_mode === 'pool_slots',
      workers: { sampling: at('sampling_worker_id'), evaluation: at('evaluation_worker_id'), postprocess: at('postprocess_worker_id') },
      workerCapacities: {
        sampling: at('sampling_worker_capacity'),
        evaluation: at('evaluation_worker_capacity'),
        postprocess: at('postprocess_worker_capacity'),
      },
      stages: {
        pipeline: [pipelineStart, postprocessEnd],
        sampling: [samplingStart, samplingEnd],
        evaluation: [evaluationStart, evaluationEnd],
        postprocess: [postprocessStart, postprocessEnd],
      },
      seconds: {
        sampling: Math.max(0, samplingEnd - samplingStart),
        evaluation: Math.max(0, evaluationEnd - evaluationStart),
        postprocess: Math.max(0, postprocessEnd - postprocessStart),
      },
    }
    const priority = runtimeRowPriority(row)
    const timestamp = typeof p.timestamp === 'number' ? p.timestamp : 0
    const key = md.source_job_id || p.id
    const existing = kept.get(key)
    if (!existing || priority > existing.priority || (priority === existing.priority && timestamp < existing.timestamp)) {
      kept.set(key, { row, priority, timestamp })
    }
  }
  if (kept.size === 0) return { origin: null, rows: [] }
  let origin = Infinity
  for (const { row } of kept.values()) origin = Math.min(origin, row.stages.pipeline[0])
  const rows = [...kept.values()].map(({ row }) => ({
    ...row,
    stages: {
      pipeline: shift(row.stages.pipeline, origin),
      sampling: shift(row.stages.sampling, origin),
      evaluation: shift(row.stages.evaluation, origin),
      postprocess: shift(row.stages.postprocess, origin),
    },
  }))
  return { origin, rows }
}

export type LaneRow = RuntimeRow & { lane: number }

/**
 * VT:16677-16705 (getLegacyRuntimeTimelineRows): lanes inferred from timestamps alone. Rows by
 * evaluation start, then generation, then pipeline start; each takes the first lane whose last
 * evaluation finished by its evaluation start (1e-6 s slack), else a new lane. Lane 0 is legacy "Worker 1".
 */
export function legacyRuntimeLanes(rows: readonly RuntimeRow[]): { rows: LaneRow[]; laneCount: number } {
  const sorted = [...rows].sort((a, b) => {
    if (a.stages.evaluation[0] !== b.stages.evaluation[0]) return a.stages.evaluation[0] - b.stages.evaluation[0]
    if (a.generation !== b.generation) return a.generation - b.generation
    return a.stages.pipeline[0] - b.stages.pipeline[0]
  })
  const laneEnds: number[] = []
  const laned = sorted.map((row) => {
    let lane = laneEnds.findIndex((end) => row.stages.evaluation[0] >= end - 1e-6)
    if (lane === -1) {
      lane = laneEnds.length
      laneEnds.push(row.stages.evaluation[1])
    } else {
      laneEnds[lane] = row.stages.evaluation[1]
    }
    return { ...row, lane }
  })
  return { rows: laned, laneCount: laneEnds.length }
}

/**
 * VT:16637-16675 (reconcileStageWorkerOverlaps): per stage, a worker's jobs (by start, finish,
 * generation, pipeline start) cannot overlap: a job starts no earlier than the worker's previous finish
 * and finishes no earlier than it starts. Durations (`seconds`) are left as recorded.
 */
function reconcile(rows: readonly RuntimeRow[]): RuntimeRow[] {
  const out = rows.map((row) => ({ ...row, stages: { ...row.stages } }))
  for (const stage of STAGES) {
    const valid = out
      .filter((row) => row.workers[stage] !== null)
      .sort((a, b) => {
        const wa = a.workers[stage] as number
        const wb = b.workers[stage] as number
        if (wa !== wb) return wa - wb
        if (a.stages[stage][0] !== b.stages[stage][0]) return a.stages[stage][0] - b.stages[stage][0]
        if (a.stages[stage][1] !== b.stages[stage][1]) return a.stages[stage][1] - b.stages[stage][1]
        if (a.generation !== b.generation) return a.generation - b.generation
        return a.stages.pipeline[0] - b.stages.pipeline[0]
      })
    const previousEnd = new Map<number, number>()
    for (const row of valid) {
      const worker = row.workers[stage] as number
      const previous = previousEnd.get(worker)
      const start = previous === undefined ? row.stages[stage][0] : Math.max(row.stages[stage][0], previous)
      const end = Math.max(start, row.stages[stage][1])
      row.stages[stage] = [start, end]
      previousEnd.set(worker, end)
    }
  }
  return out
}

/** VT:16613-16635 (computePeakConcurrency): most spans open at once; a span ending where another starts does not overlap it. */
function peakConcurrency(spans: readonly Span[]): number {
  const events: [number, number][] = []
  for (const [start, end] of spans) {
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) continue
    events.push([start, 1], [end, -1])
  }
  events.sort((a, b) => (a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1]))
  let current = 0
  let peak = 0
  for (const [, delta] of events) {
    current += delta
    peak = Math.max(peak, current)
  }
  return peak
}

/** One worker slot of a stage pool: legacy lane "Sampling W1" is `{ stage: 'sampling', worker: 1 }`. */
export type PoolLane = { stage: Stage; worker: number }

export type PoolRuntime = {
  /** The pool-slot rows, stage spans reconciled per worker. */
  rows: RuntimeRow[]
  /** Per stage: the largest recorded capacity, a row without one counting its worker id. */
  capacities: Record<Stage, number>
  /** Worker 1 of each pool, then worker 2, …; a pool contributes while the index is within its capacity. */
  lanes: PoolLane[]
  peaks: Record<Stage, number>
}

/**
 * VT:16707-16766 (getPoolRuntimeTimelineData): the rows the runner recorded in pool-slot mode, their
 * stages reconciled per worker, the pools' capacities, lanes and peak concurrency. Null when there are
 * no such rows or every capacity is 0 (the legacy "no pool-based runtime data").
 */
export function poolRuntime(rows: readonly RuntimeRow[]): PoolRuntime | null {
  const pool = rows.filter((row) => row.poolSlots)
  if (pool.length === 0) return null
  const reconciled = reconcile(pool)
  const capacityOf = (stage: Stage) =>
    reconciled.reduce((max, row) => Math.max(max, row.workerCapacities[stage] || row.workers[stage] || 0), 0)
  const capacities = { sampling: capacityOf('sampling'), evaluation: capacityOf('evaluation'), postprocess: capacityOf('postprocess') }
  if (capacities.sampling === 0 && capacities.evaluation === 0 && capacities.postprocess === 0) return null
  const lanes: PoolLane[] = []
  const laneCount = Math.max(capacities.sampling, capacities.evaluation, capacities.postprocess)
  for (let worker = 1; worker <= laneCount; worker += 1) {
    for (const stage of STAGES) if (worker <= capacities[stage]) lanes.push({ stage, worker })
  }
  const peakOf = (stage: Stage) => peakConcurrency(reconciled.map((row) => row.stages[stage]))
  return {
    rows: reconciled,
    capacities,
    lanes,
    peaks: { sampling: peakOf('sampling'), evaluation: peakOf('evaluation'), postprocess: peakOf('postprocess') },
  }
}

/** A drawn job: per stage its lane and its bar, which starts at the (reconciled) start and lasts the recorded duration. */
export type TimelineRow = {
  id: string
  generation: number
  isIslandCopy: boolean
  correct: boolean
  name: string
  model: string | null
  attempts: Attempts
  /** Pool mode: an index into `lanes`; legacy mode: the job's one lane, the same for every stage. Null: not drawn. */
  lanes: Record<Stage, number | null>
  /** Null exactly where the lane is. */
  stages: Record<Stage, Span | null>
  seconds: Record<Stage, number>
}

type TimelineFrame = {
  /** Earliest pipeline start to latest postprocess finish of the drawn rows. */
  range: Span
  /** A bar is drawn at least this long: max(0.25 s, 0.3% of the range); the parent applies it. */
  minBarSeconds: number
  rows: TimelineRow[]
}

/**
 * The timeline's three cases: pool lanes (with the per-pool "peak / capacity" summary), lanes inferred
 * from timestamps (legacy mode), or nothing to draw.
 */
export type RuntimeTimeline =
  | { mode: 'empty' }
  | (TimelineFrame & { mode: 'pool'; lanes: PoolLane[]; capacities: Record<Stage, number>; peaks: Record<Stage, number> })
  | (TimelineFrame & { mode: 'legacy'; laneCount: number })

/** VT:16506-16515 (getMinVisibleRuntimeBarMs, getDisplayRuntimeBarMs), in seconds: a bar lasts max(duration, this). */
const minBarSeconds = (rangeSeconds: number) =>
  !Number.isFinite(rangeSeconds) || rangeSeconds <= 0 ? 0.25 : Math.max(0.25, rangeSeconds * 0.003)

function frame<R extends RuntimeRow>(rows: readonly R[], laneOf: (row: R, stage: Stage) => number | null): TimelineFrame {
  let start = Infinity
  let end = -Infinity
  for (const row of rows) {
    start = Math.min(start, row.stages.pipeline[0])
    end = Math.max(end, row.stages.pipeline[1])
  }
  const timelineRows = rows.map((row): TimelineRow => {
    const lanes = { sampling: laneOf(row, 'sampling'), evaluation: laneOf(row, 'evaluation'), postprocess: laneOf(row, 'postprocess') }
    const bar = (stage: Stage): Span | null =>
      lanes[stage] === null ? null : [row.stages[stage][0], row.stages[stage][0] + row.seconds[stage]]
    return {
      id: row.id,
      generation: row.generation,
      isIslandCopy: row.isIslandCopy,
      correct: row.correct,
      name: row.name,
      model: row.model,
      attempts: row.attempts,
      lanes,
      stages: { sampling: bar('sampling'), evaluation: bar('evaluation'), postprocess: bar('postprocess') },
      seconds: row.seconds,
    }
  })
  return { range: [start, end], minBarSeconds: minBarSeconds(end - start), rows: timelineRows }
}

/**
 * VT:16823-17037 (renderRuntimeTimelinePlot) with the data part of VT:16768-16821
 * (getRuntimeTimelineLayout): pool lanes when `pool` has rows, a stage drawn only for a nonzero worker
 * id; otherwise every runtime row on the legacy inferred lanes; otherwise empty. `pool` is
 * `poolRuntime(rows)`. A stage whose worker has no lane within its pool's capacity is not drawn (legacy
 * placed it outside the plotted lanes).
 */
export function runtimeTimeline(rows: readonly RuntimeRow[], pool: PoolRuntime | null): RuntimeTimeline {
  if (pool && pool.rows.length) {
    const index = new Map(pool.lanes.map((lane, i) => [`${lane.stage}:${lane.worker}`, i]))
    const laneOf = (row: RuntimeRow, stage: Stage) => {
      const worker = row.workers[stage]
      return worker ? index.get(`${stage}:${worker}`) ?? null : null
    }
    return { mode: 'pool', lanes: pool.lanes, capacities: pool.capacities, peaks: pool.peaks, ...frame(pool.rows, laneOf) }
  }
  const legacy = legacyRuntimeLanes(rows)
  if (legacy.rows.length === 0) return { mode: 'empty' }
  return { mode: 'legacy', laneCount: legacy.laneCount, ...frame(legacy.rows, (row) => row.lane) }
}

/**
 * Why the throughput charts have nothing to show (VT:17469-17488, VT:16942-16946): no programs at all;
 * no program with every stage timestamp (the timeline is empty too); or runtime rows but none in pool
 * mode (the timeline still draws legacy lanes). Null when the pool charts have data.
 */
export type RuntimeEmptyState = 'noPrograms' | 'noRuntimeRows' | 'noPoolRows'

export function runtimeEmptyState(programCount: number, rows: readonly RuntimeRow[], pool: PoolRuntime | null): RuntimeEmptyState | null {
  if (programCount === 0) return 'noPrograms'
  if (rows.length === 0) return 'noRuntimeRows'
  if (!pool || pool.rows.length === 0) return 'noPoolRows'
  return null
}

export type SeriesPoint = { t: number; value: number }

export type OccupancySeries = {
  /** A step curve: starts at (start, 0); every start or finish adds the count before and after it. */
  points: SeriesPoint[]
  start: number
  end: number
  capacity: number
  totalSeconds: number
  /** Time spent at each count of busy workers, by count; counts never reached are absent. */
  secondsByCount: { count: number; seconds: number }[]
  avgOccupied: number
  utilizationPct: number
  /** Share of time at exactly `capacity` busy workers. */
  fullOccupancyPct: number
  idlePct: number
}

/**
 * VT:17039-17107 (computeOccupancySeries): busy workers of one stage over time, from the rows' spans of
 * positive length. Null when there are no rows or spans, or the capacity is not positive.
 */
export function occupancySeries(rows: readonly RuntimeRow[], stage: Stage, capacity: number): OccupancySeries | null {
  if (!rows.length || !Number.isFinite(capacity) || capacity <= 0) return null
  const spans = rows
    .map((row) => row.stages[stage])
    .filter(([start, end]) => Number.isFinite(start) && Number.isFinite(end) && end > start)
  if (!spans.length) return null
  let start = Infinity
  let end = -Infinity
  const events: [number, number][] = []
  for (const [s, e] of spans) {
    start = Math.min(start, s)
    end = Math.max(end, e)
    events.push([s, 1], [e, -1])
  }
  events.sort((a, b) => (a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1]))
  const points: SeriesPoint[] = [{ t: start, value: 0 }]
  const byCount = new Map<number, number>()
  let current = 0
  let previous = start
  for (const [time, delta] of events) {
    if (time > previous) byCount.set(current, (byCount.get(current) ?? 0) + (time - previous))
    points.push({ t: time, value: current })
    current += delta
    points.push({ t: time, value: current })
    previous = time
  }
  if (end > previous) byCount.set(current, (byCount.get(current) ?? 0) + (end - previous))
  const totalSeconds = end - start
  let occupied = 0
  for (const [count, seconds] of byCount) occupied += count * seconds
  const pct = (seconds: number) => (totalSeconds > 0 ? (seconds / totalSeconds) * 100 : 0)
  return {
    points,
    start,
    end,
    capacity,
    totalSeconds,
    secondsByCount: [...byCount].sort((a, b) => a[0] - b[0]).map(([count, seconds]) => ({ count, seconds })),
    avgOccupied: totalSeconds > 0 ? occupied / totalSeconds : 0,
    utilizationPct: totalSeconds > 0 ? (occupied / (capacity * totalSeconds)) * 100 : 0,
    fullOccupancyPct: pct(byCount.get(capacity) ?? 0),
    idlePct: pct(byCount.get(0) ?? 0),
  }
}

/** VT:17110-17118 (percentile): nearest rank, the ⌈pct·n⌉-th smallest. */
function percentile(values: readonly number[], pct: number): number | null {
  if (!values.length) return null
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil((pct / 100) * sorted.length) - 1))]
}

export type ThroughputSummary = {
  /** Earliest pipeline start to latest postprocess finish. */
  wallClockSeconds: number
  completedJobs: number
  /** Null where the evaluation pool has no occupancy series. */
  evalFullOccupancyPct: number | null
  evalUtilizationPct: number | null
  evalIdlePct: number | null
  /** Jobs per minute of wall clock; 0 when the span is 0. */
  completionsPerMinute: number
}

/** VT:17128-17170 (renderThroughputSummary): the six summary cards. Null without rows. */
export function throughputSummary(pool: Pick<PoolRuntime, 'rows' | 'capacities'>): ThroughputSummary | null {
  const { rows, capacities } = pool
  if (!rows.length) return null
  const evaluation = occupancySeries(rows, 'evaluation', capacities.evaluation)
  let start = Infinity
  let end = -Infinity
  for (const row of rows) {
    start = Math.min(start, row.stages.pipeline[0])
    end = Math.max(end, row.stages.pipeline[1])
  }
  const minutes = (end - start) / 60
  return {
    wallClockSeconds: end - start,
    completedJobs: rows.length,
    evalFullOccupancyPct: evaluation ? evaluation.fullOccupancyPct : null,
    evalUtilizationPct: evaluation ? evaluation.utilizationPct : null,
    evalIdlePct: evaluation ? evaluation.idlePct : null,
    completionsPerMinute: minutes > 0 ? rows.length / minutes : 0,
  }
}

/**
 * VT:17172-17247 (renderThroughputOccupancyPlot): each pool's busy-worker curve; its capacity line runs
 * at `series.capacity` from `series.start` to `series.end`. Pools without a series are left out; an
 * empty result is the legacy "no occupancy data".
 */
export function occupancyCurves(pool: Pick<PoolRuntime, 'rows' | 'capacities'>): { stage: Stage; series: OccupancySeries }[] {
  return STAGES.flatMap((stage) => {
    const series = occupancySeries(pool.rows, stage, pool.capacities[stage])
    return series ? [{ stage, series }] : []
  })
}

/**
 * VT:17249-17323 (renderThroughputOccupancyPercentPlot): each pool's busy share of its capacity in %,
 * and the span of the 100% line (earliest start to latest end of the drawn curves). Null when no pool
 * has a curve (the legacy "no normalized occupancy data").
 */
export function occupancyPercentCurves(
  pool: Pick<PoolRuntime, 'rows' | 'capacities'>,
): { curves: { stage: Stage; points: SeriesPoint[] }[]; range: Span } | null {
  let start = Infinity
  let end = -Infinity
  const curves = STAGES.flatMap((stage) => {
    const capacity = pool.capacities[stage]
    const series = occupancySeries(pool.rows, stage, capacity)
    if (!series || !Number.isFinite(capacity) || capacity <= 0) return []
    start = Math.min(start, series.start)
    end = Math.max(end, series.end)
    return [{ stage, points: series.points.map(({ t, value }) => ({ t, value: (value / capacity) * 100 })) }]
  })
  return curves.length ? { curves, range: [start, end] } : null
}

/**
 * VT:17325-17354 (renderThroughputEvalDistributionPlot): the share of wall clock (in %) spent with
 * 0, 1, …, capacity evaluation workers busy; `full` marks the bar at capacity. Null without an
 * evaluation series (the legacy "no evaluation occupancy data").
 */
export function evalOccupancyDistribution(
  pool: Pick<PoolRuntime, 'rows' | 'capacities'>,
): { count: number; sharePct: number; full: boolean }[] | null {
  const capacity = pool.capacities.evaluation
  const series = occupancySeries(pool.rows, 'evaluation', capacity)
  if (!series) return null
  const byCount = new Map(series.secondsByCount.map(({ count, seconds }) => [count, seconds]))
  return Array.from({ length: Math.floor(capacity) + 1 }, (_, count) => ({
    count,
    sharePct: series.totalSeconds > 0 ? ((byCount.get(count) ?? 0) / series.totalSeconds) * 100 : 0,
    full: count === capacity,
  }))
}

/**
 * VT:17356-17385 (renderThroughputCompletionRatePlot): evaluations finished per minute, in one-minute
 * buckets from the first finish; `t` is the bucket start. Only buckets with completions appear; empty
 * without rows (the legacy "no completion data").
 */
export function completionsPerMinute(rows: readonly RuntimeRow[]): SeriesPoint[] {
  if (!rows.length) return []
  const first = rows.reduce((min, row) => Math.min(min, row.stages.evaluation[1]), Infinity)
  const buckets = new Map<number, number>()
  for (const row of rows) {
    const bucket = Math.floor((row.stages.evaluation[1] - first) / 60) * 60 + first
    buckets.set(bucket, (buckets.get(bucket) ?? 0) + 1)
  }
  return [...buckets].sort((a, b) => a[0] - b[0]).map(([t, value]) => ({ t, value }))
}

/**
 * VT:17387-17423 (renderThroughputDurationTable): p50, p95 (nearest rank) and max of each stage's
 * recorded duration. Empty without rows (the legacy "no duration data").
 */
export function stageDurations(rows: readonly RuntimeRow[]): { stage: Stage; p50: number; p95: number; max: number }[] {
  if (!rows.length) return []
  return STAGES.map((stage) => {
    const values = rows.map((row) => row.seconds[stage])
    return { stage, p50: percentile(values, 50) as number, p95: percentile(values, 95) as number, max: Math.max(...values) }
  })
}

export type StageUtilization = {
  stage: Stage
  capacity: number
  avgOccupied: number | null
  utilizationPct: number | null
  fullOccupancyPct: number | null
  idlePct: number | null
}

/** VT:17425-17461 (renderThroughputUtilizationTable): one row per pool; null figures where it has no occupancy series. */
export function stageUtilization(pool: Pick<PoolRuntime, 'rows' | 'capacities'>): StageUtilization[] {
  return STAGES.map((stage) => {
    const capacity = pool.capacities[stage]
    const series = occupancySeries(pool.rows, stage, capacity)
    return {
      stage,
      capacity,
      avgOccupied: series ? series.avgOccupied : null,
      utilizationPct: series ? series.utilizationPct : null,
      fullOccupancyPct: series ? series.fullOccupancyPct : null,
      idlePct: series ? series.idlePct : null,
    }
  })
}
