// The progress view's runtime timeline and throughput numbers (award-redesign 6.3); node strips the TypeScript types.
import assert from 'node:assert/strict'
import test from 'node:test'
import {
  completionsPerMinute,
  evalOccupancyDistribution,
  legacyRuntimeLanes,
  occupancyPercentCurves,
  occupancySeries,
  poolRuntime,
  runtimeEmptyState,
  runtimeRows,
  runtimeTimeline,
  stageDurations,
  stageUtilization,
  throughputSummary,
} from './throughput.ts'

// Epoch seconds of the run's first pipeline start; every fixture time is an offset from it.
const T0 = 1_700_000_000

/** A job with every stage timestamp: offsets in seconds, sampling → evaluation → postprocess. */
const job = (id, { pipeline, sampling = [0, 1], evaluation = [1, 2], postprocess = [2, 3] } = {}, extra = {}) => ({
  id,
  generation: extra.generation ?? 1,
  timestamp: extra.timestamp ?? 0,
  correct: extra.correct ?? true,
  combined_score: 'score' in extra ? extra.score : 1,
  metadata: {
    pipeline_started_at: T0 + (pipeline ?? sampling[0]),
    sampling_started_at: T0 + sampling[0],
    sampling_finished_at: T0 + sampling[1],
    evaluation_started_at: T0 + evaluation[0],
    evaluation_finished_at: T0 + evaluation[1],
    postprocess_started_at: T0 + postprocess[0],
    postprocess_finished_at: T0 + postprocess[1],
    ...extra.metadata,
  },
})

/** Pool-slot metadata: worker ids per stage and the capacities the runner recorded. */
const slots = (workers, capacities = {}) => ({
  timeline_lane_mode: 'pool_slots',
  sampling_worker_id: workers.sampling ?? null,
  evaluation_worker_id: workers.evaluation ?? null,
  postprocess_worker_id: workers.postprocess ?? null,
  sampling_worker_capacity: capacities.sampling ?? null,
  evaluation_worker_capacity: capacities.evaluation ?? null,
  postprocess_worker_capacity: capacities.postprocess ?? null,
})

const ids = (rows) => rows.map((r) => r.id)

test('island copies of one job keep a non-copy over a copy, then a correct row, then a scored one, in the job’s first place', () => {
  const { rows } = runtimeRows([
    job('copy', {}, { metadata: { source_job_id: 'j', _is_island_copy: true } }),
    job('solo'),
    job('failed', {}, { correct: false, metadata: { source_job_id: 'j' } }),
    job('kept', {}, { metadata: { source_job_id: 'j' } }),
    job('spawned', {}, { metadata: { source_job_id: 'j', _spawned_island: true } }),
  ])
  assert.deepEqual(ids(rows), ['kept', 'solo'])
})

test('rows of one job with the same priority keep the earlier program, a missing timestamp counting as 0, else the first seen', () => {
  const late = job('late', {}, { timestamp: 20, metadata: { source_job_id: 'j' } })
  const early = job('early', {}, { timestamp: 10, metadata: { source_job_id: 'j' } })
  assert.deepEqual(ids(runtimeRows([late, early]).rows), ['early'])
  const undated = { ...job('undated', {}, { metadata: { source_job_id: 'j' } }), timestamp: null }
  assert.deepEqual(ids(runtimeRows([early, undated]).rows), ['undated'])
  const twin = job('twin', {}, { timestamp: 10, metadata: { source_job_id: 'j' } })
  assert.deepEqual(ids(runtimeRows([early, twin]).rows), ['early'])
})

test('a failed evaluation’s raw score and a "false" string earn no priority (legacy counted both)', () => {
  const rawScore = job('raw', {}, { correct: false, score: 5, timestamp: 20, metadata: { source_job_id: 'j' } })
  const noScore = job('none', {}, { correct: false, score: null, timestamp: 10, metadata: { source_job_id: 'j' } })
  assert.deepEqual(ids(runtimeRows([rawScore, noScore]).rows), ['none'])
  const falseString = job('string', {}, { correct: 'false', timestamp: 10, metadata: { source_job_id: 'k' } })
  const passed = job('passed', {}, { correct: true, timestamp: 20, metadata: { source_job_id: 'k' } })
  const [row] = runtimeRows([falseString, passed]).rows
  assert.equal(row.id, 'passed')
  assert.equal(runtimeRows([falseString]).rows[0].correct, false)
})

test('a row needs all seven stage timestamps; numeric strings are read, empty strings and nulls are missing', () => {
  const asStrings = job('strings')
  for (const [key, value] of Object.entries(asStrings.metadata)) asStrings.metadata[key] = String(value)
  const blank = job('blank', {}, { metadata: { evaluation_finished_at: '' } })
  const missing = job('missing', {}, { metadata: { postprocess_started_at: null } })
  assert.deepEqual(ids(runtimeRows([asStrings, blank, missing]).rows), ['strings'])
})

test('times are seconds since the earliest pipeline start, which is the origin; durations are never negative', () => {
  const { origin, rows } = runtimeRows([
    job('a', { pipeline: 5, sampling: [6, 8], evaluation: [8, 20], postprocess: [20, 21] }),
    job('b', { pipeline: 2, sampling: [3, 2.5], evaluation: [4, 9], postprocess: [9, 12] }),
  ])
  assert.equal(origin, T0 + 2)
  assert.deepEqual(rows[0].stages, { pipeline: [3, 19], sampling: [4, 6], evaluation: [6, 18], postprocess: [18, 19] })
  assert.deepEqual(rows[1].seconds, { sampling: 0, evaluation: 5, postprocess: 3 })
})

test('legacy lanes: by evaluation start, each job takes the first lane free by then (1e-6 s slack), else a new one', () => {
  const { rows } = runtimeRows([
    job('x', { pipeline: 0, evaluation: [0, 10] }),
    job('z', { pipeline: 0, evaluation: [10 - 4e-7, 12] }),
    job('w', { pipeline: 0, evaluation: [8, 9] }),
    job('y', { pipeline: 0, evaluation: [5, 8] }),
  ])
  const laned = legacyRuntimeLanes(rows)
  assert.deepEqual(laned.rows.map((r) => [r.id, r.lane]), [['x', 0], ['y', 1], ['w', 1], ['z', 0]])
  assert.equal(laned.laneCount, 2)
})

test('pool slots: worker 1 of each pool, then worker 2, … within each pool’s capacity, a row without one counting its worker id', () => {
  const { rows } = runtimeRows([
    job('a', {}, { metadata: slots({ sampling: 1, evaluation: 1, postprocess: 1 }, { sampling: 1, evaluation: 2 }) }),
    job('b', { sampling: [1, 2], evaluation: [2, 3], postprocess: [3, 4] }, { metadata: slots({ sampling: 1, evaluation: 2, postprocess: 3 }) }),
    job('legacy', {}, { metadata: { evaluation_worker_id: 9 } }),
  ])
  const pool = poolRuntime(rows)
  assert.deepEqual(ids(pool.rows), ['a', 'b'])
  assert.deepEqual(pool.capacities, { sampling: 1, evaluation: 2, postprocess: 3 })
  assert.deepEqual(pool.lanes, [
    { stage: 'sampling', worker: 1 },
    { stage: 'evaluation', worker: 1 },
    { stage: 'postprocess', worker: 1 },
    { stage: 'evaluation', worker: 2 },
    { stage: 'postprocess', worker: 2 },
    { stage: 'postprocess', worker: 3 },
  ])
})

test('a worker’s overlapping jobs are serialised: the later one starts at the earlier finish and its bar keeps the recorded duration', () => {
  const { rows } = runtimeRows([
    job('a', { sampling: [0, 2], evaluation: [2, 10], postprocess: [10, 11] }, { metadata: slots({ sampling: 1, evaluation: 1, postprocess: 1 }, { evaluation: 2 }) }),
    job('b', { sampling: [2, 4], evaluation: [5, 12], postprocess: [12, 13] }, { metadata: slots({ sampling: 1, evaluation: 1, postprocess: 1 }, { evaluation: 2 }) }),
  ])
  const pool = poolRuntime(rows)
  const b = pool.rows[1]
  assert.deepEqual(b.stages.evaluation, [10, 12])
  assert.equal(b.seconds.evaluation, 7)
  // Back-to-back jobs on one worker never count as two at once.
  assert.deepEqual(pool.peaks, { sampling: 1, evaluation: 1, postprocess: 1 })

  const timeline = runtimeTimeline(rows, pool)
  assert.equal(timeline.mode, 'pool')
  assert.deepEqual(timeline.range, [0, 13])
  assert.equal(timeline.minBarSeconds, 0.25)
  const drawn = timeline.rows.find((r) => r.id === 'b')
  assert.deepEqual(drawn.lanes, { sampling: 0, evaluation: 1, postprocess: 2 })
  assert.deepEqual(drawn.stages, { sampling: [2, 4], evaluation: [10, 17], postprocess: [12, 13] })
})

test('in pool mode a stage is drawn only on a nonzero worker with a lane in its pool', () => {
  const { rows } = runtimeRows([
    job('d', {}, { metadata: slots({ sampling: 0, evaluation: 3, postprocess: 1 }, { evaluation: 2 }) }),
  ])
  const timeline = runtimeTimeline(rows, poolRuntime(rows))
  assert.deepEqual(timeline.lanes, [
    { stage: 'evaluation', worker: 1 },
    { stage: 'postprocess', worker: 1 },
    { stage: 'evaluation', worker: 2 },
  ])
  assert.deepEqual(timeline.rows[0].lanes, { sampling: null, evaluation: null, postprocess: 1 })
  assert.deepEqual(timeline.rows[0].stages, { sampling: null, evaluation: null, postprocess: [2, 3] })
})

test('without pool slots the timeline draws every stage of a job on its inferred lane', () => {
  const { rows } = runtimeRows([
    job('x', { evaluation: [1, 10], postprocess: [10, 11] }),
    job('y', { evaluation: [2, 4], postprocess: [4, 5] }),
  ])
  const timeline = runtimeTimeline(rows, poolRuntime(rows))
  assert.equal(timeline.mode, 'legacy')
  assert.equal(timeline.laneCount, 2)
  assert.deepEqual(timeline.rows.map((r) => [r.id, r.lanes]), [
    ['x', { sampling: 0, evaluation: 0, postprocess: 0 }],
    ['y', { sampling: 1, evaluation: 1, postprocess: 1 }],
  ])
  assert.deepEqual(timeline.range, [0, 11])
})

const overlapping = () => {
  const { rows } = runtimeRows([
    job('a', { pipeline: 0, sampling: [0, 0], evaluation: [0, 10], postprocess: [10, 10] }, { metadata: slots({ evaluation: 1 }, { evaluation: 2 }) }),
    job('b', { pipeline: 0, sampling: [0, 5], evaluation: [5, 15], postprocess: [15, 30] }, { metadata: slots({ evaluation: 2 }, { evaluation: 2 }) }),
  ])
  return poolRuntime(rows)
}

test('occupancy over two overlapping jobs: the busy-worker steps and the time at each count', () => {
  const series = occupancySeries(overlapping().rows, 'evaluation', 2)
  assert.deepEqual(series.points.map(({ t, value }) => [t, value]), [
    [0, 0], [0, 0], [0, 1], [5, 1], [5, 2], [10, 2], [10, 1], [15, 1], [15, 0],
  ])
  assert.deepEqual(series.secondsByCount, [{ count: 1, seconds: 10 }, { count: 2, seconds: 5 }])
  assert.equal(series.totalSeconds, 15)
  assert.equal(series.avgOccupied, 20 / 15)
  assert.equal(series.utilizationPct, (20 / 30) * 100)
  assert.equal(series.fullOccupancyPct, (5 / 15) * 100)
  assert.equal(series.idlePct, 0)
  assert.equal(occupancySeries(overlapping().rows, 'evaluation', 0), null)
})

test('the throughput figures for those two jobs: summary, shares by busy count, % of capacity and the utilisation table', () => {
  const pool = overlapping()
  assert.deepEqual(throughputSummary(pool), {
    wallClockSeconds: 30,
    completedJobs: 2,
    evalFullOccupancyPct: (5 / 15) * 100,
    evalUtilizationPct: (20 / 30) * 100,
    evalIdlePct: 0,
    completionsPerMinute: 4,
  })
  assert.deepEqual(evalOccupancyDistribution(pool), [
    { count: 0, sharePct: 0, full: false },
    { count: 1, sharePct: (10 / 15) * 100, full: false },
    { count: 2, sharePct: (5 / 15) * 100, full: true },
  ])
  const percent = occupancyPercentCurves(pool)
  assert.deepEqual(percent.curves.map((c) => c.stage), ['evaluation'])
  assert.deepEqual(percent.curves[0].points.map((p) => p.value), [0, 0, 50, 50, 100, 100, 50, 50, 0])
  assert.deepEqual(percent.range, [0, 15])
  // Sampling and postprocess have no workers recorded, hence no capacity and no figures.
  assert.deepEqual(stageUtilization(pool).map((r) => [r.stage, r.capacity, r.utilizationPct]), [
    ['sampling', 0, null],
    ['evaluation', 2, (20 / 30) * 100],
    ['postprocess', 0, null],
  ])
})

test('completions per minute: one-minute buckets from the first evaluation finish, only buckets with completions', () => {
  const finishing = (id, at) => job(id, { pipeline: 0, sampling: [0, 1], evaluation: [1, at], postprocess: [at, at + 1] })
  const { rows } = runtimeRows([10, 40, 69.5, 70, 195].map((at, i) => finishing(`j${i}`, at)))
  assert.deepEqual(completionsPerMinute(rows), [{ t: 10, value: 3 }, { t: 70, value: 1 }, { t: 190, value: 1 }])
  assert.deepEqual(completionsPerMinute([]), [])
})

test('a zero wall-clock span gives zero completions per minute', () => {
  const { rows } = runtimeRows([job('flat', { sampling: [0, 0], evaluation: [0, 0], postprocess: [0, 0] }, { metadata: slots({ evaluation: 1 }) })])
  assert.equal(throughputSummary({ rows, capacities: { sampling: 0, evaluation: 1, postprocess: 0 } }).completionsPerMinute, 0)
})

test('stage durations: nearest-rank p50 and p95 and the max of the recorded durations', () => {
  const { rows } = runtimeRows([3, 1, 5, 2, 4].map((s, i) => job(`j${i}`, { evaluation: [1, 1 + s] })))
  assert.deepEqual(stageDurations(rows).find((r) => r.stage === 'evaluation'), { stage: 'evaluation', p50: 3, p95: 5, max: 5 })
  assert.deepEqual(stageDurations([]), [])
})

test('a run without stage timestamps has no runtime rows and nothing on the timeline', () => {
  const programs = [{ id: 'a', generation: 0, timestamp: T0, correct: true, combined_score: 1, metadata: { patch_name: 'init' } }]
  const runtime = runtimeRows(programs)
  assert.deepEqual(runtime, { origin: null, rows: [] })
  const pool = poolRuntime(runtime.rows)
  assert.equal(pool, null)
  assert.deepEqual(runtimeTimeline(runtime.rows, pool), { mode: 'empty' })
  assert.equal(runtimeEmptyState(programs.length, runtime.rows, pool), 'noRuntimeRows')
})

test('empty states: no programs, runtime rows but no pool data (including pools with no capacity), and pool data', () => {
  assert.equal(runtimeEmptyState(0, [], null), 'noPrograms')
  const { rows: plain } = runtimeRows([job('a')])
  assert.equal(runtimeEmptyState(1, plain, poolRuntime(plain)), 'noPoolRows')
  const { rows: idle } = runtimeRows([job('b', {}, { metadata: slots({}) })])
  assert.equal(poolRuntime(idle), null)
  assert.equal(runtimeEmptyState(1, idle, poolRuntime(idle)), 'noPoolRows')
  const pool = overlapping()
  assert.equal(runtimeEmptyState(2, pool.rows, pool), null)
})
