# Demo workspace generator

> **DEMO · 演示数据，非真实研究** — everything this generator writes is synthetic.
> No research is run, no model is called, no embedding is computed, nothing touches the
> network, and no number in the output is a real result.

`build_demo_workspace.py` builds a labelled research workspace and an ordinary-runs root.
Use them to judge UI designs on realistic content: several directions in different
states, hundreds of programs per run, prompt evolution, meta summaries, plots and
throughput timing.

## Usage

Run it from the repository root, using the repository's existing environment:

```bash
uv run --offline --no-sync python shinka/research/webui/demo/build_demo_workspace.py \
    --workspace <empty dir>/workspace --runs <empty dir>/runs

uv run --offline --no-sync shinka_research ui <empty dir>/workspace \
    --runs-root <empty dir>/runs --port 8771 --no-browser
```

Both targets must be missing or empty; the generator refuses to write into a directory
that already has content. When it finishes, it prints a JSON summary covering every
direction state and every database.

## What gets written

The output goes through the same storage code a real run uses: `Notebook`,
`prepare_data` / `scaffold_experiment`, `append_log`, `library`, `ProgramDatabase`
(with the default island assignment strategy and `_migrate_program` for migrations),
`SystemPromptDatabase`, `with_pipeline_timing` and `QueryResult.to_dict()`.

### Workspace

| Directory | State |
|---|---|
| `d1-momentum-regimes` | Active. Four hypotheses: supported, refuted, inconclusive (two experiments) and running. The running experiment's database is still growing, so the notebook does not count its trials yet. Budget $80, partly spent. Lockbox sealed. Probes recorded. Events span 8 to 18 October. |
| `d2-overnight-gap` | Concluded. One hypothesis confirmed on the lockbox and one refuted. Lockbox opened; `CONFIRMATION.md` written. |
| `d3-volume-shock` | New. One proposed hypothesis, with no data, no experiments and no events (empty states). |
| `d4-short-reversal` | Over budget: spend is about twice the $8 budget. One hypothesis is challenged but has no verdict yet. The other has a `run-failed` experiment (`h002/e01`) in which the evaluator fails on about 75% of programs from generation 15 on. |
| `library/` | Nine wiki pages (concept, paper, finding), an index, a log, two compiled raw sources and one pending link. The paper pages are fictional. |

Every experiment that has trials gets `results/programs.sqlite` with 150–320 programs,
25–36 generations, 3–4 islands with migrations, and three demo model names whose
posteriors (`metadata.llm_result.model_posteriors`) change each generation. About 8% of
evaluations fail. Each experiment also gets:

- Patch types from Shinka's taxonomy: `init` for the generation-0 seeds, then a mix of
  `diff`, `full`, `cross` and `fix`. A `fix` patch repairs a program whose evaluation
  failed, so its parent has `correct = 0`. Every program has a short `patch_name`
  (`tune_fast`, `add_regime_filter`, `fix_look_ahead_detected`, ...) and integer
  `novelty_attempt`, `resample_attempt` and `patch_attempt`. The DEMO label is in
  `patch_description`, the code and the feedback.
- Synthetic embeddings with five-cluster structure on every program, failed ones
  included, plus PCA 2D/3D coordinates and cluster IDs.
- `metadata.code_analysis_metrics` (including `lines_of_code`), written by
  `ProgramDatabase.add`, and `metadata.compute_time` in seconds. These are the default
  axes of the MAPs view.
- 3–4 public metrics, and private metrics that include the sentinel
  `DEMO_PRIVATE_SENTINEL_20261019_MUST_STAY_HIDDEN`. The API must never return it.
- Program code, unified diffs, stdout/stderr and feedback.
- Pool-slot timing on every program: `timeline_lane_mode = "pool_slots"`, the seven
  stage timestamps from `pipeline_started_at` to `postprocess_finished_at`, worker IDs
  and capacities for sampling, evaluation and postprocessing, and `source_job_id`.
- `prompts.sqlite`: 6–12 prompts with parent links, percentile fitness and
  `metadata.llm.cost`.
- `meta/meta_N.txt`, written at four processed counts in the layout of
  `MetaSummarizer.write_meta_output`: `# INDIVIDUAL PROGRAM SUMMARIES` (one
  `**Program Name: ...**` entry per program since the previous file),
  `# GLOBAL INSIGHTS SCRATCHPAD` and `# META RECOMMENDATIONS`. The figures in each
  section come from the demo database.
- `gen_N/results/plots/*.png` (matplotlib), for four generations.

### Ordinary runs root

Five runs from three tasks, with no research association:

| Run | Size |
|---|---|
| `circle_packing_demo/run_20261012_093000` | Tiny: 12 programs, 2 islands. |
| `circle_packing_demo/run_20261015_141500` | Large: 500 programs, 40 generations, 4 islands, 12 prompts. |
| `bin_packing_demo/run_20261013_101000` | Empty database: schema only, no programs. |
| `bin_packing_demo/run_20261016_200000` | Medium: 180 programs. |
| `quant_signal_demo/run_20261017_083000` | Medium: 220 programs. Unknown cost: one of its models is an unpriced headless agent (see below). |

## Costs

Program costs are written the way the runner writes them: `metadata.api_costs`,
`metadata.embed_cost` and `metadata.novelty_cost` on every program (seeds have no API
cost, programs without novelty checks have no novelty cost), and `metadata.meta_cost` only
on the program each meta update charged. Prompt records carry `metadata.llm.cost`.

`rundb.run_cost` counts an absent field as a recorded zero, as the runner's budget total
does, so every research direction and every ordinary run has a known cost except one:

- `runs/quant_signal_demo/run_20261017_083000` (`RunSpec(cost_unknown=True)`)

Its first model is `headless/demo-agent`, a headless CLI agent with no price list. Its
programs carry `headless_pricing_unknown: true` and `api_costs: 0.0`, which core telemetry
treats as pricing-unknown. `/get_database_stats` therefore returns `total_cost: null` for
that run, the runs list shows its cost as unknown, and the runs total becomes a lower bound.
Research experiments cannot use headless models (`shinka_research run` refuses them), so no
direction has an unknown spend.

## Determinism

The seed is fixed (`SEED = 20261019`). A scoped demo clock stands in for wall-clock time
inside `shinka.database.dbase`, `shinka.database.islands`, `notebook._now` and
`library._now`. Building twice into the same path therefore gives byte-identical
SQLite files and notebooks.

Some values are absolute paths, because the real research code records them:

- `DataSplit.dev_path`
- `Experiment.data_path`
- the data path embedded in the frozen `evaluate.py`, and so `evaluator_hash`
- the skeptic's `best_program`

Builds into different directories are therefore identical only after those paths are
normalised. Even then, `evaluator_hash` still differs between them.

## Known limits

- Plot titles are ASCII (`DEMO - synthetic, not real research`), because matplotlib's
  default fonts cannot render the Chinese label.
- The figures for skeptic, verdict and lockbox come from a fixed plan. The labels are
  then derived by the real `skeptic.decide()`. The trial counts and the noise bar come
  from the demo databases through `skeptic.direction_trials()`.
