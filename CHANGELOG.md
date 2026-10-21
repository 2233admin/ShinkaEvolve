# Changelog

All notable changes to `shinka-evolve` are documented in this file.

## TBD

### Added

- Added a unified read-only Research workspace with independent ordinary runs,
  comparison links and the fork's redesigned full experiment analysis workbench.
  Research links retain direction/hypothesis context, while analysis URLs retain
  tabs, selected programs and comparison selections across refresh and history.
  The restored renderer uses local pinned dependencies, shared dark/light tokens,
  keyboard navigation and protected same-origin data access.
- Added Chinese and English to the Research workbench: one copy table for every
  React screen, a 中/EN switch remembered across reloads and kept in step with
  the restored analysis pages, and withheld skeptic/verdict notes and unreadable
  directions worded by the page (`/api/snapshot` events now carry `withheld`
  with a `null` note; `/api/workspace` returns the error kind, not a sentence).
- Added per-direction evidence to `/api/workspace` (hypothesis status counts,
  best in-sample score, noise bar, evaluated and failed programs, last
  activity), computed exactly as the direction page computes it, and the
  program id on every `/api/snapshot` trial point.
- Added the Research workbench's native shell: a rail with section counts and
  every direction's state (shape and word), a top bar with breadcrumb, an
  “Updated” time that refreshes on click and flags a lost connection, theme and
  language switches, a Ctrl K palette for pages, directions, runs, library pages
  and actions, and on phones a bottom tab bar with a settings sheet. The data
  refreshes every 30 seconds while the page is visible without losing the
  selection. Demo workspaces show the DEMO label once instead of in every title.
- Added the Research workbench's native directions overview: a data sentence
  whose counts jump to the first matching row, totals of trials, evaluated and
  failed programs, spend and budget, and a sortable table in which every row
  shares one evidence scale (best record against its noise line), with a
  totals row and an activity column fed by the new `/api/workspace` `events`
  field (the newest 30 events across directions, withheld notes kept hidden).
  Unreadable directions stay as error rows outside the totals; phones get one
  card per direction. The overview never requests `/api/snapshot`.
- Added the Research workbench's native direction report, read in the order the
  argument runs: a sortable hypothesis table, the selected hypothesis's claim,
  mechanism and falsification with its full notebook text, every trial against
  the noise line the chart draws itself (integer trial numbers, failures in
  their own band), the check ledger, and beside it the verdict, the lockbox
  confirmation and each experiment. A trial point, by click or by keyboard,
  opens that program's code in its experiment's analysis; an experiment opens
  the program its verdict rests on. Back returns to the same hypothesis with the
  scroll position, sort, open text and event filter as left. The timeline and
  the filterable event stream are native too, so the workbench no longer loads
  Astro UXDS web components.
- Added the Research workbench's native shared library: the index, the log and
  every wiki page are links that can be shared and reopened, sources waiting to
  be compiled are named with the command that marks them done, and a missing
  page says so instead of silently showing the index. `[[page]]` links open the
  library page from any Markdown in the workbench, including hypothesis text;
  Markdown links use the text colour instead of the signal orange.
- Added the Research workbench's native run list: runs grouped by evaluator
  (research experiments under their direction, ordinary runs under their task)
  with every legacy column plus failures, runtime and update time, search (`/`
  focuses it), a kind filter, sorting and a selection that survives filtering,
  all kept in the URL. Totals always cover every run, with a subtotal while
  filtered and a lower bound while a cost is unknown or unread, or while a row
  keeps its earlier numbers because its latest read failed; stall and cost
  thresholds are shown by type weight instead of a second colour. The run list
  refreshes every 10 seconds, and the top bar can pause automatic refresh for
  the visit.
- Added the Research workbench's native compare page: cumulative best, program
  count or valid count against generation and against cumulative spend for any
  two or more runs, point for point as the old compare page drew them, with
  each run told apart by shape and line and its value readable by pointer or
  arrow keys; a leaderboard of the best valid programs, each opening its code;
  and run-level metrics with the better value in bold. Runs from different
  tasks keep their score curves, leaderboards and best apart by task, since
  scores from different evaluators do not compare. The metric and the selection
  live in the URL, and the page follows automatic refresh and its pause.
- Added background research runs: `shinka_research run --detach` returns at
  once and keeps the run going after its window closes (no console, a session
  or process group of its own, output in the experiment's `run.log`), and
  `shinka_research watch <direction-or-workspace>` records the end of a run
  whose processes vanished without doing so. A run's end is recorded exactly
  once, by whichever of the waiting command, the background supervisor or
  `watch` gets there first; one experiment runs once at a time, overlapping
  runs of a hypothesis share the status they restore, and only when none of
  them added a program. A run killed while creating its database is recorded
  too, as adding nothing.
- Added round grants for the research agent: `shinka_research grant <dir>
  <hid>... --attempts N --max-cost USD [--allow NS.FIELD]` records the owner's
  approval of one round, and where `SHINKA_RESEARCH_AGENT` is set every `run`
  needs `--grant`, takes one of its attempts, always runs in the background,
  passes `shinka_run` only the `--set` keys the grant allows and is capped at
  the smaller of the grant's cap and the direction budget left after the caps
  of grant runs in progress (counted on top of what a resumed experiment
  already spent; a soft cap: shinka_run stops proposing once committed cost
  reaches it, and work in flight still finishes). A used-up grant pauses the
  round, logged once. Grants and attempts
  live in `ledger.json`, checked against `results.tsv`, so a ledger changed by
  hand stops grant runs; manual runs are unchanged and their end events say
  "manual run", and `probe` now logs each probe when it starts. Opening the
  lockbox there takes `confirm --approval <id>` from the owner's
  `shinka_research approve`: the opening spends it in the same lock, `--yes`
  alone is refused, and an interrupted opening is finished only with the
  approval that opened it; a manual `confirm --yes` is unchanged.
- Added a read-only view of the agent's rounds to the Research workspace API
  and to `shinka_research brief`: each grant's attempts and the state of their
  runs (running, ended, ended without a record, or lost), the runs in
  progress, lockbox approvals, and what waits on the owner (a round over, an
  unrecorded end, a lost attempt, a grant the agent cannot use and why, an
  unfinished opening, a ledger that does not match the run log), each with
  what to do, plus the round's state. It names no process, command or path
  and none of the validation figures a run keeps for its end. The owner's next
  grant after a round's attempts ran out answers the round, whatever hypotheses
  it names.
- Added native Google Gemini 3.7 Flash support at its introductory 2026 price.
- Added native Google Gemini 3.6 Flash support with current pricing and
  level-based thinking controls. Thanks @anantgar.
- Added optional Weights & Biases logging with resumable run IDs, per-individual
  scores, timing and cost metrics, and final run summaries in PR #149. PR #177
  adds monotonic population/island snapshots and a secured metadata boundary.
  Thanks @anantgar.
- Added Verilog/SystemVerilog as a first-class evolution target in PR #137,
  including `.sv` task detection, syntax validation, EVOLVE-BLOCK marker support,
  failure-artifact rendering, and a self-contained RTLLM example. Thanks @Tyronita.
- Added startup pricing refresh from models.dev with conditional HTTP caching,
  validated offline fallback, provider-qualified model discovery, and per-run
  pricing snapshots in PR #141. The tutorial now includes a no-cost catalog
  preflight. Thanks @rodmarkun.
- Added DeepSeek V4 Flash and V4 Pro pricing entries to the DeepSeek LLM pricing catalog.
- Added controls to reduce evaluation stdout bloat: `JobConfig.eval_verbose` suppresses framework evaluation progress output, and `DatabaseConfig.max_stdout_log_chars` can persist only the tail of `stdout_log` metadata while keeping full logs on disk. Thanks @marcopirazzini.
- Added SLURM support for `numeric_threads_per_job`, applying numeric-library thread caps consistently across local and SLURM evaluation jobs. Thanks @marcopirazzini.
- Added `shinka_research watch <dir> --since NAME` and `brief <dir> --since NAME`
  so a scheduled shell job can wake a research agent only on news: the owner's
  grants and lockbox approvals and the end of any run, which `watch` alone
  never reported for a run that recorded its own end. `watch --since` exits 3
  while there is news NAME has not read and never marks it read; `brief
  --since` lists it and marks it read once printed (`<dir>/.watch/NAME.json`),
  so a wake that never reached the agent is reported again. The research skill
  documents the terminal job (`wakeOn=failure`, `missedWindow=catch_up_one`).
- Added a "Lesson for the skill" section to the research report template: at
  most one lesson per round about how to work rather than about the market
  (when it applies, the lesson, the `results.tsv` events that show it, and a
  confidence), or "None". It is a candidate for the owner to review, not an
  adopted rule: the research skill instructs the agent not to act on a lesson
  from an earlier report (guidance only, not enforced in code), and the owner
  decides whether the skill changes. Adapted from google-research/harness2's
  lesson entries, without its automatic adoption.

### Changed

- Reused provider SDK clients and their HTTP connection pools across compatible
  requests, with explicit credential/configuration isolation, process-safe sync
  caching, event-loop-scoped async caching, and cache shutdown helpers.
- Reduced Python complexity-analysis overhead by parsing each candidate AST once
  while preserving the existing metrics contract. Thanks @dexhunter.
- Relaxed the exact HTTPX dependency pin to a `>=0.27` compatibility lower
  bound in issue #180. Thanks @htmai-880.
- Removed the automatic Claude Code Review pull-request workflow.
- Moved the Research workbench to one token stylesheet
  (`app/src/design/tokens.css`) for the dark default and light themes; the
  remembered theme now applies before the first paint.
- Replaced Carbon in the Research workbench with react-aria-components 1.22.1:
  glossary terms and table definitions open a react-aria tooltip (also on a
  tap), and notices are a project component with Carbon's low-contrast look.
  `@carbon/react`, `@carbon/styles` and the unused `@carbon/charts-react` are
  gone, and the workbench stylesheet shrinks from 1.28 MB to 64 kB. Carbon's
  stylesheet also fetched IBM Plex Mono from IBM's CDN; the semibold figures now
  use a bundled copy instead.
- Changed the Research workbench URLs to one page per `view`
  (`directions`, `direction`, `runs`, `compare`, `run`, `library`); the restored
  analysis pages' state now appears as `select`, `runs`, `run` and `program`, and
  `/api/snapshot` experiment links open `/?view=run&run=…`. The old
  `analysis=`/`compare=` and view-less `?direction=` links are not redirected.
- Changed `shinka_research confirm` to open the lockbox only after every check
  that leaves it untouched has passed, and then to freeze, under the direction's
  lock, each supported hypothesis's program, evaluator and data hashes, trial
  count and the thresholds before any candidate runs on it. A candidate that
  times out, exits with an error, returns invalid positions, looks ahead or was
  changed after opening is now recorded as `unconfirmed` with that category
  alone (its output may carry lockbox figures) instead of aborting the whole
  confirm; results reach the notebook together. Running `confirm --yes` again
  after an interruption finishes the frozen hypotheses under the frozen rules
  and records any whose candidate had already started as interrupted rather
  than running it twice, or rewrites missing `CONFIRMATION.md` and log lines.
  Until it finishes, experiment commands refuse and a second `confirm` is
  refused at once; it does not open while a run is still adding trials, since
  every trial counts toward the DSR. A probe now counts as a trial when it
  starts, so one that fails or is killed is counted too.

### Fixed

- Fixed the Research workspace UI answering requests addressed to any host
  name: a page on another site that points its own name at 127.0.0.1 (DNS
  rebinding) could read notebooks and code. It now answers only a `Host` of
  `127.0.0.1:<port>` or `localhost:<port>`, refusing any other with 421, and
  only GET (HEAD gets 405).
- Fixed restored analysis frontiers admitting failed zero-valued scores, stale
  program details after rapid selection, lost selected-program links, and absent
  cost/diversity measurements being presented as measured zero values.
- Fixed Research run statistics and the restored analysis pages reporting
  every real run's cost as unknown: `meta_cost` is written only on programs a
  meta update charged, and both demanded all four cost fields on every program.
  The research budget, workspace, direction snapshot, run statistics, analysis
  cost tab and compare page now share one rule (`rundb.run_cost`): non-copy
  programs, absent fields as zero, unpriced headless calls and malformed values
  as unknown, prompt evolution in every total. Island copies no longer double
  count cost, read-only connections close after failed reads, and a budgeted
  direction with unknown spend refuses new runs.
- Fixed run cost totals that missed real spend or counted some twice, as
  measured against a stub model's own charge ledger: a resumed run counted
  island copies again and ignored failed proposals; paid steps of a failed
  meta update or final summary were recorded nowhere; surplus proposals
  discarded at the target, paid empty replies and prompt evolutions that
  produced no prompt were dropped; and a results directory holding only
  generation 0 plus failures resumed from zero. One rule
  (`shinka.database.cost`) now totals a run for the runner's resume base and
  `rundb.run_cost`: non-copy programs, one `attempt_log` charge event for each
  paid proposal or prompt evolution that never became a row, and prompt
  evolution. Proposals still in flight at the target finish and are charged
  instead of being cancelled (up to `surplus_proposal_wait_s`, 600 s by
  default; one still running then is cancelled and leaves the marker), a
  deliberate change from upstream, whose two recovery tests in
  `tests/test_async_runner_recovery.py` now expect it. A charge the database
  refuses, or whose amount
  cannot be known, leaves `cost-unrecorded.json`, and the run's total reads
  unknown until the missing charge is recorded durably (deleting the file is
  not that); with `evo.max_api_costs` set a resume then refuses, as does a
  research grant run in any direction, and a run that cannot even leave the
  marker stops with `ChargeNotRecordedError`. The analysis cost tab takes its
  total from the server and lists the rest as other charges, and the compare
  spend curve ends at the run total. A results directory holding only
  generation 0 now resumes instead of running its seed again.
- Fixed `shinka_research run` accepting a `headless/*` model configured in the
  experiment's `task/shinka.yaml`: only `--set` arguments were checked, so a
  headless CLI agent, which works inside the results directory beside the
  out-of-sample metrics, could still mutate a research run. The four model
  lists (`llm_models`, `meta_llm_models`, `novelty_llm_models`,
  `prompt_llm_models`, under `evo` or `evo_config`) are now read as
  `shinka_run` reads them and refused before any grant attempt is taken;
  hypothesis text that merely mentions a headless model still runs.
- Fixed run list totals that dropped what an unknown-cost run had recorded:
  `/get_database_stats` sent `null` for the unknown part and nothing else, so
  the workbench's `≥` total counted such a run's priced charges as zero and
  the restored `/analysis/index.html` showed `—` for the whole total. Run
  statistics now carry `recorded_cost` (everything the run recorded: its cost
  when known, otherwise a lower bound); both run lists sum it and mark the
  total `≥` when a run's cost is unknown or its statistics could not be read.
- Fixed the restored run list painting the prompt-evolution count in white on
  a near-white chip.
- Fixed the Research workbench's trial counts while a run is in progress: the
  overview, the direction header, the hypothesis table and each experiment card
  used the notebook's figure, which `run` writes only when it returns, so a
  running experiment showed fewer trials than evaluated programs. Workspace and
  direction snapshots now count evaluated programs in the run databases, as the
  skeptic counts trials for the DSR, and build every count, best score and trial
  point of one response from a single read of each run. `shinka_research brief`
  and `status`, which the research agent reads each round, count the same way.
- Fixed concurrent `shinka_research` commands discarding each other's notebook
  changes: a probe, skeptic or run that finished after another command saved
  wrote back its stale copy of the whole notebook, and two first events could
  truncate `results.tsv`. Writers now take a per-direction lock
  (`.research.lock`), re-read the notebook, evaluate outside the lock and
  replace `notebook.json` atomically, so readers never see half a file; a
  failed run restores the hypothesis's earlier status only if nothing moved it
  on meanwhile.

- Fixed Anthropic response parsing to dispatch by content-block type, avoiding
  crashes and truncated output for redacted, thinking-only, and multi-block
  responses in PR #188. Thanks @Atharva-Kanherkar.
- Fixed Anthropic and DeepSeek token/cost accounting parity across synchronous
  and asynchronous queries, including thinking-token decomposition and safe
  zero-cost fallback for models missing from the pricing catalog. Runtime
  models.dev refreshes now preserve pinned embedding-price overrides.
- Fixed diff insertions (empty SEARCH blocks) splicing the payload directly
  against the EVOLVE-BLOCK-END marker, which corrupted the marker line, let
  consecutive insertions merge code lines into invalid programs, and caused
  marker validation to reject every insertion patch for block-comment
  languages such as Wolfram. Reported in issue #183. Thanks
  @Atharva-Kanherkar.
- Fixed Gemini retry handling so unsupported structured-output requests fail
  immediately, synchronous retries are paced, and sync/async calls share the
  same default thinking budget.
- Fixed Rust candidate validation on stable toolchains by replacing the
  nightly-only parser flag with isolated, non-mutating `rustfmt` syntax checks
  in PR #179. Thanks @Atharva-Kanherkar.
- Fixed Azure OpenAI LLM client construction to use the v1 base URL contract,
  avoiding doubled `/openai/v1/openai` request paths and 404 responses reported
  in issue #147.
- Fixed OpenAI retry handling so ShinkaEvolve respects provider `retry_after` hints from transient Cloudflare/API 5xx responses.
- Fixed DeepSeek V4 kwargs so ShinkaEvolve omits `temperature` for thinking-mode calls and includes model kwargs in retry error logs.
- Fixed Claude Opus 4.8 kwargs so ShinkaEvolve omits the deprecated `temperature` parameter for Anthropic and Bedrock calls.
- Fixed OpenAI reasoning model kwargs so ShinkaEvolve omits the deprecated `temperature` parameter for GPT-5-series Responses API calls.
- Fixed DeepSeek reasoning model kwargs so ShinkaEvolve passes DeepSeek thinking-mode controls and `reasoning_effort` for V4 models.
- Fixed Google GenAI client setup to detect broken IPv6 connectivity to Google API hosts and prefer IPv4 when needed.
- Fixed LLM pricing boolean metadata normalization so reasoning-model flags handle whitespace and mixed CSV value types.
- Fixed OpenAI Responses parsing to find assistant text by content type while ignoring reasoning/tool text when no message output exists.
- Fixed quiet evaluation mode so `run_shinka_eval(..., verbose=False)` also suppresses result-save framework stdout while preserving the default verbose behavior.

## 0.0.7 - 2026-06-02

### Added

- Added Gemini 3.5 Flash pricing to the Google LLM pricing catalog.
- Added Claude Opus 4.8 pricing entries for the Anthropic API and Amazon Bedrock (`us.anthropic.claude-opus-4-8`) in the LLM pricing catalog.
- Added Wolfram Language as a first-class evolution target: registry entries (`wolfram`, `wl`, `wls`, `mathematica`), code-fence and EVOLVE-BLOCK marker support, and end-to-end coverage in `apply_diff` / `apply_full`.
- Added EVOLVE-BLOCK marker validation in `shinka.edit.marker_validation`, catching the LLM failure mode where a block-comment-language marker is emitted without its closing delimiter (Wolfram, Markdown), which would silently trap the candidate body inside a comment.
- Added the `examples/wolfram_gcd_sum` task: deoptimized seed for `S(N) = sum_{i,j in 1..N} GCD(i,j)`. The evaluator calibrates the baseline on every run by timing the seed through the same `RepeatedTiming` harness as the candidate, and the Wolfram-side timeout is configurable via `WOLFRAM_GCD_MAX_SECONDS`.
- Added Go as a first-class evolution target with `go`/`golang` language registration, `.go` task detection, WebUI failure-artifact support, regression coverage, and the `examples/go_collatz_steps` task.

### Fixed

- Fixed the WebUI dashboard so runs that have completed only the initial generation remain visible instead of being filtered out as empty results.
- Fixed the WebUI Ensembling tab so single-model Headless runs render visible data points and keep the top y-axis tick labels in view.

## 0.0.6 - 2026-05-03

### Added

- Added Headless CLI-backed LLM provider support via `headless/<agent>` model strings, defaulting to `npx -y @roberttlange/headless` for subscription-backed Codex, Claude, and other local agent calls.
- Added Headless startup validation, prompt artifacts, usage/cost parsing, and a `examples/sine_approx_headless` task showing API-free mutation calls with embeddings disabled.
- Added Vertex AI authentication support for Gemini LLM and embedding clients in PR #125. Thanks @wu375.
- Added async-runner validation for configured LLM and embedding model environment access before run artifacts are created in PR #127. Thanks @RobertTLange.
- Added GPT-5.5 and GPT-5.5 Pro entries to the OpenAI LLM pricing catalog.
- Added Fortran evolution support, including language detection, patch application, validation, visualization metadata, and a compiled heat-diffusion example in PR #131.

### Fixed

- Fixed bandit sampler resume from legacy or changed `llm_models` state so saved per-arm arrays are resized, name-aligned when possible, and cost-aware UCB range state remains active after loading `bandit_state.pkl` in PR #130, addressing issue #129.
- Fixed the WebUI embedding similarity heatmap so hydrated programs with empty embeddings no longer leave the tab stuck on `Loading full embedding data...`, and stale cached full-program data is refetched after summary updates.

## 0.0.5 - 2026-04-22

### Added

- Added the GitHub Pages documentation website for `shinka-evolve`.
- Added interactive async-throughput and UCB bandit-selection demos to the documentation website.
- Added dashboard sorting controls to the local WebUI so result cards can be reordered by the active setting.
- Added a `Hide Plot` / `Show Plot` toggle for the WebUI Throughput tab runtime timeline while keeping the plot visible by default.
- Added Claude Opus 4.7 pricing entries for the Anthropic API and Amazon Bedrock (`anthropic.claude-opus-4-7`) in the LLM pricing catalog.

### Changed

- Changed `shinka_run` startup output to use a minimal `Shinka CLI` banner while other launch paths keep the full gradient banner.
- Changed `enable_controlled_oversubscription` to default to `false` across the shared `EvolutionConfig` baseline and packaged Hydra evolution presets.
- Changed the WebUI meta header to show the active results directory directly in the info panel.
- Changed the WebUI Throughput tab runtime timeline to scale its height with worker-lane count so each worker keeps a dedicated visible row.

### Fixed

- Fixed async run-time regressions from proposal-failure persistence by keeping terminal failed proposals in `attempt_log` plus `failure.json` artifacts instead of inserting them into the main `programs` table during evolution runs.
- Fixed the local WebUI to render failed proposal nodes from `attempt_log` plus `failure.json` so failure lineage remains visible without storing synthetic failed programs in the main results database.
- Fixed failed terminal proposal and prompt-evolution cost accounting so the runtime API budget counter now reflects those spend buckets, and added runtime-timeline metadata for failed proposal nodes rendered from `attempt_log`.
- Fixed bandit summary tables to preserve readable `local/<model>` and `openrouter/<model>` labels while stripping endpoint and API-key query details from local OpenAI-compatible model labels.
- Fixed the LLM bandit sampling summary to use the same 120-column table width as the program, patch, and other Rich summaries.
- Fixed the default `AsymmetricUCB` bandit summary to omit the `div` and `log mean` columns for a more compact Rich table.
- Fixed the compare view so negative best scores remain visible instead of being clamped away by the WebUI summary logic.
- Fixed documentation links and metadata URLs to use the deployed GitHub Pages path capitalization (`/ShinkaEvolve/`).
- Fixed oversubscription regression coverage and docs so adaptive proposal backlog behavior is now clearly opt-in instead of implied by defaults.
- Fixed Gemini client timeout handling so the shared second-based timeout is converted to the millisecond unit expected by `google-genai`, avoiding accidental `1.2s` read timeouts on long-running requests.
- Fixed async runtime regressions from sampling/evaluation worker-lane persistence by falling back to timestamp-based throughput lane inference instead of storing those lane IDs in program metadata.
- Fixed local WebUI cache staleness by disabling browser caching for the main HTML shells (`index.html`, `viz_tree.html`, `compare.html`) served by the local visualization server.
- Fixed WebUI Throughput tab hydration so `right_tab=throughput` restores reliably after data loads and the generation runtime timeline renders pool stages using timestamp-inferred sampling/evaluation lanes on the current Plotly build.

## 0.0.4 - 2026-04-06

### Added

- Added a new `shinka_models` CLI that inspects the current environment plus discovered `.env` files and reports which priced LLM and embedding models are available to use.
- Added per-model `api_key_env` support for local OpenAI-compatible LLM and embedding backends, allowing credentials to be selected inline via `local/<model>@http(s)://...?...api_key_env=ENV_VAR`.

### Changed

- Changed `shinka_models` default output to a compact JSON object with separate `embedding` and `llm` model lists, while `--verbose` now emits provider-level availability details with the same top-level lists.
- Updated the `shinka-run` skill so run planning must validate mutation, meta-recommendation, prompt-evolution, and embedding models against `shinka_models` before launching evolution.
- Improved local async runtime scaling by launching evaluation subprocesses with the active project interpreter, capping per-process numeric-library thread fan-out, and reducing local monitor polling latency.
- Moved prompt-fitness percentile recomputation off the prompt side-effect hot path and onto a debounced background task using fresh read-only database connections.
- Updated the circle-packing scaling presets to run for `100` generations and reduced `max_patch_resamples` to `1` for the small / medium / large benchmark configs.

### Fixed

- Fixed adaptive proposal targeting so invalid `proposal_target_hard_cap` values below evaluation capacity no longer silently disable oversubscription.
- Fixed prompt-percentile background refresh to avoid SQLite thread-affinity failures when recomputing prompt fitness during async runs.
- Fixed Windows Unicode handling for async candidate-file I/O, diff summaries, and best-path exports by forcing UTF-8 reads and writes on LLM-generated artifacts.

## 0.0.3 - 2026-04-04

### Added

- Added local OpenAI-compatible embedding endpoint support via `evo.embedding_model=local/<model>@http(s)://host[:port]/v1`.
- Added `CONTRIBUTING.md` plus GitHub issue and pull-request templates to document the contribution flow.
- Added Python throughput plotting utilities in `shinka.plots` for generation runtime timelines and normalized occupancy-over-time views.
- Added a durable SQLite `generation_event_log` journal for async generation lifecycle debugging, including stop and persistence-failure events.
- Added regression coverage for the new Python throughput plotting helpers, including pool-slot prep, occupancy math, and legend/layout behavior.
- Added regression coverage for concurrent async completed-job persistence so multi-worker postprocessing throughput stays exercised.
- Added regression coverage for lightweight program summaries and WebUI embed-tab hydration so similarity matrices only render from fully loaded embedding data.

### Changed

- Reworked async completion handling so completed scheduler jobs are detected immediately, evaluation slots are released before persistence finishes, and shutdown now waits for queued completed-job batches plus post-persistence side effects to drain.
- Moved database archive / best-program / island maintenance off the insert hot path via deferred replay hooks, while letting async writers use fresh worker-local connections and merge runtime metadata back into the shared DB state.
- Expanded pipeline timing metadata with post-evaluation queue wait, post-persistence side-effect timing, and summary statistics for end-to-end async throughput analysis.
- Tuned `examples/circle_packing/shinka_long.yaml` for a smaller long-run preset and ignored generated `results*` / `shinka_scale*` artifacts in the repo root.
- Renamed the local backend guide from `docs/support_local_llm.md` to `docs/support_local_models.md` and expanded it to cover local embedding backends alongside local LLMs.
- Refactored async code validation to use a shared subprocess helper across Python, Rust, Swift, JSON, and C++ validators without changing validation behavior.
- Updated `examples/circle_packing/load_results.ipynb` to include the new throughput plots at the bottom of the notebook.
- Updated `examples/circle_packing/load_results.ipynb` and `examples/circle_packing/shinka_long.yaml` for the latest large async circle-packing run analysis setup.
- Refined Python throughput plot legends to use compact centered panels below each subplot for cleaner notebook rendering.
- Reduced the async generation journal to high-signal failure/stop events only so persistence debugging does not add heavy hot-path overhead.

### Fixed

- Fixed completion-time accounting so retried or duplicate-persisted jobs keep the original scheduler completion timestamp instead of inflating evaluation duration.
- Fixed the async job monitor to finalize cleanly once the generation target is reached, even when no jobs remain active at the polling boundary.
- Fixed high-concurrency SQLite persistence regressions by covering deferred maintenance replay, multi-writer overlap, and shutdown drain behavior with new recovery and database tests.
- Fixed async proposal scheduling so `num_generations` is now a hard cap on assigned proposal generations instead of launching extra `gen_*` attempts to compensate for failed or discarded work.
- Fixed async evaluation slot lifecycle bugs so local evaluation concurrency no longer exceeds `max_evaluation_jobs` through stale double-release of reassigned worker slots.
- Fixed retry-time completion accounting so successful DB retries refresh async progress before the hard generation-budget stop condition is evaluated.
- Fixed async database retry races by treating in-flight `source_job_id` inserts as already claimed, preventing duplicate persisted programs while timed-out writes are still finishing in worker threads.
- Fixed async resume/recovery bookkeeping so restarted runs continue from the number of persisted completed programs instead of stopping early when failed proposals or hung local evals left gaps in generation IDs.
- Fixed WebUI meta-analysis labeling so `meta_*.txt` snapshots are presented as meta updates / processed-count checkpoints instead of misleading generation numbers.
- Fixed duplicate-retry recovery so already-persisted jobs replay post-persistence side effects exactly once, restoring missing meta-memory updates after DB timeouts.
- Fixed SQLite persistence stability by increasing busy timeouts and the outer async DB-add timeout for long high-concurrency runs.
- Fixed Python throughput plot preparation so frames without optional metadata columns like `is_island_copy`, `patch_name`, or `model_name` still render correctly.
- Fixed legacy throughput accounting in both the Python plotter and WebUI Throughput tab so reused worker lanes no longer show impossible peaks like `31/20`.
- Fixed the WebUI embed tab so summary-only loads or single lazily hydrated programs no longer produce a misleading 1x1 similarity matrix instead of the full run.

## 0.0.2 - 2026-03-22

### Added

- Added adaptive proposal oversubscription controls and documentation for bounded async proposal backlogs.
- Added a new `Throughput` tab in the WebUI with runtime timeline, worker occupancy, normalized occupancy, occupancy distribution, completion-rate, and utilization summaries.
- Added regression coverage for WebUI runtime timeline and Throughput-tab structure.

### Changed

- Improved async runtime accounting so evaluation timing starts at evaluation-slot acquisition instead of scheduler submission.
- Improved runtime timeline rendering in the WebUI, including better legend placement and spawned-island copy deduplication for resource-usage plots.
- Improved the embedding similarity matrix layout so large runs preserve cell size and scroll cleanly instead of collapsing visually.
- Expanded `docs/async_evolution.md` with detailed explanations of controlled oversubscription settings and tuning heuristics.

### Fixed

- Prevented duplicate async program persistence for the same completed scheduler job by treating `source_job_id` writes as idempotent.
- Fixed inflated runtime timeline peaks caused by counting spawned-island copies as separate runtime jobs.
- Fixed runtime timeline legend overlap and related layout issues in the WebUI.

## 0.0.1 - 2026-03-12

First PyPI release for `shinka-evolve`.

### Added

- Initial PyPI release for `shinka-evolve`.
- Trusted publishing workflow via GitHub Actions.
- Packaged Hydra presets and release artifact checks for PyPI builds.
- Added a full async pipeline via `ShinkaEvolveRunner` for concurrent proposal generation and evaluation.
- Added prompt co-evolution support, including a system-prompt archive, prompt mutation, and prompt fitness tracking.
- Added island sampling strategies via `shinka/database/island_sampler.py` (`uniform`, `equal`, `proportional`, `weighted`).
- Added fix-mode prompts for incorrect-only populations.
- Added new plotting modules:
  - `shinka/plots/plot_costs.py`
  - `shinka/plots/plot_evals.py`
  - `shinka/plots/plot_time.py`
  - `shinka/plots/plot_llm.py`
- Added new documentation:
  - `docs/async_evolution.md`
  - `docs/design/dynamic_evolve_markers.md`
  - `docs/design/evaluation_cascades.md`
- Added the `examples/game_2048` example.

### Changed

- Refactored the API around `ShinkaEvolveRunner` and the async evolution pipeline.
- Added prompt co-evolution, expanded island logic, provider-based LLM and embedding modules, and a major WebUI refresh.
- Preserved original shorthand launch syntax such as `variant=...`, `task=...`, `database=...`, and `cluster=...`.
- Expanded island and parent sampling logic, including dynamic island spawning on stagnation.
- Refactored the LLM and embedding stack into provider-based modules.
- `ShinkaEvolveRunner` gained stronger resume behavior, fix-mode sampling fallback, and richer metadata/cost accounting.
- Database model expanded with dynamic island spawning controls, island selection strategy config, and `system_prompt_id` lineage.
- `shinka/core/wrap_eval.py` gained per-run process parallelism, deterministic ordering, clearer worker error surfacing, early stopping, optional plot artifacts, and NaN/Inf score guards.
- WebUI backend/frontend expanded with summary/count/details/prompts/stats/plots endpoints plus dashboard and compare views.
- README and install docs were updated to prefer PyPI install and document `--config-dir` for user-defined presets.
- `pyproject.toml` packaging/dependency updates:
  - `google-generativeai` -> `google-genai`
  - added `psutil`
  - pinned `httpx==0.27`
  - updated setuptools packaging config

### Cost Budgeting

- `max_api_costs` became a first-class runtime budget guard in evolution runners.
- Budget checks use committed cost:
  - realized DB costs (`api_costs`, `embed_cost`, `novelty_cost`, `meta_cost`)
  - plus estimated cost of in-flight work
- Once the budget is reached, new proposals stop and the runner drains ongoing jobs.
- If `num_generations` is omitted, `max_api_costs` is required to bound the run.
