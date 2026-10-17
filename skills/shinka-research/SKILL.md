---
name: shinka-research
description: Run hypothesis-driven quant research on top of ShinkaEvolve with the `shinka_research` CLI. Use when the user gives a research direction or idea (not just parameters to tune) and wants falsifiable hypotheses proposed, tested by evolution against a frozen walk-forward evaluator, challenged on an untouched holdout, and recorded as verdicts in a research notebook.
---

# Shinka Research Skill
Turn a research direction into evidence-backed verdicts, autoresearch style:
the human writes `program.md`, you act as the researcher, Shinka evolves the
code, and a frozen evaluator plus a skeptic judge the result.

## When to Use
- The user gives a direction ("does order-flow imbalance predict short-term
  returns on BTC perps?") or an idea they want explored, not a single program
  to optimize.
- The user has bar data (CSV with a timestamp column and `close`).

Do not use this skill for plain program optimization with an existing task
directory (use `shinka-run`) or for non-quant tasks (not supported yet).

## Roles and hard rules
- **The human owns `program.md` and the evaluator settings** (data, costs,
  splits, thresholds, budget). You never edit `task/evaluate.py` or the data
  after `scaffold`; `run` and `skeptic` refuse tampered experiments.
- **Never look at out-of-sample numbers while designing.** Score variants with
  `shinka_research probe` (public metrics only). Do not open `metrics.json`
  files, `skeptic.md`, or skeptic evidence of an experiment until it is
  finished; steer only with `combined_score`, `public` and `text_feedback`.
- **Every attempt goes through `shinka_research`** (scaffold/run), so it is
  counted as a trial for the Deflated Sharpe Ratio. Do not test strategy
  variants on the holdout by hand.
- **Stop for the human twice**: after proposing hypotheses (they choose) and
  after verdicts (they review). Skip a checkpoint only if `program.md` says so.
- **The lockbox is the human's.** The tail of the data is sealed at the first
  `scaffold` and is opened once per direction by `confirm --yes`. Never run
  `confirm` unless the human says to close the direction.
- **Respect the budget** in `program.md`: hypotheses, generations, and USD
  (pass `--set evo.max_api_costs=<usd>` to each run).

## Workflow

### 0. Set up the direction
```bash
shinka_research init <dir> --direction "<one sentence>"
```
Ask the human to fill `<dir>/program.md` (data path, bar size, costs, budget,
thresholds, prior knowledge). If a notebook already exists, run
`shinka_research status <dir>` and read every `hypotheses/*.md` and
`experiments/*/*/skeptic.md` first: build on verdicts, do not repeat refuted
ideas unless you state what is different.

### 1. Research and propose hypotheses
- Search the literature and practitioner sources for the direction (web
  search; papers, known anomalies, market microstructure). Note what is known
  to work, what decayed, and typical pitfalls.
- Write 3–8 hypotheses with **different mechanisms**, each falsifiable on the
  available data. Use `templates/hypothesis.md`; cite sources under
  "Related work".
- Add each one:
```bash
shinka_research hypothesis add <dir> --title "<short claim>" --file h.md
```
- **Checkpoint:** present the list (id, claim, mechanism, why it might fail)
  and let the human select. Mark the rest:
```bash
shinka_research hypothesis status <dir> <hid> selected   # or dropped
```

### 2. Design the experiment (AI-Scientist stage 1: initial implementation)
For each selected hypothesis write a seed `initial.py`: the simplest faithful,
causal expression of the claim. Copy the structure of
`examples/quant_research/initial.py`: logic inside `EVOLVE-BLOCK`,
`run_experiment(bars)` outside it returning positions in [-1, 1] on the same
index. Then freeze the evaluator with the settings from `program.md`:
```bash
shinka_research scaffold <dir> <hid> --initial initial.py --data <bars.csv> \
    --lockbox-fraction 0.2 --cost-bps <bps> --periods-per-year <n> \
    --oos-fraction 0.3 --n-folds 3
```
`--data` and `--lockbox-fraction` are fixed by the direction's first
experiment; later experiments omit `--data` (other data needs a new
direction). Then check the seed:
```bash
shinka_research probe <dir> <hid> <dir>/experiments/<hid>/e01/task/initial.py
```
`probe` scores a program with the frozen evaluator and prints only public
in-sample metrics; it must report `"correct": true`.
If the seed is incorrect (look-ahead, NaN, wrong index), fix the seed and
scaffold a new experiment — never edit the frozen task.

### 3. Evolve (stages 2–3: baseline tuning, then creative research)
Runs on the same experiment resume; `--num-generations` is the total target.
```bash
# Stage 2: tune the faithful version with small diffs
shinka_research run <dir> <hid> --num-generations 10 -- \
    --set 'evo.patch_types=["diff"]' --set 'evo.patch_type_probs=[1.0]' \
    --set evo.max_api_costs=<usd>
# Stage 3: explore richer expressions of the same mechanism
shinka_research run <dir> <hid> --num-generations 40 -- \
    --set 'evo.patch_types=["diff","full","cross"]' \
    --set 'evo.patch_type_probs=[0.5,0.3,0.2]' --set evo.max_api_costs=<usd>
```
Choose models from `shinka_models` (`--set evo.llm_models='[...]'`). Between
stages, inspect progress with the `shinka-inspect` skill (public metrics only)
and stop early if the in-sample score has clearly plateaued.

### 4. Ablate (stage 4: what actually matters)
Copy the best program (`shinka-inspect`, or the top program in the run), remove
one component at a time, and score each variant with the frozen evaluator:
```bash
shinka_research probe <dir> <hid> ablation_no_filter.py
```
Record which components carry the in-sample edge in the hypothesis file.

### 5. Challenge and decide
```bash
shinka_research skeptic <dir> <hid>      # DSR, PBO, sub-periods, baselines, decay
shinka_research verdict <dir> <hid>      # records the rule-based verdict
```
Use the thresholds from `program.md` (`--dsr`, `--min-positive-fraction`,
`--subperiods`, `--max-pbo`). These verdicts use the validation segment,
which every round reuses, so treat `supported` as development-grade evidence. Only override with `--label ... --reason "..."` when the
rule clearly misreads the evidence (e.g. the edge is entirely from one
crisis), and say so in the report. "Inconclusive" is a valid result.

### 6. Report
Write `<dir>/REPORT.md` (template: `templates/report.md`): question, what was
tried, verdict per hypothesis with key evidence (OOS Sharpe, DSR with trial
count, baselines, ablation findings), what this rules out, and the most
promising next hypotheses. **Checkpoint:** hand it to the human.

### 7. Next round
New hypotheses should follow from verdicts: deepen supported mechanisms
(robustness, other instruments, capacity), and explain refuted ones. Every new
experiment increases the trial count, so the bar for "supported" rises — that
is intended.

### 8. Close the direction: lockbox, then forward test
When the human decides the direction is done:
```bash
shinka_research confirm <dir>            # shows what will be opened
shinka_research confirm <dir> --yes      # opens the lockbox, once
```
Every `supported` hypothesis becomes `confirmed` or `unconfirmed`
(`CONFIRMATION.md`). Add the outcome to `REPORT.md`. The lockbox cannot be
reopened; further work on the same idea needs a new direction with new data.
The final gate is outside this tool: run `confirmed` strategies on a paper
trading account with data that did not exist when the research started, and
only then consider real capital.

## Notebook layout
```
<dir>/program.md            human: direction, constraints, budget, thresholds
<dir>/notebook.json         index: hypotheses, statuses, experiments, verdicts
<dir>/hypotheses/<hid>.md   one hypothesis per file
<dir>/experiments/<hid>/<eid>/task/      frozen evaluate.py, initial.py, shinka.yaml
<dir>/experiments/<hid>/<eid>/results/   Shinka run (programs.sqlite)
<dir>/experiments/<hid>/<eid>/skeptic.md and best_program.py
<dir>/data/dev.csv          development data (the lockbox tail is not in it)
<dir>/REPORT.md
<dir>/CONFIRMATION.md       lockbox results (after `confirm`)
```

Design rationale: `docs/research/design.md` in the ShinkaEvolve fork.
