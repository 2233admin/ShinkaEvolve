---
name: shinka-research
description: Run hypothesis-driven quant research on top of ShinkaEvolve with the `shinka_research` CLI. Use when the user gives a research direction or idea (not just parameters to tune) and wants falsifiable hypotheses proposed from a curated research library, tested by evolution against a frozen walk-forward evaluator, challenged on an untouched holdout, and recorded as verdicts that feed back into the library.
---

# Shinka Research Skill
Turn a research direction into evidence-backed verdicts. Two ideas from
Andrej Karpathy are combined:

- **autoresearch**: the human writes `program.md` ("programming the program"),
  you act as the researcher, Shinka evolves the code, and a frozen evaluator
  nobody may touch judges it. Every attempt is logged (`results.tsv`).
- **LLM knowledge base**: the human drops sources into `library/raw/`; you
  compile them into an interlinked markdown wiki, answer research questions
  from it, lint it, and file your findings back, so knowledge compounds across
  directions.

## When to Use
- The user gives a direction ("does order-flow imbalance predict short-term
  returns on BTC perps?") or an idea to explore, not one program to optimize.
- The user has bar data (CSV with a timestamp column and `close`).

Not for plain program optimization (use `shinka-run`) or non-quant tasks.

## Roles and hard rules
- **The human owns `program.md`, the evaluator settings and the raw sources.**
  You never edit `task/evaluate.py` or the data after `scaffold`; `run`,
  `probe` and `skeptic` refuse tampered experiments. You never edit or delete
  files in `library/raw/`; you may add web sources you found (step 1).
- **Never look at out-of-sample numbers while designing an experiment.**
  Score variants with `shinka_research probe` (public metrics only). Do not
  open `metrics.json`, `skeptic.md` or skeptic evidence of an experiment until
  it is finished. Finished verdicts (validation-segment evidence) may inform
  later rounds; that reuse is what the trial count and the lockbox account for.
- **Every attempt goes through `shinka_research`** (scaffold/run/probe), so it
  counts as a trial for the Deflated Sharpe Ratio.
- **Checkpoints**: stop for the human after proposing hypotheses and after
  verdicts, unless `program.md` sets `Autonomy: autonomous`. In autonomous
  mode never pause to ask whether to continue: keep going until the budget is
  spent or the human interrupts. When out of ideas, re-read the wiki, combine
  near-misses, or try a different mechanism.
- **The lockbox is the human's.** Never run `confirm` unless the human says to
  close the direction.
- **Budget**: `init --budget-usd` makes `run` refuse once the direction's LLM
  spend reaches it; also pass `--set evo.max_api_costs=<usd>` per run.
- **Simplicity**: when two programs score about the same in-sample, prefer
  the simpler one. A small change with a clear gain beats a big rewrite with a
  marginal one.

## Workspace layout
```
<workspace>/library/raw/         human sources: PDFs, papers, clips, notes, links.md
<workspace>/library/wiki/        your pages: concepts, papers, findings
<workspace>/library/index.md     catalog of wiki pages (you maintain it)
<workspace>/library/log.md       ingests, compiles, lints, findings
<workspace>/<direction>/         one research direction (notebook, experiments)
```
One library serves every direction in the workspace.

## Workflow

### 0. Set up and get the brief
```bash
shinka_research init <workspace>/<direction> --direction "<one sentence>" --budget-usd <usd>
shinka_research brief <workspace>/<direction>
```
Ask the human to fill `program.md` (data, costs, budget, thresholds, autonomy)
and to drop sources into `library/raw/` (or `shinka_research library add
<dir> <file-or-url>`). Read the brief every round before proposing anything:
it lists hypotheses with verdicts and evidence, trials, spend, the lockbox
state, uncompiled sources and recent events.

### 1. Compile the library, then research
1. `shinka_research library pending <dir>` lists new or changed raw sources.
   For each: read it (fetch every URL in `links.md`), then create or update
   wiki pages using `templates/wiki_page.md`: one page per paper/source and
   one per concept (mechanism, anomaly, pitfall), cross-linked with
   `[[page-name]]`, each claim citing its raw source. Update `index.md`.
   Then `shinka_research library compiled <dir> <raw/...>` (or `--all`).
2. Research gaps with web search. Save useful sources into `raw/` with
   `library add` (so they get compiled and cited like the human's), never
   from memory alone.
3. Ask the wiki the direction's question: what is known to work, what decayed,
   which mechanisms are untested. When an answer is worth keeping, write it as
   a new wiki page and `shinka_research library log <dir> qa "<question>"`.

### 2. Propose hypotheses
Write 3–8 hypotheses with **different mechanisms**, each falsifiable on the
available data, using `templates/hypothesis.md` and citing wiki pages under
"Related work". Do not repeat refuted ideas unless you state what is different.
```bash
shinka_research hypothesis add <dir> --title "<short claim>" --file h.md
```
**Checkpoint:** present id, claim, mechanism and why it might fail; the human
selects. `shinka_research hypothesis status <dir> <hid> selected|dropped`.

### 3. Design the experiment (stage 1: initial implementation)
Write a seed `initial.py`: the simplest faithful, causal expression of the
claim, structured like `examples/quant_research/initial.py` (logic inside
`EVOLVE-BLOCK`, `run_experiment(bars)` returning positions in [-1, 1]).
```bash
shinka_research scaffold <dir> <hid> --initial initial.py --data <bars.csv> \
    --lockbox-fraction 0.2 --cost-bps <bps> --periods-per-year <n> \
    --oos-fraction 0.3 --n-folds 3
shinka_research probe <dir> <hid> <dir>/experiments/<hid>/e01/task/initial.py
```
`--data` and `--lockbox-fraction` are fixed by the direction's first
experiment; later ones omit `--data`. `probe` must report `"correct": true`;
otherwise fix the seed and scaffold a new experiment (never edit the task).

### 4. Evolve (stages 2–3: baseline tuning, then creative research)
Runs resume; `--num-generations` is the total target.
```bash
shinka_research run <dir> <hid> --num-generations 10 -- \
    --set 'evo.patch_types=["diff"]' --set 'evo.patch_type_probs=[1.0]' \
    --set evo.max_api_costs=<usd>
shinka_research run <dir> <hid> --num-generations 40 -- \
    --set 'evo.patch_types=["diff","full","cross"]' \
    --set 'evo.patch_type_probs=[0.5,0.3,0.2]' --set evo.max_api_costs=<usd>
```
Choose models from `shinka_models`. Between stages inspect progress with
`shinka-inspect` (public metrics only); stop early on a clear plateau.
Add `--detach` to keep a run going after the window closes: it returns at
once, the output goes to `<dir>/experiments/<hid>/<eid>/run.log`, and the run
records its own end. `shinka_research watch <dir>` (a direction or a whole
workspace) records the end of a run whose processes vanished without doing
so; it prints nothing when there is nothing new. One experiment runs once at
a time; different experiments can run side by side.

### 5. Ablate (stage 4)
Remove one component of the best program at a time and score each variant:
`shinka_research probe <dir> <hid> ablation_x.py`. Record what carries the
in-sample edge in the hypothesis file.

### 6. Challenge and decide
```bash
shinka_research skeptic <dir> <hid>      # DSR, PBO, sub-periods, baselines, decay
shinka_research verdict <dir> <hid>      # rule-based verdict
```
Use the thresholds from `program.md`. Override only with `--label ...
--reason "..."` when the rule clearly misreads the evidence. "Inconclusive" is
a valid result.

### 7. File the finding back into the library
For each verdict, create or update a `finding` wiki page for the mechanism
(`templates/wiki_page.md`): direction, data and period, verdict, validation
Sharpe, DSR with trial count, PBO, what the ablations showed, link to the
hypothesis. Never write lockbox numbers before `confirm`. Update `index.md`
and `shinka_research library log <dir> finding "<hid>: <verdict>"`.

### 8. Lint the library (each round)
Check the wiki for contradictions (two pages disagreeing, a finding that
contradicts a cited paper), claims without a source, stale findings, missing
pages for frequently mentioned concepts, and promising connections. Fix what
you can, list what needs the human, and
`shinka_research library log <dir> lint "<summary>"`.

### 9. Report, then next round
Write `REPORT.md` (template: `templates/report.md`). **Checkpoint** (unless
autonomous). Then `brief` again: new hypotheses should follow from verdicts
and the wiki. Every experiment raises the trial count and the bar for
"supported" — intended.

### 10. Close the direction: lockbox, then forward test
When the human decides the direction is done:
```bash
shinka_research confirm <dir> --data <bars.csv>          # shows what opens
shinka_research confirm <dir> --data <bars.csv> --yes    # opens it, once
```
All `supported` hypotheses become `confirmed` or `unconfirmed`
(`CONFIRMATION.md`). If `confirm` stops midway (a crash, Ctrl-C), run the same
`--yes` command again: it finishes the hypotheses it froze under the rules it
froze, and any whose program had already started on the lockbox becomes
`unconfirmed` (interrupted) instead of running again. It does not open the
lockbox while a `run` is still under way (a `probe` counts as a trial as soon as
it starts); until it finishes, `run`, `probe`, `skeptic`, `verdict` and
`scaffold` refuse. Now add lockbox results
to the finding pages and the report. The final gate is outside this tool:
paper-trade `confirmed` strategies on data that did not exist when the
research started.

## Direction layout
```
<dir>/program.md            human: direction, constraints, budget, autonomy
<dir>/notebook.json         hypotheses, statuses, experiments, verdicts, budget
<dir>/results.tsv           one line per event (scaffold, run, skeptic, verdict, confirm)
<dir>/hypotheses/<hid>.md
<dir>/experiments/<hid>/<eid>/{task,results,skeptic.md,best_program.py}
<dir>/data/dev.csv          development data (the lockbox tail is not in it)
<dir>/REPORT.md, CONFIRMATION.md
```

Design rationale: `docs/research/design.md` in the ShinkaEvolve fork.
