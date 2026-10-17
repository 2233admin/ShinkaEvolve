# Quant research example

A minimal research direction for `shinka_research`: does time-series momentum
survive transaction costs on a (synthetic) instrument?

```bash
cd examples/quant_research
python make_data.py bars.csv                       # or bring your own CSV: timestamp,close,...

shinka_research init runs/momentum --direction "Time-series momentum"
shinka_research hypothesis add runs/momentum \
    --title "TS momentum survives costs" --file hypothesis.md
shinka_research scaffold runs/momentum h001 --initial initial.py --data bars.csv
shinka_research run runs/momentum h001 --num-generations 20 \
    -- --set 'evo.llm_models=["gpt-5-mini"]'       # anything after -- goes to shinka_run
shinka_research status runs/momentum
```

`scaffold` freezes the evaluator (`task/evaluate.py`, config embedded) and
hashes it together with the data; `run` refuses to start if either changed.
The mutation LLM only sees in-sample walk-forward metrics; the out-of-sample
holdout is stored as private metrics for the skeptic.

After evolution, challenge the result on the holdout and record the verdict:

```bash
shinka_research skeptic runs/momentum h001   # DSR, PBO, sub-periods, baselines
shinka_research verdict runs/momentum h001
```

The first `scaffold` seals the last 20% of the data as a lockbox (only
`data/dev.csv` reaches experiments). When the direction is finished, open it
once to confirm the supported hypotheses:

```bash
shinka_research confirm runs/momentum --data bars.csv --yes
```

To let a coding agent drive the whole loop (literature search, hypothesis
tree, staged evolution, ablations, report), point it at
`skills/shinka-research/SKILL.md`.

See `docs/research/design.md` for the full design.
