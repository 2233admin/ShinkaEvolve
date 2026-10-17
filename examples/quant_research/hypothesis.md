## Statement
Trailing returns carry information about the next bar's return on this
instrument (time-series momentum), and the effect survives 5 bps costs.

## Test
Evolve a causal position rule seeded with 20-bar momentum. Score on
in-sample walk-forward Sharpe; judge on the untouched out-of-sample holdout.

## Supported if
Out-of-sample DSR >= 0.95 and at least 2/3 of out-of-sample sub-periods have
positive Sharpe.

## Refuted if
Out-of-sample Sharpe <= 0.

## Risks
Regime dependence; high turnover eating the edge; overfitting the in-sample folds.
