"""Skeptic: try to break an experiment's best result on the untouched holdout.

Checks (all on out-of-sample returns the mutation LLM never saw):

* Deflated Sharpe Ratio (Bailey & López de Prado, 2014). The number of trials
  is every program evaluated across the whole research direction; the
  dispersion of trial Sharpe ratios is estimated from the in-sample scores of
  the correct programs (an approximation: OOS Sharpe of every trial is not
  computed).
* Sub-period stability: share of OOS segments with positive Sharpe.
* Baselines: buy-and-hold and the seed program over the same holdout.
* In-sample -> out-of-sample decay.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats

from shinka.research import rundb
from shinka.research.notebook import Experiment, Notebook, Status, experiment_dir
from shinka.research.quant import (
    QuantConfig,
    load_bars,
    positions_for,
    sharpe,
    split_index,
    strategy_returns,
)
from shinka.research.scaffold import load_config, verify_experiment

EULER_GAMMA = 0.5772156649015329


class Thresholds(BaseModel):
    dsr_supported: float = 0.95
    min_positive_fraction: float = 2.0 / 3.0
    n_subperiods: int = 3


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` null strategies."""
    if n_trials <= 1 or var_trials <= 0.0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_trials) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(returns: pd.Series, n_trials: int, var_trials: float) -> float:
    """Probability that the true per-period Sharpe exceeds the multiple-testing
    benchmark, accounting for sample length, skewness and kurtosis."""
    r = returns.to_numpy(dtype=float)
    t = len(r)
    std = r.std(ddof=1) if t > 1 else 0.0
    if t < 3 or std == 0.0:
        return 0.0
    sr = r.mean() / std
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    sr0 = expected_max_sharpe(n_trials, var_trials)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0.0:
        return 0.0
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(t - 1) / math.sqrt(denom)))


def positive_fraction(returns: pd.Series, n: int, periods_per_year: float) -> float:
    parts = [p for p in np.array_split(returns.to_numpy(), n) if len(p) > 1]
    if not parts:
        return 0.0
    return sum(sharpe(pd.Series(p), periods_per_year) > 0 for p in parts) / len(parts)


def decide(report: Dict[str, Any], th: Thresholds) -> tuple[Status, str]:
    if report["oos_sharpe"] <= 0.0:
        return Status.refuted, f"OOS Sharpe {report['oos_sharpe']:.2f} <= 0"
    if (
        report["dsr"] >= th.dsr_supported
        and report["oos_positive_fraction"] >= th.min_positive_fraction
    ):
        return Status.supported, (
            f"DSR {report['dsr']:.3f} >= {th.dsr_supported} over "
            f"{report['n_trials']} trials; {report['oos_positive_fraction']:.0%} "
            "of OOS sub-periods positive"
        )
    return Status.inconclusive, (
        f"OOS Sharpe {report['oos_sharpe']:.2f} > 0 but DSR {report['dsr']:.3f} "
        f"(needs {th.dsr_supported}) or sub-period stability "
        f"{report['oos_positive_fraction']:.0%} (needs "
        f"{th.min_positive_fraction:.0%}) falls short"
    )


def _oos_returns(
    program: Path, bars: pd.DataFrame, cfg: QuantConfig
) -> tuple[pd.Series, pd.Series]:
    rets = strategy_returns(positions_for(program, bars), bars["close"], cfg.cost_bps)
    cut = split_index(len(bars), cfg.oos_fraction)
    return rets.iloc[:cut], rets.iloc[cut:]


def challenge(
    root: Path, nb: Notebook, hid: str, exp: Experiment, th: Thresholds
) -> Dict[str, Any]:
    """Run all checks for one experiment and return the evidence report."""
    verify_experiment(root, hid, exp)
    exp_dir = experiment_dir(root, hid, exp.id)
    task_dir, results_dir = exp_dir / "task", exp_dir / "results"
    cfg = load_config(task_dir)
    bars = load_bars(Path(cfg.data_path), cfg.timestamp_col)
    ppy = cfg.periods_per_year

    pid, code, is_score = rundb.best_program(results_dir)
    best_path = exp_dir / "best_program.py"
    best_path.write_text(code)
    is_rets, oos = _oos_returns(best_path, bars, cfg)
    _, seed_oos = _oos_returns(task_dir / "initial.py", bars, cfg)
    hold = bars["close"].pct_change().fillna(0.0)
    hold_oos = hold.iloc[split_index(len(bars), cfg.oos_fraction) :]

    scores: List[float] = []
    for h in nb.hypotheses:
        for e in h.experiments:
            rd = experiment_dir(root, h.id, e.id) / "results"
            if (rd / rundb.DB_FILENAME).exists():
                scores.extend(rundb.correct_scores(rd))
    per_period = np.asarray(scores) / math.sqrt(ppy)
    var_trials = float(per_period.var(ddof=1)) if len(per_period) > 1 else 0.0
    n_trials = max(nb.total_trials, 1)

    oos_sharpe = sharpe(oos, ppy)
    is_sharpe = sharpe(is_rets, ppy)
    return {
        "program_id": pid,
        "is_score": is_score,
        "is_sharpe": is_sharpe,
        "oos_sharpe": oos_sharpe,
        "oos_bars": len(oos),
        "decay": oos_sharpe / is_sharpe if is_sharpe else None,
        "n_trials": n_trials,
        "trial_sharpe_var_per_period": var_trials,
        "expected_max_sharpe_annual": expected_max_sharpe(n_trials, var_trials)
        * math.sqrt(ppy),
        "dsr": deflated_sharpe(oos, n_trials, var_trials),
        "oos_positive_fraction": positive_fraction(oos, th.n_subperiods, ppy),
        "baseline_seed_oos_sharpe": sharpe(seed_oos, ppy),
        "baseline_buy_hold_oos_sharpe": sharpe(hold_oos, ppy),
        "best_program": str(best_path),
    }


def render_report(hid: str, eid: str, report: Dict[str, Any], label: Status) -> str:
    rows = "\n".join(
        f"| {k} | {v:.4g} |" if isinstance(v, float) else f"| {k} | {v} |"
        for k, v in report.items()
    )
    return (
        f"# Skeptic report: {hid}/{eid}\n\n"
        f"Rule-based verdict: **{label.value}**\n\n"
        f"| check | value |\n|---|---|\n{rows}\n"
    )
