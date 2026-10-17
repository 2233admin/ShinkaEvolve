"""Quant evaluation harness for Shinka Research.

The evolved program exposes ``run_experiment(bars) -> positions``: a target
position in [-1, 1] decided at each bar's close. The harness owns everything
else (P&L, costs, splits, metrics), so the program can never touch how it is
scored.

Evaluation is mapped onto ``run_shinka_eval`` runs:

* run 0 receives the full bar history and produces the positions that are
  scored;
* runs 1..k receive the history truncated at random cut points. A causal
  strategy must produce the same position at each cut whether or not later
  bars exist; any mismatch marks the program incorrect (look-ahead).

In-sample fold Sharpe ratios are ``public`` (they drive selection); the
out-of-sample holdout is ``private`` so the mutation LLM never sees it.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class QuantConfig(BaseModel):
    """Fixed evaluation settings for one experiment. Hashed into the notebook."""

    data_path: str = Field(description="Absolute path to a CSV with timestamp + close.")
    timestamp_col: str = "timestamp"
    oos_fraction: float = Field(0.3, gt=0.0, lt=1.0)
    n_folds: int = Field(3, ge=1)
    cost_bps: float = Field(5.0, ge=0.0, description="Cost per unit turnover, bps.")
    periods_per_year: float = Field(252.0, gt=0.0)
    lookahead_checks: int = Field(5, ge=0)
    seed: int = 0


def load_bars(path: Path, timestamp_col: str = "timestamp") -> pd.DataFrame:
    bars = pd.read_csv(path, parse_dates=[timestamp_col]).set_index(timestamp_col)
    bars = bars.sort_index()
    if "close" not in bars.columns:
        raise ValueError(f"{path} has no 'close' column")
    if bars.index.has_duplicates:
        raise ValueError(f"{path} has duplicate timestamps")
    return bars


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Backtest and metrics
# --------------------------------------------------------------------------


def strategy_returns(
    positions: pd.Series, close: pd.Series, cost_bps: float
) -> pd.Series:
    """Net per-bar returns of holding ``positions[t]`` from close t to t+1."""
    pos = positions.reindex(close.index).astype(float)
    asset_ret = close.pct_change()
    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])
    gross = pos.shift(1) * asset_ret
    cost = turnover.shift(1) * cost_bps / 1e4
    return (gross - cost).fillna(0.0)


def sharpe(returns: pd.Series, periods_per_year: float) -> float:
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    if std == 0.0 or not math.isfinite(std):
        return 0.0
    return float(returns.mean()) / std * math.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0


def annual_return(returns: pd.Series, periods_per_year: float) -> float:
    if len(returns) == 0:
        return 0.0
    growth = float((1.0 + returns).prod())
    if growth <= 0.0:
        return -1.0
    return growth ** (periods_per_year / len(returns)) - 1.0


def summarize(returns: pd.Series, periods_per_year: float) -> Dict[str, float]:
    return {
        "sharpe": sharpe(returns, periods_per_year),
        "annual_return": annual_return(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
    }


def split_index(n: int, oos_fraction: float) -> int:
    """First out-of-sample position."""
    return int(round(n * (1.0 - oos_fraction)))


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate_positions(positions: Any, bars: pd.DataFrame) -> Optional[str]:
    """Return an error message, or None if the output is well-formed."""
    if not isinstance(positions, pd.Series):
        return f"run_experiment must return a pandas Series, got {type(positions)}"
    if not positions.index.equals(bars.index):
        return "positions index must equal the bars index"
    values = positions.to_numpy(dtype=float, na_value=np.nan)
    if not np.all(np.isfinite(values)):
        return "positions contain NaN or inf"
    if np.any(np.abs(values) > 1.0 + 1e-9):
        return "positions must be within [-1, 1]"
    return None


def lookahead_cuts(n: int, k: int, seed: int) -> List[int]:
    """Cut positions (inclusive last index) in the back 3/4 of the history."""
    if k == 0 or n < 8:
        return []
    rng = np.random.default_rng(seed)
    lo = n // 4
    cuts = rng.choice(np.arange(lo, n - 1), size=min(k, n - 1 - lo), replace=False)
    return sorted(int(c) for c in cuts)


# --------------------------------------------------------------------------
# Shinka adapter
# --------------------------------------------------------------------------


def evaluate_positions(
    positions: pd.Series, bars: pd.DataFrame, cfg: QuantConfig
) -> Dict[str, Any]:
    """Score full-history positions into Shinka's metrics format."""
    rets = strategy_returns(positions, bars["close"], cfg.cost_bps)
    cut = split_index(len(bars), cfg.oos_fraction)
    is_rets, oos_rets = rets.iloc[:cut], rets.iloc[cut:]
    folds = np.array_split(np.arange(len(is_rets)), cfg.n_folds)
    fold_sharpes = [
        sharpe(is_rets.iloc[idx], cfg.periods_per_year) for idx in folds if len(idx)
    ]
    turnover = float(positions.diff().abs().iloc[:cut].mean())
    is_sum = summarize(is_rets, cfg.periods_per_year)
    oos_sum = summarize(oos_rets, cfg.periods_per_year)
    score = float(np.mean(fold_sharpes))
    public = {
        "is_sharpe_mean": score,
        "is_sharpe_std": float(np.std(fold_sharpes)),
        "is_fold_sharpes": fold_sharpes,
        "is_annual_return": is_sum["annual_return"],
        "is_max_drawdown": is_sum["max_drawdown"],
        "is_turnover": turnover,
    }
    private = {f"oos_{k}": v for k, v in oos_sum.items()}
    feedback = (
        f"In-sample fold Sharpe: {', '.join(f'{s:.2f}' for s in fold_sharpes)} "
        f"(mean {score:.2f}). Annual return {is_sum['annual_return']:.1%}, "
        f"max drawdown {is_sum['max_drawdown']:.1%}, "
        f"mean turnover {turnover:.3f} per bar, costs {cfg.cost_bps} bps."
    )
    return {
        "combined_score": score,
        "public": public,
        "private": private,
        "text_feedback": feedback,
    }


def run_quant_eval(
    program_path: str, results_dir: str, cfg: QuantConfig
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Entry point used by a task directory's evaluate.py."""
    from shinka.core import run_shinka_eval

    bars = load_bars(Path(cfg.data_path), cfg.timestamp_col)
    cuts = lookahead_cuts(len(bars), cfg.lookahead_checks, cfg.seed)
    views = [bars] + [bars.iloc[: c + 1] for c in cuts]

    def get_kwargs(i: int) -> Dict[str, Any]:
        return {"bars": views[i].copy()}

    def aggregate(results: List[Any]) -> Dict[str, Any]:
        full = results[0]
        err = validate_positions(full, bars)
        if err is None:
            for cut, view, partial in zip(cuts, views[1:], results[1:]):
                err = validate_positions(partial, view)
                if err is None and not math.isclose(
                    float(full.iloc[cut]), float(partial.iloc[-1]), abs_tol=1e-9
                ):
                    err = (
                        f"look-ahead detected: position at {bars.index[cut]} "
                        "changes when later bars are removed"
                    )
                if err:
                    break
        if err:
            raise ValueError(err)
        return evaluate_positions(full, bars, cfg)

    return run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=len(views),
        get_experiment_kwargs=get_kwargs,
        aggregate_metrics_fn=aggregate,
    )
