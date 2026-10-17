import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shinka.research.quant import (
    QuantConfig,
    evaluate_positions,
    load_bars,
    lookahead_cuts,
    max_drawdown,
    run_quant_eval,
    sharpe,
    strategy_returns,
    validate_positions,
)

CHEATING_PROGRAM = """
import pandas as pd

def run_experiment(bars):
    # Uses the next bar's return: look-ahead.
    nxt = bars["close"].pct_change().shift(-1).fillna(0.0)
    return (nxt > 0).astype(float)
"""


def test_strategy_returns_applies_position_to_next_bar_and_costs() -> None:
    idx = pd.date_range("2020-01-01", periods=4)
    close = pd.Series([100.0, 110.0, 99.0, 99.0], index=idx)
    pos = pd.Series([1.0, 1.0, -1.0, 0.0], index=idx)
    rets = strategy_returns(pos, close, cost_bps=10.0)
    # bar1: held +1 from bar0 (+10%), entry cost 1 unit * 10bps
    assert rets.iloc[0] == 0.0
    assert rets.iloc[1] == pytest.approx(0.10 - 0.001)
    # bar2: held +1 (-10%), no turnover at bar1
    assert rets.iloc[2] == pytest.approx(-0.10)
    # bar3: held -1 (0%), flipped at bar2 => 2 units turnover
    assert rets.iloc[3] == pytest.approx(-0.002)


def test_metrics_basic() -> None:
    flat = pd.Series([0.0] * 10)
    assert sharpe(flat, 252) == 0.0
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)


def test_validate_positions(bars: pd.DataFrame) -> None:
    ok = pd.Series(0.0, index=bars.index)
    assert validate_positions(ok, bars) is None
    assert "Series" in validate_positions(ok.to_numpy(), bars)
    assert "within" in validate_positions(ok + 2, bars)
    assert "NaN" in validate_positions(ok.where(ok.index != ok.index[3]), bars)
    assert "index" in validate_positions(ok.iloc[1:], bars)


def test_lookahead_cuts_are_deterministic_and_in_range() -> None:
    cuts = lookahead_cuts(100, 5, seed=3)
    assert cuts == lookahead_cuts(100, 5, seed=3)
    assert len(cuts) == 5 and all(25 <= c < 99 for c in cuts)
    assert lookahead_cuts(100, 0, seed=3) == []


def test_evaluate_positions_splits_public_and_private(bars: pd.DataFrame) -> None:
    cfg = QuantConfig(data_path="unused.csv", n_folds=3)
    pos = pd.Series(1.0, index=bars.index)
    m = evaluate_positions(pos, bars, cfg)
    assert len(m["public"]["is_fold_sharpes"]) == 3
    assert m["combined_score"] == pytest.approx(np.mean(m["public"]["is_fold_sharpes"]))
    assert set(m["private"]) == {"oos_sharpe", "oos_annual_return", "oos_max_drawdown"}
    assert not any(k.startswith("oos") for k in m["public"])
    assert "oos" not in m["text_feedback"].lower()


def _config(bars_csv: Path) -> QuantConfig:
    return QuantConfig(data_path=str(bars_csv), lookahead_checks=4)


def test_run_quant_eval_scores_causal_program(
    tmp_path: Path, bars_csv: Path, example_dir: Path
) -> None:
    metrics, correct, err = run_quant_eval(
        str(example_dir / "initial.py"),
        str(tmp_path / "res"),
        _config(bars_csv),
    )
    assert correct, err
    assert np.isfinite(metrics["combined_score"])
    saved = json.loads((tmp_path / "res" / "metrics.json").read_text())
    assert "oos_sharpe" in saved["private"]


def test_run_quant_eval_rejects_lookahead(tmp_path: Path, bars_csv: Path) -> None:
    program = tmp_path / "cheat.py"
    program.write_text(CHEATING_PROGRAM)
    _, correct, err = run_quant_eval(
        str(program), str(tmp_path / "res"), _config(bars_csv)
    )
    assert not correct
    assert "look-ahead" in err


def test_load_bars_requires_close(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("timestamp,open\n2020-01-01,1\n")
    with pytest.raises(ValueError, match="close"):
        load_bars(p)
