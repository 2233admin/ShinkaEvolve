import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shinka.research import cli
from shinka.research.notebook import Notebook, Status
from shinka.research.skeptic import (
    Thresholds,
    decide,
    deflated_sharpe,
    expected_max_sharpe,
    positive_fraction,
)


def _returns(mean: float, n: int = 1000, seed: int = 0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(mean, 0.01, n))


def test_expected_max_sharpe_grows_with_trials() -> None:
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(10, 0.0) == 0.0
    assert 0 < expected_max_sharpe(10, 0.01) < expected_max_sharpe(1000, 0.01)


def test_deflated_sharpe_penalises_many_trials() -> None:
    strong = _returns(0.002)
    assert deflated_sharpe(strong, 1, 0.0) > 0.99
    single = deflated_sharpe(_returns(0.0005), 1, 0.0)
    many = deflated_sharpe(_returns(0.0005), 500, 0.002)
    assert many < single
    assert deflated_sharpe(pd.Series([0.0] * 10), 1, 0.0) == 0.0


def test_positive_fraction() -> None:
    r = pd.Series(np.r_[np.full(50, 0.01), np.full(50, -0.01)])
    r = r + np.random.default_rng(0).normal(0, 1e-4, 100)
    assert positive_fraction(r, 2, 252) == pytest.approx(0.5)


def test_decide_rules() -> None:
    th = Thresholds()
    base = {
        "n_trials": 10,
        "dsr": 0.99,
        "oos_positive_fraction": 1.0,
        "baseline_buy_hold_oos_sharpe": 0.2,
    }
    assert decide({**base, "oos_sharpe": -0.1}, th)[0] is Status.refuted
    assert decide({**base, "oos_sharpe": 1.0}, th)[0] is Status.supported
    weak = {**base, "oos_sharpe": 0.3, "dsr": 0.6}
    assert decide(weak, th)[0] is Status.inconclusive
    # Strong statistics but no better than passive exposure: not supported.
    below_hold = {**base, "oos_sharpe": 0.15}
    assert decide(below_hold, th)[0] is Status.inconclusive
    # Likely an artefact of picking the best of many trials.
    overfit = {**base, "oos_sharpe": 1.0, "pbo": 0.8}
    assert decide(overfit, th)[0] is Status.inconclusive


def _fake_results(results_dir: Path, programs: list) -> None:
    results_dir.mkdir(parents=True)
    with sqlite3.connect(results_dir / "programs.sqlite") as conn:
        conn.execute(
            "CREATE TABLE programs (id TEXT, code TEXT, combined_score REAL, "
            "correct INTEGER, metadata TEXT)"
        )
        conn.executemany("INSERT INTO programs VALUES (?, ?, ?, ?, '{}')", programs)


def test_skeptic_and_verdict_cli(
    tmp_path: Path, example_dir: Path, bars_csv: Path
) -> None:
    root = tmp_path / "d"
    seed = (example_dir / "initial.py").read_text()
    cli.main(["init", str(root), "--direction", "momentum"])
    cli.main(
        ["hypothesis", "add", str(root), "--title", "TS momentum",
         "--file", str(example_dir / "hypothesis.md")]
    )  # fmt: skip
    cli.main(
        ["scaffold", str(root), "h001", "--initial", str(example_dir / "initial.py"),
         "--data", str(bars_csv)]
    )  # fmt: skip
    results = root / "experiments" / "h001" / "e01" / "results"
    _fake_results(
        results,
        [("p0", seed, 0.3, 1), ("p1", seed.replace("(20)", "(5)"), 0.5, 1),
         ("p2", "broken", 9.9, 0)],
    )  # fmt: skip
    nb = Notebook.load(root)
    nb.get("h001").experiments[0].n_evaluated = 3
    nb.save(root)

    with pytest.raises(SystemExit):
        cli.main(["verdict", str(root), "h001"])  # not challenged yet

    cli.main(["skeptic", str(root), "h001"])
    nb = Notebook.load(root)
    rep = nb.get("h001").experiments[0].skeptic
    assert nb.get("h001").status is Status.challenged
    assert rep["program_id"] == "p1"  # best correct, not the broken 9.9
    assert rep["n_trials"] == 3
    assert 0.0 <= rep["dsr"] <= 1.0
    assert (results.parent / "skeptic.md").exists()
    assert "(5)" in (results.parent / "best_program.py").read_text()

    with pytest.raises(SystemExit):  # override needs a reason
        other = "supported" if rep["rule_label"] != "supported" else "refuted"
        cli.main(["verdict", str(root), "h001", "--label", other])

    cli.main(["verdict", str(root), "h001"])
    hyp = Notebook.load(root).get("h001")
    assert hyp.status.value == rep["rule_label"]
    assert hyp.verdict and hyp.verdict.evidence["experiment"] == "e01"
