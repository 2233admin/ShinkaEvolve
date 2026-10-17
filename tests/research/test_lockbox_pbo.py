from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shinka.research import cli
from shinka.research.notebook import Notebook, Status
from shinka.research.skeptic import pbo


def test_pbo_noise_is_near_half_and_real_edge_is_low() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, (2000, 30))
    assert 0.3 < pbo(noise, 10) < 0.7
    edge = noise.copy()
    edge[:, 0] += 0.004  # one trial with a persistent edge
    assert pbo(edge, 10) < 0.1
    assert np.isnan(pbo(noise[:, :1], 10))


def _direction(tmp_path: Path, example_dir: Path, bars_csv: Path) -> Path:
    root = tmp_path / "d"
    cli.main(["init", str(root), "--direction", "momentum"])
    cli.main(
        ["hypothesis", "add", str(root), "--title", "TS momentum",
         "--file", str(example_dir / "hypothesis.md")]
    )  # fmt: skip
    cli.main(
        ["scaffold", str(root), "h001", "--initial", str(example_dir / "initial.py"),
         "--data", str(bars_csv), "--lockbox-fraction", "0.25"]
    )  # fmt: skip
    return root


def test_scaffold_seals_lockbox(tmp_path: Path, example_dir: Path, bars_csv: Path):
    root = _direction(tmp_path, example_dir, bars_csv)
    nb = Notebook.load(root)
    full = pd.read_csv(bars_csv)
    dev = pd.read_csv(nb.data.dev_path)
    assert nb.data.n_total == len(full) and nb.data.n_dev == len(dev) == 600
    assert str(pd.Timestamp(full["timestamp"].iloc[600])) == nb.data.lockbox_start
    evaluate = (root / "experiments/h001/e01/task/evaluate.py").read_text()
    assert "dev.csv" in evaluate and str(bars_csv) not in evaluate

    # Later experiments reuse the direction's data; other data is refused.
    cli.main(["scaffold", str(root), "h001", "--initial",
              str(example_dir / "initial.py")])  # fmt: skip
    other = tmp_path / "other.csv"
    full.iloc[:700].to_csv(other, index=False)
    with pytest.raises(ValueError, match="new direction"):
        cli.main(["scaffold", str(root), "h001", "--initial",
                  str(example_dir / "initial.py"), "--data", str(other)])  # fmt: skip


LOCKBOX_PEEK = """
import pandas as pd

def run_experiment(bars):
    pos = pd.Series(0.0, index=bars.index)
    if len(bars) > 600:
        # Causal on development data, peeks at the end of the lockbox.
        pos.iloc[605] = 1.0 if bars["close"].iloc[-1] > bars["close"].iloc[0] else -1.0
    return pos
"""


def _mark_supported(root: Path, code: str) -> None:
    nb = Notebook.load(root)
    hyp = nb.get("h001")
    exp = hyp.experiments[0]
    exp.skeptic = {"rule_label": "supported", "rule_reason": "test"}
    hyp.status = Status.supported
    nb.save(root)
    (root / "experiments/h001/e01/best_program.py").write_text(code)


def test_confirm_opens_lockbox_once(tmp_path: Path, example_dir: Path, bars_csv: Path):
    root = _direction(tmp_path, example_dir, bars_csv)
    with pytest.raises(SystemExit, match="no supported"):
        cli.main(["confirm", str(root)])
    _mark_supported(root, (example_dir / "initial.py").read_text())
    with pytest.raises(SystemExit, match="--yes"):
        cli.main(["confirm", str(root)])
    assert not Notebook.load(root).data.opened

    cli.main(["confirm", str(root), "--yes"])
    nb = Notebook.load(root)
    hyp = nb.get("h001")
    assert nb.data.opened
    assert hyp.status in (Status.confirmed, Status.unconfirmed)
    assert hyp.confirmation["lockbox_bars"] == 200
    assert (root / "CONFIRMATION.md").exists()
    with pytest.raises(SystemExit, match="already opened"):
        cli.main(["confirm", str(root), "--yes"])


def test_confirm_detects_lookahead_in_lockbox(
    tmp_path: Path, example_dir: Path, bars_csv: Path
):
    root = _direction(tmp_path, example_dir, bars_csv)
    _mark_supported(root, LOCKBOX_PEEK)
    with pytest.raises(ValueError, match="look-ahead"):
        cli.main(["confirm", str(root), "--yes"])
    assert not Notebook.load(root).data.opened
