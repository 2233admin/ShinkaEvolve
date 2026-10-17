from pathlib import Path

import pytest

from shinka.research import cli
from shinka.research.notebook import Notebook, Status, Verdict
from shinka.research.scaffold import verify_experiment


def _init(tmp_path: Path, example_dir: Path) -> Path:
    root = tmp_path / "direction"
    cli.main(["init", str(root), "--direction", "Does momentum work?"])
    cli.main(
        [
            "hypothesis", "add", str(root),
            "--title", "TS momentum survives costs",
            "--file", str(example_dir / "hypothesis.md"),
        ]
    )  # fmt: skip
    return root


def test_init_and_hypothesis(tmp_path: Path, example_dir: Path) -> None:
    root = _init(tmp_path, example_dir)
    nb = Notebook.load(root)
    assert nb.direction == "Does momentum work?"
    assert (root / "program.md").exists()
    assert nb.hypotheses[0].id == "h001"
    assert "Statement" in (root / "hypotheses" / "h001.md").read_text()

    cli.main(["hypothesis", "status", str(root), "h001", "dropped"])
    assert Notebook.load(root).get("h001").status is Status.dropped


def test_scaffold_freezes_evaluator(
    tmp_path: Path, example_dir: Path, bars_csv: Path
) -> None:
    root = _init(tmp_path, example_dir)
    cli.main(
        [
            "scaffold", str(root), "h001",
            "--initial", str(example_dir / "initial.py"),
            "--data", str(bars_csv),
        ]
    )  # fmt: skip
    nb = Notebook.load(root)
    exp = nb.get("h001").experiments[0]
    task = root / "experiments" / "h001" / "e01" / "task"
    for name in ("initial.py", "evaluate.py", "shinka.yaml"):
        assert (task / name).exists()
    assert "TS momentum survives costs" in (task / "shinka.yaml").read_text()
    assert nb.get("h001").status is Status.selected

    verify_experiment(root, Notebook.load(root), "h001", exp)
    (task / "evaluate.py").write_text("# tampered\n")
    with pytest.raises(RuntimeError, match="modified"):
        verify_experiment(root, Notebook.load(root), "h001", exp)


def test_verify_detects_data_change(
    tmp_path: Path, example_dir: Path, bars_csv: Path
) -> None:
    root = _init(tmp_path, example_dir)
    cli.main(
        [
            "scaffold", str(root), "h001",
            "--initial", str(example_dir / "initial.py"),
            "--data", str(bars_csv),
        ]
    )  # fmt: skip
    exp = Notebook.load(root).get("h001").experiments[0]
    with bars_csv.open("a") as f:
        f.write("2099-01-01,1.0\n")
    with pytest.raises(RuntimeError, match="data changed"):
        verify_experiment(root, Notebook.load(root), "h001", exp)


def test_generated_evaluator_runs(
    tmp_path: Path, example_dir: Path, bars_csv: Path
) -> None:
    import subprocess
    import sys

    root = _init(tmp_path, example_dir)
    cli.main(
        [
            "scaffold", str(root), "h001",
            "--initial", str(example_dir / "initial.py"),
            "--data", str(bars_csv),
        ]
    )  # fmt: skip
    task = root / "experiments" / "h001" / "e01" / "task"
    # Shinka copies evaluate.py into the results dir before running it.
    elsewhere = tmp_path / "copied"
    elsewhere.mkdir()
    (elsewhere / "evaluate.py").write_text((task / "evaluate.py").read_text())
    out = subprocess.run(
        [
            sys.executable, str(elsewhere / "evaluate.py"),
            "--program_path", str(task / "initial.py"),
            "--results_dir", str(tmp_path / "res"),
        ],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert out.stdout.strip().splitlines()[-1] == "correct"


def test_verdict_rules_and_trials(tmp_path: Path, example_dir: Path) -> None:
    root = _init(tmp_path, example_dir)
    nb = Notebook.load(root)
    with pytest.raises(ValueError):
        nb.set_status("h001", Status.supported)
    nb.add_experiment("h001", "a", "/d.csv", "b").n_evaluated = 7
    nb.set_verdict("h001", Verdict(label=Status.refuted, reason="OOS Sharpe < 0"))
    assert nb.get("h001").status is Status.refuted
    assert nb.total_trials == 7


def test_probe_shows_public_metrics_only(
    tmp_path: Path, example_dir: Path, bars_csv: Path, capsys
) -> None:
    import json

    root = _init(tmp_path, example_dir)
    cli.main(
        [
            "scaffold", str(root), "h001",
            "--initial", str(example_dir / "initial.py"),
            "--data", str(bars_csv),
        ]
    )  # fmt: skip
    capsys.readouterr()
    cli.main(["probe", str(root), "h001", str(example_dir / "initial.py")])
    out = capsys.readouterr().out
    shown = json.loads(out)
    assert shown["correct"] is True
    assert "is_sharpe_mean" in shown["public"]
    assert "oos" not in out.lower()
    assert Notebook.load(root).total_trials == 1  # probes count as trials


def test_status_guard_and_headless_guard(tmp_path: Path, example_dir: Path) -> None:
    root = _init(tmp_path, example_dir)
    nb = Notebook.load(root)
    nb.add_experiment("h001", "a", "/d.csv", "b")
    nb.get("h001").status = Status.running
    nb.save(root)
    with pytest.raises(ValueError, match="running"):
        cli.main(["hypothesis", "status", str(root), "h001", "dropped"])
    with pytest.raises(SystemExit, match="headless"):
        cli.main(
            ["run", str(root), "h001", "--",
             "--set", 'evo.llm_models=["headless/claude"]']
        )  # fmt: skip
