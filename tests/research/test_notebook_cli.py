import sqlite3
from contextlib import closing
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
    nb = Notebook.load(root)
    exp = nb.get("h001").experiments[0]
    with open(nb.data.dev_path, "a") as f:
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


def test_verdict_rules(tmp_path: Path, example_dir: Path) -> None:
    root = _init(tmp_path, example_dir)
    nb = Notebook.load(root)
    with pytest.raises(ValueError):
        nb.set_status("h001", Status.supported)
    nb.add_experiment("h001", "a", "/d.csv", "b")
    nb.set_verdict("h001", Verdict(label=Status.refuted, reason="OOS Sharpe < 0"))
    assert nb.get("h001").status is Status.refuted


def test_status_and_brief_count_the_trials_of_a_run_in_progress(
    tmp_path: Path, example_dir: Path, capsys
) -> None:
    root = _init(tmp_path, example_dir)
    nb = Notebook.load(root)
    nb.add_experiment("h001", "a", "/d.csv", "b").n_probed = 1
    nb.save(root)
    results = root / "experiments" / "h001" / "e01" / "results"
    results.mkdir(parents=True)
    with closing(sqlite3.connect(results / "programs.sqlite")) as conn, conn:
        conn.execute(
            "CREATE TABLE programs (id TEXT, code TEXT, combined_score REAL, "
            "correct INTEGER, metadata TEXT)"
        )
        conn.executemany(
            "INSERT INTO programs VALUES (?, '', 0.1, 1, '{}')", [("p1",), ("p2",)]
        )
    # The notebook still says 0 evaluated, as it does until a run returns.
    capsys.readouterr()
    cli.main(["status", str(root)])
    cli.main(["brief", str(root)])
    out = capsys.readouterr().out
    assert "Total trials: 3" in out and "e01(2)" in out
    assert "Trials so far: 3" in out


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
    # Probes count as trials.
    assert Notebook.load(root).get("h001").experiments[0].n_probed == 1


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
