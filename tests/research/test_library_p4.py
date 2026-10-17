import json
import sqlite3
from pathlib import Path

import pytest

from shinka.research import cli, library
from shinka.research.notebook import RESULTS_LOG, Notebook
from shinka.research.rundb import total_cost


def _init(ws: Path, example_dir: Path, *extra: str) -> Path:
    root = ws / "momentum"
    cli.main(["init", str(root), "--direction", "momentum", *extra])
    cli.main(
        ["hypothesis", "add", str(root), "--title", "TS momentum",
         "--file", str(example_dir / "hypothesis.md")]
    )  # fmt: skip
    return root


def test_library_is_shared_by_the_workspace(tmp_path: Path, example_dir: Path) -> None:
    root = _init(tmp_path, example_dir)
    lib = tmp_path / "library"
    assert (lib / "raw").is_dir() and (lib / "wiki").is_dir()
    assert (lib / "index.md").exists() and (lib / "log.md").exists()
    assert library.library_root(root) == lib.resolve()


def test_add_pending_compiled_cycle(tmp_path: Path, example_dir: Path, capsys) -> None:
    root = _init(tmp_path, example_dir)
    paper = tmp_path / "paper.md"
    paper.write_text("# Time-series momentum\n")
    # files and URLs; workspace path works as well as a direction path
    cli.main(["library", "add", str(root), str(paper), "https://example.org/tsmom"])
    cli.main(["library", "add", str(tmp_path), str(paper)])  # name clash -> -1
    lib = tmp_path / "library"
    assert (lib / "raw/paper.md").exists() and (lib / "raw/paper-1.md").exists()
    assert "https://example.org/tsmom" in (lib / "raw/links.md").read_text()

    capsys.readouterr()
    cli.main(["library", "pending", str(root)])
    assert capsys.readouterr().out.split() == [
        "raw/links.md", "raw/paper-1.md", "raw/paper.md",
    ]  # fmt: skip

    cli.main(["library", "compiled", str(root), "raw/paper.md"])
    assert library.pending(lib) == ["raw/links.md", "raw/paper-1.md"]
    cli.main(["library", "compiled", str(root), "--all"])
    assert library.pending(lib) == []

    # a changed source becomes pending again (e.g. a new link appended)
    cli.main(["library", "add", str(root), "https://example.org/more"])
    assert library.pending(lib) == ["raw/links.md"]

    cli.main(["library", "log", str(root), "lint", "no contradictions found"])
    log = (lib / "log.md").read_text()
    assert "[compiled] raw/paper.md" in log and "[lint] no contradictions" in log


def _fake_results(results: Path, costs: list) -> None:
    results.mkdir(parents=True)
    with sqlite3.connect(results / "programs.sqlite") as conn:
        conn.execute(
            "CREATE TABLE programs (id TEXT, code TEXT, combined_score REAL, "
            "correct INTEGER, metadata TEXT)"
        )
        for i, c in enumerate(costs):
            meta = json.dumps({"api_costs": c, "embed_cost": 0.01})
            conn.execute(
                "INSERT INTO programs VALUES (?, '', ?, 1, ?)", (f"p{i}", 0.1 * i, meta)
            )
        copy = json.dumps({"api_costs": 99.0, "_is_island_copy": True})
        conn.execute("INSERT INTO programs VALUES ('c', '', 9.0, 1, ?)", (copy,))


def test_total_cost_excludes_copies(tmp_path: Path) -> None:
    _fake_results(tmp_path / "r", [0.5, 0.25])
    assert total_cost(tmp_path / "r") == pytest.approx(0.77)
    assert total_cost(tmp_path / "missing") == 0.0


def test_budget_stops_runs(tmp_path: Path, example_dir: Path, bars_csv: Path) -> None:
    root = _init(tmp_path, example_dir, "--budget-usd", "1.0")
    assert Notebook.load(root).budget_usd == 1.0
    cli.main(
        ["scaffold", str(root), "h001", "--initial", str(example_dir / "initial.py"),
         "--data", str(bars_csv)]
    )  # fmt: skip
    _fake_results(root / "experiments/h001/e01/results", [0.6, 0.6])
    with pytest.raises(SystemExit, match="budget exhausted"):
        cli.main(["run", str(root), "h001"])


def test_brief_and_results_log(
    tmp_path: Path, example_dir: Path, bars_csv: Path, capsys
) -> None:
    root = _init(tmp_path, example_dir, "--budget-usd", "5")
    cli.main(
        ["scaffold", str(root), "h001", "--initial", str(example_dir / "initial.py"),
         "--data", str(bars_csv)]
    )  # fmt: skip
    (tmp_path / "library/raw/note.md").write_text("idea")
    capsys.readouterr()
    cli.main(["brief", str(root)])
    out = capsys.readouterr().out
    assert "# Brief: momentum" in out
    assert "of $5.00" in out
    assert "1 raw sources not compiled yet" in out
    assert "h001 [selected] TS momentum" in out
    assert "scaffold" in out  # recent events from results.tsv

    rows = (root / RESULTS_LOG).read_text().splitlines()
    assert rows[0].split("\t") == ["time", "hypothesis", "experiment", "event", "note"]
    assert rows[1].split("\t")[1:4] == ["h001", "e01", "scaffold"]
