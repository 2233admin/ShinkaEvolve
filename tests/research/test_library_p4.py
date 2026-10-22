import json
import sqlite3
from pathlib import Path

import pytest

from shinka.research import cli, grants, library
from shinka.research.notebook import RESULTS_LOG, Notebook
from shinka.research.rundb import RunCost, run_cost


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


def test_add_logs_whether_the_agent_or_a_human_added_it(
    tmp_path: Path, example_dir: Path, monkeypatch
) -> None:
    root = _init(tmp_path, example_dir)
    lib = tmp_path / "library"
    mine, theirs = tmp_path / "mine.md", tmp_path / "theirs.md"
    mine.write_text("owner's note\n")
    theirs.write_text("agent's clip\n")
    monkeypatch.delenv(grants.AGENT_ENV, raising=False)
    cli.main(["library", "add", str(root), str(mine), "https://example.org/a"])
    monkeypatch.setenv(grants.AGENT_ENV, "1")
    cli.main(["library", "add", str(root), str(theirs), "https://example.org/b"])

    adds = [
        line.split(" ", 2)[2]
        for line in (lib / "log.md").read_text().splitlines()
        if "[add " in line
    ]
    assert adds == [
        "[add human] raw/mine.md", "[add human] https://example.org/a",
        "[add agent] raw/theirs.md", "[add agent] https://example.org/b",
    ]  # fmt: skip
    # the origin lives in the log only: raw/ holds the sources and nothing else
    assert library.pending(lib) == ["raw/links.md", "raw/mine.md", "raw/theirs.md"]


def test_brief_says_where_the_library_came_from(
    tmp_path: Path, example_dir: Path, capsys
) -> None:
    root = _init(tmp_path, example_dir)
    lib = tmp_path / "library"
    with (lib / "log.md").open("a") as f:
        f.write("- 2026-01-01T00:00:00+00:00 [add] raw/old.md\n")  # before origins
        f.write("- 2026-02-01T00:00:00+00:00 [add human] https://example.org/a\n")
        f.write("- 2026-03-01T00:00:00+00:00 [add agent] raw/clip.md\n")
        f.write("- 2026-03-02T00:00:00+00:00 [add agent] https://example.org/b\n")
        f.write("- 2026-03-03T00:00:00+00:00 [lint] nothing\n")
    for name in ("old.md", "clip.md", "dropped.md"):  # dropped.md: copied by hand
        (lib / "raw" / name).write_text(name)
    # two links `add` wrote (counted by their log lines) and one typed by hand
    (lib / "raw" / "links.md").write_text(
        "# Links\n\n- https://example.org/a (added 2026-02-01)\n"
        "- https://example.org/b (added 2026-03-02)\n- https://example.org/hand\n"
    )
    for name, kind in (("tsmom", "paper"), ("h001", "finding  # this research")):
        (lib / "wiki" / f"{name}.md").write_text(f"---\ntype: {kind}\n---\n# {name}\n")
    # the demo workspace's pages: no front matter, a standalone Type line
    (lib / "wiki" / "demo.md").write_text("# demo\n\n> DEMO\n\nType: finding\n")
    (lib / "wiki" / "notes.md").write_text("# notes\n\nno type given\n")

    capsys.readouterr()
    cli.main(["brief", str(root)])
    assert (
        "- Sources: 2 added by the agent, 1 by people, 3 with no recorded origin; "
        "last added 2026-03-02T00:00:00+00:00; 2 of 4 wiki pages are findings"
    ) in capsys.readouterr().out.splitlines()


def test_brief_of_an_empty_library(tmp_path: Path, example_dir: Path, capsys) -> None:
    root = _init(tmp_path, example_dir)
    capsys.readouterr()
    cli.main(["brief", str(root)])
    assert (
        "- Sources: 0 added by the agent, 0 by people; none added yet; "
        "0 of 0 wiki pages are findings"
    ) in capsys.readouterr().out.splitlines()


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


def test_run_cost_excludes_copies(tmp_path: Path) -> None:
    _fake_results(tmp_path / "r", [0.5, 0.25])
    # Absent fields are recorded zeros, as the runner totals them.
    assert run_cost(tmp_path / "r") == pytest.approx(RunCost(0.77, True, 0.0, True))
    assert run_cost(tmp_path / "missing") == RunCost(0.0, True, 0.0, True)
    # prompt-evolution charges live in prompts.sqlite
    with sqlite3.connect(tmp_path / "r" / "prompts.sqlite") as conn:
        conn.execute("CREATE TABLE system_prompts (id TEXT, metadata TEXT)")
        conn.execute(
            "INSERT INTO system_prompts VALUES ('s', ?)",
            (json.dumps({"llm": {"cost": 0.2}}),),
        )
        conn.execute("INSERT INTO system_prompts VALUES ('t', NULL)")
    assert run_cost(tmp_path / "r").recorded_usd == pytest.approx(0.97)
    assert run_cost(tmp_path / "r").known


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


def test_budget_refuses_runs_when_spend_is_unknown(
    tmp_path: Path, example_dir: Path, bars_csv: Path
) -> None:
    root = _init(tmp_path, example_dir, "--budget-usd", "1.0")
    cli.main(
        ["scaffold", str(root), "h001", "--initial", str(example_dir / "initial.py"),
         "--data", str(bars_csv)]
    )  # fmt: skip
    # $0.10 recorded, far under budget, but one cost is not an amount.
    _fake_results(root / "experiments/h001/e01/results", [0.1, "n/a"])
    with pytest.raises(SystemExit, match="budget cannot be checked"):
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
