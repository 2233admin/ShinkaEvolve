"""Runs that outlive their window, and their ends recorded once (research-agent 2.2).

A stand-in for shinka_run writes program rows at a set pace, then exits or
hangs; detached runs are real background processes.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from shinka.research import cli
from shinka.research.notebook import RESULTS_LOG, Notebook, Status
from shinka.research.state import live_work, work_entries

FAKE_SHINKA_RUN = """
import json, os, sqlite3, sys, time
from pathlib import Path

args = sys.argv[1:]
results = Path(args[args.index("--results_dir") + 1])
results.mkdir(parents=True, exist_ok=True)
# One plan per experiment, named by the results directory's parent.
plan = {"rows": 0, "pause": 0.0, "wait": 0.0, "code": 0}
plan.update(json.loads(os.environ["FAKE_SHINKA_RUN"]).get(results.parent.name, {}))
db = results / "programs.sqlite"
with sqlite3.connect(db) as conn:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS programs (id TEXT, code TEXT, generation INTEGER, "
        "timestamp REAL, correct INTEGER, combined_score REAL, metadata TEXT)"
    )
for i in range(plan["rows"]):
    time.sleep(plan["pause"])
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO programs VALUES (?, '', ?, ?, 1, 0.1, '{}')",
            (f"p{time.time_ns()}", i, time.time()),
        )
time.sleep(plan["wait"])
if plan.get("hang"):
    time.sleep(600)
sys.exit(plan["code"])
"""

# A window of its own: it starts the run in the background and closes.
LAUNCH = """
import sys
from shinka.research import cli

cli.SHINKA_RUN = [sys.executable, sys.argv[2]]
cli.main(["run", sys.argv[1], "h001", "--detach"])
"""

WATCH = """
import sys
from pathlib import Path
from shinka.research.runs import watch

watch(Path(sys.argv[1]))
"""


@pytest.fixture
def fake(tmp_path: Path) -> Path:
    path = tmp_path / "fake_shinka_run.py"
    path.write_text(FAKE_SHINKA_RUN)
    return path


@pytest.fixture
def direction(tmp_path: Path, example_dir: Path, bars_csv: Path, fake, monkeypatch):
    root = tmp_path / "direction"
    cli.main(["init", str(root), "--direction", "Background run fixture"])
    cli.main(
        [
            "hypothesis", "add", str(root), "--title", "runs",
            "--file", str(example_dir / "hypothesis.md"),
        ]
    )  # fmt: skip
    cli.main(
        [
            "scaffold", str(root), "h001",
            "--initial", str(example_dir / "initial.py"),
            "--data", str(bars_csv),
        ]
    )  # fmt: skip
    monkeypatch.setattr(cli, "SHINKA_RUN", [sys.executable, str(fake)])
    yield root.resolve()
    _kill(root)
    left = _left_running(root.resolve())
    assert not left, f"processes outlived the test: {[p.pid for p in left]}"


def _plan(monkeypatch, **experiments: dict) -> None:
    monkeypatch.setenv("FAKE_SHINKA_RUN", json.dumps(experiments))


def _until(check, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.1)


def _ends(root: Path) -> list:
    events = [
        row.split("\t")[3] for row in (root / RESULTS_LOG).read_text().splitlines()[1:]
    ]
    return [e for e in events if e in ("run", "run-failed")]


def _named(root: Path) -> list:
    """The processes a run's leases name, with everything they started; never
    this test's own process, which in-process `cli.main` registers first."""
    found = {}
    for entry in work_entries(root):
        for named in [entry, *entry.get("workers", [])]:
            if named["pid"] == os.getpid():
                continue
            try:
                process = psutil.Process(named["pid"])
                if abs(process.create_time() - named["created"]) < 1.0:
                    for p in [process, *process.children(recursive=True)]:
                        found[p.pid] = p
            except psutil.Error:
                pass  # already gone
    return list(found.values())


def _kill(root: Path) -> None:
    """Kill what the leases name and all it started, and wait until each of
    those processes is gone."""
    processes = _named(root)
    for process in processes:
        try:
            process.kill()
        except psutil.Error:
            pass
    _, running = psutil.wait_procs(processes, timeout=30)
    assert not running, f"still running after kill: {[p.pid for p in running]}"
    assert not live_work(root)


def _left_running(root: Path, timeout: float = 10.0) -> list:
    """Processes whose command line names the direction and that do not end on
    their own within ``timeout``, a lease's or not; killed, so none outlives
    the test. Every process these tests start is a Python interpreter, and
    reading only those command lines keeps the scan quick on Windows."""
    needle = os.path.normcase(str(root))
    mention = []
    for process in psutil.process_iter(["name"]):
        name = (process.info["name"] or "").lower()
        if process.pid == os.getpid() or "python" not in name:
            continue
        try:
            line = " ".join(process.cmdline())
        except psutil.Error:
            continue
        if needle in os.path.normcase(line):
            mention.append(process)
    _, running = psutil.wait_procs(mention, timeout=timeout)
    for process in running:
        process.kill()
    return running


def test_a_detached_run_goes_on_after_its_window_closes_and_ends_once(
    direction: Path, fake: Path, monkeypatch
) -> None:
    _plan(monkeypatch, e01=dict(rows=3, pause=0.5))
    window = subprocess.run(
        [sys.executable, "-c", LAUNCH, str(direction), str(fake)], timeout=60
    )
    assert window.returncode == 0
    assert live_work(direction), "the run should outlive the window that started it"
    _until(lambda: not work_entries(direction))
    assert _ends(direction) == ["run"]
    assert Notebook.load(direction).get("h001").experiments[0].n_evaluated == 3
    assert (direction / "experiments" / "h001" / "e01" / "run.log").exists()


def test_a_killed_run_is_recorded_once_by_two_watches(
    direction: Path, monkeypatch
) -> None:
    _plan(monkeypatch, e01=dict(hang=True))
    cli.main(["run", str(direction), "h001", "--detach"])
    # The supervisor owns the lease, and the stand-in shinka_run is its worker.
    _until(lambda: work_entries(direction)[0].get("workers"))
    (entry,) = work_entries(direction)
    assert "shinka.research.runs" in psutil.Process(entry["pid"]).cmdline()
    (worker,) = entry["workers"]
    assert "fake_shinka_run.py" in " ".join(psutil.Process(worker["pid"]).cmdline())
    _kill(direction)
    # Nothing the lease leaves out, such as an interpreter behind a launcher,
    # goes on adding trials.
    assert not _left_running(direction)
    watchers = [
        subprocess.Popen([sys.executable, "-c", WATCH, str(direction)])
        for _ in range(2)
    ]
    for watcher in watchers:
        assert watcher.wait(timeout=60) == 0
    assert _ends(direction) == ["run-failed"]
    assert work_entries(direction) == []
    # It added nothing, so the hypothesis is back where it was.
    assert Notebook.load(direction).get("h001").status is Status.selected


def _second_experiment(direction: Path, example_dir: Path) -> None:
    cli.main(
        [
            "scaffold", str(direction), "h001",
            "--initial", str(example_dir / "initial.py"),
        ]
    )  # fmt: skip


def test_one_experiment_runs_once_at_a_time_and_another_runs_beside_it(
    direction: Path, example_dir: Path, monkeypatch
) -> None:
    _plan(monkeypatch, e01=dict(hang=True), e02=dict(hang=True))
    cli.main(["run", str(direction), "h001", "--detach"])
    with pytest.raises(SystemExit, match="already has a run in progress"):
        cli.main(["run", str(direction), "h001"])
    _second_experiment(direction, example_dir)
    cli.main(["run", str(direction), "h001", "--eid", "e02", "--detach"])
    assert sorted(w["experiment"] for w in live_work(direction)) == ["e01", "e02"]
    _kill(direction)
    cli.main(["watch", str(direction)])
    assert _ends(direction) == ["run-failed", "run-failed"]
    # Neither added anything: back to the status before either started.
    assert Notebook.load(direction).get("h001").status is Status.selected


@pytest.mark.parametrize(
    "plans",
    [
        # The failed run ends first, while the other is still adding programs.
        dict(e01=dict(code=1, wait=0.5), e02=dict(rows=2, pause=1.5)),
        # The run that added a program ends first; the failed one later.
        dict(e01=dict(code=1, wait=3.0), e02=dict(rows=1)),
    ],
)
def test_a_hypothesis_stays_running_when_any_of_its_runs_added_programs(
    direction: Path, example_dir: Path, monkeypatch, plans
) -> None:
    _second_experiment(direction, example_dir)
    _plan(monkeypatch, **plans)
    cli.main(["run", str(direction), "h001", "--eid", "e02", "--detach"])
    cli.main(["run", str(direction), "h001", "--eid", "e01", "--detach"])
    _until(lambda: not work_entries(direction))
    assert sorted(_ends(direction)) == ["run", "run-failed"]
    assert Notebook.load(direction).get("h001").status is Status.running


def test_a_detached_run_that_cannot_start_leaves_no_lease(
    direction: Path, monkeypatch
) -> None:
    def cannot_start(*_args, **_kwargs):
        raise OSError("no interpreter")

    monkeypatch.setattr(cli.subprocess, "Popen", cannot_start)
    with pytest.raises(OSError):
        cli.main(["run", str(direction), "h001", "--detach"])
    assert work_entries(direction) == []
    assert _ends(direction) == ["run-failed"]
    assert Notebook.load(direction).get("h001").status is Status.selected


def test_a_foreground_run_still_waits_and_records_its_end(
    direction: Path, monkeypatch
) -> None:
    _plan(monkeypatch, e01=dict(rows=2))
    cli.main(["run", str(direction), "h001"])
    assert _ends(direction) == ["run"]
    assert work_entries(direction) == []
    assert Notebook.load(direction).get("h001").experiments[0].n_evaluated == 2
