"""Runs that outlive their window, their ends recorded once (research-agent 2.2),
and the owner's round grants that agent runs take attempts from (2.3).

A stand-in for shinka_run records what it was asked, writes program rows at a
set pace and cost, then exits or hangs; detached runs are real background
processes.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from shinka.research import cli
from shinka.research.grants import AGENT_ENV, LEDGER
from shinka.research.notebook import RESULTS_LOG, Notebook, Status
from shinka.research.runs import watch
from shinka.research.state import hand_over, live_work, register_work, work_entries

FAKE_SHINKA_RUN = """
import json, os, sqlite3, sys, time
from pathlib import Path

args = sys.argv[1:]
results = Path(args[args.index("--results_dir") + 1])
results.mkdir(parents=True, exist_ok=True)
with (results / "argv.jsonl").open("a") as asked:
    asked.write(json.dumps(args) + "\\n")
# One plan per experiment, named by the results directory's parent.
plan = {"rows": 0, "pause": 0.0, "wait": 0.0, "code": 0, "cost": 0.0}
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
            "INSERT INTO programs VALUES (?, '', ?, ?, 1, 0.1, ?)",
            (f"p{time.time_ns()}", i, time.time(),
             json.dumps({"api_costs": plan["cost"]})),
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

# An agent's run in a window of its own, for two to race for one attempt.
RACE = """
import sys
from shinka.research import cli

cli.SHINKA_RUN = [sys.executable, sys.argv[2]]
try:
    cli.main(["run", sys.argv[1], "h001", "--eid", sys.argv[3], "--grant", "g001"])
except SystemExit as refused:
    print("refused:", refused.code)
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


def test_a_run_killed_while_creating_its_database_is_still_recorded(
    direction: Path,
) -> None:
    # A run that died after creating programs.sqlite and before its tables.
    work = register_work(
        direction, "h001", "e01", cmd=[], evaluated_before=0,
        previous_status="selected", previous_verdict=None, before=0,
    )  # fmt: skip
    results = direction / "experiments" / "h001" / "e01" / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "programs.sqlite").write_bytes(b"")
    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    ended.wait()
    hand_over(direction, work, ended.pid)
    with Notebook.edit(direction) as nb:
        nb.get("h001").status = Status.running
    assert watch(direction) == ["h001/e01: run-failed"]
    assert work_entries(direction) == []
    (note,) = [note for event, note in _log(direction) if event == "run-failed"]
    assert note.startswith("ended unobserved, 0 new programs") and "spend n/a" in note
    assert Notebook.load(direction).get("h001").status is Status.selected


def test_a_foreground_run_still_waits_and_records_its_end(
    direction: Path, monkeypatch
) -> None:
    _plan(monkeypatch, e01=dict(rows=2))
    cli.main(["run", str(direction), "h001"])
    assert _ends(direction) == ["run"]
    assert work_entries(direction) == []
    assert Notebook.load(direction).get("h001").experiments[0].n_evaluated == 2


# ---- Round grants (2.3) -----------------------------------------------------


def _log(root: Path) -> list:
    """(event, note) of every run-log line."""
    rows = (root / RESULTS_LOG).read_text().splitlines()[1:]
    return [tuple(row.split("\t")[3:5]) for row in rows]


def _asked(root: Path, eid: str = "e01") -> list:
    """The arguments each run of the experiment passed shinka_run."""
    path = root / "experiments" / "h001" / eid / "results" / "argv.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    return [json.loads(line) for line in lines]


def _stops_at(argv: list) -> float:
    """The cost cap a shinka_run command ends with."""
    assert argv[-2] == "--set" and argv[-1].startswith("evo.max_api_costs=")
    return float(argv[-1].split("=", 1)[1])


def _ledger(root: Path) -> list:
    path = root / LEDGER
    return json.loads(path.read_text()) if path.exists() else []


def _attempts(root: Path) -> list:
    return [r["number"] for r in _ledger(root) if r["kind"] == "attempt"]


def _grant(root: Path, *terms: str) -> str:
    cli.main(["grant", str(root), "h001", *terms])
    return [r for r in _ledger(root) if r["kind"] == "grant"][-1]["id"]


def test_an_agent_run_needs_a_round_the_owner_granted(
    direction: Path, example_dir: Path, monkeypatch
) -> None:
    cli.main(
        [
            "hypothesis", "add", str(direction), "--title", "other",
            "--file", str(example_dir / "hypothesis.md"),
        ]
    )  # fmt: skip
    cli.main(
        [
            "scaffold",
            str(direction),
            "h002",
            "--initial",
            str(example_dir / "initial.py"),
        ]
    )
    grant = _grant(direction, "--attempts", "1", "--max-cost", "0.5")
    monkeypatch.setenv(AGENT_ENV, "1")
    for argv, refusal in (
        (["h001"], "needs a round the owner granted"),
        (["h001", "--grant", "g009"], "no grant g009"),
        (["h002", "--grant", grant], "h002 is not in grant"),
    ):
        with pytest.raises(SystemExit, match=refusal):
            cli.main(["run", str(direction), *argv])
    # Nor can the agent grant itself a round.
    with pytest.raises(SystemExit, match="the owner's to give"):
        cli.main(
            ["grant", str(direction), "h002", "--attempts", "1", "--max-cost", "1"]
        )
    assert work_entries(direction) == [] and _attempts(direction) == []
    assert not [e for e, _ in _log(direction) if e in ("run", "run-failed")]
    nb = Notebook.load(direction)
    assert [h.status for h in nb.hypotheses] == [Status.selected, Status.selected]


def test_two_agent_runs_racing_for_the_last_attempt_start_one_run(
    direction: Path, example_dir: Path, fake: Path, monkeypatch
) -> None:
    # Two experiments, so neither is refused for running the same one.
    _second_experiment(direction, example_dir)
    _grant(direction, "--attempts", "1", "--max-cost", "0.5")
    monkeypatch.setenv(AGENT_ENV, "1")
    _plan(monkeypatch, e01=dict(rows=1), e02=dict(rows=1))
    racers = [
        subprocess.Popen(
            [sys.executable, "-c", RACE, str(direction), str(fake), eid],
            stdout=subprocess.PIPE,
            text=True,
        )
        for eid in ("e01", "e02")
    ]
    said = [racer.communicate(timeout=120)[0] for racer in racers]
    assert [r.returncode for r in racers] == [0, 0]
    refusals = [out for out in said if "this round is over" in out]
    assert len(refusals) == 1, said
    _until(lambda: not work_entries(direction))
    assert _attempts(direction) == [1]
    assert [e for e, _ in _log(direction)].count("run") == 1


def test_a_round_ends_when_its_attempts_are_used(direction: Path, monkeypatch) -> None:
    grant = _grant(direction, "--attempts", "2", "--max-cost", "0.5")
    monkeypatch.setenv(AGENT_ENV, "1")
    _plan(monkeypatch, e01=dict(rows=1))
    for _ in range(2):
        cli.main(["run", str(direction), "h001", "--grant", grant])
        _until(lambda: not work_entries(direction))
    # Taking the last attempt pauses the round, before the agent asks again.
    assert [e for e, _ in _log(direction)].count("round-paused") == 1
    for _ in range(2):
        with pytest.raises(SystemExit, match="this round is over"):
            cli.main(["run", str(direction), "h001", "--grant", grant])
    assert _attempts(direction) == [1, 2]
    ends = [note for event, note in _log(direction) if event == "run"]
    taken = [re.search(r"grant g\d+ attempt (\d) of 2", note) for note in ends]
    assert [t.group(1) for t in taken] == ["1", "2"]
    # Said once, however often the agent asks again.
    assert [e for e, _ in _log(direction)].count("round-paused") == 1


@pytest.mark.parametrize(
    "passthrough, refusal",
    [
        (["--set", "evo.max_api_costs=9"], "not the agent's to set"),
        (["--set=evo.max_api_costs=9"], "not the agent's to set"),
        # shinka_run's parser would take this prefix for --set.
        (["--se", "evo.max_api_costs=9"], "only --set"),
        (["--set", "db.num_islands=3"], "not among grant"),
        (["--max-proposal-jobs", "8"], "only --set"),
        (["--set", "evo.llm_models=not json"], "Invalid JSON"),
    ],
)
def test_an_agent_run_passes_only_settings_its_grant_allows(
    direction: Path, monkeypatch, passthrough, refusal
) -> None:
    allow = ("--allow", "evo.llm_models")
    grant = _grant(direction, "--attempts", "1", "--max-cost", "0.5", *allow)
    monkeypatch.setenv(AGENT_ENV, "1")
    with pytest.raises(SystemExit, match=refusal):
        cli.main(["run", str(direction), "h001", "--grant", grant, "--", *passthrough])
    assert _attempts(direction) == [] and work_entries(direction) == []


def test_allowed_settings_and_the_exact_cap_reach_shinka_run(
    direction: Path, monkeypatch
) -> None:
    allow = ("--allow", "evo.llm_models")
    # However small, the owner's cap reaches shinka_run as given.
    grant = _grant(direction, "--attempts", "1", "--max-cost", "1e-7", *allow)
    monkeypatch.setenv(AGENT_ENV, "1")
    _plan(monkeypatch, e01=dict(rows=1))
    models = 'evo.llm_models=["gpt-5-mini"]'
    cli.main(["run", str(direction), "h001", "--grant", grant, "--", "--set", models])
    _until(lambda: not work_entries(direction))
    (argv,) = _asked(direction)
    assert argv[-4:-2] == ["--set", models]
    assert _stops_at(argv) == 1e-7


def test_a_grant_refuses_settings_that_defeat_what_is_frozen(direction: Path) -> None:
    for key in (
        "evo.max_api_costs",
        "job.eval_program_path",
        "job.extra_cmd_args",
        "db.db_path",
        "evo.no_such_field",
    ):
        with pytest.raises(SystemExit, match=re.escape(key)):
            cli.main(
                [
                    "grant", str(direction), "h001", "--attempts", "1",
                    "--max-cost", "1", "--allow", key,
                ]
            )  # fmt: skip
    assert _ledger(direction) == []


def test_a_grant_run_stops_at_its_cap_or_at_the_budget_left(
    direction: Path, example_dir: Path, monkeypatch
) -> None:
    with Notebook.edit(direction) as nb:
        nb.budget_usd = 1.5
    # The owner's own run spent $0.70 on e01.
    _plan(monkeypatch, e01=dict(rows=1, cost=0.7))
    cli.main(["run", str(direction), "h001"])
    _second_experiment(direction, example_dir)
    _second_experiment(direction, example_dir)
    grant = _grant(direction, "--attempts", "3", "--max-cost", "0.5")
    monkeypatch.setenv(AGENT_ENV, "1")
    _plan(monkeypatch, e01=dict(hang=True), e02=dict(hang=True))
    # e01 resumes its database, whose $0.70 shinka_run counts toward the cap.
    cli.main(["run", str(direction), "h001", "--eid", "e01", "--grant", grant])
    _until(lambda: len(_asked(direction)) == 2)
    assert _stops_at(_asked(direction)[-1]) == pytest.approx(0.7 + 0.5)
    # e01's run may still spend $0.50 of the $0.80 left, so e02 gets $0.30.
    cli.main(["run", str(direction), "h001", "--eid", "e02", "--grant", grant])
    _until(lambda: _asked(direction, "e02"))
    assert _stops_at(_asked(direction, "e02")[-1]) == pytest.approx(0.3)
    # Nothing is left for e03, and the refusal takes no attempt.
    with pytest.raises(SystemExit, match="no room"):
        cli.main(["run", str(direction), "h001", "--eid", "e03", "--grant", grant])
    assert _attempts(direction) == [1, 2]


def test_a_grant_run_waits_while_a_run_with_no_stop_point_spends_the_budget(
    direction: Path, example_dir: Path, monkeypatch
) -> None:
    with Notebook.edit(direction) as nb:
        nb.budget_usd = 1.0
    _second_experiment(direction, example_dir)
    grant = _grant(direction, "--attempts", "1", "--max-cost", "0.5")
    # The owner's own run, with no cap of the grant's kind.
    _plan(monkeypatch, e01=dict(hang=True))
    cli.main(["run", str(direction), "h001", "--eid", "e01", "--detach"])
    monkeypatch.setenv(AGENT_ENV, "1")
    with pytest.raises(SystemExit, match="no stop point"):
        cli.main(["run", str(direction), "h001", "--eid", "e02", "--grant", grant])
    assert _attempts(direction) == []


def test_a_manual_run_keeps_its_own_cap_and_says_it_was_manual(
    direction: Path, monkeypatch
) -> None:
    _plan(monkeypatch, e01=dict(rows=1))
    cli.main(["run", str(direction), "h001", "--", "--set", "evo.max_api_costs=0.25"])
    (argv,) = _asked(direction)
    assert _stops_at(argv) == 0.25
    (note,) = [note for event, note in _log(direction) if event == "run"]
    assert "manual run, run " in note


def test_a_changed_ledger_stops_grant_runs_but_not_manual_ones(
    direction: Path, monkeypatch
) -> None:
    grant = _grant(direction, "--attempts", "2", "--max-cost", "0.5")
    _plan(monkeypatch, e01=dict(rows=1))
    # Outside the agent environment a grant run waits, like any manual run.
    cli.main(["run", str(direction), "h001", "--grant", grant])
    records = _ledger(direction)
    for changed in (
        # The attempt taken back.
        [r for r in records if r["kind"] != "attempt"],
        # More attempts than granted.
        [dict(r, attempts=5) if r["kind"] == "grant" else r for r in records],
    ):
        (direction / LEDGER).write_text(json.dumps(changed))
        with pytest.raises(SystemExit, match="does not match the run log"):
            cli.main(["run", str(direction), "h001", "--grant", grant])
    cli.main(["run", str(direction), "h001"])
    assert [e for e, _ in _log(direction) if e == "run"] == ["run", "run"]


def test_a_grant_run_refuses_run_settings_changed_since_the_grant(
    direction: Path, example_dir: Path, monkeypatch
) -> None:
    grant = _grant(direction, "--attempts", "2", "--max-cost", "0.5")
    settings = direction / "experiments" / "h001" / "e01" / "task" / "shinka.yaml"
    settings.write_text(settings.read_text() + "db:\n  num_islands: 9\n")
    monkeypatch.setenv(AGENT_ENV, "1")
    with pytest.raises(SystemExit, match="run settings changed"):
        cli.main(["run", str(direction), "h001", "--grant", grant])
    # An experiment scaffolded since runs as scaffold wrote it, and only so.
    _second_experiment(direction, example_dir)
    other = settings.parents[2] / "e02" / "task" / "shinka.yaml"
    scaffolded = other.read_text()
    other.write_text(scaffolded + "job:\n  eval_program_path: elsewhere.py\n")
    with pytest.raises(SystemExit, match="run settings changed"):
        cli.main(["run", str(direction), "h001", "--eid", "e02", "--grant", grant])
    other.write_text(scaffolded)
    _plan(monkeypatch, e02=dict(rows=1))
    cli.main(["run", str(direction), "h001", "--eid", "e02", "--grant", grant])
    _until(lambda: not work_entries(direction))
    assert _attempts(direction) == [1]


def test_each_probe_is_logged_when_it_starts(
    direction: Path, example_dir: Path, monkeypatch
) -> None:
    def broken(*_args, **_kwargs):
        raise RuntimeError("the evaluation broke")

    monkeypatch.setattr(cli, "run_quant_eval", broken)
    program = str(example_dir / "initial.py")
    for flag in ("", "1"):
        monkeypatch.setenv(AGENT_ENV, flag)
        with pytest.raises(RuntimeError):
            cli.main(["probe", str(direction), "h001", program])
    probes = [note for event, note in _log(direction) if event == "probe"]
    assert probes == ["initial.py, manual", "initial.py, agent"]
    assert Notebook.load(direction).get("h001").experiments[0].n_probed == 2


def test_an_agent_run_goes_to_the_background(direction: Path, monkeypatch) -> None:
    grant = _grant(direction, "--attempts", "1", "--max-cost", "0.5")
    monkeypatch.setenv(AGENT_ENV, "1")
    _plan(monkeypatch, e01=dict(wait=5))
    cli.main(["run", str(direction), "h001", "--grant", grant])
    # It returned at once: the run goes on, owned by its supervisor.
    (entry,) = work_entries(direction)
    assert entry["pid"] != os.getpid()
