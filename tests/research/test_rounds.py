"""The read-only view of agent work: rounds, runs, approvals and what waits on
the owner (research-agent 3.2), built from the files the CLI writes."""

import json
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from shinka.research import cli, grants, rounds, ui
from shinka.research.grants import LEDGER
from shinka.research.notebook import LockboxOpening, LockboxTarget, Notebook
from shinka.research.runs import finish_run
from shinka.research.state import hand_over, register_work

# A validation figure a lease keeps for its end; it must never leave the view.
HELD_OUT = 9.87654


@pytest.fixture
def direction(tmp_path: Path, example_dir: Path, bars_csv: Path) -> Path:
    root = tmp_path / "workspace" / "d"
    cli.main(["init", str(root), "--direction", "Rounds fixture"])
    cli.main(
        [
            "hypothesis", "add", str(root), "--title", "rounds",
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
    return root.resolve()


def _grant(root: Path, attempts: int) -> dict:
    cli.main(
        ["grant", str(root), "h001", "--attempts", str(attempts), "--max-cost", "0.5"]
    )
    return [r for r in grants.load(root) if r["kind"] == "grant"][-1]


def _attempt(root: Path, grant: dict, number: int, lease: bool = True) -> str:
    """Attempt ``number`` of ``grant`` and its lease, as `run --grant` writes
    them; the lease is this process's own, so it reads as alive."""
    work = uuid.uuid4().hex
    grants.take_attempt(root, grant, number, "h001", "e01", work, 0.5, 0.5)
    if lease:
        register_work(
            root, "h001", "e01", work_id=work, cmd=["shinka_run", str(root)],
            origin={
                "env": "agent", "grant": grant["id"], "attempt": number,
                "of": grant["attempts"], "cap": 0.5, "cost_before": 0.0,
            },
            evaluated_before=0, previous_status="selected",
            previous_verdict={
                "label": "supported", "reason": "x",
                "evidence": {"oos_sharpe": HELD_OUT},
            },
            before=0,
        )  # fmt: skip
    return work


def _dies(root: Path, work: str) -> None:
    """Hand the lease to a process that has ended."""
    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    ended.wait()
    hand_over(root, work, ended.pid)


def _get(base: str, path: str) -> dict:
    return json.loads(urllib.request.urlopen(base + path, timeout=30).read())


def _aged(root: Path) -> None:
    """Make the last attempt an hour old, as one a crash left long ago."""
    path = root / LEDGER
    records = json.loads(path.read_text())
    an_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    records[-1]["time"] = an_hour_ago.isoformat(timespec="seconds")
    path.write_text(json.dumps(records))


def test_the_view_tells_each_run_and_what_waits_on_the_owner(direction: Path) -> None:
    first = _grant(direction, 3)
    finish_run(direction, _attempt(direction, first, 1), 0)
    _attempt(direction, first, 2)
    _dies(direction, _attempt(direction, first, 3))
    second = _grant(direction, 1)
    _attempt(direction, second, 1, lease=False)
    _aged(direction)
    cli.main(["approve", str(direction)])

    view = rounds.overview(direction)
    one, two = view["grants"]
    assert [(r["state"], r["event"]) for r in one["runs"]] == [
        ("ended", "run"),
        ("running", None),
        ("unrecorded", None),
    ]
    # Its attempt was taken and its run never registered.
    assert [r["state"] for r in two["runs"]] == ["lost"]
    assert (one["used"], two["used"]) == (3, 1)
    # g001's runs are not all recorded, and g002 came after it anyway; g002's
    # lost attempt never ends, so its round is over and the loss is listed.
    pending = {item["kind"]: item for item in view["pending"]}
    assert sorted(pending) == ["lost", "round-over", "unrecorded"]
    assert (pending["lost"]["grant"], pending["lost"]["attempt"]) == ("g002", 1)
    assert pending["round-over"]["grant"] == "g002"
    assert pending["unrecorded"]["do"] == "shinka_research watch <dir>"
    (running,) = view["running"]
    assert running["attempt"] == 2 and running["env"] == "agent"
    assert set(running) == {
        "hypothesis", "experiment", "env", "grant", "attempt", "of", "cap", "since",
    }  # fmt: skip
    assert [(a["id"], a["used"], a["moot"]) for a in view["approvals"]] == [
        ("a001", None, False)
    ]
    assert view["state"] == "running"

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(ui._Handler, root=direction.parent)
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        query = urllib.parse.urlencode({"direction": direction.name})
        snapshot = _get(base, f"/api/snapshot?{query}")
        (row,) = _get(base, "/api/workspace")["directions"]
    finally:
        server.shutdown()
        server.server_close()
    served = snapshot["rounds"]
    assert served["state"] == "running"
    assert [r["state"] for r in served["grants"][0]["runs"]] == [
        "ended", "running", "unrecorded",
    ]  # fmt: skip
    assert row["rounds"] == {"state": "running", "pending": 3}
    text = json.dumps(served)
    # Nothing a lease keeps for itself: its held-out evidence, process or paths.
    for leaked in (str(HELD_OUT), '"pid"', '"cmd"', str(direction)):
        assert leaked not in text
    # The timeline tells who ran it, without the key that records it once.
    (ended,) = [e for e in snapshot["events"] if e["event"] == "run"]
    assert ended["note"].endswith("$0.00 of $0.5")


def test_a_ledger_that_does_not_match_the_log_is_reported_without_details(
    direction: Path,
) -> None:
    _grant(direction, 2)
    path = direction / LEDGER
    records = json.loads(path.read_text())
    records[0]["attempts"] = 5
    path.write_text(json.dumps(records))
    view = rounds.overview(direction)
    assert (view["state"], view["problem"], view["grants"]) == (
        "unknown",
        "mismatch",
        [],
    )
    assert [item["kind"] for item in view["pending"]] == ["ledger"]
    path.write_text("{not json")
    assert rounds.overview(direction)["problem"] == "unreadable"
    assert str(direction) not in json.dumps(rounds.overview(direction))


def test_an_open_grant_the_agent_cannot_use_says_why(direction: Path) -> None:
    _grant(direction, 1)
    settings = direction / "experiments" / "h001" / "e01" / "task" / "shinka.yaml"
    settings.write_text(settings.read_text() + "db:\n  num_islands: 9\n")
    with Notebook.edit(direction) as nb:
        nb.budget_usd = 0.0

    def reasons() -> list:
        view = rounds.overview(direction)
        assert view["state"] == "blocked"
        return sorted(i["reason"] for i in view["pending"] if i["kind"] == "stuck")

    assert reasons() == ["budget", "settings"]
    # The owner's own run, with no stop point, may spend whatever is left.
    with Notebook.edit(direction) as nb:
        nb.budget_usd = 10.0
    register_work(direction, "h001", "e01", origin={"env": "manual"})
    assert reasons() == ["open-ended", "settings"]


def test_an_unfinished_opening_names_the_approval_that_finishes_it(
    direction: Path,
) -> None:
    cli.main(["approve", str(direction)])
    cli.main(["approve", str(direction)])
    target = LockboxTarget(
        hypothesis="h001", experiment="e01", program_id="p", code_sha256="c",
        evaluator_sha256="e", data_path="d", data_sha256="s", n_trials=1,
        var_trials=0.0,
    )  # fmt: skip
    with Notebook.edit(direction) as nb:
        opening = LockboxOpening(source_sha256="x", thresholds={}, targets=[target])
        grants.use_approval(direction, "a001", opening.started)
        nb.data.opened, nb.data.opening = True, opening
    view = rounds.overview(direction)
    (item,) = [i for i in view["pending"] if i["kind"] == "lockbox-unfinished"]
    assert item["approval"] == "a001" and item["do"].endswith("--approval a001")
    # The other approval can open nothing any more.
    assert [(a["id"], a["used"] is not None, a["moot"]) for a in view["approvals"]] == [
        ("a001", True, False),
        ("a002", False, True),
    ]
    assert view["state"] == "blocked"


def test_brief_gives_the_terminal_agent_the_same_reading(
    direction: Path, capsys
) -> None:
    _grant(direction, 2)
    capsys.readouterr()
    cli.main(["brief", str(direction)])
    out = capsys.readouterr().out
    assert "## Agent rounds (open)" in out
    assert "- g001: h001, 0 of 2 attempts taken, each stops at $0.5" in out


def test_a_run_still_creating_its_database_leaves_the_view_readable(
    direction: Path,
) -> None:
    grant = _grant(direction, 2)
    with Notebook.edit(direction) as nb:
        nb.budget_usd = 10.0
    work = _attempt(direction, grant, 1)
    # The live run has created programs.sqlite and not yet its tables.
    results = direction / "experiments" / "h001" / "e01" / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "programs.sqlite").write_bytes(b"")
    view = rounds.overview(direction)
    assert (view["problem"], view["state"]) == (None, "running")
    assert view["grants"][0]["runs"][0]["spent_usd"] == 0.0
    # With nothing alive it is a store without tables, which stays an error,
    # reported rather than raised.
    _dies(direction, work)
    assert rounds.overview(direction)["problem"] == "unreadable"


def test_an_attempt_without_its_lease_yet_is_starting_and_only_later_lost(
    direction: Path,
) -> None:
    grant = _grant(direction, 1)
    # A run paused after taking its attempt, before registering its lease.
    _attempt(direction, grant, 1, lease=False)
    view = rounds.overview(direction)
    assert [run["state"] for run in view["grants"][0]["runs"]] == ["starting"]
    assert {item["kind"] for item in view["pending"]} & {"lost", "round-over"} == set()
    assert view["state"] == "running"
    # Only a crash leaves it so for long: then it is lost, and its round over.
    _aged(direction)
    view = rounds.overview(direction)
    assert [run["state"] for run in view["grants"][0]["runs"]] == ["lost"]
    assert {item["kind"] for item in view["pending"]} >= {"lost", "round-over"}
