import contextlib
import json
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pytest

from shinka.research import cli, grants, skeptic
from shinka.research.grants import AGENT_ENV, LEDGER
from shinka.research.notebook import Notebook, Status, Verdict
from shinka.research.skeptic import pbo
from shinka.research.state import WORK_FILENAME, file_lock


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


def _mark_supported(root: Path, code: str, hid: str = "h001") -> None:
    results = root / f"experiments/{hid}/e01/results"
    results.mkdir(parents=True)
    with sqlite3.connect(results / "programs.sqlite") as conn:
        conn.execute(
            "CREATE TABLE programs (id TEXT, code TEXT, combined_score REAL, "
            "correct INTEGER, metadata TEXT)"
        )
        conn.execute("INSERT INTO programs VALUES ('p1', ?, 1.0, 1, '{}')", (code,))
    nb = Notebook.load(root)
    exp = nb.get(hid).experiments[0]
    exp.skeptic = {"rule_label": "supported", "rule_reason": "t", "program_id": "p1"}
    nb.get(hid).status = Status.challenged
    nb.set_verdict(
        hid,
        Verdict(
            label=Status.supported,
            reason="t",
            evidence={"experiment": "e01", **exp.skeptic},
        ),
    )
    nb.save(root)


def test_confirm_opens_lockbox_once(tmp_path: Path, example_dir: Path, bars_csv: Path):
    root = _direction(tmp_path, example_dir, bars_csv)
    with pytest.raises(SystemExit, match="no supported"):
        cli.main(["confirm", str(root), "--data", str(bars_csv)])
    _mark_supported(root, (example_dir / "initial.py").read_text())
    with pytest.raises(SystemExit, match="--yes"):
        cli.main(["confirm", str(root), "--data", str(bars_csv)])
    assert not Notebook.load(root).data.opened

    cli.main(["confirm", str(root), "--data", str(bars_csv), "--yes"])
    nb = Notebook.load(root)
    hyp = nb.get("h001")
    assert nb.data.opened
    assert hyp.status in (Status.confirmed, Status.unconfirmed)
    assert hyp.confirmation["lockbox_bars"] == 200
    assert (root / "CONFIRMATION.md").exists()
    with pytest.raises(SystemExit, match="already opened"):
        cli.main(["confirm", str(root), "--data", str(bars_csv), "--yes"])


def test_a_program_that_looks_ahead_inside_the_lockbox_is_unconfirmed(
    tmp_path: Path, example_dir: Path, bars_csv: Path
):
    root = _direction(tmp_path, example_dir, bars_csv)
    _mark_supported(root, LOCKBOX_PEEK)
    cli.main(["confirm", str(root), "--data", str(bars_csv), "--yes"])
    nb = Notebook.load(root)
    assert nb.data.opened and not nb.data.unfinished
    assert nb.get("h001").status is Status.unconfirmed
    # Only the category: the program's own output may carry lockbox figures.
    assert nb.get("h001").confirmation == {
        "experiment": "e01",
        "program_id": "p1",
        "lockbox_start": nb.data.lockbox_start,
        "passed": False,
        "failure": "look-ahead",
    }


def test_confirm_checks_source_and_evaluator(
    tmp_path: Path, example_dir: Path, bars_csv: Path
):
    root = _direction(tmp_path, example_dir, bars_csv)
    _mark_supported(root, (example_dir / "initial.py").read_text())
    other = tmp_path / "other.csv"
    other.write_text(bars_csv.read_text() + "2099-01-01,1.0\n")
    with pytest.raises(ValueError, match="not the data"):
        cli.main(["confirm", str(root), "--data", str(other), "--yes"])
    (root / "experiments/h001/e01/task/evaluate.py").write_text("# tampered\n")
    with pytest.raises(RuntimeError, match="modified"):
        cli.main(["confirm", str(root), "--data", str(bars_csv), "--yes"])
    nb = Notebook.load(root)
    assert not nb.data.opened
    assert "source_path" not in Notebook.path(root).read_text()


class _Crash(Exception):
    pass


def _confirm(root: Path, bars_csv: Path, *extra: str) -> None:
    cli.main(["confirm", str(root), "--data", str(bars_csv), "--yes", *extra])


def _disk_error(*_args) -> None:
    raise OSError("disk")


def test_a_confirm_that_died_after_the_program_saw_the_lockbox_never_reruns_it(
    tmp_path: Path, example_dir: Path, bars_csv: Path, monkeypatch
):
    root = _direction(tmp_path, example_dir, bars_csv)
    _mark_supported(root, (example_dir / "initial.py").read_text())
    calls = []

    def dies(*_args, **_kwargs):
        calls.append(1)
        raise _Crash

    monkeypatch.setattr(skeptic, "positions_for", dies)
    with pytest.raises(_Crash):
        _confirm(root, bars_csv)
    nb = Notebook.load(root)
    assert nb.data.opened and nb.data.unfinished
    assert nb.data.opening.target("h001").state == "scoring"
    # Until it finishes, nothing may change the hypotheses it froze.
    with pytest.raises(SystemExit, match="unfinished"):
        cli.main(["skeptic", str(root), "h001"])

    _confirm(root, bars_csv)
    nb = Notebook.load(root)
    assert calls == [1]
    assert not nb.data.unfinished
    assert nb.get("h001").status is Status.unconfirmed
    assert nb.get("h001").confirmation["failure"] == "interrupted"


def test_a_confirm_that_died_before_the_program_ran_finishes_under_its_frozen_rules(
    tmp_path: Path, example_dir: Path, bars_csv: Path, monkeypatch
):
    root = _direction(tmp_path, example_dir, bars_csv)
    _mark_supported(root, (example_dir / "initial.py").read_text())
    monkeypatch.setattr(skeptic, "load_bars", _disk_error)
    with pytest.raises(OSError):
        _confirm(root, bars_csv)
    nb = Notebook.load(root)
    assert nb.data.unfinished and nb.data.opening.target("h001").state == "pending"

    monkeypatch.undo()
    # A resume cannot change the rules after the lockbox was opened.
    _confirm(root, bars_csv, "--dsr", "0")
    nb = Notebook.load(root)
    rep = nb.get("h001").confirmation
    assert not nb.data.unfinished and rep["lockbox_bars"] == 200
    assert rep["passed"] == (
        rep["lockbox_sharpe"] > max(0.0, rep["lockbox_buy_hold_sharpe"])
        and rep["dsr"] >= 0.95
        and rep["lockbox_positive_fraction"] >= 2 / 3
    )


def test_a_program_changed_after_the_lockbox_opened_is_not_run(
    tmp_path: Path, example_dir: Path, bars_csv: Path, monkeypatch
):
    root = _direction(tmp_path, example_dir, bars_csv)
    _mark_supported(root, (example_dir / "initial.py").read_text())
    monkeypatch.setattr(skeptic, "load_bars", _disk_error)
    with pytest.raises(OSError):
        _confirm(root, bars_csv)
    monkeypatch.undo()
    results = root / "experiments/h001/e01/results/programs.sqlite"
    with sqlite3.connect(results) as conn:
        conn.execute("UPDATE programs SET code = ? WHERE id = 'p1'", (LOCKBOX_PEEK,))
    calls = []
    monkeypatch.setattr(skeptic, "positions_for", lambda *_a, **_k: calls.append(1))
    _confirm(root, bars_csv)
    nb = Notebook.load(root)
    assert calls == []
    assert nb.get("h001").confirmation["failure"] == "changed"


def test_a_second_confirm_is_refused_without_opening_anything(
    tmp_path: Path, example_dir: Path, bars_csv: Path
):
    root = _direction(tmp_path, example_dir, bars_csv)
    _mark_supported(root, (example_dir / "initial.py").read_text())
    held, release = threading.Event(), threading.Event()

    def another_confirm() -> None:
        with file_lock(root / cli.CONFIRM_LOCK):
            held.set()
            release.wait(timeout=60)

    holder = threading.Thread(target=another_confirm)
    holder.start()
    assert held.wait(timeout=10)
    try:
        with pytest.raises(SystemExit, match="another confirm"):
            _confirm(root, bars_csv)
    finally:
        release.set()
        holder.join(timeout=10)
    assert not Notebook.load(root).data.opened


def test_no_lockbox_result_is_readable_before_every_target_is_scored(
    tmp_path: Path, example_dir: Path, bars_csv: Path, monkeypatch
):
    root = _direction(tmp_path, example_dir, bars_csv)
    hypothesis = ["--file", str(example_dir / "hypothesis.md")]
    cli.main(["hypothesis", "add", str(root), "--title", "second", *hypothesis])
    initial = example_dir / "initial.py"
    cli.main(["scaffold", str(root), "h002", "--initial", str(initial)])
    _mark_supported(root, initial.read_text())
    _mark_supported(root, initial.read_text(), "h002")
    readable, real = [], skeptic.positions_for

    def what_a_program_could_read(*args, **kwargs):
        # Candidate programs can read the notebook while they run.
        readable.append([h.confirmation for h in Notebook.load(root).hypotheses])
        return real(*args, **kwargs)

    monkeypatch.setattr(skeptic, "positions_for", what_a_program_could_read)
    _confirm(root, bars_csv)
    assert readable == [[None, None], [None, None]]
    nb = Notebook.load(root)
    assert [h.confirmation["lockbox_bars"] for h in nb.hypotheses] == [200, 200]


def test_a_confirm_that_died_after_recording_writes_its_files_on_the_next_run(
    tmp_path: Path, example_dir: Path, bars_csv: Path, monkeypatch
):
    root = _direction(tmp_path, example_dir, bars_csv)
    _mark_supported(root, (example_dir / "initial.py").read_text())

    def dies(*_args):
        raise _Crash

    monkeypatch.setattr(cli, "append_log", dies)
    with pytest.raises(_Crash):
        _confirm(root, bars_csv)
    monkeypatch.undo()
    nb = Notebook.load(root)
    assert nb.data.opened and not nb.data.unfinished
    assert not (root / "CONFIRMATION.md").exists()

    scored = []
    monkeypatch.setattr(skeptic, "positions_for", lambda *_a, **_k: scored.append(1))
    _confirm(root, bars_csv)
    assert scored == []
    assert "h001" in (root / "CONFIRMATION.md").read_text()
    rows = (root / "results.tsv").read_text().splitlines()[1:]
    assert [row.split("\t")[3] for row in rows].count("confirm") == 1
    with pytest.raises(SystemExit, match="already opened"):
        _confirm(root, bars_csv)


def test_a_probe_counts_as_a_trial_from_the_moment_it_starts(
    tmp_path: Path, example_dir: Path, bars_csv: Path, monkeypatch
):
    root = _direction(tmp_path, example_dir, bars_csv)
    _mark_supported(root, (example_dir / "initial.py").read_text())
    started, release = threading.Event(), threading.Event()

    def crashes_later(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=60)
        raise _Crash

    def probe() -> None:
        with contextlib.suppress(_Crash):
            cli.main(["probe", str(root), "h001", str(example_dir / "initial.py")])

    monkeypatch.setattr(cli, "run_quant_eval", crashes_later)
    probing = threading.Thread(target=probe)
    probing.start()
    assert started.wait(timeout=60)
    try:
        # Still evaluating, yet already in the trial count the lockbox freezes.
        _confirm(root, bars_csv)
        assert Notebook.load(root).data.opening.target("h001").n_trials == 2
    finally:
        release.set()
        probing.join(timeout=60)
    assert Notebook.load(root).get("h001").experiments[0].n_probed == 1


def test_work_left_by_a_dead_process_does_not_hold_the_lockbox(
    tmp_path: Path, example_dir: Path, bars_csv: Path
):
    root = _direction(tmp_path, example_dir, bars_csv)
    _mark_supported(root, (example_dir / "initial.py").read_text())
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    entry = {"id": "x", "hypothesis": "h001", "experiment": "e01", "pid": gone.pid}
    (root / WORK_FILENAME).write_text(json.dumps([{**entry, "created": 0.0}]))
    _confirm(root, bars_csv)
    assert Notebook.load(root).data.opened


def test_a_run_whose_command_died_holds_the_lockbox_while_its_child_runs(
    tmp_path: Path, example_dir: Path, bars_csv: Path
):
    root = _direction(tmp_path, example_dir, bars_csv)
    _mark_supported(root, (example_dir / "initial.py").read_text())
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        worker = {"pid": child.pid, "created": psutil.Process(child.pid).create_time()}
        entry = {"id": "x", "hypothesis": "h001", "experiment": "e01"}
        work = {**entry, "pid": gone.pid, "created": 0.0, "workers": [worker]}
        (root / WORK_FILENAME).write_text(json.dumps([work]))
        with pytest.raises(SystemExit, match="still being added"):
            _confirm(root, bars_csv)
    finally:
        child.kill()
        child.wait()
    _confirm(root, bars_csv)
    assert Notebook.load(root).data.opened


def test_a_view_of_nan_positions_cannot_pass_the_look_ahead_check(
    tmp_path: Path, example_dir: Path, bars_csv: Path, monkeypatch
):
    root = _direction(tmp_path, example_dir, bars_csv)
    _mark_supported(root, (example_dir / "initial.py").read_text())
    # NaN differs from nothing under a tolerance check, so it would look causal.
    monkeypatch.setattr(skeptic, "run_view", lambda *a: np.full(a[3], np.nan))
    _confirm(root, bars_csv)
    assert (
        Notebook.load(root).get("h001").confirmation["failure"] == "invalid-positions"
    )


# ---- The owner's lockbox approval (research-agent 2.4) ----------------------


def _approve(root: Path) -> str:
    cli.main(["approve", str(root)])
    ledger = json.loads((root / LEDGER).read_text())
    return [r for r in ledger if r["kind"] == "approval"][-1]["id"]


def _confirm_with(root: Path, bars_csv: Path, *extra: str) -> None:
    cli.main(["confirm", str(root), "--data", str(bars_csv), *extra])


def test_an_agent_opens_the_lockbox_only_with_the_owners_approval(
    tmp_path: Path, example_dir: Path, bars_csv: Path, monkeypatch
):
    root = _direction(tmp_path, example_dir, bars_csv)
    _mark_supported(root, (example_dir / "initial.py").read_text())
    monkeypatch.setenv(AGENT_ENV, "1")
    # --yes is not enough there, and the agent cannot approve itself.
    with pytest.raises(SystemExit, match="lockbox approval"):
        _confirm(root, bars_csv)
    with pytest.raises(SystemExit, match="the owner's to give"):
        cli.main(["approve", str(root)])
    with pytest.raises(SystemExit, match="no lockbox approval a009"):
        _confirm_with(root, bars_csv, "--approval", "a009")
    assert not Notebook.load(root).data.opened
    monkeypatch.delenv(AGENT_ENV)
    approval = _approve(root)
    monkeypatch.setenv(AGENT_ENV, "1")
    _confirm_with(root, bars_csv, "--approval", approval)
    nb = Notebook.load(root)
    assert nb.data.opened and not nb.data.unfinished
    uses = [
        r
        for r in json.loads((root / LEDGER).read_text())
        if r["kind"] == "approval-used"
    ]
    assert [(u["approval"], u["opening"]) for u in uses] == [
        (approval, nb.data.opening.started)
    ]


def test_an_approval_is_spent_before_the_opening_and_only_once(
    tmp_path: Path, example_dir: Path, bars_csv: Path, monkeypatch
):
    root = _direction(tmp_path, example_dir, bars_csv)
    _mark_supported(root, (example_dir / "initial.py").read_text())
    first, second = _approve(root), _approve(root)
    spend = grants.use_approval

    def dies_after_spending(*args):
        spend(*args)
        raise _Crash

    monkeypatch.setattr(grants, "use_approval", dies_after_spending)
    monkeypatch.setenv(AGENT_ENV, "1")
    with pytest.raises(_Crash):
        _confirm_with(root, bars_csv, "--approval", first)
    # A crash between the two leaves the lockbox sealed, never opened unpaid.
    assert not Notebook.load(root).data.opened
    monkeypatch.setattr(grants, "use_approval", spend)
    with pytest.raises(SystemExit, match=f"approval {first} was already used"):
        _confirm_with(root, bars_csv, "--approval", first)
    _confirm_with(root, bars_csv, "--approval", second)
    assert Notebook.load(root).data.opened


def test_an_agent_finishes_an_opening_only_with_the_approval_that_opened_it(
    tmp_path: Path, example_dir: Path, bars_csv: Path, monkeypatch
):
    root = _direction(tmp_path, example_dir, bars_csv)
    _mark_supported(root, (example_dir / "initial.py").read_text())
    opened_with, other = _approve(root), _approve(root)
    monkeypatch.setenv(AGENT_ENV, "1")
    load_bars = skeptic.load_bars
    monkeypatch.setattr(skeptic, "load_bars", _disk_error)
    with pytest.raises(OSError):
        _confirm_with(root, bars_csv, "--approval", opened_with)
    assert Notebook.load(root).data.unfinished
    monkeypatch.setattr(skeptic, "load_bars", load_bars)
    for extra, refusal in (
        (["--yes"], "lockbox approval"),
        (["--approval", other], "did not open this lockbox"),
    ):
        with pytest.raises(SystemExit, match=refusal):
            _confirm_with(root, bars_csv, *extra)
    _confirm_with(root, bars_csv, "--approval", opened_with)
    assert not Notebook.load(root).data.unfinished
