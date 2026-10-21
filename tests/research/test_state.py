"""Concurrent writers of a direction's state (research-agent task 2.1).

Two terminals, or a terminal and the agent service, run shinka_research at the
same time; separate processes stand in for them.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import psutil
import pytest

from shinka.research import cli
from shinka.research.notebook import Experiment, Notebook, Status
from shinka.research.state import alive, atomic_write_text, file_lock

WRITER = """
import json, sys
from pathlib import Path
from shinka.research.state import atomic_write_text, file_lock
counter = Path(sys.argv[1])
for _ in range(int(sys.argv[2])):
    with file_lock(counter.with_name(".research.lock")):
        n = json.loads(counter.read_text())["n"]
        atomic_write_text(counter, json.dumps({"n": n + 1}))
"""

READER = """
import json, sys
from pathlib import Path
from shinka.research.state import read_text
counter, stop = Path(sys.argv[1]), Path(sys.argv[2])
reads = 0
while not stop.exists():
    json.loads(read_text(counter))
    reads += 1
print(reads)
"""

EDITOR = """
import sys
from pathlib import Path
from shinka.research import cli
from shinka.research.notebook import Notebook
root, tag, body = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
for k in range(5):
    cli.main(["hypothesis", "add", str(root), "--title", f"{tag}-{k}", "--file", body])
    for _ in range(2):
        with Notebook.edit(root) as nb:
            nb.hypotheses[0].experiments[0].n_probed += 1
"""


def _spawn(code: str, *args: object) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", code, *map(str, args)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _finish(process: subprocess.Popen) -> str:
    out, err = process.communicate(timeout=180)
    assert process.returncode == 0, err
    return out


def test_concurrent_writers_lose_no_update_and_a_reader_never_sees_half_a_file(
    tmp_path: Path,
) -> None:
    counter = tmp_path / "counter.json"
    atomic_write_text(counter, json.dumps({"n": 0}))
    stop = tmp_path / "stop"
    reader = _spawn(READER, counter, stop)
    writers = [_spawn(WRITER, counter, 50) for _ in range(4)]
    for writer in writers:
        _finish(writer)
    stop.touch()
    assert int(_finish(reader)) > 0
    assert json.loads(counter.read_text()) == {"n": 200}


def test_a_lock_held_by_this_process_can_be_taken_again(tmp_path: Path) -> None:
    # A locked edit that writes the run log takes the same lock twice.
    with file_lock(tmp_path / ".research.lock"):
        with file_lock(tmp_path / ".research.lock"):
            pass


def test_concurrent_commands_keep_every_hypothesis_and_probe(tmp_path: Path) -> None:
    root = tmp_path / "direction"
    body = tmp_path / "hypothesis.md"
    body.write_text("Statement.\n")
    cli.main(["init", str(root), "--direction", "Concurrent fixture"])
    cli.main(["hypothesis", "add", str(root), "--title", "first", "--file", str(body)])
    nb = Notebook.load(root)
    nb.hypotheses[0].experiments.append(
        Experiment(id="e01", evaluator_hash="e", data_path="d", data_hash="h")
    )
    nb.save(root)

    editors = [_spawn(EDITOR, root, f"p{i}", body) for i in range(4)]
    for editor in editors:
        _finish(editor)

    nb = Notebook.load(root)
    titles = sorted(h.title for h in nb.hypotheses[1:])
    assert titles == sorted(f"p{i}-{k}" for i in range(4) for k in range(5))
    assert [h.id for h in nb.hypotheses] == [f"h{n:03d}" for n in range(1, 22)]
    for h in nb.hypotheses:
        assert h.title in (root / "hypotheses" / f"{h.id}.md").read_text()
    assert nb.hypotheses[0].experiments[0].n_probed == 40


def _scaffolded(tmp_path: Path, example_dir: Path, bars_csv: Path) -> Path:
    root = tmp_path / "direction"
    cli.main(["init", str(root), "--direction", "Concurrent command fixture"])
    cli.main(
        [
            "hypothesis", "add", str(root), "--title", "probed",
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
    return root


def test_a_long_evaluation_does_not_hold_the_direction(
    tmp_path: Path, example_dir: Path, bars_csv: Path, monkeypatch
) -> None:
    root = _scaffolded(tmp_path, example_dir, bars_csv)
    hypothesis = ["--file", str(example_dir / "hypothesis.md")]
    started, release = threading.Event(), threading.Event()

    def slow_eval(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=60)
        return {"combined_score": 0.1}, True, None

    monkeypatch.setattr(cli, "run_quant_eval", slow_eval)
    program = str(example_dir / "initial.py")
    probe = threading.Thread(
        target=cli.main, args=(["probe", str(root), "h001", program],)
    )
    probe.start()
    assert started.wait(timeout=60)
    add = threading.Thread(
        target=cli.main,
        args=(["hypothesis", "add", str(root), "--title", "during", *hypothesis],),
    )
    add.start()
    add.join(timeout=10)
    try:
        assert not add.is_alive(), "hypothesis add waited for the probe"
    finally:
        release.set()
        probe.join(timeout=60)
    nb = Notebook.load(root)
    assert [h.title for h in nb.hypotheses] == ["probed", "during"]
    assert nb.hypotheses[0].experiments[0].n_probed == 1


def test_a_failed_run_keeps_a_status_another_command_set_meanwhile(
    tmp_path: Path, example_dir: Path, bars_csv: Path, monkeypatch
) -> None:
    root = _scaffolded(tmp_path, example_dir, bars_csv)

    class FailsWhileAnotherCommandChallenges:
        """A run whose process fails after another command moved the hypothesis on."""

        def __init__(self, _cmd):
            self.pid = os.getpid()

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def wait(self):
            with Notebook.edit(root) as nb:
                assert nb.get("h001").status is Status.running
                nb.get("h001").status = Status.challenged
            return 1

        def kill(self):
            pass

    monkeypatch.setattr(cli.subprocess, "Popen", FailsWhileAnotherCommandChallenges)
    with pytest.raises(SystemExit):
        cli.main(["run", str(root), "h001"])
    assert Notebook.load(root).get("h001").status is Status.challenged


def test_a_challenge_that_raced_another_change_records_nothing(
    tmp_path: Path, example_dir: Path, bars_csv: Path, monkeypatch
) -> None:
    root = _scaffolded(tmp_path, example_dir, bars_csv)

    def challenge_while_a_run_starts(_root, _nb, hid, _exp, _th):
        with Notebook.edit(root) as nb:
            nb.get(hid).status = Status.running
        return {}

    monkeypatch.setattr(cli, "challenge", challenge_while_a_run_starts)
    monkeypatch.setattr(cli, "decide", lambda _report, _th: (Status.supported, "x"))
    with pytest.raises(SystemExit, match="changed while the challenge ran"):
        cli.main(["skeptic", str(root), "h001"])
    nb = Notebook.load(root)
    assert nb.get("h001").status is Status.running
    assert nb.get("h001").experiments[0].skeptic is None
    assert not (root / "experiments" / "h001" / "e01" / "skeptic.md").exists()


class _Inaccessible:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    def create_time(self) -> float:
        raise psutil.AccessDenied(self.pid)


class _Zombie:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    def create_time(self) -> float:
        return 5.0

    def status(self) -> str:
        return psutil.STATUS_ZOMBIE


def test_a_run_whose_process_cannot_be_inspected_is_not_taken_for_over(
    monkeypatch,
) -> None:
    # So confirm keeps the lockbox sealed while it may still be adding trials.
    monkeypatch.setattr(psutil, "Process", _Inaccessible)
    assert alive({"pid": 1234, "created": 5.0})


def test_a_finished_process_not_yet_reaped_is_over(monkeypatch) -> None:
    monkeypatch.setattr(psutil, "Process", _Zombie)
    assert not alive({"pid": 1234, "created": 5.0})
