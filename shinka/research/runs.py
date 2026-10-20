"""Research runs that outlive the window that started them (research-agent 2.2).

A run registers as work in its direction (``state.register_work``) together
with what its end restores. That end is recorded exactly once, by whichever
gets there first: the command that waited for the run, its detached
supervisor, or ``watch`` finding every process of the run gone.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from shinka.research.notebook import (
    RESULTS_LOG,
    Hypothesis,
    Notebook,
    Status,
    Verdict,
    append_log,
    experiment_dir,
)
from shinka.research.rundb import best_score, count_evaluated, run_cost
from shinka.research.state import (
    alive,
    attach_worker,
    drop_work,
    file_lock,
    find_work,
    hand_over,
    lock_path,
    record_exit,
    work_entries,
)

# A detached run has no console: its output goes here, beside its results.
RUN_LOG = "run.log"

_END_EVENTS = ("run", "run-failed")


def _run_key(entry: Dict[str, Any]) -> str:
    """How a run's end event names the run (its full id), so a retried end is
    not written twice."""
    return f"run {entry['id']}"


def _end_logged(root: Path, entry: Dict[str, Any]) -> bool:
    path = Path(root) / RESULTS_LOG
    rows = path.read_text().splitlines()[1:] if path.exists() else []
    key = _run_key(entry)
    for row in rows:
        _time, hid, eid, event, note = (row.split("\t", 4) + [""] * 5)[:5]
        if (hid, eid) == (entry["hypothesis"], entry["experiment"]):
            if event in _END_EVENTS and note.endswith(key):
                return True
    return False


def hypothesis_evaluated(root: Path, hyp: Hypothesis) -> int:
    """Evaluated programs across a hypothesis's experiments."""
    return sum(
        count_evaluated(experiment_dir(root, hyp.id, e.id) / "results")
        for e in hyp.experiments
    )


def finish_run(
    root: Path, work_id: str, code: Optional[int], dead_only: bool = False
) -> Optional[str]:
    """Record the end of a registered run: its trial count, the hypothesis's
    earlier status when its runs added nothing, and one event. An exit code kept
    with the run wins over ``code``, which is None for a run whose processes
    vanished before anyone saw it end; ``dead_only`` leaves a run alone while
    any of its processes lives. Every step is idempotent and the entry goes
    last, so a crash midway leaves it for the next attempt. Returns the event,
    or None when there was nothing to record."""
    with file_lock(lock_path(root)):
        entry = find_work(root, work_id)
        if entry is None or (dead_only and alive(entry)):
            return None
        code = entry.get("exit_code", code)
        hid, eid = entry["hypothesis"], entry["experiment"]
        results = experiment_dir(root, hid, eid) / "results"
        before = entry["evaluated_before"]
        with Notebook.edit(root) as nb:
            hyp = nb.get(hid)
            exp = next(e for e in hyp.experiments if e.id == eid)
            exp.n_evaluated = count_evaluated(results)
            # Overlapping runs of a hypothesis share one baseline: it goes back
            # to its earlier status only when the last of them fails and none
            # added a program, and only while still marked running (another
            # command may have moved it on meanwhile).
            others = any(
                w["hypothesis"] == hid and w["id"] != work_id and alive(w)
                for w in work_entries(root)
            )
            idle = hypothesis_evaluated(root, hyp) == entry["before"]
            if code != 0 and not others and idle and hyp.status is Status.running:
                hyp.status = Status(entry["previous_status"])
                previous = entry["previous_verdict"]
                hyp.verdict = None if previous is None else Verdict(**previous)
        event = "run" if code == 0 else "run-failed"
        if not _end_logged(root, entry):
            best = best_score(results)
            cost = run_cost(results)
            append_log(
                root, hid, eid, event,
                ("ended unobserved, " if code is None else "")
                + f"{exp.n_evaluated - before} new programs, "
                f"{exp.n_evaluated} total, best IS score "
                f"{'n/a' if best is None else f'{best:.3f}'}, "
                f"spend {'' if cost.known else '>='}${cost.recorded_usd:.2f}, "
                + _run_key(entry),
            )  # fmt: skip
        drop_work(root, work_id)
    return event


def watch(root: Path) -> List[str]:
    """Record the end of every run of a direction whose processes are all gone
    (a killed supervisor, a closed foreground window): one line per run."""
    lines = []
    for entry in work_entries(root):
        event = finish_run(root, entry["id"], None, dead_only=True)
        if event:
            lines.append(f"{entry['hypothesis']}/{entry['experiment']}: {event}")
    return lines


def detached() -> Dict[str, Any]:
    """Popen options for a process that survives closing the window it was
    started from: no console and a session or process group of its own."""
    options: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        options["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        options["start_new_session"] = True
    return options


def supervise(root: Path, work_id: str) -> None:
    """Run a registered run's command and record its end; the body of a
    detached run."""
    # The command that started it hands the run over too; doing it here as
    # well covers that command dying first.
    hand_over(root, work_id, os.getpid())
    entry = find_work(root, work_id)
    if entry is None:
        return  # watch recorded it before this process got going
    log = experiment_dir(root, entry["hypothesis"], entry["experiment"]) / RUN_LOG
    # Without a console of its own, a console child would open a window.
    window = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    code: Optional[int] = 1
    try:
        with log.open("a") as out, subprocess.Popen(
            entry["cmd"], stdin=subprocess.DEVNULL, stdout=out,
            stderr=subprocess.STDOUT, **window,
        ) as proc:  # fmt: skip
            attach_worker(root, work_id, proc.pid)
            code = proc.wait()
            record_exit(root, work_id, code)
    finally:
        finish_run(root, work_id, code)


if __name__ == "__main__":
    supervise(Path(sys.argv[1]), sys.argv[2])
