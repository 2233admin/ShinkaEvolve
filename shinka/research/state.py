"""Locking and atomic replacement for a direction's state files.

Every writer takes the direction's lock, reads the latest copy, changes it and
replaces the file in one step, so two terminals or the agent service never lose
each other's updates and a reader never sees half a file. Readers (the research
UI) take no lock: a replaced file is either the old one or the new one.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List

import psutil

LOCK_FILENAME = ".research.lock"
WORK_FILENAME = ".work.json"

if os.name == "nt":
    import msvcrt

    def _try_lock(fd: int) -> bool:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


# Both OS locks conflict between two handles of one process, so a nested
# acquisition would wait on itself: the process holds each lock once and
# counts nested holders of it.
_registry = threading.Lock()
_guards: Dict[str, threading.RLock] = {}
_depth: Dict[str, int] = {}


def lock_path(root: Path) -> Path:
    return Path(root) / LOCK_FILENAME


@contextlib.contextmanager
def file_lock(path: Path, timeout: float = 120.0) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` (created if missing) across processes;
    re-entrant within a process."""
    key = os.path.normcase(os.path.abspath(path))
    with _registry:
        guard = _guards.setdefault(key, threading.RLock())
    if not guard.acquire(timeout=timeout):
        raise TimeoutError(f"could not lock {path} within {timeout:.0f} s")
    try:
        if _depth.get(key, 0):
            _depth[key] += 1
            try:
                yield
            finally:
                _depth[key] -= 1
            return
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            deadline = time.monotonic() + timeout
            while not _try_lock(fd):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"could not lock {path} within {timeout:.0f} s; "
                        "another shinka_research command holds it"
                    )
                time.sleep(0.02)
            _depth[key] = 1
            try:
                yield
            finally:
                _depth[key] = 0
                _unlock(fd)
        finally:
            os.close(fd)
    finally:
        guard.release()


def atomic_write_text(path: Path, text: str, attempts: int = 100) -> None:
    """Replace ``path`` with ``text`` in one step, with the encoding and newline
    handling of ``Path.write_text``."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    # Windows refuses to replace a file that a reader has open; the reader lets
    # go within moments.
    for _ in range(attempts - 1):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.02)
    try:
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_text(path: Path, attempts: int = 100) -> str:
    """``Path.read_text`` that waits out a concurrent replacement, which Windows
    may briefly refuse to open."""
    for _ in range(attempts - 1):
        try:
            return Path(path).read_text()
        except PermissionError:
            time.sleep(0.02)
    return Path(path).read_text()


# A run registers here for its duration: confirm refuses to freeze the
# direction's trial count while one is still adding trials, a second run of the
# same experiment is refused, and the run's end is recorded from its entry. An
# entry names its processes (the command, a detached supervisor, the
# shinka_run child) by pid and start time, which survive neither a crash nor
# pid reuse, so an entry whose processes are all gone belongs to a run that
# ended unobserved.


def work_entries(root: Path) -> List[Dict[str, Any]]:
    path = Path(root) / WORK_FILENAME
    return json.loads(read_text(path)) if path.exists() else []


def _save_work(root: Path, entries: List[Dict[str, Any]]) -> None:
    atomic_write_text(Path(root) / WORK_FILENAME, json.dumps(entries))


def register_work(
    root: Path, hid: str, eid: str, work_id: str | None = None, **details: Any
) -> str:
    """Record that this process is adding trials to an experiment, with what its
    end needs (``details``); returns the entry's id (``work_id`` when a record
    made first names it). Take the direction's lock around it together with the
    checks that allowed the work, so others see both or neither."""
    entry = {
        **details,
        "id": work_id or uuid.uuid4().hex,
        "hypothesis": hid,
        "experiment": eid,
        "pid": os.getpid(),
        "created": psutil.Process().create_time(),
    }
    with file_lock(lock_path(root)):
        _save_work(root, work_entries(root) + [entry])
    return entry["id"]


def record_exit(root: Path, work_id: str, code: int) -> None:
    """Keep the work's exit code with it the moment it is known, so whoever
    records the end reports what happened even if the waiter dies first."""
    with file_lock(lock_path(root)):
        entries = work_entries(root)
        for entry in entries:
            if entry["id"] == work_id:
                entry["exit_code"] = code
        _save_work(root, entries)


def hand_over(root: Path, work_id: str, pid: int, replacing: int | None = None) -> None:
    """Make ``pid`` the process that owns the work: a detached run's supervisor
    takes over from the command that started it, which may live on (an agent
    service) long after the run ends. With ``replacing``, only while that
    process still owns it, so the command's hand-over never undoes the
    supervisor's own."""
    try:
        created = psutil.Process(pid).create_time()
    except psutil.Error:
        created = 0.0  # it has already ended; the entry then reads as ended too
    with file_lock(lock_path(root)):
        entries = work_entries(root)
        for entry in entries:
            if entry["id"] == work_id and replacing in (None, entry["pid"]):
                entry["pid"], entry["created"] = pid, created
        _save_work(root, entries)


def attach_worker(root: Path, work_id: str, pid: int) -> None:
    """Add a process doing the work: it may go on adding trials after the
    command that registered the work is gone. On Windows it is usually a
    launcher (a console script's .exe, a venv's python.exe) with the
    interpreter behind it; a launcher waits for that interpreter and takes it
    down when killed, so the launcher's pid stands for both."""
    try:
        worker = {"pid": pid, "created": psutil.Process(pid).create_time()}
    except psutil.Error:
        return  # it has already ended
    with file_lock(lock_path(root)):
        entries = work_entries(root)
        for entry in entries:
            if entry["id"] == work_id:
                entry["workers"] = entry.get("workers", []) + [worker]
        _save_work(root, entries)


def alive(entry: Dict[str, Any]) -> bool:
    """Whether any process of a work entry is still running."""

    def running(process: Dict[str, Any]) -> bool:
        try:
            started = psutil.Process(process["pid"]).create_time()
        except psutil.Error:
            return False
        return abs(started - process["created"]) < 1.0

    return any(running(p) for p in [entry, *entry.get("workers", [])])


def live_work(root: Path) -> List[Dict[str, Any]]:
    """Registered work with a process still alive."""
    return [e for e in work_entries(root) if alive(e)]


def find_work(root: Path, work_id: str) -> Dict[str, Any] | None:
    return next((e for e in work_entries(root) if e["id"] == work_id), None)


def drop_work(root: Path, work_id: str) -> None:
    """Remove a work entry; the last step of recording the work's end, so a
    crash before it leaves the entry for the next attempt."""
    with file_lock(lock_path(root)):
        _save_work(root, [e for e in work_entries(root) if e["id"] != work_id])
