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


# Work that adds trials (a run, a probe) registers here for its duration, so
# confirm can refuse to freeze the direction's trial count while any is still
# under way. An entry names its processes (the command and any child doing the
# work) by pid and start time, which survive neither a crash nor pid reuse, so
# an entry whose processes are all gone is simply ignored.


def _work(root: Path) -> List[Dict[str, Any]]:
    path = Path(root) / WORK_FILENAME
    return json.loads(read_text(path)) if path.exists() else []


def register_work(root: Path, hid: str, eid: str) -> str:
    """Record that this process is adding trials to an experiment; returns the
    entry's id. Take the direction's lock around it together with the checks
    that allowed the work, so confirm sees both or neither."""
    entry = {
        "id": uuid.uuid4().hex,
        "hypothesis": hid,
        "experiment": eid,
        "pid": os.getpid(),
        "created": psutil.Process().create_time(),
    }
    with file_lock(lock_path(root)):
        atomic_write_text(Path(root) / WORK_FILENAME, json.dumps(_work(root) + [entry]))
    return entry["id"]


def finish_work(root: Path, work_id: str) -> None:
    with file_lock(lock_path(root)):
        rest = [e for e in _work(root) if e["id"] != work_id]
        atomic_write_text(Path(root) / WORK_FILENAME, json.dumps(rest))


def attach_worker(root: Path, work_id: str, pid: int) -> None:
    """Add a child process doing the work: it may go on adding trials after the
    command that registered the work is gone."""
    try:
        worker = {"pid": pid, "created": psutil.Process(pid).create_time()}
    except psutil.Error:
        return  # it has already ended
    with file_lock(lock_path(root)):
        entries = _work(root)
        for entry in entries:
            if entry["id"] == work_id:
                entry["workers"] = entry.get("workers", []) + [worker]
        atomic_write_text(Path(root) / WORK_FILENAME, json.dumps(entries))


def live_work(root: Path) -> List[Dict[str, Any]]:
    """Registered work with a process still alive."""

    def alive(process: Dict[str, Any]) -> bool:
        try:
            started = psutil.Process(process["pid"]).create_time()
        except psutil.Error:
            return False
        return abs(started - process["created"]) < 1.0

    return [e for e in _work(root) if any(alive(p) for p in [e, *e.get("workers", [])])]
