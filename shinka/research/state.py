"""Locking and atomic replacement for a direction's state files.

Every writer takes the direction's lock, reads the latest copy, changes it and
replaces the file in one step, so two terminals or the agent service never lose
each other's updates and a reader never sees half a file. Readers (the research
UI) take no lock: a replaced file is either the old one or the new one.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path
from typing import Dict, Iterator

LOCK_FILENAME = ".research.lock"

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
