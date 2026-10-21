"""Read-only helpers over a Shinka results directory."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Tuple, TypeVar
from urllib.parse import quote

# Administrative copies (migration and dynamic island spawning) are not trials
# and not charges; the cost rule owns the predicate.
from shinka.database.cost import NOT_COPY as _NOT_COPY
from shinka.database.cost import (
    Spend,
    program_spend,
    prompt_spend,
    recorded_in_full,
)

DB_FILENAME = "programs.sqlite"
T = TypeVar("T")


def has_programs(results_dir: Path) -> bool:
    """Whether the results database exists with its programs table. A run
    creates the file before its tables, so one caught or killed in between has
    neither; the readers below still treat that as an error, as for any
    damaged store, and only reconciling a dead run reads it as empty."""
    if not (Path(results_dir) / DB_FILENAME).exists():
        return False
    ((n,),) = _query(
        results_dir,
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'programs'",
    )
    return bool(n)


def count_evaluated(results_dir: Path) -> int:
    """Programs actually evaluated, excluding administrative island copies."""
    if not (Path(results_dir) / DB_FILENAME).exists():
        return 0
    ((n,),) = _query(results_dir, f"SELECT COUNT(*) FROM programs WHERE {_NOT_COPY}")
    return int(n)


def _connect(
    results_dir: Path, filename: str = DB_FILENAME
) -> "closing[sqlite3.Connection]":
    db = Path(results_dir) / filename
    if not db.exists():
        raise FileNotFoundError(db)
    # Encode the native filename so UNC paths never become URI authorities.
    # closing(): the connection's own context manager only commits, and an open
    # handle keeps the file locked on Windows after a failed read.
    return closing(
        sqlite3.connect(f"file:{quote(str(db.resolve()), safe='')}?mode=ro", uri=True)
    )


def _query(
    results_dir: Path, sql: str, params: tuple = (), filename: str = DB_FILENAME
) -> list:
    with _connect(results_dir, filename) as conn:
        return conn.execute(sql, params).fetchall()


def _read(results_dir: Path, filename: str, total: Callable[..., T]) -> T:
    with _connect(results_dir, filename) as conn:
        return total(conn)


def best_program(results_dir: Path) -> Tuple[str, str, float]:
    """(id, code, combined_score) of the best correct evaluated program."""
    rows = _query(
        results_dir,
        "SELECT id, code, combined_score FROM programs "
        f"WHERE correct = 1 AND {_NOT_COPY} "
        "ORDER BY combined_score DESC LIMIT 1",
    )
    if not rows:
        raise LookupError(f"No correct programs in {results_dir}")
    pid, code, score = rows[0]
    return str(pid), str(code), float(score)


def correct_scores(results_dir: Path) -> List[float]:
    """combined_score of every correct evaluated program."""
    rows = _query(
        results_dir,
        f"SELECT combined_score FROM programs WHERE correct = 1 AND {_NOT_COPY}",
    )
    return [float(s) for (s,) in rows if s is not None]


def trials(results_dir: Path) -> List[Tuple[float, int, Optional[float], str]]:
    """(timestamp, generation, in-sample score or None if incorrect, program id)
    of every evaluated program, oldest first."""
    if not (Path(results_dir) / DB_FILENAME).exists():
        return []
    rows = _query(
        results_dir,
        "SELECT timestamp, generation, correct, combined_score, id FROM programs "
        f"WHERE {_NOT_COPY} ORDER BY timestamp",
    )
    return [
        (float(t), int(g), float(s) if ok and s is not None else None, str(pid))
        for t, g, ok, s, pid in rows
    ]


def top_correct_programs(results_dir: Path, limit: int) -> List[Tuple[str, str]]:
    """(id, code) of the best correct programs, distinct code, by score."""
    rows = _query(
        results_dir,
        "SELECT id, code FROM programs "
        f"WHERE correct = 1 AND {_NOT_COPY} ORDER BY combined_score DESC",
    )
    seen, out = set(), []
    for pid, code in rows:
        if code not in seen:
            seen.add(code)
            out.append((str(pid), str(code)))
        if len(out) == limit:
            break
    return out


def program_code(results_dir: Path, program_id: str) -> str:
    rows = _query(results_dir, "SELECT code FROM programs WHERE id = ?", (program_id,))
    if not rows:
        raise LookupError(f"program {program_id} not in {results_dir}")
    return str(rows[0][0])


PROMPT_DB_FILENAME = "prompts.sqlite"


class RunCost(NamedTuple):
    """Recorded LLM spend of a run (USD), by the one cost rule
    (``shinka.database.cost``): evaluated programs without island copies,
    charge events of paid work that never became a program, and prompt
    evolution. A part whose ``*_known`` flag is false holds an unpriced or
    malformed entry, or the run marked a charge it could not record, so its
    amount is only a lower bound."""

    programs_usd: float
    programs_known: bool
    prompts_usd: float
    prompts_known: bool

    @property
    def recorded_usd(self) -> float:
        return self.programs_usd + self.prompts_usd

    @property
    def known(self) -> bool:
        return self.programs_known and self.prompts_known


def run_cost(results_dir: Path, db_name: str = DB_FILENAME) -> RunCost:
    """LLM spend of a run. The budget, grants, the workspace, the run
    statistics and the runner's resume base all use this one rule, so a run
    never shows two costs."""
    results_dir = Path(results_dir)
    programs = Spend(0.0, True)
    if (results_dir / db_name).exists():
        programs = _read(results_dir, db_name, program_spend)
    prompts = Spend(0.0, True)
    if (results_dir / PROMPT_DB_FILENAME).exists():
        prompts = _read(results_dir, PROMPT_DB_FILENAME, prompt_spend)
    return RunCost(
        programs.usd,
        programs.known and recorded_in_full(results_dir),
        prompts.usd,
        prompts.known,
    )


def best_score(results_dir: Path) -> float | None:
    if not (Path(results_dir) / DB_FILENAME).exists():
        return None
    ((score,),) = _query(
        results_dir,
        f"SELECT MAX(combined_score) FROM programs WHERE correct = 1 AND {_NOT_COPY}",
    )
    return None if score is None else float(score)
