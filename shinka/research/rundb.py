"""Read-only helpers over a Shinka results directory."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Tuple

DB_FILENAME = "programs.sqlite"


def count_evaluated(results_dir: Path) -> int:
    """Programs actually evaluated, excluding administrative island copies."""
    if not (Path(results_dir) / DB_FILENAME).exists():
        return 0
    ((n,),) = _query(results_dir, f"SELECT COUNT(*) FROM programs WHERE {_NOT_COPY}")
    return int(n)


# Administrative copies (migration and dynamic island spawning) are not trials;
# same predicate as shinka.database.dbase's evaluated_count.
_NOT_COPY = (
    "COALESCE(json_extract(metadata, '$._is_island_copy'), 0) = 0 "
    "AND COALESCE(json_extract(metadata, '$._spawned_island'), 0) = 0"
)


def _query(results_dir: Path, sql: str) -> list:
    db = Path(results_dir) / DB_FILENAME
    if not db.exists():
        raise FileNotFoundError(db)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return conn.execute(sql).fetchall()


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
