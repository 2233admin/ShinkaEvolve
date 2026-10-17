"""Read-only helpers over a Shinka results directory."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_FILENAME = "programs.sqlite"


def count_evaluated(results_dir: Path) -> int:
    """Programs actually evaluated, excluding administrative island copies."""
    db = Path(results_dir) / DB_FILENAME
    if not db.exists():
        return 0
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM programs WHERE "
            "COALESCE(json_extract(metadata, '$._is_island_copy'), 0) = 0"
        ).fetchone()
    return int(n)
