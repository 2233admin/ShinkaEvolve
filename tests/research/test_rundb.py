import json
import sqlite3
from pathlib import Path

from shinka.research.rundb import count_evaluated


def test_count_evaluated_excludes_island_copies(tmp_path: Path) -> None:
    assert count_evaluated(tmp_path) == 0
    with sqlite3.connect(tmp_path / "programs.sqlite") as conn:
        conn.execute("CREATE TABLE programs (id TEXT, metadata TEXT)")
        rows = [("a", "{}"), ("b", None), ("c", json.dumps({"_is_island_copy": True}))]
        conn.executemany("INSERT INTO programs VALUES (?, ?)", rows)
    assert count_evaluated(tmp_path) == 2
