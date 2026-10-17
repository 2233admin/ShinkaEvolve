"""Read-only helpers over a Shinka results directory."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

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


def _query(results_dir: Path, sql: str, params: tuple = ()) -> list:
    db = Path(results_dir) / DB_FILENAME
    if not db.exists():
        raise FileNotFoundError(db)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return conn.execute(sql, params).fetchall()


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


def trials(results_dir: Path) -> List[Tuple[float, int, Optional[float]]]:
    """(timestamp, generation, in-sample score or None if incorrect) of every
    evaluated program, oldest first."""
    if not (Path(results_dir) / DB_FILENAME).exists():
        return []
    rows = _query(
        results_dir,
        "SELECT timestamp, generation, correct, combined_score FROM programs "
        f"WHERE {_NOT_COPY} ORDER BY timestamp",
    )
    return [
        (float(t), int(g), float(s) if ok and s is not None else None)
        for t, g, ok, s in rows
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


_COST = " + ".join(
    f"CASE WHEN json_type(metadata, '$.{k}') IN ('integer', 'real') "
    f"THEN json_extract(metadata, '$.{k}') ELSE 0 END"
    for k in ("api_costs", "embed_cost", "novelty_cost", "meta_cost")
)


PROMPT_DB_FILENAME = "prompts.sqlite"


def total_cost(results_dir: Path) -> float:
    """LLM spend of a run (USD), as Shinka totals it: program costs plus
    prompt-evolution costs (``prompts.sqlite``, metadata.llm.cost)."""
    results_dir = Path(results_dir)
    cost = 0.0
    if (results_dir / DB_FILENAME).exists():
        ((c,),) = _query(
            results_dir,
            f"SELECT COALESCE(SUM({_COST}), 0) FROM programs WHERE {_NOT_COPY}",
        )
        cost += float(c)
    prompts = results_dir / PROMPT_DB_FILENAME
    if prompts.exists():
        with sqlite3.connect(f"file:{prompts}?mode=ro", uri=True) as conn:
            ((c,),) = conn.execute(
                "SELECT COALESCE(SUM(CASE WHEN json_type(metadata, '$.llm.cost') "
                "IN ('integer', 'real') THEN json_extract(metadata, '$.llm.cost') "
                "ELSE 0 END), 0) FROM system_prompts"
            ).fetchall()
        cost += float(c)
    return cost


def best_score(results_dir: Path) -> float | None:
    if not (Path(results_dir) / DB_FILENAME).exists():
        return None
    ((score,),) = _query(
        results_dir,
        f"SELECT MAX(combined_score) FROM programs WHERE correct = 1 AND {_NOT_COPY}",
    )
    return None if score is None else float(score)
