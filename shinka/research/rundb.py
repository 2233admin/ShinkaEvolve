"""Read-only helpers over a Shinka results directory."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple
from urllib.parse import quote

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


def _query(
    results_dir: Path, sql: str, params: tuple = (), filename: str = DB_FILENAME
) -> list:
    db = Path(results_dir) / filename
    if not db.exists():
        raise FileNotFoundError(db)
    # Encode the native filename so UNC paths never become URI authorities.
    # closing(): the connection's own context manager only commits, and an open
    # handle keeps the file locked on Windows after a failed read.
    with closing(
        sqlite3.connect(f"file:{quote(str(db.resolve()), safe='')}?mode=ro", uri=True)
    ) as conn:
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
COST_FIELDS = ("api_costs", "embed_cost", "novelty_cost", "meta_cost")


class RunCost(NamedTuple):
    """Recorded LLM spend of a run (USD). A part whose ``*_known`` flag is false
    holds an unpriced or malformed entry, so its amount is only a lower bound."""

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


def _cost_field(path: str) -> Tuple[str, str]:
    """(amount, recorded) SQL for one cost field of valid JSON ``metadata``.
    An absent or null field is a recorded zero: the runner writes a field only
    for the steps that ran (``meta_cost`` only on a program a meta update
    charged) and totals missing ones as 0. A present value must be a finite
    number >= 0; anything else leaves the cost unknown."""
    kind = f"json_type(metadata, '$.{path}')"
    value = f"json_extract(metadata, '$.{path}')"
    number = f"({kind} IN ('integer', 'real') AND {value} >= 0 AND {value} < 1e999)"
    recorded = f"({kind} IS NULL OR {kind} = 'null' OR {number})"
    return f"(CASE WHEN {number} THEN {value} ELSE 0 END)", recorded


def _cost_sql(table: str, amount: str, recorded: str, skip: str) -> str:
    """Total of ``amount`` and whether every row is recorded. Null metadata is
    a recorded zero, malformed metadata is unknown, and rows matching ``skip``
    are not charges. CASE keeps the JSON functions off malformed rows."""
    return (
        "SELECT COALESCE(SUM(amount), 0), COALESCE(MIN(recorded), 1) FROM ("
        "SELECT CASE WHEN metadata IS NULL THEN 0 WHEN json_valid(metadata) "
        f"THEN CASE WHEN {skip} THEN 0 ELSE {amount} END ELSE 0 END AS amount, "
        "CASE WHEN metadata IS NULL THEN 1 WHEN json_valid(metadata) "
        f"THEN CASE WHEN {skip} OR {recorded} THEN 1 ELSE 0 END ELSE 0 END "
        f"AS recorded FROM {table})"
    )


_FIELDS = {field: _cost_field(field) for field in COST_FIELDS}
_MODEL = (
    "COALESCE("
    + ", ".join(
        f"NULLIF(json_extract(metadata, '$.{p}'), '')"
        for p in ("model_name", "model", "llm_result.model_name", "llm_result.model")
    )
    + ", '')"
)
# A headless call without pricing carries an api_costs that is not an amount;
# core telemetry leaves it out and counts it as pricing-unknown (dbase.py).
_UNPRICED = (
    f"({_MODEL} GLOB 'headless/*' "
    "AND COALESCE(json_extract(metadata, '$.headless_pricing_unknown'), 0) = 1)"
)
_PROGRAMS_COST = _cost_sql(
    "programs",
    f"(CASE WHEN {_UNPRICED} THEN 0 ELSE {_FIELDS['api_costs'][0]} END) + "
    + " + ".join(_FIELDS[f][0] for f in COST_FIELDS[1:]),
    f"(NOT {_UNPRICED} AND " + " AND ".join(r for _, r in _FIELDS.values()) + ")",
    f"NOT ({_NOT_COPY})",
)
_PROMPTS_COST = _cost_sql("system_prompts", *_cost_field("llm.cost"), "0")


def run_cost(results_dir: Path, db_name: str = DB_FILENAME) -> RunCost:
    """LLM spend of a run: its evaluated (non-copy) programs plus prompt
    evolution (``prompts.sqlite``). The budget, the workspace and the run
    statistics all use this one rule, so a run never shows two costs."""
    results_dir = Path(results_dir)
    programs, programs_known = 0.0, 1
    if (results_dir / db_name).exists():
        ((programs, programs_known),) = _query(
            results_dir, _PROGRAMS_COST, filename=db_name
        )
    prompts, prompts_known = 0.0, 1
    if (results_dir / PROMPT_DB_FILENAME).exists():
        ((prompts, prompts_known),) = _query(
            results_dir, _PROMPTS_COST, filename=PROMPT_DB_FILENAME
        )
    return RunCost(
        float(programs), bool(programs_known), float(prompts), bool(prompts_known)
    )


def best_score(results_dir: Path) -> float | None:
    if not (Path(results_dir) / DB_FILENAME).exists():
        return None
    ((score,),) = _query(
        results_dir,
        f"SELECT MAX(combined_score) FROM programs WHERE correct = 1 AND {_NOT_COPY}",
    )
    return None if score is None else float(score)
