"""What a run has spent, by one rule.

The runner's resume base (``ShinkaEvolveRunner._get_total_api_costs``) and the
readers that total a run (``shinka.research.rundb.run_cost``: the research
budget, grants, workspace and run statistics) use these queries, so a resumed
run stops where its budget says and every page shows the same total.

A run is charged for its evaluated programs, the paid work that never became
one, and its prompt evolution:

- A program row charges its ``api_costs``, ``embed_cost``, ``novelty_cost`` and
  ``meta_cost``. Island copies (``_is_island_copy`` from migration,
  ``_spawned_island`` from dynamic spawning) repeat their source's fields and
  are not charges. Meta recommendations charge the program that triggered them,
  or the best one for the end-of-run summary, paid steps of a failed update
  included.
- Paid work without a program row charges its one ``attempt_log`` event
  (``details.node_kind`` in ``CHARGE_KINDS``): a terminal failed proposal, a
  surplus proposal discarded once the run reached its target, and a prompt
  evolution that produced no prompt. Events are never merged by generation: a
  generation that fails again later is a second charge. A database from before
  ``attempt_log`` has none.
- Prompt evolution charges ``metadata.llm.cost`` of each ``system_prompts`` row
  in ``prompts.sqlite``.

An absent or null field is a recorded zero: the runner writes a field only for
the steps that ran. A present value must be a finite number >= 0. Anything
else, malformed JSON (including a charge event's), or a headless call without
pricing leaves the total unknown: ``usd`` is then a lower bound. A charge the
runner could not write to its database leaves a marker next to it
(``UNRECORDED``), which keeps every total of that run unknown until the owner
accounts for the charge and removes the marker; it is a signal, not a second
ledger, so a write that did land is never counted twice. Paid work never
recorded at all (a run killed with proposals in flight) is in no total.
Callers read through their own connection (read-only for readers), so a live
run's database is never locked for writing. Each database's total is one
statement, so it reads one snapshot of that database; a run's two databases
(``programs.sqlite``, ``prompts.sqlite``) are read one after the other.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, NamedTuple, Tuple

PROGRAM_FIELDS = ("api_costs", "embed_cost", "novelty_cost", "meta_cost")
EVENT_FIELDS = ("api_costs", "embed_cost", "novelty_cost")

FAILED_PROPOSAL = "failed_proposal"
DISCARDED_PROPOSAL = "discarded_proposal"
FAILED_PROMPT_EVOLUTION = "failed_prompt_evolution"
CHARGE_KINDS = (FAILED_PROPOSAL, DISCARDED_PROPOSAL, FAILED_PROMPT_EVOLUTION)
# attempt_log statuses a charge event is written with.
CHARGE_STATUSES = ("failed", "discarded")
UNRECORDED = "cost-unrecorded.json"

# Administrative copies are neither trials nor charges; the predicate of the
# evaluated counts in dbase's population telemetry.
NOT_COPY = (
    "COALESCE(json_extract(metadata, '$._is_island_copy'), 0) = 0 "
    "AND COALESCE(json_extract(metadata, '$._spawned_island'), 0) = 0"
)


class Spend(NamedTuple):
    """Recorded spend in USD; when ``known`` is false some charge has no valid
    amount and ``usd`` is only a lower bound."""

    usd: float
    known: bool


def _field(column: str, path: str) -> Tuple[str, str]:
    """(amount, recorded) SQL for one cost field of a valid JSON ``column``."""
    kind = f"json_type({column}, '$.{path}')"
    value = f"json_extract({column}, '$.{path}')"
    number = f"({kind} IN ('integer', 'real') AND {value} >= 0 AND {value} < 1e999)"
    recorded = f"({kind} IS NULL OR {kind} = 'null' OR {number})"
    return f"(CASE WHEN {number} THEN {value} ELSE 0 END)", recorded


def _total(
    table: str,
    column: str,
    amount: str,
    recorded: str,
    skip: str,
    where: str,
    null_recorded: bool = True,
) -> str:
    """Total (``usd``) of ``amount`` over ``table`` rows matching ``where``,
    and whether every row is recorded (``known``). A null ``column`` is a
    recorded zero unless ``null_recorded`` is false, malformed JSON is
    unknown, and rows matching ``skip`` are not charges. CASE keeps the JSON
    functions off malformed rows, where they would raise."""
    null = 1 if null_recorded else 0
    return (
        "SELECT COALESCE(SUM(amount), 0) AS usd, COALESCE(MIN(recorded), 1) AS known "
        f"FROM (SELECT CASE WHEN {column} IS NULL THEN 0 WHEN json_valid({column}) "
        f"THEN CASE WHEN {skip} THEN 0 ELSE {amount} END ELSE 0 END AS amount, "
        f"CASE WHEN {column} IS NULL THEN {null} WHEN json_valid({column}) "
        f"THEN CASE WHEN {skip} OR {recorded} THEN 1 ELSE 0 END ELSE 0 END "
        f"AS recorded FROM {table} WHERE {where})"
    )


_PROGRAM = {field: _field("metadata", field) for field in PROGRAM_FIELDS}
_MODEL = (
    "COALESCE("
    + ", ".join(
        f"NULLIF(json_extract(metadata, '$.{p}'), '')"
        for p in ("model_name", "model", "llm_result.model_name", "llm_result.model")
    )
    + ", '')"
)
# A headless call without pricing carries an api_costs that is not an amount;
# the population telemetry leaves it out and counts it as pricing-unknown.
_UNPRICED = (
    f"({_MODEL} GLOB 'headless/*' "
    "AND COALESCE(json_extract(metadata, '$.headless_pricing_unknown'), 0) = 1)"
)
_PROGRAMS = _total(
    "programs",
    "metadata",
    f"(CASE WHEN {_UNPRICED} THEN 0 ELSE {_PROGRAM['api_costs'][0]} END) + "
    + " + ".join(_PROGRAM[f][0] for f in PROGRAM_FIELDS[1:]),
    f"(NOT {_UNPRICED} AND " + " AND ".join(r for _, r in _PROGRAM.values()) + ")",
    f"NOT ({NOT_COPY})",
    "1",
)


def _quoted(values: Tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


_EVENT = {field: _field("details", field) for field in EVENT_FIELDS}
# Every event with a charge status is read. Only an object naming another kind
# in text is not a charge; an object naming a charge kind counts its amounts;
# anything else (null, malformed, JSON null, {}, a missing or non-text kind)
# might be a charge, so it leaves the total unknown instead of dropping out.
_CHARGE = (
    "(json_type(details) = 'object' AND "
    f"json_extract(details, '$.node_kind') IN ({_quoted(CHARGE_KINDS)}))"
)
_EVENTS = _total(
    "attempt_log",
    "details",
    f"(CASE WHEN {_CHARGE} THEN "
    + " + ".join(amount for amount, _ in _EVENT.values())
    + " ELSE 0 END)",
    f"({_CHARGE} AND "
    + " AND ".join(recorded for _, recorded in _EVENT.values())
    + ")",
    "(json_type(details) = 'object' AND json_type(details, '$.node_kind') = 'text' "
    f"AND json_extract(details, '$.node_kind') NOT IN ({_quoted(CHARGE_KINDS)}))",
    f"status IN ({_quoted(CHARGE_STATUSES)})",
    null_recorded=False,
)
_PROGRAMS_AND_EVENTS = (
    "SELECT p.usd + e.usd, MIN(p.known, e.known) "
    f"FROM ({_PROGRAMS}) AS p, ({_EVENTS}) AS e"
)

_PROMPTS = _total(
    "system_prompts", "metadata", *_field("metadata", "llm.cost"), "0", "1"
)


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def program_spend(conn: sqlite3.Connection) -> Spend:
    """Charges recorded in a run's ``programs.sqlite``: its evaluated programs
    and its charge events."""
    sql = _PROGRAMS_AND_EVENTS if _has_table(conn, "attempt_log") else _PROGRAMS
    usd, known = conn.execute(sql).fetchone()
    return Spend(float(usd), bool(known))


def prompt_spend(conn: sqlite3.Connection) -> Spend:
    """Prompt-evolution charges recorded in a run's ``prompts.sqlite``."""
    usd, known = conn.execute(_PROMPTS).fetchone()
    return Spend(float(usd), bool(known))


def mark_unrecorded(results_dir: Path, charge: Dict[str, Any]) -> None:
    """Record, atomically, that a paid charge of this run did not reach its
    database. Repeated failures keep the latest charge; any marker makes the
    run's total unknown. Raises when the disk takes no write either."""
    target = Path(results_dir) / UNRECORDED
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps({"time": time.time(), **charge}, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def recorded_in_full(results_dir: Path) -> bool:
    """False when a charge of this run is known not to have been recorded."""
    return not (Path(results_dir) / UNRECORDED).exists()
