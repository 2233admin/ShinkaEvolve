"""The run-cost rule (``shinka.database.cost``) on databases built by hand, and
the places that act on an unknown total: ``run --grant`` refuses, the overview
lists the experiment as stuck, and the runner marks or stops on a charge it
could not record.

Every amount is a power of two, so sums compare exactly and a difference names
the charges it is made of. Databases get the production schema
(``ProgramDatabase``, ``SystemPromptDatabase``); rows are inserted raw so a case
can hold what no writer produces (malformed JSON, text amounts).
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

import shinka.core.async_runner as async_runner
from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.core.async_runner import ChargeNotRecordedError
from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.database.cost import (
    UNRECORDED,
    mark_unrecorded,
    program_spend,
    prompt_spend,
    recorded_in_full,
)
from shinka.database.prompt_dbase import SystemPromptConfig, SystemPromptDatabase
from shinka.launch import LocalJobConfig
from shinka.pricing.catalog import get_catalog
from shinka.research import cli, grants, rounds
from shinka.research.grants import AGENT_ENV
from shinka.research.notebook import Notebook, experiment_dir
from shinka.research.rundb import run_cost
from shinka.research.state import work_entries

A, B, C, D, E, F = (2.0**-k for k in range(1, 7))  # 0.5, 0.25, ... 0.015625


def _programs_db(
    results: Path, programs: list, events: list | None, *, attempt_log: bool = True
) -> Path:
    """``programs.sqlite`` with ``programs`` rows (metadata dict, raw string or
    None) and ``attempt_log`` rows ``(generation, status, details)`` (details a
    dict, raw string or None). ``attempt_log=False`` builds a database from
    before that table, by hand."""
    results.mkdir(parents=True, exist_ok=True)
    path = results / "programs.sqlite"
    if attempt_log:
        db = ProgramDatabase(DatabaseConfig(db_path=str(path)), embedding_model="")
        db.close()
    with closing(sqlite3.connect(path)) as conn:
        if not attempt_log:
            conn.execute(
                "CREATE TABLE programs (id TEXT PRIMARY KEY, code TEXT NOT NULL, "
                "language TEXT NOT NULL, generation INTEGER NOT NULL, "
                "timestamp REAL NOT NULL, correct BOOLEAN, combined_score REAL, "
                "metadata TEXT)"
            )
        conn.executemany(
            "INSERT INTO programs (id, code, language, generation, timestamp, "
            "correct, combined_score, metadata) VALUES (?, '', 'python', ?, 0.0, 1, "
            "0.0, ?)",
            [
                (
                    f"p{i}",
                    i,
                    md if md is None or isinstance(md, str) else json.dumps(md),
                )
                for i, md in enumerate(programs)
            ],
        )
        if attempt_log:
            assert events is not None
            conn.executemany(
                "INSERT INTO attempt_log (generation, stage, status, details, "
                "created_at) VALUES (?, 'test', ?, ?, 0.0)",
                [
                    (
                        g,
                        status,
                        d if d is None or isinstance(d, str) else json.dumps(d),
                    )
                    for g, status, d in events
                ],
            )
        conn.commit()
    return path


def _prompts_db(results: Path, metadata: list) -> Path:
    results.mkdir(parents=True, exist_ok=True)
    path = results / "prompts.sqlite"
    db = SystemPromptDatabase(SystemPromptConfig(db_path=str(path)))
    db.close()
    with closing(sqlite3.connect(path)) as conn:
        conn.executemany(
            "INSERT INTO system_prompts (id, prompt_text, generation, patch_type, "
            "timestamp, metadata) VALUES (?, 'x', ?, 'full', 0.0, ?)",
            [
                (
                    f"s{i}",
                    i,
                    md if md is None or isinstance(md, str) else json.dumps(md),
                )
                for i, md in enumerate(metadata)
            ],
        )
        conn.commit()
    return path


def _failed(kind: str, **costs) -> dict:
    return {"node_kind": kind, **costs}


# (programs, events, usd, known). A charge event is (generation, status, details).
PROGRAM_CASES = {
    "four fields of a row are summed": (
        [{"api_costs": A, "embed_cost": B, "novelty_cost": C, "meta_cost": D}],
        [],
        A + B + C + D,
        True,
    ),
    "null metadata and null or absent fields are recorded zeros": (
        [None, {}, {"api_costs": None, "meta_cost": None}, {"api_costs": B}],
        [],
        B,
        True,
    ),
    "island copies are not charges": (
        [
            {"api_costs": A, "meta_cost": E},
            {"api_costs": A, "meta_cost": E, "_is_island_copy": True},
            {"api_costs": A, "embed_cost": B, "_spawned_island": True},
        ],
        [],
        A + E,
        True,
    ),
    "one failed proposal counts": (
        [{"api_costs": A}],
        [(3, "failed", _failed("failed_proposal", api_costs=B))],
        A + B,
        True,
    ),
    "the same generation failing twice counts twice": (
        [{"api_costs": A}],
        [
            (3, "failed", _failed("failed_proposal", api_costs=B, embed_cost=F)),
            (3, "failed", _failed("failed_proposal", api_costs=B, novelty_cost=E)),
        ],
        A + 2 * B + F + E,
        True,
    ),
    "a discarded proposal and a failed prompt evolution count": (
        [],
        [
            (
                5,
                "discarded",
                _failed(
                    "discarded_proposal", api_costs=C, embed_cost=D, novelty_cost=E
                ),
            ),
            (2, "failed", _failed("failed_prompt_evolution", api_costs=F)),
        ],
        C + D + E + F,
        True,
    ),
    "an explicit non-charge node_kind is skipped": (
        [{"api_costs": A}],
        [(1, "failed", _failed("evaluation", api_costs=B))],
        A,
        True,
    ),
    # A charge-status row that does not name its kind might be a charge.
    "a failed row with SQL NULL details leaves the total unknown": (
        [{"api_costs": A}],
        [(1, "failed", None)],
        A,
        False,
    ),
    "a failed row with JSON null details leaves the total unknown": (
        [{"api_costs": A}],
        [(1, "failed", "null")],
        A,
        False,
    ),
    "a failed row with empty details leaves the total unknown": (
        [{"api_costs": A}],
        [(1, "failed", {})],
        A,
        False,
    ),
    "a failed row without a node_kind leaves the total unknown": (
        [{"api_costs": A}],
        [(1, "failed", {"api_costs": C})],
        A,
        False,
    ),
    "a discarded row with a non-text node_kind leaves the total unknown": (
        [{"api_costs": A}],
        [(1, "discarded", {"node_kind": 3, "api_costs": C})],
        A,
        False,
    ),
    "a non-charge status is skipped": (
        [{"api_costs": A}],
        [
            (1, "started", _failed("failed_proposal", api_costs=B)),
            (1, "succeeded", _failed("discarded_proposal", api_costs=C)),
        ],
        A,
        True,
    ),
    "malformed details of a non-charge status are not read": (
        [{"api_costs": A}],
        [(1, "started", "{not json")],
        A,
        True,
    ),
    "a failed row with malformed details leaves the total unknown": (
        [{"api_costs": A}],
        [
            (1, "failed", _failed("failed_proposal", api_costs=B)),
            (2, "failed", "{not json"),
        ],
        A + B,
        False,
    ),
    "a discarded row with malformed details leaves the total unknown": (
        [],
        [(2, "discarded", "[1, 2")],
        0.0,
        False,
    ),
    "a text amount in a charge event is unknown": (
        [{"api_costs": A}],
        [(1, "failed", _failed("failed_proposal", api_costs="0.25"))],
        A,
        False,
    ),
    "a negative amount in a charge event is unknown": (
        [],
        [(1, "failed", _failed("failed_proposal", api_costs=-B))],
        0.0,
        False,
    ),
    "a text meta_cost stays unknown": (
        [{"api_costs": A, "meta_cost": "0.5"}, {"api_costs": B}],
        [],
        A + B,
        False,
    ),
    "malformed program metadata is unknown": (
        [{"api_costs": A}, "{oops"],
        [],
        A,
        False,
    ),
    "a headless call without pricing is unknown": (
        [
            {"api_costs": A},
            {
                "model_name": "headless/codex",
                "headless_pricing_unknown": True,
                "api_costs": 0.0,
            },
        ],
        [],
        A,
        False,
    ),
}


@pytest.mark.parametrize(
    "programs, events, usd, known",
    list(PROGRAM_CASES.values()),
    ids=list(PROGRAM_CASES),
)
def test_program_spend_and_run_cost_follow_the_rule(
    tmp_path, programs, events, usd, known
):
    path = _programs_db(tmp_path, programs, events)
    with closing(sqlite3.connect(path)) as conn:
        spend = program_spend(conn)
    assert math.isclose(spend.usd, usd, rel_tol=0, abs_tol=1e-12), spend
    assert spend.known is known, spend

    cost = run_cost(tmp_path)
    assert math.isclose(cost.recorded_usd, usd, rel_tol=0, abs_tol=1e-12), cost
    assert cost.programs_known is known and cost.known is known, cost
    assert cost.prompts_usd == 0.0 and cost.prompts_known, cost


def test_a_database_from_before_attempt_log_reads_known(tmp_path):
    _programs_db(
        tmp_path,
        [{"api_costs": A, "embed_cost": B}, {"meta_cost": C}, None],
        None,
        attempt_log=False,
    )
    with closing(sqlite3.connect(tmp_path / "programs.sqlite")) as conn:
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'attempt_log'"
        ).fetchone()
        spend = program_spend(conn)
    assert spend.known and math.isclose(spend.usd, A + B + C, rel_tol=0, abs_tol=1e-12)
    cost = run_cost(tmp_path)
    assert cost.known and math.isclose(
        cost.recorded_usd, A + B + C, rel_tol=0, abs_tol=1e-12
    )


PROMPT_CASES = {
    "prompt costs are summed": (
        [{"llm": {"cost": B}}, {"llm": {"cost": D}}],
        B + D,
        True,
    ),
    "a prompt without a cost is a recorded zero": ([None, {}, {"llm": {}}], 0.0, True),
    "a text prompt cost is unknown": (
        [{"llm": {"cost": B}}, {"llm": {"cost": "x"}}],
        B,
        False,
    ),
    "malformed prompt metadata is unknown": ([{"llm": {"cost": B}}, "{bad"], B, False),
}


@pytest.mark.parametrize(
    "metadata, usd, known", list(PROMPT_CASES.values()), ids=list(PROMPT_CASES)
)
def test_prompt_spend_is_part_of_run_cost(tmp_path, metadata, usd, known):
    _programs_db(tmp_path, [{"api_costs": A}], [])
    path = _prompts_db(tmp_path, metadata)
    with closing(sqlite3.connect(path)) as conn:
        spend = prompt_spend(conn)
    assert spend.known is known
    assert math.isclose(spend.usd, usd, rel_tol=0, abs_tol=1e-12), spend

    cost = run_cost(tmp_path)
    assert cost.programs_known and cost.prompts_known is known and cost.known is known
    assert math.isclose(cost.recorded_usd, A + usd, rel_tol=0, abs_tol=1e-12), cost


def test_the_unrecorded_marker_makes_run_cost_unknown_until_removed(tmp_path):
    _programs_db(
        tmp_path,
        [{"api_costs": A}],
        [(1, "failed", _failed("failed_proposal", api_costs=B))],
    )
    assert run_cost(tmp_path).known and recorded_in_full(tmp_path)

    mark_unrecorded(tmp_path, {"generation": 2, "api_costs": C})
    marker = tmp_path / UNRECORDED
    assert json.loads(marker.read_text(encoding="utf-8"))["api_costs"] == C
    cost = run_cost(tmp_path)
    assert not recorded_in_full(tmp_path)
    assert not cost.known and not cost.programs_known, cost
    # The marker is a signal, not a ledger: its charge is not added.
    assert math.isclose(cost.recorded_usd, A + B, rel_tol=0, abs_tol=1e-12), cost

    marker.unlink()
    cost = run_cost(tmp_path)
    assert cost.known and recorded_in_full(tmp_path), cost
    assert math.isclose(cost.recorded_usd, A + B, rel_tol=0, abs_tol=1e-12), cost


def test_the_marker_alone_makes_an_empty_results_dir_unknown(tmp_path):
    mark_unrecorded(tmp_path, {"generation": 0, "api_costs": A})
    cost = run_cost(tmp_path)
    assert not cost.known and cost.recorded_usd == 0.0, cost


# --- run --grant and the overview, on an experiment whose cost is unknown ---


@pytest.fixture
def direction(tmp_path: Path, example_dir: Path, bars_csv: Path, monkeypatch) -> Path:
    root = tmp_path / "direction"
    cli.main(["init", str(root), "--direction", "Unknown cost fixture"])
    cli.main(
        [
            "hypothesis", "add", str(root), "--title", "cost",
            "--file", str(example_dir / "hypothesis.md"),
        ]
    )  # fmt: skip
    cli.main(
        [
            "scaffold", str(root), "h001",
            "--initial", str(example_dir / "initial.py"),
            "--data", str(bars_csv),
        ]
    )  # fmt: skip
    # Never reached: a run that got this far would fail loudly.
    monkeypatch.setattr(
        cli, "SHINKA_RUN", [sys.executable, "-c", "raise SystemExit('launched')"]
    )
    return root.resolve()


def _grant(root: Path) -> str:
    cli.main(["grant", str(root), "h001", "--attempts", "2", "--max-cost", "0.5"])
    return [r for r in grants.load(root) if r["kind"] == "grant"][-1]["id"]


def _unknown_by_marker(results: Path) -> None:
    _programs_db(results, [{"api_costs": A}], [])
    mark_unrecorded(results, {"generation": 1, "api_costs": B})


def _unknown_by_row(results: Path) -> None:
    _programs_db(results, [{"api_costs": A}, {"api_costs": "n/a"}], [])


def _unknown_by_event(results: Path) -> None:
    _programs_db(results, [{"api_costs": A}], [(1, "failed", "{truncated")])


UNKNOWN = {
    "marker": _unknown_by_marker,
    "unknown-cost row": _unknown_by_row,
    "malformed charge event": _unknown_by_event,
}


@pytest.mark.parametrize("make_unknown", list(UNKNOWN.values()), ids=list(UNKNOWN))
def test_a_grant_run_refuses_an_experiment_whose_cost_is_unknown(
    direction: Path, monkeypatch, make_unknown
):
    assert Notebook.load(direction).budget_usd is None  # no budget to check
    grant = _grant(direction)
    results = experiment_dir(direction, "h001", "e01") / "results"
    make_unknown(results)
    assert not run_cost(results).known

    monkeypatch.setenv(AGENT_ENV, "1")
    with pytest.raises(SystemExit, match="unknown amount") as refused:
        cli.main(["run", str(direction), "h001", "--grant", grant])
    assert refused.value.code not in (None, 0)
    records = grants.load(direction)
    assert [r for r in records if r["kind"] == "attempt"] == [], records
    assert work_entries(direction) == []


@pytest.mark.parametrize("make_unknown", list(UNKNOWN.values()), ids=list(UNKNOWN))
def test_the_overview_lists_an_open_grants_unknown_cost_as_stuck(
    direction: Path, make_unknown
):
    grant = _grant(direction)
    results = experiment_dir(direction, "h001", "e01") / "results"
    make_unknown(results)

    stuck = [i for i in rounds.overview(direction)["pending"] if i["kind"] == "stuck"]
    assert {
        "kind": "stuck",
        "reason": "cost-unknown",
        "grant": grant,
        "hypothesis": "h001",
        "experiment": "e01",
    }.items() <= next(
        (i for i in stuck if i.get("reason") == "cost-unknown"), {}
    ).items(), stuck


def test_the_overview_drops_cost_unknown_once_the_cost_is_known(direction: Path):
    _grant(direction)
    results = experiment_dir(direction, "h001", "e01") / "results"
    _unknown_by_marker(results)
    reasons = lambda: [  # noqa: E731
        i.get("reason") for i in rounds.overview(direction)["pending"]
    ]
    assert "cost-unknown" in reasons()
    (results / UNRECORDED).unlink()
    assert "cost-unknown" not in reasons()


# --- the runner's required recorder for a charge with no program row ---


@pytest.fixture
def runner(tmp_path: Path, monkeypatch) -> ShinkaEvolveRunner:
    """A real runner, never started; its database is replaced per test."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-offline")
    monkeypatch.setattr(async_runner, "refresh_model_catalog", get_catalog)
    task = tmp_path / "task"
    task.mkdir()
    (task / "initial.py").write_text("def f():\n    return 0\n", encoding="utf-8")
    (task / "evaluate.py").write_text("", encoding="utf-8")
    return ShinkaEvolveRunner(
        evo_config=EvolutionConfig(
            num_generations=2,
            llm_models=["gpt-5-mini"],
            llm_dynamic_selection=None,
            meta_llm_models=None,
            embedding_model=None,
            novelty_llm_models=None,
            init_program_path=str(task / "initial.py"),
            results_dir=str(tmp_path / "results"),
        ),
        job_config=LocalJobConfig(
            eval_program_path=str(task / "evaluate.py"),
            python_executable=sys.executable,
        ),
        db_config=DatabaseConfig(num_islands=1),
        verbose=False,
    )


class RefusingDB:
    def __init__(self):
        self.calls = 0

    async def record_attempt_event_async(self, **kwargs):
        self.calls += 1
        raise sqlite3.OperationalError("database is locked")


class RecordingDB:
    def __init__(self):
        self.events = []

    async def record_attempt_event_async(self, **kwargs):
        self.events.append(kwargs)


CHARGE = dict(
    generation=3,
    stage="proposal",
    status="failed",
    details={"node_kind": "failed_proposal", "api_costs": B, "embed_cost": E},
)


def test_a_recorded_charge_leaves_no_marker(runner, tmp_path):
    db = RecordingDB()
    runner.async_db = db
    asyncio.run(runner._record_charge_event(**CHARGE))
    assert db.events == [CHARGE]
    assert not runner.cost_limit_reached and runner._fatal_error is None
    assert recorded_in_full(Path(runner.results_dir))


def test_a_refused_charge_writes_the_marker_and_stops_proposals(runner):
    db = RefusingDB()
    runner.async_db = db
    asyncio.run(runner._record_charge_event(**CHARGE))  # no raise
    assert db.calls == 1  # one write, no retry
    results = Path(runner.results_dir)
    marker = json.loads((results / UNRECORDED).read_text(encoding="utf-8"))
    assert marker["node_kind"] == "failed_proposal" and marker["generation"] == 3
    assert marker["api_costs"] == B and marker["embed_cost"] == E
    assert runner.cost_limit_reached
    assert runner._fatal_error is None
    assert not runner.finalization_complete.is_set()
    assert not run_cost(results).known


def test_a_charge_neither_recorded_nor_marked_is_fatal(runner, tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a file where the results directory's parent should be")
    runner.results_dir = str(blocker / "results")
    runner.async_db = RefusingDB()
    with pytest.raises(ChargeNotRecordedError):
        asyncio.run(runner._record_charge_event(**CHARGE))
    assert isinstance(runner._fatal_error, ChargeNotRecordedError)
    assert runner.finalization_complete.is_set()
    assert runner.should_stop.is_set()
    assert runner.cost_limit_reached


# A charge of nothing is every field absent, None or a finite non-bool 0; any
# other value is a charge, so a refused write must leave the marker.
NOT_NOTHING = {
    "false": False,
    "true": True,
    "nan": float("nan"),
    "inf": float("inf"),
    "string": "x",
    "negative": -B,
}


@pytest.mark.parametrize("api_costs", list(NOT_NOTHING.values()), ids=list(NOT_NOTHING))
def test_a_refused_charge_that_is_not_zero_writes_the_marker(runner, api_costs):
    db = RefusingDB()
    runner.async_db = db
    details = {
        "node_kind": "failed_proposal",
        "api_costs": api_costs,
        "embed_cost": None,
        "novelty_cost": None,
    }
    asyncio.run(
        runner._record_charge_event(
            generation=4, stage="proposal", status="failed", details=details
        )
    )
    assert db.calls == 1
    results = Path(runner.results_dir)
    marker = json.loads((results / UNRECORDED).read_text(encoding="utf-8"))
    assert marker["generation"] == 4 and marker["node_kind"] == "failed_proposal"
    if isinstance(api_costs, float) and math.isnan(api_costs):
        assert math.isnan(marker["api_costs"])
    else:
        assert marker["api_costs"] == api_costs
        assert type(marker["api_costs"]) is type(api_costs)
    assert runner.cost_limit_reached
    assert runner._fatal_error is None
    assert not recorded_in_full(results)
    assert not run_cost(results).known


NOTHING = {
    "int-zero": dict(api_costs=0, embed_cost=0, novelty_cost=0),
    "float-zero": dict(api_costs=0.0, embed_cost=0.0, novelty_cost=0.0),
    "negative-zero": dict(api_costs=-0.0, embed_cost=-0.0, novelty_cost=-0.0),
    "none": dict(api_costs=None, embed_cost=None, novelty_cost=None),
    "absent": dict(),
}


@pytest.mark.parametrize("costs", list(NOTHING.values()), ids=list(NOTHING))
def test_a_refused_charge_of_nothing_leaves_no_marker(runner, costs):
    db = RefusingDB()
    runner.async_db = db
    asyncio.run(
        runner._record_charge_event(
            generation=4,
            stage="proposal",
            status="failed",
            details={"node_kind": "failed_proposal", **costs},
        )
    )
    assert db.calls == 1  # still written, best effort
    results = Path(runner.results_dir)
    assert not (results / UNRECORDED).exists()
    assert recorded_in_full(results)
    assert not runner.cost_limit_reached and runner._fatal_error is None
