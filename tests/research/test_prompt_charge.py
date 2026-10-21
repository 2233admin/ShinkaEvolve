"""Prompt evolution's charge reaches the run cost exactly once.

``ShinkaEvolveRunner._maybe_evolve_prompt`` books the cost of an evolution
call, then stores the new prompt; the stored row's ``metadata.llm.cost``
carries the charge. ``SystemPromptDatabase.add`` commits the row before its
archive, best-prompt and metadata upkeep, so ``add`` can raise with the row in
place. The runner then asks the database: a row present needs nothing more, a
row absent gets one ``failed_prompt_evolution`` charge event, and a lookup that
fails leaves the cost rule's marker so the total reads unknown.

A real runner is built and never started; its program and prompt databases are
real SQLite files under ``tmp_path``, the parent sampler and the evolver are
stubs. No network, no model call. Amounts are powers of two, so sums compare
exactly.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

import shinka.core.async_runner as async_runner
from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase
from shinka.database.cost import (
    CHARGE_STATUSES,
    FAILED_PROMPT_EVOLUTION,
    UNRECORDED,
    recorded_in_full,
)
from shinka.database.prompt_dbase import (
    SystemPromptConfig,
    SystemPromptDatabase,
    create_system_prompt,
)
from shinka.launch import LocalJobConfig
from shinka.pricing.catalog import get_catalog
from shinka.research.rundb import run_cost

COST = 2.0**-4  # one prompt-evolution call
GENERATION = 3  # the program generation the evolution happens at


class ParentSampler:
    def __init__(self, parent):
        self.parent = parent

    def sample(self):
        return self.parent


class StubEvolver:
    """Stands in for ``AsyncSystemPromptEvolver``: one paid call, returning
    ``prompt`` (or None for a failed evolution) and its cost."""

    def __init__(self, prompt, cost: float):
        self.prompt = prompt
        self.cost = cost
        self.calls = 0

    async def evolve(self, **kwargs):
        self.calls += 1
        return self.prompt, "full", self.cost


@pytest.fixture
def runner(tmp_path: Path, monkeypatch) -> ShinkaEvolveRunner:
    """A real runner with prompt evolution due on its next call, never
    started; real databases under its results directory."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-offline")
    monkeypatch.setattr(async_runner, "refresh_model_catalog", get_catalog)
    task = tmp_path / "task"
    task.mkdir()
    (task / "initial.py").write_text("def f():\n    return 0\n", encoding="utf-8")
    (task / "evaluate.py").write_text("", encoding="utf-8")
    results = tmp_path / "results"
    runner = ShinkaEvolveRunner(
        evo_config=EvolutionConfig(
            num_generations=5,
            llm_models=["gpt-5-mini"],
            llm_dynamic_selection=None,
            meta_llm_models=None,
            embedding_model=None,
            novelty_llm_models=None,
            evolve_prompts=True,
            prompt_evolution_interval=1,
            init_program_path=str(task / "initial.py"),
            results_dir=str(results),
        ),
        job_config=LocalJobConfig(
            eval_program_path=str(task / "evaluate.py"),
            python_executable=sys.executable,
        ),
        db_config=DatabaseConfig(num_islands=1),
        verbose=False,
    )
    results.mkdir(parents=True, exist_ok=True)
    runner.db = ProgramDatabase(
        DatabaseConfig(db_path=str(results / "programs.sqlite")), embedding_model=""
    )
    runner.prompt_db = SystemPromptDatabase(
        SystemPromptConfig(db_path=str(results / "prompts.sqlite"))
    )
    parent = create_system_prompt("You write programs.")  # costs nothing
    runner.prompt_db.add(parent)
    runner.prompt_sampler_evo = ParentSampler(parent)
    runner.completed_generations = GENERATION
    yield runner
    runner.prompt_db.close()
    runner.db.close()


def _evolved(runner) -> object:
    """The prompt the evolution call returns, its cost on the row."""
    parent = runner.prompt_sampler_evo.parent
    return create_system_prompt(
        "You write faster programs.",
        parent_id=parent.id,
        generation=runner.prompt_db.last_generation + 1,
        program_generation=GENERATION,
        patch_type="full",
        metadata={"llm": {"cost": COST, "model": "gpt-5-mini"}},
    )


def _evolve(runner, evolver: StubEvolver) -> None:
    """One ``_maybe_evolve_prompt`` call on a real async program database,
    built inside the loop that uses it."""

    async def go():
        runner.async_db = AsyncProgramDatabase(runner.db)
        try:
            await runner._maybe_evolve_prompt()
        finally:
            runner.async_db.executor.shutdown(wait=True)
            runner.async_db.write_executor.shutdown(wait=True)

    runner.prompt_evolver = evolver
    asyncio.run(go())
    assert evolver.calls == 1
    assert runner.prompt_api_cost == COST and runner.total_api_cost == COST


def _charge_events(results: Path) -> list[tuple[int, str, str, dict]]:
    """(generation, stage, status, details) of every charge-status attempt
    event."""
    with closing(sqlite3.connect(results / "programs.sqlite")) as conn:
        rows = conn.execute(
            "SELECT generation, stage, status, details FROM attempt_log WHERE "
            f"status IN ({', '.join('?' for _ in CHARGE_STATUSES)})",
            CHARGE_STATUSES,
        ).fetchall()
    return [(int(g), stage, status, json.loads(d)) for g, stage, status, d in rows]


def _prompt_costs(results: Path) -> dict[str, float]:
    """Each prompt row's ``metadata.llm.cost`` (0.0 when absent)."""
    with closing(sqlite3.connect(results / "prompts.sqlite")) as conn:
        rows = conn.execute("SELECT id, metadata FROM system_prompts").fetchall()
    return {
        pid: float(json.loads(md or "{}").get("llm", {}).get("cost", 0.0))
        for pid, md in rows
    }


def _assert_one_charge_event(results: Path) -> None:
    events = _charge_events(results)
    assert len(events) == 1, events
    generation, stage, status, details = events[0]
    assert generation == GENERATION and stage == "prompt_evolution", events
    assert status == "failed", events
    assert details["node_kind"] == FAILED_PROMPT_EVOLUTION, events
    assert details["patch_type"] == "full" and details["api_costs"] == COST, events


@pytest.mark.parametrize(
    "upkeep_error",
    [RuntimeError("archive upkeep failed"), sqlite3.OperationalError("locked")],
    ids=["other-error", "sqlite-error-retried"],
)
def test_a_committed_row_whose_upkeep_fails_carries_the_cost_once(
    runner, monkeypatch, upkeep_error
):
    """``add`` commits the row, then archive upkeep raises. A sqlite error makes
    ``add``'s retry insert the same id again, which raises IntegrityError; the
    committed row still carries the cost, so no event is written."""
    prompt = _evolved(runner)
    upkeep_calls = []
    lookups = []

    def broken_upkeep(*args, **kwargs):
        upkeep_calls.append(args)
        raise upkeep_error

    real_get = runner.prompt_db.get

    def spied_get(prompt_id):
        lookups.append(prompt_id)
        return real_get(prompt_id)

    monkeypatch.setattr(runner.prompt_db, "_update_archive", broken_upkeep)
    monkeypatch.setattr(runner.prompt_db, "get", spied_get)
    _evolve(runner, StubEvolver(prompt, COST))

    # add raised from its upkeep, and the runner asked whether the row exists.
    assert len(upkeep_calls) == 1 and lookups == [prompt.id]
    results = Path(runner.results_dir)
    assert real_get(prompt.id) is not None
    assert _prompt_costs(results)[prompt.id] == COST
    assert _charge_events(results) == []
    assert recorded_in_full(results) and not runner.cost_limit_reached
    cost = run_cost(results)
    assert cost.known and cost.recorded_usd == COST, cost
    assert cost.prompts_usd == COST and cost.programs_usd == 0.0, cost


def test_a_prompt_add_refused_before_commit_gets_one_charge_event(runner, monkeypatch):
    prompt = _evolved(runner)

    def refused(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(runner.prompt_db, "add", refused)
    _evolve(runner, StubEvolver(prompt, COST))

    results = Path(runner.results_dir)
    assert runner.prompt_db.get(prompt.id) is None
    _assert_one_charge_event(results)
    assert recorded_in_full(results) and not runner.cost_limit_reached
    cost = run_cost(results)
    assert cost.known and cost.recorded_usd == COST, cost
    assert cost.programs_usd == COST and cost.prompts_usd == 0.0, cost


def test_a_failed_add_whose_lookup_fails_marks_the_cost_unknown(runner, monkeypatch):
    """Whether the row exists cannot be told: an event could count the cost
    twice and none could drop it, so the run's total reads unknown."""
    prompt = _evolved(runner)

    def refused(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(runner.prompt_db, "add", refused)
    monkeypatch.setattr(runner.prompt_db, "get", refused)
    _evolve(runner, StubEvolver(prompt, COST))

    results = Path(runner.results_dir)
    assert _charge_events(results) == []
    marker = json.loads((results / UNRECORDED).read_text(encoding="utf-8"))
    assert marker["node_kind"] == FAILED_PROMPT_EVOLUTION, marker
    assert marker["generation"] == GENERATION and marker["api_costs"] == COST, marker
    assert runner.cost_limit_reached and runner._fatal_error is None
    cost = run_cost(results)
    assert not cost.known and cost.recorded_usd == 0.0, cost


def test_a_paid_evolution_without_a_prompt_gets_one_charge_event(runner):
    _evolve(runner, StubEvolver(None, COST))

    results = Path(runner.results_dir)
    assert sum(_prompt_costs(results).values()) == 0.0
    _assert_one_charge_event(results)
    assert recorded_in_full(results) and not runner.cost_limit_reached
    cost = run_cost(results)
    assert cost.known and cost.recorded_usd == COST, cost
