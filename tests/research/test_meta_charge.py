"""A paid meta update whose cost the runner cannot store leaves the run's
total unknown, and one whose output file fails still stores its cost.

Each test builds a real run directory (``ShinkaEvolveRunner._setup_async``: a
real ``programs.sqlite`` holding generation 0, evaluated by the real local
scheduler). Only the meta summarizer is a stub, reporting a paid update, and
the one database or file write under test fails. The cost rule
(``shinka.research.rundb.run_cost``) then reads the run directory as any
reader would.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sqlite3
import sys
import time
import uuid
from contextlib import closing
from pathlib import Path

import pytest

import shinka.core.async_runner as async_runner
from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.core.async_runner import AsyncRunningJob, PersistedProgramEvent
from shinka.database import DatabaseConfig, Program
from shinka.database.async_dbase import AsyncProgramDatabase
from shinka.database.cost import UNRECORDED, recorded_in_full
from shinka.launch import LocalJobConfig
from shinka.pricing.catalog import get_catalog
from shinka.research.rundb import run_cost

MODEL = "gpt-5-mini"
META_COST = 2.0**-4

INITIAL = """\
# EVOLVE-BLOCK-START
def score():
    return 0.0
# EVOLVE-BLOCK-END
"""

EVALUATE = """\
import argparse, importlib.util, json, os

p = argparse.ArgumentParser()
p.add_argument("--program_path")
p.add_argument("--results_dir")
a = p.parse_args()
spec = importlib.util.spec_from_file_location("prog", a.program_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = float(m.score())
os.makedirs(a.results_dir, exist_ok=True)
with open(os.path.join(a.results_dir, "correct.json"), "w") as f:
    json.dump({"correct": True, "error": None}, f)
with open(os.path.join(a.results_dir, "metrics.json"), "w") as f:
    json.dump({"combined_score": s, "public": {"score": s}, "private": {}}, f)
"""


class MetaOutputFailed(Exception):
    """The meta output file could not be written."""


class MetadataWriteFailed(Exception):
    """The database did not take a program's metadata."""


class StubMeta:
    """Stands in for the async meta summarizer: when ``fire`` is set, the next
    update is due and pays ``META_COST``; its output file fails when
    ``fail_output`` is set."""

    def __init__(self, fire: bool, fail_output: bool = False):
        self.fire = fire
        self.fail_output = fail_output
        self.evaluated_since_last_meta: list[Program] = []
        self.updates = 0
        self.output_writes = 0

    def add_evaluated_program(self, program: Program) -> None:
        self.evaluated_since_last_meta.append(program)

    def should_update_meta(self, meta_rec_interval) -> bool:
        return self.fire

    async def update_meta_memory_async(self, best_program=None):
        self.updates += 1
        self.evaluated_since_last_meta = []
        return "recommendations", META_COST

    async def write_meta_output_async(self, results_dir: str) -> None:
        self.output_writes += 1
        if self.fail_output:
            raise MetaOutputFailed("meta output disk full")


@pytest.fixture
def offline(monkeypatch):
    """No pricing refresh, a dummy key, loopback-only sockets."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-offline")
    monkeypatch.setattr(async_runner, "refresh_model_catalog", get_catalog)
    real_connect = socket.socket.connect

    def loopback_only(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise OSError(f"network blocked in test: {address!r}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)


def fail_metadata_writes_carrying_meta_cost(monkeypatch) -> list[str]:
    """Make the database refuse every metadata write that carries a
    ``meta_cost``; every other write goes through. Returns the refused ids."""
    real = AsyncProgramDatabase.update_program_metadata_async
    refused: list[str] = []

    async def update_program_metadata_async(self, program_id, metadata):
        if "meta_cost" in metadata:
            refused.append(program_id)
            raise MetadataWriteFailed("database is locked")
        await real(self, program_id, metadata)

    monkeypatch.setattr(
        AsyncProgramDatabase,
        "update_program_metadata_async",
        update_program_metadata_async,
    )
    return refused


def _runner(root: Path, meta: StubMeta) -> ShinkaEvolveRunner:
    task = root / "task"
    task.mkdir(parents=True, exist_ok=True)
    (task / "initial.py").write_text(INITIAL, encoding="utf-8")
    (task / "evaluate.py").write_text(EVALUATE, encoding="utf-8")
    runner = ShinkaEvolveRunner(
        evo_config=EvolutionConfig(
            patch_types=["full"],
            patch_type_probs=[1.0],
            num_generations=3,
            max_patch_attempts=1,
            max_patch_resamples=1,
            max_novelty_attempts=1,
            llm_models=[MODEL],
            llm_dynamic_selection=None,
            meta_llm_models=None,
            meta_rec_interval=1,
            embedding_model=None,
            novelty_llm_models=None,
            evolve_prompts=False,
            init_program_path=str(task / "initial.py"),
            results_dir=str(root / "results"),
            max_api_costs=None,
        ),
        job_config=LocalJobConfig(
            eval_program_path=str(task / "evaluate.py"),
            python_executable=sys.executable,
        ),
        db_config=DatabaseConfig(num_islands=1),
        verbose=False,
        max_evaluation_jobs=1,
        max_proposal_jobs=1,
        max_db_workers=1,
    )
    runner.meta_summarizer = meta
    return runner


async def _set_up(runner: ShinkaEvolveRunner) -> None:
    """The run's real setup: database, generation 0 evaluated for real."""
    async_runner.activate_model_catalog(runner.pricing_snapshot)
    await runner._setup_async()


async def _tear_down(runner: ShinkaEvolveRunner) -> None:
    if runner.async_db is not None:
        await runner.async_db.close_async()
    runner.scheduler.shutdown()


def _results(runner: ShinkaEvolveRunner) -> Path:
    return Path(runner.results_dir)


def _metadata(runner: ShinkaEvolveRunner, generation: int) -> dict:
    """The stored metadata of the run's one program of ``generation``."""
    with closing(sqlite3.connect(_results(runner) / "programs.sqlite")) as conn:
        rows = conn.execute(
            "SELECT metadata FROM programs WHERE generation = ?", (generation,)
        ).fetchall()
    assert len(rows) == 1, rows
    return json.loads(rows[0][0])


def _marker(runner: ShinkaEvolveRunner) -> dict:
    return json.loads((_results(runner) / UNRECORDED).read_text(encoding="utf-8"))


def _assert_marked(runner: ShinkaEvolveRunner, generation: int, program_id: str):
    """The paid meta update is marked unrecorded: the run's total reads
    unknown and no more proposals start."""
    results = _results(runner)
    assert not recorded_in_full(results)
    assert not run_cost(results).known
    assert runner.cost_limit_reached is True
    marker = json.dumps(_marker(runner))
    assert program_id in marker, marker
    assert str(META_COST) in marker, marker
    assert "meta_update" in marker, marker
    assert "meta_cost" not in _metadata(runner, generation)


def test_a_periodic_meta_cost_whose_metadata_write_fails_marks_the_run(
    tmp_path, offline, monkeypatch
):
    """A completed program's side effects run a paid meta update, then the
    database refuses the metadata that carries its cost. The side effects
    still finish, the cost is in the runner's total, and the run is marked
    unrecorded."""
    meta = StubMeta(fire=False)
    runner = _runner(tmp_path, meta)

    async def run():
        await _set_up(runner)
        try:
            assert meta.updates == 0
            assert recorded_in_full(_results(runner))
            seed = await runner.async_db.get_best_program_async()
            program = Program(
                id=str(uuid.uuid4()),
                code=INITIAL,
                parent_id=seed.id,
                generation=1,
                correct=True,
                combined_score=1.0,
                public_metrics={"score": 1.0},
                metadata={"patch_name": "child"},
            )
            await runner.async_db.add_program_async(program)
            assert "meta_cost" not in _metadata(runner, 1)
            before = runner.total_api_cost
            refused = fail_metadata_writes_carrying_meta_cost(monkeypatch)
            meta.fire = True
            now = time.time()
            job = AsyncRunningJob(
                job_id="job-1",
                exec_fname=f"{runner.results_dir}/gen_1/main.py",
                results_dir=f"{runner.results_dir}/gen_1/results",
                start_time=now,
                proposal_started_at=now,
                evaluation_submitted_at=now,
                evaluation_started_at=now,
                generation=1,
                parent_id=seed.id,
            )
            await runner._apply_persisted_program_side_effects(
                PersistedProgramEvent(
                    job=job,
                    program=program,
                    evaluation_finished_at=now,
                    postprocess_started_at=now,
                    postprocess_finished_at=now,
                )
            )
            assert meta.updates == 1
            assert refused == [program.id]
            assert runner.total_api_cost == before + META_COST
            assert runner._fatal_error is None
            _assert_marked(runner, 1, program.id)
        finally:
            await _tear_down(runner)

    asyncio.run(run())


def test_an_initial_meta_cost_is_stored_when_its_output_file_fails(tmp_path, offline):
    """The initial program's paid meta update succeeds but its output file
    cannot be written. Setup goes on to store the cost with the program, so
    the run's total holds it and stays complete."""
    meta = StubMeta(fire=True, fail_output=True)
    runner = _runner(tmp_path, meta)

    async def run():
        await _set_up(runner)
        try:
            assert meta.updates == 1
            assert meta.output_writes == 1
            stored = _metadata(runner, 0)
            assert stored["meta_cost"] == META_COST
            other = sum(
                float(stored.get(field) or 0.0)
                for field in ("api_costs", "embed_cost", "novelty_cost")
            )
            results = _results(runner)
            assert recorded_in_full(results)
            cost = run_cost(results)
            assert cost.known
            assert cost.recorded_usd == pytest.approx(other + META_COST)
            assert runner.total_api_cost == pytest.approx(other + META_COST)
            assert runner.cost_limit_reached is False
            assert runner.completed_generations == 1
        finally:
            await _tear_down(runner)

    asyncio.run(run())


def test_an_initial_meta_cost_whose_metadata_write_fails_marks_the_run(
    tmp_path, offline, monkeypatch
):
    """The initial program's paid meta update succeeds, then the database
    refuses the metadata that carries its cost. Setup fails, and the run is
    marked unrecorded first."""
    meta = StubMeta(fire=True)
    runner = _runner(tmp_path, meta)
    refused = fail_metadata_writes_carrying_meta_cost(monkeypatch)

    async def run():
        try:
            with pytest.raises(MetadataWriteFailed, match="database is locked"):
                await _set_up(runner)
            assert meta.updates == 1
            assert len(refused) == 1
            assert runner.total_api_cost >= META_COST
            assert runner._fatal_error is None
            _assert_marked(runner, 0, refused[0])
        finally:
            await _tear_down(runner)

    asyncio.run(run())
