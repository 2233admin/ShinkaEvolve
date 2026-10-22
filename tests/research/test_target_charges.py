"""Work the async runner drops at its target is charged exactly once.

Each test builds a real run directory (``ShinkaEvolveRunner._setup_async``:
a real ``programs.sqlite`` holding generation 0, evaluated by the real local
scheduler), puts surplus work in flight, and lets the runner's own job monitor
(``_job_monitor_task``) reach the target and clean up. Only the scheduler the
monitor talks to afterwards and the LLM are fakes. Every charge is read back
through the cost rule (``shinka.database.cost.program_spend``) and counted per
dropped job in ``attempt_log``.

Costs are distinct binary fractions per job and per field, so a total equals
the sum exactly and a missing or doubled field shows in it.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest

import shinka.core.async_runner as async_runner
import shinka.llm.llm as llm_module
from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.core.async_runner import AsyncRunningJob
from shinka.database import DatabaseConfig
from shinka.database.cost import (
    DISCARDED_PROPOSAL,
    UNRECORDED,
    Spend,
    program_spend,
    recorded_in_full,
)
from shinka.launch import LocalJobConfig
from shinka.llm.client import async_client_cache_scope
from shinka.llm.providers import QueryResult
from shinka.pricing.catalog import get_catalog

MODEL = "gpt-5-mini"
MONITOR_TIMEOUT_S = 10.0
PROPOSAL_COST = 2.0**-5
PROPOSAL_DELAY_S = 0.3

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


class StubLLM:
    """Stands in for ``shinka.llm.llm.query_async``: every call is a proposal
    that pays ``PROPOSAL_COST`` after ``PROPOSAL_DELAY_S`` and returns a valid
    full rewrite."""

    def __init__(self):
        self.calls = 0

    async def __call__(
        self,
        msg,
        system_msg,
        msg_history,
        output_model=None,
        model_posteriors=None,
        **kwargs,
    ):
        await asyncio.sleep(PROPOSAL_DELAY_S)
        self.calls += 1
        content = (
            f"<NAME>stub_{self.calls}</NAME>\n<DESCRIPTION>stub</DESCRIPTION>\n"
            "```python\n# EVOLVE-BLOCK-START\n"
            "def score():\n    return 1.0\n"
            "# EVOLVE-BLOCK-END\n```\n"
        )
        return QueryResult(
            content=content,
            msg=msg,
            system_msg=system_msg,
            new_msg_history=[*msg_history, {"role": "assistant", "content": content}],
            model_name=kwargs.get("model_name", MODEL),
            kwargs=kwargs,
            input_tokens=1,
            output_tokens=1,
            cost=PROPOSAL_COST,
            model_posteriors=model_posteriors,
        )


class FakeScheduler:
    """The scheduler edge the job monitor talks to once the run is set up.

    ``cancels`` maps a job id to what ``cancel_job_async`` returns; a job
    reports running until its ``done_after`` status checks have passed."""

    job_type = "local"

    def __init__(self, cancels=None, done_after=None):
        self.cancels = dict(cancels or {})
        self.done_after = dict(done_after or {})
        self.cancel_calls: list[str] = []
        self.submitted: list[str] = []
        self.status_checks: dict[str, int] = {}

    async def batch_check_status_async(self, jobs):
        out = []
        for job in jobs:
            job_id = str(job.job_id)
            seen = self.status_checks.get(job_id, 0) + 1
            self.status_checks[job_id] = seen
            limit = self.done_after.get(job_id)
            out.append(limit is None or seen <= limit)
        return out

    async def cancel_job_async(self, job_id):
        self.cancel_calls.append(str(job_id))
        return self.cancels.get(str(job_id), False)

    async def submit_async_nonblocking(self, exec_fname, results_dir):
        self.submitted.append(exec_fname)
        return f"job-for-{exec_fname}"

    async def get_job_results_async(self, job_id, results_dir):
        raise AssertionError(f"a job dropped at the target was read: {job_id}")

    def shutdown(self):
        pass


@pytest.fixture
def offline(monkeypatch):
    """No pricing refresh, a dummy key, loopback-only sockets, the stub LLM."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-offline")
    monkeypatch.setattr(async_runner, "refresh_model_catalog", get_catalog)
    real_connect = socket.socket.connect

    def loopback_only(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise OSError(f"network blocked in test: {address!r}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    stub = StubLLM()
    monkeypatch.setattr(llm_module, "query_async", stub)
    return stub


def _runner(root: Path, num_generations: int) -> ShinkaEvolveRunner:
    task = root / "task"
    task.mkdir(parents=True, exist_ok=True)
    (task / "initial.py").write_text(INITIAL, encoding="utf-8")
    (task / "evaluate.py").write_text(EVALUATE, encoding="utf-8")
    runner = ShinkaEvolveRunner(
        evo_config=EvolutionConfig(
            patch_types=["full"],
            patch_type_probs=[1.0],
            num_generations=num_generations,
            max_patch_attempts=1,
            max_patch_resamples=1,
            max_novelty_attempts=1,
            llm_models=[MODEL],
            llm_dynamic_selection=None,
            meta_llm_models=None,
            meta_rec_interval=None,
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
        max_evaluation_jobs=3,
        max_proposal_jobs=1,
        max_db_workers=1,
    )
    runner.surplus_proposal_wait_s = 2.0
    return runner


async def _set_up(runner: ShinkaEvolveRunner, scheduler: FakeScheduler):
    """The run's real setup (database, generation 0 evaluated for real), then
    the fake scheduler for everything after it. Returns the real scheduler."""
    async_runner.activate_model_catalog(runner.pricing_snapshot)
    await runner._setup_async()
    await runner._verify_database_ready()
    real = runner.scheduler
    runner.scheduler = scheduler
    return real


async def _tear_down(runner: ShinkaEvolveRunner, real_scheduler) -> None:
    await runner.async_db.close_async()
    real_scheduler.shutdown()


async def _monitor_to_the_end(runner: ShinkaEvolveRunner) -> None:
    """Run the runner's own job monitor until it finalizes the run."""
    await asyncio.wait_for(runner._job_monitor_task(), MONITOR_TIMEOUT_S)
    assert runner.finalization_complete.is_set()
    assert runner._fatal_error is None


async def _surplus_job(
    runner: ShinkaEvolveRunner,
    job_id: str,
    generation: int,
    api: float,
    embed: float,
    novelty: float,
) -> AsyncRunningJob:
    """A submitted evaluation job holding a real evaluation slot, its
    proposal's costs as the runner records them at submission."""
    now = time.time()
    job = AsyncRunningJob(
        job_id=job_id,
        exec_fname=f"{runner.results_dir}/gen_{generation}/main.py",
        results_dir=f"{runner.results_dir}/gen_{generation}/results",
        start_time=now,
        proposal_started_at=now,
        evaluation_submitted_at=now,
        evaluation_started_at=now,
        generation=generation,
        evaluation_worker_id=await runner.evaluation_slot_pool.acquire(),
        meta_patch_data={"api_costs": api},
        embed_cost=embed,
        novelty_cost=novelty,
    )
    runner.submitted_jobs[job_id] = job
    return job


def _db(runner: ShinkaEvolveRunner) -> Path:
    return Path(runner.results_dir) / "programs.sqlite"


def _spend(runner: ShinkaEvolveRunner) -> Spend:
    with closing(sqlite3.connect(_db(runner))) as conn:
        return program_spend(conn)


def _discards(runner: ShinkaEvolveRunner) -> list[dict]:
    """Every charge-status event, decoded, in write order."""
    with closing(sqlite3.connect(_db(runner))) as conn:
        rows = conn.execute(
            "SELECT generation, status, details FROM attempt_log "
            "WHERE status IN ('failed', 'discarded') ORDER BY id"
        ).fetchall()
    return [
        {"generation": int(g), "status": status, **json.loads(details or "{}")}
        for g, status, details in rows
    ]


def _program_generations(runner: ShinkaEvolveRunner) -> list[int]:
    with closing(sqlite3.connect(_db(runner))) as conn:
        rows = conn.execute("SELECT generation FROM programs").fetchall()
    return sorted(int(g) for (g,) in rows)


def _assert_charged_once(events, job_id, reason, api, embed, novelty):
    mine = [e for e in events if e.get("source_job_id") == job_id]
    assert len(mine) == 1, (job_id, events)
    (event,) = mine
    assert event["status"] == "discarded", event
    assert event["node_kind"] == DISCARDED_PROPOSAL, event
    assert event["reason"] == reason, event
    assert (event["api_costs"], event["embed_cost"], event["novelty_cost"]) == (
        api,
        embed,
        novelty,
    ), event


def _assert_spend_grew_by(before: Spend, after: Spend, amount: float, events):
    assert before.known and after.known, (before, after)
    assert after.usd - before.usd == amount, (before, after, amount, events)


def test_jobs_cancelled_at_the_target_are_charged_once(tmp_path, offline):
    """Two jobs run at the target and the scheduler cancels them; a proposal
    that passed the target check just before the target submits a third job
    while the cleanup waits for it. Each is charged once, as one discard event
    with its own costs, and none becomes a program row."""
    costs = {
        "job-a": (2.0**-3, 2.0**-9, 2.0**-12),
        "job-b": (2.0**-4, 2.0**-10, 2.0**-13),
        "job-late": (2.0**-6, 2.0**-11, 2.0**-14),
    }
    scheduler = FakeScheduler(cancels={job_id: True for job_id in costs})

    async def run():
        runner = _runner(tmp_path, num_generations=1)
        real = await _set_up(runner, scheduler)
        try:
            await runner._update_completed_generations()
            assert runner.completed_generations == 1  # generation 0: the target
            before = _spend(runner)
            runner.running_jobs = [
                await _surplus_job(runner, "job-a", 1, *costs["job-a"]),
                await _surplus_job(runner, "job-b", 2, *costs["job-b"]),
            ]

            # Simulated: what the real proposal path does once a proposal that
            # passed the target check gets an evaluation slot (async_runner
            # _generate_evolved_proposal: submit, then append to running_jobs).
            async def late_proposal():
                await asyncio.sleep(0.2)
                job = await _surplus_job(runner, "job-late", 3, *costs["job-late"])
                runner.running_jobs.append(job)

            task = asyncio.create_task(late_proposal(), name="proposal_3")
            runner.active_proposal_tasks["late"] = task
            runner.assigned_generations.add(3)

            await _monitor_to_the_end(runner)

            events = _discards(runner)
            assert len(events) == 3, events
            for job_id, (api, embed, novelty) in costs.items():
                _assert_charged_once(
                    events, job_id, "cancelled at the target", api, embed, novelty
                )
            # job-a and job-b were cancelled by the first pass; the second
            # pass saw only the late job.
            assert scheduler.cancel_calls == ["job-a", "job-b", "job-late"]
            assert runner.running_jobs == [] and runner.submitted_jobs == {}
            assert runner.evaluation_slot_pool.in_use == 0
            assert _program_generations(runner) == [0]
            _assert_spend_grew_by(
                before, _spend(runner), sum(sum(c) for c in costs.values()), events
            )
            assert recorded_in_full(runner.results_dir)
            assert not (Path(runner.results_dir) / UNRECORDED).exists()
        finally:
            await _tear_down(runner, real)

    asyncio.run(run())


def test_a_job_whose_cancel_fails_is_charged_once_when_it_completes(tmp_path, offline):
    """The scheduler cannot cancel the job: the target cleanup does not charge
    it, however often it retries the cancel; it is charged once when it
    completes after the target, without its results being read."""
    api, embed, novelty = 2.0**-3, 2.0**-9, 2.0**-12
    # Running for its first three status checks, so the monitor's target
    # cleanup tries (and fails) to cancel it on more than one pass.
    scheduler = FakeScheduler(cancels={"job-stuck": False}, done_after={"job-stuck": 3})

    async def run():
        runner = _runner(tmp_path, num_generations=1)
        real = await _set_up(runner, scheduler)
        try:
            await runner._update_completed_generations()
            assert runner.completed_generations == 1
            before = _spend(runner)
            job = await _surplus_job(runner, "job-stuck", 1, api, embed, novelty)
            runner.running_jobs = [job]

            await _monitor_to_the_end(runner)

            # The cleanup retried the cancel (before and after its proposal
            # wait, on each pass that saw the job running) and charged nothing.
            assert len(scheduler.cancel_calls) >= 2, scheduler.cancel_calls
            assert set(scheduler.cancel_calls) == {"job-stuck"}
            assert scheduler.status_checks == {"job-stuck": 4}
            events = _discards(runner)
            assert len(events) == 1, events
            _assert_charged_once(
                events, "job-stuck", "completed after the target", api, embed, novelty
            )
            assert runner.running_jobs == [] and runner.submitted_jobs == {}
            assert runner.evaluation_slot_pool.in_use == 0
            assert _program_generations(runner) == [0]
            _assert_spend_grew_by(before, _spend(runner), api + embed + novelty, events)
            assert recorded_in_full(runner.results_dir)
        finally:
            await _tear_down(runner, real)

    asyncio.run(run())


def test_a_retry_queued_job_dropped_at_the_target_is_charged_once(tmp_path, offline):
    """A completed job whose database write failed waits in the retry queue
    when the run reaches its target: it is dropped, charged once with its
    costs, and never retried into a program row."""
    api, embed, novelty = 2.0**-4, 2.0**-10, 2.0**-13
    scheduler = FakeScheduler()

    async def run():
        runner = _runner(tmp_path, num_generations=1)
        real = await _set_up(runner, scheduler)
        try:
            await runner._update_completed_generations()
            assert runner.completed_generations == 1
            before = _spend(runner)
            job = await _surplus_job(runner, "job-retry", 1, api, embed, novelty)
            await runner._release_evaluation_slot_once(job)  # it had completed
            job.db_retry_count = 1
            runner.failed_jobs_for_retry = {"job-retry": job}

            await _monitor_to_the_end(runner)

            events = _discards(runner)
            assert len(events) == 1, events
            _assert_charged_once(
                events,
                "job-retry",
                "queued for retry at the target",
                api,
                embed,
                novelty,
            )
            assert runner.failed_jobs_for_retry == {}
            assert runner.submitted_jobs == {}
            assert scheduler.cancel_calls == []
            assert _program_generations(runner) == [0]
            _assert_spend_grew_by(before, _spend(runner), api + embed + novelty, events)
            assert recorded_in_full(runner.results_dir)
        finally:
            await _tear_down(runner, real)

    asyncio.run(run())


def test_a_proposal_finishing_after_the_target_is_charged_once(tmp_path, offline):
    """A real proposal task is mid-call when the run reaches its target. The
    cleanup waits for it (``surplus_proposal_wait_s``); it finishes inside the
    wait, books what it paid as one discard event, submits no job and takes no
    evaluation slot, and the run's total stays complete."""
    scheduler = FakeScheduler()

    async def run():
        runner = _runner(tmp_path, num_generations=2)
        real = await _set_up(runner, scheduler)
        try:
            await runner._update_completed_generations()
            assert runner.completed_generations == 1
            before = _spend(runner)

            await runner._start_proposals(1)
            (task,) = runner.active_proposal_tasks.values()
            assert task.get_name() == "proposal_1"
            # The target is reached while the proposal is in flight. In a run
            # that happens when island copies count toward it (test_run_spend);
            # set before the task first runs, so it is mid-call either way.
            runner.completed_generations = runner.evo_config.num_generations

            started = time.monotonic()
            await _monitor_to_the_end(runner)
            waited = time.monotonic() - started

            assert task.done() and not task.cancelled()
            assert task.result() is None  # discarded, no running job
            assert offline.calls == 1, "the proposal never paid for a call"
            assert waited < runner.surplus_proposal_wait_s, waited
            events = _discards(runner)
            assert events == [
                {
                    "generation": 1,
                    "status": "discarded",
                    "node_kind": DISCARDED_PROPOSAL,
                    "reason": "finished after the target",
                    "api_costs": PROPOSAL_COST,
                    "embed_cost": 0.0,
                    "novelty_cost": 0.0,
                }
            ], events
            assert scheduler.submitted == []
            assert runner.evaluation_slot_pool.peak_in_use == 0
            assert runner.running_jobs == [] and runner.submitted_jobs == {}
            assert runner.active_proposal_tasks == {}
            assert _program_generations(runner) == [0]
            _assert_spend_grew_by(before, _spend(runner), PROPOSAL_COST, events)
            assert recorded_in_full(runner.results_dir)
        finally:
            await _tear_down(runner, real)

    asyncio.run(_in_client_scope(run))


async def _in_client_scope(run):
    """``run_async``'s provider-client scope around ``run``."""
    async with async_client_cache_scope():
        await run()
