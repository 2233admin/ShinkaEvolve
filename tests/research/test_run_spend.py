"""The run-cost rule against the real spend of a real run.

``ShinkaEvolveRunner.run_async`` drives a trivial task end to end with the LLM
replaced at its lowest layer (``shinka.llm.llm.query_async``) by a stub that
keeps its own ledger of every charge it returns. That ledger is the spend; no
expected value is derived from ``run_cost`` or the runner.

One scenario, run once per module, holds every kind of charge the rule has to
see:

- dynamic islands copy a paid program (``_spawned_island`` rows);
- terminal failed proposals (recorded only in ``attempt_log``): generation
  ``FAIL_GEN`` of run 1 and the first generation of the resumed run fail all
  their attempts, and later generations succeed;
- meta recommendations: one update whose last step fails after paying, a
  successful update, and the final meta summary at the end of each run;
- a stop and a resume in the same results directory.

Each invariant is its own test, so a fix turns them green one by one: at the
resume the runner's loaded base equals the ledger so far, and after each run
``run_cost(results).recorded_usd`` equals the ledger so far. Where a run stops
is deliberately not asserted.

Each kind of charge costs a different power of two, so every sum is exact and
a failure message's difference names the charges it is made of.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import math
import socket
import sqlite3
import sys
import time
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

import shinka.core.async_runner as async_runner
import shinka.llm.llm as llm_module
from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.core.async_runner import ChargeNotRecordedError
from shinka.core.async_summarizer import AsyncMetaSummarizer
from shinka.database import AsyncProgramDatabase, DatabaseConfig, ProgramDatabase
from shinka.database.cost import CHARGE_KINDS, EVENT_FIELDS, UNRECORDED
from shinka.launch import LocalJobConfig
from shinka.llm.providers import QueryResult
from shinka.pricing.catalog import get_catalog
from shinka.prompts import (
    META_STEP1_SYSTEM_MSG,
    META_STEP2_SYSTEM_MSG,
    META_STEP3_SYSTEM_MSG,
)
from shinka.prompts.prompts_prompt_evo import PROMPT_EVO_SYSTEM_BASE
from shinka.research.rundb import run_cost

COSTS = {
    "mutation": 2.0**-3,  # a proposal call that returns a usable rewrite
    "failed_proposal": 2.0**-10,  # a proposal call of a failing generation
    "meta": 2.0**-6,  # a meta step during the run (failed update included)
    "meta_final": 2.0**-14,  # a meta step of the end-of-run summary
}
MODEL = "gpt-5-mini"

FAIL_GEN = 4  # run 1's failing generation; gens 1-3 make the islands stagnate
RUN1_GENERATIONS = 8  # generations 0..7
RUN2_EXTRA_GENERATIONS = 5  # run 2: num_generations = run 1's last generation + 5
META_REC_INTERVAL = 4
RUN_TIMEOUT_S = 45
# Liveness backstop. A failed proposal does not wake the coordinator, so the
# next one starts through stuck recovery (async_runner.py:2516, :5387-5396).
STALL_TIMEOUT_S = 3.0

ISLANDS = dict(
    num_islands=2,
    enable_dynamic_islands=True,
    stagnation_threshold=2,
    island_spawn_strategy="best",
)

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

# The generation a proposal call belongs to, and whether a meta call is the
# end-of-run summary. Set inside the task that makes the calls; the batch
# queries it gathers copy its context.
GENERATION = contextvars.ContextVar("generation", default=None)
FINAL_SUMMARY = contextvars.ContextVar("final_summary", default=False)

META_STEPS = {
    META_STEP1_SYSTEM_MSG: "step1",
    META_STEP2_SYSTEM_MSG: "step2",
    META_STEP3_SYSTEM_MSG: "step3",
}


class LedgerLLM:
    """Stands in for ``shinka.llm.llm.query_async`` and records every charge.

    A proposal call returns a valid full rewrite unless ``fails(generation)``;
    then it gets prose without a code block (non-empty, so the runner reads
    its cost) and the generation exhausts its attempts. The first meta step-3
    call returns empty content: that update has paid for all three steps and
    then fails."""

    def __init__(self):
        self.charges: list[dict] = []
        self.fails = lambda generation: False
        self.step3_failures_left = 1

    @property
    def total(self) -> float:
        return sum(c["cost"] for c in self.charges)

    def breakdown(self, start: int = 0, stop: int | None = None) -> dict:
        out: dict = defaultdict(float)
        for c in self.charges[start:stop]:
            out[c["kind"]] += c["cost"]
        return dict(out)

    async def __call__(
        self,
        msg,
        system_msg,
        msg_history,
        output_model=None,
        model_posteriors=None,
        **kwargs,
    ):
        await asyncio.sleep(0.02)
        step = META_STEPS.get(system_msg)
        generation = GENERATION.get()
        n = len(self.charges) + 1
        if step is not None and FINAL_SUMMARY.get():
            kind, cost, content = (
                f"meta_final:{step}",
                COSTS["meta_final"],
                f"final {n}",
            )
        elif step is not None:
            cost = COSTS["meta"]
            if step == "step3" and self.step3_failures_left:
                self.step3_failures_left -= 1
                kind, content = "meta:failed_update", ""
            else:
                kind, content = f"meta:{step}", f"meta {step} {n}"
        elif generation is not None and self.fails(generation):
            kind, cost = "failed_proposal", COSTS["failed_proposal"]
            content = (
                f"<NAME>nothing_{n}</NAME>\n<DESCRIPTION>no code</DESCRIPTION>\n"
                "I could not produce a program this time.\n"
            )
        else:
            # A proposal call outside a tagged proposal task is a path this
            # test does not know; recorded, and rejected by the fixture.
            kind = "mutation" if generation is not None else "untagged"
            cost = COSTS["mutation"]
            score = 1.0 if generation == 1 else 0.5  # gen 1 best: stagnation
            content = (
                f"<NAME>stub_{n}</NAME>\n<DESCRIPTION>stub {n}</DESCRIPTION>\n"
                "```python\n# EVOLVE-BLOCK-START\n"
                f"def score():\n    return {score!r}  # gen {generation} call {n}\n"
                "# EVOLVE-BLOCK-END\n```\n"
            )
        self.charges.append(dict(kind=kind, generation=generation, cost=cost))
        return QueryResult(
            content=content,
            msg=msg,
            system_msg=system_msg,
            new_msg_history=[*msg_history, {"role": "assistant", "content": content}],
            model_name=kwargs.get("model_name", MODEL),
            kwargs=kwargs,
            input_tokens=1,
            output_tokens=1,
            cost=cost,
            model_posteriors=model_posteriors,
        )


def _offline(patch: pytest.MonkeyPatch) -> SimpleNamespace:
    """No pricing refresh, a dummy key, loopback-only sockets; tags proposal
    and final-summary calls; records the resume base the runner loads."""
    patch.delenv("SHINKA_RESEARCH_AGENT", raising=False)
    patch.setenv("OPENAI_API_KEY", "sk-test-offline")
    patch.setattr(async_runner, "refresh_model_catalog", get_catalog)
    real_connect = socket.socket.connect

    def loopback_only(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise OSError(f"network blocked in test: {address!r}")
        return real_connect(self, address)

    patch.setattr(socket.socket, "connect", loopback_only)

    probe = SimpleNamespace(loaded=[], stalls=[])
    real_total = ShinkaEvolveRunner._get_total_api_costs

    async def recording_total(self):
        value = await real_total(self)
        probe.loaded.append(value)
        return value

    real_proposal = ShinkaEvolveRunner._generate_proposal_async

    async def tagged_proposal(self, generation, task_id):
        GENERATION.set(generation)  # this proposal task's own context
        return await real_proposal(self, generation, task_id)

    real_final = AsyncMetaSummarizer.perform_final_summary_async

    async def tagged_final(self, *args, **kwargs):
        FINAL_SUMMARY.set(True)  # runs inside asyncio.wait_for's own task
        return await real_final(self, *args, **kwargs)

    real_stuck = ShinkaEvolveRunner._handle_stuck_system

    async def recording_stuck(self):
        last = self.last_progress_time
        if last is not None and time.time() - last > self.stuck_detection_timeout:
            probe.stalls.append(self.completed_generations)
        return await real_stuck(self)

    patch.setattr(ShinkaEvolveRunner, "_get_total_api_costs", recording_total)
    patch.setattr(ShinkaEvolveRunner, "_generate_proposal_async", tagged_proposal)
    patch.setattr(AsyncMetaSummarizer, "perform_final_summary_async", tagged_final)
    patch.setattr(ShinkaEvolveRunner, "_handle_stuck_system", recording_stuck)
    return probe


def _run(patch, root: Path, stub: LedgerLLM, num_generations: int):
    task = root / "task"
    task.mkdir(parents=True, exist_ok=True)
    (task / "initial.py").write_text(INITIAL, encoding="utf-8")
    (task / "evaluate.py").write_text(EVALUATE, encoding="utf-8")
    patch.setattr(llm_module, "query_async", stub)
    runner = ShinkaEvolveRunner(
        evo_config=EvolutionConfig(
            patch_types=["full"],
            patch_type_probs=[1.0],
            num_generations=num_generations,
            max_patch_attempts=2,
            max_patch_resamples=1,
            max_novelty_attempts=1,
            llm_models=[MODEL],
            llm_dynamic_selection=None,
            meta_llm_models=[MODEL],
            meta_rec_interval=META_REC_INTERVAL,
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
        db_config=DatabaseConfig(**ISLANDS),
        verbose=False,
        # One at a time: island spawning, and so the copy rows, are the same
        # on every run.
        max_evaluation_jobs=1,
        max_proposal_jobs=1,
        max_db_workers=1,
    )
    runner.stuck_detection_timeout = STALL_TIMEOUT_S  # liveness backstop only
    asyncio.run(asyncio.wait_for(runner.run_async(), RUN_TIMEOUT_S))
    return runner


def _programs(results: Path) -> list[tuple[int, dict]]:
    with closing(sqlite3.connect(results / "programs.sqlite")) as conn:
        rows = conn.execute("SELECT generation, metadata FROM programs").fetchall()
    return [(int(g), json.loads(md or "{}")) for g, md in rows]


def _is_copy(md: dict) -> bool:
    return bool(md.get("_is_island_copy") or md.get("_spawned_island"))


def _row_cost(md: dict) -> float:
    return sum(
        float(md.get(f) or 0.0)
        for f in ("api_costs", "embed_cost", "novelty_cost", "meta_cost")
    )


def _failed_proposals(results: Path) -> list[tuple[int, float]]:
    with closing(sqlite3.connect(results / "programs.sqlite")) as conn:
        rows = conn.execute(
            "SELECT generation, details FROM attempt_log WHERE status = 'failed' "
            "AND json_extract(details, '$.node_kind') = 'failed_proposal' ORDER BY id"
        ).fetchall()
    return [
        (
            int(g),
            sum(
                float(json.loads(d).get(f) or 0.0)
                for f in ("api_costs", "embed_cost", "novelty_cost")
            ),
        )
        for g, d in rows
    ]


def _snapshot(results: Path, stub: LedgerLLM, runner) -> dict:
    rows = _programs(results)
    return dict(
        ledger=stub.total,
        ledger_by_kind=stub.breakdown(),
        run_cost=run_cost(results).recorded_usd,
        non_copy_rows=sum(_row_cost(m) for _, m in rows if not _is_copy(m)),
        copy_rows=sum(_row_cost(m) for _, m in rows if _is_copy(m)),
        attempt_log_failed=_failed_proposals(results),
        runner_in_memory=runner.total_api_cost,
    )


def _assert_stub_sees_every_paid_call(rows, stub: LedgerLLM) -> None:
    """Each evaluated row's api_costs is exactly the ledger's usable-rewrite
    charges for its generation: the stub is the only thing that charges, and
    the runner keeps the cost the stub returns."""
    for g, md in rows:
        if _is_copy(md):
            continue
        charged = sum(
            c["cost"]
            for c in stub.charges
            if c["generation"] == g and c["kind"] == "mutation"
        )
        assert float(md.get("api_costs") or 0.0) == charged, (g, md, charged)
        assert not md.get("embed_cost") and not md.get("novelty_cost"), md


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    root = tmp_path_factory.mktemp("spend")
    results = root / "results"
    stub = LedgerLLM()
    with pytest.MonkeyPatch.context() as patch:
        probe = _offline(patch)

        # Run 1: fresh; generation FAIL_GEN fails all its attempts, the
        # generations after it succeed.
        stub.fails = lambda g: g == FAIL_GEN
        runner = _run(patch, root, stub, RUN1_GENERATIONS)
        run1_charges = len(stub.charges)
        after_run1 = _snapshot(results, stub, runner)
        rows = _programs(results)
        last = max(g for g, _ in rows)
        assert probe.loaded == [0.0]  # a fresh directory has no earlier spend
        _assert_stub_sees_every_paid_call(rows, stub)
        assert any(_is_copy(m) and _row_cost(m) > 0 for _, m in rows), rows
        assert [g for g, _ in after_run1["attempt_log_failed"]] == [FAIL_GEN]
        assert FAIL_GEN not in {g for g, _ in rows} and last > FAIL_GEN, rows
        kinds = stub.breakdown()
        assert "meta:failed_update" in kinds, kinds  # paid, then failed
        assert "meta:step3" in kinds, kinds  # a later update succeeded
        assert "meta_final:step3" in kinds, kinds  # the final summary ran

        # Run 2: resume in the same directory. It starts at last + 1
        # (async_runner.py:5021); that generation fails, the next ones succeed.
        # (A failed generation at the top of run 1 is not retried: its gen_N
        # directory makes the resumed proposal abort at :2614-2635.)
        resumed_fail = last + 1
        stub.fails = lambda g: g == resumed_fail
        runner = _run(patch, root, stub, last + RUN2_EXTRA_GENERATIONS)
        after_run2 = _snapshot(results, stub, runner)
        rows = _programs(results)
        _assert_stub_sees_every_paid_call(rows, stub)
        assert len(probe.loaded) == 2, probe.loaded  # fresh base, resume base
        assert [g for g, _ in after_run2["attempt_log_failed"]] == [
            FAIL_GEN,
            resumed_fail,
        ], (
            [(c["kind"], c["generation"]) for c in stub.charges[run1_charges:]],
            [(g, _is_copy(m)) for g, m in rows],
            probe.stalls,
        )
        assert any(not _is_copy(m) and g > resumed_fail for g, m in rows), rows
        assert "meta_final:step3" in stub.breakdown(run1_charges), stub.breakdown(
            run1_charges
        )
        assert "untagged" not in stub.breakdown(), stub.breakdown()

    return SimpleNamespace(
        after_run1=after_run1,
        after_run2=after_run2,
        loaded=probe.loaded[1],
        spent_run1=after_run1["ledger"],
        spent_total=after_run2["ledger"],
        run1_by_kind=stub.breakdown(0, run1_charges),
        run2_by_kind=stub.breakdown(run1_charges),
        stalls=probe.stalls,
    )


def _explain(got: float, want: float, **context) -> str:
    return f"got {got}, ledger {want}, diff {got - want}; " + "; ".join(
        f"{k}={v}" for k, v in context.items()
    )


def test_run_cost_after_a_run_is_its_spend(scenario):
    s = scenario
    got = s.after_run1["run_cost"]
    assert math.isclose(got, s.spent_run1, rel_tol=0, abs_tol=1e-12), _explain(
        got, s.spent_run1, run1_ledger_by_kind=s.run1_by_kind, state=s.after_run1
    )


def test_resume_base_is_the_spend_so_far(scenario):
    s = scenario
    assert math.isclose(s.loaded, s.spent_run1, rel_tol=0, abs_tol=1e-12), _explain(
        s.loaded,
        s.spent_run1,
        run1_ledger_by_kind=s.run1_by_kind,
        state=s.after_run1,
    )


def test_run_cost_after_a_resumed_run_is_the_total_spend(scenario):
    s = scenario
    got = s.after_run2["run_cost"]
    assert math.isclose(got, s.spent_total, rel_tol=0, abs_tol=1e-12), _explain(
        got,
        s.spent_total,
        run1_ledger_by_kind=s.run1_by_kind,
        run2_ledger_by_kind=s.run2_by_kind,
        state=s.after_run2,
    )


def test_run_cost_reads_a_legacy_database(tmp_path):
    """Guard for the fix, passes today: a results dir from before attempt_log
    and the copy flags still reads, as the total of its program rows."""
    rows = [
        ("a", {"api_costs": 0.5, "embed_cost": 0.25}),
        ("b", {"api_costs": 0.125, "meta_cost": 0.0625, "novelty_cost": 0.03125}),
        ("c", {"patch_name": "initial_program"}),
        ("d", None),
    ]
    with closing(sqlite3.connect(tmp_path / "programs.sqlite")) as conn:
        conn.execute(
            "CREATE TABLE programs (id TEXT PRIMARY KEY, code TEXT NOT NULL, "
            "generation INTEGER, correct BOOLEAN, combined_score REAL, "
            "timestamp REAL, metadata TEXT)"
        )
        conn.executemany(
            "INSERT INTO programs VALUES (?, '', 0, 1, 0.0, 0.0, ?)",
            [(pid, None if md is None else json.dumps(md)) for pid, md in rows],
        )
        conn.commit()

    cost = run_cost(tmp_path)
    assert cost.known
    assert math.isclose(cost.recorded_usd, 0.96875, rel_tol=0, abs_tol=1e-12)


# --- The reopened rule: one run per charge path, the ledger as ground truth ---

EMPTY_COST = 2.0**-18  # a paid reply without content
PROMPT_EVO_COST = 2.0**-16  # a prompt-evolution call
EMPTY_PROPOSAL_GEN = 2

# Generation g's evaluation takes 0.6 * g seconds, so the later generations are
# still in flight when the run reaches its target.
SLOW_EVALUATE = EVALUATE.replace(
    "a = p.parse_args()\n",
    "a = p.parse_args()\n"
    "import time\n"
    "gen = os.path.basename(os.path.dirname(os.path.abspath(a.program_path)))\n"
    "time.sleep(0.6 * int(gen.rsplit('_', 1)[1]))\n",
)


class ScriptedLLM(LedgerLLM):
    """``LedgerLLM`` that also answers prompt evolution, never fails an in-run
    meta update unless told, and can return one paid reply without content
    on a chosen path (``empty``: ``proposal`` for generation
    ``EMPTY_PROPOSAL_GEN``, ``meta_step1``, ``prompt_evo``), or fail the final
    summary's last step after it paid."""

    def __init__(self, empty=(), final_step3_fails=False):
        super().__init__()
        self.step3_failures_left = 0
        self.empty = set(empty)
        self.final_step3_fails = final_step3_fails

    def _take(self, path: str) -> bool:
        if path in self.empty:
            self.empty.discard(path)
            return True
        return False

    async def __call__(
        self,
        msg,
        system_msg,
        msg_history,
        output_model=None,
        model_posteriors=None,
        **kwargs,
    ):
        step = META_STEPS.get(system_msg)
        generation = GENERATION.get()
        final = FINAL_SUMMARY.get()
        scripted = None
        if isinstance(system_msg, str) and system_msg.startswith(
            PROMPT_EVO_SYSTEM_BASE
        ):
            if self._take("prompt_evo"):
                scripted = ("empty:prompt_evo", EMPTY_COST, "")
            else:
                n = len(self.charges) + 1
                text = " ".join(["Write fast, correct, well-tested code."] * 4)
                scripted = (
                    "prompt_evo",
                    PROMPT_EVO_COST,
                    f"<NAME>prompt_{n}</NAME>\n<DESCRIPTION>d</DESCRIPTION>\n"
                    f"<PROMPT>{text} ({n})</PROMPT>",
                )
        elif final and step == "step3" and self.final_step3_fails:
            scripted = ("meta_final:failed_step3", COSTS["meta_final"], "")
        elif not final and step == "step1" and self._take("meta_step1"):
            scripted = ("empty:meta_step1", EMPTY_COST, "")
        elif (
            step is None and generation == EMPTY_PROPOSAL_GEN and self._take("proposal")
        ):
            scripted = ("empty:proposal", EMPTY_COST, "")
        if scripted is None:
            return await super().__call__(
                msg, system_msg, msg_history, output_model, model_posteriors, **kwargs
            )
        await asyncio.sleep(0.02)
        kind, cost, content = scripted
        self.charges.append(dict(kind=kind, generation=generation, cost=cost))
        return QueryResult(
            content=content,
            msg=msg,
            system_msg=system_msg,
            new_msg_history=[*msg_history, {"role": "assistant", "content": content}],
            model_name=kwargs.get("model_name", MODEL),
            kwargs=kwargs,
            input_tokens=1,
            output_tokens=1,
            cost=cost,
            model_posteriors=model_posteriors,
        )


def _run_with(
    patch,
    root: Path,
    stub: LedgerLLM,
    num_generations: int,
    *,
    evaluate: str = EVALUATE,
    evo: dict | None = None,
    db: dict | None = None,
    jobs: int = 1,
    proposals: int = 1,
    stall: float = STALL_TIMEOUT_S,
    final_summary_timeout_s: float | None = None,
):
    """``_run`` with the settings a path needs."""
    task = root / "task"
    task.mkdir(parents=True, exist_ok=True)
    (task / "initial.py").write_text(INITIAL, encoding="utf-8")
    (task / "evaluate.py").write_text(evaluate, encoding="utf-8")
    patch.setattr(llm_module, "query_async", stub)
    config = dict(
        patch_types=["full"],
        patch_type_probs=[1.0],
        num_generations=num_generations,
        max_patch_attempts=2,
        max_patch_resamples=1,
        max_novelty_attempts=1,
        llm_models=[MODEL],
        llm_dynamic_selection=None,
        meta_llm_models=[MODEL],
        meta_rec_interval=META_REC_INTERVAL,
        embedding_model=None,
        novelty_llm_models=None,
        evolve_prompts=False,
        init_program_path=str(task / "initial.py"),
        results_dir=str(root / "results"),
        max_api_costs=None,
    )
    config.update(evo or {})
    runner = ShinkaEvolveRunner(
        evo_config=EvolutionConfig(**config),
        job_config=LocalJobConfig(
            eval_program_path=str(task / "evaluate.py"),
            python_executable=sys.executable,
        ),
        db_config=DatabaseConfig(**{**ISLANDS, **(db or {})}),
        verbose=False,
        max_evaluation_jobs=jobs,
        max_proposal_jobs=proposals,
        max_db_workers=1,
    )
    runner.stuck_detection_timeout = stall
    if final_summary_timeout_s is not None:
        runner.final_summary_timeout_s = final_summary_timeout_s
    asyncio.run(asyncio.wait_for(runner.run_async(), RUN_TIMEOUT_S))
    return runner


def _charge_events(results: Path) -> list[tuple[int, str, str, float]]:
    """(generation, status, node_kind, amount) of each charge-status event."""
    with closing(sqlite3.connect(results / "programs.sqlite")) as conn:
        rows = conn.execute(
            "SELECT generation, status, details FROM attempt_log "
            "WHERE status IN ('failed', 'discarded') ORDER BY id"
        ).fetchall()
    out = []
    for g, status, details in rows:
        d = json.loads(details or "{}")
        amount = sum(float(d.get(f) or 0.0) for f in EVENT_FIELDS)
        out.append((int(g), status, d.get("node_kind"), amount))
    return out


def _assert_run_cost_is_the_ledger(results: Path, stub: LedgerLLM, **context):
    cost = run_cost(results)
    assert stub.total > 0, stub.charges
    assert cost.known, (cost, context)
    assert math.isclose(cost.recorded_usd, stub.total, rel_tol=0, abs_tol=1e-12), (
        _explain(
            cost.recorded_usd,
            stub.total,
            ledger_by_kind=stub.breakdown(),
            programs_usd=cost.programs_usd,
            prompts_usd=cost.prompts_usd,
            events=_charge_events(results),
            **context,
        )
    )


def test_discarded_surplus_proposals_are_in_run_cost(tmp_path):
    """Three proposals and three evaluations at a time, each generation's
    evaluation slower than the last, and a stagnant generation spawns an
    island at once: the copy row counts toward the target while the later
    generations are in flight, so the run reaches its target with paid work
    it drops. That work is charged once, as a ``discarded_proposal`` event."""
    results = tmp_path / "results"
    stub = ScriptedLLM()
    with pytest.MonkeyPatch.context() as patch:
        _offline(patch)
        runner = _run_with(
            patch,
            tmp_path,
            stub,
            8,
            evaluate=SLOW_EVALUATE,
            db=dict(stagnation_threshold=1),
            jobs=3,
            proposals=3,
        )
    events = _charge_events(results)
    discarded = [e for e in events if e[2] == "discarded_proposal"]
    assert discarded, (
        "setup produced no surplus: "
        f"events={events}, rows={[(g, _is_copy(m)) for g, m in _programs(results)]}"
    )
    assert all(
        status == "discarded" and amount > 0 for _, status, _, amount in discarded
    )
    _assert_run_cost_is_the_ledger(
        results, stub, discarded=discarded, runner_in_memory=runner.total_api_cost
    )


def test_resuming_a_run_with_only_generation_0_loads_its_failed_proposals(tmp_path):
    """Every proposal of run 1 fails, so the database holds generation 0 and
    failed-proposal events only. The resumed run's cost base is run 1's spend
    (it used to load 0 below ``last_iteration > 0``), and it resumes instead
    of adding a second generation 0."""
    results = tmp_path / "results"
    stub = ScriptedLLM()
    with pytest.MonkeyPatch.context() as patch:
        probe = _offline(patch)
        stub.fails = lambda g: g >= 1
        _run_with(patch, tmp_path, stub, 3, stall=1.0)
        run1 = stub.total
        rows = _programs(results)
        assert [g for g, m in rows if not _is_copy(m)] == [0], rows
        failed = [e for e in _charge_events(results) if e[2] == "failed_proposal"]
        assert failed and stub.breakdown().get("failed_proposal", 0) > 0, failed
        # The case the old rule missed: the base was loaded only past it.
        with closing(sqlite3.connect(results / "programs.sqlite")) as conn:
            (last,) = conn.execute(
                "SELECT value FROM metadata_store WHERE key = 'last_iteration'"
            ).fetchone()
        assert int(last) == 0, last
        _assert_run_cost_is_the_ledger(results, stub, run=1)

        stub.fails = lambda g: False
        loads_before = len(probe.loaded)
        _run_with(patch, tmp_path, stub, 5, stall=1.0)
        base = probe.loaded[loads_before]
    assert math.isclose(base, run1, rel_tol=0, abs_tol=1e-12), _explain(
        base, run1, run1_failed_events=failed
    )
    rows = _programs(results)
    assert [g for g, m in rows if g == 0 and not _is_copy(m)] == [0], rows
    _assert_run_cost_is_the_ledger(results, stub, run=2)


def _copy_target(results: Path, program_id: str) -> str:
    """The row a copy's charge belongs to: its original, followed through."""
    with closing(sqlite3.connect(results / "programs.sqlite")) as conn:
        while True:
            (md,) = conn.execute(
                "SELECT metadata FROM programs WHERE id = ?", (program_id,)
            ).fetchone()
            md = json.loads(md or "{}")
            if md.get("_is_island_copy"):
                program_id = md["_original_program_id"]
            elif md.get("_spawned_island"):
                program_id = md["_spawned_from_program_id"]
            else:
                return program_id


def _best_is_a_copy(patch) -> list:
    """Make the database's best program an island copy of it (or, failing a
    copy of the best, the best-scoring copy); returns the ids handed out."""
    handed = []
    real_best = AsyncProgramDatabase.get_best_program_async

    async def copy_of_best(self):
        best = await real_best(self)

        def pick():
            with closing(sqlite3.connect(self.sync_db.config.db_path)) as conn:
                rows = conn.execute(
                    "SELECT id, combined_score, metadata FROM programs"
                ).fetchall()
            copies = []
            for pid, score, md in rows:
                md = json.loads(md or "{}")
                source = md.get("_original_program_id") or md.get(
                    "_spawned_from_program_id"
                )
                if _is_copy(md):
                    copies.append((source == getattr(best, "id", None), score, pid))
            if not copies:
                return best
            pid = max(copies)[2]
            db = ProgramDatabase(self.sync_db.config, read_only=True)
            try:
                return db.get(pid)
            finally:
                db.close()

        chosen = await asyncio.get_event_loop().run_in_executor(None, pick)
        handed.append(getattr(chosen, "id", None))
        return chosen

    patch.setattr(AsyncProgramDatabase, "get_best_program_async", copy_of_best)
    return handed


@pytest.mark.parametrize("best", ["evaluated program", "island copy"])
def test_a_failed_final_summary_charges_the_best_programs_original_row(tmp_path, best):
    """The end-of-run summary's last step fails after its steps paid. The cost
    is on a program row the rule counts: the best one's, or its original's
    when the best is an island copy."""
    results = tmp_path / "results"
    stub = ScriptedLLM(final_step3_fails=True)
    with pytest.MonkeyPatch.context() as patch:
        _offline(patch)
        handed = _best_is_a_copy(patch) if best == "island copy" else None
        # No update during the run: every meta charge is the final summary's.
        _run_with(patch, tmp_path, stub, 6, evo=dict(meta_rec_interval=1000))
    kinds = stub.breakdown()
    final = sum(v for k, v in kinds.items() if k.startswith("meta_final:"))
    assert "meta_final:failed_step3" in kinds and final > 0, kinds
    assert not any(k.startswith("meta:") for k in kinds), kinds
    _assert_run_cost_is_the_ledger(results, stub, best=best)

    with closing(sqlite3.connect(results / "programs.sqlite")) as conn:
        rows = conn.execute("SELECT id, metadata FROM programs").fetchall()
    charged = {
        pid: md["meta_cost"]
        for pid, md in ((pid, json.loads(md or "{}")) for pid, md in rows)
        if md.get("meta_cost")
    }
    assert len(charged) == 1, charged
    ((target, meta_cost),) = charged.items()
    assert math.isclose(meta_cost, final, rel_tol=0, abs_tol=1e-12), (charged, kinds)
    md = json.loads(dict(rows)[target] or "{}")
    assert not _is_copy(md), (target, md)
    if handed is not None:
        copy_id = handed[-1]
        assert copy_id != target and _is_copy(
            json.loads(dict(rows)[copy_id] or "{}")
        ), (copy_id, handed)
        assert target == _copy_target(results, copy_id), (target, copy_id)


EMPTY_PATHS = {
    "proposal": dict(),
    "meta_step1": dict(),
    "prompt_evo": dict(
        evolve_prompts=True,
        prompt_evolution_interval=1,
        prompt_patch_types=["full"],
        prompt_patch_type_probs=[1.0],
    ),
}


@pytest.mark.parametrize("path", list(EMPTY_PATHS))
def test_a_paid_reply_without_content_is_in_run_cost(tmp_path, path):
    """One reply comes back empty but paid; the difference from the ledger, if
    any, is exactly ``EMPTY_COST``."""
    results = tmp_path / "results"
    stub = ScriptedLLM(empty={path})
    with pytest.MonkeyPatch.context() as patch:
        _offline(patch)
        _run_with(patch, tmp_path, stub, 6, evo=EMPTY_PATHS[path])
    assert stub.breakdown().get(f"empty:{path}") == EMPTY_COST, stub.breakdown()
    if path == "prompt_evo":
        assert stub.breakdown().get("prompt_evo", 0) > 0, stub.breakdown()
    _assert_run_cost_is_the_ledger(results, stub, path=path)


def _refuse_charge_writes(patch) -> list:
    """The database refuses every attempt-log write; returns the charge
    events it refused."""
    refused = []

    async def refuse(self, generation, stage, status, details=None):
        if (details or {}).get("node_kind") in CHARGE_KINDS:
            refused.append((generation, status, dict(details)))
        raise sqlite3.OperationalError("database is locked")

    patch.setattr(AsyncProgramDatabase, "record_attempt_event_async", refuse)
    return refused


def test_a_charge_the_database_refuses_leaves_the_run_cost_unknown(tmp_path):
    results = tmp_path / "results"
    stub = ScriptedLLM()
    stub.fails = lambda g: g == 2
    with pytest.MonkeyPatch.context() as patch:
        _offline(patch)
        refused = _refuse_charge_writes(patch)
        runner = _run_with(patch, tmp_path, stub, 6, stall=1.0)
    assert [(g, d["node_kind"]) for g, _, d in refused] == [(2, "failed_proposal")]
    assert runner.cost_limit_reached
    marker = json.loads((results / UNRECORDED).read_text(encoding="utf-8"))
    unrecorded = sum(float(marker.get(f) or 0.0) for f in EVENT_FIELDS)
    assert unrecorded == stub.breakdown()["failed_proposal"], (marker, stub.breakdown())
    cost = run_cost(results)
    assert not cost.known, cost
    # The marker is the only charge missing from the recorded total.
    assert math.isclose(
        cost.recorded_usd + unrecorded, stub.total, rel_tol=0, abs_tol=1e-12
    ), _explain(cost.recorded_usd + unrecorded, stub.total, ledger=stub.breakdown())


def test_a_charge_neither_recorded_nor_marked_stops_the_run(tmp_path):
    stub = ScriptedLLM()
    stub.fails = lambda g: g == 2
    with pytest.MonkeyPatch.context() as patch:
        _offline(patch)
        refused = _refuse_charge_writes(patch)

        def disk_full(results_dir, charge):
            raise OSError(28, "No space left on device")

        patch.setattr(async_runner, "mark_unrecorded", disk_full)
        with pytest.raises(ChargeNotRecordedError):
            _run_with(patch, tmp_path, stub, 6, stall=1.0)
    assert [(g, d["node_kind"]) for g, _, d in refused] == [(2, "failed_proposal")]
    # It stopped at the unrecorded charge: no later generation was proposed.
    assert max(c["generation"] or 0 for c in stub.charges) == 2, stub.charges


class HangingFinalLLM(ScriptedLLM):
    """``ScriptedLLM`` whose end-of-run summary hangs at step 2, after step 1
    paid: the call never returns, so nothing records what step 1 cost."""

    async def __call__(self, msg, system_msg, msg_history, *args, **kwargs):
        if FINAL_SUMMARY.get() and META_STEPS.get(system_msg) == "step2":
            await asyncio.sleep(3600)
        return await super().__call__(msg, system_msg, msg_history, *args, **kwargs)


def test_a_final_summary_that_times_out_leaves_the_run_cost_unknown(tmp_path):
    """The end-of-run summary outlives ``final_summary_timeout_s`` after one of
    its steps paid. The run still finishes, and its total reads unknown."""
    results = tmp_path / "results"
    stub = HangingFinalLLM()
    with pytest.MonkeyPatch.context() as patch:
        _offline(patch)
        # No update during the run: every meta charge is the final summary's.
        runner = _run_with(
            patch,
            tmp_path,
            stub,
            6,
            evo=dict(meta_rec_interval=1000),
            final_summary_timeout_s=0.5,
        )
    kinds = stub.breakdown()
    # It timed out mid-summary, after paying, not before the summary began.
    assert kinds.get("meta_final:step1", 0) > 0, kinds
    assert not {"meta_final:step2", "meta_final:step3"} & kinds.keys(), kinds
    assert runner.cost_limit_reached
    assert runner._fatal_error is None, runner._fatal_error
    marker = json.loads((results / UNRECORDED).read_text(encoding="utf-8"))
    assert marker["node_kind"] == "final_meta_summary", marker
    assert marker.get("program"), marker
    cost = run_cost(results)
    assert not cost.known, cost
