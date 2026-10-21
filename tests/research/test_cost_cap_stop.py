"""Where the real async runner stops under ``evo.max_api_costs``.

Drives ``ShinkaEvolveRunner.run_async`` end to end on a trivial task, with the
LLM replaced at its lowest layer (``shinka.llm.llm.query_async``, which
``AsyncLLMClient.query`` awaits) by a stub that returns a valid full rewrite and
charges a fixed ``COST`` per call. No network, no embeddings, no meta / novelty
/ prompt evolution, so the stub is the only thing that charges.

Stop bound. Proposal cost lands in ``total_api_cost`` when sampling finishes
(``async_runner.py:2968-2974``). The coordinator stops starting proposals once
``total + active * avg_cost >= cap`` (``_get_committed_cost``) but starts each
batch whole, and a batch never exceeds ``P = min(max_evaluation_jobs,
max_proposal_jobs)`` in-flight proposals. With a fixed cost the estimate is
exact or high, so the last batch starts at committed <= cap - COST and the run
ends at most ``(P - 1) * COST`` over the cap. ``avg_cost`` lives in memory and
is 0 again after a resume, so the first resumed batch is unestimated; its P
proposals are also within ``(P - 1) * COST`` of any cap above the base. A high
estimate can only start fewer proposals (see the backstop note in ``_run``).
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
from types import SimpleNamespace

import pytest

import shinka.core.async_runner as async_runner
import shinka.llm.llm as llm_module
from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
from shinka.llm.providers import QueryResult
from shinka.pricing.catalog import get_catalog
from shinka.research.rundb import run_cost

COST = 0.125  # exact in binary, so sums compare exactly
MODEL = "gpt-5-mini"
NUM_GENERATIONS = 1000  # never the binding limit
RUN_TIMEOUT_S = 90
STALL_TIMEOUT_S = 3.0

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
    """Stands in for ``shinka.llm.llm.query_async``: one call, one ``COST``."""

    def __init__(self, score_of=lambda n: float(n)):
        self.calls = 0
        self.score_of = score_of

    async def __call__(
        self,
        msg,
        system_msg,
        msg_history,
        output_model=None,
        model_posteriors=None,
        **kwargs,
    ):
        self.calls += 1
        n = self.calls
        await asyncio.sleep(0.2)  # keep proposals in flight together
        content = (
            f"<NAME>stub_{n}</NAME>\n<DESCRIPTION>stub {n}</DESCRIPTION>\n"
            "```python\n# EVOLVE-BLOCK-START\n"
            f"def score():\n    return {self.score_of(n)!r}  # call {n}\n"
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
            cost=COST,
            model_posteriors=model_posteriors,
        )


@pytest.fixture
def offline(monkeypatch):
    """No pricing refresh, a dummy key for the env check, loopback-only sockets;
    records the resume base the runner loads and any stall it recovers from."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-offline")
    monkeypatch.setattr(async_runner, "refresh_model_catalog", get_catalog)
    real_connect = socket.socket.connect

    def loopback_only(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise OSError(f"network blocked in test: {address!r}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    probe = SimpleNamespace(loaded=[], stalls=[])
    real_total = ShinkaEvolveRunner._get_total_api_costs

    async def recording_total(self):
        value = await real_total(self)
        probe.loaded.append(value)
        return value

    real_stuck = ShinkaEvolveRunner._handle_stuck_system

    async def recording_stuck(self):
        last = self.last_progress_time
        if last is not None and time.time() - last > self.stuck_detection_timeout:
            probe.stalls.append(
                dict(
                    total=self.total_api_cost,
                    committed=self._get_committed_cost(),
                    cost_limit_reached=self.cost_limit_reached,
                    cap=self.evo_config.max_api_costs,
                )
            )
        return await real_stuck(self)

    monkeypatch.setattr(ShinkaEvolveRunner, "_get_total_api_costs", recording_total)
    monkeypatch.setattr(ShinkaEvolveRunner, "_handle_stuck_system", recording_stuck)
    return probe


def _task(root: Path) -> tuple[Path, Path]:
    task = root / "task"
    task.mkdir(parents=True, exist_ok=True)
    (task / "initial.py").write_text(INITIAL, encoding="utf-8")
    (task / "evaluate.py").write_text(EVALUATE, encoding="utf-8")
    return task / "initial.py", task / "evaluate.py"


def _run(
    monkeypatch, probe, root: Path, cap: float, stub: StubLLM, **db
) -> ShinkaEvolveRunner:
    probe.stalls.clear()
    initial, evaluate = _task(root)
    monkeypatch.setattr(llm_module, "query_async", stub)
    runner = ShinkaEvolveRunner(
        evo_config=EvolutionConfig(
            patch_types=["full"],
            patch_type_probs=[1.0],
            num_generations=NUM_GENERATIONS,
            llm_models=[MODEL],
            llm_dynamic_selection=None,
            meta_rec_interval=None,
            embedding_model=None,
            novelty_llm_models=None,
            evolve_prompts=False,
            init_program_path=str(initial),
            results_dir=str(root / "results"),
            max_api_costs=cap,
        ),
        job_config=LocalJobConfig(
            eval_program_path=str(evaluate), python_executable=sys.executable
        ),
        db_config=DatabaseConfig(**db),
        verbose=False,
        max_evaluation_jobs=4,
        max_proposal_jobs=4,
        max_db_workers=2,
    )
    # Liveness backstop, not a cap knob: a proposal's cost reaches
    # total_api_cost (async_runner.py:2971) before its task leaves
    # active_proposal_tasks (:2696, after an awaited lock unlink), so a check in
    # that window counts it twice and can latch cost_limit_reached below the
    # cap; the run then idles until stuck recovery (60 s by default) starts one
    # proposal. Recorded in probe.stalls.
    runner.stuck_detection_timeout = STALL_TIMEOUT_S
    asyncio.run(asyncio.wait_for(runner.run_async(), RUN_TIMEOUT_S))
    return runner


def _in_flight(runner: ShinkaEvolveRunner) -> int:
    return min(runner.max_evaluation_jobs, runner.max_proposal_jobs)


def _copy_costs(results_dir: Path) -> list[float]:
    """api_costs carried by island-copy rows (initial copies, spawned islands)."""
    with closing(sqlite3.connect(results_dir / "programs.sqlite")) as conn:
        rows = conn.execute("SELECT metadata FROM programs").fetchall()
    costs = []
    for (raw,) in rows:
        md = json.loads(raw or "{}")
        if md.get("_is_island_copy") or md.get("_spawned_island"):
            costs.append(float(md.get("api_costs") or 0.0))
    return costs


def _report(label: str, **values) -> None:
    print(f"\n[cap-stop] {label}: " + ", ".join(f"{k}={v}" for k, v in values.items()))


def test_fresh_then_resumed_runs_stop_within_one_batch_of_cap(
    monkeypatch, tmp_path, offline
):
    results = tmp_path / "results"

    # (a) fresh directory, cap C
    cap = 1.0
    stub = StubLLM()
    runner = _run(monkeypatch, offline, tmp_path, cap, stub)
    p = _in_flight(runner)
    spend = runner.total_api_cost
    recorded = run_cost(results).recorded_usd
    _report(
        "fresh",
        cap=cap,
        P=p,
        proposals=stub.calls,
        spend=spend,
        recorded=recorded,
        overshoot=spend - cap,
        completed=runner.completed_generations,
        stalls=offline.stalls,
    )
    assert offline.loaded == [0.0]  # a fresh directory has no earlier spend
    assert runner.cost_limit_reached
    assert runner.completed_generations < NUM_GENERATIONS
    assert spend == stub.calls * COST == recorded
    assert cap <= spend <= cap + (p - 1) * COST

    # (b) resume with the research layer's injection: earlier spend + C2
    for c2 in (0.75, COST):  # C2 >= P*COST, then C2 < P*COST
        cost_before = run_cost(results).recorded_usd
        stops_at = cost_before + c2
        stub = StubLLM(score_of=lambda n, base=stub.calls + 100: float(base + n))
        runner = _run(monkeypatch, offline, tmp_path, stops_at, stub)
        p = _in_flight(runner)
        loaded = offline.loaded[-1]
        spend = runner.total_api_cost
        _report(
            f"resume C2={c2}",
            cost_before=cost_before,
            runner_loaded=loaded,
            stops_at=stops_at,
            P=p,
            new_proposals=stub.calls,
            spend=spend,
            new_spend=spend - loaded,
            overshoot=spend - stops_at,
            stalls=offline.stalls,
        )
        assert loaded == cost_before  # base neither ignored nor inflated
        assert runner.cost_limit_reached
        assert spend == loaded + stub.calls * COST == run_cost(results).recorded_usd
        assert stops_at <= spend <= stops_at + (p - 1) * COST
        assert stub.calls >= 1  # did not stop at once


def test_resumed_grant_gets_its_cap_when_island_copies_carry_cost(
    monkeypatch, tmp_path, offline
):
    """Dynamic island spawning copies a paid program (``_spawned_island``).
    The copy repeats its source's cost fields but is not a charge, so the
    runner's resume base (``_get_total_api_costs``) and ``rundb.run_cost``
    both skip it: the base is exactly ``cost_before`` and the grant gets its
    whole cap."""
    results = tmp_path / "results"
    islands = dict(
        num_islands=2,
        enable_dynamic_islands=True,
        stagnation_threshold=2,
        island_spawn_strategy="best",
    )
    # Gen 1 is the best ever, so every later generation is stagnant.
    stub = StubLLM(score_of=lambda n: 1.0 if n == 1 else 0.5)
    _run(monkeypatch, offline, tmp_path, 1.0, stub, **islands)
    copies = _copy_costs(results)
    paid_copies = [c for c in copies if c > 0]
    assert paid_copies, f"no island copy carried cost; copies={copies}"

    cost_before = run_cost(results).recorded_usd
    c2 = 0.75
    stops_at = cost_before + c2
    stub = StubLLM(score_of=lambda n: 0.5)
    runner = _run(monkeypatch, offline, tmp_path, stops_at, stub, **islands)
    loaded = offline.loaded[-1]
    _report(
        "island resume",
        cost_before=cost_before,
        runner_loaded=loaded,
        paid_copy_rows=len(paid_copies),
        copy_cost=sum(paid_copies),
        stops_at=stops_at,
        new_proposals=stub.calls,
        new_spend=runner.total_api_cost - loaded,
        spend=runner.total_api_cost,
        stalls=offline.stalls,
    )
    assert loaded == cost_before, (loaded, cost_before, sum(paid_copies))
    assert runner.cost_limit_reached
    assert (
        stops_at <= runner.total_api_cost <= stops_at + (_in_flight(runner) - 1) * COST
    )
