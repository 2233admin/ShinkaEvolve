"""DEMO · 演示数据，非真实研究 — deterministic demo workspace generator.

Writes a clearly labelled research workspace (four directions in different
states plus a shared library) and an ordinary-runs root so the web UI can be
judged on rich content. Nothing here runs research, calls a model, computes a
real embedding or touches the network: every number is synthesised from a
fixed seed and every artefact says so.

Files are written through the same storage code a real run uses
(``Notebook``, ``prepare_data``/``scaffold_experiment``, ``append_log``,
``library``, ``ProgramDatabase``, ``SystemPromptDatabase``,
``with_pipeline_timing``). A scoped demo clock stands in for wall-clock time
inside those modules so a rerun into the same path is byte-for-byte equal.

Usage::

    uv run --offline --no-sync python shinka/research/webui/demo/build_demo_workspace.py \
        --workspace <empty dir> --runs <empty dir>
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import heapq
import json
import math
import random
import re
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from shinka.core.pipeline_timing import with_pipeline_timing
from shinka.database import (
    DatabaseConfig,
    Program,
    ProgramDatabase,
    SystemPrompt,
    SystemPromptConfig,
    SystemPromptDatabase,
)
from shinka.database import dbase as dbase_module
from shinka.database import islands as islands_module
from shinka.database.islands import (
    CombinedIslandManager,
    DefaultIslandAssignmentStrategy,
)
from shinka.llm.providers.result import QueryResult
from shinka.research import library, notebook, rundb
from shinka.research.notebook import (
    Notebook,
    Status,
    Verdict,
    append_log,
    experiment_dir,
)
from shinka.research.quant import QuantConfig, split_index
from shinka.research.scaffold import prepare_data, scaffold_experiment
from shinka.research.skeptic import (
    Thresholds,
    decide,
    direction_trials,
    expected_max_sharpe,
    render_report,
)

LABEL = "DEMO · 演示数据，非真实研究"
# Stored only in private_metrics; the API must never return it.
SENTINEL = "DEMO_PRIVATE_SENTINEL_20261019_MUST_STAY_HIDDEN"
SEED = 20261019
PPY = 252.0


def ts(text: str) -> float:
    """UTC timestamp of an ISO-like ``YYYY-MM-DD HH:MM`` string."""
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def iso(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Demo clock: replaces wall-clock time inside the storage modules.
# ---------------------------------------------------------------------------


class DemoClock:
    """Duck-types the ``time`` module for ``shinka.database``; other attributes
    (``sleep``, ``perf_counter``) still come from the real module."""

    def __init__(self) -> None:
        self.now = ts("2026-09-01 00:00")

    def time(self) -> float:
        return self.now

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)

    def at(self, when: str | float) -> "DemoClock":
        self.now = ts(when) if isinstance(when, str) else float(when)
        return self


@contextlib.contextmanager
def demo_clock() -> Iterator[DemoClock]:
    clock = DemoClock()
    saved = (dbase_module.time, islands_module.time, notebook._now, library._now)
    dbase_module.time = clock
    islands_module.time = clock
    notebook._now = lambda: iso(clock.now)
    library._now = lambda: iso(clock.now)
    try:
        yield clock
    finally:
        (dbase_module.time, islands_module.time, notebook._now, library._now) = saved


# ---------------------------------------------------------------------------
# Synthetic tasks: program text templates, metrics and failure modes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    kind: str
    title: str
    scale: float
    header: Tuple[str, ...]
    params: Dict[str, Tuple[float, float, float, bool]]
    signature: str
    body_head: Tuple[str, ...]
    features: Dict[str, Tuple[str, ...]]
    body_tail: Tuple[str, ...]
    failures: Tuple[str, ...]


QUANT = Task(
    kind="quant",
    title="volatility-scaled momentum strategy",
    scale=0.06,
    header=(
        '"""Volatility-scaled momentum strategy (demo program)."""',
        "",
        "import numpy as np",
        "import pandas as pd",
    ),
    params={
        "FAST": (12, 5, 30, True),
        "SLOW": (63, 40, 200, True),
        "VOL_WINDOW": (20, 10, 60, True),
        "VOL_TARGET": (0.15, 0.08, 0.30, False),
        "ENTRY_Z": (0.25, 0.0, 1.0, False),
        "REGIME_DAMP": (0.5, 0.0, 1.0, False),
        "BRAKE_LEVEL": (0.15, 0.05, 0.4, False),
        "SMOOTH": (3, 1, 10, True),
    },
    signature="def run_experiment(bars: pd.DataFrame) -> pd.Series:",
    body_head=(
        '    close = bars["close"]',
        "    returns = close.pct_change().fillna(0.0)",
        "    trend = close.rolling(FAST).mean() / close.rolling(SLOW).mean() - 1.0",
        "    vol = returns.rolling(VOL_WINDOW).std() * np.sqrt(252)",
        "    edge = ENTRY_Z * vol / np.sqrt(252)",
        "    signal = np.sign(trend).where(trend.abs() > edge, 0.0)",
    ),
    features={
        "regime_filter": (
            "    calm = vol < vol.rolling(250, min_periods=60).median()",
            "    signal = signal.where(calm, signal * REGIME_DAMP)",
        ),
        "trend_confirm": (
            "    signal = signal.where(np.sign(close.diff(SLOW)) == signal, 0.0)",
        ),
        "drawdown_brake": (
            "    drawdown = close / close.cummax() - 1.0",
            "    signal = signal.where(drawdown > -BRAKE_LEVEL, 0.0)",
        ),
        "volume_gate": (
            '    if "volume" in bars:',
            '        median_volume = bars["volume"].rolling(20).median()',
            '        signal = signal.where(bars["volume"] > 0.8 * median_volume, 0.0)',
        ),
        "smooth_entry": ("    signal = signal.rolling(SMOOTH, min_periods=1).mean()",),
        "mean_revert_highvol": (
            "    stretch = (close / close.rolling(10).mean() - 1.0) / edge.clip(lower=1e-4)",
            "    signal = signal.where(vol < vol.quantile(0.8), -np.sign(stretch))",
        ),
    },
    body_tail=(
        "    position = signal * (VOL_TARGET / vol.clip(lower=0.05))",
        "    return position.clip(-1.0, 1.0).fillna(0.0)",
    ),
    failures=(
        "look-ahead detected: positions on the first 1187 bars changed when "
        "later bars were removed",
        "run_experiment returned NaN positions after the warm-up window",
        "KeyError: 'volume' (column not present in the frozen data)",
        "evaluation timed out after 300 s",
        "positions outside [-1, 1] (max 1.84)",
    ),
)

PACKING = Task(
    kind="packing",
    title="circle packing heuristic (26 circles in the unit square)",
    scale=0.022,
    header=(
        '"""Ring-based circle packing in the unit square (demo program)."""',
        "",
        "import numpy as np",
    ),
    params={
        "RING_COUNT": (3, 2, 5, True),
        "RING_GAP": (0.12, 0.05, 0.2, False),
        "JITTER": (0.02, 0.0, 0.08, False),
        "SHRINK": (0.995, 0.98, 0.9999, False),
        "ITERATIONS": (200, 50, 800, True),
    },
    signature="def construct_packing(n: int = 26) -> tuple[np.ndarray, np.ndarray]:",
    body_head=(
        "    rng = np.random.default_rng(0)",
        "    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)",
        "    rings = np.arange(n) % RING_COUNT",
        "    radius = 0.5 - RING_GAP * (rings + 1)",
        "    centers = 0.5 + np.c_[np.cos(angles), np.sin(angles)] * radius[:, None]",
        "    centers += rng.normal(0.0, JITTER, centers.shape)",
        "    radii = np.full(n, 0.5 / np.sqrt(n))",
    ),
    features={
        "corner_seeds": (
            "    centers[:4] = [[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]]",
        ),
        "edge_snap": (
            "    centers = np.clip(centers, radii[:, None], 1.0 - radii[:, None])",
        ),
        "pairwise_relax": (
            "    for _ in range(ITERATIONS):",
            "        gaps = np.linalg.norm(centers[:, None] - centers[None], axis=-1)",
            "        radii = np.minimum(radii, (gaps + np.eye(n)).min(axis=1) / 2.0)",
        ),
        "grow_radii": ("    radii = radii / SHRINK",),
        "hex_lattice": (
            "    rows = int(np.ceil(np.sqrt(n)))",
            "    offsets = (np.arange(n) // rows % 2) * 0.5 / rows",
            "    centers[:, 0] = (np.arange(n) % rows + 0.5) / rows + offsets",
        ),
    },
    body_tail=(
        "    radii = radii * SHRINK",
        "    return centers, radii",
    ),
    failures=(
        "circles 7 and 19 overlap by 0.0031",
        "circle 3 extends outside the unit square",
        "construct_packing returned 25 circles, expected 26",
        "evaluation timed out after 120 s",
    ),
)

BINPACK = Task(
    kind="binpack",
    title="online bin packing heuristic",
    scale=0.006,
    header=('"""First-fit-decreasing bin packing heuristic (demo program)."""',),
    params={
        "SORT_POWER": (1.0, 0.5, 2.0, False),
        "SLACK": (0.02, 0.0, 0.1, False),
        "LOOKAHEAD": (2, 0, 6, True),
        "SPLIT_THRESHOLD": (0.6, 0.3, 0.9, False),
    },
    signature="def pack(items: list[float], capacity: float = 1.0) -> list[list[float]]:",
    body_head=(
        "    order = sorted(items, key=lambda x: -(x**SORT_POWER))",
        "    bins: list[list[float]] = []",
        "    loads: list[float] = []",
    ),
    features={
        "large_first": (
            "    large = [x for x in order if x > SPLIT_THRESHOLD]",
            "    order = large + [x for x in order if x <= SPLIT_THRESHOLD]",
        ),
        "pair_lookahead": (
            "    order = order[:LOOKAHEAD] + sorted(order[LOOKAHEAD:], reverse=True)",
        ),
        "tight_slack": ("    capacity = capacity - SLACK / 2.0",),
        "best_fit": (
            "    def room(i: int) -> float:",
            "        return capacity - loads[i]",
        ),
    },
    body_tail=(
        "    for item in order:",
        "        for i, load in enumerate(loads):",
        "            if load + item <= capacity - SLACK:",
        "                bins[i].append(item)",
        "                loads[i] += item",
        "                break",
        "        else:",
        "            bins.append([item])",
        "            loads.append(item)",
        "    return bins",
    ),
    failures=(
        "bin 12 over capacity (1.014)",
        "item 0.731 was dropped",
        "pack() raised IndexError: list index out of range",
    ),
)


def render_code(task: Task, params: Dict[str, float], features: List[str]) -> str:
    lines = [
        f"# {LABEL} — synthetic program text for UI demos; never executed or scored.",
        *task.header,
        "",
        "# EVOLVE-BLOCK-START",
    ]
    for name, (_, _, _, is_int) in task.params.items():
        value = params[name]
        lines.append(f"{name} = {int(value) if is_int else round(float(value), 4)}")
    lines += ["", "", task.signature, f"    # {LABEL}", *task.body_head]
    for name, snippet in task.features.items():
        if name in features:
            lines += snippet
    lines += [*task.body_tail, "# EVOLVE-BLOCK-END", ""]
    return "\n".join(lines)


def public_metrics(
    task: Task, rng: random.Random, score: float, params: Dict[str, float]
) -> Dict[str, float]:
    if task.kind == "quant":
        return {
            "is_sharpe_std": round(abs(rng.gauss(0.28, 0.08)), 4),
            "is_annual_return": round(0.085 * score + rng.gauss(0.0, 0.012), 4),
            "is_max_drawdown": round(
                min(-0.03, max(-0.6, -0.27 + 0.07 * score - abs(rng.gauss(0, 0.03)))),
                4,
            ),
            "is_turnover": round(0.3 / params["SMOOTH"] + rng.uniform(0.01, 0.06), 4),
        }
    if task.kind == "packing":
        return {
            "num_circles": 26,
            "min_gap": round(abs(rng.gauss(0.002, 0.0015)), 5),
            "validity_margin": round(rng.uniform(0.0, 0.004), 5),
            "runtime_s": round(params["ITERATIONS"] / 400 + rng.uniform(0.1, 0.6), 3),
        }
    return {
        "bins_used": int(round(100 / max(score, 0.5))),
        "overflow_items": 0,
        "runtime_ms": round(rng.uniform(4.0, 30.0) + params["LOOKAHEAD"] * 3, 2),
    }


def private_metrics(task: Task, rng: random.Random, score: float) -> Dict[str, Any]:
    if task.kind == "quant":
        held = {
            "oos_sharpe": round(score * rng.uniform(0.2, 0.9) + rng.gauss(0, 0.2), 4),
            "oos_annual_return": round(rng.gauss(0.04, 0.05), 4),
            "oos_max_drawdown": round(-abs(rng.gauss(0.18, 0.06)), 4),
        }
    else:
        held = {"holdout_score": round(score * rng.uniform(0.95, 1.0), 5)}
    return {**held, "demo_private_sentinel": SENTINEL}


def text_feedback(task: Task, score: float, pub: Dict[str, float]) -> str:
    if task.kind == "quant":
        folds = [score + d for d in (-0.07, 0.02, 0.05)]
        return (
            f"{LABEL}. In-sample fold Sharpe: {', '.join(f'{s:.2f}' for s in folds)} "
            f"(mean {score:.2f}). Annual return {pub['is_annual_return']:.1%}, "
            f"max drawdown {pub['is_max_drawdown']:.1%}, "
            f"mean turnover {pub['is_turnover']:.3f} per bar, costs 5.0 bps."
        )
    if task.kind == "packing":
        return (
            f"{LABEL}. Valid packing of 26 circles, sum of radii {score:.4f}, "
            f"smallest gap {pub['min_gap']:.4f}."
        )
    return f"{LABEL}. Fill ratio {score:.4f} using {pub['bins_used']} bins."


# ---------------------------------------------------------------------------
# Synthetic mutation models.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Model:
    name: str
    skill: float
    spread: float
    cost: float
    latency: Tuple[float, float]


MODELS = (
    Model("demo/model-aurora", 1.0, 1.0, 0.040, (35.0, 80.0)),
    Model("demo/model-basalt", 0.55, 0.7, 0.011, (14.0, 38.0)),
    Model("demo/model-cirrus", 1.2, 1.6, 0.075, (50.0, 120.0)),
)
# A headless CLI agent with no price list. Core telemetry leaves its api_costs
# out and counts its programs as pricing-unknown (shinka/database/dbase.py), so
# a run that uses it has an unknown cost.
UNPRICED_MODEL = "headless/demo-agent"
PATCH_WEIGHTS = {"diff": 0.52, "full": 0.2, "cross": 0.14, "fix": 0.14}


# ---------------------------------------------------------------------------
# Run simulation (pure Python, fixed seed) -> replayable events.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    key: str
    task: Task
    n_programs: int
    n_generations: int
    n_islands: int
    seed_score: float
    ceiling: float
    start: str
    fail_rate: float = 0.08
    fail_from: Optional[int] = None
    fail_spike: float = 0.7
    n_prompts: int = 8
    meta: bool = True
    plots: bool = True
    workers: Tuple[int, int, int] = (4, 3, 2)
    # The run's first model is the unpriced headless agent, so the run's cost is
    # unknown and the UI's unknown-cost state stays testable. Ordinary runs only:
    # `shinka_research run` refuses headless models for research experiments.
    cost_unknown: bool = False


@dataclass
class Record:
    order: int
    id: str
    generation: int
    parent_id: Optional[str]
    island: int = -1
    params: Dict[str, float] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    code: str = ""
    code_diff: Optional[str] = None
    score: float = 0.0
    correct: bool = True
    intended: float = 0.0  # score the evaluator would have given; kept on failure
    error: str = ""
    patch_type: str = "init"
    model: Optional[str] = None
    cluster: int = 0
    embedding: List[float] = field(default_factory=list)
    pca2: List[float] = field(default_factory=list)
    pca3: List[float] = field(default_factory=list)
    timestamp: float = 0.0
    finished: float = 0.0
    archive_ids: List[str] = field(default_factory=list)
    top_k_ids: List[str] = field(default_factory=list)
    prompt_id: Optional[str] = None
    public: Dict[str, Any] = field(default_factory=dict)
    private: Dict[str, Any] = field(default_factory=dict)
    feedback: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptRecord:
    id: str
    index: int
    parent_id: Optional[str]
    program_generation: int
    timestamp: float
    focus: str
    model: str
    cost: float
    program_ids: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    percentiles: List[float] = field(default_factory=list)
    program_count: int = 0


@dataclass
class Simulation:
    spec: RunSpec
    initial_code: str
    events: List[Tuple[str, Any]]
    records: List[Record]
    prompts: List[PromptRecord]
    meta_files: Dict[int, str]
    migrations: int


def det_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def allocate(total: int, weights: List[float]) -> List[int]:
    """Integer counts >= 1 proportional to weights, summing to ``total``."""
    raw = [w / sum(weights) * (total - len(weights)) for w in weights]
    counts = [1 + int(r) for r in raw]
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - int(raw[i]), -i))
    for i in reversed(order[len(order) - (total - sum(counts)) :]):
        counts[i] += 1
    return counts


def mutate(
    rng: random.Random,
    task: Task,
    parent: Record,
    patch: str,
    inspiration: Optional[Record],
) -> Tuple[Dict[str, float], List[str], str, str]:
    params = dict(parent.params)
    features = list(parent.features)
    names = list(task.params)
    note = ""
    if patch == "cross" and inspiration is not None:
        for name in names:
            if rng.random() < 0.5:
                params[name] = inspiration.params[name]
        borrowed = [f for f in inspiration.features if f not in features]
        if borrowed:
            features.append(rng.choice(borrowed))
        note = f"crossover with island {inspiration.island} program"
    count = {"diff": rng.randint(1, 2), "full": 3, "cross": 1, "fix": 1}[patch]
    tuned = rng.sample(names, count)
    for name in tuned:
        default, lo, hi, is_int = task.params[name]
        value = params[name] * (1.0 + rng.gauss(0.0, 0.22)) + rng.gauss(0.0, 0.02)
        value = min(hi, max(lo, value))
        params[name] = int(round(value)) if is_int else round(value, 4)
    toggled = ""
    if patch == "diff" and rng.random() < 0.35:
        choice = rng.choice(list(task.features))
        if choice in features:
            features.remove(choice)
            toggled = f"drop_{choice}"
        else:
            features.append(choice)
            toggled = f"add_{choice}"
        note = note or toggled.replace("_", " ", 1)
    if patch == "full":
        features = rng.sample(list(task.features), rng.randint(1, 3))
        note = f"rewrite around {', '.join(sorted(features))}"
    features = [f for f in task.features if f in features]
    if patch == "cross":
        name = f"crossover_{tuned[0].lower()}"
    elif patch == "full":
        name = f"rewrite_{features[0]}"
    elif patch == "fix":
        name = "fix_" + "_".join(re.findall(r"[a-z]+", parent.error.lower())[:3])
        note = f"repair the parent's evaluation failure ({parent.error})"
    else:
        name = toggled or f"tune_{tuned[0].lower()}"
    note = note or f"retune {', '.join(t.lower() for t in tuned)}"
    return params, features, name, note


def unified_diff(old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="a/main.py",
            tofile="b/main.py",
        )
    )


def posterior(stats: Dict[str, List[int]], models) -> Dict[str, float]:
    rates = [(stats[m.name][0] + 1) / (stats[m.name][1] + 2) for m in models]
    weights = [math.exp(4.0 * r) for r in rates]
    total = sum(weights)
    return {
        m.name: round(0.1 / 3 + 0.9 * w / total, 4) for m, w in zip(models, weights)
    }


class Simulator:
    def __init__(self, spec: RunSpec):
        self.spec = spec
        self.task = spec.task
        self.rng = random.Random(f"{SEED}:{spec.key}")
        self.nrng = np.random.default_rng(
            [SEED, sum(ord(c) * (i + 1) for i, c in enumerate(spec.key))]
        )
        self.records: List[Record] = []
        self.inserted: List[Record] = []
        self.by_id: Dict[str, Record] = {}
        self.island_now: Dict[str, int] = {}
        self.events: List[Tuple[str, Any]] = []
        self.models = (
            (replace(MODELS[0], name=UNPRICED_MODEL), *MODELS[1:])
            if spec.cost_unknown
            else MODELS
        )
        self.model_stats = {m.name: [0, 0] for m in self.models}
        self.posteriors: Dict[int, Dict[str, float]] = {}
        self.prompts: List[PromptRecord] = []
        self.pending: List[Tuple[float, int, Record]] = []
        self.migration_gens = [g for g in (10, 20, 30) if g < spec.n_generations]
        self.migrations = 0
        self.correct_scores: List[float] = []
        self.fixed: set = set()
        k = 5
        self.centres = self.nrng.normal(0.0, 2.0, (k, 16))
        self.drift = self.nrng.normal(0.0, 1.0, (k, 16))
        self.breakthrough = (
            self.rng.randrange(spec.n_islands),
            int(spec.n_generations * self.rng.uniform(0.5, 0.7)),
        )
        self.breakthrough_done = False

    # -- helpers ----------------------------------------------------------

    def embed(self, cluster: int, generation: int) -> List[float]:
        vec = (
            self.centres[cluster]
            + 0.45 * self.nrng.normal(0.0, 1.0, 16)
            + 0.015 * generation * self.drift[cluster]
        )
        return [round(float(v), 5) for v in vec]

    def child_score(self, parent: Record, generation: int, model: Model, patch: str):
        spec, scale = self.spec, self.task.scale
        span = spec.ceiling - spec.seed_score
        headroom = max(0.0, (spec.ceiling - parent.score) / span)
        phase = max(0.0, 1.0 - generation / spec.n_generations) ** 0.8
        drift = scale * model.skill * (2.4 * headroom * phase - 0.3)
        spread = scale * model.spread * {"diff": 1.0, "full": 1.8, "cross": 1.3}[patch]
        delta = drift + self.rng.gauss(0.0, spread)
        if self.rng.random() < 0.14:
            delta -= abs(self.rng.gauss(0.0, 3.0 * scale))
        island, gen = self.breakthrough
        if (
            not self.breakthrough_done
            and generation >= gen
            and self.island_now[parent.id] == island
        ):
            self.breakthrough_done = True
            delta += 5.0 * scale
        return round(parent.score + delta, 5)

    def schedule(self, slots: List[float], ready: float, duration: float):
        slot = min(range(len(slots)), key=lambda i: (slots[i], i))
        start = max(ready, slots[slot])
        slots[slot] = start + duration
        return slot + 1, start, start + duration

    # -- insertion (DB order) ---------------------------------------------

    def flush(self, until: float) -> None:
        while self.pending and self.pending[0][0] <= until:
            _, _, rec = heapq.heappop(self.pending)
            self.insert(rec)

    def migrate(self, generation: int, at: float) -> None:
        moved = set()
        for source in range(self.spec.n_islands):
            members = [r for r in self.inserted if self.island_now[r.id] == source]
            eligible = [r for r in members if r.correct and r.generation > 0]
            if len(members) <= 1 or not eligible:
                continue
            elite = max(eligible, key=lambda r: (r.score, r.order))
            candidates = [r for r in eligible if r is not elite and r.id not in moved]
            count = min(len(candidates), max(1, int(len(members) * 0.05)))
            for rec in self.rng.sample(candidates, count):
                dest = self.rng.choice(
                    [i for i in range(self.spec.n_islands) if i != source]
                )
                moved.add(rec.id)
                self.island_now[rec.id] = dest
                self.events.append(
                    (
                        "migrate",
                        {
                            "id": rec.id,
                            "from": source,
                            "to": dest,
                            "generation": generation,
                            "at": round(at, 3),
                        },
                    )
                )
                self.migrations += 1

    def insert(self, rec: Record) -> None:
        while self.migration_gens and rec.generation >= self.migration_gens[0]:
            self.migrate(self.migration_gens.pop(0), rec.finished - 0.5)
        if rec.parent_id:
            rec.island = self.island_now[rec.parent_id]
        else:
            # Same rule as DefaultIslandAssignmentStrategy for correct seeds.
            seeded = {self.island_now[r.id] for r in self.inserted if r.correct}
            rec.island = min(i for i in range(self.spec.n_islands) if i not in seeded)
        self.island_now[rec.id] = rec.island
        self.inserted.append(rec)
        self.events.append(("insert", rec))
        if rec.correct:
            below = sum(s < rec.score for s in self.correct_scores)
            pct = below / len(self.correct_scores) if self.correct_scores else 0.5
            self.correct_scores.append(rec.score)
            if rec.prompt_id is not None:
                prompt = next(p for p in self.prompts if p.id == rec.prompt_id)
                prompt.scores.append(rec.score)
                prompt.percentiles.append(round(pct, 5))
        if rec.model is not None and rec.parent_id:
            stats = self.model_stats[rec.model]
            stats[1] += 1
            stats[0] += int(rec.correct and rec.score > self.by_id[rec.parent_id].score)

    # -- prompts ----------------------------------------------------------

    def prompt_for(self, generation: int, when: float) -> Optional[PromptRecord]:
        spec = self.spec
        if spec.n_prompts <= 0:
            return None
        due = [
            round(k * (spec.n_generations - 1) / spec.n_prompts)
            for k in range(spec.n_prompts)
        ]
        while len(self.prompts) < len(due) and due[len(self.prompts)] <= generation:
            index = len(self.prompts)
            parent = None
            if self.prompts:
                parent = max(
                    self.prompts,
                    key=lambda p: (
                        sum(p.percentiles) / max(1, len(p.percentiles)),
                        p.index,
                    ),
                )
            focus = PROMPT_FOCUS[index % len(PROMPT_FOCUS)]
            self.prompts.append(
                PromptRecord(
                    id=det_uuid(self.rng),
                    index=index,
                    parent_id=parent.id if parent else None,
                    program_generation=due[index],
                    timestamp=round(when - 30.0 if index else when, 3),
                    focus=focus,
                    model=self.rng.choice(MODELS).name,
                    cost=round(self.rng.uniform(0.004, 0.02), 5) if index else 0.0,
                )
            )
        weights = [
            sum(p.percentiles) / max(1, len(p.percentiles))
            + 0.25
            + (0.6 if p is self.prompts[-1] else 0.0)
            for p in self.prompts
        ]
        prompt = self.rng.choices(self.prompts, weights)[0]
        prompt.program_ids.append("")  # placeholder replaced by caller
        prompt.program_count += 1
        return prompt

    # -- main loop --------------------------------------------------------

    def run(self) -> Simulation:
        spec, task, rng = self.spec, self.task, self.rng
        n_sample, n_eval, n_post = spec.workers
        start = ts(spec.start)
        sampling = [start] * n_sample
        evaluating = [start] * n_eval
        posting = [start] * n_post
        gens = [0] * spec.n_islands
        if spec.n_programs > spec.n_islands:
            weights = [rng.uniform(0.6, 1.4) for _ in range(spec.n_generations - 1)]
            for g, count in enumerate(
                allocate(spec.n_programs - spec.n_islands, weights), start=1
            ):
                gens += [g] * count
        defaults = {name: v[0] for name, v in task.params.items()}
        initial_code = render_code(task, defaults, [])
        release = start
        for order, generation in enumerate(gens):
            if generation > 0 and release == start:
                self.flush(float("inf"))
                release = max(r.finished for r in self.inserted) + 5.0
            ready = start if generation == 0 else release
            slot_free = min(sampling)
            proposal_at = max(ready, slot_free)
            self.flush(proposal_at)
            rec = Record(
                order=order, id=det_uuid(rng), generation=generation, parent_id=None
            )
            prompt = self.prompt_for(generation, proposal_at)
            if prompt is not None:
                prompt.program_ids[-1] = rec.id
                rec.prompt_id = prompt.id
            if generation == 0:
                self.seed(rec, defaults, initial_code, order)
                model, sample_dur = None, 0.5
                patch, desc = "init", "initial program from the experiment seed"
            else:
                if generation not in self.posteriors:
                    self.posteriors[generation] = posterior(
                        self.model_stats, self.models
                    )
                post = self.posteriors[generation]
                model = rng.choices(self.models, [post[m.name] for m in self.models])[0]
                patch = rng.choices(list(PATCH_WEIGHTS), list(PATCH_WEIGHTS.values()))[
                    0
                ]
                desc = self.propose(rec, model, patch)
                sample_dur = rng.uniform(*model.latency)
            fail_p = spec.fail_rate
            if spec.fail_from is not None and generation >= spec.fail_from:
                fail_p = spec.fail_spike
            rec.correct = generation == 0 or rng.random() >= fail_p
            eval_dur = (
                rng.uniform(40.0, 160.0)
                if rec.correct
                else rng.choice([rng.uniform(5.0, 30.0), 300.0])
            )
            busy_sampling = sum(t > proposal_at for t in sampling)
            s_worker, s_start, s_end = self.schedule(sampling, proposal_at, sample_dur)
            busy_eval = sum(t > s_end for t in evaluating)
            e_worker, e_start, e_end = self.schedule(evaluating, s_end, eval_dur)
            busy_post = sum(t > e_end for t in posting)
            p_worker, p_start, p_end = self.schedule(
                posting, e_end, rng.uniform(0.8, 4.0)
            )
            rec.timestamp = round(p_start + 0.2, 3)
            rec.finished = round(p_end, 3)
            self.finish(rec, model, desc, eval_dur)
            rec.metadata = with_pipeline_timing(
                {
                    **rec.metadata,
                    "results_missing": False,
                    "safe_processing": True,
                    "source_job_id": f"demo-job-{order:04d}",
                    "source_generation": generation,
                    "timeline_lane_mode": "pool_slots",
                    "sampling_worker_id": s_worker,
                    "evaluation_worker_id": e_worker,
                    "postprocess_worker_id": p_worker,
                    "active_proposals_at_start": busy_sampling + 1,
                    "running_eval_jobs_at_submit": busy_eval,
                    "db_workers_in_use_at_postprocess_start": busy_post + 1,
                    "sampling_worker_capacity": n_sample,
                    "evaluation_worker_capacity": n_eval,
                    "postprocess_worker_capacity": n_post,
                },
                pipeline_started_at=round(s_start, 3),
                sampling_started_at=round(s_start, 3),
                sampling_finished_at=round(s_end, 3),
                evaluation_started_at=round(e_start, 3),
                evaluation_finished_at=round(e_end, 3),
                postprocess_started_at=round(p_start, 3),
                postprocess_finished_at=round(p_end, 3),
            )
            self.records.append(rec)
            self.by_id[rec.id] = rec
            heapq.heappush(self.pending, (rec.finished, order, rec))
        self.flush(float("inf"))
        self.project()
        meta_files = self.meta() if spec.meta else {}
        return Simulation(
            spec,
            initial_code,
            self.events,
            self.records,
            self.prompts,
            meta_files,
            self.migrations,
        )

    def seed(self, rec: Record, defaults, initial_code: str, order: int) -> None:
        rec.island = order
        rec.params = dict(defaults)
        rec.code = initial_code
        rec.cluster = order % len(self.centres)
        if order:
            for name in self.rng.sample(list(self.task.params), 2):
                _, lo, hi, is_int = self.task.params[name]
                value = min(hi, max(lo, defaults[name] * self.rng.uniform(0.8, 1.2)))
                rec.params[name] = int(round(value)) if is_int else round(value, 4)
            rec.code = render_code(self.task, rec.params, [])
        rec.score = round(
            self.spec.seed_score + self.rng.gauss(0.0, 0.5 * self.task.scale), 5
        )

    def propose(self, rec: Record, model: Model, patch: str) -> str:
        rng = self.rng
        island = rng.randrange(self.spec.n_islands)
        pool = [
            r for r in self.inserted if r.correct and self.island_now[r.id] == island
        ]
        pool.sort(key=lambda r: (-r.score, r.order))
        if rng.random() < 0.15:
            parent = rng.choice(pool)
        else:
            parent = rng.choices(
                pool, [1.0 / (i + 1) ** 1.2 for i in range(len(pool))]
            )[0]
        correct = sorted(
            (r for r in self.inserted if r.correct), key=lambda r: (-r.score, r.order)
        )
        others = [r for r in correct if self.island_now[r.id] != island]
        inspiration = others[0] if others else None
        if patch == "cross" and inspiration is None:
            patch = "diff"
        broken = [
            r
            for r in self.inserted
            if not r.correct
            and r.generation < rec.generation
            and r.id not in self.fixed
        ]
        if patch == "fix" and not broken:
            patch = "diff"
        if patch == "fix":
            # Shinka's fix patches repair a program whose evaluation failed.
            parent = rng.choice(broken[-10:])
            self.fixed.add(parent.id)
        rec.parent_id = parent.id
        rec.patch_type = patch
        rec.archive_ids = (
            [inspiration.id]
            if patch == "cross" and inspiration
            else [rng.choice(correct[:10]).id]
        )
        rec.top_k_ids = [correct[0].id] if correct[0].id != parent.id else []
        rec.params, rec.features, name, note = mutate(
            rng, self.task, parent, patch, inspiration if patch == "cross" else None
        )
        rec.code = render_code(self.task, rec.params, rec.features)
        rec.code_diff = unified_diff(parent.code, rec.code)
        rec.model = model.name
        if patch == "full" and rng.random() < 0.45:
            rec.cluster = rng.randrange(len(self.centres))
        elif patch == "cross" and inspiration is not None and rng.random() < 0.5:
            rec.cluster = inspiration.cluster
        else:
            rec.cluster = parent.cluster if rng.random() < 0.9 else rng.randrange(5)
        if patch == "fix":
            noise = rng.gauss(0.0, 0.6 * self.task.scale)
            rec.score = round(parent.intended + noise, 5)
        else:
            rec.score = self.child_score(parent, rec.generation, model, patch)
        rec.metadata["patch_name"] = name
        return note

    def finish(self, rec: Record, model, desc: str, eval_dur: float):
        rng, task = self.rng, self.task
        rec.embedding = self.embed(rec.cluster, rec.generation)
        rec.intended = rec.score
        if rec.correct:
            rec.public = public_metrics(task, rng, rec.score, rec.params)
            rec.private = private_metrics(task, rng, rec.score)
            rec.feedback = text_feedback(task, rec.score, rec.public)
            stdout = (
                f"[{LABEL}] evaluating {rec.id[:8]} (generation {rec.generation})\n"
                f"[demo] combined_score={rec.score:.4f}\n[demo] correct\n"
            )
            stderr = ""
        else:
            error = rng.choice(task.failures)
            rec.error = error
            rec.score = 0.0
            rec.feedback = f"{LABEL}. Evaluation failed: {error}"
            stdout = (
                f"[{LABEL}] evaluating {rec.id[:8]} (generation {rec.generation})\n"
            )
            stderr = (
                "Traceback (most recent call last):\n"
                '  File "evaluate.py", line 37, in <module>\n'
                f"RuntimeError: {error}\n"
            )
        tokens_in = rng.randint(3500, 9000)
        tokens_out = rng.randint(400, 2600)
        api = 0.0 if model is None else round(model.cost * rng.uniform(0.6, 1.5), 5)
        unpriced = model is not None and model.name == UNPRICED_MODEL
        if unpriced:
            api = 0.0  # a headless agent reports no price
        checks = 0 if model is None else rng.choice([0, 0, 1, 1, 2])
        thought = (
            f"## {LABEL}\n\nSynthetic reasoning note (no model was called). "
            f"Plan: {desc}; keep the evaluator contract and the EVOLVE-BLOCK intact."
        )
        rng.random()  # one draw per program keeps the seeded sequence unchanged
        embed_cost = round(rng.uniform(0.00002, 0.00008), 7)
        meta: Dict[str, Any] = {
            "demo_label": LABEL,
            "api_costs": api,
            "patch_type": rec.patch_type,
            "patch_name": rec.metadata.get("patch_name", "initial_program"),
            "patch_description": f"{LABEL} — {desc}",
            "num_applied": 1,
            "error_attempt": None,
            "novelty_attempt": max(1, checks),
            "resample_attempt": 1 if rng.random() < 0.85 else 2,
            "patch_attempt": 1 if rng.random() < 0.8 else 2,
            "novelty_checks_performed": checks,
            "novelty_explanation": "" if not checks else f"{LABEL}: synthetic check",
            "max_similarity": None if not checks else round(rng.uniform(0.6, 0.97), 4),
            "system_prompt_id": rec.prompt_id,
            "embed_cost": embed_cost,
            "novelty_cost": round(checks * rng.uniform(0.001, 0.004), 6),
            # meta_cost appears only where a meta update charged the program
            # (meta()), as the runner writes it.
            "stdout_log": stdout,
            "stderr_log": stderr,
            "compute_time": round(eval_dur, 3),
            "thought": thought,
        }
        if unpriced:
            meta["headless_pricing_unknown"] = True
        if model is not None:
            meta["model_name"] = model.name
            meta["llm_result"] = QueryResult(
                content=(
                    f"<NAME>{meta['patch_name']}</NAME>\n"
                    f"<DESCRIPTION>{meta['patch_description']}</DESCRIPTION>\n"
                    "<DIFF>(see code_diff)</DIFF>"
                ),
                msg=f"{LABEL}: synthetic user message, never sent to a model.",
                system_msg=f"{LABEL}: synthetic system message.",
                new_msg_history=[],
                model_name=model.name,
                kwargs={"temperature": 0.75, "max_tokens": 8192},
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                cost=api,
                input_cost=round(api * 0.4, 6),
                output_cost=round(api * 0.6, 6),
                thought=thought,
                model_posteriors=self.posteriors[rec.generation],
            ).to_dict()
        rec.metadata = meta

    def project(self) -> None:
        """PCA of the synthetic embeddings, sign-fixed for determinism."""
        if not self.records:
            return
        matrix = np.array([r.embedding for r in self.records])
        centred = matrix - matrix.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        for row in vt:
            if row[np.argmax(np.abs(row))] < 0:
                row *= -1.0
        coords = centred @ vt[:3].T
        for rec, xyz in zip(self.records, coords):
            rec.pca3 = [round(float(v), 4) for v in xyz]
            rec.pca2 = rec.pca3[:2]

    def meta(self) -> Dict[int, str]:
        """``meta/meta_N.txt`` in the layout of ``MetaSummarizer.write_meta_output``:
        program summaries since the last update, a global insights scratchpad
        and numbered recommendations."""
        n = len(self.inserted)
        counts = sorted(
            {max(10, int(round(n * f / 10.0)) * 10) for f in (0.2, 0.45, 0.7, 0.95)}
        )
        note = f"> {LABEL}: written by the demo generator, not by a language model."
        files = {}
        previous, previous_best = 0, None
        for count in counts:
            if count > n:
                continue
            seen = self.inserted[:count]
            good = [r for r in seen if r.correct]
            best = max(good, key=lambda r: (r.score, -r.order))
            cost = round(self.rng.uniform(0.03, 0.09), 5)
            best.metadata["meta_cost"] = round(
                best.metadata.get("meta_cost", 0.0) + cost, 5
            )
            summaries = "\n\n".join(
                self.program_summary(r) for r in self.inserted[previous:count]
            )
            insights = self.insights(seen, good, best, previous_best)
            advice = self.recommendations(seen, good, best)
            files[count] = (
                "# INDIVIDUAL PROGRAM SUMMARIES\n\n"
                "The following are summaries of individual programs "
                "evaluated since the last meta update:\n\n"
                f"{note}\n\n{summaries}\n\n"
                "# GLOBAL INSIGHTS SCRATCHPAD\n\n"
                "The following are global insights about optimization "
                "approaches and their effectiveness:\n\n"
                f"{note}\n\n{insights}\n\n"
                "# META RECOMMENDATIONS\n\n"
                "The following are actionable recommendations for the next "
                "program generations:\n\n"
                f"{note}\n\n{advice}\n"
            )
            previous, previous_best = count, best
        return files

    def program_summary(self, rec: Record) -> str:
        desc = rec.metadata["patch_description"].split(" — ", 1)[-1]
        features = ", ".join(rec.features) or "the baseline signal only"
        if rec.correct:
            shown = list(rec.public.items())[:3]
            figures = ", ".join(
                f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in shown
            )
            performance = f"Combined score {rec.score:.4f} ({figures})."
        else:
            performance = "Evaluation failed, so the combined score is 0."
        feedback = rec.feedback.removeprefix(f"{LABEL}. ")
        return (
            f"**Program Name: {rec.metadata['patch_name']} "
            f"({rec.id[:8]}, gen {rec.generation}, island {rec.island})**\n"
            f"- **Implementation**: {desc[0].upper()}{desc[1:]}; uses {features}.\n"
            f"- **Performance**: {performance}\n"
            f"- **Feedback**: {feedback}"
        )

    def patch_rates(self, seen: List[Record]) -> Dict[str, Tuple[int, int]]:
        """(improved, total) per patch type; a child improves when it is correct
        and beats its parent's score. ``fix`` patches start from a failed parent,
        so they count as improved when the repaired program is correct."""
        rates: Dict[str, Tuple[int, int]] = {}
        for r in seen:
            if r.parent_id is None:
                continue
            up, total = rates.get(r.patch_type, (0, 0))
            better = r.correct and r.score > self.by_id[r.parent_id].score
            rates[r.patch_type] = (up + int(better), total + 1)
        return rates

    def insights(self, seen, good, best, previous_best) -> str:
        top = sorted(good, key=lambda r: (-r.score, r.order))[:5]
        shared = [f for f in self.task.features if all(f in r.features for r in top)]
        rates = self.patch_rates(seen)
        repaired = rates.pop("fix", None)
        ranked = sorted(rates, key=lambda p: (-rates[p][0] / rates[p][1], p))
        errors: Dict[str, int] = {}
        for r in seen:
            if not r.correct:
                errors[r.error] = errors.get(r.error, 0) + 1
        islands: Dict[int, float] = {}
        for r in good:
            isl = self.island_now[r.id]
            islands[isl] = max(islands.get(isl, r.score), r.score)
        models: Dict[str, int] = {}
        for r in seen:
            if r.model:
                models[r.model] = models.get(r.model, 0) + 1
        lines = ["## Successful Algorithmic Patterns"]
        lines += [
            f"- **{r.metadata['patch_name']}** ({r.id[:8]}, gen {r.generation}) "
            f"scores {r.score:.4f} with {', '.join(r.features) or 'no extra features'}"
            + (" — **current best**" if r is best else "")
            for r in top[:3]
        ]
        if ranked:
            p = ranked[0]
            lines.append(
                f"- `{p}` patches improve on their parent most often "
                f"({rates[p][0]} of {rates[p][1]})."
            )
        if repaired:
            lines.append(
                f"- `fix` patches repaired {repaired[0]} of {repaired[1]} failed "
                "parents."
            )
        lines += ["", "## Ineffective Approaches"]
        if ranked:
            p = ranked[-1]
            lines.append(
                f"- `{p}` patches improve least often ({rates[p][0]} of {rates[p][1]})."
            )
        lines += [
            f"- {count} failure(s): {error}"
            for error, count in sorted(errors.items(), key=lambda e: (-e[1], e[0]))[:3]
        ] or ["- No evaluation failures so far."]
        lines += ["", "## Implementation Insights"]
        lines.append(
            "- The top five programs share "
            + (", ".join(f"`{f}`" for f in shared) if shared else "no single feature")
            + "."
        )
        lines.append(
            "- Best program settings: "
            + ", ".join(f"{k}={v}" for k, v in sorted(best.params.items()))
            + "."
        )
        lines += ["", "## Performance Analysis"]
        lines.append(
            f"- {len(good)} of {len(seen)} programs are correct "
            f"({len(seen) - len(good)} failed)."
        )
        if previous_best is not None:
            lines.append(
                f"- Best score moved from {previous_best.score:.4f} to "
                f"{best.score:.4f} since the last meta update."
            )
        lines += [f"- Island {i}: best {s:.4f}" for i, s in sorted(islands.items())]
        lines += [f"- {m}: {c} proposals" for m, c in sorted(models.items())]
        return "\n".join(lines)

    def recommendations(self, seen, good, best) -> str:
        rates = self.patch_rates(seen)
        rates.pop("fix", None)
        ranked = sorted(rates, key=lambda p: (-rates[p][0] / rates[p][1], p))
        unused = [f for f in self.task.features if f not in best.features]
        repaired = {r.parent_id for r in seen if r.patch_type == "fix"}
        failed = sum(not r.correct and r.id not in repaired for r in seen)
        items = [
            f"Start from **{best.metadata['patch_name']}** ({best.id[:8]}, score "
            f"{best.score:.4f}) and retune one parameter at a time.",
            (
                f"Favour `{ranked[0]}` patches; they improve on their parent most "
                "often so far."
                if ranked
                else "Collect more child programs before favouring a patch type."
            ),
            (
                f"Try adding `{unused[0]}` to the best program as a small diff."
                if unused
                else "Try removing one feature from the best program to simplify it."
            ),
            (
                f"Repair the {failed} failed programs that no `fix` patch has "
                "repaired yet before exploring new ideas."
                if failed
                else "Keep the evaluator contract unchanged; no failure awaits a fix."
            ),
            "Keep every change inside the EVOLVE-BLOCK.",
        ]
        return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))


PROMPT_FOCUS = (
    "baseline instructions: improve the score while keeping the contract",
    "prefer small, testable parameter changes over rewrites",
    "explain the mechanism behind every change in one sentence",
    "penalise turnover and fragile thresholds",
    "borrow structure from the best program of another island",
    "avoid repeating failed edits listed in the feedback",
    "simplify: remove features that do not pay for themselves",
    "explore one bold structural idea per proposal",
    "stabilise: favour robust settings across all folds",
    "combine the two strongest feature families",
    "tighten edge cases that caused evaluation failures",
    "trade a little score for much lower complexity",
)


def simulate(spec: RunSpec) -> Simulation:
    return Simulator(spec).run()


# ---------------------------------------------------------------------------
# Writers (real storage code).
# ---------------------------------------------------------------------------


def write_plot(path: Path, title: str, kind: str, values: List[float], rng) -> None:
    fig = Figure(figsize=(4.0, 2.4), dpi=80)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    if kind == "curve":
        steps = np.cumsum([rng.gauss(0.0004 * values[0], 0.01) for _ in range(250)])
        ax.plot(np.exp(steps), color="#2b7a78", linewidth=1.2)
        ax.set_ylabel("equity (synthetic)")
    else:
        ax.bar(range(len(values)), values, color="#c8963e")
        ax.set_ylabel("score (synthetic)")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", metadata={"Software": None})


def write_run(results: Path, sim: Simulation, clock: DemoClock) -> Dict[str, Any]:
    spec = sim.spec
    results.mkdir(parents=True, exist_ok=True)
    config = DatabaseConfig(
        db_path=str(results / rundb.DB_FILENAME),
        num_islands=spec.n_islands,
        archive_size=40,
        migration_interval=10,
        migration_rate=0.0,  # demo migrations are replayed explicitly below
        enable_dynamic_islands=False,
    )
    clock.at(ts(spec.start))
    db = ProgramDatabase(config)
    db.island_manager = CombinedIslandManager(
        db.cursor,
        db.conn,
        config,
        assignment_strategy=DefaultIslandAssignmentStrategy(db.cursor, db.conn, config),
    )
    migrator = db.island_manager.migration_strategy
    for kind, payload in sim.events:
        if kind == "migrate":
            clock.at(payload["at"])
            migrator._migrate_program(
                payload["id"], payload["from"], payload["to"], payload["generation"]
            )
            db.conn.commit()
            continue
        rec: Record = payload
        program = Program(
            id=rec.id,
            code=rec.code,
            parent_id=rec.parent_id,
            archive_inspiration_ids=rec.archive_ids,
            top_k_inspiration_ids=rec.top_k_ids,
            generation=rec.generation,
            timestamp=rec.timestamp,
            code_diff=rec.code_diff,
            combined_score=rec.score,
            public_metrics=rec.public,
            private_metrics=rec.private,
            text_feedback=rec.feedback,
            correct=rec.correct,
            embedding=rec.embedding,
            embedding_pca_2d=rec.pca2,
            embedding_pca_3d=rec.pca3,
            embedding_cluster_id=rec.cluster,
            metadata=rec.metadata,
            system_prompt_id=rec.prompt_id,
        )
        clock.at(rec.finished)
        db.add(program, defer_maintenance=True)
        db.run_post_add_maintenance(program)
        if program.island_idx != rec.island:
            raise RuntimeError(
                f"{spec.key}: island {program.island_idx} != simulated {rec.island}"
            )
    db.save()
    db.close()

    if sim.prompts:
        prompt_db = SystemPromptDatabase(
            SystemPromptConfig(db_path=str(results / rundb.PROMPT_DB_FILENAME))
        )
        for p in sim.prompts:
            clock.at(p.timestamp)
            correct = len(p.scores)
            prompt_db.add(
                SystemPrompt(
                    id=p.id,
                    prompt_text=(
                        f"# {LABEL}\n# Synthetic system prompt v{p.index}; written by "
                        "the demo generator and never sent to a model.\n\n"
                        f"You are improving a {spec.task.title}.\n"
                        "Only edit code inside the EVOLVE-BLOCK.\n"
                        f"Focus for this version: {p.focus}.\n"
                    ),
                    name=f"demo_prompt_v{p.index}",
                    description=f"{LABEL} — {p.focus}",
                    parent_id=p.parent_id,
                    generation=p.index,
                    program_generation=p.program_generation,
                    patch_type="init"
                    if p.index == 0
                    else ("diff", "full")[p.index % 2],
                    timestamp=p.timestamp,
                    program_count=p.program_count,
                    correct_program_count=correct,
                    total_percentile=round(sum(p.percentiles), 5),
                    fitness=round(sum(p.percentiles) / correct, 5) if correct else 0.0,
                    program_scores=p.scores,
                    program_ids=p.program_ids,
                    metadata={
                        "demo_label": LABEL,
                        "llm": {"model_name": p.model, "cost": p.cost},
                    },
                )
            )
        prompt_db.save()
        prompt_db.close()

    for count, text in sim.meta_files.items():
        meta = results / "meta"
        meta.mkdir(exist_ok=True)
        (meta / f"meta_{count}.txt").write_text(text, encoding="utf-8")

    plot_gens: List[int] = []
    correct = [r for r in sim.records if r.correct]
    if spec.plots and correct:
        best = max(correct, key=lambda r: (r.score, -r.order))
        last = max(r.generation for r in sim.records)
        plot_gens = sorted({0, last // 2, best.generation, last})
        rng = random.Random(f"{SEED}:{spec.key}:plots")
        for gen in plot_gens:
            scores = [r.score for r in sim.records if r.generation == gen and r.correct]
            plots = results / f"gen_{gen}" / "results" / "plots"
            title = f"DEMO - synthetic, not real research (gen {gen})"
            write_plot(plots / "equity_curve.png", title, "curve", scores or [0.0], rng)
            write_plot(
                plots / "generation_scores.png", title, "bars", scores or [0.0], rng
            )

    return {
        "programs": len(sim.records),
        "generations": len({r.generation for r in sim.records}),
        "islands": spec.n_islands,
        "failed": sum(not r.correct for r in sim.records),
        "migrations": sim.migrations,
        "prompts": len(sim.prompts),
        "meta_files": sorted(sim.meta_files),
        "plot_generations": plot_gens,
        "best_score": max((r.score for r in correct), default=None),
    }


def write_empty_run(results: Path, clock: DemoClock, start: str) -> Dict[str, Any]:
    results.mkdir(parents=True, exist_ok=True)
    clock.at(start)
    db = ProgramDatabase(DatabaseConfig(db_path=str(results / rundb.DB_FILENAME)))
    db.save()
    db.close()
    return {"programs": 0}


def write_label(path: Path, text: str) -> None:
    path.write_text(
        f"# {LABEL}\n\n{text}\n\nGenerated by shinka/research/webui/demo/"
        "build_demo_workspace.py from a fixed seed. No research was run, no model "
        "was called, and no number here is a real result.\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Research directions.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkepticPlan:
    at: str
    oos_sharpe: float
    hold: float
    dsr: float
    positive_fraction: float
    pbo: float
    seed_oos: float
    verdict_at: Optional[str] = None


@dataclass(frozen=True)
class ExperimentPlan:
    scaffold_at: str
    run: RunSpec
    probes: int = 0
    finished: bool = True
    run_failed: bool = False


@dataclass(frozen=True)
class HypothesisPlan:
    title: str
    body: str
    added_at: str
    experiments: Tuple[ExperimentPlan, ...] = ()
    skeptic: Optional[SkepticPlan] = None


@dataclass(frozen=True)
class ConfirmPlan:
    at: str
    lockbox_sharpe: float
    lockbox_buy_hold: float
    dsr: float
    positive_fraction: float
    annual_return: float
    max_drawdown: float


@dataclass(frozen=True)
class DirectionPlan:
    dirname: str
    direction: str
    created: str
    budget_usd: float
    data_seed: int
    hypotheses: Tuple[HypothesisPlan, ...]
    confirm: Optional[ConfirmPlan] = None


def source_bars(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-02", periods=1600, name="timestamp")
    rets = rng.normal(0.0003, 0.011, len(index))
    close = 100.0 * np.exp(np.cumsum(rets))
    opens = close * np.exp(rng.normal(0.0, 0.004, len(index)))
    frame = pd.DataFrame(
        {
            "open": opens.round(4),
            "high": (np.maximum(opens, close) * 1.004).round(4),
            "low": (np.minimum(opens, close) * 0.996).round(4),
            "close": close.round(4),
            "volume": rng.integers(800_000, 3_000_000, len(index)),
        },
        index=index,
    )
    frame.to_csv(path)


def program_md(plan: DirectionPlan) -> str:
    return (
        "# Research program\n\n"
        f"> {LABEL}. Every number in this direction is synthetic.\n\n"
        f"## Direction\n{plan.direction}\n\n"
        "## Data\n- File: synthetic daily bars generated by the demo builder\n"
        "- Instrument, bar size, period: DEMO-INDEX, daily, 2018-2024 (synthetic)\n"
        "- Periods per year: 252\n\n"
        "## Evaluator settings\n- Costs: 5 bps per unit turnover\n"
        "- Lockbox fraction: 0.2\n"
        "- Validation fraction of the development data: 0.3, in-sample folds: 3\n\n"
        f"## Budget\n- USD per direction: {plan.budget_usd:.0f} (demo)\n\n"
        "## Autonomy\n- Autonomy: checkpoints\n\n"
        "## Verdict thresholds\n- supported: DSR >= 0.95, >= 2/3 of validation "
        "sub-periods positive, PBO <= 0.5, beats buy-and-hold\n"
        "- refuted: out-of-sample Sharpe <= 0\n- otherwise: inconclusive\n"
    )


def skeptic_report(
    root: Path, nb: Notebook, hid: str, eid: str, plan: SkepticPlan
) -> Dict[str, Any]:
    """Same keys as ``skeptic.challenge``; the trial count and noise bar are
    computed from the demo databases, the out-of-sample figures are synthetic."""
    exp_dir = experiment_dir(root, hid, eid)
    pid, code, is_score = rundb.best_program(exp_dir / "results")
    best_path = exp_dir / "best_program.py"
    best_path.write_text(code, encoding="utf-8")
    n_trials, var = direction_trials(root, nb, PPY)
    n_dev = nb.data.n_dev
    is_sharpe = round(is_score * 0.97, 4)
    return {
        "program_id": pid,
        "is_score": is_score,
        "is_sharpe": is_sharpe,
        "oos_sharpe": plan.oos_sharpe,
        "oos_bars": n_dev - split_index(n_dev, 0.3),
        "decay": plan.oos_sharpe / is_sharpe if is_sharpe else None,
        "n_trials": n_trials,
        "trial_sharpe_var_per_period": var,
        "expected_max_sharpe_annual": expected_max_sharpe(n_trials, var)
        * math.sqrt(PPY),
        "dsr": plan.dsr,
        "oos_positive_fraction": plan.positive_fraction,
        "pbo": plan.pbo,
        "baseline_seed_oos_sharpe": plan.seed_oos,
        "baseline_buy_hold_oos_sharpe": plan.hold,
        "best_program": str(best_path),
    }


def build_direction(
    workspace: Path, plan: DirectionPlan, clock: DemoClock, staging: Path
) -> Dict[str, Any]:
    """Replay the direction's history through the research storage code, one
    step at a time in timestamp order, so the event log stays append-only."""
    root = workspace / plan.dirname
    root.mkdir()
    clock.at(plan.created)
    Notebook(
        direction=plan.direction, budget_usd=plan.budget_usd, created=iso(clock.now)
    ).save(root)
    library.ensure(library.library_root(root))
    (root / "program.md").write_text(program_md(plan), encoding="utf-8")
    source = staging / f"{plan.dirname}-source.csv"
    source_bars(source, plan.data_seed)
    th = Thresholds()
    runs: Dict[str, Any] = {}

    def add_hypothesis(hp: HypothesisPlan, hid: str) -> None:
        nb = Notebook.load(root)
        hyp = nb.add_hypothesis(root, hp.title, hp.body)
        assert hyp.id == hid, (hyp.id, hid)
        hyp.created = iso(clock.now)
        nb.save(root)

    def scaffold(hid: str, eid: str, sim: Simulation) -> None:
        nb = Notebook.load(root)
        split = prepare_data(root, nb, source, "timestamp", 0.2)
        initial = staging / plan.dirname / hid / eid / "initial.py"
        initial.parent.mkdir(parents=True)
        initial.write_text(sim.initial_code, encoding="utf-8")
        cfg = QuantConfig(data_path=split.dev_path, timestamp_col="timestamp")
        exp, _ = scaffold_experiment(root, nb, hid, initial, cfg)
        assert exp.id == eid, (exp.id, eid)
        exp.created = iso(clock.now)
        nb.save(root)
        append_log(root, hid, eid, "scaffold", f"seed {initial.name}")

    def run_start(hid: str, eid: str, sim: Simulation) -> None:
        # `run` marks the hypothesis running before Shinka starts.
        nb = Notebook.load(root)
        nb.get(hid).status, nb.get(hid).verdict = Status.running, None
        nb.save(root)
        results = experiment_dir(root, hid, eid) / "results"
        runs[f"{hid}/{eid}"] = write_run(results, sim, clock)

    def run_end(hid: str, eid: str, ep: ExperimentPlan) -> None:
        # `run` records the evaluated programs only when Shinka exits.
        results = experiment_dir(root, hid, eid) / "results"
        nb = Notebook.load(root)
        exp = next(e for e in nb.get(hid).experiments if e.id == eid)
        exp.n_evaluated = rundb.count_evaluated(results)
        exp.n_probed = ep.probes
        nb.save(root)
        best = rundb.best_score(results)
        cost = rundb.run_cost(results)
        append_log(
            root,
            hid,
            eid,
            "run-failed" if ep.run_failed else "run",
            f"{exp.n_evaluated} new programs, {exp.n_evaluated} total, best IS "
            f"score {'n/a' if best is None else f'{best:.3f}'}, "
            f"spend {'' if cost.known else '>='}${cost.recorded_usd:.2f}",
        )

    def skeptic(hid: str, sp: SkepticPlan) -> None:
        nb = Notebook.load(root)
        hyp = nb.get(hid)
        exp = hyp.experiments[-1]
        report = skeptic_report(root, nb, hid, exp.id, sp)
        label, reason = decide(report, th)
        exp.skeptic = {
            **report,
            "rule_label": label.value,
            "rule_reason": reason,
            "thresholds": th.model_dump(
                include={"dsr_supported", "min_positive_fraction", "max_pbo"}
            ),
        }
        hyp.status, hyp.verdict = Status.challenged, None
        nb.save(root)
        append_log(root, hid, exp.id, "skeptic", f"rule says {label.value}: {reason}")
        (experiment_dir(root, hid, exp.id) / "skeptic.md").write_text(
            render_report(hid, exp.id, report, label) + f"\n{reason}\n",
            encoding="utf-8",
        )

    def verdict(hid: str) -> None:
        # `verdict` without --label adopts the rule-based label and reason.
        nb = Notebook.load(root)
        exp = nb.get(hid).experiments[-1]
        label = Status(exp.skeptic["rule_label"])
        reason = exp.skeptic["rule_reason"]
        nb.set_verdict(
            hid,
            Verdict(
                label=label,
                reason=reason,
                evidence={"experiment": exp.id, **exp.skeptic},
                created=iso(clock.now),
            ),
        )
        nb.save(root)
        append_log(root, hid, exp.id, "verdict", f"{label.value}: {reason}")

    steps: List[Tuple[float, int, Callable[[], None]]] = []

    def at(when: str | float, action: Callable[[], None]) -> None:
        steps.append((ts(when) if isinstance(when, str) else when, len(steps), action))

    for i, hp in enumerate(plan.hypotheses):
        hid = f"h{i + 1:03d}"
        at(hp.added_at, partial(add_hypothesis, hp, hid))
        for j, ep in enumerate(hp.experiments):
            eid = f"e{j + 1:02d}"
            sim = simulate(ep.run)
            at(ep.scaffold_at, partial(scaffold, hid, eid, sim))
            at(ep.run.start, partial(run_start, hid, eid, sim))
            if ep.finished:
                end = max(r.finished for r in sim.records) + 45.0
                at(end, partial(run_end, hid, eid, ep))
        if hp.skeptic is not None:
            at(hp.skeptic.at, partial(skeptic, hid, hp.skeptic))
            if hp.skeptic.verdict_at is not None:
                at(hp.skeptic.verdict_at, partial(verdict, hid))
    if plan.confirm is not None:
        at(plan.confirm.at, partial(confirm_direction, root, plan.confirm, th, clock))
    for when, _, action in sorted(steps, key=lambda s: (s[0], s[1])):
        clock.at(when)
        action()

    nb = Notebook.load(root)
    return {
        "states": {h.id: h.status.value for h in nb.hypotheses},
        "lockbox_opened": nb.data.opened if nb.data else None,
        "runs": runs,
    }


def confirm_direction(root: Path, cp: ConfirmPlan, th: Thresholds, clock: DemoClock):
    """Mirror ``cmd_confirm`` with synthetic lockbox figures."""
    clock.at(cp.at)
    nb = Notebook.load(root)
    hids = [h.id for h in nb.hypotheses if h.status is Status.supported]
    n_trials, _ = direction_trials(root, nb, PPY)
    reports = {}
    for hid in hids:
        hyp = nb.get(hid)
        rep = {
            "experiment": hyp.verdict.evidence["experiment"],
            "program_id": hyp.verdict.evidence["program_id"],
            "lockbox_start": nb.data.lockbox_start,
            "lockbox_bars": nb.data.n_total - nb.data.n_dev,
            "lockbox_sharpe": cp.lockbox_sharpe,
            "lockbox_annual_return": cp.annual_return,
            "lockbox_max_drawdown": cp.max_drawdown,
            "lockbox_buy_hold_sharpe": cp.lockbox_buy_hold,
            "n_trials": n_trials,
            "dsr": cp.dsr,
            "lockbox_positive_fraction": cp.positive_fraction,
        }
        rep["passed"] = bool(
            rep["lockbox_sharpe"] > max(0.0, rep["lockbox_buy_hold_sharpe"])
            and rep["dsr"] >= th.dsr_supported
            and rep["lockbox_positive_fraction"] >= th.min_positive_fraction
        )
        reports[hid] = rep
    nb.data.opened = True
    lines = [
        f"# Lockbox confirmation\n\n> {LABEL}\n\nLockbox from {nb.data.lockbox_start}.\n"
    ]
    for hid, rep in reports.items():
        hyp = nb.get(hid)
        hyp.confirmation = rep
        hyp.status = Status.confirmed if rep["passed"] else Status.unconfirmed
        rows = "\n".join(
            f"| {k} | {v:.4g} |" if isinstance(v, float) else f"| {k} | {v} |"
            for k, v in rep.items()
        )
        lines.append(
            f"## {hid}: {hyp.title} — {hyp.status.value}\n\n"
            f"| check | value |\n|---|---|\n{rows}\n"
        )
    nb.save(root)
    for hid, rep in reports.items():
        append_log(
            root,
            hid,
            rep["experiment"],
            "confirm",
            f"{nb.get(hid).status.value}: lockbox Sharpe {rep['lockbox_sharpe']:.2f}, "
            f"DSR {rep['dsr']:.3f}",
        )
    (root / "CONFIRMATION.md").write_text("\n".join(lines), encoding="utf-8")


def hyp_body(claim: str, mechanism: str, test: str) -> str:
    return (
        f"> {LABEL}\n\n## Claim\n{claim}\n\n## Mechanism\n{mechanism}\n\n"
        f"## Falsification\n{test}\n"
    )


def q(
    key: str,
    n: int,
    gens: int,
    islands: int,
    seed: float,
    ceiling: float,
    start: str,
    **kw,
):
    return RunSpec(key, QUANT, n, gens, islands, seed, ceiling, start, **kw)


DIRECTIONS = (
    DirectionPlan(
        dirname="d1-momentum-regimes",
        direction=f"{LABEL} · 日线动量在不同波动率状态下是否稳健",
        created="2026-10-08 09:00",
        budget_usd=80.0,
        data_seed=101,
        hypotheses=(
            HypothesisPlan(
                title=f"{LABEL} · 波动率缩放的 12/63 日动量在验证段仍有正 Sharpe",
                body=hyp_body(
                    "按目标波动率缩放仓位后，中期动量扣费后仍有正收益。",
                    "波动率缩放让高波动期的亏损变小，动量本身的趋势延续不变。",
                    "验证段 Sharpe <= 0 即否定；DSR 不足则为不确定。",
                ),
                added_at="2026-10-08 09:05",
                experiments=(
                    ExperimentPlan(
                        "2026-10-08 09:20",
                        q("d1-h001-e01", 320, 36, 4, 0.32, 1.45, "2026-10-08 09:30"),
                        probes=3,
                    ),
                ),
                skeptic=SkepticPlan(
                    "2026-10-09 10:00",
                    1.12,
                    0.41,
                    0.973,
                    1.0,
                    0.18,
                    0.22,
                    verdict_at="2026-10-09 10:20",
                ),
            ),
            HypothesisPlan(
                title=f"{LABEL} · 5 日超短回看动量优于买入持有",
                body=hyp_body(
                    "极短回看窗口能更早捕捉趋势。",
                    "短窗口反应快，但换手高、噪声大。",
                    "验证段 Sharpe <= 0 即否定。",
                ),
                added_at="2026-10-08 09:06",
                experiments=(
                    ExperimentPlan(
                        "2026-10-08 14:00",
                        q("d1-h002-e01", 220, 30, 3, 0.18, 0.78, "2026-10-08 14:10"),
                    ),
                ),
                skeptic=SkepticPlan(
                    "2026-10-09 11:00",
                    -0.21,
                    0.41,
                    0.08,
                    0.333,
                    0.71,
                    0.05,
                    verdict_at="2026-10-09 11:15",
                ),
            ),
            HypothesisPlan(
                title=f"{LABEL} · 高波动状态下切换为均值回归能提升稳定性",
                body=hyp_body(
                    "高波动期改用均值回归，可以减少动量在震荡市中的回撤。",
                    "高波动期价格更容易过度反应后回落。",
                    "需要 DSR >= 0.95 且子区间多数为正才算支持。",
                ),
                added_at="2026-10-10 09:00",
                experiments=(
                    ExperimentPlan(
                        "2026-10-10 09:15",
                        q("d1-h003-e01", 180, 28, 3, 0.25, 0.95, "2026-10-10 09:30"),
                    ),
                    ExperimentPlan(
                        "2026-10-11 10:00",
                        q("d1-h003-e02", 260, 34, 4, 0.3, 1.1, "2026-10-11 10:15"),
                        probes=2,
                    ),
                ),
                skeptic=SkepticPlan(
                    "2026-10-13 14:30",
                    0.58,
                    0.41,
                    0.71,
                    0.333,
                    0.38,
                    0.19,
                    verdict_at="2026-10-13 15:00",
                ),
            ),
            HypothesisPlan(
                title=f"{LABEL} · 加入成交量确认可降低换手且不损失收益",
                body=hyp_body(
                    "只在成交量高于中位数时开仓，换手下降而 Sharpe 不降。",
                    "低量突破更容易失败。",
                    "验证段 Sharpe 低于 h001 或换手不降则不支持。",
                ),
                added_at="2026-10-10 09:01",
                experiments=(
                    ExperimentPlan(
                        "2026-10-18 08:00",
                        q("d1-h004-e01", 150, 25, 3, 0.33, 1.2, "2026-10-18 08:30"),
                        finished=False,
                    ),
                ),
            ),
        ),
    ),
    DirectionPlan(
        dirname="d2-overnight-gap",
        direction=f"{LABEL} · 隔夜跳空的延续效应（已结题）",
        created="2026-09-28 09:00",
        budget_usd=40.0,
        data_seed=202,
        hypotheses=(
            HypothesisPlan(
                title=f"{LABEL} · 大幅高开后当日延续，扣费后仍为正",
                body=hyp_body(
                    "高开超过 1 个标准差的交易日，收盘方向延续开盘方向。",
                    "隔夜信息在开盘后继续被消化。",
                    "封存段 Sharpe 不超过买入持有即不确认。",
                ),
                added_at="2026-09-28 09:10",
                experiments=(
                    ExperimentPlan(
                        "2026-09-28 10:00",
                        q("d2-h001-e01", 240, 32, 4, 0.28, 1.3, "2026-09-28 10:15"),
                        probes=1,
                    ),
                ),
                skeptic=SkepticPlan(
                    "2026-09-30 09:00",
                    1.05,
                    0.36,
                    0.968,
                    0.667,
                    0.22,
                    0.15,
                    verdict_at="2026-09-30 09:30",
                ),
            ),
            HypothesisPlan(
                title=f"{LABEL} · 小幅低开后当日反转",
                body=hyp_body(
                    "小幅低开的交易日，日内倾向于回补缺口。",
                    "小幅低开多为噪声，流动性提供者会把价格拉回。",
                    "验证段 Sharpe <= 0 即否定。",
                ),
                added_at="2026-09-28 09:12",
                experiments=(
                    ExperimentPlan(
                        "2026-09-30 14:00",
                        q("d2-h002-e01", 160, 26, 3, 0.15, 0.6, "2026-09-30 14:15"),
                    ),
                ),
                skeptic=SkepticPlan(
                    "2026-10-02 09:00",
                    -0.34,
                    0.36,
                    0.04,
                    0.0,
                    0.66,
                    0.02,
                    verdict_at="2026-10-02 09:20",
                ),
            ),
        ),
        confirm=ConfirmPlan(
            "2026-10-04 10:00", 0.94, 0.38, 0.961, 0.667, 0.112, -0.087
        ),
    ),
    DirectionPlan(
        dirname="d3-volume-shock",
        direction=f"{LABEL} · 成交量冲击后的价格漂移（新方向）",
        created="2026-10-17 16:00",
        budget_usd=20.0,
        data_seed=303,
        hypotheses=(
            HypothesisPlan(
                title=f"{LABEL} · 成交量突增 3 倍后 5 日内正向漂移",
                body=hyp_body(
                    "成交量超过 20 日均量 3 倍的交易日之后，价格在 5 日内继续漂移。",
                    "大资金分批建仓，信息逐步进入价格。",
                    "验证段 Sharpe <= 0 即否定。",
                ),
                added_at="2026-10-17 16:10",
            ),
        ),
    ),
    DirectionPlan(
        dirname="d4-short-reversal",
        direction=f"{LABEL} · 短期反转（预算超支与评估失败示例）",
        created="2026-10-14 09:00",
        budget_usd=8.0,
        data_seed=404,
        hypotheses=(
            HypothesisPlan(
                title=f"{LABEL} · 3 日反转在低流动性日更强",
                body=hyp_body(
                    "过去 3 日跌幅最大的交易日之后买入，低成交量时效果更强。",
                    "低流动性时价格冲击更大，回补也更明显。",
                    "验证段 Sharpe <= 0 即否定。",
                ),
                added_at="2026-10-14 09:10",
                experiments=(
                    ExperimentPlan(
                        "2026-10-14 09:30",
                        q("d4-h001-e01", 200, 28, 3, 0.2, 0.9, "2026-10-14 09:45"),
                    ),
                ),
                skeptic=SkepticPlan(
                    "2026-10-15 10:00", 0.31, 0.44, 0.42, 0.333, 0.55, 0.1
                ),
            ),
            HypothesisPlan(
                title=f"{LABEL} · 反转信号叠加波动率过滤（评估器多次失败）",
                body=hyp_body(
                    "只在波动率低于中位数时交易反转信号。",
                    "高波动期反转更容易继续下跌。",
                    "验证段 Sharpe <= 0 即否定。",
                ),
                added_at="2026-10-15 11:00",
                experiments=(
                    ExperimentPlan(
                        "2026-10-15 11:20",
                        q(
                            "d4-h002-e01",
                            150,
                            25,
                            3,
                            0.22,
                            0.85,
                            "2026-10-15 11:30",
                            fail_from=15,
                            fail_spike=0.75,
                        ),
                        run_failed=True,
                    ),
                ),
            ),
        ),
    ),
)


ORDINARY_RUNS = (
    (
        "circle_packing_demo",
        "run_20261012_093000",
        RunSpec(
            "runs-cp-tiny",
            PACKING,
            12,
            3,
            2,
            2.05,
            2.63,
            "2026-10-12 09:30",
            n_prompts=0,
            meta=False,
            plots=False,
            workers=(2, 2, 1),
        ),
    ),
    (
        "circle_packing_demo",
        "run_20261015_141500",
        RunSpec(
            "runs-cp-large",
            PACKING,
            500,
            40,
            4,
            2.05,
            2.635,
            "2026-10-15 14:15",
            n_prompts=12,
            workers=(6, 4, 2),
        ),
    ),
    ("bin_packing_demo", "run_20261013_101000", None),
    (
        "bin_packing_demo",
        "run_20261016_200000",
        RunSpec(
            "runs-bp-medium",
            BINPACK,
            180,
            26,
            3,
            0.86,
            0.965,
            "2026-10-16 20:00",
            n_prompts=6,
        ),
    ),
    (
        "quant_signal_demo",
        "run_20261017_083000",
        RunSpec(
            "runs-qs-medium",
            QUANT,
            220,
            30,
            3,
            0.25,
            1.2,
            "2026-10-17 08:30",
            n_prompts=7,
            cost_unknown=True,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Shared library.
# ---------------------------------------------------------------------------

WIKI = {
    "concept-deflated-sharpe-ratio": (
        "concept",
        "Deflated Sharpe ratio: how likely the best Sharpe beats the best of n null "
        "strategies.",
        "The deflated Sharpe ratio (DSR) corrects an observed Sharpe ratio for the "
        "number of variants tried and the variance of their Sharpe ratios. The skeptic "
        "counts every evaluated program and every probe of the direction as a trial.",
    ),
    "concept-probability-of-backtest-overfitting": (
        "concept",
        "PBO via combinatorially symmetric cross-validation.",
        "PBO estimates how often the in-sample winner ranks below the median out of "
        "sample when the history is split into symmetric halves. The skeptic computes "
        "it over the best distinct programs of one experiment.",
    ),
    "concept-lockbox": (
        "concept",
        "The lockbox is the untouched tail of the data, opened once per direction.",
        "`confirm` scores every supported hypothesis on the lockbox at the same time, "
        "then marks it opened; a new direction needs new data.",
    ),
    "concept-volatility-regimes": (
        "concept",
        "Splitting history into calm and turbulent regimes by rolling volatility.",
        "A regime filter compares rolling volatility with its long-run median. It is a "
        "frequent ingredient of the demo momentum programs.",
    ),
    "paper-demo-regime-momentum": (
        "paper",
        "Fictional paper note used only to show the paper page layout.",
        "This page summarises a fictional paper invented for the demo. It does not "
        "describe any real publication or result.",
    ),
    "paper-demo-overnight-drift": (
        "paper",
        "Fictional paper note about overnight gaps, invented for the demo.",
        "Invented for the demo; cite nothing from it.",
    ),
    "finding-d1-h001-vol-scaled-momentum": (
        "finding",
        "d1/h001 supported on synthetic data; lockbox still sealed.",
        "Synthetic demo finding: the volatility-scaled momentum program passed the "
        "skeptic rules on the demo data. Nothing was learned about real markets.",
    ),
    "finding-d1-h002-short-lookback": (
        "finding",
        "d1/h002 refuted on synthetic data: turnover eats the edge.",
        "Synthetic demo finding: out-of-sample Sharpe below zero in the demo numbers.",
    ),
    "finding-d2-h001-overnight-gap": (
        "finding",
        "d2/h001 confirmed on the synthetic lockbox.",
        "Synthetic demo finding: the demo lockbox confirmation passed. Not a result.",
    ),
}


def build_library(workspace: Path, clock: DemoClock, staging: Path) -> Dict[str, Any]:
    lib = library.ensure(workspace / library.LIBRARY_DIRNAME)
    clock.at("2026-09-27 15:00")
    notes = staging / "regime-ideas.md"
    notes.write_text(
        f"# {LABEL}\n\nRaw notes: ideas for regime-aware momentum (synthetic).\n",
        encoding="utf-8",
    )
    abstracts = staging / "demo-paper-abstracts.txt"
    abstracts.write_text(
        f"{LABEL}\nFictional abstracts used to demo the library. Not real papers.\n",
        encoding="utf-8",
    )
    library.add(lib, str(notes), "human")
    clock.at("2026-09-27 15:05")
    library.add(lib, str(abstracts), "human")
    clock.at("2026-09-27 16:00")
    for name, (kind, summary, text) in WIKI.items():
        (lib / "wiki" / f"{name}.md").write_text(
            f"# {name}\n\n> {LABEL}\n\nType: {kind}\n\n{summary}\n\n{text}\n",
            encoding="utf-8",
        )
    (lib / "index.md").write_text(
        library.INDEX_TEMPLATE
        + "\n"
        + "".join(
            f"- [[{name}]] ({kind}): {summary}\n"
            for name, (kind, summary, _) in WIKI.items()
        ),
        encoding="utf-8",
    )
    library.mark_compiled(lib, library.pending(lib))
    library.log(lib, "compile", f"{LABEL}: wrote {len(WIKI)} wiki pages")
    clock.at("2026-10-04 11:00")
    library.log(lib, "finding", f"{LABEL}: filed finding-d2-h001 after confirmation")
    clock.at("2026-10-09 12:00")
    library.log(lib, "finding", f"{LABEL}: filed finding-d1-h001 and finding-d1-h002")
    clock.at("2026-10-18 09:00")
    library.add(lib, "https://example.org/demo/volume-shock-notes", "human")
    library.log(lib, "lint", f"{LABEL}: 1 raw source waiting to be compiled")
    return {"wiki_pages": len(library.wiki_pages(lib)), "pending": library.pending(lib)}


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def empty_target(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        sys.exit(f"refusing to write into non-empty target: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def build(workspace: Path, runs: Path) -> Dict[str, Any]:
    random.seed(SEED)
    np.random.seed(SEED)
    summary: Dict[str, Any] = {"workspace": str(workspace), "runs_root": str(runs)}
    with demo_clock() as clock, tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        write_label(
            workspace / "DEMO_DATA.md",
            "Demo research workspace: four directions in different states and a "
            "shared library.",
        )
        summary["library"] = build_library(workspace, clock, staging)
        summary["directions"] = {
            plan.dirname: build_direction(workspace, plan, clock, staging)
            for plan in DIRECTIONS
        }
        write_label(
            runs / "DEMO_DATA.md",
            "Demo ordinary runs root: five runs from three tasks, no research link.",
        )
        summary["runs"] = {}
        for task, name, spec in ORDINARY_RUNS:
            results = runs / task / name
            if spec is None:
                summary["runs"][f"{task}/{name}"] = write_empty_run(
                    results, clock, "2026-10-13 10:10"
                )
            else:
                summary["runs"][f"{task}/{name}"] = write_run(
                    results, simulate(spec), clock
                )
            write_label(results / "DEMO_DATA.md", f"Demo ordinary run {task}/{name}.")
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    args = parser.parse_args(argv)
    workspace = empty_target(args.workspace)
    runs = empty_target(args.runs)
    if workspace == runs or workspace in runs.parents or runs in workspace.parents:
        sys.exit("--workspace and --runs must be separate directories")
    summary = build(workspace, runs)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
