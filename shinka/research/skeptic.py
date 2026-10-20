"""Skeptic: try to break an experiment's best result on the untouched holdout.

Checks (all on out-of-sample returns the mutation LLM never saw):

* Deflated Sharpe Ratio (Bailey & López de Prado, 2014). The number of trials
  is every program evaluated across the whole research direction; the
  dispersion of trial Sharpe ratios is estimated from the in-sample scores of
  the correct programs (an approximation: OOS Sharpe of every trial is not
  computed).
* Sub-period stability: share of OOS segments with positive Sharpe.
* Baselines: buy-and-hold and the seed program over the same holdout.
* In-sample -> out-of-sample decay.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats

from shinka.research import rundb
from shinka.research.notebook import (
    DataSplit,
    Experiment,
    LockboxTarget,
    Notebook,
    Status,
    experiment_dir,
)
from shinka.research.quant import (
    InvalidPositions,
    ProgramFailed,
    ProgramTimeout,
    QuantConfig,
    annual_return,
    check_positions,
    file_sha256,
    load_bars,
    max_drawdown,
    positions_for,
    run_view,
    sharpe,
    split_index,
    strategy_returns,
)
from shinka.research.scaffold import load_config, verify_experiment

EULER_GAMMA = 0.5772156649015329


class Thresholds(BaseModel):
    dsr_supported: float = 0.95
    min_positive_fraction: float = 2.0 / 3.0
    n_subperiods: int = 3
    max_pbo: float = 0.5
    pbo_blocks: int = 16
    pbo_max_programs: int = 64


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` null strategies."""
    if n_trials <= 1 or var_trials <= 0.0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_trials) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(returns: pd.Series, n_trials: int, var_trials: float) -> float:
    """Probability that the true per-period Sharpe exceeds the multiple-testing
    benchmark, accounting for sample length, skewness and kurtosis."""
    r = returns.to_numpy(dtype=float)
    t = len(r)
    std = r.std(ddof=1) if t > 1 else 0.0
    if t < 3 or std == 0.0:
        return 0.0
    sr = r.mean() / std
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    sr0 = expected_max_sharpe(n_trials, var_trials)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0.0:
        return 0.0
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(t - 1) / math.sqrt(denom)))


def positive_fraction(returns: pd.Series, n: int, periods_per_year: float) -> float:
    parts = [p for p in np.array_split(returns.to_numpy(), n) if len(p) > 1]
    if not parts:
        return 0.0
    return sum(sharpe(pd.Series(p), periods_per_year) > 0 for p in parts) / len(parts)


def pbo(returns: np.ndarray, n_blocks: int = 16) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey, Borwein,
    López de Prado & Zhu, 2014).

    ``returns`` is a (T, N) matrix of per-bar returns of N trials. Rows are cut
    into ``n_blocks`` contiguous blocks; for every way of choosing half the
    blocks as in-sample, the in-sample best trial is ranked out-of-sample.
    PBO is the share of splits where it lands at or below the median.
    """
    t, n = returns.shape
    if n < 2 or n_blocks < 2 or n_blocks % 2 or t < 2 * n_blocks:
        return float("nan")
    blocks = np.array_split(np.arange(t), n_blocks)
    cnt = np.array([len(b) for b in blocks], dtype=float)
    s1 = np.stack([returns[b].sum(axis=0) for b in blocks])
    s2 = np.stack([(returns[b] ** 2).sum(axis=0) for b in blocks])
    tot1, tot2, totc = s1.sum(axis=0), s2.sum(axis=0), cnt.sum()

    def _sr(a1: np.ndarray, a2: np.ndarray, c: float) -> np.ndarray:
        mean = a1 / c
        var = np.maximum(a2 / c - mean**2, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(var > 0, mean / np.sqrt(var), 0.0)

    below = total = 0
    for combo in combinations(range(n_blocks), n_blocks // 2):
        idx = list(combo)
        c_in = cnt[idx].sum()
        in_sr = _sr(s1[idx].sum(axis=0), s2[idx].sum(axis=0), c_in)
        out_sr = _sr(
            tot1 - s1[idx].sum(axis=0), tot2 - s2[idx].sum(axis=0), totc - c_in
        )
        best = int(np.argmax(in_sr))
        rank = 1 + int(np.sum(out_sr < out_sr[best]))  # 1 = worst
        omega = rank / (n + 1)
        below += omega <= 0.5
        total += 1
    return below / total


def decide(report: Dict[str, Any], th: Thresholds) -> tuple[Status, str]:
    """Rule-based verdict. Failing to show an edge is inconclusive, not a
    refutation; only a non-positive OOS Sharpe refutes."""
    if report["oos_sharpe"] <= 0.0:
        return Status.refuted, f"OOS Sharpe {report['oos_sharpe']:.2f} <= 0"
    hold = report["baseline_buy_hold_oos_sharpe"]
    if report["oos_sharpe"] <= hold:
        return Status.inconclusive, (
            f"OOS Sharpe {report['oos_sharpe']:.2f} does not beat buy-and-hold "
            f"({hold:.2f})"
        )
    pbo_ok = not (report.get("pbo", float("nan")) > th.max_pbo)  # NaN passes
    if (
        report["dsr"] >= th.dsr_supported
        and report["oos_positive_fraction"] >= th.min_positive_fraction
        and pbo_ok
    ):
        return Status.supported, (
            f"DSR {report['dsr']:.3f} >= {th.dsr_supported} over "
            f"{report['n_trials']} trials; {report['oos_positive_fraction']:.0%} "
            f"of OOS sub-periods positive; beats buy-and-hold ({hold:.2f})"
        )
    return Status.inconclusive, (
        f"OOS Sharpe {report['oos_sharpe']:.2f} > 0 but DSR {report['dsr']:.3f} "
        f"(needs {th.dsr_supported}), sub-period stability "
        f"{report['oos_positive_fraction']:.0%} (needs "
        f"{th.min_positive_fraction:.0%}) or PBO {report.get('pbo', float('nan')):.2f} "
        f"(max {th.max_pbo}) falls short"
    )


def _oos_returns(
    program: Path, bars: pd.DataFrame, cfg: QuantConfig
) -> tuple[pd.Series, pd.Series]:
    rets = strategy_returns(
        positions_for(program, cfg, bars), bars["close"], cfg.cost_bps
    )
    cut = split_index(len(bars), cfg.oos_fraction)
    return rets.iloc[:cut], rets.iloc[cut:]


def direction_trials(root: Path, nb: Notebook, ppy: float) -> tuple[int, float]:
    """(number of trials, variance of per-period trial Sharpe) for the whole
    direction: every evaluated program plus every probe."""
    scores: List[float] = []
    n_trials = 0
    for h in nb.hypotheses:
        for e in h.experiments:
            n_trials += e.n_probed
            rd = experiment_dir(root, h.id, e.id) / "results"
            if (rd / rundb.DB_FILENAME).exists():
                scores.extend(rundb.correct_scores(rd))
                n_trials += rundb.count_evaluated(rd)
    per_period = np.asarray(scores) / math.sqrt(ppy)
    var_trials = float(per_period.var(ddof=1)) if len(per_period) > 1 else 0.0
    return max(n_trials, 1), var_trials


def _experiment_pbo(
    results_dir: Path, bars: pd.DataFrame, cfg: QuantConfig, th: Thresholds
) -> float:
    """PBO over the experiment's best distinct programs on development data."""
    programs = rundb.top_correct_programs(results_dir, th.pbo_max_programs)
    if len(programs) < 2:
        return float("nan")
    cols = []
    with tempfile.TemporaryDirectory() as tmp:
        for pid, code in programs:
            path = Path(tmp) / f"p_{len(cols)}.py"
            path.write_text(code)
            pos = positions_for(path, cfg, bars)
            cols.append(strategy_returns(pos, bars["close"], cfg.cost_bps).to_numpy())
    return pbo(np.column_stack(cols), th.pbo_blocks)


def challenge(
    root: Path, nb: Notebook, hid: str, exp: Experiment, th: Thresholds
) -> Dict[str, Any]:
    """Run all checks for one experiment and return the evidence report."""
    verify_experiment(root, nb, hid, exp)
    exp_dir = experiment_dir(root, hid, exp.id)
    task_dir, results_dir = exp_dir / "task", exp_dir / "results"
    cfg = load_config(task_dir)
    bars = load_bars(Path(cfg.data_path), cfg.timestamp_col)
    ppy = cfg.periods_per_year

    pid, code, is_score = rundb.best_program(results_dir)
    best_path = exp_dir / "best_program.py"
    best_path.write_text(code)
    is_rets, oos = _oos_returns(best_path, bars, cfg)
    _, seed_oos = _oos_returns(task_dir / "initial.py", bars, cfg)
    hold = bars["close"].pct_change().fillna(0.0)
    hold_oos = hold.iloc[split_index(len(bars), cfg.oos_fraction) :]

    n_trials, var_trials = direction_trials(root, nb, ppy)
    pbo_value = _experiment_pbo(results_dir, bars, cfg, th)

    oos_sharpe = sharpe(oos, ppy)
    is_sharpe = sharpe(is_rets, ppy)
    return {
        "program_id": pid,
        "is_score": is_score,
        "is_sharpe": is_sharpe,
        "oos_sharpe": oos_sharpe,
        "oos_bars": len(oos),
        "decay": oos_sharpe / is_sharpe if is_sharpe else None,
        "n_trials": n_trials,
        "trial_sharpe_var_per_period": var_trials,
        "expected_max_sharpe_annual": expected_max_sharpe(n_trials, var_trials)
        * math.sqrt(ppy),
        "dsr": deflated_sharpe(oos, n_trials, var_trials),
        "oos_positive_fraction": positive_fraction(oos, th.n_subperiods, ppy),
        "pbo": pbo_value,
        "baseline_seed_oos_sharpe": sharpe(seed_oos, ppy),
        "baseline_buy_hold_oos_sharpe": sharpe(hold_oos, ppy),
        "best_program": str(best_path),
    }


def render_report(hid: str, eid: str, report: Dict[str, Any], label: Status) -> str:
    rows = "\n".join(
        f"| {k} | {v:.4g} |" if isinstance(v, float) else f"| {k} | {v} |"
        for k, v in report.items()
    )
    return (
        f"# Skeptic report: {hid}/{eid}\n\n"
        f"Rule-based verdict: **{label.value}**\n\n"
        f"| check | value |\n|---|---|\n{rows}\n"
    )


def _code_sha256(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def confirm_target(root: Path, nb: Notebook, hid: str) -> LockboxTarget:
    """Everything ``confirm`` checks for one supported hypothesis before the
    lockbox is touched: the experiment and program its verdict names, the
    frozen evaluator and development data, and the direction's trials for the
    DSR. Returns what the lockbox opening freezes for it."""
    hyp = nb.get(hid)
    if hyp.verdict is None or hyp.status is not Status.supported:
        raise ValueError(f"{hid} is not supported")
    eid = hyp.verdict.evidence["experiment"]
    exp = next((e for e in hyp.experiments if e.id == eid), None)
    if exp is None:
        raise ValueError(f"{hid}'s verdict names experiment {eid}, which it lacks")
    verify_experiment(root, nb, hid, exp)
    exp_dir = experiment_dir(root, hid, eid)
    pid = hyp.verdict.evidence["program_id"]
    code = rundb.program_code(exp_dir / "results", pid)
    ppy = load_config(exp_dir / "task").periods_per_year
    n_trials, var_trials = direction_trials(root, nb, ppy)
    return LockboxTarget(
        hypothesis=hid,
        experiment=eid,
        program_id=pid,
        code_sha256=_code_sha256(code),
        evaluator_sha256=exp.evaluator_hash,
        data_path=exp.data_path,
        data_sha256=exp.data_hash,
        n_trials=n_trials,
        var_trials=var_trials,
    )


def check_source(source_hash: str, source: Path) -> None:
    """Raise unless ``source`` is the file the direction was split from."""
    if file_sha256(Path(source)) != source_hash:
        raise ValueError(f"{source} is not the data this direction was split from")


def confirm(
    root: Path,
    split: DataSplit,
    target: LockboxTarget,
    th: Thresholds,
    source: Path,
    source_sha256: str,
    tmp: Path,
    started: Callable[[], None],
) -> Dict[str, Any]:
    """Score one frozen target's program on the lockbox.

    A program, evaluator or development data that differs from what the target
    froze is not run. ``started`` runs just before the program first sees
    lockbox bars; from then on a retry would be a second look. The program's
    own failures there give a failed report naming only their category, since
    its output may carry lockbox figures. Returns the evidence and does not
    modify the notebook."""
    hid, pid = target.hypothesis, target.program_id
    exp_dir = experiment_dir(root, hid, target.experiment)
    base = {
        "experiment": target.experiment,
        "program_id": pid,
        "lockbox_start": split.lockbox_start,
    }
    changed = {**base, "passed": False, "failure": "changed"}
    try:
        code = rundb.program_code(exp_dir / "results", pid)
        evaluator = file_sha256(exp_dir / "task" / "evaluate.py")
        data = file_sha256(Path(target.data_path))
    except (LookupError, FileNotFoundError):
        return changed
    if (_code_sha256(code), evaluator, data) != (
        target.code_sha256,
        target.evaluator_sha256,
        target.data_sha256,
    ):
        return changed
    best = Path(tmp) / f"{hid}_{pid}.py"
    best.write_text(code)
    # This function and the program's own process read one copy, checked against
    # the frozen hash, so the source cannot change between the check and a read.
    data = Path(tmp) / f"{hid}_source{Path(source).suffix}"
    shutil.copyfile(source, data)
    if file_sha256(data) != source_sha256:
        raise ValueError(f"{source} is not the data this direction was split from")
    cfg = load_config(exp_dir / "task").model_copy(
        update={"data_path": str(data.resolve())}
    )
    bars = load_bars(data, split.timestamp_col)
    ppy = cfg.periods_per_year

    started()
    try:
        full = positions_for(best, cfg, bars).to_numpy()
        n_lock = len(bars) - split.n_dev
        for n in (split.n_dev + n_lock // 3, split.n_dev + 2 * n_lock // 3):
            part = run_view(
                str(best), cfg.data_path, cfg.timestamp_col, n, cfg.view_timeout_s
            )
            # NaN differences compare false, so a view must be valid positions
            # before it can prove causality.
            if check_positions(part, n):
                return {**base, "passed": False, "failure": "invalid-positions"}
            if np.any(np.abs(full[:n] - part) > 1e-9):
                return {**base, "passed": False, "failure": "look-ahead"}
    except ProgramTimeout:
        return {**base, "passed": False, "failure": "timeout"}
    except ProgramFailed:
        return {**base, "passed": False, "failure": "error"}
    except InvalidPositions:
        return {**base, "passed": False, "failure": "invalid-positions"}

    rets = strategy_returns(
        pd.Series(full, index=bars.index), bars["close"], cfg.cost_bps
    )
    lock = rets.iloc[split.n_dev :]
    hold = bars["close"].pct_change().fillna(0.0).iloc[split.n_dev :]
    n_trials, var_trials = target.n_trials, target.var_trials
    report = {
        **base,
        "lockbox_bars": len(lock),
        "lockbox_sharpe": sharpe(lock, ppy),
        "lockbox_annual_return": annual_return(lock, ppy),
        "lockbox_max_drawdown": max_drawdown(lock),
        "lockbox_buy_hold_sharpe": sharpe(hold, ppy),
        "n_trials": n_trials,
        "dsr": deflated_sharpe(lock, n_trials, var_trials),
        "lockbox_positive_fraction": positive_fraction(lock, th.n_subperiods, ppy),
    }
    passed = (
        report["lockbox_sharpe"] > max(0.0, report["lockbox_buy_hold_sharpe"])
        and report["dsr"] >= th.dsr_supported
        and report["lockbox_positive_fraction"] >= th.min_positive_fraction
    )
    report["passed"] = bool(passed)
    return report
