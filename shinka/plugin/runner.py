"""plugin/runner.py — universal keep/revert loop, autoresearch-compatible.

Two modes:
  agent  — karpathy-style: outer loop does eval + git keep/revert.
           The agent (Claude / Codex) edits mutable files externally.
  shinka — ShinkaEvolve UCB loop: calls evaluate.py directly.
           evaluate.py already contains run_shinka_eval with validate/aggregate.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .loader import Plugin

# ShinkaEvolve root (two levels up from this file: shinka/plugin/runner.py)
_SHINKA_ROOT = str(Path(__file__).parent.parent.parent)


def _eval_env() -> dict:
    """Subprocess env with ShinkaEvolve root injected into PYTHONPATH."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _SHINKA_ROOT + (os.pathsep + existing if existing else "")
    return env


# ── eval ─────────────────────────────────────────────────────────────────────

def run_eval_once(plugin: Plugin, results_dir: str | None = None) -> float:
    """Run evaluate.py for plugin.mutable[0], return metric score."""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = results_dir or tmp
        eval_path = plugin.abs(plugin.eval_module)
        mutable_path = plugin.abs(plugin.mutable[0])

        subprocess.run(
            [
                sys.executable, str(eval_path),
                "--program_path", str(mutable_path),
                "--results_dir", out_dir,
            ],
            cwd=str(plugin.plugin_dir),
            env=_eval_env(),
            check=True,
        )

        metrics_file = Path(out_dir) / "metrics.json"
        with open(metrics_file) as f:
            metrics = json.load(f)

    score = float(metrics.get(plugin.metric, -1e9))
    return score if plugin.higher_is_better else -score


# ── git helpers ───────────────────────────────────────────────────────────────

def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + args, cwd=str(cwd), check=True, capture_output=True)


def _git_commit(plugin: Plugin, msg: str) -> None:
    for f in plugin.mutable:
        _git(["add", f], plugin.plugin_dir)
    _git(["commit", "-m", msg], plugin.plugin_dir)


def _git_revert(plugin: Plugin) -> None:
    for f in plugin.mutable:
        _git(["checkout", "HEAD", "--", f], plugin.plugin_dir)


# ── prepare ───────────────────────────────────────────────────────────────────

def _run_prepare(plugin: Plugin) -> None:
    if not plugin.prepare:
        return
    prepare_path = plugin.abs(plugin.prepare)
    if prepare_path.exists():
        print(f"[plugin] prepare: {prepare_path.name}")
        subprocess.run(
            [sys.executable, str(prepare_path)],
            cwd=str(plugin.plugin_dir),
            env=_eval_env(),
            check=True,
        )


# ── agent-mode loop ───────────────────────────────────────────────────────────

def run_loop(
    plugin: Plugin,
    results_root: str = "results",
    mode: str = "agent",
) -> None:
    """
    Karpathy-compatible outer loop.

    mode='agent':  expects the agent to have already edited mutable files.
                   Call this after each agent edit to evaluate + keep/revert.
    mode='shinka': delegates to evaluate.py which calls run_shinka_eval
                   (the agent/UCB loop is inside ShinkaEvolve itself).
    """
    _run_prepare(plugin)

    if mode == "shinka":
        # ShinkaEvolve has its own loop — just delegate to evaluate.py main()
        eval_path = plugin.abs(plugin.eval_module)
        mutable_path = plugin.abs(plugin.mutable[0])
        results_dir = Path(plugin.plugin_dir) / results_root / "shinka"
        results_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable, str(eval_path),
                "--program_path", str(mutable_path),
                "--results_dir", str(results_dir),
            ],
            cwd=str(plugin.plugin_dir),
            env=_eval_env(),
            check=True,
        )
        return

    # agent mode: keep/revert loop
    best: float = -1e18
    results_root_path = Path(plugin.plugin_dir) / results_root
    results_root_path.mkdir(parents=True, exist_ok=True)

    # baseline
    run_dir = str(results_root_path / "gen_000")
    try:
        best = run_eval_once(plugin, run_dir)
        _git_commit(plugin, f"[plugin] baseline {plugin.name}: {best:.4f}")
        print(f"[plugin] baseline {best:.4f}")
    except Exception as e:
        print(f"[plugin] baseline eval failed: {e}")

    print(
        f"\n[plugin] '{plugin.name}' ready — {plugin.mutable[0]} is the mutable file.\n"
        f"         Edit it, then call: python run_plugin.py --plugin . --step\n"
        f"         Or run an agent pointed at program.md.\n"
        f"         Best so far: {best:.4f}  (metric={plugin.metric})\n"
    )


def run_step(plugin: Plugin, gen: int, results_root: str = "results") -> tuple[float, bool]:
    """
    Single keep/revert step — call after the agent has edited mutable files.
    Returns (score, kept).
    """
    run_dir = str(Path(plugin.plugin_dir) / results_root / f"gen_{gen:03d}")
    try:
        score = run_eval_once(plugin, run_dir)
    except Exception as e:
        print(f"[plugin] gen {gen} eval failed: {e} — reverting")
        _git_revert(plugin)
        return -1e18, False

    state_file = Path(plugin.plugin_dir) / results_root / ".best"
    best = float(state_file.read_text()) if state_file.exists() else -1e18

    if score > best:
        # verification gate: run verify_cmd before committing
        if plugin.verify_cmd:
            vresult = subprocess.run(
                plugin.verify_cmd,
                shell=True,
                cwd=str(plugin.plugin_dir),
                env=_eval_env(),
            )
            if vresult.returncode != 0:
                _git_revert(plugin)
                print(f"[plugin] gen {gen:03d} REVERT verify_cmd failed (score {score:.4f})")
                return score, False

        state_file.write_text(str(score))
        _git_commit(plugin, f"[plugin] gen {gen:03d} {plugin.name}: {score:.4f} (+{score-best:.4f})")
        print(f"[plugin] gen {gen:03d} KEEP  {score:.4f} (prev best {best:.4f})")
        return score, True
    else:
        _git_revert(plugin)
        print(f"[plugin] gen {gen:03d} REVERT {score:.4f} < best {best:.4f}")
        return score, False
