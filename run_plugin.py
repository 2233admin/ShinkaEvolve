#!/usr/bin/env python
"""run_plugin.py — universal autoresearch-compatible plugin runner.

Usage:
    # Show plugin info
    python run_plugin.py --plugin forge_evolve/

    # Init baseline (first run)
    python run_plugin.py --plugin forge_evolve/ --init

    # After agent edits initial.py, evaluate + keep/revert
    python run_plugin.py --plugin forge_evolve/ --step --gen 1

    # Delegate to ShinkaEvolve UCB loop
    python run_plugin.py --plugin forge_evolve/ --mode shinka

Examples with domains:
    python run_plugin.py --plugin D:/projects/FORGE/forge_evolve/
    python run_plugin.py --plugin D:/projects/hist-mat/hmc_evolve/
    python run_plugin.py --plugin D:/projects/quant-terminal/shinka_evo/
"""

import argparse
import sys
from pathlib import Path

# Allow running from ShinkaEvolve root without install
sys.path.insert(0, str(Path(__file__).parent))

from shinka.plugin import load, run_loop, run_eval_once
from shinka.plugin.runner import run_step


def cmd_info(plugin):
    print(f"Plugin:      {plugin.name}")
    print(f"Description: {plugin.description}")
    print(f"Dir:         {plugin.plugin_dir}")
    print(f"Mutable:     {plugin.mutable}")
    print(f"Eval:        {plugin.eval_module} -> {plugin.eval_fn}()")
    print(f"Metric:      {plugin.metric}  ({'higher' if plugin.higher_is_better else 'lower'} is better)")
    if plugin.prepare:
        print(f"Prepare:     {plugin.prepare}")


def cmd_eval(plugin):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        score = run_eval_once(plugin, tmp)
    print(f"Score: {score:.4f}  ({plugin.metric})")


def main():
    p = argparse.ArgumentParser(description="Universal autoresearch plugin runner")
    p.add_argument("--plugin", required=True, help="Path to plugin directory (contains plugin.yaml)")
    p.add_argument("--mode", default="agent", choices=["agent", "shinka"],
                   help="agent=karpathy loop, shinka=ShinkaEvolve UCB")
    p.add_argument("--init", action="store_true", help="Run baseline eval and prepare")
    p.add_argument("--step", action="store_true", help="Evaluate current state, keep/revert")
    p.add_argument("--eval", action="store_true", help="Run eval once, print score (no git)")
    p.add_argument("--gen", type=int, default=1, help="Generation number (for --step)")
    p.add_argument("--results", default="results", help="Results root dir")
    args = p.parse_args()

    plugin = load(args.plugin)

    if args.eval:
        cmd_eval(plugin)
    elif args.step:
        run_step(plugin, args.gen, args.results)
    elif args.init:
        run_loop(plugin, max_iters=1, results_root=args.results, mode="agent")
    elif args.mode == "shinka":
        run_loop(plugin, results_root=args.results, mode="shinka")
    else:
        cmd_info(plugin)
        print("\nRun with --init to start, or --eval to test current state.")


if __name__ == "__main__":
    main()
