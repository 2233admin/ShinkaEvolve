"""Run one candidate program on one view of the bars, in its own process.

Invoked by ``shinka.research.quant`` as ``python -I -m shinka.research.child``.
The evaluator process never imports candidate code: each view runs in a fresh
interpreter and only a float array crosses back (``.npy``, no pickle), so the
candidate cannot patch the scorer or carry state between views.
"""

import argparse
import importlib.util
import sys

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--program", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--timestamp-col", required=True)
    p.add_argument("--n-bars", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from shinka.research.quant import load_bars, validate_positions

    bars = load_bars(a.data, a.timestamp_col).iloc[: a.n_bars]
    spec = importlib.util.spec_from_file_location("program", a.program)
    if spec is None or spec.loader is None:
        sys.exit(f"cannot load {a.program}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    positions = module.run_experiment(bars.copy())
    err = validate_positions(positions, bars)
    if err:
        sys.exit(err)
    np.save(a.out, positions.to_numpy(dtype=float), allow_pickle=False)


if __name__ == "__main__":
    main()
