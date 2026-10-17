"""Generate synthetic daily bars with weak, regime-dependent momentum.

Usage: python make_data.py [out.csv] [--n 3000] [--seed 0]
"""

import argparse

import numpy as np
import pandas as pd


def make_bars(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Two regimes: trending (positive autocorrelation) and mean-reverting.
    regime = np.zeros(n, dtype=int)
    for t in range(1, n):
        regime[t] = regime[t - 1] if rng.random() > 0.01 else 1 - regime[t - 1]
    phi = np.where(regime == 0, 0.08, -0.05)
    eps = rng.normal(0.0, 0.01, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi[t] * r[t - 1] + eps[t]
    close = 100.0 * np.exp(np.cumsum(r))
    index = pd.bdate_range("2010-01-01", periods=n, name="timestamp")
    return pd.DataFrame({"close": close}, index=index)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", nargs="?", default="bars.csv")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    make_bars(a.n, a.seed).to_csv(a.out)
    print(a.out)
