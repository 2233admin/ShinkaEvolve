"""Seed strategy for the example research direction.

`run_experiment` is called by the frozen evaluator; only the EVOLVE-BLOCK is
mutated.
"""

import numpy as np
import pandas as pd


# EVOLVE-BLOCK-START
def generate_positions(bars: pd.DataFrame) -> pd.Series:
    """Time-series momentum: long when the trailing 20-bar return is positive."""
    trailing = bars["close"].pct_change(20)
    return np.sign(trailing).fillna(0.0)


# EVOLVE-BLOCK-END


def run_experiment(bars: pd.DataFrame) -> pd.Series:
    positions = generate_positions(bars)
    return pd.Series(positions, index=bars.index, dtype=float).clip(-1.0, 1.0)
