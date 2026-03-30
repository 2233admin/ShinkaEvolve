"""Backfill model_posteriors into programs.sqlite metadata from bandit_state.pkl.

The bandit_state.pkl stores AsymmetricUCB internal arrays (n_submitted, n_completed,
s, divs, …) but does NOT store arm_names. Arm names are inferred in order of first
appearance from the programs table's metadata.model_name field, which matches the
order models were registered in the bandit when the run started.

The posterior written to each program's metadata is the *global* bandit posterior at
the time this script runs — i.e., the final learned weights, not the per-step weights
at program creation time.  This is sufficient for inspection / visualisation; a
per-step replay is not possible without the full history.

Usage
-----
    py -3.11 scripts/backfill_posteriors.py <results_dir>
    py -3.11 scripts/backfill_posteriors.py D:/projects/quant-terminal/shinka_evo/evo_results_tension
    py -3.11 scripts/backfill_posteriors.py D:/projects/quant-terminal/shinka_evo/evo_results_v3

    # dry-run (print posteriors, do not modify DB)
    py -3.11 scripts/backfill_posteriors.py <results_dir> --dry-run

    # override arm names (comma-separated, same order as bandit arms)
    py -3.11 scripts/backfill_posteriors.py <results_dir> --arms "model-a,model-b,model-c"
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_nan(obj):
    """Recursively replace NaN/Inf floats with None for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.ndarray):
        return _clean_nan(obj.tolist())
    return obj


# ---------------------------------------------------------------------------
# Bandit posterior computation
# ---------------------------------------------------------------------------

def _compute_posteriors_asymmetric_ucb(state: dict, n_arms: int) -> np.ndarray:
    """Re-implement AsymmetricUCB.posterior() from raw state arrays.

    Returns a probability vector of length n_arms.
    The bandit uses UCB scores with epsilon-greedy: the winner(s) get
    (1 - epsilon) weight, losers share epsilon.  We use epsilon=0 here
    to compute a "pure" probability ranking; what we really want is the
    normalised UCB score vector, not a one-hot selection.

    Strategy: compute UCB scores and softmax-normalise them so every arm
    gets a meaningful weight reflecting its relative UCB value.
    """
    n_submitted = state["n_submitted"]          # shape (n_arms,)
    n_completed = state["n_completed"]
    s           = state["s"]                    # accumulated reward (log or linear)
    divs        = state["divs"]                 # effective count for mean
    obs_max     = float(state.get("obs_max", np.nan))
    obs_min     = float(state.get("obs_min", np.nan))

    n = np.maximum(n_submitted, n_completed)    # effective pulls per arm

    # Detect exponential (log-space) mode: s contains -inf for unseen arms
    use_exp = np.any(np.isneginf(s) & (n > 0))  # simplistic heuristic
    # More reliable: if all s values are <= 0 or -inf, assume log-space
    # The actual flag is stored in AsymmetricUCB instance, not in pickle.
    # We detect it by checking whether obs_min is -inf (asymmetric + exp mode).
    use_exp_asym = not np.isfinite(obs_min)

    # Compute empirical mean per arm
    denom = np.maximum(divs, 1e-7)
    if use_exp_asym:
        # log-space mean
        mean_log = s - np.log(denom)
        # normalise relative to obs_max
        if np.isfinite(obs_max):
            norm_means = np.exp(mean_log - obs_max)
        else:
            norm_means = np.exp(mean_log)
    else:
        raw_mean = s / denom
        if np.isfinite(obs_min) and np.isfinite(obs_max) and (obs_max - obs_min) > 1e-9:
            norm_means = (raw_mean - obs_min) / (obs_max - obs_min)
        else:
            norm_means = raw_mean

    # UCB exploration bonus (standard UCB1 formula)
    t = float(n.sum())
    if t <= 0:
        # No data yet: uniform distribution
        return np.full(n_arms, 1.0 / n_arms)

    log_t = math.log(max(t, 2.0))
    n_safe = np.maximum(n, 1.0)
    exploration = np.sqrt(2.0 * log_t / n_safe)

    # Zero out exploration for arms never submitted (treat as worst)
    unseen = n <= 0
    if np.any(unseen):
        norm_means[unseen] = 0.0
        exploration[unseen] = 0.0

    ucb_scores = norm_means + exploration

    # Replace any remaining NaN/Inf with 0 before normalising
    ucb_scores = np.where(np.isfinite(ucb_scores), ucb_scores, 0.0)

    # Shift to non-negative then normalise to a probability vector
    ucb_min = ucb_scores.min()
    if ucb_min < 0:
        ucb_scores = ucb_scores - ucb_min

    total = ucb_scores.sum()
    if total <= 0:
        return np.full(n_arms, 1.0 / n_arms)

    return ucb_scores / total


def compute_posteriors(state: dict) -> np.ndarray:
    """Given a raw bandit_state dict, return the posterior probability vector."""
    n_arms = len(state["n_submitted"])
    return _compute_posteriors_asymmetric_ucb(state, n_arms)


# ---------------------------------------------------------------------------
# Arm-name inference
# ---------------------------------------------------------------------------

def infer_arm_names(conn: sqlite3.Connection, n_arms: int) -> List[str]:
    """Infer arm names from the DB.

    Arms are assigned in order of first appearance among programs that have
    a metadata.model_name field.  This matches the order models were first
    seen during the run (which is how the bandit typically registers arms).
    """
    cur = conn.execute(
        "SELECT metadata FROM programs WHERE metadata IS NOT NULL ORDER BY timestamp ASC"
    )
    seen: list = []
    seen_set: set = set()
    for (meta_text,) in cur:
        try:
            meta = json.loads(meta_text)
        except (json.JSONDecodeError, TypeError):
            continue
        mn = meta.get("model_name")
        if mn and mn not in seen_set:
            seen.append(mn)
            seen_set.add(mn)
        if len(seen) >= n_arms:
            break

    if len(seen) == n_arms:
        return seen

    # Fallback: pad with generic names
    while len(seen) < n_arms:
        seen.append(f"arm_{len(seen)}")
    return seen[:n_arms]


# ---------------------------------------------------------------------------
# Per-program metadata enrichment
# ---------------------------------------------------------------------------

def _build_posterior_payload(
    arm_names: List[str],
    posteriors: np.ndarray,
    state: dict,
) -> dict:
    """Build the model_posteriors dict plus summary stats."""
    payload: dict = {}

    # Main posteriors: {model_name: probability}
    payload["model_posteriors"] = {
        name: _clean_nan(float(prob))
        for name, prob in zip(arm_names, posteriors)
    }

    # Additional bandit stats for reference
    n_sub  = state["n_submitted"]
    n_comp = state["n_completed"]
    divs   = state["divs"]
    s      = state["s"]
    denom  = np.maximum(divs, 1e-7)

    # Mean reward per arm (raw, in whatever space the bandit uses)
    raw_means = (s / denom).tolist()

    payload["bandit_stats"] = {
        name: {
            "n_submitted": _clean_nan(float(n_sub[i])),
            "n_completed": _clean_nan(float(n_comp[i])),
            "raw_mean":    _clean_nan(float(raw_means[i])),
        }
        for i, name in enumerate(arm_names)
    }

    return payload


# ---------------------------------------------------------------------------
# Main backfill logic
# ---------------------------------------------------------------------------

def backfill(
    results_dir: Path,
    arm_names_override: Optional[List[str]] = None,
    dry_run: bool = False,
) -> None:
    pkl_path = results_dir / "bandit_state.pkl"
    db_path  = results_dir / "programs.sqlite"

    if not pkl_path.exists():
        print(f"ERROR: bandit_state.pkl not found: {pkl_path}")
        sys.exit(1)
    if not db_path.exists():
        print(f"ERROR: programs.sqlite not found: {db_path}")
        sys.exit(1)

    # Load bandit state
    with open(pkl_path, "rb") as f:
        state = pickle.load(f)

    n_arms = len(state["n_submitted"])
    print(f"Loaded bandit_state.pkl: {n_arms} arm(s)")
    print(f"  n_submitted : {state['n_submitted']}")
    print(f"  n_completed : {state['n_completed']}")
    print(f"  s           : {state['s']}")
    print(f"  divs        : {state['divs']}")
    print(f"  baseline    : {state['baseline']}")
    print(f"  obs_max     : {state['obs_max']}")
    print(f"  obs_min     : {state['obs_min']}")

    # Compute posteriors
    posteriors = compute_posteriors(state)
    print(f"\nComputed posteriors: {posteriors}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Resolve arm names
    if arm_names_override:
        if len(arm_names_override) != n_arms:
            print(
                f"ERROR: --arms provided {len(arm_names_override)} names "
                f"but bandit has {n_arms} arms"
            )
            conn.close()
            sys.exit(1)
        arm_names = arm_names_override
        print(f"\nArm names (from --arms override): {arm_names}")
    else:
        arm_names = infer_arm_names(conn, n_arms)
        print(f"\nArm names (inferred from DB): {arm_names}")

    # Build the payload to store in every program's metadata
    posterior_payload = _build_posterior_payload(arm_names, posteriors, state)
    print("\nPosterior payload to write:")
    print(json.dumps(posterior_payload, indent=2))

    if dry_run:
        print("\n[dry-run] No changes written.")
        conn.close()
        return

    # Update every program
    cur = conn.execute("SELECT id, metadata FROM programs")
    rows = cur.fetchall()
    total = len(rows)
    updated = 0

    for row in rows:
        prog_id   = row["id"]
        meta_text = row["metadata"]
        try:
            meta = json.loads(meta_text) if meta_text else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}

        meta["model_posteriors"] = posterior_payload["model_posteriors"]
        meta["bandit_stats"]     = posterior_payload["bandit_stats"]

        conn.execute(
            "UPDATE programs SET metadata = ? WHERE id = ?",
            (json.dumps(_clean_nan(meta)), prog_id),
        )
        updated += 1

    conn.commit()
    conn.close()

    print(f"\nDone: updated {updated}/{total} programs in {db_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backfill model_posteriors from bandit_state.pkl into programs.sqlite"
    )
    parser.add_argument(
        "results_dir",
        help="Path to the results directory containing bandit_state.pkl and programs.sqlite",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying the database",
    )
    parser.add_argument(
        "--arms",
        default=None,
        help="Comma-separated arm names in bandit order (overrides auto-inference from DB)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: not a directory: {results_dir}")
        sys.exit(1)

    arm_names_override = None
    if args.arms:
        arm_names_override = [a.strip() for a in args.arms.split(",")]

    backfill(
        results_dir=results_dir,
        arm_names_override=arm_names_override,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
