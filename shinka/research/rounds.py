"""What the owner and both conversations read about agent work: the rounds,
their runs, the lockbox approvals and what waits on the owner (research-agent
3.2).

Read-only and lock-free, so the workbench can poll it: the leases first, then
the ledger, then the run log, because a run writes its attempt before its
lease and every grant or approval is logged before it is recorded; a mismatch
seen once is read again before it is reported. Only listed fields leave it: no
pids, commands or paths, nothing a lease keeps for its end (a cleared
verdict's evidence, say), and no refusal text, which names paths. Pending items
are what the files show; the agent's own requests are not recorded anywhere
yet (research-agent design.md).
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shinka.research import grants
from shinka.research.notebook import RESULTS_LOG, Notebook, experiment_dir
from shinka.research.rundb import run_cost
from shinka.research.runs import end_logged, unbuilt
from shinka.research.state import alive, read_text, work_entries


def direction_spend(
    root: Path, nb: Notebook, live: List[Dict[str, Any]] = ()
) -> Tuple[float, bool]:
    """Recorded LLM spend (USD) across every experiment of the direction and
    whether all of it is priced; unpriced calls make it a lower bound. An
    experiment whose run in ``live`` is still creating its database has spent
    nothing yet; any other database without tables stays an error."""
    starting = {(w["hypothesis"], w["experiment"]) for w in live}
    costs = []
    for h in nb.hypotheses:
        for e in h.experiments:
            results = experiment_dir(root, h.id, e.id) / "results"
            if (h.id, e.id) in starting and unbuilt(results):
                continue
            costs.append(run_cost(results))
    return sum(c.recorded_usd for c in costs), all(c.known for c in costs)


def _spent_since(root: Path, lease: Dict[str, Any]) -> float:
    results = experiment_dir(root, lease["hypothesis"], lease["experiment"]) / "results"
    if unbuilt(results):
        # A run that has only just created its database has recorded nothing.
        return 0.0
    return run_cost(results).recorded_usd - lease["origin"]["cost_before"]


def held(root: Path, live: List[Dict[str, Any]]) -> float:
    """What grant runs in progress may still spend: each stops at its cap."""
    return sum(
        max(0.0, w["origin"]["cap"] - _spent_since(root, w))
        for w in live
        if "cap" in (w.get("origin") or {})
    )


def _room(root: Path, nb: Notebook, live: List[Dict[str, Any]]) -> float:
    """The direction budget a new grant run could have; rounded so the
    subtraction's float noise reads as no room."""
    assert nb.budget_usd is not None
    spent, _known = direction_spend(root, nb, live)
    return round(nb.budget_usd - spent - held(root, live), 9)


def _running(lease: Dict[str, Any]) -> Dict[str, Any]:
    origin = lease.get("origin") or {}
    created = lease.get("created") or 0.0
    return {
        "hypothesis": lease["hypothesis"],
        "experiment": lease["experiment"],
        "env": origin.get("env", "manual"),
        "grant": origin.get("grant"),
        "attempt": origin.get("attempt"),
        "of": origin.get("of"),
        "cap": origin.get("cap"),
        "since": datetime.fromtimestamp(created, timezone.utc).isoformat(
            timespec="seconds"
        )
        if created
        else None,
    }


# How long an attempt with neither a lease nor an end may still be starting:
# `run` writes the attempt and the lease in one lock, moments apart, so only a
# stalled writer takes longer, and only a crash leaves it lost for good.
STARTING = 60.0


def _age(time: str) -> float:
    """Seconds since an ISO time the ledger recorded; infinite if unreadable."""
    try:
        return (
            datetime.now(timezone.utc) - datetime.fromisoformat(time)
        ).total_seconds()
    except (TypeError, ValueError):
        return float("inf")


def overview(root: Path) -> Dict[str, Any]:
    """The direction's agent work; never raises for a file it cannot read,
    reporting the problem instead."""
    try:
        view = _overview(Path(root))
        if view["problem"] == "mismatch":
            # A run may have started or ended between two reads.
            view = _overview(Path(root))
    except (sqlite3.Error, OSError, ValueError, KeyError, TypeError):
        # A file it cannot read, or one of an unexpected shape: a damaged
        # results database, a grant naming a hypothesis the notebook lacks.
        return _problem("unreadable", "files", [])
    return view


def _overview(root: Path) -> Dict[str, Any]:
    try:
        nb = Notebook.load(root)
        leases = work_entries(root)
        lives = {lease["id"]: alive(lease) for lease in leases}
        # Its end goes into the log before the lease goes: a lease whose end
        # is logged is just being dropped.
        pending = [
            {
                "kind": "unrecorded",
                "hypothesis": lease["hypothesis"],
                "experiment": lease["experiment"],
                "do": "shinka_research watch <dir>",
            }
            for lease in leases
            if not lives[lease["id"]] and not end_logged(root, lease)
        ]
    except (OSError, ValueError, KeyError, TypeError):
        return _problem("unreadable", "files", [])
    running = [_running(lease) for lease in leases if lives[lease["id"]]]
    try:
        ledger = grants.verify(root, grants.load(root), leases)
        rows = grants.log_rows(root)
    except grants.Refused as refused:
        unreadable = "cannot be read" in str(refused)
        problem = "unreadable" if unreadable else "mismatch"
        return _problem(problem, "ledger", running, pending)
    except (OSError, ValueError):
        return _problem("unreadable", "ledger", running, pending)

    ended = grants.ended_attempts(rows)
    by_id = {lease["id"]: lease for lease in leases}
    rounds = []
    for gid, grant in ledger.grants.items():
        runs = []
        for attempt in ledger.attempts[gid]:
            lease = by_id.get(attempt["work"])
            run = {
                "attempt": attempt["number"],
                "hypothesis": attempt["hypothesis"],
                "experiment": attempt["experiment"],
                "cap": attempt["cap"],
                "event": None,
                "spent_usd": None,
            }
            if attempt["work"] in ended:
                run.update(state="ended", event=ended[attempt["work"]][1])
            elif lease is not None and lives[lease["id"]]:
                run.update(state="running", spent_usd=_spent_since(root, lease))
            elif lease is not None:
                run["state"] = "unrecorded"
            elif _age(attempt["time"]) < STARTING:
                # Its attempt is written before its lease: still starting.
                run["state"] = "starting"
            else:
                # Taken, and its run never registered (a crash in between).
                run["state"] = "lost"
            runs.append(run)
        rounds.append(
            {
                "id": gid,
                "hypotheses": grant["hypotheses"],
                "attempts": grant["attempts"],
                "used": len(runs),
                "max_cost": grant["max_cost"],
                "allow": grant["allow"],
                "time": grant["time"],
                "note": grant["note"],
                "runs": runs,
            }
        )

    for i, r in enumerate(rounds):
        pending += [
            {
                "kind": "lost",
                "grant": r["id"],
                "attempt": run["attempt"],
                "do": "this attempt was taken and its run never started, so it "
                "has no result; nothing can recover it",
            }
            for run in r["runs"]
            if run["state"] == "lost"
        ]
        # A lost attempt will never end, so it does not hold the round open.
        done = r["used"] == r["attempts"] and all(
            run["state"] in ("ended", "lost") for run in r["runs"]
        )
        # A later grant for any of its hypotheses opened the next round.
        followed = any(
            set(r["hypotheses"]) & set(s["hypotheses"]) for s in rounds[i + 1 :]
        )
        if done and not followed:
            pending.append(
                {
                    "kind": "round-over",
                    "grant": r["id"],
                    "hypotheses": r["hypotheses"],
                    "do": "review the round; the next needs `shinka_research grant`",
                }
            )

    stuck = []
    open_grants = [r for r in rounds if r["used"] < r["attempts"]]
    if open_grants and nb.budget_usd is not None:
        live = [lease for lease in leases if lives[lease["id"]]]
        for lease in live:
            if "cap" not in (lease.get("origin") or {}):
                stuck.append(
                    {
                        "reason": "open-ended",
                        "hypothesis": lease["hypothesis"],
                        "experiment": lease["experiment"],
                        "do": "grant runs wait until this run with no stop point ends",
                    }
                )
        if not stuck:
            if not direction_spend(root, nb, live)[1]:
                stuck.append(
                    {
                        "reason": "budget-unknown",
                        "do": "price or remove the runs whose cost is unknown",
                    }
                )
            elif _room(root, nb, live) <= 0:
                stuck.append(
                    {"reason": "budget", "do": "the direction budget has no room left"}
                )
    for r in open_grants:
        grant = ledger.grants[r["id"]]
        for hid in r["hypotheses"]:
            for exp in nb.get(hid).experiments:
                if not grants.settings_kept(root, grant, hid, exp.id):
                    stuck.append(
                        {
                            "reason": "settings",
                            "grant": r["id"],
                            "hypothesis": hid,
                            "experiment": exp.id,
                            "do": "shinka.yaml changed since the grant; restore it or "
                            "grant anew",
                        }
                    )
                results = experiment_dir(root, hid, exp.id) / "results"
                if not unbuilt(results) and not run_cost(results).known:
                    stuck.append(
                        {
                            "reason": "cost-unknown",
                            "grant": r["id"],
                            "hypothesis": hid,
                            "experiment": exp.id,
                            "do": "resolve the experiment's unknown cost; a grant "
                            "run's cap cannot hold from a lower bound",
                        }
                    )
    pending += [{"kind": "stuck", **item} for item in stuck]

    opened_by = {use["opening"]: aid for aid, use in ledger.uses.items()}
    if nb.data is not None and nb.data.unfinished:
        assert nb.data.opening is not None
        aid = opened_by.get(nb.data.opening.started)
        pending.append(
            {
                "kind": "lockbox-unfinished",
                "approval": aid,
                "do": "shinka_research confirm <dir> --data <file> "
                + (f"--approval {aid}" if aid else "--yes"),
            }
        )
    opened = nb.data is not None and nb.data.opened
    approvals = [
        {
            "id": aid,
            "time": approval["time"],
            "note": approval["note"],
            "used": ledger.uses[aid]["opening"] if aid in ledger.uses else None,
            # Left unspent when the owner opened the lockbox by hand.
            "moot": opened and aid not in ledger.uses,
        }
        for aid, approval in ledger.approvals.items()
    ]

    kinds = {item["kind"] for item in pending}
    if kinds & {"stuck", "lockbox-unfinished"}:
        state = "blocked"
    elif any(run["env"] == "agent" for run in running) or any(
        run["state"] == "starting" for r in rounds for run in r["runs"]
    ):
        state = "running"
    elif "round-over" in kinds:
        state = "waiting"
    elif open_grants:
        state = "open"
    else:
        state = "idle"
    return {
        "state": state,
        "problem": None,
        "running": running,
        "grants": rounds,
        "approvals": approvals,
        "pending": pending,
    }


def _problem(
    problem: str,
    where: str,
    running: List[Dict[str, Any]],
    pending: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    item = (
        {
            "kind": "ledger",
            "problem": problem,
            "do": "compare ledger.json with results.tsv; grant runs and agent "
            "confirms are refused until they match",
        }
        if where == "ledger"
        else {
            "kind": "unreadable",
            "problem": problem,
            "do": "a direction file cannot be read: notebook.json, .work.json or "
            "an experiment's results database",
        }
    )
    return {
        "state": "unknown",
        "problem": problem,
        "running": running,
        "grants": [],
        "approvals": [],
        "pending": (pending or []) + [item],
    }


def summary(view: Dict[str, Any]) -> Dict[str, Any]:
    """The workspace overview's part of it."""
    return {"state": view["state"], "pending": len(view["pending"])}


# Run-log events an agent did not cause and should hear about: the owner's
# grants and lockbox approvals, and every run's end (a run that ends normally
# records its own end, so `watch` alone would never see it). The agent's own
# steps (scaffold, probe, skeptic, verdict, confirm, round-paused) are not news
# to it, and starting a run writes no event.
NEWS_EVENTS = ("grant", "approval", "run", "run-failed")


def news(
    root: Path, cursor: Optional[Dict[str, Any]]
) -> Tuple[List[str], Dict[str, Any]]:
    """The news events of the run log after ``cursor`` (``None``: from the
    start), and the cursor at the end of what was read. A cursor is the number
    of complete event lines read and the last of them; when the log no longer
    has that line there (rewritten by hand), every event is news again, so a
    lost or stale cursor can only repeat news, never hide it."""
    path = Path(root) / RESULTS_LOG
    # After the header; the last piece has no newline yet: empty, or a line
    # still being written, left for the next read.
    rows = read_text(path).split("\n")[1:-1] if path.exists() else []
    seen = 0
    if cursor is not None:
        count, last = cursor.get("rows"), cursor.get("last")
        if (
            isinstance(count, int)
            and 0 < count <= len(rows)
            and rows[count - 1] == last
        ):
            seen = count
    fresh = [
        row for row in rows[seen:] if (row.split("\t") + [""] * 4)[3] in NEWS_EVENTS
    ]
    return fresh, {"rows": len(rows), "last": rows[-1] if rows else None}
