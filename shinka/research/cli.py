"""`shinka_research` command line interface."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import tempfile
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shinka.research import grants, library
from shinka.research.notebook import (
    RESULTS_LOG,
    DataSplit,
    LockboxOpening,
    Notebook,
    Status,
    Verdict,
    append_log,
    experiment_dir,
)
from shinka.research.quant import QuantConfig, run_quant_eval
from shinka.research.rundb import count_evaluated, run_cost
from shinka.research.skeptic import (
    Thresholds,
    challenge,
    check_source,
    confirm,
    confirm_target,
    decide,
    render_report,
)
from shinka.research.scaffold import (
    SHINKA_CONFIG_FILENAME,
    load_config,
    prepare_data,
    scaffold_experiment,
    verify_experiment,
)
from shinka.research.runs import (
    RUN_LOG,
    detached,
    finish_run,
    hypothesis_evaluated,
    watch,
)
from shinka.research.state import (
    atomic_write_text,
    attach_worker,
    file_lock,
    hand_over,
    live_work,
    lock_path,
    record_exit,
    register_work,
)

PROGRAM_TEMPLATE = """# Research program

## Direction
{direction}

## Data
- File: <absolute path to CSV with timestamp + close (+ other columns)>
- Instrument, bar size, period: <...>
- Periods per year: <252 for daily equities, 365 for daily crypto, ...>

## Evaluator settings
- Costs: <bps per unit turnover>
- Lockbox fraction: 0.2 (sealed at the first scaffold, opened once by `confirm`)
- Validation fraction of the development data: 0.3, in-sample folds: 3

## Budget
- Max hypotheses per round: <n>
- Generations per experiment: <stage 2 / stage 3>
- USD per experiment: <...>; per direction: set by `init --budget-usd`

## Autonomy
- Autonomy: checkpoints   # or "autonomous": run until the budget is spent;
  # or "authorized-round": finish the round the owner granted, then pause

## Library
- Sources for this direction are in ../library/raw/ (papers, clips, notes, links.md)

## Verdict thresholds
- supported: DSR >= 0.95, >= 2/3 of validation sub-periods positive, PBO <= 0.5,
  beats buy-and-hold
- refuted: out-of-sample Sharpe <= 0
- otherwise: inconclusive

## Notes for the researcher agent
<prior knowledge, ideas to avoid, things to prioritise>
"""


def cmd_init(args: argparse.Namespace) -> None:
    root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(lock_path(root)):
        if Notebook.path(root).exists():
            sys.exit(f"{root} already has a notebook")
        Notebook(direction=args.direction, budget_usd=args.budget_usd).save(root)
    library.ensure(library.library_root(root))
    program = root / "program.md"
    if not program.exists():
        program.write_text(PROGRAM_TEMPLATE.format(direction=args.direction))
    print(root)


def cmd_hypothesis_add(args: argparse.Namespace) -> None:
    root = Path(args.dir)
    body = Path(args.file).read_text() if args.file else sys.stdin.read()
    with Notebook.edit(root) as nb:
        hyp = nb.add_hypothesis(root, args.title, body)
    print(hyp.id)


def cmd_hypothesis_status(args: argparse.Namespace) -> None:
    with Notebook.edit(Path(args.dir)) as nb:
        nb.set_status(args.hid, Status(args.status))


def _not_confirming(nb: Notebook) -> None:
    """Experiment changes wait while a lockbox confirmation is unfinished: when
    it finishes, it records over the hypotheses it froze."""
    if nb.data is not None and nb.data.unfinished:
        sys.exit(
            "a lockbox confirmation is unfinished; `confirm --yes` finishes it, and "
            "experiments wait until then"
        )


def cmd_scaffold(args: argparse.Namespace) -> None:
    root = Path(args.dir)
    # The experiment number and the sealed split are allocated here, so the
    # whole scaffold holds the direction's lock.
    with Notebook.edit(root) as nb:
        _not_confirming(nb)
        split = prepare_data(
            root,
            nb,
            Path(args.data) if args.data else None,
            args.timestamp_col,
            args.lockbox_fraction,
        )
        cfg = QuantConfig(
            data_path=split.dev_path,
            timestamp_col=split.timestamp_col,
            oos_fraction=args.oos_fraction,
            n_folds=args.n_folds,
            cost_bps=args.cost_bps,
            periods_per_year=args.periods_per_year,
            lookahead_checks=args.lookahead_checks,
        )
        exp, task_dir = scaffold_experiment(root, nb, args.hid, Path(args.initial), cfg)
    append_log(root, args.hid, exp.id, "scaffold", f"seed {Path(args.initial).name}")
    print(task_dir)


def direction_spend(root: Path, nb: Notebook) -> Tuple[float, bool]:
    """Recorded LLM spend (USD) across every experiment of the direction and
    whether all of it is priced; unpriced calls make it a lower bound."""
    costs = [
        run_cost(experiment_dir(root, h.id, e.id) / "results")
        for h in nb.hypotheses
        for e in h.experiments
    ]
    return sum(c.recorded_usd for c in costs), all(c.known for c in costs)


def _held(root: Path, live: List[Dict[str, Any]]) -> float:
    """What grant runs in progress may still spend: each stops at its cap."""
    held = 0.0
    for w in live:
        origin = w.get("origin") or {}
        if "cap" in origin:
            results = experiment_dir(root, w["hypothesis"], w["experiment"]) / "results"
            now = run_cost(results).recorded_usd
            held += max(0.0, origin["cost_before"] + origin["cap"] - now)
    return held


def cmd_grant(args: argparse.Namespace) -> None:
    if grants.in_agent_env():
        sys.exit(
            "a round grant is the owner's to give: from a terminal without "
            f"{grants.AGENT_ENV}, or from the workbench"
        )
    root = Path(args.dir).resolve()
    try:
        grant = grants.add_grant(
            root, args.hids, args.attempts, args.max_cost, args.allow, args.note
        )
    except grants.Refused as refused:
        sys.exit(str(refused))
    print(grants.terms(grant))


# The evolution command a run starts; the lease keeps the full command, so a
# detached supervisor runs exactly what was checked here.
SHINKA_RUN = ["shinka_run"]


def cmd_run(args: argparse.Namespace, extra: List[str]) -> None:
    if any("headless/" in a for a in extra):
        # Headless CLI agents work inside results_dir, which holds private
        # (out-of-sample) metrics; they would see the holdout.
        sys.exit(
            "headless/* mutation models are not allowed: they can read OOS metrics"
        )
    agent = grants.in_agent_env()
    if agent and args.grant is None:
        sys.exit(
            "an agent run needs a round the owner granted: `run --grant <id>` "
            "(`shinka_research grant` records one)"
        )
    root = Path(args.dir).resolve()
    # A run that ended unobserved is recorded first, so its leftover running
    # mark does not pass for this run's earlier status.
    for line in watch(root):
        print(line)
    with Notebook.edit(root) as nb:
        _not_confirming(nb)
        hyp, exp = _pick_experiment(nb, args.hid, args.eid)
        live = live_work(root)
        if any((w["hypothesis"], w["experiment"]) == (hyp.id, exp.id) for w in live):
            sys.exit(
                f"{hyp.id}/{exp.id} already has a run in progress; two runs would "
                "write the same programs.sqlite"
            )
        verify_experiment(root, nb, hyp.id, exp)
        spent = 0.0
        if nb.budget_usd is not None:
            spent, known = direction_spend(root, nb)
            if spent >= nb.budget_usd:
                sys.exit(
                    f"direction budget exhausted: {'' if known else '>='}${spent:.2f} "
                    f"of ${nb.budget_usd:.2f}"
                )
            # A lower bound cannot show that the budget still has room.
            if not known:
                sys.exit(
                    "direction budget cannot be checked: some runs have an unknown "
                    f"cost (recorded ${spent:.2f} of ${nb.budget_usd:.2f})"
                )
        exp_dir = experiment_dir(root, hyp.id, exp.id)
        results_dir = exp_dir / "results"
        work = uuid.uuid4().hex
        origin: Dict[str, Any] = {"env": "agent" if agent else "manual"}
        passthrough = extra
        if args.grant is not None:
            # Every refusal comes before the attempt is taken, and the attempt
            # before the lease, so no run goes without one.
            try:
                grant, used, tokens = grants.check_run(
                    root, args.grant, hyp.id, exp.id, extra, live
                )
            except grants.Refused as refused:
                sys.exit(str(refused))
            cap = grant["max_cost"]
            if nb.budget_usd is not None:
                # A run with no stop point (the owner's own, say) may spend any
                # of what is left, so nothing can be held for this one.
                open_ended = [w for w in live if "cap" not in (w.get("origin") or {})]
                if open_ended:
                    w = open_ended[0]
                    sys.exit(
                        f"{w['hypothesis']}/{w['experiment']} has a run in progress "
                        "with no stop point, so the direction budget cannot hold room "
                        "for a grant run; it waits until that run ends"
                    )
                # Rounded so the subtraction's float noise reads as no room.
                left = round(nb.budget_usd - spent - _held(root, live), 9)
                cap = min(cap, left)
                if cap <= 0:
                    sys.exit(
                        f"the direction budget has no room for a grant run: "
                        f"${spent:.2f} of ${nb.budget_usd:.2f} spent, the rest held "
                        "by grant runs in progress"
                    )
            # shinka_run counts what the experiment's database already spent
            # toward its cap when it resumes it.
            cost_before = run_cost(results_dir).recorded_usd
            stops_at = cost_before + cap
            passthrough = [part for token in tokens for part in ("--set", token)]
            passthrough += ["--set", f"evo.max_api_costs={stops_at!r}"]
            origin.update(
                grant=grant["id"],
                attempt=used + 1,
                of=grant["attempts"],
                cap=cap,
                cost_before=cost_before,
            )
            grants.take_attempt(
                root, grant, used + 1, hyp.id, exp.id, work, cap, stops_at
            )
        cmd = [
            *SHINKA_RUN,
            "--task-dir", str(exp_dir / "task"),
            "--results_dir", str(results_dir),
            "--num_generations", str(args.num_generations),
            "--config-fname", SHINKA_CONFIG_FILENAME,
            *passthrough,
        ]  # fmt: skip
        # The lease, with the checks that allowed it: confirm cannot freeze the
        # trial count while this run adds trials, a second run of this
        # experiment is refused, and the run's end is recorded from it. A run
        # joining another live run of the hypothesis shares its baseline, so the
        # hypothesis goes back to the status it had before any of them.
        joined = next((w for w in live if w["hypothesis"] == hyp.id), None)
        baseline = (
            {k: joined[k] for k in ("previous_status", "previous_verdict", "before")}
            if joined
            else {
                "previous_status": hyp.status.value,
                "previous_verdict": hyp.verdict.model_dump(mode="json")
                if hyp.verdict
                else None,
                "before": hypothesis_evaluated(root, hyp),
            }
        )
        register_work(
            root,
            hyp.id,
            exp.id,
            work_id=work,
            cmd=cmd,
            origin=origin,
            evaluated_before=count_evaluated(results_dir),
            **baseline,
        )
        hyp.status, hyp.verdict = Status.running, None

    # An agent's run outlives the window or page that asked for it.
    if args.detach or agent:
        try:
            supervisor = subprocess.Popen(
                [sys.executable, "-m", "shinka.research.runs", str(root), work],
                **detached(),
            )
        except BaseException:
            finish_run(root, work, 1)
            raise
        # The supervisor owns the run from here; this command may live on. If
        # the supervisor took over first, its own hand-over stands.
        hand_over(root, work, supervisor.pid, replacing=os.getpid())
        print(
            f"{hyp.id}/{exp.id} is running in the background (pid {supervisor.pid}); "
            f"its output goes to {exp_dir / RUN_LOG}, and `shinka_research watch "
            f"{root}` records its end if nothing else does"
        )
        return
    print("$ " + " ".join(cmd), flush=True)
    code = 1
    try:
        # subprocess.call, keeping the child's pid: the child adds the trials
        # and may outlive this command.
        with subprocess.Popen(cmd) as proc:
            try:
                attach_worker(root, work, proc.pid)
                code = proc.wait()
                record_exit(root, work, code)
            except BaseException:
                proc.kill()
                raise
    finally:
        # Always record what was evaluated, even on Ctrl-C, so trials are counted.
        finish_run(root, work, code)
    if code:
        sys.exit(code)


def _pick_experiment(nb: Notebook, hid: str, eid: Optional[str]):
    hyp = nb.get(hid)
    if not hyp.experiments:
        sys.exit(f"{hid} has no experiment; run `scaffold` first")
    if eid is None:
        return hyp, hyp.experiments[-1]
    return hyp, next(e for e in hyp.experiments if e.id == eid)


def cmd_probe(args: argparse.Namespace) -> None:
    """Score a program with the frozen evaluator, showing public metrics only."""
    root = Path(args.dir)
    # Probing is a manual search step and a trial for the DSR from the moment it
    # starts, so one that fails or is killed is counted too, and a lockbox
    # opened meanwhile has already frozen it. The run log records it in the same
    # lock, so the log and the count agree.
    with file_lock(lock_path(root)):
        with Notebook.edit(root) as nb:
            _not_confirming(nb)
            hyp, exp = _pick_experiment(nb, args.hid, args.eid)
            verify_experiment(root, nb, hyp.id, exp)
            exp.n_probed += 1
        who = "agent" if grants.in_agent_env() else "manual"
        append_log(root, hyp.id, exp.id, "probe", f"{Path(args.program).name}, {who}")
    cfg = load_config(experiment_dir(root, hyp.id, exp.id) / "task")
    with (
        tempfile.TemporaryDirectory() as tmp,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        metrics, correct, err = run_quant_eval(args.program, tmp, cfg)
    shown = {
        "correct": correct,
        "error": err,
        "combined_score": metrics.get("combined_score"),
        "public": metrics.get("public"),
        "text_feedback": metrics.get("text_feedback"),
    }
    print(json.dumps(shown, indent=2, default=float))


def cmd_skeptic(args: argparse.Namespace) -> None:
    root = Path(args.dir)
    nb = Notebook.load(root)
    _not_confirming(nb)
    hyp, exp = _pick_experiment(nb, args.hid, args.eid)
    th = _thresholds(args)
    # What the challenge starts from; a change to it meanwhile (a verdict, a
    # confirmation, a run, another challenge) is newer than this result.
    start = (hyp.status, hyp.verdict, exp.skeptic)
    report = challenge(root, nb, hyp.id, exp, th)
    label, reason = decide(report, th)
    # The challenge ran unlocked; its result goes onto the latest notebook.
    with Notebook.edit(root) as nb:
        _not_confirming(nb)
        hyp, exp = _pick_experiment(nb, hyp.id, exp.id)
        if (hyp.status, hyp.verdict, exp.skeptic) != start:
            sys.exit(
                f"{hyp.id} changed while the challenge ran (now {hyp.status.value}); "
                "nothing was recorded, challenge it again"
            )
        exp.skeptic = {
            **report,
            "rule_label": label.value,
            "rule_reason": reason,
            "thresholds": th.model_dump(
                include={"dsr_supported", "min_positive_fraction", "max_pbo"}
            ),
        }
        hyp.status, hyp.verdict = Status.challenged, None
    append_log(root, hyp.id, exp.id, "skeptic", f"rule says {label.value}: {reason}")
    out = experiment_dir(root, hyp.id, exp.id) / "skeptic.md"
    out.write_text(render_report(hyp.id, exp.id, report, label) + f"\n{reason}\n")
    print(out.read_text())


def _thresholds(args: argparse.Namespace) -> Thresholds:
    return Thresholds(
        dsr_supported=args.dsr,
        min_positive_fraction=args.min_positive_fraction,
        n_subperiods=args.subperiods,
        max_pbo=args.max_pbo,
    )


# Held for a confirm's whole run: a target found mid-scoring then belongs to a
# confirm that died, never to one still running.
CONFIRM_LOCK = ".confirm.lock"


def cmd_confirm(args: argparse.Namespace) -> None:
    root = Path(args.dir)
    entered = False
    try:
        with file_lock(root / CONFIRM_LOCK, timeout=0):
            entered = True
            _confirm(root, args)
    except TimeoutError:
        if entered:
            raise
        sys.exit("another confirm is running for this direction; wait for it to end")


def _confirmed(nb: Notebook, hid: str, report: Dict[str, Any]) -> None:
    assert nb.data is not None and nb.data.opening is not None
    hyp = nb.get(hid)
    hyp.confirmation = report
    hyp.status = Status.confirmed if report["passed"] else Status.unconfirmed
    nb.data.opening.target(hid).state = "done"


def _confirm(root: Path, args: argparse.Namespace) -> None:
    nb = Notebook.load(root)
    split = nb.data
    if split is None:
        sys.exit("this direction has no data yet")
    if split.opened and not split.unfinished:
        if split.opening is not None and not (root / CONFIRMATION).exists():
            # The results are in the notebook but that confirm died before
            # writing them out; only the events it did not log are added.
            logged = _confirm_events(root, split.opening.started)
            hids = [t.hypothesis for t in split.opening.targets]
            _publish(root, nb, [hid for hid in hids if hid not in logged])
            return
        sys.exit("the lockbox was already opened; start a new direction with new data")
    if split.unfinished:
        assert split.opening is not None
        left = [t.hypothesis for t in split.opening.targets if t.state != "done"]
        if not args.yes:
            sys.exit(
                f"The lockbox opening of {split.opening.started} did not finish. "
                f"This finishes it for {', '.join(left)} with the programs and rules "
                "it froze. Re-run with --yes to proceed."
            )
        check_source(split.opening.source_sha256, Path(args.data))
        th = Thresholds(**split.opening.thresholds)
    else:
        # The lockbox opens once, so every supported hypothesis is confirmed together.
        hids = [h.id for h in nb.hypotheses if h.status is Status.supported]
        if not hids:
            sys.exit("no supported hypotheses to confirm; the lockbox stays closed")
        if not args.yes:
            sys.exit(
                f"This opens the lockbox (from {split.lockbox_start}) once for the "
                f"whole direction and scores {', '.join(hids)}. Re-run with --yes "
                "to proceed."
            )
        th = _thresholds(args)
        # Every check that leaves the lockbox untouched, so a mistake here, such
        # as the wrong --data, keeps it sealed.
        targets = [confirm_target(root, nb, hid) for hid in hids]
        check_source(split.source_hash, Path(args.data))
        with Notebook.edit(root) as nb:
            assert nb.data is not None
            # Every trial counts toward the DSR, so the count is frozen only
            # once no run or probe is still adding to it.
            busy = [f"{w['hypothesis']}/{w['experiment']}" for w in live_work(root)]
            if busy:
                sys.exit(
                    f"trials are still being added ({', '.join(busy)}); the lockbox "
                    "stays sealed, confirm once that work has ended"
                )
            supported = [h.id for h in nb.hypotheses if h.status is Status.supported]
            if (
                nb.data.opened
                or [confirm_target(root, nb, h) for h in supported] != targets
            ):
                sys.exit(
                    "the direction changed while confirm checked it; the lockbox "
                    "stays sealed, run confirm again"
                )
            nb.data.opened = True
            nb.data.opening = LockboxOpening(
                source_sha256=nb.data.source_hash,
                thresholds=th.model_dump(),
                targets=targets,
            )
    split = nb.data
    assert split is not None and split.opening is not None
    # A target already scoring belongs to a confirm that died after its program
    # saw the lockbox, whether or not it got a result: running it again would be
    # a second look.
    recorded: Dict[str, Dict[str, Any]] = {
        t.hypothesis: {
            "experiment": t.experiment,
            "program_id": t.program_id,
            "lockbox_start": split.lockbox_start,
            "passed": False,
            "failure": "interrupted",
        }
        for t in split.opening.targets
        if t.state == "scoring"
    }
    with tempfile.TemporaryDirectory() as tmp:
        for t in [t for t in split.opening.targets if t.state == "pending"]:

            def started(hid: str = t.hypothesis) -> None:
                with Notebook.edit(root) as fresh:
                    assert fresh.data is not None and fresh.data.opening is not None
                    fresh.data.opening.target(hid).state = "scoring"

            recorded[t.hypothesis] = confirm(
                root,
                split,
                t,
                th,
                Path(args.data),
                split.opening.source_sha256,
                Path(tmp),
                started,
            )
    # Every result reaches the notebook at once: candidate programs can read it,
    # so a program scored later must not find an earlier one's lockbox figures.
    with Notebook.edit(root) as nb:
        assert nb.data is not None and nb.data.opening is not None
        for hid, report in recorded.items():
            _confirmed(nb, hid, report)
        nb.data.opening.finish()
    _publish(root, nb, list(recorded))


CONFIRMATION = "CONFIRMATION.md"


def _confirm_events(root: Path, since: str) -> set:
    """Hypotheses with a confirm event in the run log from ``since`` on."""
    path = root / RESULTS_LOG
    rows = path.read_text().splitlines()[1:] if path.exists() else []
    fields = [(row.split("\t") + [""] * 4)[:4] for row in rows]
    return {
        hid for time, hid, _eid, event in fields if event == "confirm" and time >= since
    }


def _publish(root: Path, nb: Notebook, log: List[str]) -> None:
    """The confirmation's run-log events, then CONFIRMATION.md, both read from
    the notebook. The Markdown comes last and in one step, so while it is
    missing some events may be too, and the next confirm adds them."""
    assert nb.data is not None and nb.data.opening is not None
    for hid in log:
        rep = nb.get(hid).confirmation or {}
        status = nb.get(hid).status.value
        note = (
            f"{status}: {rep['failure']} on the lockbox"
            if "failure" in rep
            else f"{status}: lockbox Sharpe {rep['lockbox_sharpe']:.2f}, "
            f"DSR {rep['dsr']:.3f}"
        )
        append_log(root, hid, rep["experiment"], "confirm", note)
    lines = [f"# Lockbox confirmation\n\nLockbox from {nb.data.lockbox_start}.\n"]
    for t in nb.data.opening.targets:
        hyp = nb.get(t.hypothesis)
        rows = "\n".join(
            f"| {k} | {v:.4g} |" if isinstance(v, float) else f"| {k} | {v} |"
            for k, v in (hyp.confirmation or {}).items()
        )
        lines.append(
            f"## {hyp.id}: {hyp.title} — {hyp.status.value}\n\n"
            f"| check | value |\n|---|---|\n{rows}\n"
        )
    atomic_write_text(root / CONFIRMATION, "\n".join(lines))
    for t in nb.data.opening.targets:
        print(f"{t.hypothesis}: {nb.get(t.hypothesis).status.value}")


def cmd_verdict(args: argparse.Namespace) -> None:
    root = Path(args.dir)
    with Notebook.edit(root) as nb:
        _not_confirming(nb)
        hyp, exp = _pick_experiment(nb, args.hid, args.eid)
        if exp.skeptic is None:
            sys.exit(f"{hyp.id}/{exp.id} has not been challenged; run `skeptic` first")
        label = Status(args.label or exp.skeptic["rule_label"])
        if args.label and label.value != exp.skeptic["rule_label"] and not args.reason:
            sys.exit("--reason is required when overriding the rule-based verdict")
        reason = args.reason or exp.skeptic["rule_reason"]
        nb.set_verdict(
            hyp.id,
            Verdict(
                label=label,
                reason=reason,
                evidence={"experiment": exp.id, **exp.skeptic},
            ),
        )
    append_log(root, hyp.id, exp.id, "verdict", f"{label.value}: {reason}")
    print(f"{hyp.id}: {label.value} — {reason}")


def _library(path: str) -> Path:
    """Accept a direction (its workspace library) or a workspace directory."""
    p = Path(path)
    lib = library.library_root(p) if Notebook.path(p).exists() else p / "library"
    return library.ensure(lib)


def cmd_library_add(args: argparse.Namespace) -> None:
    lib = _library(args.dir)
    for source in args.sources:
        print(library.add(lib, source).relative_to(lib).as_posix())


def cmd_library_pending(args: argparse.Namespace) -> None:
    for rel in library.pending(_library(args.dir)):
        print(rel)


def cmd_library_compiled(args: argparse.Namespace) -> None:
    lib = _library(args.dir)
    rels = library.pending(lib) if args.all else args.files
    library.mark_compiled(lib, rels)
    print(f"{len(rels)} marked compiled")


def cmd_library_log(args: argparse.Namespace) -> None:
    library.log(_library(args.dir), args.kind, args.text)


def _evidence(h) -> str:
    ev = h.verdict.evidence if h.verdict else {}
    keys = (("is_sharpe", "IS"), ("oos_sharpe", "val"), ("dsr", "DSR"), ("pbo", "PBO"))
    parts = [
        f"{label} {ev[k]:.2f}" for k, label in keys if isinstance(ev.get(k), float)
    ]
    return ", ".join(parts)


def _lockbox(split: DataSplit) -> str:
    if split.unfinished:
        return "OPENED, confirmation unfinished (confirm --yes finishes it)"
    return "OPENED" if split.opened else "sealed"


def cmd_brief(args: argparse.Namespace) -> None:
    """Digest for the researcher agent before proposing the next round."""
    root = Path(args.dir)
    nb = Notebook.load(root)
    lib = library.library_root(root)
    spent, known = direction_spend(root, nb)
    budget = f" of ${nb.budget_usd:.2f}" if nb.budget_usd is not None else ""
    out = [f"# Brief: {nb.direction}", ""]
    trials = _trial_count(root, nb)
    out.append(f"- Trials so far: {trials} (raise the bar for 'supported')")
    out.append(f"- LLM spend: {'' if known else '>='}${spent:.2f}{budget}")
    if nb.data:
        out.append(f"- Lockbox: {_lockbox(nb.data)} (from {nb.data.lockbox_start})")
    pend = library.pending(lib) if lib.exists() else []
    out.append(
        f"- Library: {len(library.wiki_pages(lib))} wiki pages, "
        f"{len(pend)} raw sources not compiled yet ({lib})"
    )
    out += ["", "## Hypotheses", ""]
    if not nb.hypotheses:
        out.append("(none yet)")
    for h in nb.hypotheses:
        line = f"- {h.id} [{h.status.value}] {h.title}"
        if h.verdict:
            line += f" — {h.verdict.reason}"
            ev = _evidence(h)
            if ev:
                line += f" ({ev})"
        out.append(line)
    log_path = root / RESULTS_LOG
    if log_path.exists():
        recent = log_path.read_text().splitlines()[1:][-args.recent :]
        out += ["", f"## Last {len(recent)} events ({RESULTS_LOG})", ""]
        out += [f"    {line}" for line in recent]
    print("\n".join(out))


def _evaluated(root: Path, hid: str, eid: str) -> int:
    return count_evaluated(experiment_dir(root, hid, eid) / "results")


def _trial_count(root: Path, nb: Notebook) -> int:
    """Every evaluated program in the run databases plus every probe, as the
    skeptic counts trials for the DSR; a run in progress counts as it goes,
    while the notebook's copy is written only once it returns."""
    return sum(
        _evaluated(root, h.id, e.id) + e.n_probed
        for h in nb.hypotheses
        for e in h.experiments
    )


def cmd_status(args: argparse.Namespace) -> None:
    root = Path(args.dir)
    nb = Notebook.load(root)
    print(f"Direction: {nb.direction}")
    print(f"Total trials: {_trial_count(root, nb)}")
    if nb.data:
        print(
            f"Lockbox: {_lockbox(nb.data)} (from {nb.data.lockbox_start}, "
            f"{nb.data.n_total - nb.data.n_dev} of {nb.data.n_total} bars)"
        )
    for h in nb.hypotheses:
        counts = [f"{e.id}({_evaluated(root, h.id, e.id)})" for e in h.experiments]
        exps = ", ".join(counts) or "-"
        line = f"{h.id}  {h.status.value:<12} {h.title}  [{exps}]"
        if h.verdict:
            line += f"\n      verdict: {h.verdict.reason}"
        print(line)


def cmd_watch(args: argparse.Namespace) -> None:
    """Prints one line per run whose end it recorded, and nothing otherwise."""
    path = Path(args.dir).resolve()
    if Notebook.path(path).exists():
        roots, named = [path], False
    else:
        roots = [d for d in sorted(path.iterdir()) if Notebook.path(d).exists()]
        named = True
    for root in roots:
        for line in watch(root):
            print(f"{root.name}: {line}" if named else line)


def cmd_ui(args: argparse.Namespace) -> None:
    from shinka.research import ui

    ui.serve(Path(args.workspace), args.port, not args.no_browser, args.runs_root)


def _add_threshold_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--dsr", type=float, default=0.95)
    s.add_argument("--min-positive-fraction", type=float, default=2.0 / 3.0)
    s.add_argument("--subperiods", type=int, default=3)
    s.add_argument("--max-pbo", type=float, default=0.5)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shinka_research",
        description="Hypothesis-driven research on top of ShinkaEvolve.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="Start a research direction")
    s.add_argument("dir")
    s.add_argument("--direction", required=True)
    s.add_argument(
        "--budget-usd", type=float, help="Stop runs once LLM spend hits this"
    )
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("hypothesis", help="Manage hypotheses")
    hsub = s.add_subparsers(dest="action", required=True)
    a = hsub.add_parser("add", help="Add a hypothesis (body from --file or stdin)")
    a.add_argument("dir")
    a.add_argument("--title", required=True)
    a.add_argument("--file")
    a.set_defaults(func=cmd_hypothesis_add)
    a = hsub.add_parser("status", help="Select or drop a hypothesis")
    a.add_argument("dir")
    a.add_argument("hid")
    a.add_argument("status", choices=["proposed", "selected", "dropped"])
    a.set_defaults(func=cmd_hypothesis_status)

    s = sub.add_parser("scaffold", help="Create a Shinka task for a hypothesis")
    s.add_argument("dir")
    s.add_argument("hid")
    s.add_argument("--initial", required=True, help="Seed program (initial.py)")
    s.add_argument(
        "--data",
        help="CSV with timestamp + close (required for the first experiment)",
    )
    s.add_argument(
        "--lockbox-fraction",
        type=float,
        default=0.2,
        help="Tail reserved for `confirm` (set once per direction)",
    )
    s.add_argument("--timestamp-col", default="timestamp")
    s.add_argument("--oos-fraction", type=float, default=0.3)
    s.add_argument("--n-folds", type=int, default=3)
    s.add_argument("--cost-bps", type=float, default=5.0)
    s.add_argument("--periods-per-year", type=float, default=252.0)
    s.add_argument("--lookahead-checks", type=int, default=5)
    s.set_defaults(func=cmd_scaffold)

    s = sub.add_parser(
        "run", help="Evolve an experiment (extra args after -- go to shinka_run)"
    )
    s.add_argument("dir")
    s.add_argument("hid")
    s.add_argument("--eid")
    s.add_argument("--num-generations", type=int, default=20)
    s.add_argument(
        "--detach",
        action="store_true",
        help="Keep running after this window closes; `watch` records an end "
        "nothing else saw",
    )
    s.add_argument(
        "--grant",
        help="The owner's round grant this run takes an attempt from; required "
        f"where {grants.AGENT_ENV} is set, where runs also always detach",
    )
    s.set_defaults(func=cmd_run, takes_extra=True)

    s = sub.add_parser(
        "probe", help="Score a program with the frozen evaluator (public only)"
    )
    s.add_argument("dir")
    s.add_argument("hid")
    s.add_argument("program")
    s.add_argument("--eid")
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("skeptic", help="Challenge an experiment on the holdout")
    s.add_argument("dir")
    s.add_argument("hid")
    s.add_argument("--eid")
    _add_threshold_args(s)
    s.set_defaults(func=cmd_skeptic)

    s = sub.add_parser(
        "confirm",
        help="Open the lockbox once and confirm supported hypotheses",
    )
    s.add_argument("dir")
    s.add_argument(
        "--data", required=True, help="The source CSV the direction was split from"
    )
    s.add_argument("--yes", action="store_true", help="Really open the lockbox")
    _add_threshold_args(s)
    s.set_defaults(func=cmd_confirm)

    s = sub.add_parser("verdict", help="Record the verdict (rule-based by default)")
    s.add_argument("dir")
    s.add_argument("hid")
    s.add_argument("--eid")
    s.add_argument("--label", choices=["supported", "refuted", "inconclusive"])
    s.add_argument("--reason")
    s.set_defaults(func=cmd_verdict)

    s = sub.add_parser("library", help="Workspace research library (raw -> wiki)")
    lsub = s.add_subparsers(dest="action", required=True)
    a = lsub.add_parser("add", help="Add files or URLs to library/raw")
    a.add_argument("dir", help="A direction or a workspace directory")
    a.add_argument("sources", nargs="+")
    a.set_defaults(func=cmd_library_add)
    a = lsub.add_parser("pending", help="List raw sources not compiled yet")
    a.add_argument("dir")
    a.set_defaults(func=cmd_library_pending)
    a = lsub.add_parser("compiled", help="Mark raw sources as compiled into the wiki")
    a.add_argument("dir")
    a.add_argument("files", nargs="*")
    a.add_argument("--all", action="store_true", help="Every pending source")
    a.set_defaults(func=cmd_library_compiled)
    a = lsub.add_parser("log", help="Append to library/log.md (lint, qa, finding)")
    a.add_argument("dir")
    a.add_argument("kind")
    a.add_argument("text")
    a.set_defaults(func=cmd_library_log)

    s = sub.add_parser("brief", help="Digest of the direction for the next round")
    s.add_argument("dir")
    s.add_argument("--recent", type=int, default=15)
    s.set_defaults(func=cmd_brief)

    s = sub.add_parser("status", help="Show the notebook")
    s.add_argument("dir")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser(
        "grant",
        help="Grant the agent one round: attempts, a per-run cap and the settings "
        "it may choose (the owner's, never from the agent environment)",
    )
    s.add_argument("dir")
    s.add_argument("hids", nargs="+", metavar="hid")
    s.add_argument("--attempts", type=int, required=True)
    s.add_argument(
        "--max-cost", type=float, required=True, help="USD each run may spend"
    )
    s.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="NS.FIELD",
        help="A shinka_run --set key the agent may choose (repeatable)",
    )
    s.add_argument("--note", default="")
    s.set_defaults(func=cmd_grant)

    s = sub.add_parser(
        "watch",
        help="Record the end of runs whose processes are all gone "
        "(a direction, or every direction of a workspace)",
    )
    s.add_argument("dir")
    s.set_defaults(func=cmd_watch)

    s = sub.add_parser("ui", help="Read-only research UI in the browser")
    s.add_argument(
        "workspace", help="Workspace directory containing research directions"
    )
    s.add_argument("--port", type=int, default=8765)
    s.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Explicit additional ordinary-run search root; omitted means workspace only",
    )
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_ui)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    args = build_parser().parse_args(argv)
    if getattr(args, "takes_extra", False):
        args.func(args, extra)
    else:
        if extra:
            sys.exit("extra arguments after -- are only accepted by `run`")
        args.func(args)


if __name__ == "__main__":
    main()
