"""`shinka_research` command line interface."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from shinka.research import library
from shinka.research.notebook import (
    RESULTS_LOG,
    Notebook,
    Status,
    Verdict,
    append_log,
    experiment_dir,
)
from shinka.research.quant import QuantConfig, run_quant_eval
from shinka.research.rundb import best_score, count_evaluated, run_cost
from shinka.research.skeptic import (
    Thresholds,
    challenge,
    confirm,
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
from shinka.research.state import file_lock, lock_path

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
- Autonomy: checkpoints   # or "autonomous": run until the budget is spent

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


def cmd_scaffold(args: argparse.Namespace) -> None:
    root = Path(args.dir)
    # The experiment number and the sealed split are allocated here, so the
    # whole scaffold holds the direction's lock.
    with Notebook.edit(root) as nb:
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


def cmd_run(args: argparse.Namespace, extra: List[str]) -> None:
    if any("headless/" in a for a in extra):
        # Headless CLI agents work inside results_dir, which holds private
        # (out-of-sample) metrics; they would see the holdout.
        sys.exit(
            "headless/* mutation models are not allowed: they can read OOS metrics"
        )
    root = Path(args.dir)
    with Notebook.edit(root) as nb:
        hyp, exp = _pick_experiment(nb, args.hid, args.eid)
        verify_experiment(root, nb, hyp.id, exp)
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
        previous_status, previous_verdict = hyp.status, hyp.verdict
        evaluated_before = count_evaluated(results_dir)
        hyp.status, hyp.verdict = Status.running, None

    cmd = [
        "shinka_run",
        "--task-dir", str(exp_dir / "task"),
        "--results_dir", str(results_dir),
        "--num_generations", str(args.num_generations),
        "--config-fname", SHINKA_CONFIG_FILENAME,
        *extra,
    ]  # fmt: skip
    print("$ " + " ".join(cmd), flush=True)
    code = 1
    try:
        code = subprocess.call(cmd)
    finally:
        # Always record what was evaluated, even on Ctrl-C, so trials are counted.
        with Notebook.edit(root) as nb:
            hyp, exp = _pick_experiment(nb, hyp.id, exp.id)
            exp.n_evaluated = count_evaluated(results_dir)
            # Undo the running mark only while it is still there: another
            # command may have moved the hypothesis on during the run.
            unchanged = hyp.status is Status.running
            if code and exp.n_evaluated == evaluated_before and unchanged:
                hyp.status, hyp.verdict = previous_status, previous_verdict
        best = best_score(results_dir)
        cost = run_cost(results_dir)
        append_log(
            root, hyp.id, exp.id, "run" if code == 0 else "run-failed",
            f"{exp.n_evaluated - evaluated_before} new programs, "
            f"{exp.n_evaluated} total, best IS score "
            f"{'n/a' if best is None else f'{best:.3f}'}, "
            f"spend {'' if cost.known else '>='}${cost.recorded_usd:.2f}",
        )  # fmt: skip
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
    nb = Notebook.load(root)
    hyp, exp = _pick_experiment(nb, args.hid, args.eid)
    verify_experiment(root, nb, hyp.id, exp)
    cfg = load_config(experiment_dir(root, hyp.id, exp.id) / "task")
    with (
        tempfile.TemporaryDirectory() as tmp,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        metrics, correct, err = run_quant_eval(args.program, tmp, cfg)
    # Probing is a manual search step: count it as a trial for the DSR. The
    # evaluation ran unlocked, so the count goes onto the latest notebook.
    with Notebook.edit(root) as nb:
        _pick_experiment(nb, hyp.id, exp.id)[1].n_probed += 1
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
    hyp, exp = _pick_experiment(nb, args.hid, args.eid)
    th = _thresholds(args)
    # What the challenge starts from; a change to it meanwhile (a verdict, a
    # confirmation, a run, another challenge) is newer than this result.
    start = (hyp.status, hyp.verdict, exp.skeptic)
    report = challenge(root, nb, hyp.id, exp, th)
    label, reason = decide(report, th)
    # The challenge ran unlocked; its result goes onto the latest notebook.
    with Notebook.edit(root) as nb:
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


def cmd_confirm(args: argparse.Namespace) -> None:
    root = Path(args.dir)
    nb = Notebook.load(root)
    if nb.data is None:
        sys.exit("this direction has no data yet")
    if nb.data.opened:
        sys.exit("the lockbox was already opened; start a new direction with new data")
    # The lockbox opens once, so every supported hypothesis is confirmed together.
    hids = [h.id for h in nb.hypotheses if h.status is Status.supported]
    if not hids:
        sys.exit("no supported hypotheses to confirm; the lockbox stays closed")
    if not args.yes:
        sys.exit(
            f"This opens the lockbox (from {nb.data.lockbox_start}) once for the whole "
            f"direction and scores {', '.join(hids)}. Re-run with --yes to proceed."
        )
    th = _thresholds(args)
    with tempfile.TemporaryDirectory() as tmp:
        reports = {
            hid: confirm(root, nb, hid, th, Path(args.data), Path(tmp)) for hid in hids
        }
    nb.data.opened = True
    lines = [f"# Lockbox confirmation\n\nLockbox from {nb.data.lockbox_start}.\n"]
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
            root, hid, rep["experiment"], "confirm",
            f"{nb.get(hid).status.value}: lockbox Sharpe {rep['lockbox_sharpe']:.2f}, "
            f"DSR {rep['dsr']:.3f}",
        )  # fmt: skip
    (root / "CONFIRMATION.md").write_text("\n".join(lines))
    for hid in hids:
        print(f"{hid}: {nb.get(hid).status.value}")


def cmd_verdict(args: argparse.Namespace) -> None:
    root = Path(args.dir)
    with Notebook.edit(root) as nb:
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


def cmd_brief(args: argparse.Namespace) -> None:
    """Digest for the researcher agent before proposing the next round."""
    root = Path(args.dir)
    nb = Notebook.load(root)
    lib = library.library_root(root)
    spent, known = direction_spend(root, nb)
    budget = f" of ${nb.budget_usd:.2f}" if nb.budget_usd is not None else ""
    out = [f"# Brief: {nb.direction}", ""]
    out.append(f"- Trials so far: {nb.total_trials} (raise the bar for 'supported')")
    out.append(f"- LLM spend: {'' if known else '>='}${spent:.2f}{budget}")
    if nb.data:
        state = "OPENED" if nb.data.opened else "sealed"
        out.append(f"- Lockbox: {state} (from {nb.data.lockbox_start})")
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


def cmd_status(args: argparse.Namespace) -> None:
    nb = Notebook.load(Path(args.dir))
    print(f"Direction: {nb.direction}")
    print(f"Total trials: {nb.total_trials}")
    if nb.data:
        state = "OPENED" if nb.data.opened else "sealed"
        print(
            f"Lockbox: {state} (from {nb.data.lockbox_start}, "
            f"{nb.data.n_total - nb.data.n_dev} of {nb.data.n_total} bars)"
        )
    for h in nb.hypotheses:
        exps = ", ".join(f"{e.id}({e.n_evaluated})" for e in h.experiments) or "-"
        line = f"{h.id}  {h.status.value:<12} {h.title}  [{exps}]"
        if h.verdict:
            line += f"\n      verdict: {h.verdict.reason}"
        print(line)


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
