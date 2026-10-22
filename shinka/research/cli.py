"""`shinka_research` command line interface."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import tempfile
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import yaml

from shinka.research import domains, grants, library, rules
from shinka.research.notebook import (
    RESULTS_LOG,
    DataSplit,
    Experiment,
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
    trial_count,
)
from shinka.research.scaffold import (
    SHINKA_CONFIG_FILENAME,
    load_config,
    prepare_data,
    scaffold_experiment,
    verify_experiment,
)
from shinka.research.rounds import direction_spend, held, news, overview
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
    read_text,
)


def cmd_init(args: argparse.Namespace) -> None:
    root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(lock_path(root)):
        if Notebook.path(root).exists():
            sys.exit(f"{root} already has a notebook")
        Notebook(
            direction=args.direction, domain=args.domain, budget_usd=args.budget_usd
        ).save(root)
    library.ensure(library.library_root(root))
    program = root / "program.md"
    if not program.exists():
        program.write_text(domains.program(args.domain, args.direction))
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
        if nb.domain != domains.QUANT:
            sys.exit(
                f"{root} is a {nb.domain} direction; `scaffold` builds quant "
                "experiments only"
            )
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


@contextlib.contextmanager
def _refused() -> Iterator[None]:
    """A refusal from the owner's approvals ends the command with its reason."""
    try:
        yield
    except grants.Refused as refused:
        sys.exit(str(refused))


def _the_owners(what: str) -> None:
    if grants.in_agent_env():
        sys.exit(
            f"{what} is the owner's to give: from a terminal without "
            f"{grants.AGENT_ENV}, or from the workbench"
        )


def cmd_grant(args: argparse.Namespace) -> None:
    _the_owners("a round grant")
    with _refused():
        grant = grants.add_grant(
            Path(args.dir).resolve(),
            args.hids,
            args.attempts,
            args.max_cost,
            args.allow,
            args.note,
        )
    print(grants.terms(grant))


def cmd_approve(args: argparse.Namespace) -> None:
    _the_owners("a lockbox approval")
    with _refused():
        approval = grants.add_approval(Path(args.dir).resolve(), args.note)
    print(grants.approval_terms(approval))


# The evolution command a run starts; the lease keeps the full command, so a
# detached supervisor runs exactly what was checked here.
SHINKA_RUN = ["shinka_run"]

# The model lists shinka_run calls. evo.task_sys_msg holds the hypothesis text,
# which may name a headless/ model without configuring one.
_MODEL_FIELDS = (
    "llm_models",
    "meta_llm_models",
    "novelty_llm_models",
    "prompt_llm_models",
)


def _headless_models(task_dir: Path) -> List[str]:
    """The headless/* models the task's shinka.yaml configures in either of
    shinka_run's names for that section (``evo``, ``evo_config``). A file that
    does not parse is left to shinka_run, which refuses it."""
    try:
        loaded = yaml.safe_load(
            (task_dir / SHINKA_CONFIG_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(loaded, dict):
        return []
    found = []
    for section in (loaded.get("evo"), loaded.get("evo_config")):
        if not isinstance(section, dict):
            continue
        for name in _MODEL_FIELDS:
            models = section.get(name) or []
            if isinstance(models, str):
                models = [models]
            if isinstance(models, list):
                found += [
                    m
                    for m in models
                    if isinstance(m, str) and m.startswith("headless/")
                ]
    return found


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
        task_dir = experiment_dir(root, hyp.id, exp.id) / "task"
        headless = _headless_models(task_dir)
        if headless:
            sys.exit(
                "headless/* mutation models are not allowed: they can read OOS "
                f"metrics ({', '.join(headless)} in {task_dir / SHINKA_CONFIG_FILENAME})"
            )
        spent = 0.0
        if nb.budget_usd is not None:
            spent, known = direction_spend(root, nb, live)
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
            with _refused():
                grant, used, tokens = grants.check_run(
                    root, args.grant, hyp.id, exp.id, extra, live
                )
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
                left = round(nb.budget_usd - spent - held(root, live), 9)
                cap = min(cap, left)
                if cap <= 0:
                    sys.exit(
                        f"the direction budget has no room for a grant run: "
                        f"${spent:.2f} of ${nb.budget_usd:.2f} spent, the rest held "
                        "by grant runs in progress"
                    )
            # shinka_run counts what the experiment's database already spent
            # toward its cap when it resumes it, by the same rule; from a lower
            # bound the run could spend past the grant's cap.
            prior = run_cost(results_dir)
            if not prior.known:
                sys.exit(
                    f"{hyp.id}/{exp.id} has spent an unknown amount (at least "
                    f"${prior.recorded_usd:.2f}), so a grant run's cap cannot "
                    "hold; the owner must resolve the unknown cost first"
                )
            cost_before = prior.recorded_usd
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
    # A changed file is named before anything else, whatever the rule.
    verify_experiment(root, nb, hyp.id, exp)
    plan = exp.plan
    if plan is None:
        # Unfrozen (legacy): thresholds from the command line, as always.
        th = _thresholds(args)
    else:
        given = _given_thresholds(args)
        if given:
            sys.exit(
                f"{hyp.id}/{exp.id}'s rule and thresholds are frozen by its plan; "
                f"drop {', '.join(given)}"
            )
        if plan.rule.kind != rules.DSR_PBO:
            sys.exit(
                f"{hyp.id}/{exp.id}'s rule {plan.rule.kind} is judged from the "
                "checker's evidence, which `skeptic` cannot gather yet "
                "(research-domains 1.3)"
            )
        th = Thresholds(**plan.rule.thresholds)
    # What the challenge starts from; a change to it meanwhile (a verdict, a
    # confirmation, a run, another challenge) is newer than this result.
    start = (hyp.status, hyp.verdict, exp.skeptic)
    report = challenge(root, nb, hyp.id, exp, th)
    if plan is None:
        label, reason = decide(report, th)
    else:
        verdict, reason = rules.judge(plan.rule, report)
        label = Status(verdict)
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
        if plan is not None:
            exp.skeptic["rule"] = plan.rule.model_dump()
        hyp.status, hyp.verdict = Status.challenged, None
    append_log(root, hyp.id, exp.id, "skeptic", f"rule says {label.value}: {reason}")
    out = experiment_dir(root, hyp.id, exp.id) / "skeptic.md"
    frozen = (
        "frozen by the plan"
        if exp.plan is not None
        else "legacy, thresholds from the command line"
    )
    out.write_text(
        render_report(hyp.id, exp.id, report, label)
        + f"\nRule: {_rule_name(exp)} ({frozen})\n\n{reason}\n"
    )
    print(out.read_text())


def _rule_name(exp: Experiment) -> str:
    """The rule an experiment is judged by; experiments without a plan are
    unfrozen (legacy) and take thresholds from the command line."""
    if exp.plan is None:
        return "unfrozen"
    return f"{exp.plan.rule.kind} v{exp.plan.rule.version}"


def _given_thresholds(args: argparse.Namespace) -> List[str]:
    return [
        flag
        for flag in _THRESHOLD_FLAGS
        if getattr(args, flag[2:].replace("-", "_")) is not None
    ]


def _thresholds(args: argparse.Namespace) -> Thresholds:
    given = {
        field: getattr(args, flag[2:].replace("-", "_"))
        for flag, field in _THRESHOLD_FLAGS.items()
    }
    return Thresholds(**{k: v for k, v in given.items() if v is not None})


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
    approval = args.approval
    if grants.in_agent_env() and approval is None:
        sys.exit(
            "an agent confirm needs the owner's lockbox approval: `confirm "
            "--approval <id>` (`shinka_research approve` records one); --yes is "
            "not enough here"
        )
    # The owner's approval stands for --yes.
    proceed = args.yes or approval is not None
    nb = Notebook.load(root)
    split = nb.data
    if split is None:
        sys.exit("this direction has no data yet")
    if split.opened and not split.unfinished:
        if split.opening is not None and not (root / CONFIRMATION).exists():
            if approval is not None:
                with _refused():
                    grants.check_opened_by(root, approval, split.opening.started)
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
        if not proceed:
            sys.exit(
                f"The lockbox opening of {split.opening.started} did not finish. "
                f"This finishes it for {', '.join(left)} with the programs and rules "
                "it froze. Re-run with --yes to proceed."
            )
        if approval is not None:
            with _refused():
                grants.check_opened_by(root, approval, split.opening.started)
        check_source(split.opening.source_sha256, Path(args.data))
        th = Thresholds(**split.opening.thresholds)
    else:
        # The lockbox opens once, so every supported hypothesis is confirmed together.
        hids = [h.id for h in nb.hypotheses if h.status is Status.supported]
        if not hids:
            sys.exit("no supported hypotheses to confirm; the lockbox stays closed")
        if not proceed:
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
            opening = LockboxOpening(
                source_sha256=nb.data.source_hash,
                thresholds=th.model_dump(),
                targets=targets,
            )
            if approval is not None:
                # Spent in the same lock, before the opening is saved.
                with _refused():
                    grants.use_approval(root, approval, opening.started)
            nb.data.opened = True
            nb.data.opening = opening
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
        if exp.plan is not None:
            # Unfrozen (legacy) experiments keep recording without this check.
            verify_experiment(root, nb, hyp.id, exp)
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
    by = "agent" if grants.in_agent_env() else "human"
    for source in args.sources:
        print(library.add(lib, source, by).relative_to(lib).as_posix())


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


def _sources_line(lib: Path) -> str:
    """Where the library's knowledge comes from: only shown, never enforced."""
    src = library.sources(lib)
    added = f"{src.agent} added by the agent, {src.human} by people"
    if src.unrecorded:
        added += f", {src.unrecorded} with no recorded origin"
    last = f"last added {src.last_added}" if src.last_added else "none added yet"
    pages = len(library.wiki_pages(lib))
    findings = len(library.finding_pages(lib))
    return f"- Sources: {added}; {last}; {findings} of {pages} wiki pages are findings"


def cmd_brief(args: argparse.Namespace) -> None:
    """Digest for the researcher agent before proposing the next round. With
    ``--since``, it also lists the news that consumer has not read and marks it
    read, after printing it."""
    root = Path(args.dir)
    nb = Notebook.load(root)
    fresh, cursor = news(root, _seen(root, args.since)) if args.since else ([], None)
    lib = library.library_root(root)
    spent, known = direction_spend(root, nb, live_work(root))
    budget = f" of ${nb.budget_usd:.2f}" if nb.budget_usd is not None else ""
    out = [f"# Brief: {nb.direction}", ""]
    trials = trial_count(root, nb)
    out.append(f"- Trials so far: {trials} (raise the bar for 'supported')")
    out.append(f"- LLM spend: {'' if known else '>='}${spent:.2f}{budget}")
    if nb.data:
        out.append(f"- Lockbox: {_lockbox(nb.data)} (from {nb.data.lockbox_start})")
    pend = library.pending(lib) if lib.exists() else []
    out.append(
        f"- Library: {len(library.wiki_pages(lib))} wiki pages, "
        f"{len(pend)} raw sources not compiled yet ({lib})"
    )
    out.append(_sources_line(lib))
    if args.since:
        out += ["", f"## News since the last brief ({args.since})", ""]
        out += [f"- {_news_line(row)}" for row in fresh] or ["(none)"]
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
    # The same reading of the rounds the workbench shows.
    view = overview(root)
    out += ["", f"## Agent rounds ({view['state']})", ""]
    for r in view["grants"]:
        out.append(
            f"- {r['id']}: {' '.join(r['hypotheses'])}, {r['used']} of "
            f"{r['attempts']} attempts taken, each stops at ${r['max_cost']:g}"
        )
    for item in view["pending"]:
        about = ", ".join(
            f"{key} {' '.join(value) if isinstance(value, list) else value}"
            for key, value in item.items()
            if key not in ("kind", "do") and value is not None
        )
        out.append(
            f"- pending {item['kind']}{f' ({about})' if about else ''}: {item['do']}"
        )
    if not view["grants"] and not view["pending"]:
        out.append("(no round granted yet)")
    log_path = root / RESULTS_LOG
    if log_path.exists():
        recent = log_path.read_text().splitlines()[1:][-args.recent :]
        out += ["", f"## Last {len(recent)} events ({RESULTS_LOG})", ""]
        out += [f"    {line}" for line in recent]
    print("\n".join(out))
    if args.since:
        # Only once printed: a brief that never reached its reader leaves the
        # news unread, so the next watch reports it again.
        seen = _seen_path(root, args.since)
        seen.parent.mkdir(exist_ok=True)
        atomic_write_text(seen, json.dumps(cursor))


# Each news consumer's cursor into the run log (`brief --since NAME`).
WATCH_DIR = ".watch"


def _consumer(name: str) -> str:
    """A news consumer's name, which names its cursor file."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", name):
        raise argparse.ArgumentTypeError(
            f"{name!r}: use lowercase letters, digits and dashes, e.g. terminal"
        )
    return name


def _seen_path(root: Path, name: str) -> Path:
    return Path(root) / WATCH_DIR / f"{name}.json"


def _seen(root: Path, name: str) -> Optional[Dict[str, Any]]:
    """A consumer's cursor: none when it has not read any yet, or when its
    file is not a cursor (repeats news rather than hiding it; the next brief
    rewrites it). Any other read error propagates: reported as news it would
    wake the agent on every tick, while as a failure it reaches the job's
    failure handling."""
    try:
        cursor = json.loads(read_text(_seen_path(root, name)))
    except (FileNotFoundError, ValueError):
        return None
    return cursor if isinstance(cursor, dict) else None


def _news_line(row: str) -> str:
    time, hid, eid, event, note = (row.split("\t") + [""] * 5)[:5]
    return f"{time} {event}" + (f" {hid}/{eid}" if hid else "") + f": {note}"


def _evaluated(root: Path, hid: str, eid: str) -> int:
    return count_evaluated(experiment_dir(root, hid, eid) / "results")


def cmd_status(args: argparse.Namespace) -> None:
    root = Path(args.dir)
    nb = Notebook.load(root)
    print(f"Direction: {nb.direction}")
    print(f"Total trials: {trial_count(root, nb)}")
    if nb.data:
        print(
            f"Lockbox: {_lockbox(nb.data)} (from {nb.data.lockbox_start}, "
            f"{nb.data.n_total - nb.data.n_dev} of {nb.data.n_total} bars)"
        )
    for h in nb.hypotheses:
        counts = [
            f"{e.id}({_evaluated(root, h.id, e.id)}, {_rule_name(e)})"
            for e in h.experiments
        ]
        exps = ", ".join(counts) or "-"
        line = f"{h.id}  {h.status.value:<12} {h.title}  [{exps}]"
        if h.verdict:
            line += f"\n      verdict: {h.verdict.reason}"
        print(line)


def cmd_watch(args: argparse.Namespace) -> None:
    """Prints one line per run whose end it recorded, and nothing otherwise.
    With ``--since``, it prints instead the news that consumer has not read
    with ``brief --since`` and exits 3 when there is any; it never marks news
    read, so a wake that never reached the agent is reported again."""
    path = Path(args.dir).resolve()
    if Notebook.path(path).exists():
        roots, named = [path], False
    else:
        roots = [d for d in sorted(path.iterdir()) if Notebook.path(d).exists()]
        named = True
    unread = False
    for root in roots:
        recorded = watch(root)
        lines = (
            [_news_line(row) for row in news(root, _seen(root, args.since))[0]]
            if args.since
            else recorded
        )
        for line in lines:
            print(f"{root.name}: {line}" if named else line)
        unread = unread or bool(lines and args.since)
    if unread:
        sys.exit(3)


def cmd_ui(args: argparse.Namespace) -> None:
    from shinka.research import ui

    ui.serve(Path(args.workspace), args.port, not args.no_browser, args.runs_root)


# Flag -> Thresholds field. The flags default to None so a command can tell
# thresholds given on the command line, which a frozen plan refuses, from the
# defaults (``Thresholds``'s own).
_THRESHOLD_FLAGS = {
    "--dsr": "dsr_supported",
    "--min-positive-fraction": "min_positive_fraction",
    "--subperiods": "n_subperiods",
    "--max-pbo": "max_pbo",
}


def _add_threshold_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--dsr", type=float, help="default 0.95")
    s.add_argument("--min-positive-fraction", type=float, help="default 2/3")
    s.add_argument("--subperiods", type=int, help="default 3")
    s.add_argument("--max-pbo", type=float, help="default 0.5")


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
    s.add_argument(
        "--domain",
        choices=domains.NAMES,
        default=domains.DEFAULT,
        help="What kind of research; written once, never changed "
        "(open: no preset, the agent drafts everything)",
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
    s.add_argument(
        "--approval",
        help="The owner's lockbox approval this opening spends (instead of --yes); "
        f"required where {grants.AGENT_ENV} is set",
    )
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
    s.add_argument(
        "--since",
        type=_consumer,
        metavar="NAME",
        help="also list the news consumer NAME has not read, and mark it read",
    )
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
        "approve",
        help="Approve opening the lockbox once (the owner's, never from the agent "
        "environment)",
    )
    s.add_argument("dir")
    s.add_argument("--note", default="")
    s.set_defaults(func=cmd_approve)

    s = sub.add_parser(
        "watch",
        help="Record the end of runs whose processes are all gone "
        "(a direction, or every direction of a workspace)",
    )
    s.add_argument("dir")
    s.add_argument(
        "--since",
        type=_consumer,
        metavar="NAME",
        help="print the news consumer NAME has not read with `brief --since` "
        "(owner grants and approvals, run ends) and exit 3 if there is any",
    )
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
