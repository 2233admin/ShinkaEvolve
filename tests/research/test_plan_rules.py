import re
import sqlite3
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from shinka.research import cli
from shinka.research.notebook import ExperimentPlan, FileSet, Notebook, Status
from shinka.research.quant import file_sha256
from shinka.research.rules import Rule, judge
from shinka.research.scaffold import task_message_sha256

PLAN_THRESHOLDS = {
    "dsr_supported": 0.5,
    "min_positive_fraction": 0.5,
    "n_subperiods": 2,
    "max_pbo": 0.9,
}


def _target(on_budget_exhausted: str) -> Rule:
    return Rule(
        kind="reach_target",
        version=1,
        thresholds={"target": 2.6},
        on_budget_exhausted=on_budget_exhausted,
    )


@pytest.mark.parametrize(
    ("best", "exhausted", "on_exhausted", "label"),
    [
        (2.6, False, "refuted", "supported"),  # reaching the target exactly
        (2.7, True, "refuted", "supported"),
        (2.5, False, "refuted", "inconclusive"),  # not reached yet
        (2.5, True, "refuted", "refuted"),
        (None, True, "refuted", "refuted"),
        (2.5, True, "insufficient", "inconclusive"),
    ],
)
def test_reach_target_boundaries(
    best: float | None, exhausted: bool, on_exhausted: str, label: str
) -> None:
    evidence = {"best_score": best, "budget_exhausted": exhausted}
    assert judge(_target(on_exhausted), evidence)[0] == label


def test_counterexample_refutes_or_stays_open() -> None:
    rule = Rule(kind="counterexample", version=1)
    assert judge(rule, {"found": True})[0] == "refuted"
    assert judge(rule, {"found": False})[0] == "inconclusive"


@pytest.mark.parametrize(
    ("rule", "evidence"),
    [
        (_target("refuted"), {"best_score": float("nan"), "budget_exhausted": True}),
        (_target("refuted"), {"best_score": 2.5, "budget_exhausted": "yes"}),
        (Rule(kind="counterexample", version=1), {"found": "false"}),
    ],
    ids=["nan-score", "string-budget-flag", "string-found"],
)
def test_malformed_checker_evidence_is_refused(rule: Rule, evidence: dict) -> None:
    with pytest.raises(ValidationError):
        judge(rule, evidence)


def _holdout(margin: float = 0.1) -> Rule:
    return Rule(
        kind="holdout_vs_baseline", version=1, thresholds={"support_margin": margin}
    )


@pytest.mark.parametrize(
    ("raw", "corrected", "label"),
    [
        (0.3, 0.2, "supported"),
        (0.3, 0.1, "inconclusive"),  # reaching the margin exactly is not above it
        (0.3, -0.4, "inconclusive"),  # many trials: the correction blocks support
        (0.0, -0.4, "refuted"),  # only the raw difference refutes
        (-0.1, -0.1, "refuted"),
    ],
)
def test_holdout_vs_baseline_boundaries(
    raw: float, corrected: float, label: str
) -> None:
    evidence = {"difference": raw, "corrected_difference": corrected}
    assert judge(_holdout(), evidence)[0] == label


@pytest.mark.parametrize(
    "evidence",
    [
        {"difference": 0.1, "corrected_difference": 0.2},  # correction raised it
        {"difference": float("nan"), "corrected_difference": 0.1},
        {"corrected_difference": 0.1},
    ],
)
def test_holdout_evidence_that_cannot_be_judged_is_refused(evidence: dict) -> None:
    with pytest.raises(ValidationError):
        judge(_holdout(), evidence)


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "bootstrap", "version": 1},
        {"kind": "dsr_pbo", "version": 2, "thresholds": PLAN_THRESHOLDS},
        {"kind": "reach_target", "version": 1, "thresholds": {"target": 1.0}},
        {
            "kind": "counterexample",
            "version": 1,
            "on_budget_exhausted": "refuted",
        },
        {
            "kind": "holdout_vs_baseline",
            "version": 1,
            "thresholds": {"support_margin": -0.1},
        },
        {
            "kind": "holdout_vs_baseline",
            "version": 1,
            "thresholds": {"support_margin": float("inf")},
        },
        {"kind": "dsr_pbo", "version": 1, "thresholds": {"dsr_supported": 0.9}},
        {
            "kind": "dsr_pbo",
            "version": 1,
            "thresholds": {**PLAN_THRESHOLDS, "extra": 1.0},
        },
        {
            "kind": "dsr_pbo",
            "version": 1,
            "thresholds": {**PLAN_THRESHOLDS, "n_subperiods": 2.5},
        },
    ],
    ids=[
        "unknown-kind",
        "unknown-version",
        "target-without-budget-rule",
        "budget-rule-elsewhere",
        "negative-margin",
        "infinite-margin",
        "missing-threshold",
        "unknown-threshold",
        "fractional-subperiods",
    ],
)
def test_rules_outside_the_closed_set_are_refused(fields: dict) -> None:
    with pytest.raises(ValidationError):
        Rule(**fields)


def test_plan_paths_stay_inside_the_direction() -> None:
    for rel in ("../outside.py", "/abs.py", "C:/abs.py", "a\\b.py"):
        with pytest.raises(ValidationError):
            FileSet(files={rel: "0" * 64})


def _fake_results(results_dir: Path, seed: str) -> None:
    results_dir.mkdir(parents=True)
    with sqlite3.connect(results_dir / "programs.sqlite") as conn:
        conn.execute(
            "CREATE TABLE programs (id TEXT, code TEXT, combined_score REAL, "
            "correct INTEGER, metadata TEXT)"
        )
        conn.executemany(
            "INSERT INTO programs VALUES (?, ?, ?, ?, '{}')",
            [("p0", seed, 0.3, 1), ("p1", seed.replace("(20)", "(5)"), 0.5, 1)],
        )


@pytest.fixture
def planned(tmp_path: Path, example_dir: Path, bars_csv: Path) -> Path:
    """A scaffolded quant experiment with a frozen plan: one checker file, one
    data file of its own, the task message and a dsr_pbo rule."""
    root = tmp_path / "direction"
    cli.main(["init", str(root), "--direction", "momentum"])
    cli.main(
        ["hypothesis", "add", str(root), "--title", "TS momentum",
         "--file", str(example_dir / "hypothesis.md")]
    )  # fmt: skip
    cli.main(
        ["scaffold", str(root), "h001", "--initial", str(example_dir / "initial.py"),
         "--data", str(bars_csv)]
    )  # fmt: skip
    checker = root / "proposals" / "p001" / "check.py"
    checker.parent.mkdir(parents=True)
    checker.write_text("def check(artifact):\n    return True\n")
    instances = root / "plans" / "h001" / "instances.json"
    instances.parent.mkdir(parents=True)
    instances.write_text("[1, 2, 3]\n")
    task_dir = root / "experiments" / "h001" / "e01" / "task"
    _fake_results(task_dir.parent / "results", (example_dir / "initial.py").read_text())
    nb = Notebook.load(root)
    exp = nb.get("h001").experiments[0]
    exp.n_evaluated = 2
    exp.plan = ExperimentPlan(
        domain="quant",
        checker=FileSet(files={"proposals/p001/check.py": file_sha256(checker)}),
        data=FileSet(files={"plans/h001/instances.json": file_sha256(instances)}),
        task_message=task_message_sha256(task_dir),
        rule=Rule(kind="dsr_pbo", version=1, thresholds=PLAN_THRESHOLDS),
        claim="empirical",
        rationale="Refuted if the validation Sharpe is not above zero.",
    )
    nb.save(root)
    return root


def _rewrite_task_message(task_dir: Path) -> None:
    config = task_dir / "shinka.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["evo"]["task_sys_msg"] += "\nAlso try leverage."
    config.write_text(yaml.safe_dump(data), encoding="utf-8")


TAMPERING = {
    "checker": (
        lambda root, task: (root / "proposals/p001/check.py").write_text("x = 1\n"),
        "check.py",
        "changed since the plan was frozen",
    ),
    "checker-missing": (
        lambda root, task: (root / "proposals/p001/check.py").unlink(),
        "check.py",
        "plan file is missing",
    ),
    "data": (
        lambda root, task: (root / "plans/h001/instances.json").write_text("[]\n"),
        "instances.json",
        "changed since the plan was frozen",
    ),
    "task-message": (
        lambda root, task: _rewrite_task_message(task),
        "shinka.yaml",
        "task message changed",
    ),
    "wrapper": (
        lambda root, task: (task / "evaluate.py").write_text("# tampered\n"),
        "evaluate.py",
        "evaluator was modified",
    ),
}


@pytest.mark.parametrize("command", ["run", "probe", "skeptic", "verdict"])
@pytest.mark.parametrize("change", list(TAMPERING))
def test_a_changed_plan_file_stops_every_command_and_is_named(
    planned: Path, example_dir: Path, command: str, change: str
) -> None:
    root = planned
    if command == "verdict":
        cli.main(["skeptic", str(root), "h001"])
    tamper, name, why = TAMPERING[change]
    tamper(root, root / "experiments" / "h001" / "e01" / "task")
    argv = {
        "run": ["run", str(root), "h001"],
        "probe": ["probe", str(root), "h001", str(example_dir / "initial.py")],
        "skeptic": ["skeptic", str(root), "h001"],
        "verdict": ["verdict", str(root), "h001"],
    }[command]
    before = Notebook.load(root)
    with pytest.raises(RuntimeError, match=rf"{re.escape(name)}.*{why}"):
        cli.main(argv)
    assert Notebook.load(root) == before


def test_a_planned_challenge_follows_the_frozen_rule(
    planned: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = planned
    for flag in (["--dsr", "0.95"], ["--subperiods", "3"]):
        with pytest.raises(SystemExit, match=re.escape(flag[0])):
            cli.main(["skeptic", str(root), "h001", *flag])
    assert Notebook.load(root).get("h001").status is Status.selected

    cli.main(["skeptic", str(root), "h001"])
    rep = Notebook.load(root).get("h001").experiments[0].skeptic
    assert rep["thresholds"] == {
        "dsr_supported": 0.5,
        "min_positive_fraction": 0.5,
        "max_pbo": 0.9,
    }
    assert rep["rule"]["kind"] == "dsr_pbo" and rep["rule"]["version"] == 1
    skeptic_md = root / "experiments" / "h001" / "e01" / "skeptic.md"
    assert "Rule: dsr_pbo v1 (frozen by the plan)" in skeptic_md.read_text()
    capsys.readouterr()
    cli.main(["status", str(root)])
    assert "e01(2, dsr_pbo v1)" in capsys.readouterr().out


def test_an_unfrozen_challenge_matches_its_planned_twin(planned: Path) -> None:
    """The same evidence judged by command-line thresholds (legacy) and by the
    same thresholds frozen in a plan reaches the same verdict."""
    root = planned
    cli.main(["skeptic", str(root), "h001"])
    frozen = Notebook.load(root).get("h001").experiments[0].skeptic
    with Notebook.edit(root) as nb:
        hyp = nb.get("h001")
        hyp.experiments[0].plan, hyp.experiments[0].skeptic = None, None
        hyp.status = Status.selected
    cli.main(
        ["skeptic", str(root), "h001", "--dsr", "0.5", "--min-positive-fraction",
         "0.5", "--subperiods", "2", "--max-pbo", "0.9"]
    )  # fmt: skip
    legacy = Notebook.load(root).get("h001").experiments[0].skeptic
    assert "rule" not in legacy
    assert {k: v for k, v in frozen.items() if k != "rule"} == legacy
    skeptic_md = root / "experiments" / "h001" / "e01" / "skeptic.md"
    assert "Rule: unfrozen (legacy" in skeptic_md.read_text()


def test_a_plan_whose_rule_needs_checker_evidence_is_not_challenged(
    planned: Path,
) -> None:
    root = planned
    nb = Notebook.load(root)
    exp = nb.get("h001").experiments[0]
    assert exp.plan is not None
    exp.plan = exp.plan.model_copy(update={"rule": _target("refuted")})
    nb.save(root)
    with pytest.raises(SystemExit, match="1.3"):
        cli.main(["skeptic", str(root), "h001"])
    assert Notebook.load(root).get("h001").experiments[0].skeptic is None


def test_a_changed_file_is_named_before_a_non_dsr_rule_is_refused(
    planned: Path,
) -> None:
    root = planned
    nb = Notebook.load(root)
    exp = nb.get("h001").experiments[0]
    assert exp.plan is not None
    exp.plan = exp.plan.model_copy(
        update={"rule": Rule(kind="counterexample", version=1)}
    )
    nb.save(root)
    (root / "proposals/p001/check.py").write_text("x = 1\n")
    with pytest.raises(RuntimeError, match="check.py.*changed since the plan"):
        cli.main(["skeptic", str(root), "h001"])
