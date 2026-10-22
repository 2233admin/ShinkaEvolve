"""Verdict rules: a closed, versioned set owned by core.

A domain or an agent picks a rule kind and gives its thresholds; it never
writes judging code. An experiment's plan freezes the kind, the version and
the thresholds, and the verdict reads nothing else: the rationale agreed with
the owner is kept for audit only. Changing what a rule does is a new version;
plans keep the version they froze.

Measurements in the evidence (scores, correct flags, a trial-corrected
difference) come from the owner-approved checker; facts such as whether the
budget is spent come from core and are never taken from a caller. Whether and
how a measurement is corrected for the number of trials is the domain's
capability, frozen with the checker; a rule only compares what it is given.

Labels are ``Status`` values ("supported", "refuted", "inconclusive"); an
"insufficient" outcome is "inconclusive".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

SUPPORTED = "supported"
REFUTED = "refuted"
INCONCLUSIVE = "inconclusive"

DSR_PBO = "dsr_pbo"
REACH_TARGET = "reach_target"
COUNTEREXAMPLE = "counterexample"
HOLDOUT_VS_BASELINE = "holdout_vs_baseline"

Judgement = tuple[str, str]


class TargetEvidence(BaseModel):
    """``reach_target``: the best score the checker accepted as correct (None
    when it accepted none) and whether the plan's budget is spent."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    best_score: Optional[float]
    budget_exhausted: StrictBool


class CounterexampleEvidence(BaseModel):
    """``counterexample``: whether the checker accepted a counterexample."""

    model_config = ConfigDict(extra="forbid")
    found: StrictBool


class HoldoutEvidence(BaseModel):
    """``holdout_vs_baseline``: the holdout metric minus the frozen baseline's,
    oriented so that higher is better. ``difference`` is the raw value;
    ``corrected_difference`` is the domain's trial correction of it, frozen
    with the checker (equal to ``difference`` where the domain corrects
    nothing). Only a raw difference <= 0 refutes, so more trials never make a
    refutation easier; the correction only gates support."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    difference: float
    corrected_difference: float

    @model_validator(mode="after")
    def _shrinks(self) -> "HoldoutEvidence":
        if self.corrected_difference > self.difference:
            raise ValueError(
                f"corrected_difference {self.corrected_difference:g} exceeds "
                f"the raw difference {self.difference:g}"
            )
        return self


def _dsr_pbo(rule: "Rule", evidence: Mapping[str, Any]) -> Judgement:
    # The quant rule as the skeptic has always applied it; it moves into the
    # quant preset with research-domains 1.5.
    from shinka.research.skeptic import Thresholds, decide

    label, reason = decide(dict(evidence), Thresholds(**rule.thresholds))
    return label.value, reason


def _reach_target(rule: "Rule", evidence: Mapping[str, Any]) -> Judgement:
    """Higher is better: the plan's target must exceed the seed's score."""
    ev = TargetEvidence.model_validate(evidence)
    target = rule.thresholds["target"]
    if ev.best_score is not None and ev.best_score >= target:
        return SUPPORTED, f"best score {ev.best_score:g} reaches the target {target:g}"
    best = "no correct result" if ev.best_score is None else f"best {ev.best_score:g}"
    if not ev.budget_exhausted:
        return INCONCLUSIVE, f"{best}, target {target:g} not reached yet"
    if rule.on_budget_exhausted == "refuted":
        return REFUTED, (
            f"budget spent with {best}: the method did not reach {target:g} "
            "within its budget"
        )
    return INCONCLUSIVE, f"budget spent with {best}, short of {target:g}"


def _counterexample(rule: "Rule", evidence: Mapping[str, Any]) -> Judgement:
    if CounterexampleEvidence.model_validate(evidence).found:
        return REFUTED, "the checker accepted a counterexample"
    return INCONCLUSIVE, "no counterexample found"


def _holdout_vs_baseline(rule: "Rule", evidence: Mapping[str, Any]) -> Judgement:
    ev = HoldoutEvidence.model_validate(evidence)
    margin = rule.thresholds["support_margin"]
    if ev.difference <= 0:
        return REFUTED, f"does not beat the baseline ({ev.difference:g} <= 0)"
    if ev.corrected_difference > margin:
        return SUPPORTED, (
            f"beats the baseline by {ev.corrected_difference:g} after the trial "
            f"correction, above {margin:g}"
        )
    return INCONCLUSIVE, (
        f"beats the baseline by {ev.difference:g}, {ev.corrected_difference:g} "
        f"after the trial correction, not above {margin:g}"
    )


def _check_dsr_pbo(th: Mapping[str, float]) -> None:
    for name in ("dsr_supported", "min_positive_fraction", "max_pbo"):
        if not 0.0 <= th[name] <= 1.0:
            raise ValueError(f"{name} must be within [0, 1], got {th[name]}")
    n = th["n_subperiods"]
    if n < 1 or n != int(n):
        raise ValueError(f"n_subperiods must be a whole number >= 1, got {n}")


def _check_target(th: Mapping[str, float]) -> None:
    if not math.isfinite(th["target"]):
        raise ValueError("target must be a finite number")


def _check_margin(th: Mapping[str, float]) -> None:
    margin = th["support_margin"]
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"support_margin must be a finite number >= 0, got {margin}")


@dataclass(frozen=True)
class _Spec:
    thresholds: FrozenSet[str]
    check: Callable[[Mapping[str, float]], None]
    judge: Callable[["Rule", Mapping[str, Any]], Judgement]


_SPECS: Dict[tuple[str, int], _Spec] = {
    (DSR_PBO, 1): _Spec(
        frozenset(
            {"dsr_supported", "min_positive_fraction", "n_subperiods", "max_pbo"}
        ),
        _check_dsr_pbo,
        _dsr_pbo,
    ),
    (REACH_TARGET, 1): _Spec(frozenset({"target"}), _check_target, _reach_target),
    (COUNTEREXAMPLE, 1): _Spec(frozenset(), lambda th: None, _counterexample),
    (HOLDOUT_VS_BASELINE, 1): _Spec(
        frozenset({"support_margin"}), _check_margin, _holdout_vs_baseline
    ),
}


class Rule(BaseModel):
    """A rule kind, its version and its thresholds, as a plan freezes them."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: str
    version: int
    thresholds: Dict[str, float] = Field(default_factory=dict)
    # reach_target only: what a spent budget without the target means.
    on_budget_exhausted: Optional[Literal["refuted", "insufficient"]] = None

    @model_validator(mode="after")
    def _known(self) -> "Rule":
        spec = _SPECS.get((self.kind, self.version))
        if spec is None:
            known = ", ".join(f"{k} v{v}" for k, v in _SPECS)
            raise ValueError(
                f"unknown rule {self.kind} v{self.version}; known: {known}"
            )
        given = set(self.thresholds)
        if given != spec.thresholds:
            missing = ", ".join(sorted(spec.thresholds - given)) or "none"
            extra = ", ".join(sorted(given - spec.thresholds)) or "none"
            raise ValueError(
                f"{self.kind} v{self.version} thresholds: missing {missing}; "
                f"unknown {extra}"
            )
        spec.check(self.thresholds)
        if (self.kind == REACH_TARGET) != (self.on_budget_exhausted is not None):
            raise ValueError(
                "on_budget_exhausted is required for reach_target and only for it"
            )
        return self


def judge(rule: Rule, evidence: Mapping[str, Any]) -> Judgement:
    """(label, reason) from the frozen rule and the evidence alone."""
    return _SPECS[(rule.kind, rule.version)].judge(rule, evidence)
