"""Research domains: which kind of research a direction does.

A direction's domain is written by ``init`` and never changes; every
experiment in the direction belongs to it, since trial counts, variance
estimates and the lockbox are direction-wide and would mix units otherwise.
Notebooks written before domains existed have none and read as ``quant``.
"""

from __future__ import annotations

from typing import Dict

QUANT = "quant"
OPEN = "open"
MATH = "math"
DEFAULT = QUANT


class UnknownDomain(ValueError):
    """A notebook names a domain this version does not know. Only that
    direction is unreadable; the workbench shows the error on its row."""


QUANT_PROGRAM = """# Research program

<!-- The owner gives the wish (`init --direction`) and the budget. The agent
drafts Direction, Constraints and Notes; sections marked (owner) are the
owner's alone: the agent never writes them. -->

## Direction
{direction}
<agent: what the wish asks, in testable terms, and what answer would settle it>

## Constraints
<agent: what the research must not do or assume (instruments, horizons, data,
methods); the owner corrects them>

## Data (owner)
- File: <absolute path to CSV with timestamp + close (+ other columns)>
- Instrument, bar size, period: <...>
- Periods per year: <252 for daily equities, 365 for daily crypto, ...>

## Evaluator settings (owner)
- Costs: <bps per unit turnover>
- Lockbox fraction: 0.2 (sealed at the first scaffold, opened once by `confirm`)
- Validation fraction of the development data: 0.3, in-sample folds: 3

## Budget (owner)
- Max hypotheses per round: <n>
- Generations per experiment: <stage 2 / stage 3>
- USD per experiment: <...>; per direction: set by `init --budget-usd`

## Autonomy (owner)
- Autonomy: checkpoints   # or "autonomous": run until the budget is spent;
  # or "authorized-round": finish the round the owner granted, then pause

## Library
- Sources for this direction are in ../library/raw/ (papers, clips, notes, links.md)

## Verdict thresholds (owner)
- supported: DSR >= 0.95, >= 2/3 of validation sub-periods positive, PBO <= 0.5,
  beats buy-and-hold
- refuted: out-of-sample Sharpe <= 0
- otherwise: inconclusive

## Notes for the researcher agent
<agent: prior knowledge, ideas to avoid, things to prioritise; the owner may add>
"""

# Domains without a quant preset: no bars, Sharpe or lockbox. Thresholds come
# from each hypothesis's frozen experiment plan, not from this file.
GENERAL_PROGRAM = """# Research program

<!-- The owner gives the wish (`init --direction`) and the budget. The agent
drafts Direction, Constraints and Notes; sections marked (owner) are the
owner's alone: the agent never writes them. -->

## Direction
{direction}
<agent: what the wish asks, in testable terms, and what answer would settle it>

## Constraints
<agent: what the research must not do or assume (data, instances, methods,
resources); the owner corrects them>

## Budget (owner)
- Max hypotheses per round: <n>
- Generations per experiment: <...>
- USD per experiment: <...>; per direction: set by `init --budget-usd`

## Autonomy (owner)
- Autonomy: checkpoints   # or "autonomous": run until the budget is spent;
  # or "authorized-round": finish the round the owner granted, then pause

## Approvals
- The checker that scores candidates is drafted by the agent and approved by
  the owner once for this direction; it is frozen from then on.
- Each hypothesis's experiment plan (task message, claim type, budget, data,
  verdict rule and thresholds, prior art) is frozen when the owner grants its
  round. Changing any of it is a new experiment that needs a new grant.

## Library
- Sources for this direction are in ../library/raw/ (papers, clips, notes, links.md)

## Notes for the researcher agent
<agent: prior knowledge, ideas to avoid, things to prioritise; the owner may add>
"""

PROGRAMS: Dict[str, str] = {
    QUANT: QUANT_PROGRAM,
    OPEN: GENERAL_PROGRAM,
    MATH: GENERAL_PROGRAM,
}
NAMES = tuple(PROGRAMS)


def check(domain: str) -> str:
    if domain not in PROGRAMS:
        raise UnknownDomain(
            f"unknown research domain {domain!r}; this version knows {', '.join(NAMES)}"
        )
    return domain


def program(domain: str, direction: str) -> str:
    """The direction's ``program.md`` as ``init`` writes it."""
    return PROGRAMS[check(domain)].format(direction=direction)
