"""Research notebook: direction -> hypotheses -> experiments -> verdicts.

Plain files under one direction directory so humans and agents can read and
diff them::

    <direction>/
      program.md              human-written direction, constraints, budget
      notebook.json           machine-readable index (this module)
      hypotheses/<hid>.md     one falsifiable hypothesis per file
      experiments/<hid>/<eid>/{task,results}
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

NOTEBOOK_FILENAME = "notebook.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Status(str, Enum):
    proposed = "proposed"
    selected = "selected"
    running = "running"
    challenged = "challenged"
    supported = "supported"
    refuted = "refuted"
    inconclusive = "inconclusive"
    confirmed = "confirmed"
    unconfirmed = "unconfirmed"
    dropped = "dropped"


VERDICTS = {Status.supported, Status.refuted, Status.inconclusive}


class Verdict(BaseModel):
    label: Status
    reason: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    created: str = Field(default_factory=_now)


class DataSplit(BaseModel):
    """Direction-wide data. The lockbox (the tail of the source file) is never
    copied into experiments; only `confirm` reads it, once per direction.

    The source path is deliberately not stored: candidate programs can read
    this notebook, so `confirm` takes the file again and checks its hash."""

    source_hash: str
    timestamp_col: str
    dev_path: str
    n_total: int
    n_dev: int
    lockbox_start: str
    opened: bool = False


class Experiment(BaseModel):
    id: str
    created: str = Field(default_factory=_now)
    evaluator_hash: str
    data_path: str
    data_hash: str
    n_evaluated: int = 0
    n_probed: int = 0
    skeptic: Optional[Dict[str, Any]] = None


class Hypothesis(BaseModel):
    id: str
    title: str
    status: Status = Status.proposed
    created: str = Field(default_factory=_now)
    experiments: List[Experiment] = Field(default_factory=list)
    verdict: Optional[Verdict] = None
    confirmation: Optional[Dict[str, Any]] = None


class Notebook(BaseModel):
    direction: str
    created: str = Field(default_factory=_now)
    budget_usd: Optional[float] = None
    data: Optional[DataSplit] = None
    hypotheses: List[Hypothesis] = Field(default_factory=list)

    # ---- persistence -----------------------------------------------------

    @staticmethod
    def path(root: Path) -> Path:
        return Path(root) / NOTEBOOK_FILENAME

    @classmethod
    def load(cls, root: Path) -> "Notebook":
        p = cls.path(root)
        if not p.exists():
            raise FileNotFoundError(f"No research notebook at {p}; run `init` first")
        return cls.model_validate_json(p.read_text())

    def save(self, root: Path) -> None:
        self.path(root).write_text(self.model_dump_json(indent=2) + "\n")

    # ---- queries ---------------------------------------------------------

    def get(self, hid: str) -> Hypothesis:
        for h in self.hypotheses:
            if h.id == hid:
                return h
        raise KeyError(f"Unknown hypothesis {hid!r}")

    @property
    def total_trials(self) -> int:
        """Variants evaluated across the whole direction (for DSR)."""
        return sum(
            e.n_evaluated + e.n_probed for h in self.hypotheses for e in h.experiments
        )

    # ---- mutations -------------------------------------------------------

    def add_hypothesis(self, root: Path, title: str, body: str) -> Hypothesis:
        hid = f"h{len(self.hypotheses) + 1:03d}"
        hyp = Hypothesis(id=hid, title=title)
        self.hypotheses.append(hyp)
        hyp_dir = Path(root) / "hypotheses"
        hyp_dir.mkdir(parents=True, exist_ok=True)
        (hyp_dir / f"{hid}.md").write_text(f"# {hid}: {title}\n\n{body.strip()}\n")
        return hyp

    def set_status(self, hid: str, status: Status) -> None:
        """Human triage only: proposed <-> selected <-> dropped."""
        triage = {Status.proposed, Status.selected, Status.dropped}
        hyp = self.get(hid)
        if status not in triage or hyp.status not in triage:
            raise ValueError(
                f"{hid} is {hyp.status.value}; only proposed/selected/dropped can be "
                "set by hand (verdicts come from `verdict`)"
            )
        hyp.status = status

    def set_verdict(self, hid: str, verdict: Verdict) -> None:
        if verdict.label not in VERDICTS:
            raise ValueError(f"{verdict.label} is not a verdict")
        hyp = self.get(hid)
        hyp.verdict = verdict
        hyp.status = verdict.label

    def add_experiment(
        self, hid: str, evaluator_hash: str, data_path: str, data_hash: str
    ) -> Experiment:
        hyp = self.get(hid)
        exp = Experiment(
            id=f"e{len(hyp.experiments) + 1:02d}",
            evaluator_hash=evaluator_hash,
            data_path=data_path,
            data_hash=data_hash,
        )
        hyp.experiments.append(exp)
        if hyp.status in (Status.proposed, Status.selected):
            hyp.status = Status.selected
        return exp


RESULTS_LOG = "results.tsv"
_LOG_HEADER = "time\thypothesis\texperiment\tevent\tnote\n"


def append_log(root: Path, hid: str, eid: str, event: str, note: str) -> None:
    """One line per research event, autoresearch's results.tsv style."""
    path = Path(root) / RESULTS_LOG
    if not path.exists():
        path.write_text(_LOG_HEADER)
    clean = " ".join(note.split())
    with path.open("a") as f:
        f.write(f"{_now()}\t{hid}\t{eid}\t{event}\t{clean}\n")


def experiment_dir(root: Path, hid: str, eid: str) -> Path:
    return Path(root) / "experiments" / hid / eid
