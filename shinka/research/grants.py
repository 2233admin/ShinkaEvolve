"""The owner's approvals for agent work: round grants and lockbox approvals.

In the agent execution environment (``SHINKA_RESEARCH_AGENT`` set) every
``run`` takes an attempt from a round grant the owner gave; a grant names its
hypotheses, how many runs it allows, where each run stops spending and which
``shinka_run`` settings the agent may choose. Opening the lockbox there takes a
lockbox approval, spent by the opening it allowed. Grants, approvals and their
uses live in ``ledger.json``; each grant's and approval's terms also go into
the run log, and the run log's end events name the attempt each grant run
used, so a ledger edited by hand no longer matches the log and what it
allows is refused. Someone with a shell as the same user can still forge
both; the check makes bypass and tampering discoverable, not impossible
(design.md).
"""

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

import yaml

from shinka.research.notebook import (
    RESULTS_LOG,
    Notebook,
    _now,
    append_log,
    experiment_dir,
)
from shinka.research.quant import file_sha256
from shinka.research.scaffold import SHINKA_CONFIG_FILENAME
from shinka.research.state import (
    atomic_write_text,
    file_lock,
    live_work,
    lock_path,
    read_text,
)

AGENT_ENV = "SHINKA_RESEARCH_AGENT"
LEDGER = "ledger.json"

# Settings no grant lets the agent choose: what they set belongs to the grant
# or to `run`, or would move evaluation, the trial database or the interpreter
# away from what is frozen and counted.
_NEVER = {
    "evo.max_api_costs": "a grant run stops at the grant's cap or the direction's "
    "remaining budget",
    "evo.results_dir": "`run` sets it",
    "evo.num_generations": "`run --num-generations` sets it",
    "evo.init_program_path": "it would replace the scaffolded seed",
    "job.eval_program_path": "it would bypass the frozen evaluator",
    "db.db_path": "trials and spend would leave the experiment's results",
    "job.python_executable": "it would choose the interpreter that evaluates",
    "job.conda_env": "it would choose the interpreter that evaluates",
    "job.activate_script": "it would run a script before every evaluation",
}

# A grant run's end event names its attempt, before the run key (runs.py).
_ATTEMPT_NOTE = re.compile(r"grant (g\d+) attempt (\d+) of \d+.*run ([0-9a-f]{32})$")


class Refused(Exception):
    """A grant or a grant run the rules do not allow; the message says why."""


def in_agent_env() -> bool:
    return bool(os.environ.get(AGENT_ENV))


def load(root: Path) -> List[Dict[str, Any]]:
    path = Path(root) / LEDGER
    if not path.exists():
        return []
    try:
        records = json.loads(read_text(path))
    except json.JSONDecodeError as error:
        raise Refused(f"{path} cannot be read ({error}); ask the owner to check it")
    return records


def terms(grant: Dict[str, Any]) -> str:
    """A grant's terms as the run log states them."""
    allow = ", ".join(grant["allow"]) or "nothing"
    return (
        f"{grant['id']}: {' '.join(grant['hypotheses'])}, {grant['attempts']} "
        f"attempts, each stops at ${grant['max_cost']:g}, the agent may set "
        f"{allow}, shinka.yaml {_digest(grant['configs'])}"
    )


def _digest(configs: Dict[str, str]) -> str:
    """One short digest of the shinka.yaml hashes a grant keeps."""
    text = json.dumps(configs, sort_keys=True).encode()
    return hashlib.sha256(text).hexdigest()[:12]


def _log_rows(root: Path) -> List[Tuple[str, str, str, str, str]]:
    path = Path(root) / RESULTS_LOG
    rows = path.read_text().splitlines()[1:] if path.exists() else []
    return [tuple((row.split("\t", 4) + [""] * 5)[:5]) for row in rows]


def _check_allow(keys: Iterable[str]) -> List[str]:
    from shinka.cli.run import _field_types

    fields = _field_types()
    allow = sorted(set(keys))
    for key in allow:
        if key in _NEVER:
            raise Refused(f"a grant cannot allow {key}: {_NEVER[key]}")
        namespace, _, name = key.partition(".")
        if name not in fields.get(namespace, {}):
            raise Refused(
                f"{key} is not a shinka_run setting (NS.FIELD, ns evo/db/job)"
            )
    return allow


def add_grant(
    root: Path,
    hypotheses: List[str],
    attempts: int,
    max_cost: float,
    allow: Iterable[str] = (),
    note: str = "",
) -> Dict[str, Any]:
    """Record the owner's grant for one round: up to ``attempts`` runs of
    ``hypotheses``, each stopping at ``max_cost`` USD, with the ``allow`` keys
    the only settings the agent may pass. The experiments the hypotheses have
    now keep the run settings (``shinka.yaml``) they have now."""
    if attempts < 1:
        raise Refused("a grant allows at least one attempt")
    if not (math.isfinite(max_cost) and max_cost > 0):
        raise Refused("a grant's per-run cap is a positive amount of USD")
    allowed = _check_allow(allow)
    with file_lock(lock_path(root)):
        nb = Notebook.load(root)
        configs = {}
        for hid in dict.fromkeys(hypotheses):
            try:
                hyp = nb.get(hid)
            except KeyError as error:
                raise Refused(str(error.args[0]))
            for exp in hyp.experiments:
                task = experiment_dir(root, hid, exp.id) / "task"
                configs[f"{hid}/{exp.id}"] = file_sha256(task / SHINKA_CONFIG_FILENAME)
        records = load(root)
        rows = _log_rows(root)
        # Numbered from the log, which only grows, so an id is never reused.
        number = 1 + sum(1 for row in rows if row[3] == "grant")
        grant = {
            "kind": "grant",
            "id": f"g{number:03d}",
            "hypotheses": list(dict.fromkeys(hypotheses)),
            "attempts": attempts,
            "max_cost": max_cost,
            "allow": allowed,
            "configs": configs,
            "time": _now(),
            "note": " ".join(note.split()),
        }
        # The log first: a crash before the ledger is written leaves an event
        # naming no grant, which no check reads.
        append_log(root, "", "", "grant", terms(grant))
        atomic_write_text(Path(root) / LEDGER, json.dumps(records + [grant], indent=1))
    return grant


class Ledger(NamedTuple):
    grants: Dict[str, Dict[str, Any]]
    attempts: Dict[str, List[Dict[str, Any]]]
    approvals: Dict[str, Dict[str, Any]]
    # Each approval's use, by approval id.
    uses: Dict[str, Dict[str, Any]]


def _verify(
    root: Path, records: List[Dict[str, Any]], live: List[Dict[str, Any]]
) -> Ledger:
    """The ledger against the run log and the live leases; raises on the first
    mismatch."""
    changed = f"{Path(root) / LEDGER} does not match the run log"
    stated = {}
    for _time, _hid, _eid, event, note in _log_rows(root):
        if event in ("grant", "approval"):
            stated[note.split(":", 1)[0]] = note
    ledger = Ledger({}, {}, {}, {})
    for record in records:
        kind = record.get("kind")
        if kind == "grant":
            gid = record["id"]
            if gid in ledger.grants or stated.get(gid) != terms(record):
                raise Refused(f"{changed} at grant {gid}; ask the owner")
            ledger.grants[gid] = record
            ledger.attempts[gid] = []
        elif kind == "attempt":
            taken = ledger.attempts.get(record["grant"])
            if taken is None or record["number"] != len(taken) + 1:
                raise Refused(f"{changed} at an attempt of {record['grant']}")
            taken.append(record)
        elif kind == "approval":
            aid = record["id"]
            if aid in ledger.approvals or stated.get(aid) != approval_terms(record):
                raise Refused(f"{changed} at approval {aid}; ask the owner")
            ledger.approvals[aid] = record
        elif kind == "approval-used":
            aid = record["approval"]
            if aid not in ledger.approvals or aid in ledger.uses:
                raise Refused(f"{changed} at a use of approval {aid}")
            ledger.uses[aid] = record
        else:
            raise Refused(f"{changed}: a record of unknown kind")
    for gid, taken in ledger.attempts.items():
        if len(taken) > ledger.grants[gid]["attempts"]:
            raise Refused(f"{changed}: {gid} has more attempts than it allows")
    works = {
        (a["grant"], a["work"]) for taken in ledger.attempts.values() for a in taken
    }
    ran = []
    for _time, _hid, _eid, event, note in _log_rows(root):
        found = _ATTEMPT_NOTE.search(note) if event in ("run", "run-failed") else None
        if found:
            ran.append((found.group(1), found.group(3)))
    ran += [
        (w["origin"]["grant"], w["id"])
        for w in live
        if w.get("origin", {}).get("grant")
    ]
    for gid, work in ran:
        if (gid, work) not in works:
            raise Refused(f"{changed}: a run of {gid} has no attempt in the ledger")
    return ledger


def _passthrough(extra: List[str], grant: Dict[str, Any]) -> List[str]:
    """The ``KEY=VALUE`` settings an agent run passes shinka_run; only
    ``--set`` with keys the grant allows, checked as shinka_run checks them."""
    from shinka.cli.run import _field_types, _parse_overrides

    tokens, rest = [], list(extra)
    while rest:
        token = rest.pop(0)
        if token == "--set" and rest and not rest[0].startswith("-"):
            tokens.append(rest.pop(0))
        elif token.startswith("--set="):
            tokens.append(token[len("--set=") :])
        else:
            raise Refused(
                "a grant run passes shinka_run only --set KEY=VALUE settings its "
                f"grant lists, not {token!r}"
            )
    for token in tokens:
        key = token.split("=", 1)[0]
        if key == "evo.max_api_costs":
            raise Refused(f"evo.max_api_costs is not the agent's to set: {_NEVER[key]}")
        if key not in grant["allow"]:
            allowed = ", ".join(grant["allow"]) or "none"
            raise Refused(
                f"{key} is not among grant {grant['id']}'s settings ({allowed})"
            )
    try:
        _parse_overrides(tokens, _field_types())
    except ValueError as error:
        raise Refused(str(error))
    return tokens


def _settings_kept(root: Path, grant: Dict[str, Any], hid: str, eid: str) -> bool:
    """Whether the experiment's shinka.yaml is as the grant found it, or, for an
    experiment scaffolded since, as scaffold writes it (the task's message)."""
    path = experiment_dir(root, hid, eid) / "task" / SHINKA_CONFIG_FILENAME
    kept = grant["configs"].get(f"{hid}/{eid}")
    if kept is not None:
        return file_sha256(path) == kept
    config = yaml.safe_load(path.read_text()) or {}
    return set(config) <= {"evo"} and set(config.get("evo") or {}) <= {"task_sys_msg"}


def check_run(
    root: Path,
    grant_id: str,
    hid: str,
    eid: str,
    extra: List[str],
    live: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], int, List[str]]:
    """Whether a run may take an attempt from ``grant_id``; under the direction
    lock, before anything is taken. Returns the grant, the attempts it has
    used and the settings to pass. A grant with none left pauses the round:
    the refusal is logged once."""
    ledger = _verify(root, load(root), live)
    grants, attempts = ledger.grants, ledger.attempts
    grant = grants.get(grant_id)
    if grant is None:
        raise Refused(
            f"no grant {grant_id}: an agent run needs a round the owner granted "
            "(`shinka_research grant`)"
        )
    if hid not in grant["hypotheses"]:
        raise Refused(
            f"{hid} is not in grant {grant_id} ({' '.join(grant['hypotheses'])})"
        )
    used = len(attempts[grant_id])
    if used >= grant["attempts"]:
        # Said when the last attempt was taken; again only if that was lost.
        _log_paused(root, grant, hid, eid)
        raise Refused(
            f"grant {grant_id}'s {grant['attempts']} attempts are used: this round is "
            "over, and the next needs the owner's grant"
        )
    if not _settings_kept(root, grant, hid, eid):
        raise Refused(
            f"{experiment_dir(root, hid, eid) / 'task' / SHINKA_CONFIG_FILENAME}: run "
            f"settings changed since grant {grant_id}; under a grant they come from "
            "the grant"
        )
    return grant, used, _passthrough(extra, grant)


def take_attempt(
    root: Path,
    grant: Dict[str, Any],
    number: int,
    hid: str,
    eid: str,
    work_id: str,
    cap: float,
    stops_at: float,
) -> None:
    """Record attempt ``number`` of ``grant``; under the lock that checked it,
    before the run's lease, so no run goes without one."""
    records = load(root)
    records.append(
        {
            "kind": "attempt",
            "grant": grant["id"],
            "number": number,
            "hypothesis": hid,
            "experiment": eid,
            "work": work_id,
            "cap": cap,
            "stops_at": stops_at,
            "time": _now(),
        }
    )
    atomic_write_text(Path(root) / LEDGER, json.dumps(records, indent=1))
    if number == grant["attempts"]:
        _log_paused(root, grant, hid, eid)


def _log_paused(root: Path, grant: Dict[str, Any], hid: str, eid: str) -> None:
    """A grant with every attempt taken pauses its round; the run log says so
    once."""
    gid = grant["id"]
    for _time, _hid, _eid, event, note in _log_rows(root):
        if event == "round-paused" and note.startswith(f"{gid}:"):
            return
    append_log(
        root, hid, eid, "round-paused",
        f"{gid}: all {grant['attempts']} attempts taken; the next round needs the "
        "owner's grant",
    )  # fmt: skip


def approval_terms(approval: Dict[str, Any]) -> str:
    """A lockbox approval's terms as the run log states them."""
    return f"{approval['id']}: the lockbox may be opened once"


def add_approval(root: Path, note: str = "") -> Dict[str, Any]:
    """Record the owner's approval to open the direction's lockbox once."""
    with file_lock(lock_path(root)):
        nb = Notebook.load(root)
        if nb.data is None:
            raise Refused("this direction has no data yet, so no lockbox to open")
        if nb.data.opened:
            raise Refused("the lockbox was already opened")
        records = load(root)
        number = 1 + sum(1 for row in _log_rows(root) if row[3] == "approval")
        approval = {
            "kind": "approval",
            "id": f"a{number:03d}",
            "time": _now(),
            "note": " ".join(note.split()),
        }
        # The log first, as for a grant.
        append_log(root, "", "", "approval", approval_terms(approval))
        records.append(approval)
        atomic_write_text(Path(root) / LEDGER, json.dumps(records, indent=1))
    return approval


def use_approval(root: Path, approval_id: str, opening: str) -> None:
    """Spend the owner's lockbox approval on the opening that starts at
    ``opening``; under the lock that records the opening, before it is saved,
    so the lockbox never opens on an approval left unspent."""
    records = load(root)
    ledger = _verify(root, records, live_work(root))
    if approval_id not in ledger.approvals:
        raise Refused(
            f"no lockbox approval {approval_id}: opening the lockbox needs the "
            "owner's approval (`shinka_research approve`)"
        )
    if approval_id in ledger.uses:
        used = ledger.uses[approval_id]["opening"]
        raise Refused(
            f"approval {approval_id} was already used (the opening of {used})"
        )
    records.append(
        {
            "kind": "approval-used",
            "approval": approval_id,
            "opening": opening,
            "time": _now(),
        }
    )
    atomic_write_text(Path(root) / LEDGER, json.dumps(records, indent=1))


def check_opened_by(root: Path, approval_id: str, opening: str) -> None:
    """Whether ``approval_id`` is the approval spent on the opening that
    started at ``opening``: finishing that opening carries it."""
    use = _verify(root, load(root), live_work(root)).uses.get(approval_id)
    if use is None or use["opening"] != opening:
        raise Refused(
            f"approval {approval_id} did not open this lockbox (opened {opening}); "
            "finishing that opening carries the approval that opened it"
        )


def origin_note(origin: Optional[Dict[str, Any]], spent: float) -> str:
    """Who started a run, as its end event says it, before the run key."""
    if not origin:
        return ""
    if not origin.get("grant"):
        return f"{origin['env']} run, "
    return (
        f"{origin['env']} run, grant {origin['grant']} attempt {origin['attempt']} "
        f"of {origin['of']}, ${spent:.2f} of ${origin['cap']:g}, "
    )
