"""Read-only workspace and direction snapshots plus the built research app.

Visibility follows the agent's rules (design.md §7.3): validation-segment
evidence appears only once a hypothesis has a verdict, and lockbox numbers
only exist after ``confirm``. The filtering happens here, so the browser
never receives data it must not show.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.parse
import webbrowser
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional

import numpy as np

from .analysis import ANALYSIS_PATHS, AnalysisHandler, confined as _confined
from . import library, rundb
from .notebook import (
    NOTEBOOK_FILENAME,
    RESULTS_LOG,
    Hypothesis,
    Notebook,
    experiment_dir,
)
from .skeptic import expected_max_sharpe

DIST = Path(__file__).parent / "webui" / "dist"
HIDDEN = "（验证段结果，出结论后显示）"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def _results(root: Path, hid: str, eid: str) -> Path:
    experiments = _confined(root, root / "experiments")
    results = _confined(experiments, experiment_dir(root, hid, eid) / "results")
    for filename in (rundb.DB_FILENAME, rundb.PROMPT_DB_FILENAME):
        _confined(experiments, results / filename)
    return results


def _experiment(root: Path, h: Hypothesis, e) -> Dict[str, Any]:
    rd = _results(root, h.id, e.id)
    times = [t for t, _, _ in rundb.trials(rd)]
    db = rd / rundb.DB_FILENAME
    out: Dict[str, Any] = {
        "id": e.id,
        "created": e.created,
        "n_evaluated": e.n_evaluated,
        "n_probed": e.n_probed,
        "best_is_score": rundb.best_score(rd),
        "spend_usd": rundb.total_cost(rd),
        "first_trial": _iso(times[0]) if times else None,
        "last_trial": _iso(times[-1]) if times else None,
        "analysis_url": "/?"
        + urllib.parse.urlencode(
            {
                "view": "runs",
                "direction": root.name,
                "hypothesis": h.id,
                "analysis": "/viz_tree.html?"
                + urllib.parse.urlencode(
                    {"db_path": f"workspace/{db.relative_to(root.parent).as_posix()}"}
                ),
            }
        )
        if db.exists()
        else None,
    }
    if h.verdict is not None:
        out["skeptic"] = e.skeptic
    return out


def _body(root: Path, hid: str) -> str:
    path = _confined(root, root / "hypotheses" / f"{hid}.md")
    return path.read_text() if path.exists() else ""


def _events(root: Path, nb: Notebook, limit: int) -> List[Dict[str, str]]:
    path = _confined(root, root / RESULTS_LOG)
    if not path.exists():
        return []
    judged = {h.id for h in nb.hypotheses if h.verdict is not None}
    rows = []
    for line in path.read_text().splitlines()[1:][-limit:]:
        time, hid, eid, event, note = (line.split("\t", 4) + [""] * 5)[:5]
        # skeptic and verdict notes quote validation figures; a re-run clears
        # the verdict, so an old verdict note must hide again too.
        if event in ("skeptic", "verdict") and hid not in judged:
            note = HIDDEN
        rows.append(
            {
                "time": time,
                "hypothesis": hid,
                "experiment": eid,
                "event": event,
                "note": note,
            }
        )
    return rows[::-1]


def _trials(root: Path, nb: Notebook) -> List[Dict[str, Any]]:
    """Every evaluated program of the direction in time order, with the noise
    bar: the expected best in-sample score of n null strategies, using the
    variance of the direction's in-sample scores (the skeptic's approximation,
    kept in the scores' own annualised units). Probes leave no program row, so
    they are not plotted; the bar counts them all from the start, which keeps
    it at least as high as the skeptic's (it counts every probe too)."""
    probes = sum(e.n_probed for h in nb.hypotheses for e in h.experiments)
    rows = []
    for h in nb.hypotheses:
        for e in h.experiments:
            rd = _results(root, h.id, e.id)
            rows += [(t, g, s, h.id, e.id) for t, g, s in rundb.trials(rd)]
    rows.sort(key=lambda r: r[0])
    scores = np.array([r[2] for r in rows if r[2] is not None])
    var = float(scores.var(ddof=1)) if len(scores) > 1 else 0.0
    out, best = [], float("-inf")
    for n, (t, g, s, hid, eid) in enumerate(rows, start=1):
        record = s is not None and s > best
        best = s if record else best
        out.append(
            {
                "n": n,
                "time": _iso(t),
                "hypothesis": hid,
                "experiment": eid,
                "generation": g,
                "score": s,
                "record": record,
                "bar": expected_max_sharpe(n + probes, var),
            }
        )
    return out


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _library(workspace: Path) -> Optional[Dict[str, Any]]:
    lib = _confined(workspace, workspace / "library")
    manifest = _confined(lib, lib / library.MANIFEST)
    if not manifest.exists():
        return None
    if not isinstance(json.loads(manifest.read_text()), dict):
        raise ValueError("Library manifest must be an object")
    raw = _confined(lib, lib / "raw")
    _confined(lib, lib / "wiki")
    # pending() hashes these files. Validate its exact read set without changing
    # the library CLI's filesystem semantics or the lexical relative names.
    for path in raw.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            _confined(lib, path)
    return {
        "index": _read(_confined(lib, lib / "index.md")),
        "log": _read(_confined(lib, lib / "log.md")),
        "pending": library.pending(lib),
        "pages": [
            {"name": name, "text": _read(_confined(lib, lib / "wiki" / f"{name}.md"))}
            for name in library.wiki_pages(lib)
        ],
    }


def snapshot(root: Path, events: int = 50) -> Dict[str, Any]:
    root = Path(root).resolve()
    _confined(root, root / NOTEBOOK_FILENAME)
    nb = Notebook.load(root)
    hyps = []
    for h in nb.hypotheses:
        exps = [_experiment(root, h, e) for e in h.experiments]
        hyps.append(
            {
                "id": h.id,
                "title": h.title,
                "status": h.status.value,
                "created": h.created,
                "body": _body(root, h.id),
                "experiments": exps,
                "verdict": h.verdict.model_dump(mode="json") if h.verdict else None,
                "confirmation": h.confirmation,
            }
        )
    data = nb.data
    return {
        "direction": nb.direction,
        "created": nb.created,
        "budget_usd": nb.budget_usd,
        "spend_usd": sum(e["spend_usd"] for h in hyps for e in h["experiments"]),
        "trials": nb.total_trials,
        "lockbox": None
        if data is None
        else {
            "opened": data.opened,
            "start": data.lockbox_start,
            "n_total": data.n_total,
            "n_dev": data.n_dev,
        },
        "hypotheses": hyps,
        "trial_points": _trials(root, nb),
        "probes": sum(e.n_probed for h in nb.hypotheses for e in h.experiments),
        "events": _events(root, nb, events),
        "library": _library(root.parent),
    }


def _workspace_root(root: Path) -> Path:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Workspace directory does not exist: {root}")
    if (root / NOTEBOOK_FILENAME).exists():
        raise ValueError(
            f"This is a direction directory; pass its workspace: {root.parent}"
        )
    return root


def _direction_root(workspace: Path, direction: str) -> Path:
    """Resolve one directory-name ID, never an arbitrary client-provided path."""
    if (
        not direction
        or direction in {".", ".."}
        or "/" in direction
        or "\\" in direction
        or PureWindowsPath(direction).drive
    ):
        raise FileNotFoundError("Unknown direction")
    root = workspace / direction
    notebook = root / NOTEBOOK_FILENAME
    if root.is_symlink() or notebook.is_symlink():
        raise FileNotFoundError("Unknown direction")
    canonical = root.resolve()
    if (
        canonical != root
        or canonical.parent != workspace
        or not root.is_dir()
        or not notebook.is_file()
    ):
        raise FileNotFoundError("Unknown direction")
    return canonical


def workspace_snapshot(root: Path) -> Dict[str, Any]:
    root = _workspace_root(root)
    directions = []
    for child in sorted(root.iterdir()):
        marker = child / NOTEBOOK_FILENAME
        if (
            child.is_symlink()
            or marker.is_symlink()
            or not marker.is_file()
            or child.resolve() != child
        ):
            continue
        try:
            direction = _direction_root(root, child.name)
            nb = Notebook.load(direction)
            directions.append(
                {
                    "id": child.name,
                    "direction": nb.direction,
                    "created": nb.created,
                    "hypotheses": len(nb.hypotheses),
                    "trials": nb.total_trials,
                    "spend_usd": sum(
                        rundb.total_cost(_results(direction, h.id, e.id))
                        for h in nb.hypotheses
                        for e in h.experiments
                    ),
                    "budget_usd": nb.budget_usd,
                    "lockbox_opened": nb.data.opened if nb.data else None,
                    "error": None,
                }
            )
        except (OSError, ValueError, sqlite3.Error) as error:
            directions.append(
                {
                    "id": child.name,
                    "direction": child.name,
                    "error": f"研究记录无法读取（{type(error).__name__}）",
                }
            )
    return {
        "name": root.name,
        "directions": directions,
        "library": _library(root),
    }


class _Handler(AnalysisHandler):
    def __init__(
        self,
        *args,
        root: Path,
        **kwargs,
    ):
        self.workspace_root = Path(root).resolve()
        super().__init__(
            *args, search_root=str(self.workspace_root), directory=str(DIST), **kwargs
        )

    def _json(self, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        if url.path in ANALYSIS_PATHS or url.path.startswith("/plot_file/"):
            super().do_GET()
            return
        if url.path not in {"/api/workspace", "/api/snapshot"}:
            SimpleHTTPRequestHandler.do_GET(self)
            return
        try:
            if url.path == "/api/workspace":
                payload = workspace_snapshot(self.workspace_root)
            else:
                ids = urllib.parse.parse_qs(url.query, keep_blank_values=True).get(
                    "direction", []
                )
                if len(ids) != 1 or not ids[0].strip():
                    self.send_error(400, "Specify exactly one nonblank direction")
                    return
                try:
                    root = _direction_root(self.workspace_root, ids[0])
                except FileNotFoundError:
                    self.send_error(404, "Unknown direction")
                    return
                payload = snapshot(root)
            self._json(payload)
        except (OSError, ValueError, sqlite3.Error) as error:
            self.send_error(
                500, f"Unable to read research data: {type(error).__name__}"
            )


def serve(
    root: Path, port: int, open_browser: bool, runs_root: Optional[Path] = None
) -> None:
    root = _workspace_root(root)
    if not (DIST / "index.html").exists():
        raise SystemExit(
            f"UI not built: run `pnpm install`, then `pnpm build` "
            f"in {DIST.parent / 'app'}"
        )
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), partial(_Handler, root=root, runs_root=runs_root)
    )
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(f"Research workspace UI for {root} at {url} (Ctrl+C to stop)", flush=True)
        if open_browser:
            webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
