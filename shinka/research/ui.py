"""Read-only research UI: one JSON snapshot of a direction plus the built app.

Visibility follows the agent's rules (design.md §7.3): validation-segment
evidence appears only once a hypothesis has a verdict, and lockbox numbers
only exist after ``confirm``. The filtering happens here, so the browser
never receives data it must not show.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from shinka.webui import visualization

from . import library, rundb
from .notebook import RESULTS_LOG, Hypothesis, Notebook, experiment_dir
from .skeptic import expected_max_sharpe

DIST = Path(__file__).parent / "webui" / "dist"
HIDDEN = "（验证段结果，出结论后显示）"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def _experiment(root: Path, h: Hypothesis, e) -> Dict[str, Any]:
    rd = experiment_dir(root, h.id, e.id) / "results"
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
        # Path for the upstream evolution tree, relative to <direction>/experiments.
        "tree_db": db.relative_to(root / "experiments").as_posix()
        if db.exists()
        else None,
    }
    if h.verdict is not None:
        out["skeptic"] = e.skeptic
    return out


def _body(root: Path, hid: str) -> str:
    path = root / "hypotheses" / f"{hid}.md"
    return path.read_text() if path.exists() else ""


def _events(root: Path, nb: Notebook, limit: int) -> List[Dict[str, str]]:
    path = root / RESULTS_LOG
    if not path.exists():
        return []
    judged = {h.id for h in nb.hypotheses if h.verdict is not None}
    rows = []
    for line in path.read_text().splitlines()[1:][-limit:]:
        time, hid, eid, event, note = (line.split("\t", 4) + [""] * 5)[:5]
        if event == "skeptic" and hid not in judged:
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
    kept in the scores' own annualised units)."""
    rows = []
    for h in nb.hypotheses:
        for e in h.experiments:
            rd = experiment_dir(root, h.id, e.id) / "results"
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
                "bar": expected_max_sharpe(n, var),
            }
        )
    return out


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _library(root: Path) -> Optional[Dict[str, Any]]:
    lib = library.library_root(root)
    if not (lib / library.MANIFEST).exists():
        return None
    return {
        "index": _read(lib / "index.md"),
        "log": _read(lib / "log.md"),
        "pending": library.pending(lib),
        "pages": [
            {"name": name, "text": _read(lib / "wiki" / f"{name}.md")}
            for name in library.wiki_pages(lib)
        ],
    }


def snapshot(
    root: Path, events: int = 50, tree_url: Optional[str] = None
) -> Dict[str, Any]:
    root = Path(root)
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
        "events": _events(root, nb, events),
        "library": _library(root),
        "tree_url": tree_url,
    }


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, root: Path, tree_url: Optional[str] = None, **kwargs):
        self.research_root = root
        self.tree_url = tree_url
        super().__init__(*args, directory=str(DIST), **kwargs)

    def do_GET(self) -> None:
        if self.path.split("?")[0] == "/api/snapshot":
            body = json.dumps(
                snapshot(self.research_root, tree_url=self.tree_url),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def serve(root: Path, port: int, open_browser: bool) -> None:
    if not (DIST / "index.html").exists():
        raise SystemExit(
            f"UI not built: run `pnpm install && pnpm build` in {DIST.parent / 'app'}"
        )
    root = Path(root)
    Notebook.load(root)  # fail early on a wrong directory
    # Upstream's evolution tree, next port, private (held-out) metrics hidden.
    tree = ThreadingHTTPServer(
        ("127.0.0.1", port + 1),
        partial(
            visualization.DatabaseRequestHandler,
            search_root=str(root / "experiments"),
            show_private=False,
            directory=str(Path(visualization.__file__).parent),
        ),
    )
    threading.Thread(target=tree.serve_forever, daemon=True).start()
    tree_url = f"http://127.0.0.1:{tree.server_address[1]}/viz_tree.html"
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), partial(_Handler, root=root, tree_url=tree_url)
    )
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Research UI for {root} at {url} (Ctrl+C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        tree.shutdown()
        tree.server_close()
