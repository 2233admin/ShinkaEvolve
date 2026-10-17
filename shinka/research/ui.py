"""Read-only research UI: one JSON snapshot of a direction plus the built app.

Visibility follows the agent's rules (design.md §7.3): validation-segment
evidence appears only once a hypothesis has a verdict, and lockbox numbers
only exist after ``confirm``. The filtering happens here, so the browser
never receives data it must not show.
"""

from __future__ import annotations

import json
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

from . import library, rundb
from .notebook import RESULTS_LOG, Hypothesis, Notebook, experiment_dir

DIST = Path(__file__).parent / "webui" / "dist"
HIDDEN = "（验证段结果，出结论后显示）"


def _experiment(root: Path, h: Hypothesis, e) -> Dict[str, Any]:
    rd = experiment_dir(root, h.id, e.id) / "results"
    out: Dict[str, Any] = {
        "id": e.id,
        "created": e.created,
        "n_evaluated": e.n_evaluated,
        "n_probed": e.n_probed,
        "best_is_score": rundb.best_score(rd),
        "spend_usd": rundb.total_cost(rd),
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


def _library(root: Path) -> Dict[str, Any]:
    lib = library.library_root(root)
    if not (lib / library.MANIFEST).exists():
        return {"wiki_pages": 0, "pending": 0}
    return {
        "wiki_pages": len(library.wiki_pages(lib)),
        "pending": len(library.pending(lib)),
    }


def snapshot(root: Path, events: int = 50) -> Dict[str, Any]:
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
        "events": _events(root, nb, events),
        "library": _library(root),
    }


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, root: Path, **kwargs):
        self.research_root = root
        super().__init__(*args, directory=str(DIST), **kwargs)

    def do_GET(self) -> None:
        if self.path.split("?")[0] == "/api/snapshot":
            body = json.dumps(snapshot(self.research_root), ensure_ascii=False).encode()
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
    Notebook.load(root)  # fail early on a wrong directory
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), partial(_Handler, root=Path(root))
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
