import json
import sqlite3
import threading
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from shinka.research import library, ui
from shinka.research.notebook import (
    Experiment,
    Notebook,
    Status,
    Verdict,
    append_log,
    experiment_dir,
)
from shinka.research.skeptic import expected_max_sharpe

SECRET = {"oos_sharpe": 1.23, "dsr": 0.97}


def _direction(root: Path) -> Path:
    root.mkdir(parents=True)
    nb = Notebook(direction="BTC momentum", budget_usd=40.0)
    judged = nb.add_hypothesis(root, "vol-targeted momentum", "Body of h001.")
    pending = nb.add_hypothesis(root, "funding reversal", "Body of h002.")
    for h in (judged, pending):
        exp = Experiment(id="e01", evaluator_hash="x", data_path="d", data_hash="y")
        exp.skeptic = dict(SECRET)
        h.experiments.append(exp)
    judged.verdict = Verdict(
        label=Status.supported, reason="all checks pass", evidence=SECRET
    )
    judged.status = Status.supported
    pending.status = Status.challenged
    nb.save(root)
    append_log(root, "h001", "e01", "skeptic", "rule says supported: val Sharpe 1.23")
    append_log(root, "h002", "e01", "skeptic", "rule says refuted: val Sharpe -0.40")
    return root


def test_validation_evidence_hidden_until_verdict(tmp_path: Path) -> None:
    snap = ui.snapshot(_direction(tmp_path / "d"))
    h1, h2 = snap["hypotheses"]
    assert h1["experiments"][0]["skeptic"] == SECRET
    assert h1["verdict"]["evidence"] == SECRET
    assert "skeptic" not in h2["experiments"][0]
    assert h2["verdict"] is None
    assert "-0.40" not in json.dumps(snap)
    notes = {e["hypothesis"]: e["note"] for e in snap["events"]}
    assert notes == {"h001": "rule says supported: val Sharpe 1.23", "h002": ui.HIDDEN}


def test_snapshot_shape(tmp_path: Path) -> None:
    snap = ui.snapshot(_direction(tmp_path / "d"))
    assert snap["direction"] == "BTC momentum"
    assert snap["lockbox"] is None
    assert snap["library"] is None
    assert snap["trial_points"] == []
    h1 = snap["hypotheses"][0]
    assert h1["body"].startswith("# h001: vol-targeted momentum")
    assert h1["experiments"][0]["best_is_score"] is None
    assert snap["events"][0]["hypothesis"] == "h002"  # newest first


def test_server_serves_snapshot(tmp_path: Path, monkeypatch) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>t</title>")
    monkeypatch.setattr(ui, "DIST", dist)
    root = _direction(tmp_path / "d")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(ui._Handler, root=root))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        snap = json.loads(urllib.request.urlopen(f"{base}/api/snapshot").read())
        assert snap["hypotheses"][1]["verdict"] is None
        assert b"<title>t</title>" in urllib.request.urlopen(f"{base}/").read()
    finally:
        server.shutdown()
        server.server_close()


def _programs(results: Path, rows: list) -> None:
    """rows: (timestamp, generation, correct, in-sample score, private metrics)."""
    results.mkdir(parents=True)
    with sqlite3.connect(results / "programs.sqlite") as conn:
        conn.execute(
            "CREATE TABLE programs (id TEXT, code TEXT, timestamp REAL, "
            "generation INTEGER, correct INTEGER, combined_score REAL, "
            "private_metrics TEXT, metadata TEXT)"
        )
        for i, (t, g, ok, score, private) in enumerate(rows):
            conn.execute(
                "INSERT INTO programs VALUES (?, '', ?, ?, ?, ?, ?, '{}')",
                (f"p{i}", t, g, ok, score, json.dumps(private)),
            )


def test_trial_points_carry_noise_bar_not_private_metrics(tmp_path: Path) -> None:
    root = _direction(tmp_path / "d")
    secret = {"oos_sharpe": -7.77}
    _programs(
        experiment_dir(root, "h001", "e01") / "results",
        [(100.0, 0, 1, 0.5, secret), (300.0, 1, 0, None, secret)],
    )
    _programs(
        experiment_dir(root, "h002", "e01") / "results",
        [(200.0, 0, 1, 1.5, secret), (400.0, 1, 1, 1.0, secret)],
    )
    snap = ui.snapshot(root, tree_url="http://127.0.0.1:1/viz_tree.html")
    pts = snap["trial_points"]
    assert [(p["hypothesis"], p["score"], p["record"]) for p in pts] == [
        ("h001", 0.5, True),
        ("h002", 1.5, True),
        ("h001", None, False),  # failed programs are trials too
        ("h002", 1.0, False),
    ]
    var = float(np.var([0.5, 1.5, 1.0], ddof=1))
    assert pts[3]["bar"] == expected_max_sharpe(4, var)
    assert pts[0]["bar"] == 0.0
    assert "-7.77" not in json.dumps(snap)
    exp = snap["hypotheses"][0]["experiments"][0]
    assert exp["tree_db"] == "h001/e01/results/programs.sqlite"
    assert exp["first_trial"].startswith("1970-01-01T00:01:40")
    assert snap["tree_url"] == "http://127.0.0.1:1/viz_tree.html"


def test_library_contents(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    root = _direction(ws / "d")
    lib = library.ensure(library.library_root(root))
    (lib / "wiki" / "momentum.md").write_text("# Momentum\n")
    (lib / "raw" / "paper.md").write_text("raw")
    snap = ui.snapshot(root)["library"]
    assert snap["pages"] == [{"name": "momentum", "text": "# Momentum\n"}]
    assert snap["pending"] == ["raw/paper.md"]
    assert snap["index"].startswith("# Library index")
