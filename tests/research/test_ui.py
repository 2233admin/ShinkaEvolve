import json
import threading
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from shinka.research import ui
from shinka.research.notebook import Experiment, Notebook, Status, Verdict, append_log

SECRET = {"oos_sharpe": 1.23, "dsr": 0.97}


def _direction(root: Path) -> Path:
    root.mkdir()
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
    assert snap["library"] == {"wiki_pages": 0, "pending": 0}
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
