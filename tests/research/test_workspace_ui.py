"""Workspace-page consumer contract, exercised over real loopback HTTP."""

import errno
import json
import sqlite3
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from shinka.research import library, ui
from shinka.research.notebook import (
    DataSplit,
    Experiment,
    Notebook,
    Status,
    append_log,
    experiment_dir,
)

CREATED = "2026-01-02T03:04:05+00:00"
HELD_OUT = "fixture-held-out-evidence-should-not-be-in-workspace-summary"
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@contextmanager
def _server(workspace: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(ui._Handler, root=workspace))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive(), "workspace HTTP server did not stop"


def _get(base: str, path: str):
    with HTTP.open(base + path, timeout=5) as response:
        assert response.status == 200
        return json.load(response)


def _snapshot_path(direction: str) -> str:
    return "/api/snapshot?" + urllib.parse.urlencode({"direction": direction})


def _assert_error(base: str, path: str, status: int) -> None:
    with pytest.raises(urllib.error.HTTPError) as caught:
        HTTP.open(base + path, timeout=5)
    with caught.value as response:
        assert response.code == status


def _direction(root: Path, title: str, *bodies: str) -> Notebook:
    root.mkdir(parents=True)
    notebook = Notebook(direction=title, created=CREATED)
    for number, body in enumerate(bodies, start=1):
        hypothesis = notebook.add_hypothesis(root, f"fixture hypothesis {number}", body)
        hypothesis.created = CREATED
    notebook.save(root)
    return notebook


def _shared_library(workspace: Path) -> dict:
    root = library.ensure(workspace / "library")
    (root / "index.md").write_text("# Fixture shared index\n", encoding="utf-8")
    (root / "log.md").write_text("# Fixture library log\n", encoding="utf-8")
    (root / "wiki" / "shared.md").write_text(
        "# Shared fixture knowledge\n", encoding="utf-8"
    )
    (root / "raw" / "source.md").write_text("Fixture source\n", encoding="utf-8")
    return {
        "index": "# Fixture shared index\n",
        "log": "# Fixture library log\n",
        "pages": [{"name": "shared", "text": "# Shared fixture knowledge\n"}],
        "pending": ["raw/source.md"],
    }


def _one_trial(results: Path, metadata: dict | None = None) -> None:
    """Persist one evaluated program, by default with a known $0.75 cost, plus
    private data."""
    results.mkdir(parents=True)
    with sqlite3.connect(results / "programs.sqlite") as connection:
        connection.execute(
            "CREATE TABLE programs (id TEXT, code TEXT, timestamp REAL, "
            "generation INTEGER, correct INTEGER, combined_score REAL, "
            "private_metrics TEXT, metadata TEXT)"
        )
        connection.execute(
            "INSERT INTO programs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "fixture-program",
                "# fixture candidate",
                100.0,
                0,
                1,
                0.5,
                json.dumps({HELD_OUT: 987.654}),
                json.dumps(metadata or {"api_costs": 0.5, "embed_cost": 0.25}),
            ),
        )


def test_unpriced_runs_make_direction_spend_a_lower_bound(tmp_path: Path) -> None:
    workspace = tmp_path / "fixture-workspace"
    root = workspace / "alpha"
    notebook = _direction(root, "Alpha fixture", "Alpha-only hypothesis body.")
    for eid in ("e01", "e02"):
        notebook.hypotheses[0].experiments.append(
            Experiment(
                id=eid,
                created=CREATED,
                evaluator_hash="fixture-evaluator-hash",
                data_path="fixture-dev.csv",
                data_hash="fixture-data-hash",
                n_evaluated=1,
            )
        )
    notebook.save(root)
    _one_trial(experiment_dir(root, "h001", "e01") / "results")
    # A headless call without pricing: its api_costs is not an amount.
    _one_trial(
        experiment_dir(root, "h001", "e02") / "results",
        {
            "model_name": "headless/agent",
            "headless_pricing_unknown": True,
            "api_costs": 12.0,
            "embed_cost": 0.25,
        },
    )

    with _server(workspace) as base:
        (row,) = _get(base, "/api/workspace")["directions"]
        snapshot = _get(base, _snapshot_path("alpha"))
    assert row["spend_usd"] == snapshot["spend_usd"] == pytest.approx(1.0)
    assert row["unknown_cost_runs"] == snapshot["unknown_cost_runs"] == 1
    experiments = snapshot["hypotheses"][0]["experiments"]
    assert [e["spend_known"] for e in experiments] == [True, False]
    assert [e["spend_usd"] for e in experiments] == pytest.approx([0.75, 0.25])


def test_workspace_summaries_and_selection_isolate_directions_and_errors(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "fixture-workspace"
    alpha_root = workspace / "alpha"
    alpha = _direction(alpha_root, "Alpha fixture", "Alpha-only hypothesis body.")
    alpha.budget_usd = 40.0
    alpha.data = DataSplit(
        source_hash="fixture-source-hash",
        timestamp_col="timestamp",
        dev_path="fixture-dev.csv",
        n_total=10,
        n_dev=8,
        lockbox_start="2025-01-01",
    )
    hypothesis = alpha.hypotheses[0]
    hypothesis.status = Status.challenged
    hypothesis.confirmation = {HELD_OUT: 987.654}
    hypothesis.experiments.append(
        Experiment(
            id="e01",
            created=CREATED,
            evaluator_hash="fixture-evaluator-hash",
            data_path="fixture-dev.csv",
            data_hash="fixture-data-hash",
            n_evaluated=1,
            n_probed=1,
            skeptic={HELD_OUT: 987.654},
        )
    )
    alpha.save(alpha_root)
    _one_trial(experiment_dir(alpha_root, "h001", "e01") / "results")
    append_log(alpha_root, "h001", "e01", "skeptic", HELD_OUT)
    _direction(
        workspace / "beta",
        "Beta fixture",
        "Beta-only hypothesis body.",
        "A second beta hypothesis.",
    )
    (workspace / "unrelated").mkdir()
    corrupt = workspace / "broken"
    corrupt.mkdir()
    (corrupt / "notebook.json").write_text("{not valid JSON", encoding="utf-8")
    expected_library = _shared_library(workspace)

    with _server(workspace) as base:
        overview = _get(base, "/api/workspace")
        assert overview["library"] == expected_library
        rows = {row["id"]: row for row in overview["directions"]}
        assert set(rows) == {"alpha", "beta", "broken"}
        assert rows["alpha"]["hypotheses"] == 1
        assert rows["beta"]["hypotheses"] == 2
        assert rows["alpha"]["trials"] == 2
        assert rows["alpha"]["spend_usd"] == 0.75
        assert rows["alpha"]["budget_usd"] == 40.0
        assert rows["beta"]["budget_usd"] is None
        assert rows["alpha"]["lockbox_opened"] is False
        assert rows["beta"]["lockbox_opened"] is None
        assert rows["broken"]["error"] is not None
        assert {"hypotheses", "trials", "spend_usd"}.isdisjoint(rows["broken"])
        serialized = json.dumps(overview)
        assert HELD_OUT not in serialized
        assert "987.654" not in serialized

        alpha_snapshot = _get(base, _snapshot_path("alpha"))
        assert alpha_snapshot["direction"] == "Alpha fixture"
        assert len(alpha_snapshot["hypotheses"]) == 1
        assert "Alpha-only hypothesis body." in alpha_snapshot["hypotheses"][0]["body"]
        assert alpha_snapshot["trials"] == 2
        assert alpha_snapshot["spend_usd"] == 0.75
        assert alpha_snapshot["library"] == expected_library

        _assert_error(base, _snapshot_path("broken"), 500)
        beta_snapshot = _get(base, _snapshot_path("beta"))
        assert beta_snapshot["direction"] == "Beta fixture"
        assert len(beta_snapshot["hypotheses"]) == 2
        assert "Beta-only hypothesis body." in beta_snapshot["hypotheses"][0]["body"]
        assert beta_snapshot["trials"] == 0
        assert beta_snapshot["library"] == expected_library
        assert "Alpha-only hypothesis body." not in json.dumps(beta_snapshot)


def test_empty_workspace_can_still_show_shared_library(tmp_path: Path) -> None:
    workspace = tmp_path / "empty-workspace"
    expected_library = _shared_library(workspace)
    with _server(workspace) as base:
        overview = _get(base, "/api/workspace")
        assert overview["directions"] == []
        assert overview["library"] == expected_library


@pytest.mark.parametrize(
    "query",
    ["", "?direction=", "?direction=%20%20", "?direction=alpha&direction=alpha"],
    ids=["missing", "empty", "whitespace", "repeated-even-if-identical"],
)
def test_snapshot_requires_exactly_one_nonblank_direction(
    tmp_path: Path, query: str
) -> None:
    workspace = tmp_path / "workspace"
    _direction(workspace / "alpha", "Alpha fixture")
    with _server(workspace) as base:
        _assert_error(base, "/api/snapshot" + query, 400)


def test_encoded_directory_name_is_decoded_exactly_once(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    direction_id = "中文 space + % literal%2F # &"
    _direction(
        workspace / direction_id,
        "Encoded-name fixture",
        "Only the literal encoded-name direction contains this body.",
    )
    with _server(workspace) as base:
        overview = _get(base, "/api/workspace")
        assert [row["id"] for row in overview["directions"]] == [direction_id]
        selected = _get(base, _snapshot_path(direction_id))
        assert selected["direction"] == "Encoded-name fixture"
        assert (
            "Only the literal encoded-name direction"
            in selected["hypotheses"][0]["body"]
        )


@pytest.mark.parametrize(
    "direction_id",
    ["missing", "../outside", r"..\outside"],
    ids=["unknown", "parent-traversal", "windows-parent-traversal"],
)
def test_unknown_and_traversing_directions_are_not_found(
    tmp_path: Path, direction_id: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _direction(tmp_path / "outside", "Outside fixture", "Never serve this body.")
    with _server(workspace) as base:
        _assert_error(base, _snapshot_path(direction_id), 404)


@pytest.mark.parametrize("link_kind", ["direction", "notebook"])
def test_symlinked_directions_and_notebooks_are_not_discovered_or_selected(
    tmp_path: Path, link_kind: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    _direction(outside, "Outside fixture", "Never serve this body.")
    alias = workspace / "alias"
    if link_kind == "direction":
        link, target = alias, outside
    else:
        alias.mkdir()
        link, target = alias / "notebook.json", outside / "notebook.json"
    try:
        link.symlink_to(target, target_is_directory=link_kind == "direction")
    except OSError as error:
        denied_on_windows = sys.platform == "win32" and (
            error.winerror in {5, 50, 1314}
            or error.errno in {errno.EACCES, errno.EPERM, errno.ENOSYS, errno.ENOTSUP}
        )
        if denied_on_windows:
            pytest.skip(f"Windows denied {link_kind} symlink creation: {error}")
        raise

    with _server(workspace) as base:
        assert _get(base, "/api/workspace")["directions"] == []
        _assert_error(base, _snapshot_path("alias"), 404)
