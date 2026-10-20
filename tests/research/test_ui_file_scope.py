"""Filesystem read-authority regressions at the real workspace HTTP seam."""

import errno
import json
import sqlite3
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing, contextmanager
from functools import partial
from http.client import HTTPException
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from shinka.research import library, ui
from shinka.research.notebook import (
    Experiment,
    Notebook,
    append_log,
    experiment_dir,
)

CREATED = "2026-01-02T03:04:05+00:00"
SECRET = "read-scope-sibling-secret"
SIBLING_SCORE = 9876.54321
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@contextmanager
def _server(workspace: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(ui._Handler, root=workspace))
    # server_close must join request threads as well as closing the listener.
    server.daemon_threads = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    try:
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive(), "workspace HTTP server did not stop"
    finally:
        server.server_close()


def _request(base: str, path: str) -> tuple[int, str]:
    try:
        response = HTTP.open(base + path, timeout=5)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.code, response.read().decode("utf-8")


def _snapshot_path(direction: str) -> str:
    return "/api/snapshot?" + urllib.parse.urlencode({"direction": direction})


def _denied(response: tuple[int, str]) -> None:
    status, text = response
    assert SECRET not in text
    assert str(SIBLING_SCORE) not in text
    assert status == 500, text


def _payload(response: tuple[int, str]) -> dict:
    status, text = response
    assert status == 200, text
    return json.loads(text)


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        denied_on_windows = sys.platform == "win32" and (
            getattr(error, "winerror", None) in {5, 50, 1314}
            or error.errno in {errno.EACCES, errno.EPERM, errno.ENOSYS, errno.ENOTSUP}
        )
        if denied_on_windows:
            pytest.skip(f"Windows denied symlink creation for {link}: {error}")
        raise


def _one_program(results: Path, *, name: str, score: float, cost: float) -> None:
    """Use the minimal programs schema consumed by the real research readers."""
    results.mkdir(parents=True)
    # sqlite3's transaction context does not close the connection on Windows.
    with closing(sqlite3.connect(results / "programs.sqlite")) as connection:
        with connection:
            connection.execute(
                "CREATE TABLE programs (id TEXT, code TEXT, timestamp REAL, "
                "generation INTEGER, correct INTEGER, combined_score REAL, "
                "private_metrics TEXT, metadata TEXT)"
            )
            connection.execute(
                "INSERT INTO programs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    name,
                    f"# {name} candidate",
                    100.0,
                    0,
                    1,
                    score,
                    json.dumps({"held_out": SECRET}),
                    json.dumps({"api_costs": cost, "embed_cost": 0.25}),
                ),
            )


def _direction(root: Path, *, sibling: bool) -> None:
    root.mkdir(parents=True)
    notebook = Notebook(
        direction="Sibling fixture" if sibling else "Selected fixture",
        created=CREATED,
        budget_usd=20.0,
    )
    body = SECRET if sibling else "Selected direction body."
    hypothesis = notebook.add_hypothesis(root, "Fixture hypothesis", body)
    hypothesis.created = CREATED
    hypothesis.experiments.append(
        Experiment(
            id="e01",
            created=CREATED,
            evaluator_hash="fixture-evaluator-hash",
            data_path="fixture-dev.csv",
            data_hash="fixture-data-hash",
            n_evaluated=1,
            n_probed=1,
        )
    )
    notebook.save(root)
    append_log(root, "h001", "e01", "run", body)
    _one_program(
        experiment_dir(root, "h001", "e01") / "results",
        name="sibling-program" if sibling else "selected-program",
        score=SIBLING_SCORE if sibling else 1.5,
        cost=2.0 if sibling else 0.5,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    _direction(root / "alpha", sibling=False)
    _direction(root / "beta", sibling=True)
    library.ensure(root / "library")
    return root


def _healthy_sibling(response: tuple[int, str]) -> None:
    snapshot = _payload(response)
    assert snapshot["direction"] == "Sibling fixture"
    assert snapshot["trials"] == 2
    assert snapshot["spend_usd"] == pytest.approx(2.25)
    assert SECRET in snapshot["hypotheses"][0]["body"]
    assert snapshot["events"][0]["note"] == SECRET
    experiment = snapshot["hypotheses"][0]["experiments"][0]
    assert experiment["best_is_score"] == SIBLING_SCORE
    assert experiment["spend_usd"] == pytest.approx(2.25)
    assert [point["score"] for point in snapshot["trial_points"]] == [SIBLING_SCORE]


def _local_denial(base: str, *, summary_reads_target: bool) -> None:
    # Exercise all three endpoints before asserting, even on the RED version.
    selected = _request(base, _snapshot_path("alpha"))
    summary_response = _request(base, "/api/workspace")
    sibling = _request(base, _snapshot_path("beta"))
    _denied(selected)
    summary = _payload(summary_response)
    rows = {row["id"]: row for row in summary["directions"]}
    assert set(rows) == {"alpha", "beta"}
    if summary_reads_target:
        assert set(rows["alpha"]) == {"id", "direction", "error"}
        assert rows["alpha"]["error"]
    # The hypothesis body is not a summary input: do not require a full
    # snapshot there.
    assert rows["beta"]["error"] is None
    assert rows["beta"]["direction"] == "Sibling fixture"
    assert rows["beta"]["hypotheses"] == 1
    assert rows["beta"]["trials"] == 2
    assert rows["beta"]["spend_usd"] == pytest.approx(2.25)
    assert SECRET not in summary_response[1]
    # The sibling's own row shows its in-sample best; the selected row must not.
    assert rows["beta"]["evidence"]["best_score"] == SIBLING_SCORE
    assert str(SIBLING_SCORE) not in json.dumps(rows["alpha"])
    _healthy_sibling(sibling)


def _shared_library_denial(base: str) -> None:
    # Both endpoints read the shared library; neither may serve its target.
    responses = [
        _request(base, _snapshot_path("alpha")),
        _request(base, "/api/workspace"),
    ]
    for response in responses:
        _denied(response)


def test_library_root_cannot_relocate_outside_workspace(
    workspace: Path, tmp_path: Path
) -> None:
    lib = workspace / "library"
    outside = tmp_path / "outside-library"
    lib.rename(outside)
    (outside / "wiki" / "secret.md").write_text(SECRET, encoding="utf-8")
    _symlink(lib, outside, directory=True)

    with _server(workspace) as base:
        _shared_library_denial(base)


def test_library_wiki_leaf_cannot_read_outside_library(workspace: Path) -> None:
    # Inside the workspace is still outside the shared library's authority.
    outside = workspace / "private-note.md"
    outside.write_text(SECRET, encoding="utf-8")
    _symlink(workspace / "library" / "wiki" / "secret.md", outside)

    with _server(workspace) as base:
        _shared_library_denial(base)


def test_library_raw_hashing_cannot_follow_a_target_outside_library(
    workspace: Path,
) -> None:
    # pending() hashes raw files rather than returning their contents. A guard
    # only on markdown response reads would miss this separate read entry.
    _symlink(
        workspace / "library" / "raw" / "source.md",
        workspace / "beta" / "hypotheses" / "h001.md",
    )

    with _server(workspace) as base:
        _shared_library_denial(base)


@pytest.mark.parametrize(
    ("relative_path", "summary_reads_target"),
    [("hypotheses/h001.md", False), ("results.tsv", True)],
    ids=["hypothesis-body", "events-log"],
)
def test_direction_text_leaf_cannot_read_a_sibling_direction(
    workspace: Path, relative_path: str, summary_reads_target: bool
) -> None:
    selected = workspace / "alpha" / relative_path
    selected.unlink()
    _symlink(selected, workspace / "beta" / relative_path)

    with _server(workspace) as base:
        _local_denial(base, summary_reads_target=summary_reads_target)


def test_experiments_root_cannot_relocate_into_a_sibling_direction(
    workspace: Path,
) -> None:
    experiments = workspace / "alpha" / "experiments"
    experiments.rename(workspace / "alpha" / "original-experiments")
    _symlink(experiments, workspace / "beta" / "experiments", directory=True)

    with _server(workspace) as base:
        _local_denial(base, summary_reads_target=True)


def test_program_database_leaf_cannot_read_sibling_results(workspace: Path) -> None:
    relative = Path("experiments/h001/e01/results/programs.sqlite")
    selected = workspace / "alpha" / relative
    selected.unlink()
    _symlink(selected, workspace / "beta" / relative)

    with _server(workspace) as base:
        _local_denial(base, summary_reads_target=True)


@pytest.mark.parametrize("manifest_text", ["[]", "null"], ids=["array", "null"])
def test_non_object_library_manifest_returns_explicit_http_error(
    workspace: Path, manifest_text: str
) -> None:
    lib = workspace / "library"
    # A real raw source is required to reach pending()'s manifest lookup.
    (lib / "raw" / "pending-note.md").write_text(
        "A source awaiting compilation.\n", encoding="utf-8"
    )
    (lib / library.MANIFEST).write_text(manifest_text, encoding="utf-8")

    failures = []
    with _server(workspace) as base:
        for path in ("/api/workspace", _snapshot_path("alpha")):
            # Attempt both routes even if the unfixed handler disconnects.
            # A transport failure must never stand in for an HTTP 500.
            try:
                status, text = _request(base, path)
            except (OSError, HTTPException) as error:
                failures.append(
                    f"{path}: expected HTTP 500, got {type(error).__name__}: {error}"
                )
            else:
                if status != 500:
                    failures.append(f"{path}: expected HTTP 500, got {status}: {text}")
    assert not failures, "\n".join(failures)


def test_object_library_manifest_keeps_concrete_pending_source(
    workspace: Path,
) -> None:
    lib = workspace / "library"
    (lib / "raw" / "pending-note.md").write_text(
        "A source awaiting compilation.\n", encoding="utf-8"
    )
    (lib / library.MANIFEST).write_text("{}\n", encoding="utf-8")

    with _server(workspace) as base:
        responses = [
            _request(base, "/api/workspace"),
            _request(base, _snapshot_path("alpha")),
        ]
    for response in responses:
        payload = _payload(response)
        assert payload["library"]["pending"] == ["raw/pending-note.md"]


@pytest.mark.skipif(
    sys.platform != "win32", reason="Windows directory junction regression"
)
def test_windows_junction_direction_alias_is_not_discovered_or_selected(
    workspace: Path,
) -> None:
    import _winapi

    beta = workspace / "beta"
    alias = workspace / "alias"
    with _server(workspace) as base:
        before_beta_response = _request(base, _snapshot_path("beta"))
        _healthy_sibling(before_beta_response)
        before_beta = _payload(before_beta_response)
        before_rows = {
            row["id"]: row
            for row in _payload(_request(base, "/api/workspace"))["directions"]
        }

        # Keep cleanup around creation too: the native call may leave an empty
        # directory behind if setting its reparse point fails.
        try:
            try:
                _winapi.CreateJunction(str(beta), str(alias))
            except OSError as error:
                if getattr(error, "winerror", None) in {5, 1314} or error.errno in {
                    errno.EACCES,
                    errno.EPERM,
                }:
                    pytest.skip(f"Windows denied directory junction creation: {error}")
                raise

            # A real mount-point junction resolves to beta without being a
            # symbolic link; do not depend on Python 3.12's Path.is_junction().
            assert alias.resolve() == beta.resolve()
            assert not alias.is_symlink()

            # Gather every endpoint before asserting the expected denial, so
            # the RED run still exercises the healthy canonical direction.
            overview_response = _request(base, "/api/workspace")
            alias_response = _request(base, _snapshot_path("alias"))
            beta_response = _request(base, _snapshot_path("beta"))
        finally:
            # On Windows rmdir removes the junction itself, not beta or its
            # contents. Never recursively delete this alias during teardown.
            try:
                alias.rmdir()
            except FileNotFoundError:
                pass

        # Check the target both while aliased and after removing the junction.
        # Full payload equality catches changes beyond the fixed healthy fields.
        _healthy_sibling(beta_response)
        assert _payload(beta_response) == before_beta
        after_cleanup = _request(base, _snapshot_path("beta"))
        _healthy_sibling(after_cleanup)
        assert _payload(after_cleanup) == before_beta
        assert not alias.exists()

        rows = {row["id"]: row for row in _payload(overview_response)["directions"]}
        assert rows["beta"] == before_rows["beta"]
        assert set(rows) == {"alpha", "beta"}
        assert alias_response[0] == 404, alias_response[1]
        assert SECRET not in alias_response[1]
        assert str(SIBLING_SCORE) not in alias_response[1]
