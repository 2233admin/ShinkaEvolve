import base64
import json
import os
import sqlite3
import threading
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import pytest

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.research import ui
from shinka.research.notebook import Experiment, Notebook, experiment_dir


def _write_program_database(path: Path, name: str, score: float) -> Program:
    path.parent.mkdir(parents=True, exist_ok=True)
    program = Program(
        id=f"{name}-program",
        code=f"def candidate():\n    return {name!r}\n",
        correct=True,
        generation=0,
        island_idx=0,
        combined_score=score,
        public_metrics={"train_score": score},
        private_metrics={"held_out_secret": f"{name}-private-secret"},
    )
    db = ProgramDatabase(
        config=DatabaseConfig(db_path=str(path), num_islands=1),
        embedding_model="",
        read_only=False,
    )
    try:
        db.add(program, defer_maintenance=True)
    finally:
        db.close()
    return program


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, dict[str, Program]]:
    root = tmp_path / "workspace"
    ordinary = root / "ordinary-run" / "results" / "programs.sqlite"
    programs = {
        ordinary.relative_to(root).as_posix(): _write_program_database(
            ordinary, "ordinary", 1.25
        )
    }
    for index, name in enumerate(("direction-a", "direction-b"), start=2):
        direction = root / name
        direction.mkdir()
        nb = Notebook(direction=name, budget_usd=40.0)
        hypothesis = nb.add_hypothesis(direction, f"{name} hypothesis", "Evidence.")
        hypothesis.experiments.append(
            Experiment(id="e01", evaluator_hash="x", data_path="d", data_hash="y")
        )
        nb.save(direction)
        path = (
            experiment_dir(direction, hypothesis.id, "e01")
            / "results"
            / "programs.sqlite"
        )
        programs[path.relative_to(root).as_posix()] = _write_program_database(
            path, name, float(index)
        )
    return root, programs


@contextmanager
def _workspace_server(root: Path, runs_root: Path | None = None) -> Iterator[str]:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(ui._Handler, root=root, runs_root=runs_root)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    try:
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join()
    finally:
        server.server_close()


def _get(base: str, endpoint: str, **query: str) -> tuple[int, bytes]:
    url = f"{base}/{endpoint.lstrip('/')}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(
        url, headers={"Origin": "https://untrusted.example"}
    )
    try:
        response = urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        assert not any(
            key.lower().startswith("access-control-allow-") for key in response.headers
        )
        return response.status, response.read()


def _json(base: str, endpoint: str, **query: str):
    status, body = _get(base, endpoint, **query)
    assert status == 200, (status, body.decode("utf-8", errors="replace"))
    return json.loads(body)


def test_workspace_analysis_discovers_runs_and_isolates_programs(workspace) -> None:
    root, expected = workspace
    with _workspace_server(root) as base:
        listing = _json(base, "list_databases")
        discovered = {
            entry["path"].removeprefix("workspace/"): entry["path"] for entry in listing
        }
        assert set(discovered) == set(expected)
        # Read A, B, then A again: cached programs must not cross run boundaries.
        paths = list(expected)
        for path in [*paths, paths[1]]:
            program = expected[path]
            programs = _json(base, "get_programs", db_path=discovered[path])
            assert [item["id"] for item in programs] == [program.id]
            assert programs[0]["code"] == program.code
            assert programs[0]["public_metrics"] == program.public_metrics
            assert programs[0]["combined_score"] == program.combined_score
    assert not (root / "ordinary-run" / "notebook.json").exists()


def test_ordinary_runs_outside_research_workspace_are_discovered_and_confined(
    workspace,
):
    root, research_programs = workspace
    runs_root = root.parent / "ordinary-history"
    db = runs_root / "task" / "historical-run" / "programs.sqlite"
    program = _write_program_database(db, "outside-workspace-ordinary", 8.75)
    unrelated = root.parent / "unrelated" / "programs.sqlite"
    _write_program_database(unrelated, "unauthorized", 99.0)
    with _workspace_server(root, runs_root=runs_root) as base:
        listing = _json(base, "list_databases")
        identifiers = {entry["path"] for entry in listing}
        assert identifiers == {
            *(f"workspace/{path}" for path in research_programs),
            "runs/task/historical-run/programs.sqlite",
        }
        rows = _json(
            base, "get_programs", db_path="runs/task/historical-run/programs.sqlite"
        )
        assert [row["id"] for row in rows] == [program.id]
        assert rows[0]["code"] == program.code
        assert rows[0]["private_metrics"] == {}
        assert program.private_metrics["held_out_secret"] not in json.dumps(rows)
        status, _ = _get(base, "get_programs", db_path=unrelated.as_posix())
        assert status == 403


@pytest.mark.parametrize(
    "endpoint", ["get_programs", "get_programs_summary", "get_program_details"]
)
def test_workspace_analysis_keeps_private_metrics_hidden(workspace, endpoint) -> None:
    root, expected = workspace
    with _workspace_server(root) as base:
        for path, program in expected.items():
            payload = _json(
                base,
                endpoint,
                db_path=f"workspace/{path}",
                id=program.id,
                show_private="true",
            )
            rows = [payload] if endpoint == "get_program_details" else payload
            assert [row["id"] for row in rows] == [program.id]
            row = rows[0]
            assert row["public_metrics"] == program.public_metrics
            assert row["combined_score"] == program.combined_score
            assert row["private_metrics"] == {}
            assert row["private_metrics_hidden"] is True
            if endpoint == "get_programs_summary":
                assert "code" not in row
            else:
                assert row["code"] == program.code
            for other in expected.values():
                secret = other.private_metrics["held_out_secret"]
                assert secret not in json.dumps(payload)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({}, None),
        ({"api_costs": 0.0}, None),
        ({"api_costs": None, "embed_cost": 0, "novelty_cost": 0, "meta_cost": 0}, None),
        ({"api_costs": 0, "embed_cost": 0, "novelty_cost": 0, "meta_cost": 0}, 0.0),
        (
            {
                "api_costs": 0.2,
                "embed_cost": 0.01,
                "novelty_cost": 0.03,
                "meta_cost": 0.04,
            },
            0.28,
        ),
    ],
)
def test_run_statistics_distinguish_unrecorded_cost_from_free(
    tmp_path: Path, metadata: dict, expected: float | None
) -> None:
    root = tmp_path / "workspace"
    path = root / "ordinary" / "programs.sqlite"
    _write_program_database(path, "cost-boundary", -0.5)

    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE programs SET metadata = ?", (json.dumps(metadata),))
    with _workspace_server(root) as base:
        stats = _json(
            base, "get_database_stats", db_path="workspace/ordinary/programs.sqlite"
        )
    assert stats["program_count"] == 1
    assert stats["best_score"] == -0.5
    if expected is None:
        assert stats["total_cost"] is None
    else:
        assert stats["total_cost"] == pytest.approx(expected)


def test_snapshot_analysis_links_open_the_correct_experiment(workspace) -> None:
    root, expected = workspace
    with _workspace_server(root) as base:
        for direction in ("direction-a", "direction-b"):
            snapshot = _json(base, "api/snapshot", direction=direction)
            hypothesis = snapshot["hypotheses"][0]
            experiment = hypothesis["experiments"][0]
            link = urllib.parse.urlsplit(experiment["analysis_url"])
            assert not link.scheme and not link.netloc
            analysis_url = urllib.parse.parse_qs(link.query)["analysis"][0]
            analysis = urllib.parse.urlsplit(analysis_url)
            assert not analysis.scheme and not analysis.netloc
            status, _ = _get(base, analysis_url)
            assert status == 200
            db_path = urllib.parse.parse_qs(analysis.query)["db_path"][0]
            program = next(
                value
                for path, value in expected.items()
                if Path(path).parts[0] == direction
            )
            rows = _json(base, "get_programs", db_path=db_path)
            assert [row["id"] for row in rows] == [program.id]
            detail = _json(base, "get_program_details", db_path=db_path, id=program.id)
            assert detail["code"] == program.code
            assert detail["public_metrics"] == program.public_metrics


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows did not grant permission to create a symlink")
        raise


@pytest.mark.parametrize("escape", ["relative", "absolute", "symlink"])
def test_workspace_analysis_rejects_external_databases(workspace, escape) -> None:
    root, _ = workspace
    outside = root.parent / "outside" / "programs.sqlite"
    program = _write_program_database(outside, "outside-secret", 99.0)
    if escape == "relative":
        path = "workspace/../outside/programs.sqlite"
    elif escape == "absolute":
        path = str(outside)
    else:
        link = root / "escaped.sqlite"
        _symlink(link, outside)
        path = f"workspace/{link.name}"
    with _workspace_server(root) as base:
        if escape == "symlink":
            listing = _json(base, "list_databases")
            assert path not in {entry["path"] for entry in listing}
        for endpoint in (
            "get_programs",
            "get_programs_summary",
            "get_program_details",
            "get_program_count",
        ):
            status, body = _get(base, endpoint, db_path=path, id=program.id)
            assert status == 403, (endpoint, status, body)
            assert program.id.encode() not in body
            assert program.code.encode() not in body
            assert b"outside-secret-private-secret" not in body


# A real one-pixel PNG, so successful plot reads cannot be replaced by blanket denial.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwM"
    "CAO+jRZkAAAAASUVORK5CYII="
)


def test_workspace_analysis_serves_only_registered_run_plots(workspace) -> None:
    root, expected = workspace
    with _workspace_server(root) as base:
        for path, program in expected.items():
            plot = (
                root / Path(path).parent / "gen_0" / "results" / "plots" / "score.png"
            )
            plot.parent.mkdir(parents=True)
            plot.write_bytes(PNG)
            plots = _json(
                base,
                "get_plots",
                db_path=f"workspace/{path}",
                generation="0",
                program_id=program.id,
            )
            assert [entry["filename"] for entry in plots] == ["score.png"]
            endpoint = "plot_file/" + urllib.parse.quote(
                Path(plots[0]["path"]).as_posix(), safe="/"
            )
            status, body = _get(base, endpoint)
            assert status == 200
            assert body == PNG


@pytest.mark.parametrize(
    "relative",
    [
        "library/wiki/private.md",
        "direction-a/notebook.json",
        "ordinary-run/results/gen_0/results/plots/notes.txt",
        "library/wiki/diagram.png",
        "unregistered/gen_0/results/plots/diagram.png",
    ],
)
def test_workspace_plot_file_cannot_read_non_plot_workspace_files(
    workspace, relative
) -> None:
    root, _ = workspace
    target = root / relative
    if target.exists():
        protected = target.read_bytes()
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        protected = PNG if target.suffix == ".png" else b"workspace-library-secret"
        target.write_bytes(protected)
    with _workspace_server(root) as base:
        status, body = _get(
            base, "plot_file/" + urllib.parse.quote(f"workspace/{relative}", safe="/")
        )
        assert status in {403, 404}, (status, body)
        assert protected not in body


@pytest.mark.parametrize("escape", ["relative", "file_symlink", "directory_symlink"])
def test_workspace_plot_file_rejects_external_artifacts(workspace, escape) -> None:
    root, expected = workspace
    outside = root.parent / "private-plots"
    outside.mkdir()
    secret = b"outside-plot-secret"
    external_plot = outside / "secret.png"
    external_plot.write_bytes(secret)
    path = next(iter(expected))
    plots = root / Path(path).parent / "gen_0" / "results" / "plots"
    if escape == "relative":
        relative = "../private-plots/secret.png"
    elif escape == "file_symlink":
        plots.mkdir(parents=True)
        link = plots / "secret.png"
        _symlink(link, external_plot)
        relative = link.relative_to(root).as_posix()
    else:
        plots.parent.mkdir(parents=True)
        _symlink(plots, outside)
        relative = (plots / "secret.png").relative_to(root).as_posix()
    with _workspace_server(root) as base:
        status, body = _get(
            base, "plot_file/" + urllib.parse.quote(f"workspace/{relative}", safe="/")
        )
        assert status == 403, (status, body)
        assert secret not in body
        if escape != "relative":
            status, body = _get(
                base,
                "get_plots",
                db_path=f"workspace/{path}",
                generation="0",
                program_id="p",
            )
            assert status in {200, 403}, (status, body)
            assert b"secret.png" not in body


@pytest.mark.parametrize("form", ["bare", "absolute", "unconfigured-root"])
def test_analysis_rejects_paths_without_an_authorized_namespace(workspace, form):
    root, expected = workspace
    relative = next(iter(expected))
    path = {
        "bare": relative,
        "absolute": str(root / relative),
        "unconfigured-root": f"runs/{relative}",
    }[form]
    with _workspace_server(root) as base:
        status, body = _get(base, "get_programs", db_path=path)
        assert status == 403
        assert expected[relative].code.encode() not in body


def test_fictional_run_cannot_read_raw_source_as_meta(workspace):
    root, _ = workspace
    raw = root / "library" / "raw"
    raw.mkdir(parents=True)
    secret = "raw-source-is-not-analysis-metadata"
    (raw / "meta_0.txt").write_text(secret, encoding="utf-8")
    with _workspace_server(root) as base:
        status, body = _get(
            base,
            "get_meta_content",
            db_path="workspace/library/raw/nonexistent.sqlite",
            processed_count="0",
        )
        assert status == 403
        assert secret.encode() not in body


def test_fake_database_file_does_not_authorize_plot_download(workspace):
    root, _ = workspace
    run = root / "not-a-run"
    plots = run / "gen_0" / "results" / "plots"
    plots.mkdir(parents=True)
    (run / "fake.sqlite").write_bytes(b"this is not a program database")
    (plots / "secret.png").write_bytes(PNG)
    with _workspace_server(root) as base:
        status, body = _get(
            base, "plot_file/workspace/not-a-run/gen_0/results/plots/secret.png"
        )
        assert status == 403
        assert PNG not in body


def test_run_discovery_excludes_source_and_non_program_databases(workspace):
    root, expected = workspace
    _write_program_database(root / "library" / "raw" / "source.sqlite", "source", 90.0)
    _write_program_database(
        root / "direction-a" / "data" / "source.sqlite", "data", 91.0
    )
    (root / "broken.sqlite").write_bytes(b"not a database")
    with _workspace_server(root) as base:
        listing = _json(base, "list_databases")
        assert {row["path"] for row in listing} == {
            f"workspace/{relative}" for relative in expected
        }


@pytest.mark.parametrize(
    ("endpoint", "error"),
    [
        ("get_database_stats", "Unable to read run statistics"),
        ("get_system_prompts", "Unable to read system prompts"),
    ],
)
@pytest.mark.parametrize("prompt_failure", ["corrupt", "missing-system-prompts"])
def test_run_reads_report_prompt_database_errors_and_recover(
    tmp_path: Path, prompt_failure: str, endpoint: str, error: str
) -> None:
    root = tmp_path / "workspace"
    path = root / "ordinary" / "programs.sqlite"
    _write_program_database(path, "stats-error-boundary", -0.5)
    metadata = {
        "api_costs": 0.2,
        "embed_cost": 0.01,
        "novelty_cost": 0.03,
        "meta_cost": 0.04,
    }
    conn = sqlite3.connect(path)
    try:
        conn.execute("UPDATE programs SET metadata = ?", (json.dumps(metadata),))
        conn.commit()
    finally:
        conn.close()

    prompt_db = path.with_name("prompts.sqlite")
    if prompt_failure == "corrupt":
        prompt_db.write_bytes(b"not a SQLite database")
    else:
        conn = sqlite3.connect(prompt_db)
        try:
            conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
            conn.commit()
        finally:
            conn.close()

    with _workspace_server(root) as base:
        status, body = _get(
            base, endpoint, db_path="workspace/ordinary/programs.sqlite"
        )
        assert status == 500, (status, body.decode("utf-8", errors="replace"))
        assert json.loads(body) == {"error": error}

        prompt_db.unlink()
        stats = _json(
            base, "get_database_stats", db_path="workspace/ordinary/programs.sqlite"
        )
        assert stats["program_count"] == 1
        assert stats["best_score"] == -0.5
        assert stats["total_cost"] == pytest.approx(0.28)
        assert stats["prompt_evo_cost"] == 0.0
        assert (
            _json(
                base, "get_system_prompts", db_path="workspace/ordinary/programs.sqlite"
            )
            == []
        )


def test_pdf_export_never_disguises_text_when_converter_is_unavailable(
    workspace, monkeypatch
):
    root, expected = workspace
    relative = next(iter(expected))
    meta = (root / relative).parent / "meta"
    meta.mkdir()
    (meta / "meta_1.txt").write_text(
        "# Saved research notes\nPDF conversion boundary.\n", encoding="utf-8"
    )
    monkeypatch.setenv("PATH", "")
    with _workspace_server(root) as base:
        status, body = _get(
            base,
            "download_meta_pdf",
            db_path=f"workspace/{relative}",
            processed_count="1",
        )
    assert status == 500
    assert b"PDF export requires wkhtmltopdf or pandoc" in body
