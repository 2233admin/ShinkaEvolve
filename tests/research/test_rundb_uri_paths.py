"""Populated research results remain readable under literal URI characters."""

import json
import sqlite3
import threading
import urllib.parse
import urllib.request
from contextlib import closing, contextmanager
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from shinka.research import rundb, ui
from shinka.research.notebook import Notebook, experiment_dir

CREATED = "2026-01-02T03:04:05+00:00"
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


def _result_databases(results: Path) -> None:
    results.mkdir(parents=True)
    # Use the real table columns without opening the upstream writable databases.
    with closing(sqlite3.connect(results / "programs.sqlite")) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE programs (
                    id TEXT PRIMARY KEY,
                    code TEXT NOT NULL,
                    language TEXT NOT NULL,
                    parent_id TEXT,
                    archive_inspiration_ids TEXT,
                    top_k_inspiration_ids TEXT,
                    generation INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    code_diff TEXT,
                    combined_score REAL,
                    public_metrics TEXT,
                    private_metrics TEXT,
                    text_feedback TEXT,
                    complexity REAL,
                    embedding TEXT,
                    embedding_pca_2d TEXT,
                    embedding_pca_3d TEXT,
                    embedding_cluster_id INTEGER,
                    correct BOOLEAN DEFAULT 0,
                    children_count INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT,
                    migration_history TEXT,
                    island_idx INTEGER,
                    system_prompt_id TEXT
                )
                """
            )
            connection.executemany(
                "INSERT INTO programs "
                "(id, code, language, generation, timestamp, correct, "
                "combined_score, metadata) VALUES (?, ?, 'python', ?, ?, ?, ?, ?)",
                [
                    (
                        "chosen-program",
                        "def candidate():\n    return 42\n",
                        1,
                        200.0,
                        1,
                        1.25,
                        json.dumps(
                            {
                                "api_costs": 1.0,
                                "embed_cost": 0.25,
                                "novelty_cost": 0.125,
                                "meta_cost": 0.125,
                            }
                        ),
                    ),
                    (
                        "incorrect-program",
                        "raise RuntimeError('incorrect candidate')\n",
                        0,
                        100.0,
                        0,
                        99.0,
                        json.dumps({"api_costs": 0.5}),
                    ),
                    (
                        "migration-copy",
                        "# not an evaluated candidate\n",
                        2,
                        50.0,
                        1,
                        1000.0,
                        json.dumps({"_is_island_copy": True, "api_costs": 100.0}),
                    ),
                    (
                        "spawn-copy",
                        "# not an evaluated candidate either\n",
                        3,
                        75.0,
                        1,
                        2000.0,
                        json.dumps({"_spawned_island": True, "api_costs": 200.0}),
                    ),
                ],
            )
    with closing(sqlite3.connect(results / "prompts.sqlite")) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE system_prompts (
                    id TEXT PRIMARY KEY,
                    prompt_text TEXT NOT NULL,
                    name TEXT,
                    description TEXT,
                    parent_id TEXT,
                    generation INTEGER NOT NULL,
                    program_generation INTEGER NOT NULL DEFAULT 0,
                    patch_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    program_count INTEGER NOT NULL DEFAULT 0,
                    correct_program_count INTEGER NOT NULL DEFAULT 0,
                    total_percentile REAL NOT NULL DEFAULT 0.0,
                    total_improvement REAL NOT NULL DEFAULT 0.0,
                    fitness REAL NOT NULL DEFAULT 0.0,
                    program_scores TEXT,
                    program_ids TEXT,
                    metadata TEXT
                )
                """
            )
            connection.executemany(
                "INSERT INTO system_prompts "
                "(id, prompt_text, generation, patch_type, timestamp, metadata) "
                "VALUES (?, ?, ?, 'initial', ?, ?)",
                [
                    (
                        "prompt-0",
                        "Initial research prompt.",
                        0,
                        90.0,
                        json.dumps({"llm": {"cost": 0.5}}),
                    ),
                    (
                        "prompt-1",
                        "Revised research prompt.",
                        1,
                        190.0,
                        json.dumps({"llm": {"cost": 0.25}}),
                    ),
                ],
            )


def test_populated_results_with_literal_uri_characters_are_read_only(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    direction_id = "中文 space + % literal%2F # &"
    direction = workspace / direction_id
    direction.mkdir(parents=True)
    notebook = Notebook(direction="Literal URI results", created=CREATED)
    hypothesis = notebook.add_hypothesis(
        direction,
        "URI-safe research results",
        "Only the literal URI-character direction contains this hypothesis body.",
    )
    hypothesis.created = CREATED
    experiment = notebook.add_experiment(
        hypothesis.id,
        evaluator_hash="fixture-evaluator-hash",
        data_path="fixture-dev.csv",
        data_hash="fixture-data-hash",
    )
    experiment.created = CREATED
    experiment.n_evaluated = 2
    notebook.save(direction)
    results = experiment_dir(direction, hypothesis.id, experiment.id) / "results"
    _result_databases(results)

    # $2.00 of evaluated program costs plus $0.75 of prompt evolution costs.
    # Administrative copies have deliberately larger scores and costs.
    assert rundb.run_cost(results) == rundb.RunCost(2.0, True, 0.75, True)
    assert rundb.count_evaluated(results) == 2
    assert rundb.trials(results) == [
        (100.0, 0, None, "incorrect-program"),
        (200.0, 1, 1.25, "chosen-program"),
    ]
    assert rundb.best_program(results) == (
        "chosen-program",
        "def candidate():\n    return 42\n",
        1.25,
    )

    with _server(workspace) as base:
        overview = _get(base, "/api/workspace")
        assert len(overview["directions"]) == 1
        row = overview["directions"][0]
        assert row["id"] == direction_id
        assert row["direction"] == "Literal URI results"
        assert row["error"] is None
        assert row["trials"] == 2
        assert row["spend_usd"] == 2.75
        assert row["unknown_cost_runs"] == 0

        query = urllib.parse.urlencode({"direction": direction_id})
        selected = _get(base, "/api/snapshot?" + query)
        assert selected["direction"] == "Literal URI results"
        assert selected["spend_usd"] == 2.75
        assert selected["unknown_cost_runs"] == 0
        assert selected["trials"] == 2
        assert len(selected["hypotheses"]) == 1
        selected_hypothesis = selected["hypotheses"][0]
        assert selected_hypothesis["id"] == hypothesis.id
        assert (
            "Only the literal URI-character direction contains this hypothesis body."
            in selected_hypothesis["body"]
        )
        assert len(selected_hypothesis["experiments"]) == 1
        selected_experiment = selected_hypothesis["experiments"][0]
        assert selected_experiment["id"] == experiment.id
        assert selected_experiment["best_is_score"] == 1.25
        assert selected_experiment["spend_usd"] == 2.75
        assert selected_experiment["spend_known"] is True
        assert [point["score"] for point in selected["trial_points"]] == [None, 1.25]

    # Attempt an actual write through the research program reader, rather than
    # checking its implementation or constructing a separate read-only connection.
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        rundb._query(
            results,
            "UPDATE programs SET code = ? WHERE id = ?",
            ("# must not replace the selected program", "chosen-program"),
        )
    assert rundb.best_program(results) == (
        "chosen-program",
        "def candidate():\n    return 42\n",
        1.25,
    )
    assert rundb.run_cost(results).recorded_usd == 2.75
