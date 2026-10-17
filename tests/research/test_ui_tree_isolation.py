import json
import threading
import urllib.parse
import urllib.request
from contextlib import contextmanager
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.research import ui
from shinka.webui import visualization


def _write_program_database(path: Path, program: Program) -> None:
    path.parent.mkdir(parents=True)
    db = ProgramDatabase(
        config=DatabaseConfig(db_path=str(path), num_islands=1),
        embedding_model="",
        read_only=False,
    )
    try:
        db.add(program, defer_maintenance=True)
    finally:
        db.close()


@contextmanager
def _tree_server(experiments: Path) -> Iterator[str]:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(
            ui._TreeHandler,
            search_root=str(experiments),
            show_private=False,
        ),
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


def test_tree_program_cache_isolates_directions_and_hides_private_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    relative_db_path = "h001/e01/results/programs.sqlite"
    experiments_a = tmp_path / "direction-a" / "experiments"
    experiments_b = tmp_path / "direction-b" / "experiments"
    db_a = experiments_a / relative_db_path
    db_b = experiments_b / relative_db_path
    program_a = Program(
        id="direction-a-program",
        code="def candidate():\n    return 'direction-a'\n",
        correct=True,
        generation=0,
        island_idx=0,
        combined_score=1.25,
        public_metrics={"train_score": 1.25},
        private_metrics={"held_out_secret": "direction-a-private-secret"},
    )
    program_b = Program(
        id="direction-b-program",
        code="def candidate():\n    return 'direction-b'\n",
        correct=True,
        generation=0,
        island_idx=0,
        combined_score=2.5,
        public_metrics={"train_score": 2.5},
        private_metrics={"held_out_secret": "direction-b-private-secret"},
    )
    _write_program_database(db_a, program_a)
    _write_program_database(db_b, program_b)

    # Cover the buggy relative key and the fixed, direction-specific absolute
    # keys without clearing unrelated caches or losing a preexisting entry.
    cache_keys = (relative_db_path, str(db_a.resolve()), str(db_b.resolve()))
    absent = object()
    previous = {key: visualization.db_cache.get(key, absent) for key in cache_keys}
    # A slow CI machine must still exercise a live cache hit on the buggy
    # handler. Only lifetime is patched: DB reads and HTTP output stay real.
    monkeypatch.setattr(visualization, "CACHE_EXPIRATION_SECONDS", float("inf"))
    try:
        for key in cache_keys:
            visualization.db_cache.pop(key, None)
        query = urllib.parse.urlencode({"db_path": relative_db_path})
        with _tree_server(experiments_a) as base_a:
            with _tree_server(experiments_b) as base_b:
                responses = []
                # Warm A and request B consecutively with exactly the same
                # relative path. Independently cold requests miss the bug.
                for base in (base_a, base_b):
                    with urllib.request.urlopen(
                        f"{base}/get_programs?{query}", timeout=10
                    ) as response:
                        assert response.status == 200
                        assert (
                            response.headers.get("Access-Control-Allow-Origin") is None
                        )
                        responses.append(response.read().decode("utf-8"))

                for text, expected in zip(responses, (program_a, program_b)):
                    programs = json.loads(text)
                    assert [program["id"] for program in programs] == [expected.id]
                    assert programs[0]["code"] == expected.code
                    assert programs[0]["public_metrics"] == expected.public_metrics
                    assert programs[0]["private_metrics"] == {}
                    assert programs[0]["private_metrics_hidden"] is True
                    assert "direction-a-private-secret" not in text
                    assert "direction-b-private-secret" not in text
    finally:
        # Both servers (including request threads) are closed before restoring
        # these entries. Remove only keys absent before this test ran.
        for key, value in previous.items():
            if value is absent:
                visualization.db_cache.pop(key, None)
            else:
                visualization.db_cache[key] = value
