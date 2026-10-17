import json
import os
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
from shinka.database.prompt_dbase import (
    SystemPrompt,
    SystemPromptConfig,
    SystemPromptDatabase,
)
from shinka.research import ui


@contextmanager
def _tree_server(experiments: Path) -> Iterator[str]:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(
            ui._TreeHandler,
            search_root=str(experiments.resolve()),
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


def _get(base: str, endpoint: str, **query: str) -> tuple[int, str]:
    url = f"{base}/{endpoint}?{urllib.parse.urlencode(query)}"
    try:
        response = urllib.request.urlopen(url, timeout=10)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, response.read().decode("utf-8")


def _assert_denied(status: int, body: str, secret: str) -> None:
    assert secret not in body
    assert status in {403, 500}, (status, body)


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows did not grant permission to create a symlink")
        raise


def _write_program_database(path: Path, failure_paths: tuple[Path, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = ProgramDatabase(
        config=DatabaseConfig(db_path=str(path), num_islands=1),
        embedding_model="",
        read_only=False,
    )
    try:
        db.add(
            Program(
                id="in-scope-parent",
                code="def candidate():\n    return 1\n",
                correct=True,
                generation=0,
                island_idx=0,
                combined_score=1.0,
            ),
            defer_maintenance=True,
        )
        for generation, failure_path in enumerate(failure_paths, start=1):
            db.record_attempt_event(
                generation=generation,
                stage="proposal",
                status="failed",
                details={
                    "node_kind": "failed_proposal",
                    "parent_id": "in-scope-parent",
                    "language": "python",
                    "failure_reason": "Proposal failed before evaluation",
                    "failure_json_path": str(failure_path),
                },
            )
    finally:
        db.close()


def _write_prompt_database(path: Path, prompt: SystemPrompt) -> None:
    db = SystemPromptDatabase(SystemPromptConfig(db_path=str(path)), read_only=False)
    try:
        db.add(prompt)
    finally:
        db.close()


def test_tree_rejects_external_sibling_prompt_database(tmp_path: Path) -> None:
    experiments = tmp_path / "direction" / "experiments"
    relative_db = "h001/e01/results/programs.sqlite"
    db_path = experiments / relative_db
    _write_program_database(db_path)
    bad_relative_db = "h001/e02/results/programs.sqlite"
    bad_db_path = experiments / bad_relative_db
    _write_program_database(bad_db_path)
    prompt_path = db_path.with_name("prompts.sqlite")
    normal_prompt = SystemPrompt(
        id="in-scope-prompt", prompt_text="Use the in-scope proposal instructions"
    )
    _write_prompt_database(prompt_path, normal_prompt)
    external_path = tmp_path / "other-direction" / "prompts.sqlite"
    secret = "outside-direction-prompt-content-issue12"
    _write_prompt_database(
        external_path,
        SystemPrompt(id="external-prompt", prompt_text=secret),
    )
    _symlink(bad_db_path.with_name("prompts.sqlite"), external_path)

    with _tree_server(experiments) as base:
        status, body = _get(base, "get_system_prompts", db_path=relative_db)
        assert status == 200
        prompts = json.loads(body)
        assert [prompt["id"] for prompt in prompts] == [normal_prompt.id]
        assert prompts[0]["prompt_text"] == normal_prompt.prompt_text

        # A different in-scope program DB has an external sibling prompt DB.
        # Keep the positive reader's files intact while its handler winds down.
        status, body = _get(base, "get_system_prompts", db_path=bad_relative_db)
        _assert_denied(status, body, secret)


def test_tree_rejects_external_meta_file_symlink(tmp_path: Path) -> None:
    experiments = tmp_path / "direction" / "experiments"
    relative_db = "h001/e01/results/programs.sqlite"
    db_path = experiments / relative_db
    _write_program_database(db_path)
    meta_dir = db_path.parent / "meta"
    meta_dir.mkdir()
    normal_content = "# In-scope proposal summary\nThe candidate scored 1.0.\n"
    (meta_dir / "meta_1.txt").write_text(normal_content, encoding="utf-8")
    external_path = tmp_path / "external-meta.txt"
    secret = "outside-direction-meta-content-issue12"
    external_path.write_text(secret, encoding="utf-8")
    _symlink(meta_dir / "meta_2.txt", external_path)

    with _tree_server(experiments) as base:
        status, body = _get(
            base, "get_meta_content", db_path=relative_db, processed_count="1"
        )
        assert status == 200
        meta = json.loads(body)
        assert meta["processed_count"] == 1
        assert meta["filename"] == "meta_1.txt"
        assert meta["content"] == normal_content

        status, body = _get(
            base, "get_meta_content", db_path=relative_db, processed_count="2"
        )
        _assert_denied(status, body, secret)


@pytest.mark.parametrize(
    "escape", ["failure_json", "generated_code_path", "fallback_code_symlink"]
)
def test_tree_rejects_external_failed_proposal_artifacts(
    tmp_path: Path, escape: str
) -> None:
    experiments = tmp_path / "direction" / "experiments"
    relative_db = "h001/e01/results/programs.sqlite"
    db_path = experiments / relative_db
    normal_dir = db_path.parent / "gen_1"
    normal_dir.mkdir(parents=True)
    normal_code = "def candidate():\n    return 'in-scope failed proposal'\n"
    normal_code_path = normal_dir / "main.py"
    normal_code_path.write_text(normal_code, encoding="utf-8")
    normal_failure_path = normal_dir / "failure.json"
    normal_explanation = "in-scope failure-artifact explanation"
    normal_payload = {
        "node_kind": "failed_proposal",
        "language": "python",
        "novelty_explanation": normal_explanation,
        "generated_code_available": True,
        "artifacts": {"generated_code_path": str(normal_code_path)},
    }
    normal_failure_path.write_text(json.dumps(normal_payload), encoding="utf-8")

    external_dir = tmp_path / "other-direction" / "failed-proposal"
    external_dir.mkdir(parents=True)
    secret = f"outside-direction-{escape}-content-issue12"
    bad_dir = db_path.parent / "gen_2"
    bad_dir.mkdir()
    bad_failure_path = bad_dir / "failure.json"
    bad_payload = dict(normal_payload)
    if escape == "failure_json":
        # The log itself selects an external JSON artifact. Its copied
        # explanation would leak even without reading any external code.
        bad_failure_path = external_dir / "failure.json"
        bad_payload["novelty_explanation"] = secret
    else:
        external_code = external_dir / "main.py"
        external_code.write_text(
            f"def candidate():\n    return '{secret}'\n", encoding="utf-8"
        )
        if escape == "generated_code_path":
            # JSON is confined, but its explicit code target is not.
            bad_payload["artifacts"] = {"generated_code_path": str(external_code)}
        else:
            # No explicit target: upstream falls back to main.py beside JSON.
            bad_payload["artifacts"] = {}
            _symlink(bad_dir / "main.py", external_code)
    bad_failure_path.write_text(json.dumps(bad_payload), encoding="utf-8")
    _write_program_database(db_path, (normal_failure_path, bad_failure_path))

    # These endpoints do not use the process-wide program-list cache; the
    # test introduces no cache keys and never clears unrelated entries.
    with _tree_server(experiments) as base:
        status, body = _get(
            base,
            "get_program_details",
            db_path=relative_db,
            id="failed:proposal:1",
        )
        assert status == 200
        failure = json.loads(body)
        assert failure["id"] == "failed:proposal:1"
        assert failure["parent_id"] == "in-scope-parent"
        assert failure["code"] == normal_code
        assert failure["metadata"]["novelty_explanation"] == normal_explanation
        assert failure["metadata"]["generated_code_available"] is True

        status, body = _get(
            base,
            "get_program_details",
            db_path=relative_db,
            id="failed:proposal:2",
        )
        _assert_denied(status, body, secret)


def test_tree_rejects_external_failure_alias_before_language_fallback(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "direction" / "experiments"
    normal_relative = "h001/e01/results/programs.sqlite"
    alias_relative = "h001/e02/results/programs.sqlite"
    normal_db = experiments / normal_relative
    normal_dir = normal_db.parent / "gen_1"
    normal_dir.mkdir(parents=True)
    (normal_dir / "main.ts").write_text(
        "export function candidate() { return 1; }\n", encoding="utf-8"
    )
    failure_path = normal_dir / "failure.json"
    failure_path.write_text("{}", encoding="utf-8")
    external_dir = tmp_path / "other-direction" / "failed-proposal"
    external_dir.mkdir(parents=True)
    suffix = "shinkaexternalsuffixissue12"
    secret = "outside-direction-fallback-code-issue12"
    (external_dir / f"main.{suffix}").write_text(secret, encoding="utf-8")
    alias = external_dir / "failure.json"
    _symlink(alias, failure_path)

    for db_path, artifact in (
        (normal_db, failure_path),
        (experiments / alias_relative, alias),
    ):
        _write_program_database(db_path)
        db = ProgramDatabase(
            config=DatabaseConfig(db_path=str(db_path), num_islands=1),
            embedding_model="",
        )
        try:
            # An explicit language would bypass the vulnerable parent glob.
            db.record_attempt_event(
                generation=1,
                stage="proposal",
                status="failed",
                details={
                    "node_kind": "failed_proposal",
                    "parent_id": "in-scope-parent",
                    "failure_json_path": str(artifact),
                },
            )
        finally:
            db.close()

    with _tree_server(experiments) as base:
        status, body = _get(base, "get_programs_summary", db_path=normal_relative)
        assert status == 200, body
        failure = next(
            node for node in json.loads(body) if node["id"] == "failed:proposal:1"
        )
        assert failure["parent_id"] == "in-scope-parent"
        assert failure["language"] == "typescript"
        status, body = _get(base, "get_programs_summary", db_path=alias_relative)
        assert suffix not in body
        _assert_denied(status, body, secret)
