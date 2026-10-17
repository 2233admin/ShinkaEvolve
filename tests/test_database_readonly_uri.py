"""Read-only database paths must retain their literal filesystem meaning."""

import sqlite3
from pathlib import Path

import pytest

from shinka.database import (
    DatabaseConfig,
    Program,
    ProgramDatabase,
    SystemPrompt,
    SystemPromptConfig,
    SystemPromptDatabase,
)


@pytest.fixture(
    params=[
        pytest.param(("record#fragment.sqlite", "record"), id="hash-in-filename"),
        pytest.param(
            ("record%25.sqlite", "record%.sqlite"),
            id="escaped-percent-in-filename",
        ),
        pytest.param(
            ("record%2Frow.sqlite", "record/row.sqlite"),
            id="literal-encoded-slash-in-filename",
        ),
        pytest.param(("record%.sqlite", None), id="bare-percent-in-filename"),
        pytest.param(("space + 研究.sqlite", None), id="space-plus-unicode"),
        pytest.param(
            ("folder#fragment/record.sqlite", "folder"),
            id="hash-in-directory",
        ),
        pytest.param(
            ("folder%2Fchild/record.sqlite", "folder/child/record.sqlite"),
            id="literal-encoded-slash-in-directory",
        ),
    ]
)
def database_paths(request: pytest.FixtureRequest, tmp_path: Path):
    """Include real decoy DB paths selected by truncation or URI decoding."""
    target, alias = request.param
    return tmp_path / target, tmp_path / alias if alias is not None else None


def _write_program(db_path: Path, program: Program) -> None:
    writer = ProgramDatabase(
        config=DatabaseConfig(db_path=str(db_path), num_islands=1),
        embedding_model="",
    )
    try:
        assert writer.add(program, defer_maintenance=True) == program.id
    finally:
        writer.close()


def _write_prompt(db_path: Path, prompt: SystemPrompt) -> None:
    writer = SystemPromptDatabase(
        config=SystemPromptConfig(db_path=str(db_path), archive_size=5)
    )
    try:
        assert writer.add(prompt) == prompt.id
    finally:
        writer.close()


def test_program_database_read_only_preserves_literal_path(database_paths):
    db_path, alias_path = database_paths
    intended = Program(
        id="intended-program",
        code="def intended():\n    return 'original # %2F + 研究'\n",
        correct=True,
        combined_score=1.0,
        generation=0,
        island_idx=0,
        complexity=1.0,
        timestamp=1.0,
    )
    decoy = Program(
        id="alias-program",
        code="def decoy():\n    return 'wrong physical database'\n",
        correct=True,
        combined_score=2.0,
        generation=0,
        island_idx=0,
        complexity=1.0,
        timestamp=2.0,
    )
    if alias_path is not None:
        # A bad URI must not silently select another valid program database.
        _write_program(alias_path, decoy)
    _write_program(db_path, intended)

    reader = ProgramDatabase(
        config=DatabaseConfig(db_path=str(db_path), num_islands=1),
        embedding_model="",
        read_only=True,
    )
    try:
        actual = reader.get(intended.id)
        assert actual is not None
        assert actual.id == intended.id
        assert actual.code == intended.code
        assert actual.combined_score == intended.combined_score
        assert reader.get(decoy.id) is None
        assert [program.id for program in reader.get_all_programs()] == [intended.id]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.conn.execute(
                "UPDATE programs SET code = ? WHERE id = ?",
                ("# forbidden native write", intended.id),
            )

        rejected = Program(
            id="rejected-program",
            code="def rejected():\n    return 0\n",
            island_idx=0,
            complexity=1.0,
            timestamp=3.0,
        )
        with pytest.raises(PermissionError, match="read-only"):
            reader.add(rejected, defer_maintenance=True)
        assert reader.get(rejected.id) is None
        assert [program.id for program in reader.get_all_programs()] == [intended.id]
    finally:
        reader.close()


def test_prompt_database_read_only_preserves_literal_path(database_paths):
    db_path, alias_path = database_paths
    intended = SystemPrompt(
        id="intended-prompt",
        prompt_text="Use the original physical database: # %2F + 研究.",
        generation=0,
        timestamp=1.0,
    )
    decoy = SystemPrompt(
        id="alias-prompt",
        prompt_text="This prompt belongs to the wrong physical database.",
        generation=0,
        timestamp=2.0,
    )
    if alias_path is not None:
        # Seed the same public schema so an accidental alias open is detectable.
        _write_prompt(alias_path, decoy)
    _write_prompt(db_path, intended)

    reader = SystemPromptDatabase(
        config=SystemPromptConfig(db_path=str(db_path), archive_size=5),
        read_only=True,
    )
    try:
        actual = reader.get(intended.id)
        assert actual is not None
        assert actual.id == intended.id
        assert actual.prompt_text == intended.prompt_text
        assert reader.get(decoy.id) is None
        assert [prompt.id for prompt in reader.get_all_prompts()] == [intended.id]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.conn.execute(
                "UPDATE system_prompts SET prompt_text = ? WHERE id = ?",
                ("forbidden native write", intended.id),
            )

        rejected = SystemPrompt(
            id="rejected-prompt",
            prompt_text="This prompt must not be persisted.",
            timestamp=3.0,
        )
        with pytest.raises(PermissionError, match="read-only"):
            reader.add(rejected)
        assert reader.get(rejected.id) is None
        assert [prompt.id for prompt in reader.get_all_prompts()] == [intended.id]
    finally:
        reader.close()
