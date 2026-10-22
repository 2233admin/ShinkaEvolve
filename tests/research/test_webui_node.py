"""The workbench's own node tests (URL routes, restored-page actions) in the research gate."""

import subprocess
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "shinka" / "research" / "webui" / "app"


def test_workbench_node_tests_pass() -> None:
    completed = subprocess.run(
        ["node", "--test"], cwd=APP, capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
