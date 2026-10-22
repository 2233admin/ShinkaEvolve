import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "quant_research"


@pytest.fixture(autouse=True)
def _outside_the_agent_environment():
    """Tests start as the owner's manual commands, whatever shell runs them;
    those about the agent environment set the marker themselves. Its own
    patcher, so a test's `monkeypatch.undo()` does not bring the marker back."""
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("SHINKA_RESEARCH_AGENT", raising=False)
        yield


def _load_example_module(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bars() -> pd.DataFrame:
    return _load_example_module("make_data").make_bars(n=800, seed=1)


@pytest.fixture
def bars_csv(tmp_path: Path, bars: pd.DataFrame) -> Path:
    path = tmp_path / "bars.csv"
    bars.to_csv(path)
    return path


@pytest.fixture
def example_dir() -> Path:
    return EXAMPLE
