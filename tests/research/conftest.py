import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "quant_research"


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
