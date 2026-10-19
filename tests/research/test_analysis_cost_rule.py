"""The restored analysis pages price a program by the rule of ``rundb.run_cost``,
so a run never shows one cost in the run list and another in its analysis."""

import json
import re
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import pytest

from shinka.research import rundb

ANALYSIS = (
    Path(__file__).resolve().parents[2] / "shinka" / "research" / "webui" / "analysis"
)
# Each page carries its own helpers; the last name is the per-program price.
PAGES = {
    "viz_tree.html": (
        "recordedNumber",
        "sumRecorded",
        "recordedCost",
        "costField",
        "programCost",
    ),
    "compare.html": ("recordedCost", "getProgramCost"),
}
CASES = [
    {},
    {"api_costs": 0.2, "embed_cost": 0.01, "novelty_cost": 0.03},
    {"api_costs": 0.2, "embed_cost": 0.01, "novelty_cost": 0.03, "meta_cost": 0.04},
    {"api_costs": None, "embed_cost": 0.5},
    {"api_costs": "n/a"},
    {"api_costs": "1.5"},
    {"api_costs": -0.1},
    {
        "model_name": "headless/agent",
        "headless_pricing_unknown": True,
        "api_costs": 12.0,
        "embed_cost": 0.25,
    },
    {"model_name": "demo/model", "headless_pricing_unknown": True, "api_costs": 0.5},
    {"_is_island_copy": True, "api_costs": 99.0},
    {"_spawned_island": True, "api_costs": "n/a"},
]


def _function(source: str, name: str) -> str:
    match = re.search(rf"function {name}\(", source)
    assert match, f"{name} is missing"
    depth = 0
    for index in range(source.index("{", match.start()), len(source)):
        depth += {"{": 1, "}": -1}.get(source[index], 0)
        if depth == 0:
            return source[match.start() : index + 1]
    raise AssertionError(f"{name} has no closing brace")


def _run(page: str, names: tuple, *lines: str):
    """Evaluate the page's own functions in node and return the printed JSON."""
    source = (ANALYSIS / page).read_text(encoding="utf-8")
    script = "\n".join([*(_function(source, name) for name in names), *lines])
    completed = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True
    )
    return json.loads(completed.stdout)


def _page_prices(page: str) -> list:
    names = PAGES[page]
    return _run(
        page,
        names,
        f"const cases = {json.dumps(CASES)};",
        f"console.log(JSON.stringify(cases.map(m => {names[-1]}({{ metadata: m }}))));",
    )


def _server_price(results: Path, metadata: dict) -> float | None:
    results.mkdir()
    with closing(sqlite3.connect(results / rundb.DB_FILENAME)) as conn:
        conn.execute("CREATE TABLE programs (id TEXT, metadata TEXT)")
        conn.execute("INSERT INTO programs VALUES ('p', ?)", (json.dumps(metadata),))
        conn.commit()
    cost = rundb.run_cost(results)
    return cost.programs_usd if cost.programs_known else None


@pytest.mark.parametrize("page", sorted(PAGES))
def test_analysis_pages_price_programs_like_run_cost(tmp_path: Path, page: str) -> None:
    expected = [_server_price(tmp_path / str(i), m) for i, m in enumerate(CASES)]
    for metadata, got, want in zip(CASES, _page_prices(page), expected):
        assert (got is None) == (want is None), metadata
        if want is not None:
            assert got == pytest.approx(want), metadata


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        # The server answers [] for a run without prompts.sqlite: a recorded zero.
        ({"metaPromptData": None, "metaPromptFailed": False}, 0.0),
        (
            {
                "metaPromptData": [
                    {"metadata": {"llm": {"cost": 0.25}}},
                    {"metadata": None},
                ],
                "metaPromptFailed": False,
            },
            0.25,
        ),
        (
            {
                "metaPromptData": [{"metadata": {"llm": {"cost": "n/a"}}}],
                "metaPromptFailed": False,
            },
            None,
        ),
        # A failed read is unknown, although no prompt is loaded.
        ({"metaPromptData": None, "metaPromptFailed": True}, None),
    ],
)
def test_analysis_prompt_total_stays_unknown_after_a_failed_read(
    window: dict, expected: float | None
) -> None:
    names = ("recordedNumber", "sumRecorded", "recordedCost", "promptCostTotal")
    got = _run(
        "viz_tree.html",
        names,
        f"globalThis.window = {json.dumps(window)};",
        "console.log(JSON.stringify(promptCostTotal()));",
    )
    assert got == (None if expected is None else pytest.approx(expected))
