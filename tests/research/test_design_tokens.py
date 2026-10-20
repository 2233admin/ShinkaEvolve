"""Both themes of the research workbench keep the contrast design.md promises.

Text roles need 4.5:1 on every surface they sit on; axis strokes, the signal
orange used in graphics and the darkest score-ramp step need 3:1 on the chart
backgrounds. A token edit that breaks either rule fails here, not in review.
"""

import re
from pathlib import Path

import pytest

TOKENS = (
    Path(__file__).resolve().parents[2]
    / "shinka"
    / "research"
    / "webui"
    / "app"
    / "src"
    / "design"
    / "tokens.css"
)
SURFACES = ("bg", "rail", "raised", "hover", "selected")
CHART_SURFACES = ("bg", "rail")
TEXT_ROLES = ("text", "muted", "faint", "signal")
GRAPHIC_ROLES = ("stroke", "signal-graphic", "ramp-0")


def _theme(name: str) -> dict[str, str]:
    """Hex tokens declared for one theme, merged over every block that names it."""
    css = re.sub(r"/\*.*?\*/", "", TOKENS.read_text(encoding="utf-8"), flags=re.S)
    tokens: dict[str, str] = {}
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if f"[data-theme='{name}']" not in selector:
            continue
        for key, value in re.findall(r"--([\w-]+)\s*:\s*(#[0-9a-fA-F]{6})\s*;", body):
            tokens[key] = value.lower()
    return tokens


def _luminance(hex_color: str) -> float:
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_text_roles_reach_4_5_on_every_surface(theme):
    tokens = _theme(theme)
    failures = [
        f"{role} on {surface}: {_contrast(tokens[role], tokens[surface]):.2f}"
        for role in TEXT_ROLES
        for surface in SURFACES
        if _contrast(tokens[role], tokens[surface]) < 4.5
    ]
    assert failures == []


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_graphic_roles_reach_3_on_chart_surfaces(theme):
    tokens = _theme(theme)
    failures = [
        f"{role} on {surface}: {_contrast(tokens[role], tokens[surface]):.2f}"
        for role in GRAPHIC_ROLES
        for surface in CHART_SURFACES
        if _contrast(tokens[role], tokens[surface]) < 3.0
    ]
    assert failures == []
