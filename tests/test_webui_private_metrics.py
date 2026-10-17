from typing import Any

from shinka.webui.visualization import _hide_private_metrics


def test_private_metrics_are_blanked_recursively() -> None:
    data: dict[str, Any] = {
        "programs": [
            {"id": "a", "public_metrics": {"x": 1}, "private_metrics": {"oos": 2}},
            {"id": "b", "private_metrics": {"oos": 3}},
        ],
        "private_metrics": {"oos": 4},
    }
    out = _hide_private_metrics(data)
    assert out["private_metrics"] == {}
    assert [p["private_metrics"] for p in out["programs"]] == [{}, {}]
    assert out["programs"][0]["public_metrics"] == {"x": 1}
    assert out["programs"][1]["private_metrics_hidden"] is True
    assert "private_metrics_hidden" not in _hide_private_metrics({"id": "c"})
    assert data["programs"][0]["private_metrics"] == {"oos": 2}  # input untouched
