"""plugin/loader.py — reads plugin.yaml into a Plugin dataclass."""

from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:
    raise ImportError("pip install pyyaml")


@dataclass
class Plugin:
    name: str
    mutable: list[str]       # files the agent may modify
    eval_module: str          # evaluate.py (must NOT be in mutable — Goodhart guard)
    metric: str               # key in metrics.json
    higher_is_better: bool
    plugin_dir: Path
    description: str = ""
    prepare: str | None = None
    program_md: str | None = None
    verify_cmd: str | None = None      # run before committing a keep; revert on failure
    metrics_aggregate: str = "last"    # how to aggregate across runs: last | max | mean

    def abs(self, rel: str) -> Path:
        return self.plugin_dir / rel


def load(plugin_dir: str | Path) -> Plugin:
    plugin_dir = Path(plugin_dir).resolve()
    cfg_path = plugin_dir / "plugin.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"plugin.yaml not found in {plugin_dir}")

    with open(cfg_path) as f:
        d = yaml.safe_load(f)

    mutable = d["mutable"]
    if isinstance(mutable, str):
        mutable = [mutable]

    eval_module = d.get("eval_module", "evaluate.py")

    metrics_aggregate = d.get("metrics_aggregate", "last")
    if metrics_aggregate not in ("last", "max", "mean"):
        raise ValueError(f"metrics_aggregate must be last|max|mean, got '{metrics_aggregate}'")

    # Goodhart guard: eval must never be mutable
    for m in mutable:
        if m == eval_module:
            raise ValueError(
                f"eval_module '{eval_module}' must not be in mutable files. "
                "An agent that rewrites its own evaluation is Goodhart's Law in action."
            )

    return Plugin(
        name=d["name"],
        description=d.get("description", ""),
        mutable=mutable,
        eval_module=eval_module,
        metric=d.get("metric", "combined_score"),
        higher_is_better=d.get("higher_is_better", True),
        prepare=d.get("prepare"),
        program_md=d.get("program_md", "program.md"),
        verify_cmd=d.get("verify_cmd"),
        metrics_aggregate=metrics_aggregate,
        plugin_dir=plugin_dir,
    )
