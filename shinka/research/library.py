"""Workspace research library, after Karpathy's "LLM knowledge base".

The human drops sources into ``raw/`` (papers, web clips, notes, links); the
researcher agent "compiles" them into an interlinked markdown wiki and files
research findings back into it. This module only keeps the books: which raw
files are new, what was compiled, and the operations log. Compiling needs an
LLM and is done by the agent following the shinka-research skill.

    <workspace>/library/
      raw/            sources as dropped by the human (links go in links.md)
      wiki/           agent-written pages: concepts, papers, findings
      index.md        catalog of wiki pages (agent-maintained)
      log.md          append-only record of ingests, compiles, lints, findings
      manifest.json   sha256 of each raw file at the time it was compiled
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from shinka.research.quant import file_sha256

LIBRARY_DIRNAME = "library"
LINKS_FILE = "links.md"
MANIFEST = "manifest.json"

INDEX_TEMPLATE = """# Library index

Catalog of wiki pages, maintained by the researcher agent. One line per page:
`- [[page-name]] (concept | paper | finding): one-sentence summary`.
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def library_root(direction_root: Path) -> Path:
    """A direction's library is shared by every direction in its workspace."""
    return Path(direction_root).resolve().parent / LIBRARY_DIRNAME


def ensure(lib: Path) -> Path:
    (lib / "raw").mkdir(parents=True, exist_ok=True)
    (lib / "wiki").mkdir(exist_ok=True)
    if not (lib / "index.md").exists():
        (lib / "index.md").write_text(INDEX_TEMPLATE)
    if not (lib / "log.md").exists():
        (lib / "log.md").write_text("# Library log\n\n")
    if not (lib / MANIFEST).exists():
        (lib / MANIFEST).write_text("{}\n")
    return lib


def log(lib: Path, kind: str, text: str) -> None:
    with (lib / "log.md").open("a") as f:
        f.write(f"- {_now()} [{kind}] {text}\n")


def add(lib: Path, source: str) -> Path:
    """Add a file (copied into raw/) or a URL (appended to raw/links.md)."""
    ensure(lib)
    raw = lib / "raw"
    if source.startswith(("http://", "https://")):
        links = raw / LINKS_FILE
        if not links.exists():
            links.write_text("# Links to ingest\n\n")
        with links.open("a") as f:
            f.write(f"- {source} (added {_now()})\n")
        log(lib, "add", source)
        return links
    src = Path(source)
    if not src.is_file():
        raise FileNotFoundError(source)
    dest = raw / src.name
    n = 1
    while dest.exists():
        dest = raw / f"{src.stem}-{n}{src.suffix}"
        n += 1
    shutil.copyfile(src, dest)
    log(lib, "add", dest.relative_to(lib).as_posix())
    return dest


def _manifest(lib: Path) -> Dict[str, str]:
    return json.loads((lib / MANIFEST).read_text())


def pending(lib: Path) -> List[str]:
    """Raw files (relative to the library) that are new or changed since
    they were last compiled."""
    if not (lib / "raw").exists():
        return []
    seen = _manifest(lib)
    out = []
    for p in sorted((lib / "raw").rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            rel = p.relative_to(lib).as_posix()
            if seen.get(rel) != file_sha256(p):
                out.append(rel)
    return out


def mark_compiled(lib: Path, rels: List[str]) -> None:
    seen = _manifest(lib)
    for rel in rels:
        path = lib / rel
        if not path.is_file():
            raise FileNotFoundError(rel)
        seen[rel] = file_sha256(path)
        log(lib, "compiled", rel)
    (lib / MANIFEST).write_text(json.dumps(seen, indent=2, sort_keys=True) + "\n")


def wiki_pages(lib: Path) -> List[str]:
    wiki = lib / "wiki"
    return sorted(p.stem for p in wiki.glob("*.md")) if wiki.exists() else []
