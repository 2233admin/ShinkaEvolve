"""Workspace research library, after Karpathy's "LLM knowledge base".

The human and the researcher agent add sources to ``raw/`` (papers, web clips,
notes, links); the agent "compiles" them into an interlinked markdown wiki and
files research findings back into it. This module only keeps the books: which
raw files are new, what was compiled, who added what, and the operations log.
Compiling needs an LLM and is done by the agent following the shinka-research
skill.

    <workspace>/library/
      raw/            sources (links go in links.md)
      wiki/           agent-written pages: concepts, papers, findings
      index.md        catalog of wiki pages (agent-maintained)
      log.md          append-only record of adds, compiles, lints, findings
      manifest.json   sha256 of each raw file at the time it was compiled
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from shinka.research.quant import file_sha256

LIBRARY_DIRNAME = "library"
LINKS_FILE = "links.md"
MANIFEST = "manifest.json"
# Who added a source: the caller tells, from SHINKA_RESEARCH_AGENT. The agent
# can unset that flag, so the origin is a cooperation marker, not proof.
ORIGINS = ("agent", "human")
# A log line written by `add`; logs from before origins were recorded say [add].
_ADD_LINE = re.compile(r"^- (\S+) \[add(?: (agent|human))?\] (.+)$")
_URL = ("http://", "https://")

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


def add(lib: Path, source: str, by: str) -> Path:
    """Add a file (copied into raw/) or a URL (appended to raw/links.md); the
    log line records ``by`` (``agent`` or ``human``) as ``[add <by>]``."""
    if by not in ORIGINS:
        raise ValueError(f"origin must be one of {ORIGINS}, not {by!r}")
    ensure(lib)
    raw = lib / "raw"
    if source.startswith(_URL):
        links = raw / LINKS_FILE
        if not links.exists():
            links.write_text("# Links to ingest\n\n")
        with links.open("a") as f:
            f.write(f"- {source} (added {_now()})\n")
        log(lib, f"add {by}", source)
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
    log(lib, f"add {by}", dest.relative_to(lib).as_posix())
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


@dataclass(frozen=True)
class Sources:
    """Who added the library's sources, as far as the log records it."""

    agent: int
    human: int
    # [add] lines from before origins were recorded, and raw files and links.md
    # URLs that no add line names (put into raw/ by hand).
    unrecorded: int
    last_added: Optional[str]


def sources(lib: Path) -> Sources:
    counts: Dict[Optional[str], int] = {"agent": 0, "human": 0, None: 0}
    named = set()
    last: Optional[str] = None
    log_file = lib / "log.md"
    lines = log_file.read_text().splitlines() if log_file.exists() else []
    for line in lines:
        m = _ADD_LINE.match(line)
        if m:
            when, by, what = m.groups()
            counts[by] += 1
            named.add(what)
            last = when
    raw = lib / "raw"
    links = raw / LINKS_FILE
    dropped = [
        p
        for p in (raw.rglob("*") if raw.exists() else [])
        if p.is_file()
        and not p.name.startswith(".")
        and p != links
        and p.relative_to(lib).as_posix() not in named
    ]
    hand_links = set()
    for line in links.read_text().splitlines() if links.is_file() else []:
        parts = line.split()
        if len(parts) > 1 and parts[0] == "-" and parts[1].startswith(_URL):
            hand_links.add(parts[1])
    unrecorded = counts[None] + len(dropped) + len(hand_links - named)
    return Sources(counts["agent"], counts["human"], unrecorded, last)


def _page_type(text: str) -> Optional[str]:
    """The page's ``type``: from the front matter (templates/wiki_page.md), or
    for a page without one, its first standalone ``Type: <kind>`` line."""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                return None
            key, _, value = line.partition(":")
            if key.strip().lower() == "type":
                return value.split("#")[0].strip().lower()
        return None
    for line in lines:
        key, _, value = line.partition(":")
        if key.strip().lower() == "type" and value.strip():
            return value.split("#")[0].strip().lower()
    return None


def finding_pages(lib: Path) -> List[str]:
    """Wiki pages whose type is ``finding``."""
    return [
        name
        for name in wiki_pages(lib)
        if _page_type((lib / "wiki" / f"{name}.md").read_text()) == "finding"
    ]
