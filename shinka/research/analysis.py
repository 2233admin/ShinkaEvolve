"""Reuse the original analysis UI without widening its read authority."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.parse
from contextlib import closing
from pathlib import Path

from shinka.database.prompt_dbase import SystemPromptConfig, SystemPromptDatabase
from shinka.webui import visualization

from . import rundb


def _recorded_cost(db: Path, table: str, fields: tuple[str, ...]) -> float | None:
    """Unknown pricing is not evidence of a free run."""
    valid = " AND ".join(
        f"json_type(metadata, '$.{field}') IN ('integer', 'real')"
        f" AND json_extract(metadata, '$.{field}') >= 0"
        for field in fields
    )
    amount = " + ".join(f"json_extract(metadata, '$.{field}')" for field in fields)
    uri = f"file:{urllib.parse.quote(str(db), safe='')}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        row = conn.execute(
            f"SELECT CASE WHEN COUNT(*) = COUNT(cost) THEN COALESCE(SUM(cost), 0) END "
            f"FROM (SELECT CASE WHEN json_valid(metadata) THEN CASE WHEN {valid} "
            f"THEN {amount} END END AS cost FROM {table})"
        ).fetchone()
    return row[0]


WEBUI = Path(__file__).parent / "webui" / "dist" / "analysis"
ANALYSIS_PATHS = {
    "/analysis/index.html",
    "/viz_tree.html",
    "/compare.html",
    "/list_databases",
    "/get_programs",
    "/get_programs_summary",
    "/get_program_count",
    "/get_program_details",
    "/get_meta_files",
    "/get_meta_content",
    "/download_meta_pdf",
    "/get_plots",
    "/get_system_prompts",
    "/get_database_stats",
}


def confined(authority: Path, path: Path) -> Path:
    target = path.resolve()
    try:
        target.relative_to(authority.resolve())
    except ValueError:
        raise PermissionError("Outside analysis read scope") from None
    return target


class AnalysisHandler(visualization.DatabaseRequestHandler):
    """Discovery is workspace-wide; artifacts remain confined to their run.

    The same adapter also works with a narrower experiments root. Private
    metrics are never opt-in here, including for the ordinary-run browser.
    """

    def __init__(self, *args, search_root=None, runs_root=None, **kwargs):
        workspace = Path(search_root or Path.cwd()).resolve()
        roots = [workspace]
        if runs_root is not None:
            roots.append(Path(runs_root).resolve())
        self.read_roots = tuple(dict.fromkeys(roots))
        kwargs["show_private"] = False
        super().__init__(*args, search_root=str(workspace), **kwargs)

    def _allowed(self, path: Path) -> Path:
        target = path.resolve()
        if not any(target.is_relative_to(root) for root in self.read_roots):
            raise PermissionError("Outside analysis read scope")
        return target

    def _requested_path(self, value: str) -> Path:
        for prefix, root in zip(("workspace/", "runs/"), self.read_roots):
            if value.startswith(prefix):
                return confined(root, root / value[len(prefix) :])
        raise PermissionError("Specify a namespaced analysis path")

    def _client_path(self, path: Path) -> str:
        for prefix, root in zip(("workspace", "runs"), self.read_roots):
            if path.is_relative_to(root):
                return f"{prefix}/{path.relative_to(root).as_posix()}"
        raise PermissionError("Outside analysis read scope")

    def _is_program_database(self, path: Path) -> bool:
        try:
            db = self._allowed(path)
            if not db.is_file() or db.suffix.lower() not in {".db", ".sqlite"}:
                return False
            root = next(root for root in self.read_roots if db.is_relative_to(root))
            parts = db.relative_to(root).parts
            if any(
                part.casefold() in {"library", "raw", "lockbox"} for part in parts[:-1]
            ):
                return False
            if (
                len(parts) > 2
                and parts[1] == "data"
                and (root / parts[0] / "notebook.json").is_file()
            ):
                return False
            uri = f"file:{urllib.parse.quote(str(db), safe='')}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(programs)")
                }
            return {"id", "code", "generation", "correct", "combined_score"} <= columns
        except (OSError, sqlite3.Error):
            return False

    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(url.query)
        original_path = self.path
        original_directory = self.directory
        original_root = self.search_root
        try:
            if len(query.get("db_path", [])) > 1:
                self.send_error(400, "Specify exactly one database")
                return
            for path in query.get("db_path", []):
                db = self._requested_path(path)
                self._get_actual_db_path(str(db))
                if url.path in {
                    "/get_meta_files",
                    "/get_meta_content",
                    "/download_meta_pdf",
                }:
                    meta = confined(db.parent, db.parent / "meta")
                    if url.path != "/get_meta_files":
                        for count in query.get(
                            "processed_count", query.get("generation", [])
                        ):
                            if not count.isdecimal():
                                raise PermissionError("Invalid meta generation")
                            target = confined(db.parent, meta / f"meta_{count}.txt")
                            if not target.exists():
                                confined(db.parent, db.parent / f"meta_{count}.txt")
                elif url.path == "/get_plots":
                    for generation in query.get("generation", []):
                        if not generation.isdecimal():
                            raise PermissionError("Invalid plot generation")
                        plots = confined(
                            db.parent,
                            db.parent / f"gen_{generation}" / "results" / "plots",
                        )
                        if plots.is_dir():
                            for child in plots.iterdir():
                                confined(plots, child)
                # Upstream caches by its incoming db_path, so dispatch with the
                # canonical path only after validating the public namespace.
                query["db_path"] = [str(db)]
                self.path = url.path + "?" + urllib.parse.urlencode(query, doseq=True)
            if url.path.startswith("/plot_file/"):
                target = self._requested_path(
                    urllib.parse.unquote(url.path[len("/plot_file/") :])
                )
                if (
                    target.suffix.lower() not in {".png", ".gif", ".jpg", ".jpeg"}
                    or target.parent.name != "plots"
                    or target.parent.parent.name != "results"
                    or not re.fullmatch(r"gen_\d+", target.parents[2].name)
                    or not any(
                        self._is_program_database(p)
                        for p in target.parents[3].iterdir()
                    )
                ):
                    raise PermissionError("Not an analysis plot")
                self.search_root = str(
                    next(
                        root for root in self.read_roots if target.is_relative_to(root)
                    )
                )
                self.path = "/plot_file/" + urllib.parse.quote(
                    target.as_posix(), safe="/"
                )
            self.directory = str(WEBUI)
            if url.path == "/analysis/index.html":
                self.path = "/index.html" + (f"?{url.query}" if url.query else "")
            super().do_GET()
        except PermissionError:
            self.send_error(403, "Outside analysis read scope")
        finally:
            self.path = original_path
            self.directory = original_directory
            self.search_root = original_root

    def _get_actual_db_path(self, db_path: str) -> str:
        db = self._allowed(Path(db_path))
        if not self._is_program_database(db):
            raise PermissionError("Not a readable Shinka program database")
        confined(db.parent, db.with_name(rundb.PROMPT_DB_FILENAME))
        self.run_scope = db.parent
        self.search_root = str(
            next(root for root in self.read_roots if db.is_relative_to(root))
        )
        return str(db)

    def handle_list_databases(self):
        rows = []
        seen = set()
        for scope in self.read_roots:
            for directory, names, files in os.walk(scope):
                parent = Path(directory)
                names[:] = [
                    name
                    for name in names
                    if (parent / name).resolve() == (parent / name).absolute()
                ]
                for name in files:
                    if name.lower() in {"prompts.db", "prompts.sqlite"} or Path(
                        name
                    ).suffix.lower() not in {".db", ".sqlite"}:
                        continue
                    path = parent / name
                    try:
                        canonical = confined(scope, path)
                    except PermissionError:
                        continue
                    if canonical in seen or not self._is_program_database(canonical):
                        continue
                    seen.add(canonical)
                    relative = path.relative_to(scope)
                    client_path = self._client_path(canonical)
                    match = re.search(r"_(\d{8}_\d{6})", relative.as_posix())
                    rows.append(
                        {
                            "path": client_path,
                            "actual_path": client_path,
                            "name": relative.parent.as_posix(),
                            "task": relative.parts[0]
                            if len(relative.parts) > 1
                            else scope.name,
                            "_date": match.group(1) if match else "0",
                        }
                    )
        rows.sort(key=lambda row: row["_date"], reverse=True)
        for row in rows:
            del row["_date"]
        self.send_json_response(rows)

    def _send_read_error(self, message: str) -> None:
        payload = json.dumps({"error": message}).encode("utf-8")
        self.send_response(500)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def handle_get_system_prompts(self, db_path: str):
        db = Path(self._get_actual_db_path(db_path))
        prompt_db = confined(db.parent, db.with_name(rundb.PROMPT_DB_FILENAME))
        if not prompt_db.exists():
            self.send_json_response([])
            return
        try:
            # Validate before the library constructor, which otherwise cannot
            # close its partially initialized connection on a schema failure.
            uri = f"file:{urllib.parse.quote(str(prompt_db), safe='')}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                for table in (
                    "system_prompts",
                    "prompt_archive",
                    "prompt_metadata_store",
                ):
                    connection.execute(f"SELECT * FROM {table} LIMIT 0")
            config = SystemPromptConfig(db_path=str(prompt_db))
            with closing(SystemPromptDatabase(config, read_only=True)) as database:
                prompts = [prompt.to_dict() for prompt in database.get_all_prompts()]
        except (OSError, sqlite3.Error):
            self._send_read_error("Unable to read system prompts")
            return
        self.send_json_response(prompts)

    def handle_get_database_stats(self, db_path: str):
        db = Path(self._get_actual_db_path(db_path))
        prompt_db = confined(db.parent, db.with_name(rundb.PROMPT_DB_FILENAME))
        try:
            self._stats_costs = (
                _recorded_cost(
                    db,
                    "programs",
                    ("api_costs", "embed_cost", "novelty_cost", "meta_cost"),
                ),
                _recorded_cost(prompt_db, "system_prompts", ("llm.cost",))
                if prompt_db.exists()
                else 0.0,
            )
        except (OSError, sqlite3.Error):
            self._send_read_error("Unable to read run statistics")
            return
        try:
            return super().handle_get_database_stats(db_path)
        finally:
            self._stats_costs = None

    def _generate_pdf(self, content: str, generation: str) -> bytes:
        pdf = super()._generate_pdf(content, generation)
        if pdf is None:
            raise RuntimeError(
                "PDF export requires wkhtmltopdf or pandoc with a PDF engine"
            )
        return pdf

    def send_json_response(self, data):
        costs = getattr(self, "_stats_costs", None)
        if costs is not None and "total_cost" in data:
            data["total_cost"], data["prompt_evo_cost"] = costs
        return super().send_json_response(data)

    def handle_get_plots(self, db_path: str, generation: str, program_id: str):
        db = Path(self._get_actual_db_path(db_path))
        plots = confined(
            db.parent, db.parent / f"gen_{generation}" / "results" / "plots"
        )
        rows = []
        if plots.is_dir():
            for path in sorted(plots.iterdir()):
                target = confined(plots, path)
                suffix = target.suffix.lower()
                if target.is_file() and suffix in {".png", ".gif", ".jpg", ".jpeg"}:
                    rows.append(
                        {
                            "filename": path.name,
                            "path": self._client_path(target),
                            "type": "animation" if suffix == ".gif" else "image",
                            "ext": suffix,
                        }
                    )
        self.send_json_response(rows)

    def _read_failure_json(self, failure_json_path):
        if failure_json_path:
            scope = getattr(self, "run_scope", Path(self.search_root))
            confined(scope, (Path(self.search_root) / failure_json_path).parent)
            confined(scope, Path(self.search_root) / failure_json_path)
        return super()._read_failure_json(failure_json_path)

    def _resolve_failed_node_code_path(self, details, failure_payload):
        path = super()._resolve_failed_node_code_path(details, failure_payload)
        scope = getattr(self, "run_scope", Path(self.search_root))
        return confined(scope, path) if path is not None else None

    def send_header(self, keyword: str, value: str) -> None:
        if keyword.lower() != "access-control-allow-origin":
            super().send_header(keyword, value)
