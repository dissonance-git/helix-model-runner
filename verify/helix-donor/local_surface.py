"""Helix-owned local MCP operations.

Commons owns the cross-project first-connection packet. This module exposes only
Helix-local operations over existing owners: capabilities, skills, Git state,
bounded repository access, validation routes, execution, and HTTP. It does not
mint a Commons bootstrap schema or first-connection sequence.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from engine.api.capabilities import CAPABILITIES
from engine.paths import ROOT
from engine.query.evidence.repositories import inspect_github_donor
from engine.query.mcp.local_code import LocalCodeAdapter
from engine.repository_transport import git_blob_sha


VERSION = "mcp-local-surface-002.0"
SHORT = "helix"
DESCRIPTION = "Helix-local capabilities, repository access, validation, and execution"
TAIL_CHARS = 12000
MAX_HTTP_BYTES = 2_000_000
ALLOWED_PROGRAMS = ("python", "git")
HTTP_PREFIXES = ("https://", "http://127.0.0.1", "http://localhost")
SKILLS_DIR = ROOT / ".agents" / "skills"
ROUTES: dict[str, list[str]] = {
    "tests": ["python", "-B", "-m", "pytest", "engine/tests", "-q"],
    "mcp-tests": ["python", "-B", "-m", "pytest", "engine/tests", "-k", "mcp", "-q"],
    "validate": ["python", "-B", "-m", "engine", "validate", "architecture"],
}


class LocalSurfaceError(ValueError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str, timeout: int = 30) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return completed.returncode, (completed.stdout or "").strip()


def git_state() -> dict[str, Any]:
    head_code, head = _git("rev-parse", "HEAD")
    branch_code, branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status_code, status = _git("status", "--porcelain")
    dirty = [line for line in status.splitlines() if line.strip()]
    ahead = behind = None
    counts_code, counts = _git("rev-list", "--left-right", "--count", "origin/main...HEAD")
    if counts_code == 0 and counts:
        parts = counts.split()
        if len(parts) == 2:
            behind, ahead = int(parts[0]), int(parts[1])
    remotes_code, remotes = _git("remote", "-v")
    return {
        "schema_version": VERSION,
        "head": head if head_code == 0 else None,
        "branch": branch if branch_code == 0 else None,
        "dirty_paths": len(dirty),
        "dirty_sample": dirty[:20],
        "ahead_of_origin_main": ahead,
        "behind_origin_main": behind,
        "merge_check_exit": counts_code,
        "status": status if status_code == 0 else "",
        "upstream": "origin/main",
        "remotes": remotes if remotes_code == 0 else "",
        "ahead_behind": f"{behind}\t{ahead}" if ahead is not None and behind is not None else "",
    }


def skills() -> list[str]:
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(
        child.name
        for child in SKILLS_DIR.iterdir()
        if child.is_dir() and (child / "SKILL.md").is_file()
    )


def skill(name: str | None = None) -> dict[str, Any]:
    available = skills()
    requested = str(name or "").strip()
    if not requested:
        return {"schema_version": VERSION, "skills": available, "count": len(available)}
    if requested not in available:
        raise LocalSurfaceError(f"unknown skill: {requested}")
    path = SKILLS_DIR / requested / "SKILL.md"
    data = path.read_bytes()
    return {
        "schema_version": VERSION,
        "name": requested,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(data),
        "bytes": len(data),
        "text": data.decode("utf-8", errors="replace"),
    }


def capabilities(
    identifier: str | None = None,
    *,
    operation: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    op = str(operation or "").strip().casefold()
    if op == "find":
        needle = str(query or "").casefold()
        hits = [name for name in sorted(CAPABILITIES) if needle in name.casefold()]
        return {"schema_version": VERSION, "operation": op, "query": query, "count": len(hits), "capabilities": hits}
    if op == "ready":
        return {"schema_version": VERSION, "operation": op, "count": len(CAPABILITIES), "capabilities": sorted(CAPABILITIES)}
    if op == "self-test":
        return {"schema_version": VERSION, "operation": op, "status": "ok" if CAPABILITIES else "empty", "count": len(CAPABILITIES)}
    requested = str(identifier or "").strip()
    if op == "read" or (not op and requested):
        entry = CAPABILITIES.get(requested)
        if entry is None:
            raise LocalSurfaceError(f"unknown capability: {requested}")
        payload: dict[str, Any] = {"schema_version": VERSION, "capability": entry}
        if op:
            payload.update(operation=op, identifier=requested)
            if isinstance(entry, dict):
                payload.update(entry)
        return payload
    if op:
        raise LocalSurfaceError(f"unknown capability operation: {op}")
    return {"schema_version": VERSION, "count": len(CAPABILITIES), "capabilities": sorted(CAPABILITIES)}


def project_identity() -> dict[str, Any]:
    """Return Helix-local identity only; Commons owns first connection."""
    state = git_state()
    tree_code, tree = _git("rev-parse", "HEAD^{tree}")
    return {
        "schema_version": VERSION,
        "project": "helix",
        "server": SHORT,
        "description": DESCRIPTION,
        "root": str(ROOT),
        "head": state["head"],
        "tree_sha": tree if tree_code == 0 else None,
        "branch": state["branch"],
        "dirty_paths": state["dirty_paths"],
        "routes": {name: " ".join(argv) for name, argv in ROUTES.items()},
        "capability_count": len(CAPABILITIES),
        "skills": skills(),
        "entry_owner": "helix-commons:projects_room_bootstrap",
        "project_entry": "describe_helix",
        "derived_projection": True,
        "canonical_truth": False,
    }


def read(
    adapter: LocalCodeAdapter,
    path: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    max_characters: int = 16000,
) -> dict[str, Any]:
    payload = adapter.read(path, start_line=start_line, end_line=end_line, max_characters=int(max_characters))
    target = (ROOT / str(path)).resolve()
    try:
        if target.is_file() and target.stat().st_size <= 5_000_000:
            data = target.read_bytes()
            payload = {**payload, "sha256": _sha256(data), "bytes": len(data)}
    except OSError:
        pass
    code, blob = _git("rev-parse", f"HEAD:{Path(str(path)).as_posix()}")
    head_blob = blob if code == 0 and blob else payload.get("head_blob_sha") or payload.get("blob_sha")
    worktree_blob = payload.get("blob_sha") or payload.get("sha256")
    payload.setdefault("head_blob_sha", payload.get("blob_sha"))
    payload.setdefault("head_blob", head_blob)
    payload.setdefault("worktree_blob", worktree_blob)
    payload.setdefault("matches_head", bool(head_blob) and head_blob == worktree_blob)
    payload.setdefault("text", payload.get("content", ""))
    return payload


def search(
    adapter: LocalCodeAdapter,
    query: str,
    *,
    roots: list[str] | None = None,
    suffixes: list[str] | None = None,
    max_matches: int = 80,
) -> dict[str, Any]:
    limit = int(max_matches)
    payload = adapter.find_text(query, roots=roots, suffixes=suffixes, max_matches=limit)
    matches = payload.get("matches") or []
    payload.setdefault("query", query)
    payload.setdefault("count", len(matches))
    payload.setdefault("results", matches)
    payload.setdefault("truncated", bool(payload.get("truncated")) or len(matches) >= limit)
    return payload


def donor_repository(
    repository: str,
    *,
    question: str,
    lane: str = "direct",
    refresh: bool = False,
    maximum_files: int = 12,
    focus_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Inspect one public GitHub repository as bounded Helix donor evidence.

    Git is the hydration transport. The cached checkout becomes rate-less local
    evidence after hydration; donor semantics remain owned by
    engine.query.evidence.repositories.
    """
    return inspect_github_donor(
        repository,
        question=question,
        lane=lane,
        refresh=refresh,
        maximum_files=maximum_files,
        focus_paths=focus_paths,
    )


def write(
    adapter: LocalCodeAdapter,
    *,
    path: str,
    operation: str | None = None,
    expected_head: str | None = None,
    expected_blob_sha: str | None = None,
    content: str | None = None,
    unified_diff: str | None = None,
    new_path: str | None = None,
    dry_run: bool = True,
    allow_protected: bool = False,
) -> dict[str, Any]:
    op = str(operation or "").strip().casefold()
    if not op:
        op = "update" if (ROOT / str(path)).resolve().exists() else "create"
    payload = adapter.apply_change(
        operation=op,
        path=path,
        expected_head=str(expected_head or "").strip() or adapter.current_head(),
        expected_blob_sha=expected_blob_sha,
        content=content,
        unified_diff=unified_diff,
        new_path=new_path,
        dry_run=dry_run,
        allow_protected=allow_protected,
    )
    if content is not None:
        encoded = str(content).encode("utf-8")
        payload.setdefault("path", str(path))
        payload.setdefault("bytes", len(encoded))
        payload.setdefault("sha256", _sha256(encoded))
    return payload


def edit(
    adapter: LocalCodeAdapter,
    *,
    path: str,
    find: str,
    replace: str,
    all: bool = False,
    expected_blob_sha: str | None = None,
    expected_head: str | None = None,
    dry_run: bool = False,
    allow_protected: bool = False,
) -> dict[str, Any]:
    target = (ROOT / str(path)).resolve()
    try:
        target.relative_to(ROOT)
    except ValueError as exc:
        raise LocalSurfaceError("path escapes the repository root") from exc
    if not target.is_file():
        raise LocalSurfaceError("repository file does not exist")
    data = target.read_bytes()
    blob_sha = git_blob_sha(data)
    sha256 = _sha256(data)
    if expected_blob_sha and str(expected_blob_sha).strip().lower() != blob_sha:
        raise LocalSurfaceError(f"blob mismatch: current={blob_sha}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LocalSurfaceError("file is not utf-8 text") from exc
    needle = str(find)
    if not needle:
        raise LocalSurfaceError("find must be non-empty")
    occurrences = text.count(needle)
    if occurrences == 0:
        raise LocalSurfaceError("anchor not found")
    if occurrences > 1 and not bool(all):
        raise LocalSurfaceError(f"anchor is ambiguous ({occurrences} matches); pass all=True or a longer anchor")
    updated = text.replace(needle, str(replace)) if all else text.replace(needle, str(replace), 1)
    encoded = updated.encode("utf-8")
    payload = adapter.apply_change(
        operation="update",
        path=str(path),
        expected_head=str(expected_head or "").strip() or adapter.current_head(),
        expected_blob_sha=blob_sha,
        content=updated,
        dry_run=bool(dry_run),
        allow_protected=allow_protected,
    )
    payload.setdefault("path", str(path))
    payload.setdefault("replaced", occurrences if all else 1)
    payload.setdefault("previous_blob_sha", blob_sha)
    payload.setdefault("previous_sha256", sha256)
    payload.setdefault("sha256", _sha256(encoded))
    payload.setdefault("bytes", len(encoded))
    return payload


def routes(*, include_readiness: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": VERSION,
        "project": "helix",
        "root_exists": ROOT.is_dir(),
        "routes": {name: " ".join(argv) for name, argv in ROUTES.items()},
        "tool_readiness": {argv[0]: shutil.which(argv[0]) is not None for argv in ROUTES.values()},
    }
    if include_readiness:
        payload["readiness"] = {
            name: {"program": argv[0], "available": shutil.which(argv[0]) is not None}
            for name, argv in ROUTES.items()
        }
    return payload


def run(program: str, *, arguments: list[str] | None = None, timeout_seconds: int = 300) -> dict[str, Any]:
    name = Path(str(program or "").strip()).name
    if name not in ALLOWED_PROGRAMS:
        raise LocalSurfaceError(
            "program is not allowlisted: " + (name or "(empty)") + " (allowed: " + ", ".join(ALLOWED_PROGRAMS) + ")"
        )
    resolved_timeout = int(timeout_seconds)
    if isinstance(timeout_seconds, bool) or not 1 <= resolved_timeout <= 900:
        raise LocalSurfaceError("timeout_seconds must be between 1 and 900")
    argv = [name, *[str(item) for item in (arguments or [])]]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=resolved_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LocalSurfaceError(f"command exceeded {resolved_timeout}s") from exc
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return {
        "schema_version": VERSION,
        "program": name,
        "arguments": argv[1:],
        "command": " ".join(argv),
        "timeout_seconds": resolved_timeout,
        "seconds": round(time.monotonic() - started, 2),
        "exit_code": completed.returncode,
        "stdout_tail": stdout[-TAIL_CHARS:],
        "stderr_tail": stderr[-TAIL_CHARS:],
        "output_truncated": len(stdout) > TAIL_CHARS or len(stderr) > TAIL_CHARS,
    }


def http(url: str, *, method: str = "GET", timeout_seconds: int = 30) -> dict[str, Any]:
    target = str(url or "").strip()
    if not target.startswith(HTTP_PREFIXES):
        raise LocalSurfaceError("url must be https:// or loopback http://")
    verb = str(method or "GET").strip().upper()
    if verb not in {"GET", "HEAD"}:
        raise LocalSurfaceError("method must be GET or HEAD")
    request = urllib.request.Request(target, method=verb)
    with urllib.request.urlopen(request, timeout=int(timeout_seconds)) as response:
        body = b"" if verb == "HEAD" else response.read(MAX_HTTP_BYTES)
        text = body.decode("utf-8", errors="replace")
        return {
            "schema_version": VERSION,
            "url": target,
            "method": verb,
            "status": response.status,
            "content_type": response.headers.get("Content-Type", ""),
            "sha256": _sha256(body),
            "bytes": len(body),
            "text_tail": text[-TAIL_CHARS:],
            "body": text,
            "truncated": len(body) >= MAX_HTTP_BYTES,
        }


def register(
    server,
    *,
    adapter: LocalCodeAdapter | None = None,
    include_capability_resource: bool = True,
    include_root_resource: bool = True,
    skip_tools: frozenset[str] = frozenset(),
) -> None:
    """Attach Helix-local operations. Commons owns project-room bootstrap."""
    code = adapter or LocalCodeAdapter()

    @server.tool()
    def helix_capabilities(identifier: str | None = None, operation: str | None = None, query: str | None = None) -> dict[str, Any]:
        return capabilities(identifier, operation=operation, query=query)

    @server.tool()
    def helix_skill(name: str | None = None) -> dict[str, Any]:
        return skill(name)

    @server.tool()
    def helix_git_state() -> dict[str, Any]:
        return git_state()

    if "helix_read" not in skip_tools:
        @server.tool()
        def helix_read(path: str, start_line: int | None = None, end_line: int | None = None, max_characters: int = 16000) -> dict[str, Any]:
            return read(code, path, start_line=start_line, end_line=end_line, max_characters=max_characters)

    @server.tool()
    def helix_search(query: str, roots: list[str] | None = None, suffixes: list[str] | None = None, max_matches: int = 80) -> dict[str, Any]:
        return search(code, query, roots=roots, suffixes=suffixes, max_matches=max_matches)

    @server.tool()
    def helix_donor_repository(
        repository: str,
        question: str,
        lane: str = "direct",
        refresh: bool = False,
        maximum_files: int = 12,
        focus_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Hydrate/cache one public GitHub repo and return bounded donor-pass evidence."""
        return donor_repository(
            repository,
            question=question,
            lane=lane,
            refresh=refresh,
            maximum_files=maximum_files,
            focus_paths=focus_paths,
        )

    @server.tool()
    def helix_write(
        path: str,
        operation: str | None = None,
        expected_head: str | None = None,
        expected_blob_sha: str | None = None,
        content: str | None = None,
        unified_diff: str | None = None,
        new_path: str | None = None,
        dry_run: bool = True,
        allow_protected: bool = False,
    ) -> dict[str, Any]:
        return write(
            code,
            path=path,
            operation=operation,
            expected_head=expected_head,
            expected_blob_sha=expected_blob_sha,
            content=content,
            unified_diff=unified_diff,
            new_path=new_path,
            dry_run=dry_run,
            allow_protected=allow_protected,
        )

    @server.tool()
    def helix_edit(
        path: str,
        find: str,
        replace: str,
        all: bool = False,
        expected_blob_sha: str | None = None,
        expected_head: str | None = None,
        dry_run: bool = False,
        allow_protected: bool = False,
    ) -> dict[str, Any]:
        return edit(
            code,
            path=path,
            find=find,
            replace=replace,
            all=all,
            expected_blob_sha=expected_blob_sha,
            expected_head=expected_head,
            dry_run=dry_run,
            allow_protected=allow_protected,
        )

    @server.tool()
    def helix_routes(include_readiness: bool = True) -> dict[str, Any]:
        return routes(include_readiness=include_readiness)

    @server.tool()
    def helix_run(program: str, arguments: list[str] | None = None, timeout_seconds: int = 300) -> dict[str, Any]:
        return run(program, arguments=arguments, timeout_seconds=timeout_seconds)

    @server.tool()
    def helix_http(url: str, method: str = "GET", timeout_seconds: int = 30) -> dict[str, Any]:
        return http(url, method=method, timeout_seconds=timeout_seconds)

    if include_root_resource:
        @server.resource("helix://root")
        def helix_root_resource() -> str:
            return json.dumps(project_identity(), ensure_ascii=False)

    @server.resource("helix://head")
    def helix_head_resource() -> str:
        return json.dumps(git_state(), ensure_ascii=False)

    if include_capability_resource:
        @server.resource("helix://capabilities/{capability_id}")
        def helix_capability_resource(capability_id: str) -> str:
            return json.dumps(capabilities(capability_id), ensure_ascii=False)

    @server.resource("helix://skills/{name}")
    def helix_skill_resource(name: str) -> str:
        return json.dumps(skill(name), ensure_ascii=False)


__all__ = [
    "VERSION",
    "LocalSurfaceError",
    "ROUTES",
    "capabilities",
    "donor_repository",
    "edit",
    "git_state",
    "http",
    "project_identity",
    "read",
    "register",
    "routes",
    "run",
    "search",
    "skill",
    "skills",
    "write",
]
