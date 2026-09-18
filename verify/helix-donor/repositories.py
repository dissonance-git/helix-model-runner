"""Ephemeral repository evidence for Helix donor passes.

A donor checkout is machine-local working material. It can be deleted at any
time. Durable Helix state is the adapted mechanism/code and the provenance
needed to explain or re-enter that transfer.

Keeping an upstream repository pinned is a separate relationship. It must be
explicitly declared by the receiving project because correctness/alignment
depends on following that upstream.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from engine.paths import STATE_CACHE


VERSION = "repository-evidence-001.0"
DEFAULT_CACHE_ROOT = STATE_CACHE / "donor-repositories"
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_TEXT_SUFFIXES = frozenset({
    ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html",
    ".java", ".js", ".jsx", ".json", ".kt", ".lua", ".md", ".mjs", ".py",
    ".rb", ".rs", ".sh", ".toml", ".ts", ".tsx", ".txt", ".yaml", ".yml",
})
_SKIP_PARTS = frozenset({
    ".git", ".venv", "build", "dist", "node_modules", "target", "vendor",
})
_STRUCTURAL_TERMS = frozenset({
    "adapter", "benchmark", "cache", "checkpoint", "compiler", "controller",
    "error", "evaluator", "event", "fallback", "graph", "index", "invariant",
    "memory", "optimizer", "parser", "pipeline", "planner", "policy",
    "provenance", "queue", "reflection", "representation", "retry", "route",
    "router", "scheduler", "search", "selector", "snapshot", "solver", "state",
    "test", "transaction", "transform", "validation", "verifier", "workflow",
})
_ROOT_PRIORITY = (
    "README.md", "README.rst", "README.txt", "README",
    "ARCHITECTURE.md", "DESIGN.md", "CONTRIBUTING.md",
    "pyproject.toml", "package.json", "Cargo.toml", "go.mod",
    "requirements.txt", "LICENSE", "LICENSE.md", "COPYING",
)


class RepositoryEvidenceError(ValueError):
    pass


def repository_identity(value: str) -> str:
    text = str(value or "").strip()
    for prefix in ("https://github.com/", "http://github.com/"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    text = text.removesuffix(".git").strip("/")
    if not _REPOSITORY_RE.fullmatch(text):
        raise RepositoryEvidenceError("repository must be GitHub owner/name")
    return text


def github_clone_url(repository: str) -> str:
    return f"https://github.com/{repository_identity(repository)}.git"


def popularity_stratum(stars: int | None) -> str | None:
    """Use the deliberately separated strata established by Helix donor passes."""
    value = int(stars or 0)
    if value == 0:
        return "zero_star"
    if 100 <= value <= 1000:
        return "middle_tail"
    if value > 5000:
        return "popular"
    return None


def _selection_key(seed: str, repository: str) -> str:
    return hashlib.sha256(
        f"{seed}|{repository_identity(repository)}".encode("utf-8")
    ).hexdigest()


def select_serendipity_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selection_seed: str,
    per_stratum: int,
) -> dict[str, Any]:
    """Seeded, task-agnostic sampling across popularity strata.

    No receiving-task text is accepted here on purpose. Candidate discovery must
    already have been task-decorrelated; this function then samples only from
    repository identity, popularity metadata, language/owner diversity, and the
    run seed.
    """
    seed = str(selection_seed or "").strip()
    if not seed:
        raise RepositoryEvidenceError("selection_seed is required")
    count = int(per_stratum)
    if count < 1:
        raise RepositoryEvidenceError("per_stratum must be positive")

    strata: dict[str, list[dict[str, Any]]] = {
        "popular": [], "middle_tail": [], "zero_star": [],
    }
    seen: set[str] = set()
    for raw in candidates:
        row = dict(raw)
        try:
            identity = repository_identity(str(row.get("full_name") or ""))
        except RepositoryEvidenceError:
            continue
        if identity in seen:
            continue
        stratum = popularity_stratum(row.get("stars"))
        if stratum is None:
            continue
        seen.add(identity)
        row["full_name"] = identity
        row["stratum"] = stratum
        row["selection_key"] = _selection_key(seed, identity)
        strata[stratum].append(row)

    selected: dict[str, list[dict[str, Any]]] = {}
    for stratum, rows in strata.items():
        rows.sort(key=lambda row: (str(row["selection_key"]), str(row["full_name"])))
        owner_cap = max(2, math.ceil(count / 8))
        language_cap = max(3, math.ceil(count / 3))
        owner_counts: Counter[str] = Counter()
        language_counts: Counter[str] = Counter()
        chosen: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        for row in rows:
            owner = str(row["full_name"]).split("/", 1)[0].casefold()
            language = str(row.get("language") or "(unknown)").casefold()
            if owner_counts[owner] >= owner_cap or language_counts[language] >= language_cap:
                deferred.append(row)
                continue
            chosen.append(row)
            owner_counts[owner] += 1
            language_counts[language] += 1
            if len(chosen) >= count:
                break
        if len(chosen) < count:
            already = {str(row["full_name"]) for row in chosen}
            for row in deferred:
                if str(row["full_name"]) in already:
                    continue
                chosen.append(row)
                already.add(str(row["full_name"]))
                if len(chosen) >= count:
                    break
        selected[stratum] = chosen

    return {
        "schema_version": VERSION,
        "selection_seed": seed,
        "selection_method": (
            "seeded hash within popularity stratum; owner/language diversity guard; "
            "no task-similarity ranking"
        ),
        "strata_available": {key: len(value) for key, value in strata.items()},
        "strata_selected": {key: len(value) for key, value in selected.items()},
        "selected": selected,
        "task_similarity_used": False,
    }


def donor_retention_policy(*, upstream_alignment: bool = False) -> dict[str, Any]:
    if upstream_alignment:
        return {
            "relationship": "upstream-alignment",
            "checkout_is_disposable_cache": True,
            "retain_exact_upstream_identity": True,
            "refresh_against_upstream": True,
            "reason": (
                "Receiving project correctness or compatibility explicitly depends "
                "on continuing alignment with this upstream."
            ),
        }
    return {
        "relationship": "ordinary-donor",
        "checkout_is_disposable_cache": True,
        "retain_exact_upstream_identity": False,
        "refresh_against_upstream": False,
        "reason": (
            "Keep only adapted mechanism/code plus source provenance and revisit "
            "conditions. Re-clone the donor if later inspection is needed."
        ),
    }


def _run_git(
    cwd: Path | None,
    args: Sequence[str],
    *,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        ["git", *[str(arg) for arg in args]],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def _cache_path(repository: str, cache_root: Path) -> Path:
    identity = repository_identity(repository)
    slug = identity.replace("/", "--")
    prefix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return Path(cache_root) / f"{prefix}-{slug}"


def hydrate_github_repository(
    repository: str,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    refresh: bool = False,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Shallow-clone a public GitHub repository into disposable machine cache."""
    identity = repository_identity(repository)
    root = _cache_path(identity, cache_root)
    root.parent.mkdir(parents=True, exist_ok=True)
    cache_hit = (root / ".git").is_dir()

    if root.exists() and not cache_hit:
        shutil.rmtree(root, ignore_errors=True)
    if not cache_hit:
        completed = _run_git(
            None,
            [
                "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
                "--no-tags", github_clone_url(identity), str(root),
            ],
            timeout=timeout_seconds,
        )
        if completed.returncode:
            raise RepositoryEvidenceError(
                "public git clone failed: "
                + (completed.stderr.strip() or completed.stdout.strip())[:600]
            )
    elif refresh:
        fetched = _run_git(
            root, ["fetch", "--depth", "1", "--no-tags", "origin", "HEAD"],
            timeout=timeout_seconds,
        )
        if fetched.returncode:
            raise RepositoryEvidenceError(
                "public git refresh failed: "
                + (fetched.stderr.strip() or fetched.stdout.strip())[:600]
            )
        advanced = _run_git(
            root, ["update-ref", "HEAD", "FETCH_HEAD"],
            timeout=30,
        )
        if advanced.returncode:
            raise RepositoryEvidenceError(
                "public git metadata refresh failed: "
                + (advanced.stderr.strip() or advanced.stdout.strip())[:600]
            )

    head = _run_git(root, ["rev-parse", "HEAD"], timeout=30)
    if head.returncode:
        raise RepositoryEvidenceError("cached donor checkout has no readable HEAD")
    return {
        "schema_version": VERSION,
        "repository": identity,
        "clone_url": github_clone_url(identity),
        "root": str(root),
        "observed_head": head.stdout.strip(),
        "cache_hit": cache_hit,
        "refreshed": bool(cache_hit and refresh),
        "cache_policy": donor_retention_policy(upstream_alignment=False),
    }


def _tracked_files(root: Path) -> list[str]:
    completed = _run_git(root, ["ls-tree", "-r", "--name-only", "HEAD"], timeout=30)
    if completed.returncode:
        raise RepositoryEvidenceError("unable to enumerate donor repository")
    return [line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if line.strip()]


def _text_file(path: Path) -> bool:
    if path.suffix.casefold() in _TEXT_SUFFIXES:
        return True
    return path.name in {"README", "LICENSE", "COPYING", "Makefile", "Dockerfile"}


def _read_bounded(path: Path, limit: int = 48_000) -> str:
    try:
        data = path.read_bytes()[:limit]
    except OSError:
        return ""
    if b"\x00" in data:
        return ""
    return data.decode("utf-8", errors="replace")


def _read_repository_text(
    root: Path,
    relative: str,
    *,
    limit: int = 48_000,
) -> tuple[str, int, bool]:
    """Hydrate one selected text blob from a no-checkout donor cache."""
    local = Path(root) / relative
    if local.is_file():
        try:
            data = local.read_bytes()
        except OSError:
            return "", 0, False
        if b"\x00" in data:
            return "", len(data), False
        text = data.decode("utf-8", errors="replace")
        return text[:limit], len(data), len(text) > limit

    if not (Path(root) / ".git").is_dir():
        return "", 0, False
    completed = _run_git(Path(root), ["show", f"HEAD:{relative}"], timeout=45)
    if completed.returncode:
        return "", 0, False
    text = completed.stdout or ""
    if "\x00" in text:
        return "", len(text.encode("utf-8", errors="replace")), False
    encoded = text.encode("utf-8", errors="replace")
    return text[:limit], len(encoded), len(text) > limit


def _repository_path_references(
    texts: Iterable[str],
    *,
    available: set[str],
) -> list[str]:
    """Find explicit repository-relative text paths mentioned by inspected files."""
    counts: Counter[str] = Counter()
    pattern = re.compile(
        r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
        r"\.(?:py|rs|go|c|cc|cpp|h|hpp|java|kt|js|jsx|ts|tsx|lua|rb|sh|"
        r"md|rst|txt|toml|yaml|yml|json)"
    )
    for text in texts:
        for match in pattern.finditer(text or ""):
            value = match.group(0).strip()
            if value in available:
                counts[value] += 1
    return [
        rel for rel, _count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def inspect_repository_tree(
    root: Path,
    *,
    repository: str,
    question: str,
    lane: str,
    tracked_files: Sequence[str] | None = None,
    observed_head: str | None = None,
    maximum_files: int = 12,
    focus_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Browse the whole tree, then hydrate only a bounded set of useful blobs."""
    identity = repository_identity(repository)
    base = Path(root)
    files = list(tracked_files) if tracked_files is not None else _tracked_files(base)
    available_paths = {str(rel).replace("\\", "/") for rel in files}
    limit = max(1, min(int(maximum_files), 40))
    terms = {
        token for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", str(question).casefold())
        if token not in {
            "about", "after", "against", "from", "have", "into", "more",
            "question", "research", "that", "the", "their", "this", "what",
            "when", "where", "which", "with",
        }
    }

    explicit_focus = {
        str(value).strip().replace("\\", "/")
        for value in (focus_paths or ())
        if str(value).strip()
    }

    suffixes: Counter[str] = Counter()
    ranked_paths: list[tuple[int, str, list[str]]] = []
    text_files = 0
    for rel in files:
        rel = str(rel).replace("\\", "/")
        path = Path(rel)
        if any(part in _SKIP_PARTS for part in path.parts):
            continue
        suffixes[path.suffix.casefold() or "(none)"] += 1
        if not _text_file(path):
            continue
        text_files += 1
        lower_path = rel.casefold()
        reasons: list[str] = []
        score = 0
        if rel in explicit_focus:
            score += 200
            reasons.append("explicit-focus")
        if path.name in _ROOT_PRIORITY:
            score += 80
            reasons.append("root-contract")
        if any(
            part.casefold() in {"test", "tests", "testing", "bench", "benchmarks", "examples"}
            for part in path.parts
        ):
            score += 24
            reasons.append("test-or-example")
        structural = sorted(term for term in _STRUCTURAL_TERMS if term in lower_path)
        if structural:
            score += 5 * len(structural)
            reasons.extend(f"structural:{term}" for term in structural[:4])
        path_matches = sorted(term for term in terms if term in lower_path)
        if path_matches:
            score += 12 * len(path_matches)
            reasons.extend(f"question-path:{term}" for term in path_matches[:4])
        if len(path.parts) <= 3:
            score += 2
            reasons.append("shallow-source")
        ranked_paths.append((score, rel, reasons))

    ranked_paths.sort(key=lambda row: (-row[0], row[1]))
    probe_limit = min(len(ranked_paths), max(limit * 2, 16))
    probed: list[dict[str, Any]] = []
    for score, rel, reasons in ranked_paths[:probe_limit]:
        text, blob_bytes, truncated = _read_repository_text(base, rel, limit=48_000)
        if not text:
            continue
        folded = text.casefold()
        content_matches = sorted(term for term in terms if term in folded)
        if content_matches:
            score += min(30, 3 * len(content_matches))
            reasons = [
                *reasons,
                *(f"question-content:{term}" for term in content_matches[:4]),
            ]
        structural_content = [
            term for term in _STRUCTURAL_TERMS if term in folded
        ]
        if structural_content:
            score += min(15, len(structural_content))
        probed.append({
            "path": rel,
            "score": score,
            "reasons": reasons,
            "bytes": blob_bytes,
            "text": text,
            "probe_truncated": truncated,
        })

    initial_paths = {str(row["path"]) for row in probed}
    path_metadata = {
        rel: (score, reasons)
        for score, rel, reasons in ranked_paths
    }
    expansion_paths = _repository_path_references(
        (str(row["text"]) for row in probed),
        available=available_paths,
    )
    expanded: list[str] = []
    for rel in expansion_paths:
        if len(expanded) >= min(limit, 12):
            break
        if rel in initial_paths:
            continue
        score, reasons = path_metadata.get(rel, (0, []))
        text, blob_bytes, truncated = _read_repository_text(base, rel, limit=48_000)
        if not text:
            continue
        folded = text.casefold()
        content_matches = sorted(term for term in terms if term in folded)
        expanded_score = score + 18
        expanded_reasons = [*reasons, "reference-expansion"]
        if content_matches:
            expanded_score += min(30, 3 * len(content_matches))
            expanded_reasons.extend(
                f"question-content:{term}" for term in content_matches[:4]
            )
        structural_content = [
            term for term in _STRUCTURAL_TERMS if term in folded
        ]
        if structural_content:
            expanded_score += min(15, len(structural_content))
        probed.append({
            "path": rel,
            "score": expanded_score,
            "reasons": expanded_reasons,
            "bytes": blob_bytes,
            "text": text,
            "probe_truncated": truncated,
        })
        expanded.append(rel)

    probed.sort(key=lambda row: (-int(row["score"]), str(row["path"])))
    chosen = probed[:limit]
    surfaces: list[dict[str, Any]] = []
    for row in chosen:
        text = str(row["text"])
        surfaces.append({
            "path": row["path"],
            "score": row["score"],
            "reasons": row["reasons"],
            "bytes": row["bytes"],
            "excerpt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "excerpt": text[:4000],
            "excerpt_truncated": bool(row["probe_truncated"]) or len(text) > 4000,
        })

    if observed_head is None and (base / ".git").is_dir():
        head = _run_git(base, ["rev-parse", "HEAD"], timeout=30)
        observed_head = head.stdout.strip() if head.returncode == 0 else None

    return {
        "schema_version": VERSION,
        "repository": identity,
        "observed_head": observed_head,
        "selection_lane": str(lane),
        "tracked_files": len(files),
        "text_files": text_files,
        "tracked_bytes": None,
        "suffix_counts": dict(suffixes.most_common(16)),
        "surfaces": surfaces,
        "surface_count": len(surfaces),
        "browse_trace": {
            "tree_paths_seen": len(files),
            "tree_browsing": "commit/tree metadata only",
            "working_tree_checkout": False if (base / ".git").is_dir() else None,
            "path_candidates_ranked": len(ranked_paths),
            "content_blobs_probed": len(probed),
            "content_blob_bytes_observed": sum(int(row["bytes"]) for row in probed),
            "reference_expansion_paths": expanded,
            "explicit_focus_paths": sorted(explicit_focus),
            "policy": (
                "Browse the full repository tree first; hydrate only a bounded "
                "question-shaped set of text blobs for donor extraction."
            ),
        },
        "retention": donor_retention_policy(upstream_alignment=False),
        "extraction_contract": {
            "source": "what system/tool was observed",
            "pressure": "what constraint forced the useful structure",
            "operation_or_invariant": "smallest transferable mechanism",
            "evidence": "exact source file/region or observed behavior",
            "receiving_hypothesis": "what the mechanism suggests for the current question",
            "transfer_test": "what must improve/fail in the receiving project if transfer is real",
            "null_is_valid": True,
        },
        "claim_boundary": (
            "Repository inspection is candidate-mechanism evidence only. Popularity, "
            "selection, source similarity, and the donor's own claims do not validate "
            "a transfer. Adapted behavior must be tested in the receiving system."
        ),
    }


def inspect_github_donor(
    repository: str,
    *,
    question: str,
    lane: str,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    refresh: bool = False,
    maximum_files: int = 12,
    focus_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    hydration = hydrate_github_repository(
        repository,
        cache_root=cache_root,
        refresh=refresh,
    )
    inspection = inspect_repository_tree(
        Path(hydration["root"]),
        repository=hydration["repository"],
        question=question,
        lane=lane,
        observed_head=hydration["observed_head"],
        maximum_files=maximum_files,
        focus_paths=focus_paths,
    )
    return {
        **inspection,
        "provider_id": "public-git",
        "role": "donor-deep-inspection",
        "status": "executed",
        "cache_hit": hydration["cache_hit"],
        "clone_url": hydration["clone_url"],
        "evidence_route": (
            hydration["clone_url"] + "@" + str(hydration["observed_head"])
        ),
    }


__all__ = [
    "DEFAULT_CACHE_ROOT",
    "RepositoryEvidenceError",
    "VERSION",
    "donor_retention_policy",
    "github_clone_url",
    "hydrate_github_repository",
    "inspect_github_donor",
    "inspect_repository_tree",
    "popularity_stratum",
    "repository_identity",
    "select_serendipity_candidates",
]
