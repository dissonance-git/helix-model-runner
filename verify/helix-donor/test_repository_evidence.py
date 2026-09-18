from __future__ import annotations

from pathlib import Path

from engine.query.evidence.repositories import (
    donor_retention_policy,
    hydrate_github_repository,
    inspect_repository_tree,
    popularity_stratum,
    repository_identity,
    select_serendipity_candidates,
)


def _candidates(prefix: str, stars: int, count: int, language: str):
    return [
        {
            "full_name": f"{prefix}{i}/repo{i}",
            "stars": stars,
            "language": language if i % 2 else "Python",
            "description": "intentionally unrelated fixture",
        }
        for i in range(count)
    ]


def test_serendipity_selection_is_seeded_stratified_and_task_agnostic() -> None:
    candidates = [
        *_candidates("popular", 9000, 20, "Rust"),
        *_candidates("middle", 500, 20, "Go"),
        *_candidates("zero", 0, 20, "Lua"),
    ]
    first = select_serendipity_candidates(
        candidates,
        selection_seed="run-027-question-shape",
        per_stratum=6,
    )
    second = select_serendipity_candidates(
        candidates,
        selection_seed="run-027-question-shape",
        per_stratum=6,
    )
    other = select_serendipity_candidates(
        candidates,
        selection_seed="different-run",
        per_stratum=6,
    )
    assert first == second
    assert first["strata_selected"] == {
        "popular": 6, "middle_tail": 6, "zero_star": 6,
    }
    assert first["task_similarity_used"] is False
    assert first["selected"] != other["selected"]


def test_popularity_strata_preserve_deliberate_separation() -> None:
    assert popularity_stratum(7000) == "popular"
    assert popularity_stratum(1000) == "middle_tail"
    assert popularity_stratum(100) == "middle_tail"
    assert popularity_stratum(0) == "zero_star"
    assert popularity_stratum(12) is None
    assert popularity_stratum(3000) is None


def test_donor_retention_is_disposable_unless_upstream_is_explicit() -> None:
    donor = donor_retention_policy()
    assert donor["relationship"] == "ordinary-donor"
    assert donor["checkout_is_disposable_cache"] is True
    assert donor["retain_exact_upstream_identity"] is False
    upstream = donor_retention_policy(upstream_alignment=True)
    assert upstream["relationship"] == "upstream-alignment"
    assert upstream["retain_exact_upstream_identity"] is True
    assert upstream["refresh_against_upstream"] is True


def test_local_inspection_extracts_bounded_transfer_surfaces(tmp_path: Path) -> None:
    files = {
        "README.md": "# Queue engine\nUses backpressure and retries.\n",
        "src/router.py": "class Router:\n    # route candidate by invariant\n    pass\n",
        "tests/test_router.py": "def test_retry_route():\n    assert True\n",
        "src/irrelevant.py": "x = 1\n",
    }
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    result = inspect_repository_tree(
        tmp_path,
        repository="example/donor",
        question="Can routing and retries improve the controller?",
        lane="serendipity",
        tracked_files=list(files),
        observed_head="a" * 40,
        maximum_files=3,
    )
    assert result["repository"] == "example/donor"
    assert result["observed_head"] == "a" * 40
    assert result["surface_count"] == 3
    assert result["retention"]["relationship"] == "ordinary-donor"
    assert result["extraction_contract"]["null_is_valid"] is True
    selected = {row["path"] for row in result["surfaces"]}
    assert "README.md" in selected
    assert "src/router.py" in selected or "tests/test_router.py" in selected


def test_repository_identity_accepts_github_url_but_rejects_arbitrary_git_url() -> None:
    assert repository_identity("https://github.com/aider-ai/aider.git") == "aider-ai/aider"
    try:
        repository_identity("https://example.com/a/b.git")
    except ValueError:
        pass
    else:
        raise AssertionError("arbitrary clone URL should not be admitted")


def test_public_git_hydration_is_metadata_first(monkeypatch, tmp_path: Path) -> None:
    calls = []

    class Completed:
        def __init__(self, returncode=0, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cwd, args, *, timeout):
        calls.append((cwd, list(args)))
        if args and args[0] == "clone":
            root = Path(args[-1])
            (root / ".git").mkdir(parents=True)
            return Completed()
        if args[:2] == ["rev-parse", "HEAD"]:
            return Completed(stdout="a" * 40)
        return Completed()

    monkeypatch.setattr(
        "engine.query.evidence.repositories._run_git",
        fake_run,
    )
    result = hydrate_github_repository(
        "example/donor",
        cache_root=tmp_path,
    )
    clone = next(args for _cwd, args in calls if args and args[0] == "clone")
    assert "--filter=blob:none" in clone
    assert "--no-checkout" in clone
    assert "checkout" not in [args[0] for _cwd, args in calls]
    assert result["observed_head"] == "a" * 40


def test_focus_paths_narrow_second_pass_without_broad_checkout(tmp_path: Path) -> None:
    files = {
        "README.md": "# Donor\nGeneral overview.\n",
        "src/router.py": "def route():\n    return 'route'\n",
        "src/special.py": "def rare_mechanism():\n    return 'selected'\n",
        "tests/test_router.py": "def test_route():\n    assert True\n",
    }
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    result = inspect_repository_tree(
        tmp_path,
        repository="example/donor",
        question="Inspect the donor mechanism.",
        lane="direct",
        tracked_files=list(files),
        observed_head="b" * 40,
        maximum_files=1,
        focus_paths=["src/special.py"],
    )
    assert result["surfaces"][0]["path"] == "src/special.py"
    assert "explicit-focus" in result["surfaces"][0]["reasons"]
    assert result["browse_trace"]["explicit_focus_paths"] == ["src/special.py"]
    assert result["browse_trace"]["path_candidates_ranked"] == len(files)


def test_repository_reference_expansion_finds_buried_owner(tmp_path: Path) -> None:
    target = "deep/nested/mechanism/core.py"
    files = {
        "README.md": (
            "# Donor\n"
            "The implementation owner is deep/nested/mechanism/core.py.\n"
        ),
        target: "def mechanism():\n    return 'transferable'\n",
    }
    for index in range(24):
        files[f"src/module_{index:02d}.py"] = f"value = {index}\n"
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    result = inspect_repository_tree(
        tmp_path,
        repository="example/donor",
        question="Find the transferable donor mechanism.",
        lane="direct",
        tracked_files=list(files),
        observed_head="c" * 40,
        maximum_files=4,
    )
    selected = {row["path"]: row for row in result["surfaces"]}
    assert target in selected
    assert "reference-expansion" in selected[target]["reasons"]
    assert target in result["browse_trace"]["reference_expansion_paths"]
    assert result["browse_trace"]["content_blobs_probed"] < len(files)
