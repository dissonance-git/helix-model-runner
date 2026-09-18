from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from engine.paths import ROOT
from engine.query.mcp import local_surface
from engine.query.mcp.local_code import LocalCodeAdapter
from engine.query.mcp.operator_server import build_operator_server
from engine.query.mcp.server import build_server


SHARED_TOOLS = (
    "helix_capabilities",
    "helix_skill",
    "helix_git_state",
    "helix_read",
    "helix_search",
    "helix_donor_repository",
    "helix_write",
    "helix_edit",
    "helix_routes",
    "helix_run",
    "helix_http",
)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_project_identity_defers_first_connection_to_commons() -> None:
    packet = local_surface.project_identity()
    assert packet["server"] == "helix"
    assert packet["root"] == str(ROOT)
    assert packet["head"] == _git_head()
    assert set(local_surface.ROUTES) <= set(packet["routes"])
    assert packet["capability_count"] > 0
    assert "mcp-native-operation" in packet["skills"]
    assert packet["entry_owner"] == "helix-commons:projects_room_bootstrap"
    assert packet["project_entry"] == "describe_helix"
    assert packet["derived_projection"] is True
    assert packet["canonical_truth"] is False
    assert "bootstrap_contract" not in packet
    assert not hasattr(local_surface, "BOOTSTRAP_CONTRACT_VERSION")


def test_no_project_owned_bootstrap_alias_remains() -> None:
    assert not hasattr(local_surface, "bootstrap")


def test_capabilities_and_skill_reads_are_identified() -> None:
    catalog = local_surface.capabilities()
    assert catalog["count"] == len(catalog["capabilities"]) > 0
    detail = local_surface.capabilities(catalog["capabilities"][0])
    assert detail["capability"]["canonical_name"] == catalog["capabilities"][0]
    skill = local_surface.skill("mcp-native-operation")
    assert skill["sha256"]
    assert skill["text"].startswith("---")
    with pytest.raises(local_surface.LocalSurfaceError, match="unknown skill"):
        local_surface.skill("not-a-skill")


def test_read_and_search_carry_blob_identity() -> None:
    adapter = LocalCodeAdapter()
    payload = local_surface.read(adapter, "engine/query/mcp/local_surface.py")
    assert payload["sha256"]
    assert payload["head_blob_sha"] == payload["blob_sha"]
    result = local_surface.search(adapter, "project_identity", roots=["engine"], max_matches=5)
    assert any("project_identity" in row["line"] for row in result["matches"])


def test_donor_repository_delegates_to_repository_evidence_owner(monkeypatch) -> None:
    observed = {}

    def fake_inspect(repository: str, **kwargs):
        observed["repository"] = repository
        observed.update(kwargs)
        return {
            "repository": repository,
            "observed_head": "a" * 40,
            "status": "executed",
            "extraction_contract": {"null_is_valid": True},
        }

    monkeypatch.setattr(local_surface, "inspect_github_donor", fake_inspect)
    result = local_surface.donor_repository(
        "example/donor",
        question="What mechanism transfers?",
        lane="direct",
        refresh=True,
        maximum_files=7,
        focus_paths=["src/core.py"],
    )
    assert result["status"] == "executed"
    assert observed == {
        "repository": "example/donor",
        "question": "What mechanism transfers?",
        "lane": "direct",
        "refresh": True,
        "maximum_files": 7,
        "focus_paths": ["src/core.py"],
    }


def test_write_is_compare_and_swap_and_dry_run_by_default(tmp_path: Path) -> None:
    adapter = LocalCodeAdapter()
    with pytest.raises(Exception):
        local_surface.write(
            adapter,
            operation="update",
            path="engine/query/mcp/local_surface.py",
            expected_head="0" * 40,
            content="not allowed",
        )
    original = (ROOT / "engine/query/mcp/local_surface.py").read_bytes()
    current = local_surface.read(adapter, "engine/query/mcp/local_surface.py")
    preview = local_surface.write(
        adapter,
        operation="update",
        path="engine/query/mcp/local_surface.py",
        expected_head=_git_head(),
        expected_blob_sha=current["blob_sha"],
        content="should not land",
        dry_run=True,
    )
    assert preview["dry_run"] is True
    assert (ROOT / "engine/query/mcp/local_surface.py").read_bytes() == original


def test_edit_uses_git_blob_identity_and_preserves_sha256_checksum() -> None:
    adapter = LocalCodeAdapter()
    path = "engine/query/mcp/local_surface.py"
    original = (ROOT / path).read_bytes()
    current = local_surface.read(adapter, path)
    preview = local_surface.edit(
        adapter,
        path=path,
        find='VERSION = "mcp-local-surface-002.0"',
        replace='VERSION = "mcp-local-surface-test"',
        expected_head=_git_head(),
        expected_blob_sha=current["blob_sha"],
        dry_run=True,
    )
    assert preview["dry_run"] is True
    assert preview["before_blob_sha"] == current["blob_sha"]
    assert preview["previous_blob_sha"] == current["blob_sha"]
    assert preview["previous_sha256"] == current["sha256"]
    assert (ROOT / path).read_bytes() == original
    with pytest.raises(local_surface.LocalSurfaceError, match="blob mismatch"):
        local_surface.edit(
            adapter,
            path=path,
            find='VERSION = "mcp-local-surface-002.0"',
            replace='VERSION = "mcp-local-surface-test"',
            expected_head=_git_head(),
            expected_blob_sha="0" * 40,
            dry_run=True,
        )


def test_run_is_allowlisted_and_bounded() -> None:
    with pytest.raises(local_surface.LocalSurfaceError, match="not allowlisted"):
        local_surface.run("curl", arguments=["https://example.com"])
    ok = local_surface.run("python", arguments=["-c", "print('surface-ok')"], timeout_seconds=60)
    assert ok["exit_code"] == 0
    assert "surface-ok" in ok["stdout_tail"]


def test_http_refuses_non_allowed_prefixes() -> None:
    with pytest.raises(local_surface.LocalSurfaceError, match="https://"):
        local_surface.http("http://example.com")
    with pytest.raises(local_surface.LocalSurfaceError, match="method"):
        local_surface.http("https://example.com", method="POST")


def test_both_servers_expose_room_tools_but_not_a_local_bootstrap() -> None:
    for builder in (build_server, build_operator_server):
        server = builder()
        names = {tool.name for tool in asyncio.run(server.list_tools())}
        for tool in SHARED_TOOLS:
            assert tool in names
        assert "helix_bootstrap" not in names
