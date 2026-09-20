"""Canonical serialization and stable content-derived identity."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    raw = value if isinstance(value, bytes) else canonical_json(value).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    """Return the established stable ID without changing historical hashes."""
    return f"{prefix}-{digest([str(part) for part in parts])[:20].upper()}"


__all__ = ["canonical_json", "digest", "stable_id"]
