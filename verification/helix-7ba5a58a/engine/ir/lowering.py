"""Target adapters for lowering IR operations and raising execution results."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from engine.kernel import digest, stable_id
from engine.ir.model import IR_VERSION, topological_order, validate_module


LOWERING_VERSION = "lowering-001.0"


@dataclass(frozen=True)
class TargetAdapter:
    """Lower one target-independent operation and normalize its result."""

    name: str
    lower: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
    raise_result: Callable[[Mapping[str, Any], Any], Mapping[str, Any]]
    supports: Callable[[Mapping[str, Any]], bool] | None = None


class LoweringRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, TargetAdapter] = {}

    def register(self, adapter: TargetAdapter) -> None:
        name = str(adapter.name or "").strip()
        if not name:
            raise ValueError("target adapter name is required")
        if name in self._adapters:
            raise ValueError(f"duplicate target adapter: {name}")
        self._adapters[name] = adapter

    def get(self, target: str) -> TargetAdapter:
        try:
            return self._adapters[str(target)]
        except KeyError as exc:
            raise KeyError(f"unknown lowering target: {target}") from exc

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


def _operation(module: Mapping[str, Any], operation_id: str) -> dict[str, Any]:
    for row in module.get("operations") or []:
        if isinstance(row, Mapping) and str(row.get("operation_id") or "") == operation_id:
            return deepcopy(dict(row))
    raise KeyError(f"unknown operation: {operation_id}")


def lower_operation(
    registry: LoweringRegistry,
    module: Mapping[str, Any],
    operation_id: str,
    *,
    target: str,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Lower one legal IR operation to a backend-native request.

    Lowering does not execute the request and does not establish correctness.
    """
    validation = validate_module(module)
    if not validation["valid"]:
        raise ValueError("cannot lower invalid IR module: " + "; ".join(validation["errors"]))
    ordered = topological_order(module)
    operation = _operation(module, operation_id)
    adapter = registry.get(target)
    if adapter.supports is not None and not adapter.supports(operation):
        raise ValueError(f"target {target} does not support operation {operation_id}")
    context_value = deepcopy(dict(context or {}))
    request = adapter.lower(deepcopy(operation), context_value)
    if not isinstance(request, Mapping):
        raise ValueError("target lowerer must return a mapping")
    body = {
        "module_id": module.get("module_id"),
        "operation_id": operation_id,
        "target": target,
        "dependency_order": ordered,
        "request": deepcopy(dict(request)),
        "context_digest": digest(context_value),
        "ir_digest": digest(module),
    }
    return {
        "schema_version": LOWERING_VERSION,
        "request_id": stable_id("REQUEST", digest(body)),
        **body,
        "executed": False,
        "verified": False,
    }


def raise_result(
    registry: LoweringRegistry,
    lowered: Mapping[str, Any],
    raw_result: Any,
) -> dict[str, Any]:
    """Normalize a backend-native result without upgrading its verification state."""
    if lowered.get("schema_version") != LOWERING_VERSION:
        raise ValueError("lowered request schema_version mismatch")
    target = str(lowered.get("target") or "")
    adapter = registry.get(target)
    normalized = adapter.raise_result(deepcopy(dict(lowered)), deepcopy(raw_result))
    if not isinstance(normalized, Mapping):
        raise ValueError("target result raiser must return a mapping")
    normalized = deepcopy(dict(normalized))
    status = str(normalized.get("status") or "").strip()
    if not status:
        raise ValueError("raised result requires status")
    body = {
        "request_id": lowered.get("request_id"),
        "module_id": lowered.get("module_id"),
        "operation_id": lowered.get("operation_id"),
        "target": target,
        "status": status,
        "result": normalized.get("result"),
        "artifacts": deepcopy(list(normalized.get("artifacts") or [])),
        "observations": deepcopy(list(normalized.get("observations") or [])),
        "diagnostics": deepcopy(list(normalized.get("diagnostics") or [])),
        "provenance": deepcopy(dict(normalized.get("provenance") or {})),
    }
    return {
        "schema_version": LOWERING_VERSION,
        "result_id": stable_id("RESULT", digest(body)),
        **body,
        "verification": deepcopy(dict(normalized.get("verification") or {})),
        "verified": False,
    }
