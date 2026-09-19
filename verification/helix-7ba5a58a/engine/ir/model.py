"""Small provider-neutral intermediate representation.

The IR is a derived program representation. It references requirements, inputs,
outputs, dependencies, capabilities, verification contracts, and provenance
without becoming authority for the domain objects those references describe.

Keep this layer deliberately small. Domain-specific semantics stay in their
own modules until repeated cross-domain use earns a shared field or operation.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

from engine.kernel import digest, stable_id


IR_VERSION = "ir-001.0"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _copy_sequence(values: Iterable[Any] | None) -> list[Any]:
    return deepcopy(list(values or ()))


def build_operation(
    name: str,
    *,
    inputs: Iterable[Any] = (),
    outputs: Iterable[Any] = (),
    requirements: Iterable[Mapping[str, Any]] = (),
    constraints: Iterable[Mapping[str, Any]] = (),
    dependencies: Iterable[str] = (),
    capabilities: Iterable[str] = (),
    verification: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one typed operation without choosing an execution target."""
    operation_name = _text(name)
    if not operation_name:
        raise ValueError("operation name is required")
    dependency_ids = list(dict.fromkeys(_text(value) for value in dependencies if _text(value)))
    capability_ids = list(dict.fromkeys(_text(value) for value in capabilities if _text(value)))
    body = {
        "name": operation_name,
        "inputs": _copy_sequence(inputs),
        "outputs": _copy_sequence(outputs),
        "requirements": [deepcopy(dict(row)) for row in requirements],
        "constraints": [deepcopy(dict(row)) for row in constraints],
        "dependencies": dependency_ids,
        "capabilities": capability_ids,
        "verification": deepcopy(dict(verification or {})),
        "attributes": deepcopy(dict(attributes or {})),
    }
    return {
        "schema_version": IR_VERSION,
        "operation_id": stable_id("OP", operation_name, digest(body)),
        **body,
        "provenance": deepcopy(dict(provenance or {})),
    }


def build_module(
    name: str,
    operations: Sequence[Mapping[str, Any]],
    *,
    inputs: Iterable[Any] = (),
    outputs: Iterable[Any] = (),
    provenance: Mapping[str, Any] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compilation unit containing target-independent operations."""
    module_name = _text(name)
    if not module_name:
        raise ValueError("module name is required")
    rows = [deepcopy(dict(row)) for row in operations]
    body = {
        "name": module_name,
        "inputs": _copy_sequence(inputs),
        "outputs": _copy_sequence(outputs),
        "operations": rows,
        "attributes": deepcopy(dict(attributes or {})),
    }
    module = {
        "schema_version": IR_VERSION,
        "module_id": stable_id("MODULE", module_name, digest(body)),
        **body,
        "provenance": deepcopy(dict(provenance or {})),
    }
    validation = validate_module(module)
    if not validation["valid"]:
        raise ValueError("invalid IR module: " + "; ".join(validation["errors"]))
    return module


def _operation_index(module: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for raw in module.get("operations") or []:
        if not isinstance(raw, Mapping):
            raise ValueError("operation must be a mapping")
        row = dict(raw)
        operation_id = _text(row.get("operation_id"))
        if not operation_id:
            raise ValueError("operation_id is required")
        if operation_id in index:
            raise ValueError(f"duplicate operation_id: {operation_id}")
        index[operation_id] = row
    return index


def topological_order(module: Mapping[str, Any]) -> list[str]:
    """Return operation ids in dependency order; reject cycles and missing edges."""
    index = _operation_index(module)
    incoming: dict[str, set[str]] = {}
    outgoing: dict[str, set[str]] = {operation_id: set() for operation_id in index}
    for operation_id, row in index.items():
        dependencies = {
            _text(value)
            for value in row.get("dependencies") or []
            if _text(value)
        }
        if operation_id in dependencies:
            raise ValueError(f"operation depends on itself: {operation_id}")
        missing = sorted(dependencies - set(index))
        if missing:
            raise ValueError(
                f"operation {operation_id} has unknown dependencies: {', '.join(missing)}"
            )
        incoming[operation_id] = set(dependencies)
        for dependency in dependencies:
            outgoing[dependency].add(operation_id)

    ready = sorted(operation_id for operation_id, values in incoming.items() if not values)
    ordered: list[str] = []
    while ready:
        operation_id = ready.pop(0)
        ordered.append(operation_id)
        for dependent in sorted(outgoing[operation_id]):
            incoming[dependent].discard(operation_id)
            if not incoming[dependent] and dependent not in ordered and dependent not in ready:
                ready.append(dependent)
        ready.sort()
    if len(ordered) != len(index):
        cyclic = sorted(set(index) - set(ordered))
        raise ValueError("operation dependency cycle: " + ", ".join(cyclic))
    return ordered


def validate_module(module: Mapping[str, Any]) -> dict[str, Any]:
    """Validate IR structure without claiming domain correctness."""
    errors: list[str] = []
    warnings: list[str] = []
    if module.get("schema_version") != IR_VERSION:
        errors.append("schema_version mismatch")
    if not _text(module.get("module_id")):
        errors.append("module_id is required")
    if not _text(module.get("name")):
        errors.append("module name is required")
    try:
        index = _operation_index(module)
        if not index:
            errors.append("module requires at least one operation")
        else:
            topological_order(module)
        for operation_id, row in index.items():
            if row.get("schema_version") != IR_VERSION:
                errors.append(f"operation {operation_id} schema_version mismatch")
            if not _text(row.get("name")):
                errors.append(f"operation {operation_id} name is required")
            if not isinstance(row.get("requirements", []), list):
                errors.append(f"operation {operation_id} requirements must be a list")
            if not isinstance(row.get("constraints", []), list):
                errors.append(f"operation {operation_id} constraints must be a list")
            if not isinstance(row.get("verification", {}), Mapping):
                errors.append(f"operation {operation_id} verification must be a mapping")
    except ValueError as exc:
        errors.append(str(exc))
    return {
        "schema_version": IR_VERSION,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "domain_verified": False,
    }
