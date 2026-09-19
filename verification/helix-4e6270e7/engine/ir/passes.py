"""Transformation passes over semantic representations."""
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from engine.kernel import digest, stable_id


TRANSFORMATION_VERSION = "transformation-001.0"
DEFECT_VERSION = "defect-001.0"
TRANSFORM_STATES = ("exact", "conditional", "approximate", "empirical", "conjectural", "invalid")
_PASS_STATUSES = {"pass", "passed", "ok", "valid", "verified", "verified_exact"}

TRANSFORM_OPERATOR_BASIS = (
    "PROJECT", "REPROJECT", "LIFT", "LOWER", "FACTOR",
    "COMPOSE", "DUALIZE", "RELAX", "TIGHTEN",
)

TRANSFORM_OPERATOR_CONTRACTS: dict[str, dict[str, str]] = {
    "PROJECT": {
        "family": "projection",
        "inverse": "REPROJECT",
        "axis": "representation",
        "effect": "change coordinates or native view while declaring hidden distinctions",
    },
    "REPROJECT": {
        "family": "projection",
        "inverse": "PROJECT",
        "axis": "representation",
        "effect": "return or move into another licensed native view",
    },
    "LIFT": {
        "family": "dimension",
        "inverse": "LOWER",
        "axis": "dimensionality",
        "effect": "introduce explicit dimensions, parameters, or distinctions",
    },
    "LOWER": {
        "family": "dimension",
        "inverse": "LIFT",
        "axis": "dimensionality",
        "effect": "remove dimensions or distinctions with information loss declared",
    },
    "FACTOR": {
        "family": "composition",
        "inverse": "COMPOSE",
        "axis": "composition",
        "effect": "separate components or interfaces",
    },
    "COMPOSE": {
        "family": "composition",
        "inverse": "FACTOR",
        "axis": "composition",
        "effect": "combine compatible components or views",
    },
    "DUALIZE": {
        "family": "duality",
        "inverse": "DUALIZE",
        "axis": "roles",
        "effect": "exchange a licensed role, direction, or viewpoint",
    },
    "RELAX": {
        "family": "constraint",
        "inverse": "TIGHTEN",
        "axis": "admissible-state-set",
        "effect": "weaken constraints or exactness with weakened obligations declared",
    },
    "TIGHTEN": {
        "family": "constraint",
        "inverse": "RELAX",
        "axis": "admissible-state-set",
        "effect": "strengthen constraints or exactness under explicit preconditions",
    },
}


def transform_operator_contract(name: str) -> dict[str, str]:
    key = str(name or "").strip().upper()
    try:
        return deepcopy(TRANSFORM_OPERATOR_CONTRACTS[key])
    except KeyError as exc:
        raise KeyError(f"unknown first-class transform operator: {name}") from exc


def _basis_operator(spec: "OperatorSpec") -> str | None:
    candidate = str(spec.basis_operator or spec.name or "").strip().upper()
    return candidate if candidate in TRANSFORM_OPERATOR_CONTRACTS else None


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    source_types: tuple[str, ...]
    target_type: str
    preconditions: tuple[str, ...]
    preserved_obligations: tuple[str, ...]
    state: str = "exact"
    basis_operator: str | None = None
    implementation: Callable[[Any], Any] | None = field(default=None, compare=False, repr=False)
    verifier: Callable[[Any, Any], Mapping[str, Any]] | None = field(default=None, compare=False, repr=False)
    certificate_requirements: tuple[str, ...] = ()
    cheapest_falsifier: str = "bounded differential comparison"
    precondition_verifier: Callable[[Any], Mapping[str, Any]] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.state not in TRANSFORM_STATES:
            raise ValueError(f"unknown transformation state: {self.state}")
        if self.basis_operator is not None and str(self.basis_operator).upper() not in TRANSFORM_OPERATOR_CONTRACTS:
            raise ValueError(f"unknown first-class transform operator: {self.basis_operator}")


class TransformationRegistry:
    def __init__(self) -> None:
        self._operators: dict[str, OperatorSpec] = {}

    def register(self, spec: OperatorSpec) -> None:
        if spec.name in self._operators:
            raise ValueError(f"duplicate operator: {spec.name}")
        self._operators[spec.name] = spec

    def get(self, name: str) -> OperatorSpec:
        try:
            return self._operators[name]
        except KeyError as exc:
            raise KeyError(f"unknown operator: {name}") from exc

    def discover(self, *, source_type: str | None = None) -> list[dict[str, Any]]:
        rows = []
        for name, spec in sorted(self._operators.items()):
            if source_type and source_type not in spec.source_types:
                continue
            basis = _basis_operator(spec)
            rows.append({
                "operator": name,
                "basis_operator": basis,
                "basis_contract": transform_operator_contract(basis) if basis else None,
                "source_types": list(spec.source_types),
                "target_type": spec.target_type,
                "preconditions": list(spec.preconditions),
                "preserved_obligations": list(spec.preserved_obligations),
                "state": spec.state,
                "certificate_requirements": list(spec.certificate_requirements),
                "cheapest_falsifier": spec.cheapest_falsifier,
                "implemented": spec.implementation is not None,
                "guarded_execution_required": bool(spec.preconditions),
                "precondition_verifier_available": spec.precondition_verifier is not None,
            })
        return rows

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._operators))


def verify_operator_preconditions(spec: OperatorSpec, source: Any) -> dict[str, Any]:
    """Evaluate declared preconditions exactly and fail closed."""
    if not spec.preconditions:
        return {
            "status": "not_required",
            "applicable": True,
            "exact": True,
            "verified_preconditions": [],
            "missing_preconditions": [],
            "wall_time_ns": 0,
        }
    if spec.precondition_verifier is None:
        raise ValueError(f"{spec.name} declares preconditions but has no precondition verifier; execution fails closed")
    started = time.perf_counter_ns()
    raw = spec.precondition_verifier(source)
    elapsed = time.perf_counter_ns() - started
    if not isinstance(raw, Mapping):
        raise ValueError(f"{spec.name} precondition verifier must return a mapping")
    result = deepcopy(dict(raw))
    status = str(result.get("status") or "").strip().lower()
    applicable = result.get("applicable")
    if applicable is None:
        applicable = status in _PASS_STATUSES
    verified = list(dict.fromkeys(str(item) for item in (result.get("verified_preconditions") or [])))
    missing = sorted(set(spec.preconditions) - set(verified))
    passed = bool(applicable) and result.get("exact") is True and not missing
    result.update({
        "status": "pass" if passed else "fail",
        "applicable": passed,
        "exact": result.get("exact") is True,
        "verified_preconditions": verified,
        "missing_preconditions": missing,
        "wall_time_ns": elapsed,
    })
    if not passed:
        reason = result.get("reason") or (
            "missing verified preconditions: " + ", ".join(missing)
            if missing else "applicability was not established exactly"
        )
        raise ValueError(f"{spec.name} preconditions not established: {reason}")
    return result


def require_guarded_exact_result_verification(spec: OperatorSpec, record: Mapping[str, Any]) -> None:
    """Reject a guarded exact transformation unless result verification passed."""
    if not spec.preconditions or spec.state != "exact":
        return
    if spec.verifier is None:
        raise ValueError(f"{spec.name} is a guarded exact transformation but has no result verifier; execution fails closed")
    verification = record.get("verification_state")
    status = str(verification.get("status") or "").strip().lower() if isinstance(verification, Mapping) else ""
    if status not in _PASS_STATUSES or (isinstance(verification, Mapping) and verification.get("exact") is False):
        raise ValueError(f"{spec.name} result verification did not pass exactly; status={status or None!r}")


def execute_transformation(
    registry: TransformationRegistry,
    operator: str,
    source: Any,
    *,
    source_id: str,
    source_type: str,
    required_obligations: Iterable[str],
    provenance: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    spec = registry.get(operator)
    required = tuple(dict.fromkeys(required_obligations))
    if source_type not in spec.source_types:
        raise ValueError(f"{operator} does not accept {source_type}")
    missing = sorted(set(required) - set(spec.preserved_obligations))
    if missing:
        raise ValueError(f"{operator} does not preserve required obligations: {missing}")
    if spec.implementation is None:
        raise NotImplementedError(operator)
    precondition_verification = verify_operator_preconditions(spec, source)
    started = time.perf_counter_ns()
    output = spec.implementation(source)
    elapsed = time.perf_counter_ns() - started
    target, details = output if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict) else (output, {})
    target_id = str(details.get("target_id") or stable_id("REP", operator, digest(target)))
    verification = dict(spec.verifier(source, target)) if spec.verifier else {"status": "not_run"}
    record = {
        "schema_version": TRANSFORMATION_VERSION,
        "transform_id": stable_id("TRANSFORM", operator, source_id, target_id, digest(details)),
        "operator": operator,
        "basis_operator": _basis_operator(spec),
        "basis_contract": transform_operator_contract(_basis_operator(spec)) if _basis_operator(spec) else None,
        "source": source_id,
        "target": target_id,
        "source_type": source_type,
        "target_type": spec.target_type,
        "preconditions": list(spec.preconditions),
        "preserved_obligations": list(spec.preserved_obligations),
        "weakened_obligations": list(details.get("weakened_obligations") or []),
        "introduced_information": list(details.get("introduced_information") or []),
        "discarded_information": list(details.get("discarded_information") or []),
        "expected_observable": details.get("expected_observable"),
        "known_risk": list(details.get("known_risk") or []),
        "construction_cost": {"wall_time_ns": elapsed, **dict(details.get("construction_cost") or {})},
        "peak_intermediate_size": details.get("peak_intermediate_size"),
        "inverse_or_recovery_route": details.get("inverse_or_recovery_route"),
        "certificate_requirements": list(spec.certificate_requirements),
        "cheapest_falsifier": spec.cheapest_falsifier,
        "implementation": f"{spec.implementation.__module__}.{spec.implementation.__name__}",
        "provenance": deepcopy(dict(provenance)),
        "verification_state": verification,
        "transformation_state": spec.state,
        "precondition_verification": precondition_verification,
        "precondition_verification_cost": {"wall_time_ns": int(precondition_verification.get("wall_time_ns") or 0)},
        "execution_owner": "engine.ir.passes.execute_transformation",
    }
    if precondition_verification.get("obligation_sufficient_state"):
        record["obligation_sufficient_state"] = deepcopy(precondition_verification["obligation_sufficient_state"])
    require_guarded_exact_result_verification(spec, record)
    return target, record


def execute_transform_program(
    registry: TransformationRegistry,
    operators: Iterable[str],
    source: Any,
    *,
    source_id: str,
    source_type: str,
    required_obligations: Iterable[str],
    provenance: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Execute a typed transformation program through the guarded IR owner."""
    names = tuple(str(name) for name in operators)
    required = tuple(dict.fromkeys(str(item) for item in required_obligations))
    current = source
    current_id = source_id
    current_type = source_type
    steps: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        current, record = execute_transformation(
            registry,
            name,
            current,
            source_id=current_id,
            source_type=current_type,
            required_obligations=required,
            provenance={**deepcopy(dict(provenance)), "program_step": index},
        )
        steps.append(record)
        current_id = str(record["target"])
        current_type = str(record["target_type"])

    recoveries = [row.get("inverse_or_recovery_route") for row in reversed(steps)]
    complete_recovery = all(recoveries) if steps else True
    body = {
        "schema_version": TRANSFORMATION_VERSION,
        "program_id": stable_id(
            "TRANSFORM-PROGRAM",
            source_id,
            names,
            [row.get("transform_id") for row in steps],
        ),
        "source": source_id,
        "target": current_id,
        "source_type": source_type,
        "target_type": current_type,
        "operators": list(names),
        "basis_operators": [row.get("basis_operator") for row in steps],
        "required_obligations": list(required),
        "steps": deepcopy(steps),
        "recovery_route": recoveries if complete_recovery else None,
        "contains_non_exact_step": any(
            row.get("transformation_state") != "exact" for row in steps
        ),
        "execution_owner": "engine.ir.passes.execute_transform_program",
        "provenance": deepcopy(dict(provenance)),
    }
    return current, body


def composition(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    defects: list[dict[str, Any]] = []
    if first.get("target") != second.get("source"):
        defects.append({"kind": "endpoint_mismatch", "first_target": first.get("target"), "second_source": second.get("source")})
    lost = set(first.get("discarded_information") or [])
    required = set(second.get("preconditions") or [])
    if lost & required:
        defects.append({"kind": "prerequisite_destroyed", "items": sorted(lost & required)})
    obligations = set(first.get("preserved_obligations") or []) & set(second.get("preserved_obligations") or [])
    if not obligations:
        defects.append({"kind": "no_composed_obligation", "items": []})
    recoverable = bool(first.get("inverse_or_recovery_route")) and bool(second.get("inverse_or_recovery_route"))
    if not recoverable:
        defects.append({"kind": "recovery_route_incomplete", "items": []})
    defect_records = [{
        "schema_version": DEFECT_VERSION,
        "defect_id": stable_id("DEFECT", first.get("transform_id"), second.get("transform_id"), item),
        "defect_type": item["kind"],
        "scope": "declared transformation pair",
        "witness": item,
    } for item in defects]
    return {
        "schema_version": TRANSFORMATION_VERSION,
        "transform_id": stable_id("TRANSFORM", "COMPOSE", first.get("transform_id"), second.get("transform_id")),
        "operator": "COMPOSE",
        "source": first.get("source"),
        "target": second.get("target"),
        "preserved_obligations": sorted(obligations),
        "construction_cost": {
            "component_wall_time_ns": sum(int(item.get("construction_cost", {}).get("wall_time_ns") or 0) for item in (first, second)),
        },
        "peak_intermediate_size": max((item.get("peak_intermediate_size") or 0) for item in (first, second)),
        "inverse_or_recovery_route": [first.get("inverse_or_recovery_route"), second.get("inverse_or_recovery_route")] if recoverable else None,
        "composition_defects": defect_records,
        "verification_state": "invalid" if defects else "composition_licensed",
    }


def commutation_comparison(
    value: Any,
    first: Callable[[Any], Any],
    second: Callable[[Any], Any],
    observer: Callable[[Any], Any],
    *,
    first_name: str,
    second_name: str,
    obligations: Iterable[str],
) -> dict[str, Any]:
    left = second(first(deepcopy(value)))
    right = first(second(deepcopy(value)))
    left_observation, right_observation = observer(left), observer(right)
    commutes = left_observation == right_observation
    return {
        "schema_version": DEFECT_VERSION,
        "defect_id": stable_id("COMMUTE", first_name, second_name, digest(value)),
        "defect_type": "none" if commutes else "noncommutation",
        "operators": [first_name, second_name],
        "obligations": list(dict.fromkeys(obligations)),
        "commutes": commutes,
        "minimal_witness": None if commutes else {"left": left_observation, "right": right_observation},
        "left_digest": digest(left_observation),
        "right_digest": digest(right_observation),
    }


__all__ = [
    "DEFECT_VERSION",
    "OperatorSpec",
    "TRANSFORMATION_VERSION",
    "TRANSFORM_OPERATOR_BASIS",
    "TRANSFORM_OPERATOR_CONTRACTS",
    "TRANSFORM_STATES",
    "TransformationRegistry",
    "commutation_comparison",
    "composition",
    "execute_transform_program",
    "execute_transformation",
    "require_guarded_exact_result_verification",
    "transform_operator_contract",
    "verify_operator_preconditions",
]
