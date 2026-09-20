"""Provider-neutral semantic contracts.

This module owns obligations, representations, and equivalence. It does not own
execution, persistence, retrieval, orchestration, or provider behavior.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterable, Mapping, Sequence

from .identity import digest, stable_id


REPRESENTATION_VERSION = "representation-001.0"
OBLIGATION_VERSION = "obligation-001.0"
EQUIVALENCE_VERSION = "equivalence-001.0"
AVAILABILITY = ("available", "unavailable", "not_applicable")


def obligation(
    name: str,
    *,
    scope: str,
    required: bool = True,
    observable: str | None = None,
    source_routes: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "schema_version": OBLIGATION_VERSION,
        "obligation_id": stable_id("OBL", name, scope, observable or ""),
        "name": name,
        "scope": scope,
        "required": required,
        "observable": observable,
        "source_routes": list(dict.fromkeys(str(item) for item in source_routes)),
    }


def representation(
    object_ids: Iterable[str],
    representation_type: str,
    *,
    construction_method: str,
    availability: str = "available",
    source_representation: str | None = None,
    visible_relations: Iterable[str] = (),
    hidden_relations: Iterable[str] = (),
    preserved_obligations: Iterable[str] = (),
    known_information_loss: Iterable[str] = (),
    recoverability: Mapping[str, Any] | None = None,
    construction_cost: Mapping[str, Any] | None = None,
    peak_construction_cost: Mapping[str, Any] | None = None,
    size_measure: Mapping[str, Any] | None = None,
    structural_widths: Mapping[str, Any] | None = None,
    domain: str = "general",
    verification_status: str = "generated",
    provenance: Mapping[str, Any] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    if availability not in AVAILABILITY:
        raise ValueError(f"unknown availability: {availability}")
    ids = tuple(dict.fromkeys(str(item) for item in object_ids))
    if not ids:
        raise ValueError("representation requires at least one exact object id")
    if availability != "available" and not reason:
        raise ValueError("unavailable and not_applicable representations require a reason")
    return {
        "schema_version": REPRESENTATION_VERSION,
        "representation_id": stable_id("REP", ids, representation_type, construction_method, source_representation or ""),
        "object_ids": list(ids),
        "representation_type": representation_type,
        "availability": availability,
        "availability_reason": reason,
        "construction_method": construction_method,
        "source_representation": source_representation,
        "visible_relations": list(dict.fromkeys(visible_relations)),
        "hidden_relations": list(dict.fromkeys(hidden_relations)),
        "preserved_obligations": list(dict.fromkeys(preserved_obligations)),
        "known_information_loss": list(dict.fromkeys(known_information_loss)),
        "recoverability": deepcopy(dict(recoverability or {})),
        "construction_cost": deepcopy(dict(construction_cost or {})),
        "peak_construction_cost": deepcopy(dict(peak_construction_cost or {})),
        "size_measure": deepcopy(dict(size_measure or {})),
        "structural_widths": deepcopy(dict(structural_widths or {})),
        "domain": domain,
        "verification_status": verification_status,
        "provenance": deepcopy(dict(provenance or {})),
    }


def equivalence(
    left_id: str,
    right_id: str,
    obligations: Iterable[str],
    *,
    equivalence_type: str,
    certificate: Mapping[str, Any],
    scope: str,
    known_non_preserved_properties: Iterable[str] = (),
    counterexamples: Iterable[Any] = (),
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    obligation_ids = tuple(dict.fromkeys(str(item) for item in obligations))
    if not obligation_ids:
        raise ValueError("equivalence requires an explicit obligation family")
    return {
        "schema_version": EQUIVALENCE_VERSION,
        "equivalence_id": stable_id("EQUIV", left_id, right_id, obligation_ids, scope),
        "left_id": left_id,
        "right_id": right_id,
        "equivalence_type": equivalence_type,
        "obligations": list(obligation_ids),
        "certificate": deepcopy(dict(certificate)),
        "scope": scope,
        "known_non_preserved_properties": list(dict.fromkeys(known_non_preserved_properties)),
        "counterexamples": deepcopy(list(counterexamples)),
        "provenance": deepcopy(dict(provenance or {})),
    }


def continuation_equivalence(
    left: Any,
    right: Any,
    continuations: Sequence[Any],
    evaluator: Callable[[Any, Any], Any],
    *,
    obligation_ids: Iterable[str],
    exhaustive_within_scope: bool,
    scope: str,
) -> dict[str, Any]:
    """Compare declared continuations; sampling cannot certify exactness."""
    observations = []
    witness = None
    for continuation in continuations:
        left_value = evaluator(left, continuation)
        right_value = evaluator(right, continuation)
        observations.append({"continuation": deepcopy(continuation), "left": left_value, "right": right_value})
        if left_value != right_value and witness is None:
            witness = observations[-1]
    exact = witness is None and exhaustive_within_scope
    status = "verified_bounded" if exact else "empirical_approximate" if witness is None else "refuted"
    certificate = {
        "continuation_scope": scope,
        "continuations_checked": len(continuations),
        "exhaustive_within_scope": exhaustive_within_scope,
        "observations_digest": digest(observations),
        "minimal_separating_witness": witness,
    }
    return {
        "equivalent": witness is None,
        "exact": exact,
        "status": status,
        "certificate": certificate,
        "equivalence": equivalence(
            stable_id("STATE", left),
            stable_id("STATE", right),
            obligation_ids,
            equivalence_type="continuation_equivalence",
            certificate=certificate,
            scope=scope,
            counterexamples=[witness] if witness else [],
        ),
    }


__all__ = [
    "AVAILABILITY",
    "EQUIVALENCE_VERSION",
    "OBLIGATION_VERSION",
    "REPRESENTATION_VERSION",
    "continuation_equivalence",
    "equivalence",
    "obligation",
    "representation",
]
