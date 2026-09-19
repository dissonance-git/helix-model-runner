"""Domain-neutral verification records."""
from __future__ import annotations

from copy import deepcopy
from enum import IntEnum
from typing import Any, Iterable, Mapping

from .identity import digest, stable_id


CERTIFICATE_VERSION = "certificate-001.0"


class VerificationLevel(IntEnum):
    V0_SCHEMA_SYNTAX = 0
    V1_DETERMINISTIC_INVARIANTS = 1
    V2_UNIT_PROPERTY_TESTS = 2
    V3_METAMORPHIC_ISOMORPHISM = 3
    V4_BOUNDED_EXHAUSTIVE = 4
    V5_INDEPENDENT_DIFFERENTIAL = 5
    V6_SAT_SMT_OPTIMIZATION_CERTIFICATE = 6
    V7_ALGEBRAIC_NUMERICAL_CERTIFICATE = 7
    V8_FORMAL_PROOF_ASSISTANT = 8
    V9_EMPIRICAL_REPLICATION = 9
    V10_EXTERNAL_SPECIALIST = 10


def verification_record(
    subject_id: str,
    level: VerificationLevel | int,
    status: str,
    *,
    verifier: str,
    certificate: Mapping[str, Any] | None = None,
    limitations: Iterable[str] = (),
) -> dict[str, Any]:
    level = VerificationLevel(level)
    return {
        "schema_version": CERTIFICATE_VERSION,
        "certificate_id": stable_id("CERT", subject_id, level.name, verifier, digest(certificate or {})),
        "subject_id": subject_id,
        "verification_level": f"V{int(level)}",
        "verification_name": level.name,
        "status": status,
        "verifier": verifier,
        "certificate": deepcopy(dict(certificate or {})),
        "limitations": list(dict.fromkeys(str(item) for item in limitations)),
    }


__all__ = ["CERTIFICATE_VERSION", "VerificationLevel", "verification_record"]
