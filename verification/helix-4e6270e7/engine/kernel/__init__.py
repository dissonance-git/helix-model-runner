"""Canonical Helix semantic kernel.

The kernel owns stable identity, provider-neutral semantic contracts, and
verification records. Intermediate-representation transformations belong to
``engine.ir``; keeping that machinery out of the kernel prevents the semantic
foundation from depending upward on one representation layer.
"""
from .identity import canonical_json, digest, stable_id
from .semantics import (
    AVAILABILITY,
    EQUIVALENCE_VERSION,
    OBLIGATION_VERSION,
    REPRESENTATION_VERSION,
    continuation_equivalence,
    equivalence,
    obligation,
    representation,
)
from .verification import CERTIFICATE_VERSION, VerificationLevel, verification_record

__all__ = [
    "AVAILABILITY",
    "CERTIFICATE_VERSION",
    "EQUIVALENCE_VERSION",
    "OBLIGATION_VERSION",
    "REPRESENTATION_VERSION",
    "VerificationLevel",
    "canonical_json",
    "continuation_equivalence",
    "digest",
    "equivalence",
    "obligation",
    "representation",
    "stable_id",
    "verification_record",
]
