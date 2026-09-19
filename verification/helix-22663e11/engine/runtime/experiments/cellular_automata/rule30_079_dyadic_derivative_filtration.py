"""RULE30-079 dyadic cyclic-derivative filtration.

This run separates a proved algebraic reduction from the still-open Rule 30
branch-skeleton implication.

For cyclic dyadic width n, write L for one cyclic shift and D = I + L over
GF(2). Then D^(2^r) = I + L^(2^r). Hence ker(D^4) is exactly the subspace of
words with period dividing four, and it has dimension four for every n >= 4.

The Rule 30 portion below only records finite exact evidence that zero-tail
branching is confined to this fixed core and that no zero-basin continuation
edge leaves it through widths 4, 8, 16, and 32.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from engine.runtime.experiments.cellular_automata.rule30_077_zero_tail_basin import (
    derivative,
    shift,
    zero_tail_basin,
)

ROOT = Path(__file__).resolve().parents[4]
ALGEBRA_WIDTHS = (4, 8, 16, 32, 64)
BASIN_WIDTHS = (4, 8, 16, 32)


def derivative_power(word: int, n: int, exponent: int) -> int:
    value = word
    for _ in range(exponent):
        value = derivative(value, n)
    return value


def gf2_rank(images: list[int]) -> int:
    """Rank of a GF(2) linear map from images of the standard basis."""
    pivots: dict[int, int] = {}
    for raw in images:
        value = int(raw)
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


def period4_lift(pattern: int, n: int) -> int:
    if n % 4:
        raise ValueError("period-4 lift requires width divisible by four")
    word = 0
    for index in range(n):
        word |= ((pattern >> (index % 4)) & 1) << index
    return word


def in_d4_core(word: int, n: int) -> bool:
    return derivative_power(word, n, 4) == 0


def state_in_d4_core(state: tuple[int, ...], n: int) -> bool:
    return all(in_d4_core(word, n) for word in state)


def verify_algebra_width(n: int) -> dict[str, object]:
    if n < 4 or n & (n - 1):
        raise ValueError("n must be a dyadic width at least four")

    powers = []
    q = 1
    while q <= n:
        # Linear maps agree on every vector iff they agree on a basis.
        identity_on_basis = all(
            derivative_power(1 << index, n, q)
            == ((1 << index) ^ shift(1 << index, n, q))
            for index in range(n)
        )
        if not identity_on_basis:
            raise AssertionError({"width": n, "power": q, "frobenius_identity": False})
        powers.append(q)
        q *= 2

    if not all(derivative_power(1 << index, n, n) == 0 for index in range(n)):
        raise AssertionError({"width": n, "nilpotent": False})

    d4_images = [derivative_power(1 << index, n, 4) for index in range(n)]
    rank_d4 = gf2_rank(d4_images)
    nullity_d4 = n - rank_d4
    if nullity_d4 != 4:
        raise AssertionError({"width": n, "nullity_d4": nullity_d4})

    lifts = {period4_lift(pattern, n) for pattern in range(16)}
    if len(lifts) != 16 or not all(in_d4_core(word, n) for word in lifts):
        raise AssertionError({"width": n, "period4_lift_certificate": False})

    # Since the kernel has dimension four it has exactly 16 words. The sixteen
    # distinct period-4 lifts already lie in the kernel, so they exhaust it.
    return {
        "width": n,
        "verified_frobenius_powers": powers,
        "D_to_width_is_zero": True,
        "rank_D4": rank_d4,
        "nullity_D4": nullity_d4,
        "kernel_D4_words": 1 << nullity_d4,
        "period4_lifts": len(lifts),
        "kernel_D4_equals_period_dividing_4": True,
    }


def verify_basin_width(n: int) -> dict[str, object]:
    depth, edges = zero_tail_basin(n)
    outdegree = Counter(source for source, _target in edges)
    branch = [state for state in depth if outdegree[state] > 1]
    branch_outside_core = [state for state in branch if not state_in_d4_core(state, n)]
    core_escape_edges = [
        (source, target)
        for source, target in edges
        if state_in_d4_core(source, n) and not state_in_d4_core(target, n)
    ]
    branch_depth_counts = Counter(depth[state] for state in branch)

    if len(branch) != 10:
        raise AssertionError({"width": n, "branch_count": len(branch)})
    if branch_outside_core:
        raise AssertionError({"width": n, "branch_outside_core": len(branch_outside_core)})
    if core_escape_edges:
        raise AssertionError({"width": n, "core_escape_edges": len(core_escape_edges)})
    if dict(sorted(branch_depth_counts.items())) != {3: 2, 4: 4, 5: 4}:
        raise AssertionError({"width": n, "branch_depth_counts": dict(branch_depth_counts)})

    return {
        "width": n,
        "basin_states": len(depth),
        "basin_edges": len(edges),
        "branch_states": len(branch),
        "branch_outside_D4_core": 0,
        "zero_basin_core_escape_edges": 0,
        "branch_depth_counts": {
            str(key): value for key, value in sorted(branch_depth_counts.items())
        },
    }


def build_result() -> dict[str, object]:
    algebra = [verify_algebra_width(n) for n in ALGEBRA_WIDTHS]
    basin = [verify_basin_width(n) for n in BASIN_WIDTHS]
    return {
        "schema_version": "rule30-dyadic-derivative-filtration-001.0",
        "status": "PASS",
        "proved_algebraic_reduction": {
            "operator": "D = I + cyclic-shift over GF(2)",
            "identity": "D^(2^r) = I + L^(2^r)",
            "dyadic_nilpotence": "D^n = 0 for n a power of two",
            "period_corollary": "D^(2^r)(x)=0 iff x has period dividing 2^r",
            "D4_kernel_dimension": 4,
            "D4_kernel_word_count": 16,
            "four_column_D4_core_tuple_count": 16 ** 4,
        },
        "algebra_checks": algebra,
        "finite_rule30_checks": basin,
        "all_dyadic_branch_theorem_reduction": {
            "obligation_A": "every zero-tail-basin branch state lies in (ker D^4)^4",
            "obligation_B": "every zero-basin continuation from a (ker D^4)^4 basin state remains in (ker D^4)^4",
            "consequence_if_A_and_B": (
                "the branch-bearing zero-basin core is the periodic lift of the fixed width-4 "
                "core, reducing arbitrary dyadic branching to a width-independent 65,536-tuple "
                "finite state space"
            ),
            "status": "open",
        },
        "claim_boundary": [
            "The dyadic derivative identities and D^4 kernel characterization are algebraic theorems.",
            "Branch confinement and zero-basin core closure are verified only at complete widths 4, 8, 16, and 32.",
            "Obligations A and B remain open for arbitrary dyadic width.",
            "No Rule 30 center-column nonperiodicity or prize claim follows.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_result()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "proved_algebraic_reduction": result["proved_algebraic_reduction"],
        "finite_rule30_checks": result["finite_rule30_checks"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
