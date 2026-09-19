"""RULE30-077 exact reverse zero-tail basin replay.

This is a finite exact graph computation for cyclic widths 4, 8, 16, and 32.
It starts at the zero tail and reverses the legal right-history relation, so it
visits only states that can actually reach the zero tail instead of enumerating
the full four-column state space.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WIDTHS = (4, 8, 16, 32)
EXPECTED = {
    4: {"states": 42, "edges": 52, "max_depth": 6},
    8: {"states": 74, "edges": 84, "max_depth": 7},
    16: {"states": 2122, "edges": 2132, "max_depth": 17},
    32: {"states": 171786, "edges": 171796, "max_depth": 106},
}


def shift(word: int, n: int, q: int) -> int:
    q %= n
    mask = (1 << n) - 1
    return word if q == 0 else ((word << q) | (word >> (n - q))) & mask


def word_period(word: int, n: int) -> int:
    for p in range(1, n + 1):
        if n % p == 0 and shift(word, n, p) == word:
            return p
    return n


def forcing(a: int, b: int, c: int, d: int, n: int) -> int:
    mask = (1 << n) - 1
    return (~(a ^ b) & mask) & (c | d)


def derivative(word: int, n: int) -> int:
    return word ^ shift(word, n, 1)


def inverse_derivative(target: int, n: int) -> tuple[int, ...]:
    """Solve D(a)=target exactly on a cyclic n-bit word.

    D has kernel {0, all-ones}. An image word has even parity, and every image
    word therefore has exactly two complementary preimages.
    """
    if target.bit_count() & 1:
        return ()
    mask = (1 << n) - 1
    bits = [(target >> i) & 1 for i in range(n)]
    preimage_bits = [0] * n
    for i in range(1, n):
        preimage_bits[i] = preimage_bits[i - 1] ^ bits[i]
    preimage = sum(bit << i for i, bit in enumerate(preimage_bits))
    if derivative(preimage, n) != target:
        raise AssertionError("cyclic derivative reconstruction failed")
    return preimage, preimage ^ mask


def reverse_predecessors(state: tuple[int, int, int, int], n: int):
    b, c, d, e = state
    target = forcing(b, c, d, e, n)
    return tuple((a, b, c, d) for a in inverse_derivative(target, n))


def zero_tail_basin(n: int):
    zero = (0, 0, 0, 0)
    queue = deque([zero])
    depth = {zero: 0}
    edges: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    while queue:
        state = queue.popleft()
        for predecessor in reverse_predecessors(state, n):
            edges.append((predecessor, state))
            if predecessor not in depth:
                depth[predecessor] = depth[state] + 1
                queue.append(predecessor)
    return depth, edges


def summarize_width(n: int) -> dict[str, object]:
    depth, edges = zero_tail_basin(n)
    states = list(depth)
    outdegree = Counter(source for source, _target in edges)
    exceptions = Counter()
    equal_full_failures = 0
    for state in states:
        periods = tuple(word_period(word, n) for word in state)
        h_period = word_period(forcing(*state, n), n)
        if h_period != max(periods[0], periods[1]):
            exceptions[(periods, h_period)] += 1
        if periods[0] == periods[1] == n and h_period < n:
            equal_full_failures += 1

    branch_states = [state for state, degree in outdegree.items() if degree > 1]
    branch_max_period = max(
        (word_period(word, n) for state in branch_states for word in state),
        default=0,
    )
    expected = EXPECTED[n]
    scalars = {
        "states": len(states),
        "edges": len(edges),
        "max_depth": max(depth.values()),
    }
    if scalars != expected:
        raise AssertionError({"width": n, "observed": scalars, "expected": expected})
    if equal_full_failures != 0:
        raise AssertionError({"width": n, "equal_full_failures": equal_full_failures})
    expected_exception = {(((4, 2, 1, 1)), 1): 4}
    if dict(exceptions) != expected_exception:
        raise AssertionError({"width": n, "exceptions": dict(exceptions)})
    if len(branch_states) != 10 or branch_max_period > 4:
        raise AssertionError(
            {
                "width": n,
                "branch_states": len(branch_states),
                "branch_max_period": branch_max_period,
            }
        )

    return {
        "width": n,
        **scalars,
        "equal_full_period_failures": equal_full_failures,
        "forcing_law_exception_count": sum(exceptions.values()),
        "forcing_law_exception_signatures": [
            {
                "periods": list(periods),
                "forcing_period": h_period,
                "count": count,
            }
            for (periods, h_period), count in sorted(exceptions.items())
        ],
        "branch_state_count": len(branch_states),
        "branch_max_component_period": branch_max_period,
        "outdegree_counts": {
            str(degree): count for degree, count in sorted(Counter(outdegree.values()).items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [summarize_width(n) for n in WIDTHS]
    payload = {
        "schema_version": "rule30-077-zero-tail-basin-001.0",
        "status": "PASS",
        "method": "exact reverse breadth-first closure from the all-zero tail",
        "rows": rows,
        "governance_test_results": [
            {"test_id": "T077-ZERO-BASIN-EXCLUSION", "outcome": "supports_prediction"},
            {"test_id": "T077-SEED-EXCEPTION-STABILITY", "outcome": "supports_prediction"},
            {"test_id": "T077-LOW-PERIOD-BRANCH-SKELETON", "outcome": "supports_prediction"},
        ],
        "claim_boundary": [
            "Exact finite cyclic widths 4, 8, 16, and 32 only.",
            "No all-dyadic theorem is claimed.",
            "State-count reduction is a representation reduction, not a measured runtime speedup claim.",
            "No Rule 30 center-column nonperiodicity or prize claim.",
        ],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
