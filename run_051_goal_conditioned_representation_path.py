"""Run 051: goal-conditioned representation path.

Tests whether answer support has a useful *direction* along a preregistered,
answer-blind sequence of increasingly goal-focused representations.

The path is:
    visible source -> layout -> structural -> goal scope

All six views from Run 048 are still scored so the retained contrastive selector
is an in-run control. Correctness is never visible to a transform or selector.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from offline_reasoning_battery import build_battery, sha256_json
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import _score_values
from run_039_exact_representation_operators import operators_for_case
from run_048_contrastive_representation_support import (
    ALL_NAMES,
    OPERATOR_NAMES,
    SIBLING_NAMES,
    build_views,
)

RUN_VERSION = "helix-goal-conditioned-representation-path-001.0"
PATH_NAMES = ("primary", "layout", "familiar", "goal-scope")
GOAL_OPERATOR_BY_FAMILY = {
    "relational-composition": "coordinate_lift",
    "abstract-transformation": "finite_exact_reencoding",
    "relational-matrix": "reversible_quotient",
    "grounded-planning": "coordinate_lift",
    "stack-language": "coordinate_lift",
}


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def _mean(vectors: list[list[float]]) -> list[float]:
    width = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(width)]


def _slope(values: list[float]) -> float:
    # OLS slope for x=[0,1,2,3]. mean(x)=1.5, sum((x-mean)^2)=5.
    if len(values) != 4:
        raise ValueError("path slope requires four stages")
    return sum((x - 1.5) * y for x, y in enumerate(values)) / 5.0


def _euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def goal_scope_view(case: dict[str, Any]) -> tuple[str, tuple[str, list[str]]]:
    desired = GOAL_OPERATOR_BY_FAMILY[case["family"]]
    operators = operators_for_case(case["family"], case["surfaces"])
    matches = [row for row in operators if row[0] == desired]
    if len(matches) != 1:
        raise AssertionError((case["family"], desired, [row[0] for row in operators]))
    op_id, prompt, choices = matches[0]
    return op_id, (prompt + "\nAnswer value:", choices)


def build_path_and_views(case: dict[str, Any]) -> tuple[dict[str, tuple[str, list[str]]], dict[str, str]]:
    views = build_views(case)
    op_id, goal_view = goal_scope_view(case)

    operators = operators_for_case(case["family"], case["surfaces"])
    op_names = {operators[i][0]: f"operator-{i + 1}" for i in range(len(operators))}
    goal_source_name = op_names[op_id]

    path = {
        "primary": views["primary"],
        "layout": views["layout"],
        "familiar": views["familiar"],
        "goal-scope": goal_view,
    }

    # The goal-scope view must be byte-identical to its already-scored exact view.
    if goal_view != views[goal_source_name]:
        raise AssertionError((case["case_id"], goal_source_name, op_id))

    return path, {
        "goal_operator_id": op_id,
        "goal_operator_view": goal_source_name,
    }


def self_test(seed: int = 20261118, cases_per_family: int = 2) -> dict[str, Any]:
    cases = build_battery(seed, cases_per_family)
    counts = Counter()
    digests = []
    operators = Counter()

    for case in cases:
        views = build_views(case)
        path, meta = build_path_and_views(case)
        if tuple(path) != PATH_NAMES:
            raise AssertionError((case["case_id"], tuple(path)))
        if tuple(views) != ALL_NAMES:
            raise AssertionError((case["case_id"], tuple(views)))

        operators[f"{case['family']}/{meta['goal_operator_id']}"] += 1
        for name, (prompt, choices) in path.items():
            if len(choices) != 4 or not prompt.endswith("Answer value:"):
                raise AssertionError((case["case_id"], name))
            counts[f"{case['family']}/{name}"] += 1
            digests.append(hashlib.sha256(json.dumps({
                "family": case["family"],
                "name": name,
                "prompt": prompt,
                "choices": choices,
            }, sort_keys=True).encode()).hexdigest())

    return {
        "schema_version": RUN_VERSION,
        "cases": len(cases),
        "path_stages": list(PATH_NAMES),
        "goal_operator_counts": dict(sorted(operators.items())),
        "path_instance_counts": dict(sorted(counts.items())),
        "transform_digest": hashlib.sha256("|".join(digests).encode()).hexdigest(),
        "selector_correctness_input": False,
        "learned_parameters": 0,
    }


def _summary(rows: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    by_family: dict[str, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    correct = 0
    for row in rows:
        ok = row["selectors"][selector] == row["correct_index"]
        correct += int(ok)
        by_family[row["family"]].append(ok)
        by_difficulty[row["difficulty"]].append(ok)
    return {
        "cases": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "by_family": {k: sum(v) / len(v) for k, v in sorted(by_family.items())},
        "by_difficulty": {k: sum(v) / len(v) for k, v in sorted(by_difficulty.items())},
    }


def _paired(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, int]:
    wins = regressions = ties = 0
    for row in rows:
        correct = row["correct_index"]
        l_ok = row["selectors"][left] == correct
        r_ok = row["selectors"][right] == correct
        if l_ok and not r_ok:
            wins += 1
        elif r_ok and not l_ok:
            regressions += 1
        else:
            ties += 1
    return {"wins": wins, "regressions": regressions, "ties": ties}


def evaluate(model_dir: str, seeds: list[int], cases_per_family: int) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()
    started = time.time()
    rows: list[dict[str, Any]] = []
    public_spec = []

    for seed in seeds:
        seed_spec = []
        for case in build_battery(seed, cases_per_family):
            uid = f"{seed}:{case['case_id']}"
            views = build_views(case)
            path, path_meta = build_path_and_views(case)
            scores = {
                name: _score_values(scorer, prompt, choices)
                for name, (prompt, choices) in views.items()
            }
            vectors = {name: [float(v) for v in row["scores"]] for name, row in scores.items()}
            predictions = {name: int(row["prediction"]) for name, row in scores.items()}

            goal_view_name = path_meta["goal_operator_view"]
            path_view_names = ("primary", "layout", "familiar", goal_view_name)
            path_vectors = [vectors[name] for name in path_view_names]

            candidate_paths = [
                [stage[c] for stage in path_vectors]
                for c in range(4)
            ]
            slopes = [_slope(values) for values in candidate_paths]
            reverse_slopes = [-value for value in slopes]
            endpoint = path_vectors[-1]
            path_mean = _mean(path_vectors)

            sibling_mean = _mean([vectors[name] for name in SIBLING_NAMES])
            operator_mean = _mean([vectors[name] for name in OPERATOR_NAMES])
            all_six_mean = _mean([vectors[name] for name in ALL_NAMES])
            contrastive = [
                operator_mean[i] - sibling_mean[i]
                for i in range(4)
            ]

            selectors = {
                "forward-path-slope": _argmax(slopes),
                "goal-scope-endpoint": _argmax(endpoint),
                "path-score-mean": _argmax(path_mean),
                "reverse-path-slope": _argmax(reverse_slopes),
                "contrastive-operator-minus-sibling": _argmax(contrastive),
                "sibling-score-mean": _argmax(sibling_mean),
                "all-six-score-mean": _argmax(all_six_mean),
                "layout-fixed": predictions["layout"],
            }

            correct = int(case["correct_index"])
            correct_support = candidate_paths[correct]
            monotonic_correct = all(
                correct_support[i + 1] >= correct_support[i]
                for i in range(3)
            )
            stage_deltas = [
                correct_support[i + 1] - correct_support[i]
                for i in range(3)
            ]

            segment_lengths = [
                _euclidean(path_vectors[i], path_vectors[i + 1])
                for i in range(3)
            ]
            path_length = sum(segment_lengths)
            direct_displacement = _euclidean(path_vectors[0], path_vectors[-1])
            tortuosity = (
                path_length / direct_displacement
                if direct_displacement > 1e-12
                else None
            )
            velocities = [
                [b - a for a, b in zip(path_vectors[i], path_vectors[i + 1], strict=True)]
                for i in range(3)
            ]
            second_differences = [
                [b - a for a, b in zip(velocities[i], velocities[i + 1], strict=True)]
                for i in range(2)
            ]
            second_difference_magnitude = sum(
                math.sqrt(sum(v * v for v in vec))
                for vec in second_differences
            )

            rows.append({
                "uid": uid,
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "goal_operator_id": path_meta["goal_operator_id"],
                "goal_operator_view": goal_view_name,
                "predictions": predictions,
                "selectors": selectors,
                "path": {
                    "view_names": list(path_view_names),
                    "candidate_support": candidate_paths,
                    "candidate_slopes": slopes,
                    "correct_support": correct_support,
                    "correct_support_slope": slopes[correct],
                    "correct_support_monotonic_nondecrease": monotonic_correct,
                    "correct_support_stage_deltas": stage_deltas,
                    "segment_lengths": segment_lengths,
                    "path_length": path_length,
                    "direct_endpoint_displacement": direct_displacement,
                    "tortuosity": tortuosity,
                    "second_difference_magnitude": second_difference_magnitude,
                },
                "contrastive_support": contrastive,
            })

            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "goal_operator_id": path_meta["goal_operator_id"],
                "view_sha256": {
                    name: hashlib.sha256(json.dumps({
                        "prompt": prompt,
                        "choices": choices,
                    }, sort_keys=True).encode()).hexdigest()
                    for name, (prompt, choices) in views.items()
                },
            })
        public_spec.append({"seed": seed, "cases": seed_spec})

    selector_names = tuple(rows[0]["selectors"])
    summaries = {name: _summary(rows, name) for name in selector_names}
    primary = "forward-path-slope"
    paired = {
        "vs-goal-scope-endpoint": _paired(rows, primary, "goal-scope-endpoint"),
        "vs-contrastive-run048": _paired(rows, primary, "contrastive-operator-minus-sibling"),
        "vs-path-mean": _paired(rows, primary, "path-score-mean"),
        "vs-reverse-path": _paired(rows, primary, "reverse-path-slope"),
    }

    monotonic = sum(row["path"]["correct_support_monotonic_nondecrease"] for row in rows)
    mean_correct_slope = sum(row["path"]["correct_support_slope"] for row in rows) / len(rows)
    mean_stage_deltas = [
        sum(row["path"]["correct_support_stage_deltas"][i] for row in rows) / len(rows)
        for i in range(3)
    ]
    mean_path_length = sum(row["path"]["path_length"] for row in rows) / len(rows)
    mean_direct = sum(row["path"]["direct_endpoint_displacement"] for row in rows) / len(rows)
    tortuosities = [row["path"]["tortuosity"] for row in rows if row["path"]["tortuosity"] is not None]
    mean_tortuosity = sum(tortuosities) / len(tortuosities) if tortuosities else None
    mean_second_diff = sum(row["path"]["second_difference_magnitude"] for row in rows) / len(rows)

    by_family = {}
    for family in sorted({row["family"] for row in rows}):
        subset = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "cases": len(subset),
            "forward_path_accuracy": sum(
                row["selectors"][primary] == row["correct_index"] for row in subset
            ) / len(subset),
            "goal_scope_endpoint_accuracy": sum(
                row["selectors"]["goal-scope-endpoint"] == row["correct_index"] for row in subset
            ) / len(subset),
            "contrastive_accuracy": sum(
                row["selectors"]["contrastive-operator-minus-sibling"] == row["correct_index"]
                for row in subset
            ) / len(subset),
            "correct_support_mean_slope": sum(
                row["path"]["correct_support_slope"] for row in subset
            ) / len(subset),
            "correct_support_monotonic_rate": sum(
                row["path"]["correct_support_monotonic_nondecrease"] for row in subset
            ) / len(subset),
        }

    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-goal-conditioned-path",
        "actor": {
            "model": "HuggingFaceTB/SmolLM2-360M",
            "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
            "weights_frozen": True,
        },
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(rows),
        "source_battery_sha256": sha256_json(public_spec),
        "path_stages": list(PATH_NAMES),
        "goal_operator_by_family": GOAL_OPERATOR_BY_FAMILY,
        "selectors": summaries,
        "paired": paired,
        "path_diagnostics": {
            "correct_support_monotonic_cases": monotonic,
            "correct_support_monotonic_rate": monotonic / len(rows),
            "mean_correct_support_slope": mean_correct_slope,
            "mean_correct_support_stage_deltas": mean_stage_deltas,
            "mean_score_space_path_length": mean_path_length,
            "mean_direct_endpoint_displacement": mean_direct,
            "mean_tortuosity": mean_tortuosity,
            "mean_second_difference_magnitude": mean_second_diff,
            "by_family": by_family,
        },
        "resources": resources,
        "rows": rows,
        "claim_boundary": (
            "Fresh public procedural test of a preregistered answer-blind representation path. "
            "The path is a candidate goal-conditioned ordering, not a proven geodesic or cognitive model. "
            "Correctness is used only after selection for evaluation. No promotion, AGI, human-cognition, "
            "or thermodynamic claim follows from this run."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--seeds", default="20261118,20261119,20261120")
    parser.add_argument("--cases-per-family", type=int, default=6)
    parser.add_argument("--output", default="run-051-goal-conditioned-representation-path.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return

    if not args.model_dir or not args.model_file or not args.expected_sha256:
        raise SystemExit("model-dir, model-file, and expected-sha256 are required")

    model_path = Path(args.model_file)
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    result = evaluate(
        args.model_dir,
        [int(v) for v in args.seeds.split(",") if v.strip()],
        args.cases_per_family,
    )
    result["actor"].update({
        "model_file_sha256": observed,
        "model_bytes": model_path.stat().st_size,
        "dtype": "float32",
        "device": "cpu",
    })
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema_version": result["schema_version"],
        "cases": result["cases"],
        "selectors": result["selectors"],
        "paired": result["paired"],
        "path_diagnostics": result["path_diagnostics"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
