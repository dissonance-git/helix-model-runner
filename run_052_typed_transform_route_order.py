"""Run 052: typed transform route order.

Fresh follow-up to Run 051. The model weights and task generator are frozen.
Transforms and selectors are answer-blind. This experiment asks whether route
direction, exact COMPOSE, and an explicit early STOP control matter more than a
monotone "keep focusing" assumption.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from offline_reasoning_battery import build_battery, sha256_json
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import build_representations, _score_values
from run_039_exact_representation_operators import (
    compose,
    operators_for_case,
    strip_answer_marker,
)

RUN_VERSION = "helix-typed-transform-route-order-001.0"
SEEDS_DEFAULT = "20261201,20261202"
CASES_PER_FAMILY_DEFAULT = 4

GOAL_OPERATOR_BY_FAMILY = {
    "relational-composition": "coordinate_lift",
    "abstract-transformation": "finite_exact_reencoding",
    "relational-matrix": "reversible_quotient",
    "grounded-planning": "coordinate_lift",
    "stack-language": "coordinate_lift",
}

BASIS_BY_OPERATOR = {
    "coordinate_lift": "LIFT",
    "finite_exact_reencoding": "REPROJECT",
    "reversible_quotient": "PROJECT",
    "topology_incidence_reencoding": "REPROJECT",
    "factor_product_decomposition": "FACTOR",
}

PROGRAMS = {
    "forward": ("primary", "source", "familiar", "goal"),
    "reverse": ("goal", "familiar", "source", "primary"),
    "goal-compose": ("source", "goal", "source_goal"),
    "other-compose": ("source", "other", "source_other"),
}


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def _mean(vectors: list[list[float]]) -> list[float]:
    width = len(vectors[0])
    return [sum(row[i] for row in vectors) / len(vectors) for i in range(width)]


def _slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        raise ValueError("route slope requires at least two stages")
    mean_x = (n - 1) / 2.0
    denom = sum((x - mean_x) ** 2 for x in range(n))
    return sum((x - mean_x) * y for x, y in enumerate(values)) / denom


def _euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _case_views(case: dict[str, Any]) -> tuple[dict[str, tuple[str, list[str]]], dict[str, Any]]:
    canonical = build_representations(case["family"], case["surfaces"])
    source_prompt, source_choices = canonical["layout"]
    familiar_prompt, familiar_choices = canonical["familiar"]
    source_stem = strip_answer_marker(source_prompt)

    surface_name = sorted(case["surfaces"])[0]
    from offline_reasoning_content_probe import surface_task_and_choices
    primary_prompt, primary_choices = surface_task_and_choices(case["surfaces"][surface_name])

    operators = operators_for_case(case["family"], case["surfaces"])
    if len(operators) != 2:
        raise AssertionError((case["case_id"], len(operators)))
    goal_id = GOAL_OPERATOR_BY_FAMILY[case["family"]]
    goal_rows = [row for row in operators if row[0] == goal_id]
    if len(goal_rows) != 1:
        raise AssertionError((case["family"], goal_id, [row[0] for row in operators]))
    goal = goal_rows[0]
    other = next(row for row in operators if row[0] != goal_id)

    goal_name, goal_stem, goal_choices = goal
    other_name, other_stem, other_choices = other

    views = {
        "primary": (primary_prompt, primary_choices),
        "source": (source_prompt, source_choices),
        "familiar": (familiar_prompt, familiar_choices),
        "goal": (goal_stem + "\nAnswer value:", goal_choices),
        "other": (other_stem + "\nAnswer value:", other_choices),
        "source_goal": (compose(source_stem, goal_stem), goal_choices),
        "source_other": (compose(source_stem, other_stem), other_choices),
    }
    meta = {
        "goal_operator_id": goal_name,
        "goal_basis_operator": BASIS_BY_OPERATOR[goal_name],
        "other_operator_id": other_name,
        "other_basis_operator": BASIS_BY_OPERATOR[other_name],
        "goal_compose_basis": "COMPOSE",
        "other_compose_basis": "COMPOSE",
    }
    return views, meta


def _summary(rows: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    correct = 0
    by_family: dict[str, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
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


def self_test(seed: int = 20261201, cases_per_family: int = 1) -> dict[str, Any]:
    rows = []
    for case in build_battery(seed, cases_per_family):
        views, meta = _case_views(case)
        if tuple(views) != (
            "primary", "source", "familiar", "goal", "other",
            "source_goal", "source_other",
        ):
            raise AssertionError(tuple(views))
        for name, (prompt, choices) in views.items():
            if len(choices) != 4 or not prompt.endswith("Answer value:"):
                raise AssertionError((case["case_id"], name))
        rows.append({
            "case_id": case["case_id"],
            "family": case["family"],
            **meta,
            "view_sha256": {
                name: hashlib.sha256(
                    json.dumps({"prompt": prompt, "choices": choices}, sort_keys=True).encode()
                ).hexdigest()
                for name, (prompt, choices) in views.items()
            },
        })
    return {
        "schema_version": RUN_VERSION,
        "status": "self-test-pass",
        "cases": len(rows),
        "programs": {name: list(stages) for name, stages in PROGRAMS.items()},
        "basis_by_operator": BASIS_BY_OPERATOR,
        "rows": rows,
        "correctness_visible_to_transform_or_selector": False,
        "learned_parameters": 0,
    }


def evaluate(model_dir: str, seeds: list[int], cases_per_family: int) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()
    started = time.time()
    rows: list[dict[str, Any]] = []
    public_spec = []

    for seed in seeds:
        seed_spec = []
        for case in build_battery(seed, cases_per_family):
            views, meta = _case_views(case)
            scores = {
                name: _score_values(scorer, prompt, choices)
                for name, (prompt, choices) in views.items()
            }
            vectors = {
                name: [float(value) for value in scored["scores"]]
                for name, scored in scores.items()
            }
            predictions = {name: int(scored["prediction"]) for name, scored in scores.items()}

            selectors: dict[str, int] = {
                "stop-layout": predictions["source"],
                "goal-endpoint": predictions["goal"],
                "other-endpoint": predictions["other"],
                "source-goal-endpoint": predictions["source_goal"],
                "source-other-endpoint": predictions["source_other"],
            }
            program_diagnostics = {}
            correct = int(case["correct_index"])

            for program_name, stages in PROGRAMS.items():
                stage_vectors = [vectors[name] for name in stages]
                candidate_paths = [
                    [stage[candidate] for stage in stage_vectors]
                    for candidate in range(4)
                ]
                slopes = [_slope(path) for path in candidate_paths]
                means = _mean(stage_vectors)
                selectors[f"{program_name}-slope"] = _argmax(slopes)
                selectors[f"{program_name}-mean"] = _argmax(means)

                correct_support = candidate_paths[correct]
                deltas = [
                    correct_support[i + 1] - correct_support[i]
                    for i in range(len(correct_support) - 1)
                ]
                segment_lengths = [
                    _euclidean(stage_vectors[i], stage_vectors[i + 1])
                    for i in range(len(stage_vectors) - 1)
                ]
                direct = _euclidean(stage_vectors[0], stage_vectors[-1])
                program_diagnostics[program_name] = {
                    "stages": list(stages),
                    "candidate_slopes": slopes,
                    "correct_support": correct_support,
                    "correct_support_slope": slopes[correct],
                    "correct_support_deltas": deltas,
                    "correct_support_monotonic_nondecrease": all(delta >= 0 for delta in deltas),
                    "path_length": sum(segment_lengths),
                    "direct_endpoint_displacement": direct,
                    "tortuosity": sum(segment_lengths) / direct if direct > 1e-12 else None,
                }

            rows.append({
                "uid": f"{seed}:{case['case_id']}",
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "transform_meta": meta,
                "predictions": predictions,
                "selectors": selectors,
                "programs": program_diagnostics,
            })
            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "transform_meta": meta,
                "view_sha256": {
                    name: hashlib.sha256(
                        json.dumps({"prompt": prompt, "choices": choices}, sort_keys=True).encode()
                    ).hexdigest()
                    for name, (prompt, choices) in views.items()
                },
            })
        public_spec.append({"seed": seed, "cases": seed_spec})

    selector_names = tuple(rows[0]["selectors"])
    summaries = {name: _summary(rows, name) for name in selector_names}
    paired = {
        "reverse-vs-forward-slope": _paired(rows, "reverse-slope", "forward-slope"),
        "goal-compose-slope-vs-stop-layout": _paired(rows, "goal-compose-slope", "stop-layout"),
        "goal-compose-mean-vs-stop-layout": _paired(rows, "goal-compose-mean", "stop-layout"),
        "source-goal-endpoint-vs-goal-endpoint": _paired(
            rows, "source-goal-endpoint", "goal-endpoint"
        ),
    }

    reverse = paired["reverse-vs-forward-slope"]
    compose_stop = paired["goal-compose-mean-vs-stop-layout"]
    compose_goal = paired["source-goal-endpoint-vs-goal-endpoint"]
    decisions = {
        "run051_reversal_replicated": (
            summaries["reverse-slope"]["accuracy"] > summaries["forward-slope"]["accuracy"]
            and reverse["wins"] > reverse["regressions"]
        ),
        "continue_past_layout_earned": (
            summaries["goal-compose-mean"]["accuracy"] > summaries["stop-layout"]["accuracy"]
            and compose_stop["wins"] > compose_stop["regressions"]
        ),
        "compose_specific_gain_earned": (
            summaries["source-goal-endpoint"]["accuracy"] > summaries["goal-endpoint"]["accuracy"]
            and compose_goal["wins"] > compose_goal["regressions"]
        ),
    }

    diagnostics = {}
    for program_name in PROGRAMS:
        subset = [row["programs"][program_name] for row in rows]
        diagnostics[program_name] = {
            "mean_correct_support_slope": sum(row["correct_support_slope"] for row in subset) / len(subset),
            "correct_support_monotonic_rate": sum(
                row["correct_support_monotonic_nondecrease"] for row in subset
            ) / len(subset),
            "mean_path_length": sum(row["path_length"] for row in subset) / len(subset),
            "mean_direct_endpoint_displacement": sum(
                row["direct_endpoint_displacement"] for row in subset
            ) / len(subset),
            "mean_tortuosity": (
                sum(row["tortuosity"] for row in subset if row["tortuosity"] is not None)
                / sum(row["tortuosity"] is not None for row in subset)
            ),
        }

    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started
    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-typed-transform-route-order",
        "actor": {
            "model": "HuggingFaceTB/SmolLM2-360M",
            "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
            "weights_frozen": True,
        },
        "helix_transform_basis_source_commit": "670962597664adf988d0c606e46772be5a25b963",
        "instantiated_basis": ["PROJECT", "REPROJECT", "LIFT", "FACTOR", "COMPOSE"],
        "uninstantiated_basis": ["LOWER", "DUALIZE", "RELAX", "TIGHTEN"],
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(rows),
        "source_battery_sha256": sha256_json(public_spec),
        "programs": {name: list(stages) for name, stages in PROGRAMS.items()},
        "selectors": summaries,
        "paired": paired,
        "decisions": decisions,
        "program_diagnostics": diagnostics,
        "resources": resources,
        "rows": rows,
        "claim_boundary": (
            "Fresh public actor probe of a frozen projection of the Helix typed transform basis. "
            "The actor is unpromoted; transforms/selectors are answer-blind; correctness is used "
            "only after selection for evaluation. This does not establish universal transform "
            "ordering, compiler correctness, protected promotion evidence, or human cognition."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--seeds", default=SEEDS_DEFAULT)
    parser.add_argument("--cases-per-family", type=int, default=CASES_PER_FAMILY_DEFAULT)
    parser.add_argument("--output", default="run-052-typed-transform-route-order.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    if not args.model_dir or not args.model_file or not args.expected_sha256:
        raise SystemExit("model-dir, model-file and expected-sha256 are required")

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
        [int(value) for value in args.seeds.split(",") if value.strip()],
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
        "decisions": result["decisions"],
        "program_diagnostics": result["program_diagnostics"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
