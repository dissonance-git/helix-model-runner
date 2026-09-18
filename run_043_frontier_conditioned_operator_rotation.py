"""Run 043: frontier-conditioned exact-operator rotation.

Evaluate the Helix API representation-frontier contract as an answer-blind compute
allocation rule. The full fresh candidate bank is measured once, then routing
policies are replayed offline at identical operator-case budgets.

Important boundary: candidate-bank coverage is not deployable answer accuracy.
No selector/verifier is assumed.
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
from offline_reasoning_content_probe import surface_task_and_choices
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import build_representations, _score_values
from run_039_exact_representation_operators import operators_for_case

RUN_VERSION = "helix-frontier-conditioned-operator-rotation-001.0"
BUDGET_FRACTIONS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
RANDOM_SEEDS = tuple(range(16))


def stable_key(uid: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}|{uid}".encode()).hexdigest(), 16)


def score_cost(scorer: AttackScorer, prompt: str, choices: list[str]) -> dict[str, int]:
    """Mirror the two score_choices calls performed by _score_values."""
    tok = scorer.tokenizer
    prompt_tokens = len(tok(prompt, add_special_tokens=True).input_ids)
    prior_tokens = len(tok("Answer value:", add_special_tokens=True).input_ids)
    choice_tokens = sum(
        len(tok(" " + str(choice), add_special_tokens=False).input_ids)
        for choice in choices
    )
    return {
        "score_forward_batches": 2,
        "score_prompt_tokens": prompt_tokens + prior_tokens,
        "score_choice_tokens": 2 * choice_tokens,
    }


def add_cost(items: list[dict[str, int]]) -> dict[str, int]:
    return {
        key: sum(item[key] for item in items)
        for key in ("score_forward_batches", "score_prompt_tokens", "score_choice_tokens")
    }


def policy_order(rows: list[dict[str, Any]], policy: str, salt: str = "0") -> list[int]:
    indices = list(range(len(rows)))
    tie = lambda i: stable_key(rows[i]["uid"], salt)
    if policy == "random":
        return sorted(indices, key=tie)
    if policy == "saturation_first":
        return sorted(indices, key=lambda i: (rows[i]["sibling_distinct_predictions"], tie(i)))
    if policy == "disagreement_first":
        return sorted(indices, key=lambda i: (-rows[i]["sibling_distinct_predictions"], tie(i)))
    if policy == "unanimous_first":
        return sorted(indices, key=lambda i: (not rows[i]["sibling_unanimous"], tie(i)))
    if policy == "split_first":
        return sorted(indices, key=lambda i: (rows[i]["sibling_unanimous"], tie(i)))
    raise ValueError(policy)


def evaluate_order(
    rows: list[dict[str, Any]],
    order: list[int],
    base_cost: dict[str, int],
    fractions: tuple[float, ...] = BUDGET_FRACTIONS,
) -> list[dict[str, Any]]:
    n = len(rows)
    base_covered = sum(bool(r["sibling_has_correct"]) for r in rows)
    curve = []
    for fraction in fractions:
        selected_count = min(n, int(math.floor(fraction * n + 0.5)))
        selected = order[:selected_count]
        selected_set = set(selected)
        rescued = sum(
            bool(row["exact_operator_rescues_sibling_failure"])
            for i, row in enumerate(rows)
            if i in selected_set
        )
        operator_cost = add_cost([rows[i]["operator_pair_cost"] for i in selected])
        total_cost = {
            key: base_cost[key] + operator_cost[key]
            for key in base_cost
        }
        curve.append({
            "operator_case_fraction": fraction,
            "operator_cases_selected": selected_count,
            "candidate_coverage_correct": base_covered + rescued,
            "candidate_coverage": (base_covered + rescued) / n,
            "new_operator_rescues": rescued,
            "rescues_per_selected_case": (rescued / selected_count) if selected_count else 0.0,
            "counterfactual_cost": total_cost,
        })
    return curve


def trapezoid_auc(curve: list[dict[str, Any]]) -> float:
    auc = 0.0
    for left, right in zip(curve, curve[1:]):
        dx = right["operator_case_fraction"] - left["operator_case_fraction"]
        auc += dx * (left["candidate_coverage"] + right["candidate_coverage"]) / 2.0
    return auc


def average_random_curves(curves: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for j in range(len(curves[0])):
        entries = [curve[j] for curve in curves]
        template = dict(entries[0])
        template["candidate_coverage_correct_mean"] = sum(
            e["candidate_coverage_correct"] for e in entries
        ) / len(entries)
        template["candidate_coverage_mean"] = sum(e["candidate_coverage"] for e in entries) / len(entries)
        template["candidate_coverage_min"] = min(e["candidate_coverage"] for e in entries)
        template["candidate_coverage_max"] = max(e["candidate_coverage"] for e in entries)
        template["new_operator_rescues_mean"] = sum(e["new_operator_rescues"] for e in entries) / len(entries)
        template.pop("candidate_coverage_correct", None)
        template.pop("candidate_coverage", None)
        template.pop("new_operator_rescues", None)
        template.pop("rescues_per_selected_case", None)
        out.append(template)
    return out


def summarize_frontier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_distinct: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_distinct[int(row["sibling_distinct_predictions"])].append(row)
        by_family[row["family"]].append(row)

    def stats(group: list[dict[str, Any]]) -> dict[str, Any]:
        rescues = sum(bool(r["exact_operator_rescues_sibling_failure"]) for r in group)
        sibling_failures = sum(not bool(r["sibling_has_correct"]) for r in group)
        return {
            "cases": len(group),
            "sibling_failures": sibling_failures,
            "exact_operator_rescues": rescues,
            "rescue_per_operator_case": rescues / len(group) if group else 0.0,
            "rescue_given_sibling_failure": rescues / sibling_failures if sibling_failures else 0.0,
        }

    return {
        "by_distinct_prediction_count": {
            str(k): stats(v) for k, v in sorted(by_distinct.items())
        },
        "by_family": {k: stats(v) for k, v in sorted(by_family.items())},
    }


def evaluate(model_dir: str, seeds: list[int], cases_per_family: int) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()
    started = time.time()
    rows = []
    spec = []

    for seed in seeds:
        cases = build_battery(seed, cases_per_family)
        seed_spec = []
        for case in cases:
            canonical = build_representations(case["family"], case["surfaces"])
            surface_names = sorted(case["surfaces"])
            primary_prompt, primary_choices = surface_task_and_choices(case["surfaces"][surface_names[0]])
            secondary_prompt, secondary_choices = surface_task_and_choices(case["surfaces"][surface_names[1]])
            op1_id, op1_stem, op1_choices = operators_for_case(case["family"], case["surfaces"])[0]
            op2_id, op2_stem, op2_choices = operators_for_case(case["family"], case["surfaces"])[1]

            prompts = {
                "primary": (primary_prompt, primary_choices),
                "secondary": (secondary_prompt, secondary_choices),
                "layout": canonical["layout"],
                "familiar": canonical["familiar"],
                "op1": (op1_stem + "\nAnswer value:", op1_choices),
                "op2": (op2_stem + "\nAnswer value:", op2_choices),
            }
            scores = {
                name: _score_values(scorer, prompt, list(choices))
                for name, (prompt, choices) in prompts.items()
            }
            preds = {name: int(value["prediction"]) for name, value in scores.items()}
            correct = int(case["correct_index"])
            sibling_names = ("primary", "secondary", "layout", "familiar")
            sibling_predictions = [preds[name] for name in sibling_names]
            sibling_has_correct = any(preds[name] == correct for name in sibling_names)
            exact_has_correct = preds["op1"] == correct or preds["op2"] == correct
            costs = {
                name: score_cost(scorer, prompt, list(choices))
                for name, (prompt, choices) in prompts.items()
            }
            operator_pair_cost = add_cost([costs["op1"], costs["op2"]])
            sibling_cost = add_cost([costs[name] for name in sibling_names])

            rows.append({
                "uid": f"{seed}:{case['case_id']}",
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "operator_ids": [op1_id, op2_id],
                **{f"{name}_prediction": pred for name, pred in preds.items()},
                "sibling_distinct_predictions": len(set(sibling_predictions)),
                "sibling_unanimous": len(set(sibling_predictions)) == 1,
                "sibling_has_correct": sibling_has_correct,
                "exact_operator_has_correct": exact_has_correct,
                "exact_operator_rescues_sibling_failure": (not sibling_has_correct) and exact_has_correct,
                "sibling_cost": sibling_cost,
                "operator_pair_cost": operator_pair_cost,
                "prompt_sha256": {
                    name: hashlib.sha256(prompt.encode()).hexdigest()
                    for name, (prompt, _choices) in prompts.items()
                },
            })
            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "operator_ids": [op1_id, op2_id],
                "source_surface_sha256": {
                    name: hashlib.sha256(prompt.encode()).hexdigest()
                    for name, prompt in case["surfaces"].items()
                },
            })
        spec.append({"seed": seed, "cases": seed_spec})

    base_cost = add_cost([row["sibling_cost"] for row in rows])
    policies = {}
    for policy in ("saturation_first", "disagreement_first", "unanimous_first", "split_first"):
        curve = evaluate_order(rows, policy_order(rows, policy, "policy"), base_cost)
        policies[policy] = {"curve": curve, "auc": trapezoid_auc(curve)}

    random_curves = [
        evaluate_order(rows, policy_order(rows, "random", f"random-{seed}"), base_cost)
        for seed in RANDOM_SEEDS
    ]
    random_aucs = [trapezoid_auc(curve) for curve in random_curves]
    policies["random"] = {
        "seeds": list(RANDOM_SEEDS),
        "curve": average_random_curves(random_curves),
        "auc_mean": sum(random_aucs) / len(random_aucs),
        "auc_min": min(random_aucs),
        "auc_max": max(random_aucs),
    }

    base_covered = sum(bool(r["sibling_has_correct"]) for r in rows)
    full_covered = sum(bool(r["sibling_has_correct"] or r["exact_operator_has_correct"]) for r in rows)
    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-frontier-routing-calibration",
        "api_contract": {
            "helix_api_commit": "8640aaf4",
            "consumes": ["research.representation-search", "research.representation-frontier"],
        },
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(rows),
        "source_battery_sha256": sha256_json(spec),
        "coverage": {
            "sibling_four_view_correct": base_covered,
            "sibling_four_view_coverage": base_covered / len(rows),
            "sibling_plus_exact_operator_correct": full_covered,
            "sibling_plus_exact_operator_coverage": full_covered / len(rows),
            "strict_exact_operator_rescues": full_covered - base_covered,
        },
        "frontier": summarize_frontier(rows),
        "policies": policies,
        "measurement_execution_resources": resources,
        "cost_accounting": {
            "base_four_sibling_views": base_cost,
            "extra_unit": "both exact family-specific operator views for one selected case",
            "note": "All operator outcomes are measured once for offline policy comparison; each policy curve is counterfactually charged only for selected operator-case evaluations.",
        },
        "rows": rows,
        "claim_boundary": (
            "Fresh public candidate-acquisition diagnostic only. Unanimity/disagreement are "
            "answer-blind routing features, not correctness evidence. Candidate-bank coverage "
            "is not deployable answer accuracy; no selector/verifier is assumed."
        ),
    }


def self_test() -> dict[str, Any]:
    rows = [
        {"uid": "a", "sibling_distinct_predictions": 1, "sibling_unanimous": True, "sibling_has_correct": False, "exact_operator_rescues_sibling_failure": True, "operator_pair_cost": {"score_forward_batches": 4, "score_prompt_tokens": 10, "score_choice_tokens": 8}},
        {"uid": "b", "sibling_distinct_predictions": 2, "sibling_unanimous": False, "sibling_has_correct": True, "exact_operator_rescues_sibling_failure": False, "operator_pair_cost": {"score_forward_batches": 4, "score_prompt_tokens": 11, "score_choice_tokens": 8}},
        {"uid": "c", "sibling_distinct_predictions": 4, "sibling_unanimous": False, "sibling_has_correct": False, "exact_operator_rescues_sibling_failure": True, "operator_pair_cost": {"score_forward_batches": 4, "score_prompt_tokens": 12, "score_choice_tokens": 8}},
        {"uid": "d", "sibling_distinct_predictions": 1, "sibling_unanimous": True, "sibling_has_correct": False, "exact_operator_rescues_sibling_failure": False, "operator_pair_cost": {"score_forward_batches": 4, "score_prompt_tokens": 13, "score_choice_tokens": 8}},
    ]
    base = {"score_forward_batches": 32, "score_prompt_tokens": 80, "score_choice_tokens": 64}
    sat = policy_order(rows, "saturation_first", "test")
    dis = policy_order(rows, "disagreement_first", "test")
    if any(rows[i]["sibling_distinct_predictions"] != 1 for i in sat[:2]):
        raise AssertionError("saturation policy failed")
    if rows[dis[0]]["sibling_distinct_predictions"] != 4:
        raise AssertionError("disagreement policy failed")
    curve = evaluate_order(rows, sat, base, (0.0, 0.5, 1.0))
    if curve[0]["candidate_coverage_correct"] != 1:
        raise AssertionError("baseline coverage failed")
    if curve[-1]["candidate_coverage_correct"] != 3:
        raise AssertionError("full coverage failed")
    if curve[-1]["counterfactual_cost"]["score_forward_batches"] != 48:
        raise AssertionError("cost accounting failed")
    return {
        "saturation_order": [rows[i]["uid"] for i in sat],
        "disagreement_order": [rows[i]["uid"] for i in dis],
        "curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--seeds", default="20261102,20261103,20261104")
    parser.add_argument("--cases-per-family", type=int, default=6)
    parser.add_argument("--output", default="run-043-frontier-conditioned-operator-rotation.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    if not args.model_dir or not args.model_file or not args.expected_sha256:
        raise SystemExit("model-dir, model-file and expected-sha256 are required")

    model_path = Path(args.model_file)
    observed = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    result = evaluate(
        args.model_dir,
        [int(value) for value in args.seeds.split(",") if value.strip()],
        args.cases_per_family,
    )
    result["actor"] = {
        "model": "HuggingFaceTB/SmolLM2-360M",
        "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        "model_file_sha256": observed,
        "model_bytes": model_path.stat().st_size,
        "dtype": "float32",
        "device": "cpu",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema_version": result["schema_version"],
        "coverage": result["coverage"],
        "frontier": result["frontier"],
        "policies": result["policies"],
        "measurement_execution_resources": result["measurement_execution_resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
