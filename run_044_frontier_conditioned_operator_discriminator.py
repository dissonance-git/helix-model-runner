"""Run 044: frontier-conditioned exact-operator discriminator.

This run asks whether a small answer-blind selector can convert the candidate
coverage created by exact representation operators into selected-answer
accuracy. Candidate coverage and selected-answer accuracy are reported
separately. The frozen actor is never trained.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from offline_reasoning_battery import build_battery, sha256_json
from offline_reasoning_content_probe import surface_task_and_choices
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import build_representations, _score_values
from run_039_exact_representation_operators import operators_for_case

RUN_VERSION = "helix-frontier-conditioned-operator-discriminator-001.0"
FIT_SEEDS = (20261102, 20261103, 20261104)
VALIDATION_SEEDS = (20261105,)
TEST_SEEDS = (20261106, 20261107, 20261108)
RIDGE_STRENGTHS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
SIBLING_REPS = ("primary", "secondary", "layout", "familiar")
ALL_REPS = SIBLING_REPS + ("op1", "op2")
FAMILIES = (
    "abstract-transformation",
    "grounded-planning",
    "relational-composition",
    "relational-matrix",
    "stack-language",
)


def stable_key(uid: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}|{uid}".encode()).hexdigest(), 16)


def rank_of(scores: list[float], candidate: int) -> int:
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order.index(candidate)


def candidate_features(
    row: dict[str, Any],
    candidate: int,
    reps: tuple[str, ...],
    *,
    family_aware: bool,
) -> list[float]:
    features: list[float] = []
    candidate_scores = []
    predictions = []
    for rep in reps:
        scores = [float(v) for v in row["scores"][rep]]
        pred = int(row["predictions"][rep])
        candidate_scores.append(scores[candidate])
        predictions.append(pred)
        features.extend([
            scores[candidate],
            rank_of(scores, candidate) / 3.0,
            1.0 if pred == candidate else 0.0,
        ])

    vote_count = sum(pred == candidate for pred in predictions)
    features.extend([
        vote_count / len(reps),
        statistics.fmean(candidate_scores),
        max(candidate_scores),
        min(candidate_scores),
        statistics.pstdev(candidate_scores),
        row["sibling_distinct_predictions"] / 4.0,
        1.0 if row["sibling_unanimous"] else 0.0,
    ])

    if "op1" in reps:
        op_predictions = [row["predictions"]["op1"], row["predictions"]["op2"]]
        features.extend([
            sum(pred == candidate for pred in op_predictions) / 2.0,
            1.0 if op_predictions[0] == op_predictions[1] else 0.0,
        ])

    if family_aware:
        features.extend([1.0 if row["family"] == family else 0.0 for family in FAMILIES])

    return features


def build_matrix(
    rows: list[dict[str, Any]],
    reps: tuple[str, ...],
    *,
    family_aware: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for row in rows:
        for candidate in range(4):
            xs.append(candidate_features(row, candidate, reps, family_aware=family_aware))
            ys.append(1.0 if candidate == row["correct_index"] else 0.0)
    return torch.tensor(xs, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64)


def fit_ridge(
    rows: list[dict[str, Any]],
    reps: tuple[str, ...],
    strength: float,
    *,
    family_aware: bool,
) -> dict[str, Any]:
    x, y = build_matrix(rows, reps, family_aware=family_aware)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-9, torch.ones_like(std), std)
    xn = (x - mean) / std
    xa = torch.cat([torch.ones((xn.shape[0], 1), dtype=xn.dtype), xn], dim=1)
    penalty = torch.eye(xa.shape[1], dtype=xa.dtype)
    penalty[0, 0] = 0.0
    lhs = xa.T @ xa + float(strength) * penalty
    rhs = xa.T @ y
    coef = torch.linalg.solve(lhs, rhs)
    return {
        "reps": list(reps),
        "family_aware": family_aware,
        "strength": float(strength),
        "mean": mean,
        "std": std,
        "coef": coef,
    }


def ridge_candidate_scores(model: dict[str, Any], row: dict[str, Any]) -> list[float]:
    reps = tuple(model["reps"])
    rows = [
        candidate_features(row, candidate, reps, family_aware=model["family_aware"])
        for candidate in range(4)
    ]
    x = torch.tensor(rows, dtype=torch.float64)
    xn = (x - model["mean"]) / model["std"]
    xa = torch.cat([torch.ones((xn.shape[0], 1), dtype=xn.dtype), xn], dim=1)
    return [float(v) for v in xa @ model["coef"]]


def ridge_predict(model: dict[str, Any], row: dict[str, Any]) -> int:
    scores = ridge_candidate_scores(model, row)
    return max(range(4), key=lambda i: (scores[i], -i))


def accuracy_for_predictor(rows: list[dict[str, Any]], predictor) -> float:
    return sum(int(predictor(row)) == int(row["correct_index"]) for row in rows) / len(rows)


def tune_ridge(
    fit_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    reps: tuple[str, ...],
    *,
    family_aware: bool,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    trials = []
    models = []
    for strength in RIDGE_STRENGTHS:
        model = fit_ridge(fit_rows, reps, strength, family_aware=family_aware)
        accuracy = accuracy_for_predictor(validation_rows, lambda row, m=model: ridge_predict(m, row))
        trials.append({"strength": float(strength), "validation_accuracy": accuracy})
        models.append(model)
    # Prefer stronger regularization on exact validation ties.
    best_index = max(range(len(trials)), key=lambda i: (trials[i]["validation_accuracy"], trials[i]["strength"]))
    return models[best_index], trials


def majority_prediction(row: dict[str, Any], reps: tuple[str, ...]) -> int:
    votes = Counter(int(row["predictions"][rep]) for rep in reps)
    best = max(votes.values())
    tied = {candidate for candidate, count in votes.items() if count == best}
    layout = int(row["predictions"]["layout"])
    if layout in tied:
        return layout
    return min(tied)


def mean_score_prediction(row: dict[str, Any], reps: tuple[str, ...]) -> int:
    values = [
        statistics.fmean(float(row["scores"][rep][candidate]) for rep in reps)
        for candidate in range(4)
    ]
    return max(range(4), key=lambda i: (values[i], -i))


def score_cost(scorer: AttackScorer, prompt: str, choices: list[str]) -> dict[str, int]:
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


def collect_rows(
    scorer: AttackScorer,
    seeds: tuple[int, ...],
    cases_per_family: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    specs = []
    for seed in seeds:
        seed_spec = []
        for case in build_battery(seed, cases_per_family):
            canonical = build_representations(case["family"], case["surfaces"])
            surface_names = sorted(case["surfaces"])
            primary_prompt, primary_choices = surface_task_and_choices(case["surfaces"][surface_names[0]])
            secondary_prompt, secondary_choices = surface_task_and_choices(case["surfaces"][surface_names[1]])
            ops = operators_for_case(case["family"], case["surfaces"])
            if len(ops) != 2:
                raise AssertionError("expected exactly two exact operators")
            op1_id, op1_stem, op1_choices = ops[0]
            op2_id, op2_stem, op2_choices = ops[1]

            prompts = {
                "primary": (primary_prompt, primary_choices),
                "secondary": (secondary_prompt, secondary_choices),
                "layout": canonical["layout"],
                "familiar": canonical["familiar"],
                "op1": (op1_stem + "\nAnswer value:", op1_choices),
                "op2": (op2_stem + "\nAnswer value:", op2_choices),
            }
            scored = {
                name: _score_values(scorer, prompt, list(choices))
                for name, (prompt, choices) in prompts.items()
            }
            predictions = {name: int(value["prediction"]) for name, value in scored.items()}
            sibling_predictions = [predictions[rep] for rep in SIBLING_REPS]
            costs = {
                name: score_cost(scorer, prompt, list(choices))
                for name, (prompt, choices) in prompts.items()
            }
            correct = int(case["correct_index"])
            row = {
                "uid": f"{seed}:{case['case_id']}",
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "operator_ids": [op1_id, op2_id],
                "scores": {name: [float(v) for v in value["scores"]] for name, value in scored.items()},
                "margins": {name: float(value["margin"]) for name, value in scored.items()},
                "predictions": predictions,
                "sibling_distinct_predictions": len(set(sibling_predictions)),
                "sibling_unanimous": len(set(sibling_predictions)) == 1,
                "exact_operator_agreement": predictions["op1"] == predictions["op2"],
                "sibling_has_correct": any(predictions[rep] == correct for rep in SIBLING_REPS),
                "all_six_has_correct": any(predictions[rep] == correct for rep in ALL_REPS),
                "sibling_cost": add_cost([costs[rep] for rep in SIBLING_REPS]),
                "operator_pair_cost": add_cost([costs["op1"], costs["op2"]]),
                "prompt_sha256": {
                    name: hashlib.sha256(prompt.encode()).hexdigest()
                    for name, (prompt, _choices) in prompts.items()
                },
            }
            rows.append(row)
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
        specs.append({"seed": seed, "cases": seed_spec})
    return rows, specs


def evaluate_predictions(rows: list[dict[str, Any]], predictions: dict[str, list[int]]) -> dict[str, Any]:
    result = {}
    for arm, preds in predictions.items():
        hits = [int(pred) == int(row["correct_index"]) for pred, row in zip(preds, rows)]
        by_family: dict[str, list[bool]] = defaultdict(list)
        by_seed: dict[int, list[bool]] = defaultdict(list)
        for hit, row in zip(hits, rows):
            by_family[row["family"]].append(hit)
            by_seed[int(row["seed"])].append(hit)
        result[arm] = {
            "correct": sum(hits),
            "cases": len(rows),
            "accuracy": sum(hits) / len(rows),
            "by_family": {k: sum(v) / len(v) for k, v in sorted(by_family.items())},
            "by_seed": {str(k): sum(v) / len(v) for k, v in sorted(by_seed.items())},
        }
    return result


def paired_delta(
    rows: list[dict[str, Any]],
    predictions: dict[str, list[int]],
    arm: str,
    baseline: str,
) -> dict[str, int]:
    wins = regressions = ties = 0
    for i, row in enumerate(rows):
        target = int(row["correct_index"])
        a = int(predictions[arm][i]) == target
        b = int(predictions[baseline][i]) == target
        if a and not b:
            wins += 1
        elif b and not a:
            regressions += 1
        else:
            ties += 1
    return {"wins": wins, "regressions": regressions, "ties": ties}


def gate_cost(rows: list[dict[str, Any]], use_all_six: list[bool]) -> dict[str, Any]:
    base = add_cost([row["sibling_cost"] for row in rows])
    extra = add_cost([row["operator_pair_cost"] for row, selected in zip(rows, use_all_six) if selected])
    total = {key: base[key] + extra[key] for key in base}
    selected = sum(use_all_six)
    return {
        "all_six_cases": selected,
        "cases": len(rows),
        "all_six_fraction": selected / len(rows),
        "mean_representation_evaluations_per_case": 4.0 + 2.0 * selected / len(rows),
        "counterfactual_score_cost": total,
    }


def evaluate(model_dir: str, cases_per_family: int) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()
    started = time.time()

    fit_rows, fit_spec = collect_rows(scorer, FIT_SEEDS, cases_per_family)
    validation_rows, validation_spec = collect_rows(scorer, VALIDATION_SEEDS, cases_per_family)
    test_rows, test_spec = collect_rows(scorer, TEST_SEEDS, cases_per_family)

    sibling_model, sibling_trials = tune_ridge(
        fit_rows, validation_rows, SIBLING_REPS, family_aware=False
    )
    all_model, all_trials = tune_ridge(
        fit_rows, validation_rows, ALL_REPS, family_aware=False
    )
    sibling_family_model, sibling_family_trials = tune_ridge(
        fit_rows, validation_rows, SIBLING_REPS, family_aware=True
    )
    all_family_model, all_family_trials = tune_ridge(
        fit_rows, validation_rows, ALL_REPS, family_aware=True
    )

    sibling_learned = [ridge_predict(sibling_model, row) for row in test_rows]
    all_learned = [ridge_predict(all_model, row) for row in test_rows]
    sibling_family_learned = [ridge_predict(sibling_family_model, row) for row in test_rows]
    all_family_learned = [ridge_predict(all_family_model, row) for row in test_rows]

    unanimous_gate = [bool(row["sibling_unanimous"]) for row in test_rows]
    inverse_gate = [not value for value in unanimous_gate]
    k = sum(unanimous_gate)
    hash_order = sorted(
        range(len(test_rows)),
        key=lambda i: stable_key(test_rows[i]["uid"], "matched-frequency-gate"),
    )
    matched_indices = set(hash_order[:k])
    matched_gate = [i in matched_indices for i in range(len(test_rows))]

    predictions = {
        "layout_fixed": [int(row["predictions"]["layout"]) for row in test_rows],
        "sibling_majority": [majority_prediction(row, SIBLING_REPS) for row in test_rows],
        "sibling_score_mean": [mean_score_prediction(row, SIBLING_REPS) for row in test_rows],
        "sibling_ridge": sibling_learned,
        "sibling_family_ridge": sibling_family_learned,
        "all_six_score_mean": [mean_score_prediction(row, ALL_REPS) for row in test_rows],
        "all_six_ridge": all_learned,
        "all_six_family_ridge": all_family_learned,
        "helix_unanimity_gate": [
            all_learned[i] if unanimous_gate[i] else sibling_learned[i]
            for i in range(len(test_rows))
        ],
        "inverse_split_gate": [
            all_learned[i] if inverse_gate[i] else sibling_learned[i]
            for i in range(len(test_rows))
        ],
        "matched_frequency_gate": [
            all_learned[i] if matched_gate[i] else sibling_learned[i]
            for i in range(len(test_rows))
        ],
    }

    arm_results = evaluate_predictions(test_rows, predictions)
    paired = {
        arm: {
            "vs_layout": paired_delta(test_rows, predictions, arm, "layout_fixed"),
            "vs_sibling_ridge": paired_delta(test_rows, predictions, arm, "sibling_ridge"),
        }
        for arm in predictions
        if arm not in ("layout_fixed", "sibling_ridge")
    }

    sibling_oracle = sum(bool(row["sibling_has_correct"]) for row in test_rows)
    all_oracle = sum(bool(row["all_six_has_correct"]) for row in test_rows)
    exact_rescues = sum(
        bool((not row["sibling_has_correct"]) and row["all_six_has_correct"])
        for row in test_rows
    )

    gate_costs = {
        "helix_unanimity_gate": gate_cost(test_rows, unanimous_gate),
        "inverse_split_gate": gate_cost(test_rows, inverse_gate),
        "matched_frequency_gate": gate_cost(test_rows, matched_gate),
        "sibling_only": gate_cost(test_rows, [False] * len(test_rows)),
        "all_six": gate_cost(test_rows, [True] * len(test_rows)),
    }

    for arm in ("helix_unanimity_gate", "inverse_split_gate", "matched_frequency_gate", "sibling_ridge", "all_six_ridge"):
        mean_reps = (
            gate_costs[arm]["mean_representation_evaluations_per_case"]
            if arm in gate_costs else (4.0 if arm == "sibling_ridge" else 6.0)
        )
        arm_results[arm]["accuracy_per_representation_evaluation"] = (
            arm_results[arm]["accuracy"] / mean_reps
        )

    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-frontier-discriminator-calibration",
        "api_contract": {
            "helix_api_commit": "8640aaf43bd1f80c27406b12bfa6767c9ea1f695",
            "consumes": ["research.representation-search", "research.representation-frontier"],
        },
        "split": {
            "fit_seeds": list(FIT_SEEDS),
            "validation_seeds": list(VALIDATION_SEEDS),
            "fresh_test_seeds": list(TEST_SEEDS),
            "cases_per_family": cases_per_family,
            "fit_cases": len(fit_rows),
            "validation_cases": len(validation_rows),
            "test_cases": len(test_rows),
            "fit_battery_sha256": sha256_json(fit_spec),
            "validation_battery_sha256": sha256_json(validation_spec),
            "test_battery_sha256": sha256_json(test_spec),
        },
        "models": {
            "sibling_ridge": {
                "strength": sibling_model["strength"],
                "family_aware": False,
                "validation_trials": sibling_trials,
            },
            "all_six_ridge": {
                "strength": all_model["strength"],
                "family_aware": False,
                "validation_trials": all_trials,
            },
            "sibling_family_ridge": {
                "strength": sibling_family_model["strength"],
                "family_aware": True,
                "validation_trials": sibling_family_trials,
            },
            "all_six_family_ridge": {
                "strength": all_family_model["strength"],
                "family_aware": True,
                "validation_trials": all_family_trials,
            },
        },
        "test": {
            "arms": arm_results,
            "paired": paired,
            "oracle_ceiling": {
                "sibling_four_view_correct": sibling_oracle,
                "sibling_four_view_coverage": sibling_oracle / len(test_rows),
                "all_six_correct": all_oracle,
                "all_six_coverage": all_oracle / len(test_rows),
                "strict_exact_operator_rescues": exact_rescues,
            },
            "gate_costs": gate_costs,
            "unanimous_cases": sum(unanimous_gate),
            "split_cases": len(test_rows) - sum(unanimous_gate),
        },
        "measurement_execution_resources": resources,
        "rows": test_rows,
        "claim_boundary": (
            "Fresh public selector diagnostic. Selected-answer accuracy is measured directly; "
            "oracle candidate coverage is reported only as a ceiling. No protected promotion "
            "or frontier-equivalence claim is authorized."
        ),
    }


def self_test() -> dict[str, Any]:
    def row(uid: str, correct: int, sibling_preds: list[int], op_preds: list[int], family: str) -> dict[str, Any]:
        predictions = {
            "primary": sibling_preds[0],
            "secondary": sibling_preds[1],
            "layout": sibling_preds[2],
            "familiar": sibling_preds[3],
            "op1": op_preds[0],
            "op2": op_preds[1],
        }
        scores = {}
        for rep, pred in predictions.items():
            values = [-2.0, -2.0, -2.0, -2.0]
            values[pred] = -0.1
            values[(pred + 1) % 4] = -1.0
            scores[rep] = values
        return {
            "uid": uid,
            "family": family,
            "correct_index": correct,
            "predictions": predictions,
            "scores": scores,
            "sibling_distinct_predictions": len(set(sibling_preds)),
            "sibling_unanimous": len(set(sibling_preds)) == 1,
        }

    fit = [
        row("a", 0, [0, 0, 0, 0], [0, 0], "abstract-transformation"),
        row("b", 1, [1, 1, 1, 1], [1, 1], "grounded-planning"),
        row("c", 2, [2, 2, 2, 2], [2, 2], "relational-composition"),
        row("d", 3, [3, 3, 3, 3], [3, 3], "relational-matrix"),
        row("e", 0, [1, 0, 1, 0], [0, 0], "stack-language"),
        row("f", 2, [1, 2, 1, 2], [2, 2], "abstract-transformation"),
    ]
    validation = fit[:4]
    model, trials = tune_ridge(fit, validation, ALL_REPS, family_aware=False)
    accuracy = accuracy_for_predictor(validation, lambda r: ridge_predict(model, r))
    if accuracy < 0.75:
        raise AssertionError(("ridge self-test accuracy", accuracy))
    sample = fit[0]
    if majority_prediction(sample, SIBLING_REPS) != 0:
        raise AssertionError("majority self-test failed")
    if mean_score_prediction(sample, ALL_REPS) != 0:
        raise AssertionError("score mean self-test failed")
    return {
        "selected_strength": model["strength"],
        "validation_accuracy": accuracy,
        "trials": trials,
        "feature_count": len(candidate_features(sample, 0, ALL_REPS, family_aware=False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--cases-per-family", type=int, default=4)
    parser.add_argument("--output", default="run-044-frontier-conditioned-operator-discriminator.json")
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

    result = evaluate(args.model_dir, args.cases_per_family)
    result["actor"] = {
        "model": "HuggingFaceTB/SmolLM2-360M",
        "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        "model_file_sha256": observed,
        "model_bytes": model_path.stat().st_size,
        "dtype": "float32",
        "device": "cpu",
        "weights_frozen": True,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema_version": result["schema_version"],
        "split": result["split"],
        "models": result["models"],
        "test": result["test"],
        "measurement_execution_resources": result["measurement_execution_resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
