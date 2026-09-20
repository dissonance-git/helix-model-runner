"""Run 061: answer-blind STOP/CONTINUE controller over the actual Helix compiler.

Fit/validation may use burned correctness to learn a tiny external routing policy.
The primary controller is serialized and hashed before fresh-test actor scoring.
At test time, controller features contain visible geometry plus source/first-pass
actor diagnostics only. No answer index, hidden generator operation, or judge
outcome enters the controller.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from offline_reasoning_battery import build_battery, sha256_json
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import _score_values
from run_058_cross_family_compiler_portfolio import (
    _compact_program_record,
    _execute,
    _source_state,
)

RUN_VERSION = "helix-stop-aware-compiler-controller-001.0"
FIT_SEEDS = (20270120, 20270121)
VALIDATION_SEEDS = (20270122,)
TEST_SEEDS = (20270123, 20270124, 20270125)
CASES_PER_FAMILY = 4
MIN_LEAF = 8
DEPTHS = (0, 1, 2, 3)

FIRST_PROGRAM = ("first", "compose")
CONTINUE_PROGRAM = ("first", "second", "compose")
FAMILIES = (
    "relational-composition",
    "abstract-transformation",
    "relational-matrix",
    "grounded-planning",
    "stack-language",
)


def _entropy(log_probs: list[float]) -> float:
    return -sum(math.exp(v) * v for v in log_probs)


def _visible_complexity(case: dict[str, Any]) -> float:
    family = case["family"]
    surfaces = case["surfaces"]
    if family == "relational-composition":
        return float(len(re.findall(
            r"\b(?:north|south|east|west) of\b",
            surfaces["prose"],
        )))
    if family == "abstract-transformation":
        return float(surfaces["compact"].count("->"))
    if family == "relational-matrix":
        return float(len(re.findall(
            r"[01]{5} \? [01]{5} = [01]{5}",
            surfaces["bits"],
        )))
    if family == "grounded-planning":
        match = re.search(r"Choose the ([0-9]+)-move route", surfaces["grid"])
        if not match:
            raise ValueError("planning visible complexity parse failed")
        return float(int(match.group(1)))
    if family == "stack-language":
        match = re.search(r"prefix=(.*); completion=\?", surfaces["compact"])
        if not match:
            raise ValueError("stack visible complexity parse failed")
        return float(len(match.group(1).strip().split()))
    raise KeyError(family)


def _geometry_features(case: dict[str, Any]) -> dict[str, float]:
    out = {f"family:{family}": float(case["family"] == family) for family in FAMILIES}
    out["visible_complexity"] = _visible_complexity(case)
    return out


def _diagnostic_features(
    source: dict[str, Any],
    first: dict[str, Any],
) -> dict[str, float]:
    a = [float(x) for x in source["scores"]]
    b = [float(x) for x in first["scores"]]
    source_top = int(source["prediction"])
    first_top = int(first["prediction"])
    shifts = [b[i] - a[i] for i in range(4)]
    return {
        "source_margin": float(source["margin"]),
        "first_margin": float(first["margin"]),
        "margin_delta": float(first["margin"] - source["margin"]),
        "source_entropy": _entropy(a),
        "first_entropy": _entropy(b),
        "entropy_delta": _entropy(b) - _entropy(a),
        "score_l1_shift": sum(abs(x) for x in shifts),
        "score_max_abs_shift": max(abs(x) for x in shifts),
        "prediction_agree": float(source_top == first_top),
        "support_shift_source_top": shifts[source_top],
        "support_shift_first_top": shifts[first_top],
    }


def _case_features(
    case: dict[str, Any],
    source: dict[str, Any],
    first: dict[str, Any],
) -> dict[str, float]:
    return {**_geometry_features(case), **_diagnostic_features(source, first)}


def _leaf(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stop_correct = sum(r["stop_prediction"] == r["correct_index"] for r in rows)
    continue_correct = sum(r["continue_prediction"] == r["correct_index"] for r in rows)
    action = "CONTINUE" if continue_correct > stop_correct else "STOP"
    return {
        "leaf": True,
        "action": action,
        "support": len(rows),
        "fit_stop_correct": stop_correct,
        "fit_continue_correct": continue_correct,
        "fit_chosen_correct": max(stop_correct, continue_correct),
    }


def _thresholds(rows: list[dict[str, Any]], feature: str) -> list[float]:
    values = sorted({float(r["features"][feature]) for r in rows})
    return [(a + b) / 2.0 for a, b in zip(values, values[1:])]


def _fit_tree(
    rows: list[dict[str, Any]],
    features: tuple[str, ...],
    depth: int,
) -> dict[str, Any]:
    base = _leaf(rows)
    if depth <= 0 or len(rows) < 2 * MIN_LEAF:
        return base

    best = None
    base_reward = int(base["fit_chosen_correct"])
    for feature in features:
        for threshold in _thresholds(rows, feature):
            left = [r for r in rows if float(r["features"][feature]) <= threshold]
            right = [r for r in rows if float(r["features"][feature]) > threshold]
            if len(left) < MIN_LEAF or len(right) < MIN_LEAF:
                continue
            l_leaf = _leaf(left)
            r_leaf = _leaf(right)
            reward = int(l_leaf["fit_chosen_correct"]) + int(r_leaf["fit_chosen_correct"])
            continues = (
                (len(left) if l_leaf["action"] == "CONTINUE" else 0)
                + (len(right) if r_leaf["action"] == "CONTINUE" else 0)
            )
            candidate = (reward, -continues, feature, -threshold, threshold, left, right)
            if best is None or candidate[:4] > best[:4]:
                best = candidate

    if best is None or best[0] <= base_reward:
        return base

    _reward, _neg_cont, feature, _neg_threshold, threshold, left, right = best
    return {
        "leaf": False,
        "feature": feature,
        "threshold": float(threshold),
        "support": len(rows),
        "left": _fit_tree(left, features, depth - 1),
        "right": _fit_tree(right, features, depth - 1),
    }


def _tree_action(tree: dict[str, Any], features: dict[str, float]) -> str:
    node = tree
    while not node["leaf"]:
        node = node["left"] if float(features[node["feature"]]) <= float(node["threshold"]) else node["right"]
    return str(node["action"])


def _tree_depth(tree: dict[str, Any]) -> int:
    if tree["leaf"]:
        return 0
    return 1 + max(_tree_depth(tree["left"]), _tree_depth(tree["right"]))


def _tree_leaves(tree: dict[str, Any]) -> int:
    if tree["leaf"]:
        return 1
    return _tree_leaves(tree["left"]) + _tree_leaves(tree["right"])


def _apply_action(row: dict[str, Any], action: str) -> int:
    return int(row["continue_prediction"] if action == "CONTINUE" else row["stop_prediction"])


def _policy_stats(rows: list[dict[str, Any]], actions: list[str]) -> dict[str, Any]:
    correct = 0
    continues = 0
    by_family: dict[str, list[bool]] = defaultdict(list)
    for row, action in zip(rows, actions, strict=True):
        pred = _apply_action(row, action)
        ok = pred == int(row["correct_index"])
        correct += int(ok)
        continues += int(action == "CONTINUE")
        by_family[row["family"]].append(ok)
    return {
        "cases": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "continues": continues,
        "continue_fraction": continues / len(rows) if rows else 0.0,
        "by_family": {k: sum(v) / len(v) for k, v in sorted(by_family.items())},
    }


def _select_tree(
    fit_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    features: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = []
    for requested_depth in DEPTHS:
        tree = _fit_tree(fit_rows, features, requested_depth)
        actions = [_tree_action(tree, r["features"]) for r in validation_rows]
        stats = _policy_stats(validation_rows, actions)
        candidates.append({
            "requested_depth": requested_depth,
            "actual_depth": _tree_depth(tree),
            "leaves": _tree_leaves(tree),
            "tree": tree,
            "validation": stats,
        })
    chosen = max(
        candidates,
        key=lambda x: (
            float(x["validation"]["accuracy"]),
            -int(x["validation"]["continues"]),
            -int(x["actual_depth"]),
            -int(x["leaves"]),
        ),
    )
    return chosen["tree"], {
        "chosen_requested_depth": chosen["requested_depth"],
        "chosen_actual_depth": chosen["actual_depth"],
        "chosen_leaves": chosen["leaves"],
        "validation": chosen["validation"],
        "candidates": [
            {
                "requested_depth": c["requested_depth"],
                "actual_depth": c["actual_depth"],
                "leaves": c["leaves"],
                "validation": c["validation"],
            }
            for c in candidates
        ],
    }


def _fit_family_lookup(rows: list[dict[str, Any]]) -> dict[str, str]:
    policy = {}
    for family in FAMILIES:
        subset = [r for r in rows if r["family"] == family]
        stop_correct = sum(r["stop_prediction"] == r["correct_index"] for r in subset)
        continue_correct = sum(r["continue_prediction"] == r["correct_index"] for r in subset)
        policy[family] = "CONTINUE" if continue_correct > stop_correct else "STOP"
    return policy


def _paired(
    rows: list[dict[str, Any]],
    left_predictions: list[int],
    right_predictions: list[int],
) -> dict[str, int]:
    wins = regressions = ties = 0
    for row, left, right in zip(rows, left_predictions, right_predictions, strict=True):
        c = int(row["correct_index"])
        l_ok = int(left) == c
        r_ok = int(right) == c
        if l_ok and not r_ok:
            wins += 1
        elif r_ok and not l_ok:
            regressions += 1
        else:
            ties += 1
    return {"wins": wins, "regressions": regressions, "ties": ties}


def _integrity_update(integrity: dict[str, Any], record: dict[str, Any]) -> None:
    compact = _compact_program_record(record)
    integrity["programs"] += 1
    integrity["all_exact"] &= not bool(compact["contains_non_exact_step"])
    integrity["all_recovery_complete"] &= bool(compact["recovery_complete"])
    for step in compact["steps"]:
        integrity["all_preconditions_pass"] &= step["precondition_status"] == "pass"
        integrity["all_result_verifications_pass"] &= step["verification_status"] == "pass"


def _compile_case(
    helix_dir: str,
    case: dict[str, Any],
    split: str,
    seed: int,
    integrity: dict[str, Any],
) -> dict[str, Any]:
    state = _source_state(case["family"], case["case_id"], case["surfaces"])
    first_state, first_record = _execute(
        helix_dir,
        state,
        FIRST_PROGRAM,
        {"run_id":"061-stop-aware-compiler-controller","split":split,"seed":seed,"case_id":case["case_id"],"program":"first-stop"},
    )
    continue_state, continue_record = _execute(
        helix_dir,
        state,
        CONTINUE_PROGRAM,
        {"run_id":"061-stop-aware-compiler-controller","split":split,"seed":seed,"case_id":case["case_id"],"program":"continue"},
    )
    _integrity_update(integrity, first_record)
    _integrity_update(integrity, continue_record)
    return {
        "state": state,
        "first_state": first_state,
        "continue_state": continue_state,
    }


def _score_calibration_case(
    scorer: AttackScorer,
    compiled: dict[str, Any],
    case: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    state = compiled["state"]
    first_state = compiled["first_state"]
    continue_state = compiled["continue_state"]
    source = _score_values(scorer, state["source_prompt"], state["source_choices"])
    first = _score_values(scorer, first_state["final_prompt"], first_state["final_choices"])
    cont = _score_values(scorer, continue_state["final_prompt"], continue_state["final_choices"])
    features = _case_features(case, source, first)
    return {
        "uid": f"{seed}:{case['case_id']}",
        "seed": seed,
        "case_id": case["case_id"],
        "family": case["family"],
        "visible_complexity": _visible_complexity(case),
        "correct_index": int(case["correct_index"]),
        "source_prediction": int(source["prediction"]),
        "stop_prediction": int(first["prediction"]),
        "continue_prediction": int(cont["prediction"]),
        "features": features,
    }


def _build_and_score_calibration(
    helix_dir: str,
    scorer: AttackScorer,
    seeds: tuple[int, ...],
    split: str,
    integrity: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    spec = []
    for seed in seeds:
        seed_spec = []
        for case in build_battery(seed, CASES_PER_FAMILY):
            compiled = _compile_case(helix_dir, case, split, seed, integrity)
            row = _score_calibration_case(scorer, compiled, case, seed)
            rows.append(row)
            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "visible_complexity": row["visible_complexity"],
                "correct_index": row["correct_index"],
                "surface_sha256": {
                    k: hashlib.sha256(v.encode()).hexdigest()
                    for k, v in sorted(case["surfaces"].items())
                },
            })
        spec.append({"seed": seed, "cases": seed_spec})
    return rows, spec


def _prediction_list(rows: list[dict[str, Any]], actions: list[str]) -> list[int]:
    return [_apply_action(row, action) for row, action in zip(rows, actions, strict=True)]


def evaluate(helix_dir: str, model_dir: str) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    integrity = {
        "programs": 0,
        "all_exact": True,
        "all_recovery_complete": True,
        "all_preconditions_pass": True,
        "all_result_verifications_pass": True,
    }
    started = time.time()

    scorer.reset_counters()
    fit_rows, fit_spec = _build_and_score_calibration(
        helix_dir, scorer, FIT_SEEDS, "fit", integrity
    )
    validation_rows, validation_spec = _build_and_score_calibration(
        helix_dir, scorer, VALIDATION_SEEDS, "validation", integrity
    )
    calibration_resources = scorer.counters()

    geometry_names = tuple(sorted(k for k in fit_rows[0]["features"] if k.startswith("family:") or k == "visible_complexity"))
    diagnostic_names = tuple(sorted(k for k in fit_rows[0]["features"] if k not in geometry_names))
    combined_names = tuple(sorted(fit_rows[0]["features"]))

    geometry_tree, geometry_selection = _select_tree(
        fit_rows, validation_rows, geometry_names
    )
    diagnostic_tree, diagnostic_selection = _select_tree(
        fit_rows, validation_rows, diagnostic_names
    )
    primary_tree, primary_selection = _select_tree(
        fit_rows, validation_rows, combined_names
    )
    family_lookup = _fit_family_lookup(fit_rows)

    frozen = {
        "schema_version": "helix-stop-controller-policy-001.0",
        "fit_seeds": list(FIT_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "min_leaf": MIN_LEAF,
        "geometry_features": list(geometry_names),
        "diagnostic_features": list(diagnostic_names),
        "combined_features": list(combined_names),
        "family_lookup": family_lookup,
        "geometry_tree": geometry_tree,
        "diagnostics_tree": diagnostic_tree,
        "primary_tree": primary_tree,
        "selection": {
            "geometry": geometry_selection,
            "diagnostics": diagnostic_selection,
            "primary": primary_selection,
        },
    }
    frozen_text = json.dumps(frozen, sort_keys=True, separators=(",", ":"))
    frozen_sha256 = hashlib.sha256(frozen_text.encode()).hexdigest()

    # Fresh-test compilation and first-pass scoring begins only after policy freeze.
    scorer.reset_counters()
    test_rows: list[dict[str, Any]] = []
    test_compiled: dict[str, dict[str, Any]] = {}
    test_spec = []
    for seed in TEST_SEEDS:
        seed_spec = []
        for case in build_battery(seed, CASES_PER_FAMILY):
            uid = f"{seed}:{case['case_id']}"
            compiled = _compile_case(helix_dir, case, "fresh-test", seed, integrity)
            state = compiled["state"]
            first_state = compiled["first_state"]
            source = _score_values(scorer, state["source_prompt"], state["source_choices"])
            first = _score_values(scorer, first_state["final_prompt"], first_state["final_choices"])
            features = _case_features(case, source, first)
            row = {
                "uid": uid,
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "visible_complexity": _visible_complexity(case),
                "correct_index": int(case["correct_index"]),
                "source_prediction": int(source["prediction"]),
                "stop_prediction": int(first["prediction"]),
                "continue_prediction": None,
                "features": features,
            }
            test_rows.append(row)
            test_compiled[uid] = compiled
            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "visible_complexity": row["visible_complexity"],
                "correct_index": row["correct_index"],
                "surface_sha256": {
                    k: hashlib.sha256(v.encode()).hexdigest()
                    for k, v in sorted(case["surfaces"].items())
                },
            })
        test_spec.append({"seed": seed, "cases": seed_spec})

    primary_actions = [_tree_action(primary_tree, r["features"]) for r in test_rows]
    geometry_actions = [_tree_action(geometry_tree, r["features"]) for r in test_rows]
    diagnostic_actions = [_tree_action(diagnostic_tree, r["features"]) for r in test_rows]
    family_actions = [family_lookup[r["family"]] for r in test_rows]
    disagreement_actions = [
        "CONTINUE" if r["source_prediction"] != r["stop_prediction"] else "STOP"
        for r in test_rows
    ]

    # Execute only the second stage selected by the frozen primary controller first.
    for row, action in zip(test_rows, primary_actions, strict=True):
        if action != "CONTINUE":
            continue
        target = test_compiled[row["uid"]]["continue_state"]
        scored = _score_values(scorer, target["final_prompt"], target["final_choices"])
        row["continue_prediction"] = int(scored["prediction"])
    primary_adaptive_resources = scorer.counters()

    # Complete the forward control only after primary actions and adaptive cost are frozen.
    for row in test_rows:
        if row["continue_prediction"] is not None:
            continue
        target = test_compiled[row["uid"]]["continue_state"]
        scored = _score_values(scorer, target["final_prompt"], target["final_choices"])
        row["continue_prediction"] = int(scored["prediction"])
    total_test_resources = scorer.counters()

    policies = {
        "source": ["STOP"] * len(test_rows),
        "always-stop-first": ["STOP"] * len(test_rows),
        "always-continue": ["CONTINUE"] * len(test_rows),
        "family-lookup": family_actions,
        "geometry-only-tree": geometry_actions,
        "diagnostics-only-tree": diagnostic_actions,
        "geometry-plus-diagnostics-tree": primary_actions,
        "disagreement-continue": disagreement_actions,
    }

    predictions: dict[str, list[int]] = {
        "source": [int(r["source_prediction"]) for r in test_rows],
    }
    for name, actions in policies.items():
        if name == "source":
            continue
        predictions[name] = _prediction_list(test_rows, actions)

    summaries = {}
    for name, preds in predictions.items():
        if name == "source":
            actions = ["STOP"] * len(test_rows)
            correct = sum(p == r["correct_index"] for p, r in zip(preds, test_rows, strict=True))
            summaries[name] = {
                "cases": len(test_rows),
                "correct": correct,
                "accuracy": correct / len(test_rows),
                "continues": 0,
                "continue_fraction": 0.0,
            }
        else:
            summaries[name] = _policy_stats(test_rows, policies[name])

    fixed_names = ("always-stop-first", "always-continue")
    better_fixed = max(
        fixed_names,
        key=lambda n: (summaries[n]["accuracy"], -summaries[n]["continues"], n),
    )
    primary_name = "geometry-plus-diagnostics-tree"
    geometry_name = "geometry-only-tree"

    paired = {
        "primary-vs-best-fixed": _paired(
            test_rows, predictions[primary_name], predictions[better_fixed]
        ),
        "primary-vs-geometry": _paired(
            test_rows, predictions[primary_name], predictions[geometry_name]
        ),
        "primary-vs-always-stop": _paired(
            test_rows, predictions[primary_name], predictions["always-stop-first"]
        ),
        "primary-vs-always-continue": _paired(
            test_rows, predictions[primary_name], predictions["always-continue"]
        ),
        "geometry-vs-best-fixed": _paired(
            test_rows, predictions[geometry_name], predictions[better_fixed]
        ),
    }

    oracle_correct = sum(
        r["stop_prediction"] == r["correct_index"]
        or r["continue_prediction"] == r["correct_index"]
        for r in test_rows
    )
    stop_only = summaries["always-stop-first"]["correct"]
    continue_only = summaries["always-continue"]["correct"]
    best_fixed_correct = max(stop_only, continue_only)

    rows_out = []
    for i, row in enumerate(test_rows):
        rows_out.append({
            "uid": row["uid"],
            "seed": row["seed"],
            "case_id": row["case_id"],
            "family": row["family"],
            "visible_complexity": row["visible_complexity"],
            "correct_index": row["correct_index"],
            "source_prediction": row["source_prediction"],
            "stop_prediction": row["stop_prediction"],
            "continue_prediction": row["continue_prediction"],
            "actions": {
                "family_lookup": family_actions[i],
                "geometry_only": geometry_actions[i],
                "diagnostics_only": diagnostic_actions[i],
                "primary": primary_actions[i],
                "disagreement_continue": disagreement_actions[i],
            },
            "features": row["features"],
        })

    primary_summary = summaries[primary_name]
    geometry_summary = summaries[geometry_name]
    best_fixed_summary = summaries[better_fixed]
    primary_pair = paired["primary-vs-best-fixed"]
    diagnostic_pair = paired["primary-vs-geometry"]

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-stop-controller-test",
        "actor": {
            "model": "HuggingFaceTB/SmolLM2-360M",
            "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
            "weights_frozen": True,
        },
        "split": {
            "fit_seeds": list(FIT_SEEDS),
            "validation_seeds": list(VALIDATION_SEEDS),
            "fresh_test_seeds": list(TEST_SEEDS),
            "cases_per_family": CASES_PER_FAMILY,
            "fit_cases": len(fit_rows),
            "validation_cases": len(validation_rows),
            "fresh_test_cases": len(test_rows),
            "fit_spec_sha256": sha256_json(fit_spec),
            "validation_spec_sha256": sha256_json(validation_spec),
            "fresh_test_spec_sha256": sha256_json(test_spec),
        },
        "frozen_policy": {
            "sha256": frozen_sha256,
            "policy": frozen,
            "frozen_before_test_actor_scoring": True,
        },
        "fresh_test": {
            "summaries": summaries,
            "best_fixed_policy": better_fixed,
            "paired": paired,
            "stop_continue_oracle": {
                "correct": oracle_correct,
                "cases": len(test_rows),
                "coverage": oracle_correct / len(test_rows),
                "best_fixed_correct": best_fixed_correct,
                "recoverable_gap_cases": oracle_correct - best_fixed_correct,
            },
        },
        "decisions": {
            "stop_aware_compiler_gain_earned": (
                primary_summary["accuracy"] > summaries["always-stop-first"]["accuracy"]
                and primary_summary["accuracy"] > summaries["always-continue"]["accuracy"]
                and primary_pair["wins"] > primary_pair["regressions"]
            ),
            "actor_diagnostics_add_routing_value": (
                primary_summary["accuracy"] > geometry_summary["accuracy"]
                and diagnostic_pair["wins"] > diagnostic_pair["regressions"]
            ),
            "simpler_geometry_policy_preferred": (
                geometry_summary["accuracy"] >= primary_summary["accuracy"]
            ),
            "compute_aware_gain_earned": (
                primary_summary["accuracy"] >= best_fixed_summary["accuracy"]
                and primary_summary["continue_fraction"] < 1.0
            ),
            "routing_headroom_remains": oracle_correct > primary_summary["correct"],
        },
        "compiler_integrity": integrity,
        "resources": {
            "calibration": calibration_resources,
            "fresh_primary_adaptive_actual": primary_adaptive_resources,
            "fresh_total_with_all_controls": total_test_resources,
            "primary_forward_cases": primary_summary["continues"],
            "primary_forward_fraction": primary_summary["continue_fraction"],
        },
        "rows": rows_out,
        "claim_boundary": (
            "Fresh public evaluation of a tiny external STOP/CONTINUE compiler controller frozen "
            "before test actor scoring. Fit/validation correctness may train/select the controller; "
            "fresh correctness is evaluation-only. No protected promotion, universal compiler policy, "
            "or general-intelligence claim follows."
        ),
        "elapsed_seconds": time.time() - started,
    }


def self_test(helix_dir: str) -> dict[str, Any]:
    integrity = {
        "programs": 0,
        "all_exact": True,
        "all_recovery_complete": True,
        "all_preconditions_pass": True,
        "all_result_verifications_pass": True,
    }
    cases = build_battery(FIT_SEEDS[0], 2)
    rows = []
    for case in cases:
        compiled = _compile_case(helix_dir, case, "self-test", FIT_SEEDS[0], integrity)
        state = compiled["state"]
        if "correct_index" in state or "correct_index" in compiled["first_state"] or "correct_index" in compiled["continue_state"]:
            raise AssertionError("correctness leaked into compiler state")
        geom = _geometry_features(case)
        rows.append({
            "case_id": case["case_id"],
            "family": case["family"],
            "visible_complexity": _visible_complexity(case),
            "geometry_features": geom,
            "first_prompt_sha256": hashlib.sha256(compiled["first_state"]["final_prompt"].encode()).hexdigest(),
            "continue_prompt_sha256": hashlib.sha256(compiled["continue_state"]["final_prompt"].encode()).hexdigest(),
        })
    return {
        "schema_version": RUN_VERSION,
        "status": "self-test-pass",
        "cases": len(rows),
        "actual_execution_owner": "engine.ir.passes.execute_transform_program",
        "correctness_visible_to_compiler": False,
        "raw_answer_index_controller_feature": False,
        "compiler_integrity": integrity,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helix-dir", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", default="run-061-stop-aware-compiler-controller.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(args.helix_dir), indent=2, sort_keys=True))
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

    result = evaluate(args.helix_dir, args.model_dir)
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
        "split": result["split"],
        "frozen_policy_sha256": result["frozen_policy"]["sha256"],
        "fresh_test": result["fresh_test"],
        "decisions": result["decisions"],
        "compiler_integrity": result["compiler_integrity"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
