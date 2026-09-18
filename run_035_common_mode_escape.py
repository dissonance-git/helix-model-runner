"""Run 035: disagreement-conditioned common-mode escape.

Four fixed representations are scored first. Split fields use a frozen family
readout. Unanimous fields trigger answer-blind cross-view synthesis because
Run 032 suggested synthesis can escape common-mode representational errors.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from offline_reasoning_battery import build_battery, sha256_json
from run_026_capability_substitution import AttackScorer
from run_032_cross_representation_invariant import (
    representations_for_case,
    score_values,
    summarize_view,
    synthesize_invariant,
)

RUN_VERSION = "helix-common-mode-escape-001.0"

FAMILY_READOUT = {
    "abstract-transformation": "layout-normalized",
    "grounded-planning": "lexicographic-primary-surface",
    "relational-composition": "alternate-secondary-surface",
    "relational-matrix": "lexicographic-primary-surface",
    "stack-language": "alternate-secondary-surface",
}


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    hits = 0
    by_family: dict[str, list[bool]] = defaultdict(list)
    by_seed: dict[int, list[bool]] = defaultdict(list)
    by_mode: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        ok = int(row[key]) == int(row["correct_index"])
        hits += int(ok)
        by_family[row["family"]].append(ok)
        by_seed[int(row["seed"])].append(ok)
        by_mode["unanimous" if row["original_all_agree"] else "split"].append(ok)
    return {
        "cases": len(rows),
        "correct": hits,
        "accuracy": hits / len(rows),
        "by_family": {
            k: sum(v) / len(v) for k, v in sorted(by_family.items())
        },
        "by_seed": {
            str(k): sum(v) / len(v) for k, v in sorted(by_seed.items())
        },
        "by_mode": {
            k: {
                "cases": len(v),
                "correct": sum(v),
                "accuracy": sum(v) / len(v),
            }
            for k, v in sorted(by_mode.items())
        },
    }


def pair_delta(
    rows: list[dict[str, Any]],
    left_key: str,
    right_key: str,
) -> dict[str, int]:
    left_only = right_only = both = neither = 0
    for row in rows:
        left = int(row[left_key]) == int(row["correct_index"])
        right = int(row[right_key]) == int(row["correct_index"])
        if left and right:
            both += 1
        elif left:
            left_only += 1
        elif right:
            right_only += 1
        else:
            neither += 1
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "both_correct": both,
        "neither_correct": neither,
    }


def evaluate(
    model_dir: str,
    seeds: list[int],
    cases_per_family: int,
) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()
    started = time.time()
    rows: list[dict[str, Any]] = []
    public_specs = []

    for seed in seeds:
        cases = build_battery(seed, cases_per_family)
        seed_spec = []
        for local_index, case in enumerate(cases):
            uid = f"{seed}:{case['case_id']}"
            reps = representations_for_case(case)
            rep_scores = {
                name: score_values(scorer, prompt, choices)
                for name, (prompt, choices) in reps.items()
            }
            predictions = {
                name: int(value["prediction"])
                for name, value in rep_scores.items()
            }
            unique_predictions = set(predictions.values())
            unanimous = len(unique_predictions) == 1

            layout_prediction = predictions["layout-normalized"]
            family_prediction = predictions[FAMILY_READOUT[case["family"]]]
            policy_prediction = family_prediction
            synthesis_prediction = None
            synthesis_state_sha256 = None
            summary_sha256 = None

            if unanimous:
                summaries = {}
                for view_index, (name, (prompt, _choices)) in enumerate(reps.items()):
                    summaries[name] = summarize_view(
                        scorer,
                        representation_name=name,
                        prompt=prompt,
                        seed=seed + local_index * 17 + view_index,
                    )
                invariant = synthesize_invariant(
                    scorer,
                    summaries,
                    seed=seed + 100000 + local_index,
                )
                layout_choices = reps["layout-normalized"][1]
                invariant_prompt = (
                    "Use the following derived shared state as the representation of the task.\n"
                    "Choose the answer value that satisfies it.\n"
                    "Shared invariant state:\n"
                    f"{invariant}\n"
                    "Answer value:"
                )
                scored = score_values(scorer, invariant_prompt, layout_choices)
                synthesis_prediction = int(scored["prediction"])
                policy_prediction = synthesis_prediction
                synthesis_state_sha256 = hashlib.sha256(
                    invariant.encode()
                ).hexdigest()
                summary_sha256 = {
                    name: hashlib.sha256(text.encode()).hexdigest()
                    for name, text in summaries.items()
                }

            correct = int(case["correct_index"])
            original_oracle = any(
                prediction == correct for prediction in predictions.values()
            )
            synthesis_adds_candidate = bool(
                unanimous
                and synthesis_prediction == correct
                and not original_oracle
            )

            rows.append({
                "uid": uid,
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "original_predictions": predictions,
                "original_all_agree": unanimous,
                "layout_prediction": layout_prediction,
                "family_readout_prediction": family_prediction,
                "policy_prediction": policy_prediction,
                "synthesis_prediction": synthesis_prediction,
                "original_oracle_contains_correct": original_oracle,
                "synthesis_adds_new_correct_candidate": synthesis_adds_candidate,
                "synthesis_state_sha256": synthesis_state_sha256,
                "summary_sha256": summary_sha256,
            })

            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "source_surface_sha256": {
                    name: hashlib.sha256(prompt.encode()).hexdigest()
                    for name, prompt in case["surfaces"].items()
                },
            })
        public_specs.append({"seed": seed, "cases": seed_spec})

    unanimous_rows = [row for row in rows if row["original_all_agree"]]
    unanimous_correct_before = sum(
        row["family_readout_prediction"] == row["correct_index"]
        for row in unanimous_rows
    )
    unanimous_correct_after = sum(
        row["policy_prediction"] == row["correct_index"]
        for row in unanimous_rows
    )
    repairs = sum(
        row["family_readout_prediction"] != row["correct_index"]
        and row["policy_prediction"] == row["correct_index"]
        for row in unanimous_rows
    )
    regressions = sum(
        row["family_readout_prediction"] == row["correct_index"]
        and row["policy_prediction"] != row["correct_index"]
        for row in unanimous_rows
    )
    new_candidates = sum(
        row["synthesis_adds_new_correct_candidate"] for row in rows
    )

    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started
    resources["representations_scored_per_case"] = 4
    resources["synthesis_triggered_cases"] = len(unanimous_rows)
    resources["summary_generations_per_triggered_case"] = 4
    resources["invariant_generations_per_triggered_case"] = 1

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-common-mode-escape-calibration",
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(rows),
        "source_battery_sha256": sha256_json(public_specs),
        "frozen_family_readout": FAMILY_READOUT,
        "trigger": {
            "rule": "all four representation predictions have one unique answer index",
            "answer_blind": True,
            "unanimous_cases": len(unanimous_rows),
            "unanimous_fraction": len(unanimous_rows) / len(rows),
        },
        "arms": {
            "fixed-layout": summarize(rows, "layout_prediction"),
            "frozen-family-readout": summarize(
                rows, "family_readout_prediction"
            ),
            "common-mode-escape-policy": summarize(rows, "policy_prediction"),
        },
        "paired_policy_vs_family": pair_delta(
            rows,
            "policy_prediction",
            "family_readout_prediction",
        ),
        "unanimous_effect": {
            "cases": len(unanimous_rows),
            "correct_before": unanimous_correct_before,
            "correct_after": unanimous_correct_after,
            "repairs": repairs,
            "regressions": regressions,
            "new_correct_candidates_absent_from_all_original_views": new_candidates,
        },
        "resources": resources,
        "rows": rows,
        "claim_boundary": (
            "Fresh public procedural replication. Representation unanimity triggers "
            "answer-blind synthesis without test labels. This run tests transfer of a "
            "common-mode escape policy only; it cannot authorize promotion, frontier "
            "equivalence, or a general reasoning claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seeds", default="20261009,20261010,20261011")
    parser.add_argument("--cases-per-family", type=int, default=6)
    parser.add_argument("--output", default="run-035-common-mode.json")
    args = parser.parse_args()

    model_path = Path(args.model_file)
    observed = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    result = evaluate(
        args.model_dir,
        [int(v) for v in args.seeds.split(",") if v.strip()],
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

    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema_version": result["schema_version"],
        "trigger": result["trigger"],
        "arms": result["arms"],
        "paired_policy_vs_family": result["paired_policy_vs_family"],
        "unanimous_effect": result["unanimous_effect"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
