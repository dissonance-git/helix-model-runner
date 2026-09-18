"""Run 040: composition specificity controls.

Test whether deterministic A+B behavior is specific to combining distinct
representations or is explainable by same-view repetition A+A / B+B.
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
from run_028_canonical_representation import _score_values
from run_037_deterministic_composition import case_views, composite_prompt

RUN_VERSION = "helix-composition-specificity-controls-001.0"


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    hits = 0
    by_family: dict[str, list[bool]] = defaultdict(list)
    by_seed: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        ok = int(row[key]) == int(row["correct_index"])
        hits += int(ok)
        by_family[row["family"]].append(ok)
        by_seed[int(row["seed"])].append(ok)
    return {
        "cases": len(rows),
        "correct": hits,
        "accuracy": hits / len(rows),
        "by_family": {k: sum(v)/len(v) for k,v in sorted(by_family.items())},
        "by_seed": {str(k): sum(v)/len(v) for k,v in sorted(by_seed.items())},
    }


def evaluate(model_dir: str, seeds: list[int], cases_per_family: int) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()
    started = time.time()
    rows = []
    public_spec = []

    for seed in seeds:
        cases = build_battery(seed, cases_per_family)
        seed_spec = []
        for case in cases:
            views, choices = case_views(case)
            prompts = {
                "A": views["A"] + "\nAnswer value:",
                "B": views["B"] + "\nAnswer value:",
                "AA": composite_prompt(["A", "A"], views),
                "BB": composite_prompt(["B", "B"], views),
                "AB": composite_prompt(["A", "B"], views),
                "BA": composite_prompt(["B", "A"], views),
            }
            scored = {
                name: _score_values(scorer, prompt, choices)
                for name, prompt in prompts.items()
            }
            preds = {name: int(value["prediction"]) for name, value in scored.items()}
            correct = int(case["correct_index"])

            singleton_correct = preds["A"] == correct or preds["B"] == correct
            duplication_correct = preds["AA"] == correct or preds["BB"] == correct
            distinct_correct = preds["AB"] == correct or preds["BA"] == correct
            strict_distinct = (not singleton_correct) and (not duplication_correct) and distinct_correct
            duplication_only = (not singleton_correct) and duplication_correct
            order_disagrees = preds["AB"] != preds["BA"]
            order_one_correct = order_disagrees and ((preds["AB"] == correct) != (preds["BA"] == correct))

            rows.append({
                "uid": f"{seed}:{case['case_id']}",
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                **{f"{name}_prediction": pred for name, pred in preds.items()},
                "singleton_correct": singleton_correct,
                "same_view_duplication_correct": duplication_correct,
                "distinct_view_composition_correct": distinct_correct,
                "strict_distinct_pair_only_correct": strict_distinct,
                "same_view_duplication_only_correct": duplication_only,
                "operator_order_disagrees": order_disagrees,
                "operator_order_one_correct": order_one_correct,
                "prompt_lengths_chars": {
                    name: len(prompt) for name, prompt in prompts.items()
                },
                "prompt_sha256": {
                    name: hashlib.sha256(prompt.encode()).hexdigest()
                    for name, prompt in prompts.items()
                },
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
        public_spec.append({"seed": seed, "cases": seed_spec})

    arms = {
        name: summarize(rows, f"{name}_prediction")
        for name in ("A", "B", "AA", "BB", "AB", "BA")
    }
    singleton_or_duplication = sum(
        row["singleton_correct"] or row["same_view_duplication_correct"]
        for row in rows
    )
    with_distinct = sum(
        row["singleton_correct"]
        or row["same_view_duplication_correct"]
        or row["distinct_view_composition_correct"]
        for row in rows
    )

    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-composition-specificity-calibration",
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(rows),
        "source_battery_sha256": sha256_json(public_spec),
        "arms": arms,
        "specificity": {
            "singleton_or_same_view_duplication_coverage_correct": singleton_or_duplication,
            "singleton_or_same_view_duplication_coverage": singleton_or_duplication / len(rows),
            "plus_distinct_view_composition_coverage_correct": with_distinct,
            "plus_distinct_view_composition_coverage": with_distinct / len(rows),
            "strict_distinct_pair_only_correct_cases": sum(
                row["strict_distinct_pair_only_correct"] for row in rows
            ),
            "same_view_duplication_only_correct_cases": sum(
                row["same_view_duplication_only_correct"] for row in rows
            ),
            "operator_order_disagreement_cases": sum(
                row["operator_order_disagrees"] for row in rows
            ),
            "operator_order_one_correct_cases": sum(
                row["operator_order_one_correct"] for row in rows
            ),
        },
        "resources": resources,
        "rows": rows,
        "claim_boundary": (
            "Fresh public procedural diagnostic only. Strict pair-specific behavior requires "
            "A and B singletons plus A+A and B+B controls to fail on the same case before "
            "A+B or B+A correctness counts as distinct-view composition evidence. No protected "
            "promotion or frontier claim is authorized."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seeds", default="20261022,20261023,20261024")
    parser.add_argument("--cases-per-family", type=int, default=4)
    parser.add_argument("--output", default="run-040-composition-specificity.json")
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
        "arms": result["arms"],
        "specificity": result["specificity"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
