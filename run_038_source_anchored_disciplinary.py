"""Run 038: source-anchored disciplinary lenses.

The deterministic layout-normalized task remains intact. Each field contributes
only a fixed interrogation frame from Helix's scientific-method catalog. No
model-generated intermediate representation is allowed.
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
from run_028_canonical_representation import build_representations, _score_values
from run_034_disciplinary_representations import FIELD_QUESTIONS, FIELDS

RUN_VERSION = "helix-source-anchored-disciplinary-lenses-001.0"


def strip_answer_marker(prompt: str) -> str:
    value = prompt.rstrip()
    suffix = "Answer value:"
    if not value.endswith(suffix):
        raise ValueError("prompt lacks answer marker")
    return value[:-len(suffix)].rstrip()


def lens_prompt(field: str, canonical_source: str) -> str:
    questions = "\n".join(
        f"- {question}" for question in FIELD_QUESTIONS[field]
    )
    return (
        "Solve the canonical reasoning task below. Keep the task semantics unchanged.\n"
        f"Use the {field} questions only as analysis coordinates; do not invent domain facts.\n"
        "If a field question does not apply, ignore it rather than forcing an analogy.\n"
        "FIELD-NATIVE QUESTIONS:\n"
        f"{questions}\n\n"
        "CANONICAL TASK:\n"
        f"{canonical_source}\n\n"
        "Answer value:"
    )


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
        "by_family": {
            k: sum(v) / len(v) for k, v in sorted(by_family.items())
        },
        "by_seed": {
            str(k): sum(v) / len(v) for k, v in sorted(by_seed.items())
        },
    }


def pairwise(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out = {}
    for i, left in enumerate(keys):
        left_set = {
            row["uid"] for row in rows
            if int(row[left]) == int(row["correct_index"])
        }
        for right in keys[i + 1:]:
            right_set = {
                row["uid"] for row in rows
                if int(row[right]) == int(row["correct_index"])
            }
            union = left_set | right_set
            inter = left_set & right_set
            out[f"{left}+{right}"] = {
                "left_only_correct": len(left_set - right_set),
                "right_only_correct": len(right_set - left_set),
                "both_correct": len(inter),
                "oracle_correct": len(union),
                "oracle_coverage": len(union) / len(rows),
                "jaccard_correct_sets": len(inter) / len(union) if union else 1.0,
            }
    return out


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
            transformed = build_representations(case["family"], case["surfaces"])
            layout_prompt, choices = transformed["layout"]
            canonical = strip_answer_marker(layout_prompt)

            baseline = _score_values(scorer, layout_prompt, choices)
            field_scores = {
                field: _score_values(
                    scorer,
                    lens_prompt(field, canonical),
                    choices,
                )
                for field in FIELDS
            }

            predictions = {
                "layout": int(baseline["prediction"]),
                **{
                    field: int(value["prediction"])
                    for field, value in field_scores.items()
                },
            }
            correct = int(case["correct_index"])
            correct_fields = [
                field for field in FIELDS
                if predictions[field] == correct
            ]
            rows.append({
                "uid": f"{seed}:{case['case_id']}",
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "layout_prediction": predictions["layout"],
                **{
                    f"{field}_prediction": predictions[field]
                    for field in FIELDS
                },
                "distinct_field_predictions": len({
                    predictions[field] for field in FIELDS
                }),
                "correct_fields": correct_fields,
                "field_only_correct": (
                    bool(correct_fields)
                    and predictions["layout"] != correct
                ),
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
                "layout_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            })
        public_spec.append({"seed": seed, "cases": seed_spec})

    arm_keys = ["layout_prediction"] + [
        f"{field}_prediction" for field in FIELDS
    ]
    summaries = {
        "layout": summarize(rows, "layout_prediction"),
        **{
            field: summarize(rows, f"{field}_prediction")
            for field in FIELDS
        },
    }

    layout_set = {
        row["uid"] for row in rows
        if row["layout_prediction"] == row["correct_index"]
    }
    field_sets = {
        field: {
            row["uid"] for row in rows
            if row[f"{field}_prediction"] == row["correct_index"]
        }
        for field in FIELDS
    }
    all_sets = [layout_set, *field_sets.values()]
    oracle = set().union(*all_sets)
    field_union = set().union(*field_sets.values())

    diversity_hist = defaultdict(int)
    for row in rows:
        diversity_hist[str(row["distinct_field_predictions"])] += 1

    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started
    resources["field_scoring_arms_per_case"] = len(FIELDS)

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-source-anchored-disciplinary-calibration",
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(rows),
        "helix_lens_source": {
            "commit": "1984a42ff8c50b35227a89f1865d6e68c86f9cc8",
            "fields": list(FIELDS),
        },
        "source_battery_sha256": sha256_json(public_spec),
        "summaries": summaries,
        "coverage": {
            "layout_correct": len(layout_set),
            "layout_accuracy": len(layout_set) / len(rows),
            "any_field_correct": len(field_union),
            "any_field_coverage": len(field_union) / len(rows),
            "nine_arm_oracle_correct": len(oracle),
            "nine_arm_oracle_coverage": len(oracle) / len(rows),
            "new_correct_cases_from_fields_beyond_layout": len(field_union - layout_set),
        },
        "prediction_diversity": {
            "field_prediction_count_histogram": dict(sorted(diversity_hist.items())),
            "cases_with_more_than_one_field_prediction": sum(
                row["distinct_field_predictions"] > 1 for row in rows
            ),
        },
        "pairwise": pairwise(rows, arm_keys),
        "resources": resources,
        "rows": rows,
        "claim_boundary": (
            "Fresh public procedural diagnostic only. The canonical task is preserved "
            "and no intermediate field state is generated. Field frames are analysis "
            "coordinates, not domain authority. No protected promotion or frontier claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seeds", default="20261017,20261018")
    parser.add_argument("--cases-per-family", type=int, default=4)
    parser.add_argument("--output", default="run-038-disciplinary-lenses.json")
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
        "summaries": result["summaries"],
        "coverage": result["coverage"],
        "prediction_diversity": result["prediction_diversity"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
