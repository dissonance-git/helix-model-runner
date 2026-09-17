"""Run 027: transfer a frozen family -> representation router.

The router mapping was fixed from Run 021 before these seeds were generated.
Fresh labels are used only for posthoc scoring.
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

from offline_reasoning_battery import ChoiceScorer, build_battery, sha256_json
from offline_reasoning_content_probe import choose_content

RUN_VERSION = "helix-representation-router-transfer-001.0"
ROUTER = {
    "abstract-transformation": "compact",
    "grounded-planning": "grid",
    "relational-composition": "prose",
    "relational-matrix": "bits",
    "stack-language": "prose",
}


def log_softmax(values: list[float]) -> list[float]:
    m = max(values)
    z = m + math.log(sum(math.exp(v - m) for v in values))
    return [v - z for v in values]


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    by_family: dict[str, list[bool]] = defaultdict(list)
    by_seed: dict[int, list[bool]] = defaultdict(list)
    hits = 0
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
            family: sum(values) / len(values)
            for family, values in sorted(by_family.items())
        },
        "by_seed": {
            str(seed): sum(values) / len(values)
            for seed, values in sorted(by_seed.items())
        },
    }


def evaluate(model_dir: str, seeds: list[int], cases_per_family: int) -> dict[str, Any]:
    scorer = ChoiceScorer(model_dir)
    started = time.time()
    rows: list[dict[str, Any]] = []
    public_specs = []

    for seed in seeds:
        cases = build_battery(seed, cases_per_family)
        public_specs.append({
            "seed": seed,
            "cases": [
                {
                    "case_id": case["case_id"],
                    "family": case["family"],
                    "difficulty": case["difficulty"],
                    "correct_index": case["correct_index"],
                    "surface_sha256": {
                        name: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                        for name, prompt in case["surfaces"].items()
                    },
                }
                for case in cases
            ],
        })
        for case in cases:
            names = sorted(case["surfaces"])
            predictions = {
                name: choose_content(scorer, case["surfaces"][name])
                for name in names
            }
            primary = names[0]
            secondary = names[1]
            selected = ROUTER[case["family"]]
            if selected not in predictions:
                raise RuntimeError(
                    f"router selected unavailable surface {selected!r} for {case['family']}"
                )
            fused_scores = [
                a + b
                for a, b in zip(
                    log_softmax(predictions[primary]["pmi_scores"]),
                    log_softmax(predictions[secondary]["pmi_scores"]),
                    strict=True,
                )
            ]
            rows.append({
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": case["correct_index"],
                "primary_surface": primary,
                "secondary_surface": secondary,
                "router_surface": selected,
                "primary_prediction": predictions[primary]["pmi_prediction"],
                "secondary_prediction": predictions[secondary]["pmi_prediction"],
                "router_prediction": predictions[selected]["pmi_prediction"],
                "fusion_prediction": argmax(fused_scores),
                "surface_agreement": (
                    predictions[primary]["pmi_prediction"]
                    == predictions[secondary]["pmi_prediction"]
                ),
            })

    arms = {
        "lexicographic-primary-surface": summarize(rows, "primary_prediction"),
        "alternate-secondary-surface": summarize(rows, "secondary_prediction"),
        "naive-log-probability-fusion": summarize(rows, "fusion_prediction"),
        "fixed-family-representation-router": summarize(rows, "router_prediction"),
    }
    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-transfer-calibration",
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "source_battery_sha256": sha256_json(public_specs),
        "frozen_router": ROUTER,
        "router_training_source": {
            "run": "021-offline-semantic-content-scoring",
            "seed": 20260917,
            "artifact_sha256": "f7846f03e51e8d7548bf299d8cfe63d919a1fae9dc29313401cd424f575736ee",
        },
        "arms": arms,
        "surface_agreement": sum(row["surface_agreement"] for row in rows) / len(rows),
        "observed_surface_scores": len(rows) * 2,
        "deployable_router_surface_scores": len(rows),
        "router_extra_learned_parameters": 0,
        "router_test_label_access": False,
        "rows": rows,
        "elapsed_seconds": time.time() - started,
        "claim_boundary": (
            "Fresh procedural public transfer calibration. The router was fixed before "
            "test-seed generation and receives only public task-family identity, never "
            "test labels. This is not protected promotion evidence or frontier equivalence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seeds", default="20260918,20260919,20260920")
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--output", default="run-027-router.json")
    args = parser.parse_args()

    model_path = Path(args.model_file)
    observed = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    result = evaluate(args.model_dir, seeds, args.cases_per_family)
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
        "source_battery_sha256": result["source_battery_sha256"],
        "arms": result["arms"],
        "surface_agreement": result["surface_agreement"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
