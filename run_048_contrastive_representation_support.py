"""Run 048: contrastive representation-support selector.

Primary frozen rule:
    delta(candidate) = mean(exact-operator normalized support)
                     - mean(sibling normalized support)

The rule was suggested by burned Run 044 inspection and is tested here on fresh
procedural cases. No parameters are fit and no correctness enters selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from offline_reasoning_battery import build_battery, sha256_json
from offline_reasoning_content_probe import surface_task_and_choices
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import build_representations, _score_values
from run_039_exact_representation_operators import operators_for_case

RUN_VERSION = "helix-contrastive-representation-support-001.0"
SIBLING_NAMES = ("primary", "secondary", "layout", "familiar")
OPERATOR_NAMES = ("operator-1", "operator-2")
ALL_NAMES = SIBLING_NAMES + OPERATOR_NAMES


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def _mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("empty vector set")
    width = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(width)]


def _majority(predictions: list[int]) -> int:
    counts = Counter(predictions)
    best = max(counts.values())
    return min(i for i, count in counts.items() if count == best)


def build_views(case: dict[str, Any]) -> dict[str, tuple[str, list[str]]]:
    surfaces = case["surfaces"]
    surface_names = sorted(surfaces)
    primary = surface_task_and_choices(surfaces[surface_names[0]])
    secondary = surface_task_and_choices(surfaces[surface_names[1]])
    canonical = build_representations(case["family"], surfaces)
    operators = operators_for_case(case["family"], surfaces)
    if len(operators) != 2:
        raise AssertionError("expected two exact operators")
    return {
        "primary": primary,
        "secondary": secondary,
        "layout": canonical["layout"],
        "familiar": canonical["familiar"],
        "operator-1": (operators[0][1] + "\nAnswer value:", operators[0][2]),
        "operator-2": (operators[1][1] + "\nAnswer value:", operators[1][2]),
    }


def self_test(seed: int = 20261115, cases_per_family: int = 2) -> dict[str, Any]:
    cases = build_battery(seed, cases_per_family)
    digests = []
    counts = Counter()
    for case in cases:
        views = build_views(case)
        if tuple(views) != ALL_NAMES:
            raise AssertionError((case["case_id"], tuple(views)))
        for name, (prompt, choices) in views.items():
            if len(choices) != 4 or not prompt.endswith("Answer value:"):
                raise AssertionError((case["case_id"], name))
            counts[f"{case['family']}/{name}"] += 1
            digests.append(hashlib.sha256(json.dumps(
                {"family": case["family"], "name": name, "prompt": prompt, "choices": choices},
                sort_keys=True,
            ).encode()).hexdigest())
    return {
        "schema_version": RUN_VERSION,
        "cases": len(cases),
        "views_per_case": len(ALL_NAMES),
        "view_instances": sum(counts.values()),
        "counts": dict(sorted(counts.items())),
        "transform_digest": hashlib.sha256("|".join(digests).encode()).hexdigest(),
        "selector_correctness_input": False,
        "learned_parameters": 0,
    }


def _summary(rows: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    correct = sum(row["selectors"][selector] == row["correct_index"] for row in rows)
    by_family: dict[str, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        ok = row["selectors"][selector] == row["correct_index"]
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
            scores = {name: _score_values(scorer, prompt, choices) for name, (prompt, choices) in views.items()}
            vectors = {name: [float(v) for v in row["scores"]] for name, row in scores.items()}
            predictions = {name: int(row["prediction"]) for name, row in scores.items()}

            sibling_mean = _mean([vectors[name] for name in SIBLING_NAMES])
            operator_mean = _mean([vectors[name] for name in OPERATOR_NAMES])
            all_mean = _mean([vectors[name] for name in ALL_NAMES])
            delta = [operator_mean[i] - sibling_mean[i] for i in range(4)]
            inverse_delta = [-v for v in delta]

            selectors = {
                "layout-fixed": predictions["layout"],
                "sibling-score-mean": _argmax(sibling_mean),
                "operator-score-mean": _argmax(operator_mean),
                "all-six-score-mean": _argmax(all_mean),
                "contrastive-operator-minus-sibling": _argmax(delta),
                "inverse-sibling-minus-operator": _argmax(inverse_delta),
                "sibling-majority": _majority([predictions[name] for name in SIBLING_NAMES]),
                "all-six-majority": _majority([predictions[name] for name in ALL_NAMES]),
            }

            correct = int(case["correct_index"])
            sibling_has_correct = any(predictions[name] == correct for name in SIBLING_NAMES)
            operator_has_correct = any(predictions[name] == correct for name in OPERATOR_NAMES)
            strict_operator_rescue = (not sibling_has_correct) and operator_has_correct

            rows.append({
                "uid": uid,
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "predictions": predictions,
                "selectors": selectors,
                "candidate_support": {
                    "sibling_mean": sibling_mean,
                    "operator_mean": operator_mean,
                    "operator_minus_sibling": delta,
                },
                "sibling_has_correct": sibling_has_correct,
                "operator_has_correct": operator_has_correct,
                "strict_operator_rescue": strict_operator_rescue,
                "operator_predictions_disagree": predictions["operator-1"] != predictions["operator-2"],
            })
            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "view_sha256": {
                    name: hashlib.sha256(json.dumps(
                        {"prompt": prompt, "choices": choices},
                        sort_keys=True,
                    ).encode()).hexdigest()
                    for name, (prompt, choices) in views.items()
                },
            })
        public_spec.append({"seed": seed, "cases": seed_spec})

    selector_names = tuple(rows[0]["selectors"])
    summaries = {name: _summary(rows, name) for name in selector_names}
    primary = "contrastive-operator-minus-sibling"
    paired = {
        "vs-sibling-score-mean": _paired(rows, primary, "sibling-score-mean"),
        "vs-all-six-score-mean": _paired(rows, primary, "all-six-score-mean"),
        "vs-inverse": _paired(rows, primary, "inverse-sibling-minus-operator"),
    }

    sibling_coverage = sum(row["sibling_has_correct"] for row in rows)
    six_coverage = sum(row["sibling_has_correct"] or row["operator_has_correct"] for row in rows)
    rescue_rows = [row for row in rows if row["strict_operator_rescue"]]
    rescue_primary_correct = sum(row["selectors"][primary] == row["correct_index"] for row in rescue_rows)

    conditionals = {}
    for state in (False, True):
        subset = [row for row in rows if row["operator_predictions_disagree"] is state]
        conditionals["operator-disagree" if state else "operator-agree"] = {
            "cases": len(subset),
            "contrastive_correct": sum(row["selectors"][primary] == row["correct_index"] for row in subset),
            "contrastive_accuracy": (
                sum(row["selectors"][primary] == row["correct_index"] for row in subset) / len(subset)
                if subset else 0.0
            ),
            "strict_operator_rescues": sum(row["strict_operator_rescue"] for row in subset),
        }

    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-contrastive-support-calibration",
        "actor": {
            "model": "HuggingFaceTB/SmolLM2-360M",
            "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
            "weights_frozen": True,
        },
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(rows),
        "source_battery_sha256": sha256_json(public_spec),
        "selectors": summaries,
        "paired": paired,
        "candidate_coverage": {
            "sibling_four_view_correct": sibling_coverage,
            "sibling_four_view_coverage": sibling_coverage / len(rows),
            "six_view_correct": six_coverage,
            "six_view_coverage": six_coverage / len(rows),
            "strict_exact_operator_rescues": len(rescue_rows),
        },
        "strict_operator_rescue_selection": {
            "cases": len(rescue_rows),
            "contrastive_correct": rescue_primary_correct,
            "contrastive_accuracy": rescue_primary_correct / len(rescue_rows) if rescue_rows else 0.0,
        },
        "conditionals": conditionals,
        "resources": resources,
        "rows": rows,
        "claim_boundary": (
            "Fresh public procedural test of a preregistered parameter-free contrastive selector. "
            "The motivating Run 044 observation was burned and contributes no evidence here. "
            "Candidate coverage is not selected-answer accuracy. No promotion or frontier claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--seeds", default="20261115,20261116,20261117")
    parser.add_argument("--cases-per-family", type=int, default=6)
    parser.add_argument("--output", default="run-048-contrastive-representation-support.json")
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
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema_version": result["schema_version"],
        "cases": result["cases"],
        "selectors": result["selectors"],
        "paired": result["paired"],
        "candidate_coverage": result["candidate_coverage"],
        "strict_operator_rescue_selection": result["strict_operator_rescue_selection"],
        "conditionals": result["conditionals"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
