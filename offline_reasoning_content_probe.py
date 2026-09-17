"""Run 021: remove the A/B/C/D bottleneck from Run 020.

Uses the exact same fresh procedural cases and surfaces, but strips the displayed
choice labels before scoring and compares the semantic answer strings directly.
This is a measurement correction, not a new benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from offline_reasoning_battery import (
    BATTERY_VERSION,
    ChoiceScorer,
    DONORS,
    build_battery,
    sha256_json,
)


PROBE_VERSION = "helix-offline-content-choice-001.0"


def surface_task_and_choices(prompt: str) -> tuple[str, list[str]]:
    """Recover the task stem and surface-native answer values from Run 020 prompts."""
    if "\nChoices:\n" in prompt:
        stem, tail = prompt.rsplit("\nChoices:\n", 1)
        block, marker = tail.rsplit("\nAnswer:", 1)
        if marker:
            raise ValueError("unexpected prose answer suffix")
        lines = block.splitlines()
        if len(lines) != 4:
            raise ValueError("expected four prose choices")
        choices = []
        for index, line in enumerate(lines):
            prefix = f"{chr(65 + index)}. "
            if not line.startswith(prefix):
                raise ValueError("malformed prose choice")
            choices.append(line[len(prefix):])
        return stem + "\nAnswer value:", choices

    lines = prompt.splitlines()
    if len(lines) < 3 or lines[-1] != "Answer:":
        raise ValueError("malformed compact surface")
    option_line = lines[-2]
    stem = "\n".join(lines[:-2])
    parts = option_line.split(" | ")
    if len(parts) != 4:
        raise ValueError("expected four compact choices")
    choices = []
    for index, part in enumerate(parts):
        prefix = f"{chr(65 + index)}="
        if not part.startswith(prefix):
            raise ValueError("malformed compact choice")
        choices.append(part[len(prefix):])
    return stem + "\nAnswer value:", choices


def choose_content(scorer: ChoiceScorer, prompt: str) -> dict[str, Any]:
    task_prompt, choices = surface_task_and_choices(prompt)
    raw = scorer.score_choices(task_prompt, choices)
    prior = scorer.score_choices("Answer value:", choices)
    pmi = [raw[i] - prior[i] for i in range(4)]
    raw_pred = max(range(4), key=lambda i: raw[i])
    pmi_pred = max(range(4), key=lambda i: pmi[i])
    raw_sorted = sorted(raw, reverse=True)
    pmi_sorted = sorted(pmi, reverse=True)
    return {
        "task_prompt_sha256": hashlib.sha256(task_prompt.encode("utf-8")).hexdigest(),
        "choice_sha256": [hashlib.sha256(c.encode("utf-8")).hexdigest() for c in choices],
        "raw_scores": raw,
        "pmi_scores": pmi,
        "raw_prediction": raw_pred,
        "pmi_prediction": pmi_pred,
        "raw_margin": raw_sorted[0] - raw_sorted[1],
        "pmi_margin": pmi_sorted[0] - pmi_sorted[1],
    }


def evaluate(model_dir: str, seed: int, cases_per_family: int) -> dict[str, Any]:
    cases = build_battery(seed, cases_per_family)
    scorer = ChoiceScorer(model_dir)
    started = time.time()
    rows = []
    for case in cases:
        predictions = {
            surface: choose_content(scorer, prompt)
            for surface, prompt in case["surfaces"].items()
        }
        rows.append({
            "case_id": case["case_id"],
            "family": case["family"],
            "difficulty": case["difficulty"],
            "correct_index": case["correct_index"],
            "predictions": predictions,
        })

    family_results: dict[str, Any] = {}
    for family in sorted({r["family"] for r in rows}):
        fam = [r for r in rows if r["family"] == family]
        surfaces = {}
        for surface in sorted(fam[0]["predictions"]):
            raw_correct = [
                r["predictions"][surface]["raw_prediction"] == r["correct_index"]
                for r in fam
            ]
            pmi_correct = [
                r["predictions"][surface]["pmi_prediction"] == r["correct_index"]
                for r in fam
            ]
            surfaces[surface] = {
                "cases": len(fam),
                "raw_accuracy": sum(raw_correct) / len(fam),
                "pmi_accuracy": sum(pmi_correct) / len(fam),
                "easy_pmi_accuracy": sum(
                    r["predictions"][surface]["pmi_prediction"] == r["correct_index"]
                    for r in fam if r["difficulty"] == "easy"
                ) / max(1, sum(r["difficulty"] == "easy" for r in fam)),
                "hard_pmi_accuracy": sum(
                    r["predictions"][surface]["pmi_prediction"] == r["correct_index"]
                    for r in fam if r["difficulty"] == "hard"
                ) / max(1, sum(r["difficulty"] == "hard" for r in fam)),
                "mean_raw_margin": sum(
                    r["predictions"][surface]["raw_margin"] for r in fam
                ) / len(fam),
                "mean_pmi_margin": sum(
                    r["predictions"][surface]["pmi_margin"] for r in fam
                ) / len(fam),
                "predicted_position_counts": dict(Counter(
                    str(r["predictions"][surface]["pmi_prediction"])
                    for r in fam
                )),
            }
        family_results[family] = {
            "surfaces": surfaces,
            "surface_prediction_agreement": sum(
                len({p["pmi_prediction"] for p in r["predictions"].values()}) == 1
                for r in fam
            ) / len(fam),
            "all_surfaces_correct": sum(
                all(p["pmi_prediction"] == r["correct_index"] for p in r["predictions"].values())
                for r in fam
            ) / len(fam),
        }

    decisions = [(r, p) for r in rows for p in r["predictions"].values()]
    overall = {
        "cases": len(rows),
        "surface_decisions": len(decisions),
        "chance_accuracy": 0.25,
        "raw_accuracy": sum(p["raw_prediction"] == r["correct_index"] for r, p in decisions) / len(decisions),
        "pmi_accuracy": sum(p["pmi_prediction"] == r["correct_index"] for r, p in decisions) / len(decisions),
        "case_surface_agreement": sum(
            len({p["pmi_prediction"] for p in r["predictions"].values()}) == 1
            for r in rows
        ) / len(rows),
        "all_surfaces_correct": sum(
            all(p["pmi_prediction"] == r["correct_index"] for p in r["predictions"].values())
            for r in rows
        ) / len(rows),
        "elapsed_seconds": time.time() - started,
    }

    public_spec = [
        {
            "case_id": c["case_id"],
            "family": c["family"],
            "difficulty": c["difficulty"],
            "correct_index": c["correct_index"],
            "surface_sha256": {
                k: hashlib.sha256(v.encode("utf-8")).hexdigest()
                for k, v in c["surfaces"].items()
            },
        }
        for c in cases
    ]
    return {
        "schema_version": PROBE_VERSION,
        "source_battery_version": BATTERY_VERSION,
        "source_battery_sha256": sha256_json(public_spec),
        "status": "public-burned-calibration",
        "seed": seed,
        "cases_per_family": cases_per_family,
        "donors": DONORS,
        "scoring": {
            "primary": "length-normalized teacher-forced semantic answer-content likelihood",
            "secondary": "pointwise mutual-information style normalization against neutral 'Answer value:' context",
            "labels_removed_before_scoring": True,
            "answer_positions_balanced_per_family": True,
            "single_scalar_score_authoritative": False,
        },
        "family_results": family_results,
        "overall_diagnostic": overall,
        "cases": rows,
        "claim_boundary": (
            "Measurement-correction run on the same fresh public calibration cases as Run 020. "
            "Not an official donor benchmark score, not protected evidence, does not establish AGI, "
            "and cannot authorize model promotion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--output", default="offline-content-choice.json")
    args = parser.parse_args()

    model_path = Path(args.model_file)
    observed = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    result = evaluate(args.model_dir, args.seed, args.cases_per_family)
    result["actor"] = {
        "model": "HuggingFaceTB/SmolLM2-360M",
        "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        "model_file_sha256": observed,
        "model_bytes": model_path.stat().st_size,
        "dtype": "float32",
        "device": "cpu",
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema_version": result["schema_version"],
        "source_battery_sha256": result["source_battery_sha256"],
        "family_results": result["family_results"],
        "overall_diagnostic": result["overall_diagnostic"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
