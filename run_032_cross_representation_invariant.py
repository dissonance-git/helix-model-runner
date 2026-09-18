"""Run 032: cross-representation invariant synthesis.

The exact frozen SmolLM2-360M actor sees several obligation-preserving views of
the same public procedural task. It first summarizes each view without answer
choices, then synthesizes a shared state from the summaries, and only afterward
scores answer values.

This tests whether cross-view structure can become a new reasoning object rather
than merely selecting or averaging an existing representation.
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

from offline_reasoning_battery import build_battery, sha256_json
from offline_reasoning_content_probe import surface_task_and_choices
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import build_representations

RUN_VERSION = "helix-cross-representation-invariant-001.0"


def log_softmax(values: list[float]) -> list[float]:
    m = max(values)
    z = m + math.log(sum(math.exp(v - m) for v in values))
    return [v - z for v in values]


def score_values(
    scorer: AttackScorer,
    prompt: str,
    choices: list[str],
) -> dict[str, Any]:
    raw = scorer.score_choices(prompt, choices)
    prior = scorer.score_choices("Answer value:", choices)
    pmi = [raw[i] - prior[i] for i in range(len(choices))]
    normalized = log_softmax(pmi)
    order = sorted(normalized, reverse=True)
    return {
        "prediction": max(range(len(choices)), key=lambda i: normalized[i]),
        "scores": normalized,
        "margin": order[0] - order[1],
        "entropy": -sum(math.exp(x) * x for x in normalized),
    }


def original_representation(prompt: str) -> tuple[str, list[str]]:
    task, choices = surface_task_and_choices(prompt)
    return task, choices


def strip_answer_marker(prompt: str) -> str:
    suffix = "\nAnswer value:"
    if prompt.endswith(suffix):
        return prompt[:-len(suffix)]
    if prompt.endswith("Answer value:"):
        return prompt[:-len("Answer value:")].rstrip()
    raise ValueError("representation prompt lacks answer marker")


def summarize_view(
    scorer: AttackScorer,
    *,
    representation_name: str,
    prompt: str,
    seed: int,
) -> str:
    visible = strip_answer_marker(prompt)
    request = (
        "Extract a compact operational state from one representation of a reasoning task.\n"
        "Do not solve the task. Do not guess an answer. No answer choices are available.\n"
        "Preserve only explicit entities, relations, transformations, constraints, and the question target.\n"
        "Remove layout and wording details that are not structurally relevant.\n"
        f"Representation: {representation_name}\n"
        "Task:\n"
        f"{visible}\n"
        "Operational state:"
    )
    return scorer.generate_text(
        request,
        max_new_tokens=48,
        seed=seed,
        sample=False,
    )


def synthesize_invariant(
    scorer: AttackScorer,
    summaries: dict[str, str],
    *,
    seed: int,
) -> str:
    blocks = "\n\n".join(
        f"[{name}]\n{text}"
        for name, text in summaries.items()
    )
    request = (
        "These are independently derived states for equivalent representations of one reasoning task.\n"
        "Construct the smallest shared state that preserves the task obligation.\n"
        "Do not solve the task and do not choose an answer. No answer choices are available.\n"
        "Keep relations that agree across views. If views disagree, preserve the disagreement explicitly instead of silently choosing one.\n"
        "Collapse wording/layout differences, but do not collapse a distinction that could change the answer.\n"
        "Preserve the question target.\n\n"
        f"{blocks}\n\n"
        "Shared invariant state:"
    )
    return scorer.generate_text(
        request,
        max_new_tokens=64,
        seed=seed,
        sample=False,
    )


def rank_order(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def deterministic_quotients(
    score_rows: dict[str, dict[str, Any]]
) -> dict[str, int]:
    names = list(score_rows)
    count = len(next(iter(score_rows.values()))["scores"])

    summed = [
        sum(score_rows[name]["scores"][i] for name in names)
        for i in range(count)
    ]

    borda = [0.0] * count
    for name in names:
        order = rank_order(score_rows[name]["scores"])
        for rank, index in enumerate(order):
            borda[index] += count - 1 - rank

    maximin = [
        min(score_rows[name]["scores"][i] for name in names)
        for i in range(count)
    ]

    median = [
        statistics.median(score_rows[name]["scores"][i] for name in names)
        for i in range(count)
    ]

    def choose(values: list[float], fallback: list[float]) -> int:
        best = max(values)
        tied = [i for i, value in enumerate(values) if value == best]
        if len(tied) == 1:
            return tied[0]
        return max(tied, key=lambda i: (fallback[i], -i))

    return {
        "representation-score-sum": choose(summed, summed),
        "representation-borda-quotient": choose(borda, summed),
        "representation-maximin-quotient": choose(maximin, summed),
        "representation-median-quotient": choose(median, summed),
    }


def summarize_predictions(
    rows: list[dict[str, Any]],
    prediction_key: str,
) -> dict[str, Any]:
    by_family: dict[str, list[bool]] = defaultdict(list)
    by_seed: dict[int, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    hits = 0
    for row in rows:
        ok = int(row[prediction_key]) == int(row["correct_index"])
        hits += int(ok)
        by_family[row["family"]].append(ok)
        by_seed[int(row["seed"])].append(ok)
        by_difficulty[row["difficulty"]].append(ok)
    return {
        "cases": len(rows),
        "correct": hits,
        "accuracy": hits / len(rows),
        "by_family": {
            key: sum(values) / len(values)
            for key, values in sorted(by_family.items())
        },
        "by_seed": {
            str(key): sum(values) / len(values)
            for key, values in sorted(by_seed.items())
        },
        "by_difficulty": {
            key: sum(values) / len(values)
            for key, values in sorted(by_difficulty.items())
        },
    }


def representations_for_case(case: dict[str, Any]) -> dict[str, tuple[str, list[str]]]:
    names = sorted(case["surfaces"])
    transformed = build_representations(case["family"], case["surfaces"])
    return {
        "lexicographic-primary-surface": original_representation(case["surfaces"][names[0]]),
        "alternate-secondary-surface": original_representation(case["surfaces"][names[1]]),
        "layout-normalized": transformed["layout"],
        "familiar-structural-normalization": transformed["familiar"],
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
            quotients = deterministic_quotients(rep_scores)

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
            invariant_scored = score_values(
                scorer,
                invariant_prompt,
                layout_choices,
            )

            original_predictions = {
                name: int(value["prediction"])
                for name, value in rep_scores.items()
            }
            correct_reps = [
                name for name, prediction in original_predictions.items()
                if prediction == case["correct_index"]
            ]
            original_oracle = bool(correct_reps)
            invariant_correct = invariant_scored["prediction"] == case["correct_index"]
            unique_original_predictions = set(original_predictions.values())
            original_all_agree = len(unique_original_predictions) == 1

            row = {
                "uid": uid,
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": case["correct_index"],
                "fixed_layout_prediction": original_predictions["layout-normalized"],
                "original_predictions": original_predictions,
                "original_margins": {
                    name: float(value["margin"])
                    for name, value in rep_scores.items()
                },
                **{key.replace("-", "_") + "_prediction": int(value) for key, value in quotients.items()},
                "invariant_prediction": int(invariant_scored["prediction"]),
                "invariant_margin": float(invariant_scored["margin"]),
                "original_oracle_contains_correct": original_oracle,
                "correct_original_representations": correct_reps,
                "original_all_agree": original_all_agree,
                "invariant_adds_new_correct_candidate": bool(invariant_correct and not original_oracle),
                "invariant_changes_unanimous_field": bool(
                    original_all_agree
                    and invariant_scored["prediction"] not in unique_original_predictions
                ),
                "invariant_changes_unanimous_field_correctly": bool(
                    original_all_agree
                    and not original_oracle
                    and invariant_correct
                ),
                "summary_sha256": {
                    name: hashlib.sha256(text.encode()).hexdigest()
                    for name, text in summaries.items()
                },
                "invariant_sha256": hashlib.sha256(invariant.encode()).hexdigest(),
                "summaries": summaries,
                "invariant": invariant,
            }
            rows.append(row)

            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": case["correct_index"],
                "source_surface_sha256": {
                    name: hashlib.sha256(prompt.encode()).hexdigest()
                    for name, prompt in case["surfaces"].items()
                },
                "derived_representation_sha256": {
                    name: hashlib.sha256(
                        json.dumps(
                            {"prompt": prompt, "choices": choices},
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()
                    for name, (prompt, choices) in reps.items()
                },
            })
        public_specs.append({"seed": seed, "cases": seed_spec})

    arms = {
        "fixed-layout": summarize_predictions(rows, "fixed_layout_prediction"),
        "representation-score-sum": summarize_predictions(
            rows, "representation_score_sum_prediction"
        ),
        "representation-borda-quotient": summarize_predictions(
            rows, "representation_borda_quotient_prediction"
        ),
        "representation-maximin-quotient": summarize_predictions(
            rows, "representation_maximin_quotient_prediction"
        ),
        "representation-median-quotient": summarize_predictions(
            rows, "representation_median_quotient_prediction"
        ),
        "answer-blind-semantic-invariant-synthesis": summarize_predictions(
            rows, "invariant_prediction"
        ),
    }

    original_oracle = sum(row["original_oracle_contains_correct"] for row in rows)
    expanded_oracle = sum(
        row["original_oracle_contains_correct"]
        or row["invariant_prediction"] == row["correct_index"]
        for row in rows
    )
    new_correct = sum(row["invariant_adds_new_correct_candidate"] for row in rows)
    unanimous_changes = sum(row["invariant_changes_unanimous_field"] for row in rows)
    unanimous_repairs = sum(
        row["invariant_changes_unanimous_field_correctly"] for row in rows
    )

    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started
    resources["representations_scored_per_case_before_synthesis"] = 4
    resources["summary_generations_per_case"] = 4
    resources["invariant_generations_per_case"] = 1

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-cross-representation-calibration",
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(rows),
        "source_battery_sha256": sha256_json(public_specs),
        "synthesis_contract": {
            "summary_inputs": "representation task stem with answer choices removed",
            "synthesis_inputs": "four actor-generated answer-blind summaries",
            "synthesis_output": "shared state preserving agreements, disagreements, and question target",
            "judge_visible_to_synthesis": False,
            "correct_index_visible_to_synthesis": False,
            "answer_choices_visible_to_summary_or_synthesis_generation": False,
            "same_frozen_actor_used_for_summary_synthesis_and_scoring": True,
        },
        "arms": arms,
        "coverage": {
            "original_four_view_oracle_correct": original_oracle,
            "original_four_view_oracle": original_oracle / len(rows),
            "original_plus_synthesized_oracle_correct": expanded_oracle,
            "original_plus_synthesized_oracle": expanded_oracle / len(rows),
            "synthesized_new_correct_candidates": new_correct,
            "synthesized_new_correct_fraction": new_correct / len(rows),
            "unanimous_original_fields_changed": unanimous_changes,
            "unanimous_wrong_fields_repaired": unanimous_repairs,
        },
        "resources": resources,
        "rows": rows,
        "claim_boundary": (
            "Fresh public procedural calibration only. Cross-view agreement is never "
            "treated as truth; external labels score consequences after synthesis. "
            "This run cannot authorize promotion, frontier equivalence, or a general "
            "recursive-improvement claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seeds", default="20261003,20261004")
    parser.add_argument("--cases-per-family", type=int, default=6)
    parser.add_argument("--output", default="run-032-cross-view.json")
    args = parser.parse_args()

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
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema_version": result["schema_version"],
        "arms": result["arms"],
        "coverage": result["coverage"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
