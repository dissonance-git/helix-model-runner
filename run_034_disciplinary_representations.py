"""Run 034: disciplinary representation diversity.

The same frozen SmolLM2-360M actor receives a deterministic layout-normalized
task representation with answer choices removed and generates an operational
state through deliberately distant Helix field lenses. The same actor then
scores the original answer values from each generated state.

No stronger teacher authors the field representations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from offline_reasoning_battery import build_battery, sha256_json
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import build_representations, _score_values

RUN_VERSION = "helix-disciplinary-representation-diversity-001.0"

FIELD_QUESTIONS = {
    "physics": [
        "What is conserved?",
        "Which symmetry or constraint organizes the state space?",
        "What interactions generate the trajectory?",
    ],
    "chemistry": [
        "What transformations are allowed?",
        "What buffering or equilibrium structure matters?",
        "Which pathway or catalyst changes the reachable state?",
    ],
    "economics": [
        "What scarce resource or incentive structure shapes the outcome?",
        "What substitution or opportunity cost is hidden?",
        "Does local optimization create a system-level externality?",
    ],
    "topology": [
        "What survives deformation?",
        "Where does qualitative structure change?",
        "Which boundaries separate reachable regions?",
    ],
    "information_theory": [
        "What information must survive?",
        "What is compressible without changing the obligation?",
        "Where is information lost, duplicated, or bottlenecked?",
    ],
    "biology": [
        "What survives perturbation?",
        "What adapts, regenerates, or compensates?",
        "Is redundancy actually degeneracy or response diversity?",
    ],
    "history": [
        "What earlier state made the present reachable?",
        "What is path-dependent?",
        "Which apparently similar states have different provenance?",
    ],
    "game_theory": [
        "What changes when another actor optimizes too?",
        "Which equilibrium is stable under unilateral deviation?",
        "What incentive creates the observed behavior?",
    ],
}

FIELDS = tuple(FIELD_QUESTIONS)


def strip_answer_marker(prompt: str) -> str:
    suffix = "Answer value:"
    value = prompt.rstrip()
    if not value.endswith(suffix):
        raise ValueError("representation lacks answer marker")
    return value[:-len(suffix)].rstrip()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def field_state(
    scorer: AttackScorer,
    *,
    field: str,
    task_state: str,
    seed: int,
) -> str:
    questions = "\n".join(
        f"- {question}" for question in FIELD_QUESTIONS[field]
    )
    prompt = (
        "Translate one reasoning obligation into a materially different field-native representation.\n"
        "Do not solve the task. Do not guess an answer. No answer choices are available.\n"
        "Preserve every distinction needed to answer the original obligation.\n"
        "Use the field's native objects, relations, constraints, transformations, and failure conditions.\n"
        "Do not merely rename nouns: make the field's characteristic structure explicit.\n"
        "If the field analogy would add unsupported semantics, preserve the exact original relation instead and mark the analogy limit.\n"
        f"FIELD: {field}\n"
        "FIELD-NATIVE QUESTIONS:\n"
        f"{questions}\n\n"
        "SOURCE TASK STATE:\n"
        f"{task_state}\n\n"
        "FIELD-NATIVE OPERATIONAL STATE:"
    )
    return scorer.generate_text(
        prompt,
        max_new_tokens=56,
        seed=seed,
        sample=False,
    )


def score_state(
    scorer: AttackScorer,
    state: str,
    choices: list[str],
) -> dict[str, Any]:
    prompt = (
        "Use this derived representation to answer the original reasoning task.\n"
        "Derived state:\n"
        f"{state}\n"
        "Answer value:"
    )
    return _score_values(scorer, prompt, choices)


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    hits = 0
    by_family: dict[str, list[bool]] = defaultdict(list)
    by_seed: dict[int, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        ok = int(row[key]) == int(row["correct_index"])
        hits += int(ok)
        by_family[row["family"]].append(ok)
        by_seed[int(row["seed"])].append(ok)
        by_difficulty[row["difficulty"]].append(ok)
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
        "by_difficulty": {
            k: sum(v) / len(v) for k, v in sorted(by_difficulty.items())
        },
    }


def pairwise_metrics(
    rows: list[dict[str, Any]],
    correct_sets: dict[str, set[str]],
) -> dict[str, Any]:
    out = {}
    names = ["layout", *FIELDS]
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a = correct_sets[left]
            b = correct_sets[right]
            union = a | b
            intersection = a & b
            out[f"{left}+{right}"] = {
                "left_only_correct": len(a - b),
                "right_only_correct": len(b - a),
                "both_correct": len(intersection),
                "oracle_correct": len(union),
                "oracle_coverage": len(union) / len(rows),
                "jaccard_correct_sets": (
                    len(intersection) / len(union) if union else 1.0
                ),
            }
    return out


def evaluate(
    model_dir: str,
    seeds: list[int],
    cases_per_family: int,
) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()
    started = time.time()
    rows: list[dict[str, Any]] = []
    public_spec = []

    for seed in seeds:
        cases = build_battery(seed, cases_per_family)
        seed_spec = []
        for local_index, case in enumerate(cases):
            uid = f"{seed}:{case['case_id']}"
            transformed = build_representations(
                case["family"],
                case["surfaces"],
            )
            layout_prompt, layout_choices = transformed["layout"]
            layout_state = strip_answer_marker(layout_prompt)
            layout_score = _score_values(
                scorer,
                layout_prompt,
                layout_choices,
            )

            states = {}
            field_scores = {}
            for field_index, field in enumerate(FIELDS):
                state = field_state(
                    scorer,
                    field=field,
                    task_state=layout_state,
                    seed=seed * 1000 + local_index * 20 + field_index,
                )
                states[field] = state
                field_scores[field] = score_state(
                    scorer,
                    state,
                    layout_choices,
                )

            predictions = {
                "layout": int(layout_score["prediction"]),
                **{
                    field: int(field_scores[field]["prediction"])
                    for field in FIELDS
                },
            }
            correct = int(case["correct_index"])
            correct_views = [
                name for name, prediction in predictions.items()
                if prediction == correct
            ]
            field_only = [
                field for field in FIELDS
                if predictions[field] == correct
                and predictions["layout"] != correct
            ]

            similarity = {}
            for i, left in enumerate(FIELDS):
                for right in FIELDS[i + 1:]:
                    similarity[f"{left}+{right}"] = SequenceMatcher(
                        None,
                        normalize_text(states[left]),
                        normalize_text(states[right]),
                    ).ratio()

            rows.append({
                "uid": uid,
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
                "correct_views": correct_views,
                "field_only_correct": field_only,
                "state_sha256": {
                    field: hashlib.sha256(states[field].encode()).hexdigest()
                    for field in FIELDS
                },
                "state_lengths_chars": {
                    field: len(states[field]) for field in FIELDS
                },
                "field_state_similarity": similarity,
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
                "layout_sha256": hashlib.sha256(layout_state.encode()).hexdigest(),
            })
        public_spec.append({"seed": seed, "cases": seed_spec})

    summaries = {
        "layout": summarize(rows, "layout_prediction"),
        **{
            field: summarize(rows, f"{field}_prediction")
            for field in FIELDS
        },
    }

    correct_sets = {
        "layout": {
            row["uid"] for row in rows
            if row["layout_prediction"] == row["correct_index"]
        },
        **{
            field: {
                row["uid"] for row in rows
                if row[f"{field}_prediction"] == row["correct_index"]
            }
            for field in FIELDS
        },
    }
    oracle_set = set().union(*correct_sets.values())
    layout_set = correct_sets["layout"]
    field_union = set().union(*(correct_sets[field] for field in FIELDS))
    exactly_one_field = []
    for row in rows:
        fields_correct = [
            field for field in FIELDS
            if row[f"{field}_prediction"] == row["correct_index"]
        ]
        if len(fields_correct) == 1:
            exactly_one_field.append({
                "uid": row["uid"],
                "field": fields_correct[0],
                "layout_correct": row["layout_prediction"] == row["correct_index"],
            })

    similarities = [
        score
        for row in rows
        for score in row["field_state_similarity"].values()
    ]
    near_identical = sum(score >= 0.90 for score in similarities)

    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started
    resources["field_generations_per_case"] = len(FIELDS)
    resources["field_scoring_arms_per_case"] = len(FIELDS)

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-disciplinary-diversity-calibration",
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(rows),
        "helix_lens_source": {
            "commit": "1984a42ff8c50b35227a89f1865d6e68c86f9cc8",
            "owner": "engine/analyze/methods/_routing_core.py",
            "fields": list(FIELDS),
        },
        "source_battery_sha256": sha256_json(public_spec),
        "generation_contract": {
            "source_representation": "layout-normalized",
            "answer_choices_visible_during_generation": False,
            "correct_index_visible_during_generation": False,
            "same_frozen_actor_generates_and_scores": True,
        },
        "summaries": summaries,
        "coverage": {
            "layout_correct": len(layout_set),
            "layout_accuracy": len(layout_set) / len(rows),
            "any_field_correct": len(field_union),
            "any_field_coverage": len(field_union) / len(rows),
            "nine_view_oracle_correct": len(oracle_set),
            "nine_view_oracle_coverage": len(oracle_set) / len(rows),
            "new_correct_cases_from_fields_beyond_layout": len(field_union - layout_set),
            "new_correct_fraction_from_fields_beyond_layout": len(field_union - layout_set) / len(rows),
            "cases_solved_by_exactly_one_field": len(exactly_one_field),
            "exactly_one_field_rows": exactly_one_field,
        },
        "pairwise": pairwise_metrics(rows, correct_sets),
        "representation_diversity": {
            "state_pair_comparisons": len(similarities),
            "mean_text_similarity": sum(similarities) / len(similarities),
            "near_identical_pairs_at_0_90": near_identical,
            "near_identical_fraction": near_identical / len(similarities),
            "note": "Text similarity is a weak collapse diagnostic, not semantic equivalence.",
        },
        "resources": resources,
        "rows": rows,
        "claim_boundary": (
            "Fresh public procedural diagnostic only. A field-native state is a "
            "representation candidate, not field authority. Oracle coverage is posthoc. "
            "No protected promotion, frontier-equivalence, or general cross-field claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seeds", default="20261007,20261008")
    parser.add_argument("--cases-per-family", type=int, default=2)
    parser.add_argument("--output", default="run-034-disciplinary.json")
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
        "representation_diversity": result["representation_diversity"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
