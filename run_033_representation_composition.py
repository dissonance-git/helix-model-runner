"""Run 033: bounded recursive representation composition.

Two answer-blind seed representations A and B are composed in both orders.
First-generation composites then re-enter as operands with A or B. External
labels are used only after each state exists.

This mirrors Helix's A+B=C / multistep composition machinery on the exact same
frozen SmolLM2-360M actor.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from offline_reasoning_battery import build_battery, sha256_json
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import build_representations, _score_values

RUN_VERSION = "helix-representation-composition-closure-001.0"


def strip_answer_marker(prompt: str) -> str:
    suffix = "Answer value:"
    if not prompt.rstrip().endswith(suffix):
        raise ValueError("representation lacks answer marker")
    return prompt.rstrip()[: -len(suffix)].rstrip()


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def compose_state(
    scorer: AttackScorer,
    *,
    left_name: str,
    left_state: str,
    right_name: str,
    right_state: str,
    seed: int,
) -> str:
    prompt = (
        "Compose two answer-blind representations of the same reasoning obligation.\n"
        "The LEFT state supplies the current frame. The RIGHT state may add explicit "
        "relations, constraints, distinctions, or unresolved disagreement.\n"
        "Do not solve the task. Do not guess an answer. No answer choices are available.\n"
        "Preserve the question target and every distinction that could change the answer.\n"
        "Collapse only redundant wording or layout. If the states conflict, record the "
        "conflict instead of silently choosing one.\n"
        "Return one compact operational state that can itself be used as an operand later.\n\n"
        f"LEFT [{left_name}]\n{left_state}\n\n"
        f"RIGHT [{right_name}]\n{right_state}\n\n"
        "COMPOSITE STATE:"
    )
    return scorer.generate_text(
        prompt,
        max_new_tokens=64,
        seed=seed,
        sample=False,
    )


def score_state(
    scorer: AttackScorer,
    state: str,
    choices: list[str],
) -> dict[str, Any]:
    prompt = (
        "Use this derived operational state to answer the original task.\n"
        "Derived state:\n"
        f"{state}\n"
        "Answer value:"
    )
    return _score_values(scorer, prompt, choices)


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
            reps = build_representations(case["family"], case["surfaces"])
            a_prompt, a_choices = reps["layout"]
            b_prompt, b_choices = reps["familiar"]

            # The two transformed arms preserve answer index, while their value
            # strings can differ. Layout is the canonical value vocabulary used
            # only after a derived state exists.
            if len(a_choices) != 4 or len(b_choices) != 4:
                raise ValueError("expected four answer values")
            A = strip_answer_marker(a_prompt)
            B = strip_answer_marker(b_prompt)

            # Depth 0 seed scoring.
            a_score = _score_values(scorer, a_prompt, a_choices)
            b_score = _score_values(scorer, b_prompt, b_choices)

            base_seed = seed * 1000 + local_index * 20
            C_ab = compose_state(
                scorer,
                left_name="A/layout",
                left_state=A,
                right_name="B/familiar",
                right_state=B,
                seed=base_seed + 1,
            )
            C_ba = compose_state(
                scorer,
                left_name="B/familiar",
                left_state=B,
                right_name="A/layout",
                right_state=A,
                seed=base_seed + 2,
            )
            c_ab_score = score_state(scorer, C_ab, a_choices)
            c_ba_score = score_state(scorer, C_ba, a_choices)

            D_aba = compose_state(
                scorer,
                left_name="C_ab",
                left_state=C_ab,
                right_name="A/layout",
                right_state=A,
                seed=base_seed + 3,
            )
            E_abb = compose_state(
                scorer,
                left_name="C_ab",
                left_state=C_ab,
                right_name="B/familiar",
                right_state=B,
                seed=base_seed + 4,
            )
            F_baa = compose_state(
                scorer,
                left_name="C_ba",
                left_state=C_ba,
                right_name="A/layout",
                right_state=A,
                seed=base_seed + 5,
            )
            G_bab = compose_state(
                scorer,
                left_name="C_ba",
                left_state=C_ba,
                right_name="B/familiar",
                right_state=B,
                seed=base_seed + 6,
            )

            depth2_states = {
                "D_aba": D_aba,
                "E_abb": E_abb,
                "F_baa": F_baa,
                "G_bab": G_bab,
            }
            depth2_scores = {
                name: score_state(scorer, state, a_choices)
                for name, state in depth2_states.items()
            }

            correct = int(case["correct_index"])
            seed_predictions = {
                "A": int(a_score["prediction"]),
                "B": int(b_score["prediction"]),
            }
            depth1_predictions = {
                "C_ab": int(c_ab_score["prediction"]),
                "C_ba": int(c_ba_score["prediction"]),
            }
            depth2_predictions = {
                name: int(value["prediction"])
                for name, value in depth2_scores.items()
            }

            seed_has = any(v == correct for v in seed_predictions.values())
            depth1_has = any(v == correct for v in depth1_predictions.values())
            depth2_has = any(v == correct for v in depth2_predictions.values())

            strict_pair = (not seed_has) and depth1_has
            strict_rebound = (not seed_has) and (not depth1_has) and depth2_has

            order_disagrees = (
                depth1_predictions["C_ab"] != depth1_predictions["C_ba"]
            )
            order_one_correct = (
                order_disagrees
                and (
                    (depth1_predictions["C_ab"] == correct)
                    != (depth1_predictions["C_ba"] == correct)
                )
            )

            successful_reused = []
            if strict_rebound:
                if (
                    depth2_predictions["D_aba"] == correct
                    or depth2_predictions["F_baa"] == correct
                ):
                    successful_reused.append("A")
                if (
                    depth2_predictions["E_abb"] == correct
                    or depth2_predictions["G_bab"] == correct
                ):
                    successful_reused.append("B")

            ancestor_states = {
                "A": compact(A),
                "B": compact(B),
                "C_ab": compact(C_ab),
                "C_ba": compact(C_ba),
            }
            exact_repeats = []
            for name, state in depth2_states.items():
                normalized = compact(state)
                parents = {
                    "D_aba": ("C_ab", "A"),
                    "E_abb": ("C_ab", "B"),
                    "F_baa": ("C_ba", "A"),
                    "G_bab": ("C_ba", "B"),
                }[name]
                for parent in parents:
                    if normalized == ancestor_states[parent]:
                        exact_repeats.append({"state": name, "ancestor": parent})

            row = {
                "uid": uid,
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "A_prediction": seed_predictions["A"],
                "B_prediction": seed_predictions["B"],
                "C_ab_prediction": depth1_predictions["C_ab"],
                "C_ba_prediction": depth1_predictions["C_ba"],
                "D_aba_prediction": depth2_predictions["D_aba"],
                "E_abb_prediction": depth2_predictions["E_abb"],
                "F_baa_prediction": depth2_predictions["F_baa"],
                "G_bab_prediction": depth2_predictions["G_bab"],
                "seed_has_correct": seed_has,
                "depth1_has_correct": depth1_has,
                "depth2_has_correct": depth2_has,
                "strict_pair_only_correct": strict_pair,
                "strict_rebound_only_correct": strict_rebound,
                "operator_order_disagrees": order_disagrees,
                "operator_order_one_correct": order_one_correct,
                "successful_reused_seed_enablers": successful_reused,
                "exact_repeats": exact_repeats,
                "state_sha256": {
                    "A": hashlib.sha256(A.encode()).hexdigest(),
                    "B": hashlib.sha256(B.encode()).hexdigest(),
                    "C_ab": hashlib.sha256(C_ab.encode()).hexdigest(),
                    "C_ba": hashlib.sha256(C_ba.encode()).hexdigest(),
                    **{
                        name: hashlib.sha256(state.encode()).hexdigest()
                        for name, state in depth2_states.items()
                    },
                },
            }
            rows.append(row)

            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "source_surface_sha256": {
                    name: hashlib.sha256(prompt.encode()).hexdigest()
                    for name, prompt in case["surfaces"].items()
                },
                "A_sha256": row["state_sha256"]["A"],
                "B_sha256": row["state_sha256"]["B"],
            })
        public_spec.append({"seed": seed, "cases": seed_spec})

    arm_keys = [
        "A_prediction",
        "B_prediction",
        "C_ab_prediction",
        "C_ba_prediction",
        "D_aba_prediction",
        "E_abb_prediction",
        "F_baa_prediction",
        "G_bab_prediction",
    ]
    arms = {key.removesuffix("_prediction"): summarize(rows, key) for key in arm_keys}

    depth0 = sum(row["seed_has_correct"] for row in rows)
    depth1_total = sum(
        row["seed_has_correct"] or row["depth1_has_correct"]
        for row in rows
    )
    depth2_total = sum(
        row["seed_has_correct"] or row["depth1_has_correct"] or row["depth2_has_correct"]
        for row in rows
    )
    strict_pair = sum(row["strict_pair_only_correct"] for row in rows)
    strict_rebound = sum(row["strict_rebound_only_correct"] for row in rows)
    order_disagree = sum(row["operator_order_disagrees"] for row in rows)
    order_one_correct = sum(row["operator_order_one_correct"] for row in rows)
    reused = Counter(
        seed_id
        for row in rows
        for seed_id in row["successful_reused_seed_enablers"]
    )
    repeats = sum(len(row["exact_repeats"]) for row in rows)

    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started
    resources["seed_representations_per_case"] = 2
    resources["first_generation_compositions_per_case"] = 2
    resources["second_generation_compositions_per_case"] = 4

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-bounded-composition-calibration",
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(rows),
        "source_battery_sha256": sha256_json(public_spec),
        "operator": {
            "ordered": True,
            "answer_blind": True,
            "derived_state_can_reenter": True,
            "maximum_composition_depth": 2,
            "seed_representations": {
                "A": "layout-normalized",
                "B": "familiar-structural-normalization",
            },
        },
        "arms": arms,
        "emergence": {
            "seed_coverage_correct": depth0,
            "seed_coverage": depth0 / len(rows),
            "through_depth1_correct": depth1_total,
            "through_depth1_coverage": depth1_total / len(rows),
            "through_depth2_correct": depth2_total,
            "through_depth2_coverage": depth2_total / len(rows),
            "strict_pair_only_correct_cases": strict_pair,
            "strict_pair_only_fraction": strict_pair / len(rows),
            "strict_rebound_only_correct_cases": strict_rebound,
            "strict_rebound_only_fraction": strict_rebound / len(rows),
            "operator_order_disagreement_cases": order_disagree,
            "operator_order_one_correct_cases": order_one_correct,
            "reused_seed_enablers_in_strict_rebounds": dict(reused),
            "exact_depth2_parent_repeats": repeats,
        },
        "resources": resources,
        "rows": rows,
        "claim_boundary": (
            "Fresh public bounded diagnostic only. Pair-only and rebound-only correctness "
            "are case-local emergence under this frozen operator, not proof of a general "
            "representation algebra. External labels are used only after states exist."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seeds", default="20261005,20261006")
    parser.add_argument("--cases-per-family", type=int, default=3)
    parser.add_argument("--output", default="run-033-composition.json")
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
        "emergence": result["emergence"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
