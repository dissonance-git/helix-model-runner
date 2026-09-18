"""Run 028: answer-blind canonical representation transfer.

The same frozen SmolLM2-360M actor is evaluated on fresh procedural tasks.
Helix, not the model, performs deterministic visible-input-only transforms:

- layout normalization: make structure explicit without changing the vocabulary;
- familiar normalization: translate visible semantics into common structural
  primitives while preserving the answer mapping.

The transformers never receive correct_index or generator latent state.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from offline_reasoning_battery import build_battery, sha256_json
from offline_reasoning_content_probe import choose_content, surface_task_and_choices
from run_026_capability_substitution import AttackScorer


RUN_VERSION = "helix-canonical-representation-transfer-001.0"


def _log_softmax(values: list[float]) -> list[float]:
    m = max(values)
    z = m + math.log(sum(math.exp(v - m) for v in values))
    return [v - z for v in values]


def _score_values(
    scorer: AttackScorer,
    prompt: str,
    choices: list[str],
) -> dict[str, Any]:
    raw = scorer.score_choices(prompt, choices)
    prior = scorer.score_choices("Answer value:", choices)
    pmi = [raw[i] - prior[i] for i in range(len(choices))]
    normalized = _log_softmax(pmi)
    order = sorted(normalized, reverse=True)
    return {
        "prediction": max(range(len(choices)), key=lambda i: normalized[i]),
        "scores": normalized,
        "margin": order[0] - order[1],
    }


def _stem_and_choices(prompt: str) -> tuple[str, list[str]]:
    task, choices = surface_task_and_choices(prompt)
    suffix = "\nAnswer value:"
    if not task.endswith(suffix):
        raise ValueError("content surface missing answer-value suffix")
    return task[: -len(suffix)], choices


def _grid_lines(value: str) -> str:
    rows = value.split("/")
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError(f"unexpected 3x3 grid: {value}")
    return "\n".join(" ".join(row) for row in rows)


def _grid_inline(value: str) -> str:
    rows = value.split("/")
    return "[" + "; ".join(" ".join(row) for row in rows) + "]"


def _relation_representations(surfaces: dict[str, str]) -> dict[str, tuple[str, list[str]]]:
    stem, choices = _stem_and_choices(surfaces["prose"])
    facts = re.findall(
        r"([A-Za-z]+) is (north|south|east|west) of ([A-Za-z]+)\.",
        stem,
    )
    query = re.search(
        r"Where is ([A-Za-z]+) relative to ([A-Za-z]+)\?",
        stem,
    )
    if not facts or query is None:
        raise ValueError("could not parse visible relational surface")
    left, right = query.groups()

    layout = (
        "Spatial relation task.\n"
        "Facts:\n"
        + "\n".join(f"- {a} is {direction} of {b}" for a, direction, b in facts)
        + f"\nQuestion: where is {left} relative to {right}?\n"
        "Answer value:"
    )

    vector = {
        "north": "(0,+1)",
        "south": "(0,-1)",
        "east": "(+1,0)",
        "west": "(-1,0)",
    }
    canonical = (
        "Represent locations with ordinary 2D coordinates.\n"
        "north=(0,+1), south=(0,-1), east=(+1,0), west=(-1,0).\n"
        "The visible facts are the following coordinate-difference equations:\n"
        + "\n".join(
            f"- {a} = {b} + {vector[direction]}"
            for a, direction, b in facts
        )
        + f"\nQuestion: determine the compass direction of {left} - {right}.\n"
        "Answer value:"
    )
    return {
        "layout": (layout, choices),
        "familiar": (canonical, choices),
    }


def _grid_representations(surfaces: dict[str, str]) -> dict[str, tuple[str, list[str]]]:
    stem, choices = _stem_and_choices(surfaces["compact"])
    grids = re.findall(r"[0-2]{3}/[0-2]{3}/[0-2]{3}", stem)
    if len(grids) < 3 or len(grids) % 2 != 1:
        raise ValueError("could not parse visible grid-transform surface")
    query = grids[-1]
    demonstrations = list(zip(grids[:-1:2], grids[1:-1:2], strict=True))
    demo_layout = []
    for index, (before, after) in enumerate(demonstrations, start=1):
        demo_layout.append(
            f"Example {index} input:\n{_grid_lines(before)}\n"
            f"Example {index} output:\n{_grid_lines(after)}"
        )
    layout = (
        "Infer one transformation T from the examples. The same T is used each time.\n"
        + "\n".join(demo_layout)
        + f"\nQuery input:\n{_grid_lines(query)}\n"
        "Return T(query).\nAnswer value:"
    )
    layout_choices = [_grid_inline(choice) for choice in choices]

    familiar = (
        "Treat every visible 3x3 grid as a matrix of positions. "
        "The examples show one unchanged positional transformation T; "
        "the digit symbols themselves remain the same values and only their positions are rearranged.\n"
        + "\n".join(demo_layout)
        + f"\nApply that same positional rule to this query matrix:\n{_grid_lines(query)}\n"
        "Answer value:"
    )
    return {
        "layout": (layout, layout_choices),
        "familiar": (familiar, layout_choices),
    }


def _matrix_representations(surfaces: dict[str, str]) -> dict[str, tuple[str, list[str]]]:
    stem, choices = _stem_and_choices(surfaces["bits"])
    demos = re.findall(r"([01]{5}) \? ([01]{5}) = ([01]{5})", stem)
    query_matches = re.findall(r"([01]{5}) \? ([01]{5}) =", stem)
    if not demos or not query_matches:
        raise ValueError("could not parse visible set-operation surface")
    qa, qb = query_matches[-1]

    layout = (
        "Infer one fixed set operation F from the examples.\n"
        + "\n".join(
            f"- F({a}, {b}) = {c}"
            for a, b, c in demos
        )
        + f"\nQuery: F({qa}, {qb}) = ?\nAnswer value:"
    )

    def vector(bits: str) -> str:
        return "[" + " ".join(bits) + "]"

    familiar = (
        "Each 5-bit vector is a membership vector for positions p q r s t: "
        "1 means present and 0 means absent. Infer the same set operation F from every demonstration.\n"
        + "\n".join(
            f"- F({vector(a)}, {vector(b)}) = {vector(c)}"
            for a, b, c in demos
        )
        + f"\nApply F to {vector(qa)} and {vector(qb)}.\n"
        "Answer value:"
    )
    familiar_choices = [vector(choice) for choice in choices]
    return {
        "layout": (layout, choices),
        "familiar": (familiar, familiar_choices),
    }


def _planning_representations(surfaces: dict[str, str]) -> dict[str, tuple[str, list[str]]]:
    stem, choices = _stem_and_choices(surfaces["grid"])
    grid_match = re.search(
        r"Grid rows top-to-bottom: ([SG#.]{4}(?:/[SG#.]{4}){3})\.",
        stem,
    )
    length_match = re.search(r"Choose the ([0-9]+)-move route", stem)
    if grid_match is None or length_match is None:
        raise ValueError("could not parse visible planning surface")
    grid = grid_match.group(1)
    length = int(length_match.group(1))
    rows = grid.split("/")
    map_text = "\n".join(" ".join(row) for row in rows)

    layout = (
        "Grid navigation task. U,R,D,L are moves. Do not enter #.\n"
        f"Map, rows top to bottom:\n{map_text}\n"
        f"Choose the exact {length}-move route from S to G.\n"
        "Answer value:"
    )
    layout_choices = [" ".join(choice) for choice in choices]

    familiar = (
        "Simulate each candidate path as a sequence of state updates on this grid. "
        "Start at S; U/R/D/L move one cell; reject a candidate immediately if it leaves "
        "the map or enters #; a valid candidate must finish exactly on G.\n"
        f"Map:\n{map_text}\nRequired moves: {length}\n"
        "Answer value:"
    )
    return {
        "layout": (layout, layout_choices),
        "familiar": (familiar, layout_choices),
    }


_BRACKET = {
    "ka": "(",
    "ti": ")",
    "mo": "[",
    "re": "]",
    "su": "{",
    "va": "}",
}


def _brackets(value: str) -> str:
    tokens = value.split()
    try:
        return "".join(_BRACKET[token] for token in tokens)
    except KeyError as exc:
        raise ValueError(f"unknown visible stack token: {exc}") from exc


def _stack_representations(surfaces: dict[str, str]) -> dict[str, tuple[str, list[str]]]:
    stem, choices = _stem_and_choices(surfaces["prose"])
    match = re.search(
        r"Prefix: ([a-z ]+)\. Which completion makes the whole sequence balanced\?",
        stem,
    )
    if match is None:
        raise ValueError("could not parse visible stack surface")
    prefix = match.group(1).strip()

    layout = (
        "Nested-token task. Closers must match the most recent unmatched opener.\n"
        "Pair table:\n"
        "- ka closes with ti\n"
        "- mo closes with re\n"
        "- su closes with va\n"
        f"Visible prefix: {prefix}\n"
        "Choose a completion that makes the entire sequence balanced.\n"
        "Answer value:"
    )

    familiar = (
        "This is ordinary balanced-bracket LIFO nesting after an exact symbol renaming:\n"
        "ka=(, ti=), mo=[, re=], su={, va=}.\n"
        f"Bracket prefix: {_brackets(prefix)}\n"
        "Choose the bracket completion that makes the full sequence balanced.\n"
        "Answer value:"
    )
    familiar_choices = [_brackets(choice) for choice in choices]
    return {
        "layout": (layout, choices),
        "familiar": (familiar, familiar_choices),
    }


_BUILDERS = {
    "relational-composition": _relation_representations,
    "abstract-transformation": _grid_representations,
    "relational-matrix": _matrix_representations,
    "grounded-planning": _planning_representations,
    "stack-language": _stack_representations,
}


def build_representations(
    family: str,
    surfaces: dict[str, str],
) -> dict[str, tuple[str, list[str]]]:
    """Build representations from visible inputs only.

    The function intentionally has no parameter for correct_index or latent
    generator state.
    """
    return _BUILDERS[family](surfaces)


def _summary(cases: list[dict[str, Any]], predictions: dict[str, int]) -> dict[str, Any]:
    by_family: dict[str, list[bool]] = defaultdict(list)
    by_seed: dict[int, list[bool]] = defaultdict(list)
    hits = 0
    for case in cases:
        ok = predictions[case["uid"]] == case["correct_index"]
        hits += int(ok)
        by_family[case["family"]].append(ok)
        by_seed[int(case["seed"])].append(ok)
    return {
        "cases": len(cases),
        "correct": hits,
        "accuracy": hits / len(cases),
        "by_family": {
            key: sum(values) / len(values)
            for key, values in sorted(by_family.items())
        },
        "by_seed": {
            str(key): sum(values) / len(values)
            for key, values in sorted(by_seed.items())
        },
    }


def _score_original_arm(
    scorer: AttackScorer,
    cases: list[dict[str, Any]],
    surface_index: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    scorer.reset_counters()
    started = time.time()
    scored = {}
    for case in cases:
        name = sorted(case["surfaces"])[surface_index]
        row = choose_content(scorer, case["surfaces"][name])
        scored[case["uid"]] = {
            "prediction": row["pmi_prediction"],
            "margin": row["pmi_margin"],
            "surface": name,
        }
    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started
    return scored, resources


def _score_transformed_arm(
    scorer: AttackScorer,
    cases: list[dict[str, Any]],
    transformed: dict[str, dict[str, tuple[str, list[str]]]],
    mode: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    scorer.reset_counters()
    started = time.time()
    scored = {}
    for case in cases:
        prompt, choices = transformed[case["uid"]][mode]
        row = _score_values(scorer, prompt, choices)
        scored[case["uid"]] = {
            **row,
            "surface": mode,
        }
    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started
    return scored, resources


def validate_transformers(seed: int = 20260921) -> dict[str, Any]:
    cases = build_battery(seed, 8)
    counts = Counter()
    digests = []
    for case in cases:
        reps = build_representations(case["family"], case["surfaces"])
        if set(reps) != {"layout", "familiar"}:
            raise AssertionError(case["case_id"])
        for mode, (prompt, choices) in reps.items():
            if len(choices) != 4 or not prompt.endswith("Answer value:"):
                raise AssertionError((case["case_id"], mode))
            counts[(case["family"], mode)] += 1
            digests.append(
                hashlib.sha256(
                    json.dumps(
                        {"family": case["family"], "mode": mode, "prompt": prompt, "choices": choices},
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
            )
    return {
        "cases": len(cases),
        "representation_count": sum(counts.values()),
        "family_mode_counts": {
            f"{family}/{mode}": value
            for (family, mode), value in sorted(counts.items())
        },
        "transform_digest": hashlib.sha256("|".join(digests).encode()).hexdigest(),
    }


def evaluate(
    model_dir: str,
    seeds: list[int],
    cases_per_family: int,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    transformed: dict[str, dict[str, tuple[str, list[str]]]] = {}
    public_spec = []

    for seed in seeds:
        generated = build_battery(seed, cases_per_family)
        seed_spec = []
        for case in generated:
            uid = f"{seed}:{case['case_id']}"
            row = {
                **case,
                "uid": uid,
                "seed": seed,
            }
            cases.append(row)
            # Critical ordering: build representation before scoring/evaluation
            # touches correct_index. The builder receives only family + visible surfaces.
            transformed[uid] = build_representations(
                case["family"],
                case["surfaces"],
            )
            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": case["correct_index"],
                "surface_sha256": {
                    name: hashlib.sha256(prompt.encode()).hexdigest()
                    for name, prompt in case["surfaces"].items()
                },
                "representation_sha256": {
                    mode: hashlib.sha256(
                        json.dumps(
                            {"prompt": prompt, "choices": choices},
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()
                    for mode, (prompt, choices) in transformed[uid].items()
                },
            })
        public_spec.append({"seed": seed, "cases": seed_spec})

    scorer = AttackScorer(model_dir)
    all_started = time.time()
    primary, primary_resources = _score_original_arm(scorer, cases, 0)
    secondary, secondary_resources = _score_original_arm(scorer, cases, 1)
    layout, layout_resources = _score_transformed_arm(
        scorer, cases, transformed, "layout"
    )
    familiar, familiar_resources = _score_transformed_arm(
        scorer, cases, transformed, "familiar"
    )

    arms = {
        "lexicographic-primary-surface": primary,
        "alternate-secondary-surface": secondary,
        "layout-normalized": layout,
        "familiar-structural-normalization": familiar,
    }
    predictions = {
        arm: {uid: row["prediction"] for uid, row in values.items()}
        for arm, values in arms.items()
    }

    high_margin = {}
    representation_choice_counts = Counter()
    oracle_hits = 0
    oracle_by_family: dict[str, list[bool]] = defaultdict(list)
    coverage_histogram = Counter()
    rows = []
    for case in cases:
        uid = case["uid"]
        candidates = {
            arm: values[uid]
            for arm, values in arms.items()
        }
        selected_arm, selected_row = max(
            candidates.items(),
            key=lambda item: (float(item[1]["margin"]), item[0]),
        )
        high_margin[uid] = int(selected_row["prediction"])
        representation_choice_counts[selected_arm] += 1
        correct_arms = [
            arm for arm, row in candidates.items()
            if int(row["prediction"]) == int(case["correct_index"])
        ]
        oracle = bool(correct_arms)
        oracle_hits += int(oracle)
        oracle_by_family[case["family"]].append(oracle)
        coverage_histogram[len(correct_arms)] += 1
        rows.append({
            "uid": uid,
            "seed": case["seed"],
            "case_id": case["case_id"],
            "family": case["family"],
            "difficulty": case["difficulty"],
            "predictions": {
                arm: int(row["prediction"])
                for arm, row in candidates.items()
            },
            "margins": {
                arm: float(row["margin"])
                for arm, row in candidates.items()
            },
            "highest_margin_arm": selected_arm,
            "highest_margin_prediction": int(selected_row["prediction"]),
            "correct_representation_count": len(correct_arms),
            "correct_representations": correct_arms,
        })

    summaries = {
        arm: _summary(cases, arm_predictions)
        for arm, arm_predictions in predictions.items()
    }
    summaries["highest-margin-selection"] = _summary(cases, high_margin)

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-transfer-calibration",
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(cases),
        "source_battery_sha256": sha256_json(public_spec),
        "answer_blind_transformer_contract": {
            "inputs": ["task family", "visible prompt surfaces"],
            "forbidden": [
                "correct_index",
                "latent generator operation",
                "hidden solution",
            ],
            "builder_signature": "build_representations(family, surfaces)",
        },
        "summaries": summaries,
        "oracle_representation_coverage": {
            "correct": oracle_hits,
            "cases": len(cases),
            "coverage": oracle_hits / len(cases),
            "by_family": {
                family: sum(values) / len(values)
                for family, values in sorted(oracle_by_family.items())
            },
            "correct_representation_count_histogram": dict(
                sorted(coverage_histogram.items())
            ),
            "oracle_role": "posthoc diagnostic only",
        },
        "highest_margin_representation_choice_counts": dict(
            representation_choice_counts
        ),
        "resources": {
            "lexicographic-primary-surface": primary_resources,
            "alternate-secondary-surface": secondary_resources,
            "layout-normalized": layout_resources,
            "familiar-structural-normalization": familiar_resources,
        },
        "rows": rows,
        "total_elapsed_seconds": time.time() - all_started,
        "claim_boundary": (
            "Fresh public procedural transfer calibration only. Representation "
            "builders are answer-blind, but this is not protected promotion evidence, "
            "frontier equivalence, or a general reasoning-capability claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--seeds", default="20260921,20260922,20260923")
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--output", default="run-028-representation.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(validate_transformers(), indent=2, sort_keys=True))
        return
    if not args.model_dir or not args.model_file or not args.expected_sha256:
        raise SystemExit("model-dir, model-file and expected-sha256 are required")

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
        "summaries": result["summaries"],
        "oracle_representation_coverage": result["oracle_representation_coverage"],
        "highest_margin_representation_choice_counts": result[
            "highest_margin_representation_choice_counts"
        ],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
