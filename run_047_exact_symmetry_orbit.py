"""Run 047: exact symmetry-orbit metamorphic reasoning probe.

Each orbit member is an answer-blind, exactly invertible task symmetry. The
correct option index is preserved because candidate values are transformed by the
same bijection as the visible task. The experiment asks whether the frozen actor
is equivariant to those symmetries and whether deterministic score aggregation
across the orbit improves selected-answer accuracy.

No transform receives correct_index, judge state, or hidden generator state.
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
from typing import Any, Callable

from offline_reasoning_battery import build_battery, sha256_json
from offline_reasoning_content_probe import surface_task_and_choices
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import build_representations, _score_values
from run_039_exact_representation_operators import operators_for_case

RUN_VERSION = "helix-exact-symmetry-orbit-001.0"
SYMMETRIES = ("identity", "symmetry-1", "symmetry-2", "symmetry-3")
VIEW_NAMES = ("layout", "familiar", "operator-1", "operator-2")


def _strip_answer_value(task: str) -> str:
    suffix = "\nAnswer value:"
    if not task.endswith(suffix):
        raise ValueError("task missing Answer value marker")
    return task[:-len(suffix)]


def _render_prose(stem: str, choices: list[str]) -> str:
    block = "\n".join(f"{chr(65+i)}. {value}" for i, value in enumerate(choices))
    return f"{stem}\nChoices:\n{block}\nAnswer:"


def _replace_tokens(text: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return text
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, mapping), key=len, reverse=True)) + r")\b")
    return pattern.sub(lambda m: mapping[m.group(1)], text)


# ---------------------------------------------------------------------------
# Relational-composition symmetries: D4-style transformations of compass frame.
# ---------------------------------------------------------------------------

DIR_TO_VEC = {
    "north": (0, 1),
    "south": (0, -1),
    "east": (1, 0),
    "west": (-1, 0),
    "northeast": (1, 1),
    "northwest": (-1, 1),
    "southeast": (1, -1),
    "southwest": (-1, -1),
}
VEC_TO_DIR = {value: key for key, value in DIR_TO_VEC.items()}


def _relation_matrix(symmetry: str) -> Callable[[int, int], tuple[int, int]]:
    if symmetry == "identity":
        return lambda x, y: (x, y)
    if symmetry == "symmetry-1":
        return lambda x, y: (y, -x)  # 90 degrees clockwise
    if symmetry == "symmetry-2":
        return lambda x, y: (-x, -y)  # 180 degrees
    if symmetry == "symmetry-3":
        return lambda x, y: (-x, y)  # east/west reflection
    raise KeyError(symmetry)


def _relation_surfaces(surfaces: dict[str, str], symmetry: str) -> dict[str, str]:
    if symmetry == "identity":
        return dict(surfaces)
    fn = _relation_matrix(symmetry)
    mapping = {
        name: VEC_TO_DIR[fn(*vec)]
        for name, vec in DIR_TO_VEC.items()
    }
    prose = _replace_tokens(surfaces["prose"], mapping)
    # The canonical and exact-operator builders consume prose for this family.
    return {**surfaces, "prose": prose}


# ---------------------------------------------------------------------------
# Abstract-grid symmetries: global permutation of the three visible symbols.
# The task transform itself is positional, so global color/symbol relabeling is
# an exact automorphism when demonstrations, query, and candidates all change.
# ---------------------------------------------------------------------------

GRID_SYMBOL_MAPS = {
    "identity": {"0": "0", "1": "1", "2": "2"},
    "symmetry-1": {"0": "1", "1": "2", "2": "0"},
    "symmetry-2": {"0": "2", "1": "1", "2": "0"},
    "symmetry-3": {"0": "1", "1": "0", "2": "2"},
}


def _map_grid_token(token: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(ch, ch) for ch in token)


def _grid_surfaces(surfaces: dict[str, str], symmetry: str) -> dict[str, str]:
    if symmetry == "identity":
        return dict(surfaces)
    mapping = GRID_SYMBOL_MAPS[symmetry]
    compact = re.sub(
        r"(?<![0-9])[0-2]{3}/[0-2]{3}/[0-2]{3}(?![0-9])",
        lambda m: _map_grid_token(m.group(0), mapping),
        surfaces["compact"],
    )
    return {**surfaces, "compact": compact}


# ---------------------------------------------------------------------------
# Relational-matrix symmetries: complement and coordinate reversal. The examples
# define the relation, so applying the same bijection to all visible vectors is
# an exact conjugacy of the finite task.
# ---------------------------------------------------------------------------

def _bit_transform(value: str, symmetry: str) -> str:
    if symmetry in {"symmetry-1", "symmetry-3"}:
        value = "".join("1" if ch == "0" else "0" for ch in value)
    if symmetry in {"symmetry-2", "symmetry-3"}:
        value = value[::-1]
    return value


def _matrix_surfaces(surfaces: dict[str, str], symmetry: str) -> dict[str, str]:
    if symmetry == "identity":
        return dict(surfaces)
    bits = re.sub(
        r"(?<![01])[01]{5}(?![01])",
        lambda m: _bit_transform(m.group(0), symmetry),
        surfaces["bits"],
    )
    return {**surfaces, "bits": bits}


# ---------------------------------------------------------------------------
# Grounded-planning symmetries: geometric board automorphisms with the move
# alphabet transformed in lockstep.
# ---------------------------------------------------------------------------

MOVE_MAPS = {
    "identity": {"U": "U", "R": "R", "D": "D", "L": "L"},
    "symmetry-1": {"U": "R", "R": "D", "D": "L", "L": "U"},
    "symmetry-2": {"U": "D", "R": "L", "D": "U", "L": "R"},
    "symmetry-3": {"U": "U", "R": "L", "D": "D", "L": "R"},
}


def _grid_chars(value: str) -> list[list[str]]:
    rows = value.split("/")
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError(f"unexpected planning grid: {value}")
    return [list(row) for row in rows]


def _planning_grid_transform(value: str, symmetry: str) -> str:
    g = _grid_chars(value)
    if symmetry == "identity":
        out = g
    elif symmetry == "symmetry-1":
        out = [list(row) for row in zip(*g[::-1])]
    elif symmetry == "symmetry-2":
        out = [list(reversed(row)) for row in reversed(g)]
    elif symmetry == "symmetry-3":
        out = [list(reversed(row)) for row in g]
    else:
        raise KeyError(symmetry)
    return "/".join("".join(row) for row in out)


def _planning_surfaces(surfaces: dict[str, str], symmetry: str) -> dict[str, str]:
    if symmetry == "identity":
        return dict(surfaces)
    task, choices = surface_task_and_choices(surfaces["grid"])
    stem = _strip_answer_value(task)
    match = re.search(r"Grid rows top-to-bottom: ([SG#.]{4}(?:/[SG#.]{4}){3})\.", stem)
    if match is None:
        raise ValueError("planning grid not found")
    old_grid = match.group(1)
    new_grid = _planning_grid_transform(old_grid, symmetry)
    stem = stem[:match.start(1)] + new_grid + stem[match.end(1):]
    move_map = MOVE_MAPS[symmetry]
    new_choices = ["".join(move_map[ch] for ch in choice) for choice in choices]
    return {**surfaces, "grid": _render_prose(stem, new_choices)}


# ---------------------------------------------------------------------------
# Stack-language symmetries: permutation of the three opener/closer pair names.
# ---------------------------------------------------------------------------

STACK_PAIR_PERMUTATIONS = {
    "identity": (0, 1, 2),
    "symmetry-1": (1, 2, 0),
    "symmetry-2": (2, 1, 0),
    "symmetry-3": (1, 0, 2),
}
STACK_PAIRS = (("ka", "ti"), ("mo", "re"), ("su", "va"))


def _stack_mapping(symmetry: str) -> dict[str, str]:
    perm = STACK_PAIR_PERMUTATIONS[symmetry]
    mapping: dict[str, str] = {}
    for source_index, target_index in enumerate(perm):
        source = STACK_PAIRS[source_index]
        target = STACK_PAIRS[target_index]
        mapping[source[0]] = target[0]
        mapping[source[1]] = target[1]
    return mapping


def _stack_surfaces(surfaces: dict[str, str], symmetry: str) -> dict[str, str]:
    if symmetry == "identity":
        return dict(surfaces)
    mapping = _stack_mapping(symmetry)
    return {
        **surfaces,
        "prose": _replace_tokens(surfaces["prose"], mapping),
        "compact": _replace_tokens(surfaces["compact"], mapping),
    }


TRANSFORMERS = {
    "relational-composition": _relation_surfaces,
    "abstract-transformation": _grid_surfaces,
    "relational-matrix": _matrix_surfaces,
    "grounded-planning": _planning_surfaces,
    "stack-language": _stack_surfaces,
}


def transform_surfaces(
    family: str,
    surfaces: dict[str, str],
    symmetry: str,
) -> dict[str, str]:
    """Apply an answer-blind family automorphism to visible surfaces only."""
    return TRANSFORMERS[family](dict(surfaces), symmetry)


def build_views(family: str, surfaces: dict[str, str]) -> dict[str, tuple[str, list[str]]]:
    canonical = build_representations(family, surfaces)
    operators = operators_for_case(family, surfaces)
    if len(operators) != 2:
        raise AssertionError("expected exactly two exact operators")
    return {
        "layout": canonical["layout"],
        "familiar": canonical["familiar"],
        "operator-1": (operators[0][1] + "\nAnswer value:", operators[0][2]),
        "operator-2": (operators[1][1] + "\nAnswer value:", operators[1][2]),
    }


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def _mean_scores(score_rows: list[list[float]]) -> list[float]:
    if not score_rows:
        raise ValueError("cannot average zero score rows")
    return [
        sum(row[index] for row in score_rows) / len(score_rows)
        for index in range(len(score_rows[0]))
    ]


def _majority(predictions: list[int]) -> int:
    counts = Counter(predictions)
    best = max(counts.values())
    return min(index for index, count in counts.items() if count == best)


def _accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    correct = sum(int(row[key]) == int(row["correct_index"]) for row in rows)
    return {"cases": len(rows), "correct": correct, "accuracy": correct / len(rows)}


def self_test(seed: int = 20261112, cases_per_family: int = 2) -> dict[str, Any]:
    cases = build_battery(seed, cases_per_family)
    counts = Counter()
    digests = []
    changed = Counter()
    for case in cases:
        identity = transform_surfaces(case["family"], case["surfaces"], "identity")
        if identity != case["surfaces"]:
            raise AssertionError((case["case_id"], "identity changed source"))
        for symmetry in SYMMETRIES:
            transformed = transform_surfaces(case["family"], case["surfaces"], symmetry)
            views = build_views(case["family"], transformed)
            if set(views) != set(VIEW_NAMES):
                raise AssertionError((case["case_id"], symmetry, sorted(views)))
            for view, (prompt, choices) in views.items():
                if len(choices) != 4 or not prompt.endswith("Answer value:"):
                    raise AssertionError((case["case_id"], symmetry, view))
                counts[f"{case['family']}/{symmetry}/{view}"] += 1
                digests.append(hashlib.sha256(json.dumps(
                    {
                        "family": case["family"],
                        "symmetry": symmetry,
                        "view": view,
                        "prompt": prompt,
                        "choices": choices,
                    },
                    sort_keys=True,
                ).encode()).hexdigest())
            if symmetry != "identity" and transformed != identity:
                changed[case["family"]] += 1
    for family in TRANSFORMERS:
        if changed[family] == 0:
            raise AssertionError(f"no non-identity transformed case observed for {family}")
    return {
        "schema_version": RUN_VERSION,
        "cases": len(cases),
        "orbit_members_per_case": len(SYMMETRIES),
        "views_per_orbit_member": len(VIEW_NAMES),
        "view_instances": sum(counts.values()),
        "changed_nonidentity_surfaces_by_family": dict(sorted(changed.items())),
        "transform_digest": hashlib.sha256("|".join(digests).encode()).hexdigest(),
        "correct_index_parameter_exposed_to_transformer": False,
    }


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
            orbit: dict[str, dict[str, Any]] = {}
            orbit_spec = {}
            for symmetry in SYMMETRIES:
                transformed = transform_surfaces(case["family"], case["surfaces"], symmetry)
                views = build_views(case["family"], transformed)
                scored_views: dict[str, Any] = {}
                for view, (prompt, choices) in views.items():
                    score = _score_values(scorer, prompt, choices)
                    scored_views[view] = {
                        "prediction": int(score["prediction"]),
                        "scores": [float(x) for x in score["scores"]],
                        "margin": float(score["margin"]),
                    }
                orbit[symmetry] = scored_views
                orbit_spec[symmetry] = {
                    view: hashlib.sha256(json.dumps(
                        {"prompt": prompt, "choices": choices},
                        sort_keys=True,
                    ).encode()).hexdigest()
                    for view, (prompt, choices) in views.items()
                }

            selectors: dict[str, int] = {}
            for view in VIEW_NAMES:
                selectors[f"{view}-identity"] = orbit["identity"][view]["prediction"]
                selectors[f"{view}-orbit-mean"] = _argmax(_mean_scores([
                    orbit[symmetry][view]["scores"] for symmetry in SYMMETRIES
                ]))

            identity_all_view_scores = _mean_scores([
                orbit["identity"][view]["scores"] for view in VIEW_NAMES
            ])
            selectors["identity-all-view-mean"] = _argmax(identity_all_view_scores)

            all_orbit_score_rows = [
                orbit[symmetry][view]["scores"]
                for symmetry in SYMMETRIES
                for view in VIEW_NAMES
            ]
            selectors["all-view-symmetry-mean"] = _argmax(_mean_scores(all_orbit_score_rows))
            selectors["identity-all-view-majority"] = _majority([
                orbit["identity"][view]["prediction"] for view in VIEW_NAMES
            ])
            selectors["all-view-symmetry-majority"] = _majority([
                orbit[symmetry][view]["prediction"]
                for symmetry in SYMMETRIES
                for view in VIEW_NAMES
            ])

            correct = int(case["correct_index"])
            identity_candidates = {
                orbit["identity"][view]["prediction"] for view in VIEW_NAMES
            }
            orbit_candidates = {
                orbit[symmetry][view]["prediction"]
                for symmetry in SYMMETRIES
                for view in VIEW_NAMES
            }

            per_view = {}
            for view in VIEW_NAMES:
                predictions = [orbit[s][view]["prediction"] for s in SYMMETRIES]
                correct_flags = [prediction == correct for prediction in predictions]
                per_view[view] = {
                    "identity_prediction": predictions[0],
                    "orbit_predictions": dict(zip(SYMMETRIES, predictions, strict=True)),
                    "orbit_prediction_invariant": len(set(predictions)) == 1,
                    "orbit_distinct_predictions": len(set(predictions)),
                    "identity_correct": correct_flags[0],
                    "orbit_any_correct": any(correct_flags),
                    "orbit_all_correct": all(correct_flags),
                    "identity_wrong_orbit_rescue": (not correct_flags[0]) and any(correct_flags[1:]),
                    "identity_correct_symmetry_break": correct_flags[0] and (not all(correct_flags[1:])),
                    "orbit_mean_prediction": selectors[f"{view}-orbit-mean"],
                }

            rows.append({
                "uid": uid,
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "per_view": per_view,
                "selectors": selectors,
                "identity_candidate_has_correct": correct in identity_candidates,
                "orbit_candidate_has_correct": correct in orbit_candidates,
                "identity_candidate_count": len(identity_candidates),
                "orbit_candidate_count": len(orbit_candidates),
            })
            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "orbit_view_sha256": orbit_spec,
            })
        public_spec.append({"seed": seed, "cases": seed_spec})

    view_summary: dict[str, Any] = {}
    for view in VIEW_NAMES:
        identity_correct = sum(row["per_view"][view]["identity_correct"] for row in rows)
        any_correct = sum(row["per_view"][view]["orbit_any_correct"] for row in rows)
        all_correct = sum(row["per_view"][view]["orbit_all_correct"] for row in rows)
        invariant = sum(row["per_view"][view]["orbit_prediction_invariant"] for row in rows)
        rescued = sum(row["per_view"][view]["identity_wrong_orbit_rescue"] for row in rows)
        broken = sum(row["per_view"][view]["identity_correct_symmetry_break"] for row in rows)
        distinct_mean = sum(row["per_view"][view]["orbit_distinct_predictions"] for row in rows) / len(rows)
        orbit_mean_correct = sum(
            int(row["selectors"][f"{view}-orbit-mean"]) == int(row["correct_index"])
            for row in rows
        )
        by_family = {}
        for family in sorted({row["family"] for row in rows}):
            subset = [row for row in rows if row["family"] == family]
            by_family[family] = {
                "cases": len(subset),
                "identity_accuracy": sum(r["per_view"][view]["identity_correct"] for r in subset) / len(subset),
                "orbit_coverage": sum(r["per_view"][view]["orbit_any_correct"] for r in subset) / len(subset),
                "equivariance_rate": sum(r["per_view"][view]["orbit_prediction_invariant"] for r in subset) / len(subset),
                "orbit_mean_accuracy": sum(
                    int(r["selectors"][f"{view}-orbit-mean"]) == int(r["correct_index"])
                    for r in subset
                ) / len(subset),
            }
        by_symmetry = {}
        for symmetry in SYMMETRIES:
            correct_count = sum(
                row["per_view"][view]["orbit_predictions"][symmetry] == row["correct_index"]
                for row in rows
            )
            by_symmetry[symmetry] = {
                "correct": correct_count,
                "accuracy": correct_count / len(rows),
            }
        view_summary[view] = {
            "cases": len(rows),
            "identity_correct": identity_correct,
            "identity_accuracy": identity_correct / len(rows),
            "orbit_any_correct": any_correct,
            "orbit_coverage": any_correct / len(rows),
            "orbit_all_correct": all_correct,
            "orbit_all_correct_rate": all_correct / len(rows),
            "equivariant_cases": invariant,
            "equivariance_rate": invariant / len(rows),
            "mean_distinct_predictions_across_orbit": distinct_mean,
            "identity_wrong_orbit_rescues": rescued,
            "identity_correct_symmetry_breaks": broken,
            "orbit_mean_correct": orbit_mean_correct,
            "orbit_mean_accuracy": orbit_mean_correct / len(rows),
            "by_family": by_family,
            "by_symmetry": by_symmetry,
        }

    selector_keys = [
        "identity-all-view-mean",
        "all-view-symmetry-mean",
        "identity-all-view-majority",
        "all-view-symmetry-majority",
    ]
    selectors = {key: _accuracy(rows, "__unused__") for key in []}
    selectors = {}
    for key in selector_keys:
        correct = sum(int(row["selectors"][key]) == int(row["correct_index"]) for row in rows)
        selectors[key] = {"cases": len(rows), "correct": correct, "accuracy": correct / len(rows)}

    identity_coverage = sum(row["identity_candidate_has_correct"] for row in rows)
    orbit_coverage = sum(row["orbit_candidate_has_correct"] for row in rows)

    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-exploratory-symmetry-calibration",
        "actor": {
            "model": "HuggingFaceTB/SmolLM2-360M",
            "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
            "weights_frozen": True,
        },
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(rows),
        "symmetries": list(SYMMETRIES),
        "views": list(VIEW_NAMES),
        "source_battery_sha256": sha256_json(public_spec),
        "view_summary": view_summary,
        "candidate_coverage": {
            "identity_views_correct": identity_coverage,
            "identity_views_coverage": identity_coverage / len(rows),
            "full_symmetry_orbit_correct": orbit_coverage,
            "full_symmetry_orbit_coverage": orbit_coverage / len(rows),
            "strict_orbit_rescues_beyond_identity_views": sum(
                (not row["identity_candidate_has_correct"]) and row["orbit_candidate_has_correct"]
                for row in rows
            ),
        },
        "selectors": selectors,
        "resources": resources,
        "rows": rows,
        "claim_boundary": (
            "Fresh public exploratory metamorphic diagnostic. Orbit transformations are deterministic "
            "visible-input-only exact symmetries. Candidate coverage is not selected-answer accuracy. "
            "Any retained mechanism requires larger fresh replication and does not authorize model promotion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--seeds", default="20261112,20261113")
    parser.add_argument("--cases-per-family", type=int, default=2)
    parser.add_argument("--output", default="run-047-exact-symmetry-orbit.json")
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
        [int(value) for value in args.seeds.split(",") if value.strip()],
        args.cases_per_family,
    )
    result["actor"].update({
        "model_file_sha256": observed,
        "model_bytes": model_path.stat().st_size,
        "dtype": "float32",
        "device": "cpu",
    })
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema_version": result["schema_version"],
        "cases": result["cases"],
        "view_summary": result["view_summary"],
        "candidate_coverage": result["candidate_coverage"],
        "selectors": result["selectors"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
