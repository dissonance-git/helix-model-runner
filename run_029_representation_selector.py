"""Run 029: fresh transfer test for answer-blind representation selection.

Run 028 established that deterministic representation changes expose substantially
more correct candidates than any one fixed representation can recover. This run
tests whether a tiny linear selector can recover part of that gap without changing
the frozen actor or seeing fresh-test labels.

The selector is intentionally simpler than neural or reflective donors:
- ridge regression over candidate correctness;
- score-only and score+public-family variants;
- nearest-support + advantage fallback to the fixed layout arm.

GEPA-style reflective representation evolution remains a later discriminator if
this simpler selector leaves most of the oracle gap unresolved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from offline_reasoning_battery import build_battery, sha256_json
from offline_reasoning_content_probe import choose_content
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import build_representations, _score_values


RUN_VERSION = "helix-representation-selection-transfer-001.0"
ARMS = (
    "lexicographic-primary-surface",
    "alternate-secondary-surface",
    "layout-normalized",
    "familiar-structural-normalization",
)
FAMILIES = (
    "abstract-transformation",
    "grounded-planning",
    "relational-composition",
    "relational-matrix",
    "stack-language",
)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def _entropy(log_probs: list[float]) -> float:
    probs = np.exp(np.asarray(log_probs, dtype=float))
    return float(-(probs * np.asarray(log_probs, dtype=float)).sum())


def _score_original(
    scorer: AttackScorer,
    prompt: str,
) -> dict[str, Any]:
    row = choose_content(scorer, prompt)
    scores = [float(value) for value in row["pmi_scores"]]
    # PMI scores need not be normalized. Selection features should be comparable
    # within the candidate, so normalize exactly as the transformed scorer does.
    maximum = max(scores)
    z = maximum + math.log(sum(math.exp(value - maximum) for value in scores))
    normalized = [value - z for value in scores]
    order = sorted(normalized, reverse=True)
    return {
        "prediction": int(row["pmi_prediction"]),
        "scores": normalized,
        "margin": float(order[0] - order[1]),
    }


def _case_rows(
    scorer: AttackScorer,
    seeds: list[int],
    cases_per_family: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    public_spec: list[dict[str, Any]] = []
    for seed in seeds:
        cases = build_battery(seed, cases_per_family)
        seed_spec = []
        for case in cases:
            uid = f"{seed}:{case['case_id']}"
            reps = build_representations(case["family"], case["surfaces"])
            names = sorted(case["surfaces"])
            # Transform before labels are read by selection/training code.
            layout_prompt, layout_choices = reps["layout"]
            familiar_prompt, familiar_choices = reps["familiar"]
            scored = {
                "lexicographic-primary-surface": _score_original(
                    scorer, case["surfaces"][names[0]]
                ),
                "alternate-secondary-surface": _score_original(
                    scorer, case["surfaces"][names[1]]
                ),
                "layout-normalized": _score_values(
                    scorer, layout_prompt, layout_choices
                ),
                "familiar-structural-normalization": _score_values(
                    scorer, familiar_prompt, familiar_choices
                ),
            }
            predictions = {
                arm: int(scored[arm]["prediction"])
                for arm in ARMS
            }
            counts = Counter(predictions.values())
            correct_index = int(case["correct_index"])
            rows.append({
                "uid": uid,
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct_index,
                "candidates": {
                    arm: {
                        **scored[arm],
                        "same_prediction_count": int(
                            counts[int(scored[arm]["prediction"])]
                        ),
                        "modal_prediction": bool(
                            counts[int(scored[arm]["prediction"])]
                            == max(counts.values())
                        ),
                        "correct": bool(
                            int(scored[arm]["prediction"]) == correct_index
                        ),
                    }
                    for arm in ARMS
                },
            })
            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
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
                    for mode, (prompt, choices) in reps.items()
                },
            })
        public_spec.append({"seed": seed, "cases": seed_spec})
    return rows, public_spec


def _candidate_features(
    row: dict[str, Any],
    arm: str,
    *,
    include_family: bool,
) -> list[float]:
    candidate = row["candidates"][arm]
    scores = np.asarray(candidate["scores"], dtype=float)
    ordered = np.sort(scores)[::-1]
    vector: list[float] = []
    vector.extend(1.0 if value == arm else 0.0 for value in ARMS)
    prediction = int(candidate["prediction"])
    vector.extend(1.0 if index == prediction else 0.0 for index in range(4))
    vector.extend(float(value) for value in scores.tolist())
    vector.extend([
        float(ordered[0]),
        float(ordered[1]),
        float(candidate["margin"]),
        _entropy(candidate["scores"]),
        float(np.std(scores)),
        float(candidate["same_prediction_count"]) / len(ARMS),
        1.0 if candidate["modal_prediction"] else 0.0,
    ])
    if include_family:
        vector.extend(
            1.0 if row["family"] == family else 0.0
            for family in FAMILIES
        )
    return vector


def _matrix(
    rows: list[dict[str, Any]],
    *,
    include_family: bool,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    features: list[list[float]] = []
    labels: list[float] = []
    ids: list[tuple[str, str]] = []
    for row in rows:
        for arm in ARMS:
            features.append(
                _candidate_features(row, arm, include_family=include_family)
            )
            labels.append(1.0 if row["candidates"][arm]["correct"] else 0.0)
            ids.append((row["uid"], arm))
    return np.asarray(features, dtype=float), np.asarray(labels), ids


class RidgeRanker:
    def __init__(self, alpha: float, include_family: bool):
        self.alpha = float(alpha)
        self.include_family = bool(include_family)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.train_scaled: np.ndarray | None = None
        self.train_ids: list[tuple[str, str]] = []

    def fit(self, rows: list[dict[str, Any]]) -> "RidgeRanker":
        x, y, ids = _matrix(rows, include_family=self.include_family)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-9] = 1.0
        z = (x - mean) / scale
        design = np.column_stack([np.ones(len(z)), z])
        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(
            design.T @ design + self.alpha * penalty,
            design.T @ y,
        )
        self.mean = mean
        self.scale = scale
        self.coef = coef
        self.train_scaled = z
        self.train_ids = ids
        return self

    def _scaled(self, row: dict[str, Any], arm: str) -> np.ndarray:
        assert self.mean is not None and self.scale is not None
        raw = np.asarray(
            _candidate_features(row, arm, include_family=self.include_family),
            dtype=float,
        )
        return (raw - self.mean) / self.scale

    def score(self, row: dict[str, Any], arm: str) -> float:
        assert self.coef is not None
        z = self._scaled(row, arm)
        return float(self.coef[0] + z @ self.coef[1:])

    def nearest_support_distance(self, row: dict[str, Any], arm: str) -> float:
        assert self.train_scaled is not None
        z = self._scaled(row, arm)
        indices = [
            index for index, (_uid, candidate_arm) in enumerate(self.train_ids)
            if candidate_arm == arm
        ]
        pool = self.train_scaled[indices]
        return float(np.sqrt(((pool - z) ** 2).mean(axis=1)).min())


def _select_ranker(
    ranker: RidgeRanker,
    row: dict[str, Any],
) -> tuple[str, dict[str, float]]:
    scores = {arm: ranker.score(row, arm) for arm in ARMS}
    selected = max(ARMS, key=lambda arm: (scores[arm], arm))
    return selected, scores


def _correct(row: dict[str, Any], arm: str) -> bool:
    return bool(row["candidates"][arm]["correct"])


def _accuracy(rows: list[dict[str, Any]], choices: dict[str, str]) -> float:
    return sum(_correct(row, choices[row["uid"]]) for row in rows) / len(rows)


def _summary(rows: list[dict[str, Any]], choices: dict[str, str]) -> dict[str, Any]:
    by_family: dict[str, list[bool]] = defaultdict(list)
    by_seed: dict[int, list[bool]] = defaultdict(list)
    hits = 0
    regressions_vs_layout = 0
    wins_vs_layout = 0
    selected = Counter()
    for row in rows:
        arm = choices[row["uid"]]
        ok = _correct(row, arm)
        layout_ok = _correct(row, "layout-normalized")
        hits += int(ok)
        selected[arm] += 1
        by_family[row["family"]].append(ok)
        by_seed[int(row["seed"])].append(ok)
        wins_vs_layout += int(ok and not layout_ok)
        regressions_vs_layout += int(layout_ok and not ok)
    return {
        "cases": len(rows),
        "correct": hits,
        "accuracy": hits / len(rows),
        "wins_vs_layout": wins_vs_layout,
        "regressions_vs_layout": regressions_vs_layout,
        "selected_representations": dict(selected),
        "by_family": {
            family: sum(values) / len(values)
            for family, values in sorted(by_family.items())
        },
        "by_seed": {
            str(seed): sum(values) / len(values)
            for seed, values in sorted(by_seed.items())
        },
    }


def _fit_family_lookup(rows: list[dict[str, Any]]) -> dict[str, str]:
    mapping = {}
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        mapping[family] = max(
            ARMS,
            key=lambda arm: (
                sum(_correct(row, arm) for row in family_rows),
                -ARMS.index(arm),
            ),
        )
    return mapping


def _fixed_choices(rows: list[dict[str, Any]], arm: str) -> dict[str, str]:
    return {row["uid"]: arm for row in rows}


def _margin_choices(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        row["uid"]: max(
            ARMS,
            key=lambda arm: (
                float(row["candidates"][arm]["margin"]),
                arm,
            ),
        )
        for row in rows
    }


def _ranker_choices(
    ranker: RidgeRanker,
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    return {
        row["uid"]: _select_ranker(ranker, row)[0]
        for row in rows
    }


def _tune_alpha(
    fit_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    include_family: bool,
) -> tuple[RidgeRanker, list[dict[str, Any]]]:
    trials = []
    best: tuple[float, float, RidgeRanker] | None = None
    for alpha in ALPHAS:
        ranker = RidgeRanker(alpha, include_family).fit(fit_rows)
        choices = _ranker_choices(ranker, validation_rows)
        accuracy = _accuracy(validation_rows, choices)
        trials.append({"alpha": alpha, "validation_accuracy": accuracy})
        key = (accuracy, alpha)
        if best is None or key > (best[0], best[1]):
            best = (accuracy, alpha, ranker)
    assert best is not None
    return best[2], trials


def _tune_gate(
    ranker: RidgeRanker,
    validation_rows: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    observations = []
    for row in validation_rows:
        selected, scores = _select_ranker(ranker, row)
        advantage = scores[selected] - scores["layout-normalized"]
        distance = ranker.nearest_support_distance(row, selected)
        observations.append((row, selected, advantage, distance))

    advantages = sorted({round(float(item[2]), 8) for item in observations})
    distances = sorted({round(float(item[3]), 8) for item in observations})
    advantage_grid = [-1e9, 0.0] + advantages
    distance_grid = distances + [1e9]
    best = None
    trials = []
    for minimum_advantage in advantage_grid:
        for maximum_distance in distance_grid:
            choices = {}
            selected_count = 0
            for row, selected, advantage, distance in observations:
                use = advantage >= minimum_advantage and distance <= maximum_distance
                arm = selected if use else "layout-normalized"
                selected_count += int(use)
                choices[row["uid"]] = arm
            accuracy = _accuracy(validation_rows, choices)
            trial = {
                "minimum_advantage": float(minimum_advantage),
                "maximum_support_distance": float(maximum_distance),
                "validation_accuracy": accuracy,
                "selector_coverage": selected_count / len(validation_rows),
            }
            trials.append(trial)
            # Prefer accuracy, then less selector intervention, then stricter support.
            key = (
                accuracy,
                -trial["selector_coverage"],
                minimum_advantage,
                -maximum_distance,
            )
            if best is None or key > best[0]:
                best = (key, trial)
    assert best is not None
    return {
        "minimum_advantage": best[1]["minimum_advantage"],
        "maximum_support_distance": best[1]["maximum_support_distance"],
    }, trials


def _gated_choices(
    ranker: RidgeRanker,
    rows: list[dict[str, Any]],
    gate: dict[str, float],
) -> tuple[dict[str, str], dict[str, Any]]:
    choices = {}
    accepted = 0
    fallback = 0
    distances = []
    advantages = []
    for row in rows:
        selected, scores = _select_ranker(ranker, row)
        advantage = scores[selected] - scores["layout-normalized"]
        distance = ranker.nearest_support_distance(row, selected)
        distances.append(distance)
        advantages.append(advantage)
        use = (
            advantage >= gate["minimum_advantage"]
            and distance <= gate["maximum_support_distance"]
        )
        choices[row["uid"]] = selected if use else "layout-normalized"
        accepted += int(use)
        fallback += int(not use)
    return choices, {
        "selector_cases": accepted,
        "fallback_cases": fallback,
        "selector_coverage": accepted / len(rows),
        "mean_support_distance": float(np.mean(distances)),
        "mean_predicted_advantage": float(np.mean(advantages)),
    }


def evaluate(
    model_dir: str,
    *,
    fit_seeds: list[int],
    validation_seeds: list[int],
    test_seeds: list[int],
    cases_per_family: int,
) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()
    started = time.time()

    fit_rows, fit_spec = _case_rows(scorer, fit_seeds, cases_per_family)
    validation_rows, validation_spec = _case_rows(
        scorer, validation_seeds, cases_per_family
    )
    test_rows, test_spec = _case_rows(scorer, test_seeds, cases_per_family)

    fixed_accuracy = {
        arm: _accuracy(fit_rows, _fixed_choices(fit_rows, arm))
        for arm in ARMS
    }
    best_fixed = max(
        ARMS,
        key=lambda arm: (fixed_accuracy[arm], -ARMS.index(arm)),
    )
    family_lookup = _fit_family_lookup(fit_rows)

    score_ranker, score_trials = _tune_alpha(
        fit_rows, validation_rows, include_family=False
    )
    family_ranker, family_trials = _tune_alpha(
        fit_rows, validation_rows, include_family=True
    )
    score_val = _accuracy(
        validation_rows, _ranker_choices(score_ranker, validation_rows)
    )
    family_val = _accuracy(
        validation_rows, _ranker_choices(family_ranker, validation_rows)
    )
    if score_val >= family_val:
        best_linear_name = "linear-score-ranker"
        best_ranker = score_ranker
    else:
        best_linear_name = "linear-score-plus-family-ranker"
        best_ranker = family_ranker

    gate, gate_trials = _tune_gate(best_ranker, validation_rows)

    test_choices = {
        "fixed-layout": _fixed_choices(test_rows, "layout-normalized"),
        "best-fixed-representation": _fixed_choices(test_rows, best_fixed),
        "highest-margin": _margin_choices(test_rows),
        "family-lookup": {
            row["uid"]: family_lookup[row["family"]]
            for row in test_rows
        },
        "linear-score-ranker": _ranker_choices(score_ranker, test_rows),
        "linear-score-plus-family-ranker": _ranker_choices(
            family_ranker, test_rows
        ),
    }
    gated, gate_test = _gated_choices(best_ranker, test_rows, gate)
    test_choices["support-gated-best-linear-ranker"] = gated

    summaries = {
        name: _summary(test_rows, choices)
        for name, choices in test_choices.items()
    }

    oracle_choices = {}
    oracle_hits = 0
    for row in test_rows:
        correct_arms = [
            arm for arm in ARMS
            if _correct(row, arm)
        ]
        if correct_arms:
            oracle_hits += 1
            oracle_choices[row["uid"]] = correct_arms[0]
        else:
            oracle_choices[row["uid"]] = "layout-normalized"

    result = {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-selector-transfer",
        "split": {
            "fit_seeds": fit_seeds,
            "validation_seeds": validation_seeds,
            "fresh_test_seeds": test_seeds,
            "cases_per_family": cases_per_family,
            "fit_cases": len(fit_rows),
            "validation_cases": len(validation_rows),
            "test_cases": len(test_rows),
        },
        "source_battery_sha256": sha256_json({
            "fit": fit_spec,
            "validation": validation_spec,
            "test": test_spec,
        }),
        "feature_contract": {
            "test_labels_visible_to_selector": False,
            "latent_generator_state_visible": False,
            "score_only_uses_family": False,
            "family_ranker_public_geometry": ["task family"],
            "all_rankers_use": [
                "representation identity",
                "predicted answer index",
                "normalized answer score vector",
                "top/second score",
                "margin",
                "entropy",
                "score spread",
                "cross-representation answer agreement",
            ],
        },
        "fit_controls": {
            "fixed_representation_accuracy": fixed_accuracy,
            "best_fixed_representation": best_fixed,
            "family_lookup": family_lookup,
        },
        "model_selection": {
            "score_ranker_trials": score_trials,
            "family_ranker_trials": family_trials,
            "selected_linear_for_gate": best_linear_name,
            "support_gate": gate,
            "gate_validation_trials": len(gate_trials),
        },
        "fresh_test": {
            "summaries": summaries,
            "oracle_representation_coverage": {
                "correct": oracle_hits,
                "cases": len(test_rows),
                "coverage": oracle_hits / len(test_rows),
            },
            "support_gate": gate_test,
        },
        "resources": {
            **scorer.counters(),
            "wall_seconds": time.time() - started,
            "representations_scored_per_case": len(ARMS),
            "learned_actor_parameters_added": 0,
            "selector_kind": "linear ridge over public calibration evidence",
        },
        "claim_boundary": (
            "Fresh public selector transfer only. Four representations are scored for "
            "diagnostic selection, so any accuracy gain is a discrimination result, "
            "not yet a compute-efficiency result. No protected promotion or frontier "
            "claim is authorized."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--fit-seeds", default="20260921,20260922")
    parser.add_argument("--validation-seeds", default="20260923")
    parser.add_argument("--test-seeds", default="20260924,20260925,20260926")
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--output", default="run-029-selector.json")
    args = parser.parse_args()

    model_path = Path(args.model_file)
    observed = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    parse = lambda value: [int(item) for item in value.split(",") if item.strip()]
    result = evaluate(
        args.model_dir,
        fit_seeds=parse(args.fit_seeds),
        validation_seeds=parse(args.validation_seeds),
        test_seeds=parse(args.test_seeds),
        cases_per_family=args.cases_per_family,
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
        "split": result["split"],
        "fit_controls": result["fit_controls"],
        "model_selection": {
            "score_ranker_trials": result["model_selection"]["score_ranker_trials"],
            "family_ranker_trials": result["model_selection"]["family_ranker_trials"],
            "selected_linear_for_gate": result["model_selection"]["selected_linear_for_gate"],
            "support_gate": result["model_selection"]["support_gate"],
        },
        "fresh_test": result["fresh_test"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
