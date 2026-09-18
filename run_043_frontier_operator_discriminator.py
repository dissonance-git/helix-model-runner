"""Run 043: frontier-conditioned exact-operator discriminator.

The experiment asks whether the extra candidate coverage exposed by exact,
source-recoverable representation operators can be converted into selected-answer
accuracy by a simple answer-blind discriminator.

It also tests the Helix representation-frontier routing rule directly:
- sibling disagreement -> stay in the sibling neighborhood and discriminate;
- sibling unanimity -> rotate outside the family into exact operators.

Fresh test labels are never visible before selection. Oracle coverage is reported
only after selection as a diagnostic ceiling.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from offline_reasoning_battery import build_battery, sha256_json
from offline_reasoning_content_probe import surface_task_and_choices
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import build_representations, _score_values
from run_039_exact_representation_operators import operators_for_case

RUN_VERSION = "helix-frontier-conditioned-operator-discriminator-001.0"
SIBLING_ARMS = ("primary", "secondary", "source", "familiar")
EXACT_ARMS = ("op1", "op2")
ALL_ARMS = SIBLING_ARMS + EXACT_ARMS
FAMILIES = (
    "abstract-transformation",
    "grounded-planning",
    "relational-composition",
    "relational-matrix",
    "stack-language",
)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def _rank(values: Sequence[float], index: int) -> int:
    order = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))
    return order.index(index)


def _case_rows(
    scorer: AttackScorer,
    seeds: list[int],
    cases_per_family: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    public_spec: list[dict[str, Any]] = []
    for seed in seeds:
        cases = build_battery(seed, cases_per_family)
        seed_spec: list[dict[str, Any]] = []
        for case in cases:
            canonical = build_representations(case["family"], case["surfaces"])
            source_prompt, source_choices = canonical["layout"]
            familiar_prompt, familiar_choices = canonical["familiar"]
            surface_names = sorted(case["surfaces"])
            primary_prompt, primary_choices = surface_task_and_choices(
                case["surfaces"][surface_names[0]]
            )
            secondary_prompt, secondary_choices = surface_task_and_choices(
                case["surfaces"][surface_names[1]]
            )
            operators = operators_for_case(case["family"], case["surfaces"])
            if len(operators) != 2:
                raise AssertionError("expected exactly two exact operators")
            op1_id, op1_stem, op1_choices = operators[0]
            op2_id, op2_stem, op2_choices = operators[1]

            prompt_choices = {
                "primary": (primary_prompt, primary_choices),
                "secondary": (secondary_prompt, secondary_choices),
                "source": (source_prompt, source_choices),
                "familiar": (familiar_prompt, familiar_choices),
                "op1": (op1_stem + "\nAnswer value:", op1_choices),
                "op2": (op2_stem + "\nAnswer value:", op2_choices),
            }
            scored = {
                name: _score_values(scorer, prompt, choices)
                for name, (prompt, choices) in prompt_choices.items()
            }
            predictions = {
                name: int(scored[name]["prediction"])
                for name in ALL_ARMS
            }
            sibling_values = [predictions[name] for name in SIBLING_ARMS]
            exact_values = [predictions[name] for name in EXACT_ARMS]
            correct = int(case["correct_index"])

            rows.append({
                "uid": f"{seed}:{case['case_id']}",
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "operator_ids": [op1_id, op2_id],
                "scores": {
                    name: [float(v) for v in scored[name]["scores"]]
                    for name in ALL_ARMS
                },
                "margins": {
                    name: float(scored[name]["margin"])
                    for name in ALL_ARMS
                },
                "predictions": predictions,
                "sibling_unanimous": len(set(sibling_values)) == 1,
                "sibling_distinct_predictions": len(set(sibling_values)),
                "exact_agree": exact_values[0] == exact_values[1],
            })
            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "operator_ids": [op1_id, op2_id],
                "source_surface_sha256": {
                    name: hashlib.sha256(prompt.encode()).hexdigest()
                    for name, prompt in case["surfaces"].items()
                },
                "derived_prompt_sha256": {
                    name: hashlib.sha256(prompt.encode()).hexdigest()
                    for name, (prompt, _choices) in prompt_choices.items()
                },
            })
        public_spec.append({"seed": seed, "cases": seed_spec})
    return rows, public_spec


def _candidate_features(
    row: dict[str, Any],
    candidate: int,
    *,
    arms: tuple[str, ...],
    include_family: bool,
) -> list[float]:
    scores_by_arm = row["scores"]
    preds = row["predictions"]
    vector: list[float] = []

    # Candidate identity is public and can absorb stable answer-position bias.
    vector.extend(1.0 if candidate == j else 0.0 for j in range(4))

    candidate_scores = []
    for arm in arms:
        scores = scores_by_arm[arm]
        value = float(scores[candidate])
        top = max(float(v) for v in scores)
        candidate_scores.append(value)
        vector.extend([
            value,
            value - top,
            float(_rank(scores, candidate)) / 3.0,
            1.0 if int(preds[arm]) == candidate else 0.0,
            float(row["margins"][arm]),
        ])

    values = np.asarray(candidate_scores, dtype=float)
    sibling_votes = sum(
        int(preds[arm]) == candidate
        for arm in SIBLING_ARMS
        if arm in arms
    )
    exact_votes = sum(
        int(preds[arm]) == candidate
        for arm in EXACT_ARMS
        if arm in arms
    )
    vector.extend([
        float(sibling_votes),
        float(exact_votes),
        float(sibling_votes + exact_votes),
        float(values.mean()),
        float(values.max()),
        float(values.min()),
        float(values.std()),
        1.0 if row["sibling_unanimous"] else 0.0,
        float(row["sibling_distinct_predictions"]) / 4.0,
        1.0 if row["exact_agree"] and all(a in arms for a in EXACT_ARMS) else 0.0,
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
    arms: tuple[str, ...],
    include_family: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x: list[list[float]] = []
    y: list[float] = []
    for row in rows:
        for candidate in range(4):
            x.append(_candidate_features(
                row,
                candidate,
                arms=arms,
                include_family=include_family,
            ))
            y.append(1.0 if candidate == int(row["correct_index"]) else 0.0)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


class CandidateRidge:
    def __init__(self, alpha: float, arms: tuple[str, ...], include_family: bool):
        self.alpha = float(alpha)
        self.arms = tuple(arms)
        self.include_family = bool(include_family)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    def fit(self, rows: list[dict[str, Any]]) -> "CandidateRidge":
        x, y = _matrix(
            rows,
            arms=self.arms,
            include_family=self.include_family,
        )
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
        return self

    def score_candidate(self, row: dict[str, Any], candidate: int) -> float:
        assert self.mean is not None
        assert self.scale is not None
        assert self.coef is not None
        raw = np.asarray(
            _candidate_features(
                row,
                candidate,
                arms=self.arms,
                include_family=self.include_family,
            ),
            dtype=float,
        )
        z = (raw - self.mean) / self.scale
        return float(self.coef[0] + z @ self.coef[1:])

    def select(self, row: dict[str, Any]) -> int:
        scores = [self.score_candidate(row, candidate) for candidate in range(4)]
        return max(range(4), key=lambda candidate: (scores[candidate], -candidate))


def _accuracy(rows: list[dict[str, Any]], selected: dict[str, int]) -> float:
    return sum(
        int(selected[row["uid"]]) == int(row["correct_index"])
        for row in rows
    ) / len(rows)


def _tune_ranker(
    fit_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    arms: tuple[str, ...],
) -> tuple[CandidateRidge, dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    best: tuple[tuple[float, int, float], float, bool] | None = None
    for include_family in (False, True):
        for alpha in ALPHAS:
            ranker = CandidateRidge(alpha, arms, include_family).fit(fit_rows)
            selected = {row["uid"]: ranker.select(row) for row in validation_rows}
            accuracy = _accuracy(validation_rows, selected)
            # Accuracy first, then prefer no family feature, then stronger regularization.
            key = (accuracy, int(not include_family), alpha)
            trials.append({
                "alpha": alpha,
                "include_family": include_family,
                "validation_accuracy": accuracy,
            })
            if best is None or key > best[0]:
                best = (key, alpha, include_family)
    assert best is not None
    chosen_alpha = best[1]
    chosen_family = best[2]
    final = CandidateRidge(chosen_alpha, arms, chosen_family).fit(
        fit_rows + validation_rows
    )
    return final, {
        "trials": trials,
        "selected_alpha": chosen_alpha,
        "selected_include_family": chosen_family,
        "selected_validation_accuracy": best[0][0],
        "refit_cases": len(fit_rows) + len(validation_rows),
    }


def _vote_select(row: dict[str, Any], arms: tuple[str, ...]) -> int:
    counts = Counter(int(row["predictions"][arm]) for arm in arms)
    max_count = max(counts.values())
    tied = [candidate for candidate, count in counts.items() if count == max_count]
    if len(tied) == 1:
        return tied[0]
    return max(
        tied,
        key=lambda candidate: (
            sum(float(row["scores"][arm][candidate]) for arm in arms),
            -candidate,
        ),
    )


def _mean_score_select(row: dict[str, Any], arms: tuple[str, ...]) -> int:
    return max(
        range(4),
        key=lambda candidate: (
            float(np.mean([row["scores"][arm][candidate] for arm in arms])),
            -candidate,
        ),
    )


def _fixed_select(row: dict[str, Any], arm: str) -> int:
    return int(row["predictions"][arm])


def _stable_gate_ids(rows: list[dict[str, Any]], count: int) -> set[str]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            ("frontier-gate-control|" + row["uid"]).encode()
        ).hexdigest(),
    )
    return {row["uid"] for row in ranked[:count]}


def _summary(
    rows: list[dict[str, Any]],
    selected: dict[str, int],
    *,
    layout: dict[str, int],
    sibling_ranker: dict[str, int],
    representation_calls: dict[str, int],
) -> dict[str, Any]:
    by_family: dict[str, list[bool]] = defaultdict(list)
    by_seed: dict[int, list[bool]] = defaultdict(list)
    hits = 0
    wins_layout = 0
    regress_layout = 0
    wins_sibling = 0
    regress_sibling = 0
    total_calls = 0
    for row in rows:
        uid = row["uid"]
        correct = int(row["correct_index"])
        ok = int(selected[uid]) == correct
        layout_ok = int(layout[uid]) == correct
        sibling_ok = int(sibling_ranker[uid]) == correct
        hits += int(ok)
        wins_layout += int(ok and not layout_ok)
        regress_layout += int(layout_ok and not ok)
        wins_sibling += int(ok and not sibling_ok)
        regress_sibling += int(sibling_ok and not ok)
        by_family[row["family"]].append(ok)
        by_seed[int(row["seed"])].append(ok)
        total_calls += int(representation_calls[uid])
    return {
        "cases": len(rows),
        "correct": hits,
        "accuracy": hits / len(rows),
        "wins_vs_layout": wins_layout,
        "regressions_vs_layout": regress_layout,
        "wins_vs_sibling_ranker": wins_sibling,
        "regressions_vs_sibling_ranker": regress_sibling,
        "representation_evaluations": total_calls,
        "mean_representation_evaluations": total_calls / len(rows),
        "correct_per_representation_evaluation": (
            hits / total_calls if total_calls else 0.0
        ),
        "by_family": {
            key: sum(values) / len(values)
            for key, values in sorted(by_family.items())
        },
        "by_seed": {
            str(key): sum(values) / len(values)
            for key, values in sorted(by_seed.items())
        },
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

    sibling_ranker_model, sibling_fit = _tune_ranker(
        fit_rows,
        validation_rows,
        arms=SIBLING_ARMS,
    )
    all_ranker_model, all_fit = _tune_ranker(
        fit_rows,
        validation_rows,
        arms=ALL_ARMS,
    )

    layout = {
        row["uid"]: _fixed_select(row, "source")
        for row in test_rows
    }
    sibling_majority = {
        row["uid"]: _vote_select(row, SIBLING_ARMS)
        for row in test_rows
    }
    sibling_mean = {
        row["uid"]: _mean_score_select(row, SIBLING_ARMS)
        for row in test_rows
    }
    sibling_ranker = {
        row["uid"]: sibling_ranker_model.select(row)
        for row in test_rows
    }
    all_mean = {
        row["uid"]: _mean_score_select(row, ALL_ARMS)
        for row in test_rows
    }
    all_ranker = {
        row["uid"]: all_ranker_model.select(row)
        for row in test_rows
    }

    unanimous_ids = {
        row["uid"] for row in test_rows if row["sibling_unanimous"]
    }
    split_ids = {
        row["uid"] for row in test_rows if not row["sibling_unanimous"]
    }
    matched_ids = _stable_gate_ids(test_rows, len(unanimous_ids))

    helix_gate = {
        row["uid"]: (
            all_ranker[row["uid"]]
            if row["uid"] in unanimous_ids
            else sibling_ranker[row["uid"]]
        )
        for row in test_rows
    }
    inverse_gate = {
        row["uid"]: (
            all_ranker[row["uid"]]
            if row["uid"] in split_ids
            else sibling_ranker[row["uid"]]
        )
        for row in test_rows
    }
    matched_gate = {
        row["uid"]: (
            all_ranker[row["uid"]]
            if row["uid"] in matched_ids
            else sibling_ranker[row["uid"]]
        )
        for row in test_rows
    }

    calls_1 = {row["uid"]: 1 for row in test_rows}
    calls_4 = {row["uid"]: 4 for row in test_rows}
    calls_6 = {row["uid"]: 6 for row in test_rows}
    calls_helix = {
        row["uid"]: 6 if row["uid"] in unanimous_ids else 4
        for row in test_rows
    }
    calls_inverse = {
        row["uid"]: 6 if row["uid"] in split_ids else 4
        for row in test_rows
    }
    calls_matched = {
        row["uid"]: 6 if row["uid"] in matched_ids else 4
        for row in test_rows
    }

    policies = {
        "layout-fixed": (layout, calls_1),
        "sibling-majority": (sibling_majority, calls_4),
        "sibling-score-mean": (sibling_mean, calls_4),
        "sibling-learned-discriminator": (sibling_ranker, calls_4),
        "all-six-score-mean": (all_mean, calls_6),
        "all-six-learned-discriminator": (all_ranker, calls_6),
        "helix-frontier-unanimity-gate": (helix_gate, calls_helix),
        "inverse-split-gate-control": (inverse_gate, calls_inverse),
        "matched-frequency-hash-gate-control": (matched_gate, calls_matched),
    }
    summaries = {
        name: _summary(
            test_rows,
            selected,
            layout=layout,
            sibling_ranker=sibling_ranker,
            representation_calls=calls,
        )
        for name, (selected, calls) in policies.items()
    }

    sibling_oracle = sum(
        any(
            int(row["predictions"][arm]) == int(row["correct_index"])
            for arm in SIBLING_ARMS
        )
        for row in test_rows
    )
    six_oracle = sum(
        any(
            int(row["predictions"][arm]) == int(row["correct_index"])
            for arm in ALL_ARMS
        )
        for row in test_rows
    )
    unanimous_wrong = [
        row for row in test_rows
        if row["sibling_unanimous"]
        and not any(
            int(row["predictions"][arm]) == int(row["correct_index"])
            for arm in SIBLING_ARMS
        )
    ]
    split_wrong = [
        row for row in test_rows
        if (not row["sibling_unanimous"])
        and not any(
            int(row["predictions"][arm]) == int(row["correct_index"])
            for arm in SIBLING_ARMS
        )
    ]

    result = {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-frontier-discriminator-transfer",
        "api_contract": {
            "helix_api_commit": "8640aaf43bd1f80c27406b12bfa6767c9ea1f695",
            "consumes": [
                "research.representation-search",
                "research.representation-frontier",
            ],
        },
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
            "test_labels_visible_before_selection": False,
            "generated_intermediate_text": False,
            "candidate_selector_outputs_one_answer": True,
            "operator_views_source_recoverable": True,
            "sibling_ranker_arms": list(SIBLING_ARMS),
            "all_ranker_arms": list(ALL_ARMS),
        },
        "model_selection": {
            "sibling_ranker": sibling_fit,
            "all_six_ranker": all_fit,
        },
        "fresh_test": {
            "summaries": summaries,
            "sibling_unanimous_cases": len(unanimous_ids),
            "sibling_split_cases": len(split_ids),
            "matched_frequency_exact_cases": len(matched_ids),
            "oracle_ceiling": {
                "four_sibling_correct": sibling_oracle,
                "four_sibling_coverage": sibling_oracle / len(test_rows),
                "six_view_correct": six_oracle,
                "six_view_coverage": six_oracle / len(test_rows),
                "note": "coverage ceiling only; not selected-answer accuracy",
            },
            "frontier_rescue": {
                "unanimous_wrong_cases": len(unanimous_wrong),
                "unanimous_wrong_exact_operator_rescues": sum(
                    any(
                        int(row["predictions"][arm]) == int(row["correct_index"])
                        for arm in EXACT_ARMS
                    )
                    for row in unanimous_wrong
                ),
                "split_wrong_cases": len(split_wrong),
                "split_wrong_exact_operator_rescues": sum(
                    any(
                        int(row["predictions"][arm]) == int(row["correct_index"])
                        for arm in EXACT_ARMS
                    )
                    for row in split_wrong
                ),
            },
        },
        "resources": {
            **scorer.counters(),
            "wall_seconds": time.time() - started,
            "diagnostic_representations_scored_per_case": 6,
            "learned_actor_parameters_added": 0,
            "selector_kind": "candidate-level ridge ranker",
            "gate_call_counts_are_simulated_from_identical_six-view_evidence": True,
        },
        "claim_boundary": (
            "Fresh public selector transfer. Oracle coverage is diagnostic only. "
            "A discriminator gain requires the fresh selector itself to return more "
            "correct answers. Gate call savings are exact with respect to the frozen "
            "six-view evidence but are simulated rather than an early-stopping wall-time "
            "measurement. Exact symbolic operators are substrate capability, not actor-only "
            "reasoning. No protected promotion or frontier-equivalence claim."
        ),
    }
    return result


def self_test(seed: int = 20261102, cases_per_family: int = 1) -> dict[str, Any]:
    rows = []
    operator_counts: dict[str, int] = defaultdict(int)
    digests: list[str] = []
    for case in build_battery(seed, cases_per_family):
        operators = operators_for_case(case["family"], case["surfaces"])
        if len(operators) != 2:
            raise AssertionError((case["case_id"], len(operators)))
        for operator_id, stem, choices in operators:
            if not stem or len(choices) != 4:
                raise AssertionError((case["case_id"], operator_id))
            operator_counts[operator_id] += 1
            digests.append(hashlib.sha256(
                json.dumps(
                    {"id": operator_id, "stem": stem, "choices": choices},
                    sort_keys=True,
                ).encode()
            ).hexdigest())
        dummy = {
            "family": case["family"],
            "scores": {
                arm: [-3.0, -2.0, -1.0, -4.0]
                for arm in ALL_ARMS
            },
            "margins": {arm: 1.0 for arm in ALL_ARMS},
            "predictions": {arm: 2 for arm in ALL_ARMS},
            "sibling_unanimous": True,
            "sibling_distinct_predictions": 1,
            "exact_agree": True,
        }
        sibling_dim = len(_candidate_features(
            dummy, 0, arms=SIBLING_ARMS, include_family=False
        ))
        all_dim = len(_candidate_features(
            dummy, 0, arms=ALL_ARMS, include_family=False
        ))
        if all_dim <= sibling_dim:
            raise AssertionError((sibling_dim, all_dim))
        rows.append({
            "case_id": case["case_id"],
            "family": case["family"],
            "sibling_feature_dim": sibling_dim,
            "all_feature_dim": all_dim,
        })
    return {
        "cases": len(rows),
        "operator_counts": dict(sorted(operator_counts.items())),
        "transform_digest": hashlib.sha256("|".join(digests).encode()).hexdigest(),
        "feature_contracts": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--fit-seeds", default="20261102,20261103,20261104")
    parser.add_argument("--validation-seeds", default="20261105")
    parser.add_argument("--test-seeds", default="20261106,20261107,20261108")
    parser.add_argument("--cases-per-family", type=int, default=4)
    parser.add_argument("--output", default="run-043-frontier-discriminator.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    if not args.model_dir or not args.model_file or not args.expected_sha256:
        raise SystemExit("model-dir, model-file and expected-sha256 are required")

    model_path = Path(args.model_file)
    observed = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    parse = lambda text: [int(v) for v in text.split(",") if v.strip()]
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
        "model_selection": result["model_selection"],
        "fresh_test": result["fresh_test"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
