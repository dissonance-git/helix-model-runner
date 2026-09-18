"""Run 031: support-aware representation cascade.

Uses the four answer-blind representations from Run 028. Calibration determines
family reliability, ordering, and stopping thresholds. Fresh-test labels are
never visible to the cascade.
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

from run_026_capability_substitution import AttackScorer
from run_029_representation_selector import ARMS, FAMILIES, _case_rows

RUN_VERSION = "helix-representation-cascade-001.0"


def correct(row: dict[str, Any], arm: str) -> bool:
    return bool(row["candidates"][arm]["correct"])


def family_reliability(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    reliability: dict[str, dict[str, float]] = {}
    ordering: dict[str, list[str]] = {}
    for family in FAMILIES:
        subset = [row for row in rows if row["family"] == family]
        if not subset:
            raise ValueError(f"missing family in fit rows: {family}")
        scores = {}
        for arm in ARMS:
            hits = sum(correct(row, arm) for row in subset)
            # Laplace-smoothed reliability is stable enough for weighted consensus.
            scores[arm] = (hits + 1.0) / (len(subset) + 2.0)
        reliability[family] = scores
        ordering[family] = sorted(
            ARMS,
            key=lambda arm: (-scores[arm], ARMS.index(arm)),
        )
    return reliability, ordering


def choose_weighted(row: dict[str, Any], weights: dict[str, float], *, weighted: bool) -> str:
    vote: dict[int, float] = defaultdict(float)
    for arm in ARMS:
        prediction = int(row["candidates"][arm]["prediction"])
        vote[prediction] += float(weights[arm] if weighted else 1.0)
    best_value = max(vote.values())
    tied_answers = {answer for answer, value in vote.items() if abs(value - best_value) < 1e-12}
    # Tie-break by most reliable arm whose prediction is tied.
    for arm in sorted(ARMS, key=lambda a: (-weights[a], ARMS.index(a))):
        if int(row["candidates"][arm]["prediction"]) in tied_answers:
            return arm
    raise AssertionError("weighted vote tie resolution failed")


def choose_highest_margin(row: dict[str, Any]) -> str:
    return max(
        ARMS,
        key=lambda arm: (
            float(row["candidates"][arm]["margin"]),
            -ARMS.index(arm),
        ),
    )


def simulate_cascade(
    row: dict[str, Any],
    *,
    order: list[str],
    weights: dict[str, float],
    threshold: float,
) -> tuple[str, int]:
    first, second = order[0], order[1]
    calls = 1
    if float(row["candidates"][first]["margin"]) >= threshold:
        return first, calls
    calls = 2
    if int(row["candidates"][first]["prediction"]) == int(row["candidates"][second]["prediction"]):
        return first, calls
    calls = 4
    return choose_weighted(row, weights, weighted=True), calls


def tune_thresholds(
    validation: list[dict[str, Any]],
    reliability: dict[str, dict[str, float]],
    ordering: dict[str, list[str]],
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds: dict[str, float] = {}
    trials: dict[str, Any] = {}
    for family in FAMILIES:
        rows = [row for row in validation if row["family"] == family]
        first = ordering[family][0]
        margins = sorted({float(row["candidates"][first]["margin"]) for row in rows})
        candidates = [-math.inf, *margins, math.inf]
        best = None
        family_trials = []
        for threshold in candidates:
            hits = 0
            calls = 0
            for row in rows:
                arm, used = simulate_cascade(
                    row,
                    order=ordering[family],
                    weights=reliability[family],
                    threshold=threshold,
                )
                hits += int(correct(row, arm))
                calls += used
            accuracy = hits / len(rows)
            mean_calls = calls / len(rows)
            trial = {
                "threshold": threshold if math.isfinite(threshold) else ("-inf" if threshold < 0 else "inf"),
                "accuracy": accuracy,
                "mean_calls": mean_calls,
            }
            family_trials.append(trial)
            key = (accuracy, -mean_calls, -threshold if math.isfinite(threshold) else (math.inf if threshold < 0 else -math.inf))
            if best is None or key > best[0]:
                best = (key, threshold, trial)
        assert best is not None
        thresholds[family] = float(best[1])
        trials[family] = {
            "selected": best[2],
            "trials": family_trials,
        }
    return thresholds, trials


def arm_choices(rows: list[dict[str, Any]], chooser) -> dict[str, str]:
    return {row["uid"]: chooser(row) for row in rows}


def summarize(rows: list[dict[str, Any]], choices: dict[str, str], *, calls: dict[str, int] | None = None) -> dict[str, Any]:
    hits = 0
    by_family: dict[str, list[bool]] = defaultdict(list)
    by_seed: dict[int, list[bool]] = defaultdict(list)
    selected = Counter()
    total_calls = 0
    call_hist = Counter()
    for row in rows:
        arm = choices[row["uid"]]
        ok = correct(row, arm)
        hits += int(ok)
        selected[arm] += 1
        by_family[row["family"]].append(ok)
        by_seed[int(row["seed"])].append(ok)
        if calls is not None:
            used = int(calls[row["uid"]])
            total_calls += used
            call_hist[used] += 1
    out = {
        "cases": len(rows),
        "correct": hits,
        "accuracy": hits / len(rows),
        "selected_representations": dict(selected),
        "by_family": {k: sum(v)/len(v) for k,v in sorted(by_family.items())},
        "by_seed": {str(k): sum(v)/len(v) for k,v in sorted(by_seed.items())},
    }
    if calls is not None:
        out["simulated_representation_calls"] = total_calls
        out["mean_representation_calls"] = total_calls / len(rows)
        out["call_histogram"] = {str(k): v for k,v in sorted(call_hist.items())}
        out["correct_per_representation_call"] = hits / total_calls if total_calls else 0.0
    return out


def pair_delta(rows: list[dict[str, Any]], a: dict[str, str], b: dict[str, str]) -> dict[str, int]:
    a_only = b_only = both = neither = 0
    for row in rows:
        ac = correct(row, a[row["uid"]])
        bc = correct(row, b[row["uid"]])
        if ac and bc:
            both += 1
        elif ac:
            a_only += 1
        elif bc:
            b_only += 1
        else:
            neither += 1
    return {"a_only": a_only, "b_only": b_only, "both": both, "neither": neither}


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
    validation_rows, validation_spec = _case_rows(scorer, validation_seeds, cases_per_family)
    test_rows, test_spec = _case_rows(scorer, test_seeds, cases_per_family)

    reliability, ordering = family_reliability(fit_rows)
    thresholds, threshold_trials = tune_thresholds(validation_rows, reliability, ordering)

    layout = {row["uid"]: "layout-normalized" for row in test_rows}
    family_lookup = {row["uid"]: ordering[row["family"]][0] for row in test_rows}
    highest_margin = arm_choices(test_rows, choose_highest_margin)
    unweighted_vote = {
        row["uid"]: choose_weighted(row, reliability[row["family"]], weighted=False)
        for row in test_rows
    }
    weighted_vote = {
        row["uid"]: choose_weighted(row, reliability[row["family"]], weighted=True)
        for row in test_rows
    }

    cascade = {}
    cascade_calls = {}
    for row in test_rows:
        arm, calls = simulate_cascade(
            row,
            order=ordering[row["family"]],
            weights=reliability[row["family"]],
            threshold=thresholds[row["family"]],
        )
        cascade[row["uid"]] = arm
        cascade_calls[row["uid"]] = calls

    oracle_hits = 0
    for row in test_rows:
        if any(correct(row, arm) for arm in ARMS):
            oracle_hits += 1

    summaries = {
        "fixed-layout": summarize(test_rows, layout, calls={row["uid"]:1 for row in test_rows}),
        "family-lookup": summarize(test_rows, family_lookup, calls={row["uid"]:1 for row in test_rows}),
        "highest-margin-four": summarize(test_rows, highest_margin, calls={row["uid"]:4 for row in test_rows}),
        "unweighted-majority-four": summarize(test_rows, unweighted_vote, calls={row["uid"]:4 for row in test_rows}),
        "calibration-weighted-vote-four": summarize(test_rows, weighted_vote, calls={row["uid"]:4 for row in test_rows}),
        "support-aware-cascade": summarize(test_rows, cascade, calls=cascade_calls),
    }

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-cascade-transfer",
        "split": {
            "fit_seeds": fit_seeds,
            "validation_seeds": validation_seeds,
            "fresh_test_seeds": test_seeds,
            "cases_per_family": cases_per_family,
            "fit_cases": len(fit_rows),
            "validation_cases": len(validation_rows),
            "test_cases": len(test_rows),
        },
        "source_battery_sha256": hashlib.sha256(
            json.dumps(
                {"fit": fit_spec, "validation": validation_spec, "test": test_spec},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "fit_reliability": reliability,
        "family_representation_order": ordering,
        "validation_thresholds": {
            family: (value if math.isfinite(value) else ("-inf" if value < 0 else "inf"))
            for family, value in thresholds.items()
        },
        "threshold_trials": threshold_trials,
        "fresh_test": {
            "summaries": summaries,
            "oracle_representation_coverage": {
                "correct": oracle_hits,
                "cases": len(test_rows),
                "coverage": oracle_hits / len(test_rows),
            },
            "cascade_vs_family_lookup": pair_delta(test_rows, cascade, family_lookup),
            "cascade_vs_layout": pair_delta(test_rows, cascade, layout),
            "weighted_vote_vs_family_lookup": pair_delta(test_rows, weighted_vote, family_lookup),
        },
        "actual_diagnostic_resources": {
            **scorer.counters(),
            "wall_seconds": time.time() - started,
            "representations_scored_per_case": 4,
            "note": "All four representations are scored in this diagnostic run so alternative policies can be compared on identical model evidence. Cascade call counts are simulated from the frozen score order."
        },
        "learned_actor_parameters_added": 0,
        "claim_boundary": (
            "Fresh public cascade transfer only. Simulated call savings are exact with respect "
            "to the frozen four-representation score evidence but are not wall-time measurements "
            "of an early-stopping implementation. No protected promotion or frontier claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--fit-seeds", default="20260921,20260922,20260923,20260924")
    parser.add_argument("--validation-seeds", default="20260925,20260926")
    parser.add_argument("--test-seeds", default="20260930,20261001,20261002")
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--output", default="run-031-cascade.json")
    args = parser.parse_args()

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
        "model":"HuggingFaceTB/SmolLM2-360M",
        "revision":"f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        "model_file_sha256":observed,
        "model_bytes":model_path.stat().st_size,
        "dtype":"float32",
        "device":"cpu",
    }
    Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({
        "schema_version":result["schema_version"],
        "split":result["split"],
        "family_representation_order":result["family_representation_order"],
        "validation_thresholds":result["validation_thresholds"],
        "fresh_test":result["fresh_test"],
        "claim_boundary":result["claim_boundary"],
    },indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
