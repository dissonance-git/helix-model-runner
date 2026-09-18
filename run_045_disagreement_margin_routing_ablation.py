"""Run 045: fresh disagreement-versus-margin routing ablation.

The experiment measures where to spend two additional sibling-representation
evaluations. It scores candidate-set coverage only; no answer selector is
executed or implied.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from offline_reasoning_battery import build_battery, sha256_json
from run_026_capability_substitution import AttackScorer
from run_032_cross_representation_invariant import representations_for_case, score_values

RUN_VERSION = "helix-disagreement-margin-routing-ablation-001.0"
BUDGETS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
RANDOM_SALTS = tuple(range(16))
CHEAP = ("layout-normalized", "familiar-structural-normalization")
ALL_VIEWS = (
    "lexicographic-primary-surface",
    "alternate-secondary-surface",
    "layout-normalized",
    "familiar-structural-normalization",
)


def stable_key(uid: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}|{uid}".encode()).hexdigest(), 16)


def policy_order(rows: list[dict[str, Any]], policy: str, salt: str = "0") -> list[int]:
    idx = list(range(len(rows)))
    tie = lambda i: stable_key(rows[i]["uid"], salt)
    if policy == "random":
        return sorted(idx, key=tie)
    if policy == "margin_only":
        return sorted(idx, key=lambda i: (rows[i]["layout_margin"], tie(i)))
    if policy == "disagreement_only":
        return sorted(idx, key=lambda i: (not rows[i]["cheap_disagree"], tie(i)))
    if policy == "disagreement_then_margin":
        return sorted(
            idx,
            key=lambda i: (
                not rows[i]["cheap_disagree"],
                rows[i]["layout_margin"],
                tie(i),
            ),
        )
    if policy == "disagreement_then_high_margin":
        return sorted(
            idx,
            key=lambda i: (
                not rows[i]["cheap_disagree"],
                -rows[i]["layout_margin"],
                tie(i),
            ),
        )
    raise ValueError(policy)


def evaluate_order(
    rows: list[dict[str, Any]],
    order: list[int],
    budgets: tuple[float, ...] = BUDGETS,
) -> list[dict[str, Any]]:
    n = len(rows)
    base_correct = sum(bool(row["layout_correct"]) for row in rows)
    curve = []
    for fraction in budgets:
        k = min(n, int(round(n * fraction)))
        selected = set(order[:k])
        benefits = sum(
            bool(row["benefit"])
            for i, row in enumerate(rows)
            if i in selected
        )
        correct = base_correct + benefits
        curve.append({
            "widen_fraction": fraction,
            "widen_cases": k,
            "candidate_coverage_correct": correct,
            "candidate_coverage": correct / n,
            "new_widening_benefits": benefits,
            "benefits_per_selected_case": benefits / k if k else 0.0,
            "mean_representation_evaluations_per_case": 2.0 + 2.0 * k / n,
        })
    return curve


def auc(curve: list[dict[str, Any]]) -> float:
    value = 0.0
    for left, right in zip(curve, curve[1:]):
        dx = right["widen_fraction"] - left["widen_fraction"]
        value += dx * (
            left["candidate_coverage"] + right["candidate_coverage"]
        ) / 2.0
    return value


def average_curves(curves: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for j in range(len(curves[0])):
        rows = [curve[j] for curve in curves]
        out.append({
            "widen_fraction": rows[0]["widen_fraction"],
            "widen_cases": rows[0]["widen_cases"],
            "candidate_coverage_mean": sum(r["candidate_coverage"] for r in rows) / len(rows),
            "candidate_coverage_min": min(r["candidate_coverage"] for r in rows),
            "candidate_coverage_max": max(r["candidate_coverage"] for r in rows),
            "new_widening_benefits_mean": sum(r["new_widening_benefits"] for r in rows) / len(rows),
            "new_widening_benefits_min": min(r["new_widening_benefits"] for r in rows),
            "new_widening_benefits_max": max(r["new_widening_benefits"] for r in rows),
            "mean_representation_evaluations_per_case": rows[0]["mean_representation_evaluations_per_case"],
        })
    return out


def family_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row)
    result = {}
    for family, group in sorted(groups.items()):
        disagree = [r for r in group if r["cheap_disagree"]]
        agree = [r for r in group if not r["cheap_disagree"]]
        result[family] = {
            "cases": len(group),
            "layout_correct": sum(bool(r["layout_correct"]) for r in group),
            "four_view_candidate_coverage_correct": sum(bool(r["full_oracle"]) for r in group),
            "benefit_cases": sum(bool(r["benefit"]) for r in group),
            "disagreement_cases": len(disagree),
            "disagreement_benefits": sum(bool(r["benefit"]) for r in disagree),
            "agreement_cases": len(agree),
            "agreement_benefits": sum(bool(r["benefit"]) for r in agree),
        }
    return result


def collect_rows(
    scorer: AttackScorer,
    seeds: list[int],
    cases_per_family: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    spec = []
    for seed in seeds:
        cases = build_battery(seed, cases_per_family)
        seed_spec = []
        for case in cases:
            reps = representations_for_case(case)
            scored = {
                name: score_values(scorer, prompt, choices)
                for name, (prompt, choices) in reps.items()
            }
            predictions = {
                name: int(scored[name]["prediction"])
                for name in ALL_VIEWS
            }
            correct = int(case["correct_index"])
            layout_correct = predictions["layout-normalized"] == correct
            full_oracle = any(predictions[name] == correct for name in ALL_VIEWS)
            cheap_disagree = predictions[CHEAP[0]] != predictions[CHEAP[1]]
            rows.append({
                "uid": f"{seed}:{case['case_id']}",
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "predictions": predictions,
                "layout_margin": float(scored["layout-normalized"]["margin"]),
                "familiar_margin": float(scored["familiar-structural-normalization"]["margin"]),
                "cheap_disagree": bool(cheap_disagree),
                "layout_correct": bool(layout_correct),
                "full_oracle": bool(full_oracle),
                "benefit": bool((not layout_correct) and full_oracle),
            })
            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "surface_sha256": {
                    name: hashlib.sha256(prompt.encode()).hexdigest()
                    for name, prompt in case["surfaces"].items()
                },
            })
        spec.append({"seed": seed, "cases": seed_spec})
    return rows, spec


def evaluate(model_dir: str, seeds: list[int], cases_per_family: int) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()
    started = time.time()

    rows, spec = collect_rows(scorer, seeds, cases_per_family)

    deterministic = {}
    for policy in (
        "margin_only",
        "disagreement_then_margin",
        "disagreement_then_high_margin",
    ):
        curve = evaluate_order(rows, policy_order(rows, policy, "fixed"))
        deterministic[policy] = {"curve": curve, "auc": auc(curve)}

    random_curves = [
        evaluate_order(rows, policy_order(rows, "random", f"random-{salt}"))
        for salt in RANDOM_SALTS
    ]
    random_aucs = [auc(curve) for curve in random_curves]

    disagree_curves = [
        evaluate_order(
            rows,
            policy_order(rows, "disagreement_only", f"disagreement-{salt}"),
        )
        for salt in RANDOM_SALTS
    ]
    disagree_aucs = [auc(curve) for curve in disagree_curves]

    disagree = [row for row in rows if row["cheap_disagree"]]
    agree = [row for row in rows if not row["cheap_disagree"]]
    wrong = [row for row in rows if not row["layout_correct"]]
    wrong_disagree = [row for row in wrong if row["cheap_disagree"]]
    wrong_agree = [row for row in wrong if not row["cheap_disagree"]]

    result = {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-routing-ablation",
        "api_contract": {
            "helix_api_commit": "8640aaf43bd1f80c27406b12bfa6767c9ea1f695",
            "consumes": ["research.representation-frontier"],
        },
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(rows),
        "source_battery_sha256": sha256_json(spec),
        "candidate_bank": {
            "layout_correct": sum(bool(r["layout_correct"]) for r in rows),
            "four_view_candidate_coverage_correct": sum(bool(r["full_oracle"]) for r in rows),
            "recoverable_widen_benefit_cases": sum(bool(r["benefit"]) for r in rows),
        },
        "disagreement_enrichment": {
            "all_cases": {
                "disagreement_cases": len(disagree),
                "disagreement_benefits": sum(bool(r["benefit"]) for r in disagree),
                "agreement_cases": len(agree),
                "agreement_benefits": sum(bool(r["benefit"]) for r in agree),
            },
            "layout_wrong_only": {
                "disagreement_cases": len(wrong_disagree),
                "disagreement_benefits": sum(bool(r["benefit"]) for r in wrong_disagree),
                "agreement_cases": len(wrong_agree),
                "agreement_benefits": sum(bool(r["benefit"]) for r in wrong_agree),
            },
        },
        "policies": {
            **deterministic,
            "random": {
                "salts": list(RANDOM_SALTS),
                "curve": average_curves(random_curves),
                "auc_mean": sum(random_aucs) / len(random_aucs),
                "auc_min": min(random_aucs),
                "auc_max": max(random_aucs),
            },
            "disagreement_only": {
                "salts": list(RANDOM_SALTS),
                "curve": average_curves(disagree_curves),
                "auc_mean": sum(disagree_aucs) / len(disagree_aucs),
                "auc_min": min(disagree_aucs),
                "auc_max": max(disagree_aucs),
            },
        },
        "by_family": family_summary(rows),
        "resources": {
            **scorer.counters(),
            "wall_seconds": time.time() - started,
        },
        "rows": rows,
        "claim_boundary": (
            "Fresh public candidate-acquisition ablation only. Policies never observe "
            "fresh correctness before routing. Candidate-set coverage is not selected-answer "
            "accuracy and does not authorize model promotion."
        ),
    }
    return result


def self_test() -> dict[str, Any]:
    rows = [
        {"uid":"a","cheap_disagree":True,"layout_margin":0.4,"layout_correct":False,"benefit":True},
        {"uid":"b","cheap_disagree":True,"layout_margin":0.1,"layout_correct":False,"benefit":True},
        {"uid":"c","cheap_disagree":False,"layout_margin":0.05,"layout_correct":True,"benefit":False},
        {"uid":"d","cheap_disagree":False,"layout_margin":0.8,"layout_correct":False,"benefit":False},
    ]
    dm = policy_order(rows, "disagreement_then_margin", "test")
    if [rows[i]["uid"] for i in dm[:2]] != ["b","a"]:
        raise AssertionError("compound ordering failed")
    margin = policy_order(rows, "margin_only", "test")
    if rows[margin[0]]["uid"] != "c":
        raise AssertionError("margin ordering failed")
    high = policy_order(rows, "disagreement_then_high_margin", "test")
    if [rows[i]["uid"] for i in high[:2]] != ["a","b"]:
        raise AssertionError("negative-control ordering failed")
    curve = evaluate_order(rows, dm, (0.0,0.5,1.0))
    if curve[0]["candidate_coverage_correct"] != 1:
        raise AssertionError("base coverage failed")
    if curve[1]["candidate_coverage_correct"] != 3:
        raise AssertionError("selected benefit accounting failed")
    return {
        "compound_order":[rows[i]["uid"] for i in dm],
        "margin_order":[rows[i]["uid"] for i in margin],
        "negative_control_order":[rows[i]["uid"] for i in high],
        "curve":curve,
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--seeds",default="20261109,20261110,20261111")
    parser.add_argument("--cases-per-family",type=int,default=6)
    parser.add_argument("--output",default="run-045-disagreement-margin-routing-ablation.json")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(),indent=2,sort_keys=True))
        return
    if not args.model_dir or not args.model_file or not args.expected_sha256:
        raise SystemExit("model-dir, model-file and expected-sha256 are required")

    model_path=Path(args.model_file)
    observed=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    result=evaluate(
        args.model_dir,
        [int(x) for x in args.seeds.split(",") if x.strip()],
        args.cases_per_family,
    )
    result["actor"]={
        "model":"HuggingFaceTB/SmolLM2-360M",
        "revision":"f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        "model_file_sha256":observed,
        "model_bytes":model_path.stat().st_size,
        "dtype":"float32",
        "device":"cpu",
        "weights_frozen":True,
    }
    Path(args.output).write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema_version":result["schema_version"],
        "candidate_bank":result["candidate_bank"],
        "disagreement_enrichment":result["disagreement_enrichment"],
        "policies":result["policies"],
        "by_family":result["by_family"],
        "resources":result["resources"],
        "claim_boundary":result["claim_boundary"],
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
