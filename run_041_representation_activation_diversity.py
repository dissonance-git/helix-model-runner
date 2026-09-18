"""Run 041: representation activation diversity.

Measure answer-blind distance between equivalent representations inside the
exact frozen SmolLM2-360M actor and test whether distant pairs have more
complementary correctness than nearby pairs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from offline_reasoning_battery import build_battery, sha256_json
from run_026_capability_substitution import AttackScorer
from run_032_cross_representation_invariant import (
    representations_for_case,
    score_values,
    strip_answer_marker,
)

RUN_VERSION = "helix-representation-activation-diversity-001.0"


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.float(), dim=0)
    b = F.normalize(b.float(), dim=0)
    return float(1.0 - torch.dot(a, b))


def lexical_distance(tokenizer, left: str, right: str) -> float:
    a = set(tokenizer(left, add_special_tokens=False).input_ids)
    b = set(tokenizer(right, add_special_tokens=False).input_ids)
    union = a | b
    if not union:
        return 0.0
    return 1.0 - (len(a & b) / len(union))


@torch.inference_mode()
def representation_embedding(
    scorer: AttackScorer,
    prompt: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    visible = strip_answer_marker(prompt)
    encoded = scorer.tokenizer(
        visible,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(scorer.device)
    attention = encoded.get("attention_mask")
    if attention is not None:
        attention = attention.to(scorer.device)
    output = scorer.model(
        input_ids=input_ids,
        attention_mask=attention,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = output.hidden_states[-1][0]
    last = hidden[-1].detach().cpu()
    mean = hidden.mean(dim=0).detach().cpu()
    return last, mean, int(input_ids.shape[1])


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2:
        return None
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    va = sum(x*x for x in da)
    vb = sum(x*x for x in db)
    if va == 0 or vb == 0:
        return None
    return sum(x*y for x,y in zip(da,db,strict=True)) / math.sqrt(va*vb)


def spearman(a: list[float], b: list[float]) -> float | None:
    return pearson(rankdata(a), rankdata(b))


def policy_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    contains = sum(row[key]["contains_correct"] for row in rows)
    complementary = sum(row[key]["exactly_one_correct"] for row in rows)
    both = sum(row[key]["both_correct"] for row in rows)
    by_family: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(bool(row[key]["contains_correct"]))
    return {
        "cases": len(rows),
        "pair_oracle_correct": contains,
        "pair_oracle_coverage": contains / len(rows),
        "exactly_one_correct_cases": complementary,
        "complementarity_rate": complementary / len(rows),
        "both_correct_cases": both,
        "by_family_pair_oracle": {
            family: sum(values)/len(values)
            for family, values in sorted(by_family.items())
        },
    }


def evaluate(model_dir: str, seeds: list[int], cases_per_family: int) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()
    started = time.time()
    embedding_calls = 0
    embedding_tokens = 0
    rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    public_spec = []
    pair_choice_counts: dict[str, Counter] = {
        "farthest_last": Counter(),
        "nearest_last": Counter(),
        "farthest_mean": Counter(),
        "nearest_mean": Counter(),
        "farthest_lexical": Counter(),
    }

    for seed in seeds:
        cases = build_battery(seed, cases_per_family)
        seed_spec = []
        for case in cases:
            uid = f"{seed}:{case['case_id']}"
            reps = representations_for_case(case)
            scored = {
                name: score_values(scorer, prompt, choices)
                for name, (prompt, choices) in reps.items()
            }
            predictions = {
                name: int(value["prediction"])
                for name, value in scored.items()
            }

            embeddings = {}
            visible_text = {}
            for name, (prompt, _choices) in reps.items():
                last, mean, tokens = representation_embedding(scorer, prompt)
                embeddings[name] = {"last": last, "mean": mean}
                visible_text[name] = strip_answer_marker(prompt)
                embedding_calls += 1
                embedding_tokens += tokens

            pairs = []
            for left, right in combinations(sorted(reps), 2):
                left_correct = predictions[left] == case["correct_index"]
                right_correct = predictions[right] == case["correct_index"]
                pair = {
                    "left": left,
                    "right": right,
                    "pair_id": f"{left}+{right}",
                    "last_distance": cosine_distance(
                        embeddings[left]["last"], embeddings[right]["last"]
                    ),
                    "mean_distance": cosine_distance(
                        embeddings[left]["mean"], embeddings[right]["mean"]
                    ),
                    "lexical_distance": lexical_distance(
                        scorer.tokenizer, visible_text[left], visible_text[right]
                    ),
                    "contains_correct": bool(left_correct or right_correct),
                    "exactly_one_correct": bool(left_correct != right_correct),
                    "both_correct": bool(left_correct and right_correct),
                }
                pairs.append(pair)
                all_pair_rows.append({
                    "uid": uid,
                    "family": case["family"],
                    **pair,
                })

            def select(metric: str, farthest: bool) -> dict[str, Any]:
                ordered = sorted(
                    pairs,
                    key=lambda row: (
                        row[metric],
                        row["pair_id"],
                    ),
                    reverse=farthest,
                )
                return ordered[0]

            selections = {
                "farthest_last": select("last_distance", True),
                "nearest_last": select("last_distance", False),
                "farthest_mean": select("mean_distance", True),
                "nearest_mean": select("mean_distance", False),
                "farthest_lexical": select("lexical_distance", True),
            }
            fixed = next(
                row for row in pairs
                if set((row["left"], row["right"])) == {
                    "layout-normalized",
                    "familiar-structural-normalization",
                }
            )
            selections["fixed_layout_familiar"] = fixed

            for key, selected in selections.items():
                if key in pair_choice_counts:
                    pair_choice_counts[key][selected["pair_id"]] += 1

            complement = [1.0 if row["exactly_one_correct"] else 0.0 for row in pairs]
            case_spearman = {
                "last": spearman(
                    [row["last_distance"] for row in pairs], complement
                ),
                "mean": spearman(
                    [row["mean_distance"] for row in pairs], complement
                ),
                "lexical": spearman(
                    [row["lexical_distance"] for row in pairs], complement
                ),
            }

            rows.append({
                "uid": uid,
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": case["correct_index"],
                "predictions": predictions,
                "selections": selections,
                "within_case_spearman_distance_vs_complementarity": case_spearman,
                "pair_rows": pairs,
            })
            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": case["correct_index"],
                "representation_sha256": {
                    name: hashlib.sha256(
                        json.dumps(
                            {"prompt": prompt, "choices": choices},
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()
                    for name, (prompt, choices) in reps.items()
                },
            })
        public_spec.append({"seed": seed, "cases": seed_spec})

    flattened = []
    for row in rows:
        flat = {
            "uid": row["uid"],
            "seed": row["seed"],
            "case_id": row["case_id"],
            "family": row["family"],
            "difficulty": row["difficulty"],
            "correct_index": row["correct_index"],
        }
        for key, value in row["selections"].items():
            flat[key] = value
        flattened.append(flat)

    metrics = {}
    for key in (
        "farthest_last",
        "nearest_last",
        "farthest_mean",
        "nearest_mean",
        "farthest_lexical",
        "fixed_layout_familiar",
    ):
        metrics[key] = policy_summary(flattened, key)

    correlations = {}
    for metric_key in ("last", "mean", "lexical"):
        vals = [
            row["within_case_spearman_distance_vs_complementarity"][metric_key]
            for row in rows
            if row["within_case_spearman_distance_vs_complementarity"][metric_key] is not None
        ]
        correlations[metric_key] = {
            "defined_cases": len(vals),
            "mean_within_case_spearman": sum(vals)/len(vals) if vals else None,
        }

    def quartiles(metric: str) -> list[dict[str, Any]]:
        ordered = sorted(all_pair_rows, key=lambda row: row[metric])
        n = len(ordered)
        out = []
        for q in range(4):
            lo = (n*q)//4
            hi = (n*(q+1))//4
            chunk = ordered[lo:hi]
            out.append({
                "quartile": q+1,
                "pairs": len(chunk),
                "mean_distance": sum(row[metric] for row in chunk)/len(chunk),
                "contains_correct_rate": sum(row["contains_correct"] for row in chunk)/len(chunk),
                "complementarity_rate": sum(row["exactly_one_correct"] for row in chunk)/len(chunk),
            })
        return out

    resources = scorer.counters()
    resources.update({
        "embedding_forward_calls": embedding_calls,
        "embedding_input_tokens": embedding_tokens,
        "wall_seconds": time.time()-started,
    })

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-activation-diversity-calibration",
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(rows),
        "source_battery_sha256": sha256_json(public_spec),
        "pair_policies": metrics,
        "pair_choice_counts": {
            key: dict(value) for key, value in pair_choice_counts.items()
        },
        "distance_vs_complementarity": {
            "within_case_spearman": correlations,
            "last_distance_quartiles": quartiles("last_distance"),
            "mean_distance_quartiles": quartiles("mean_distance"),
            "lexical_distance_quartiles": quartiles("lexical_distance"),
        },
        "resources": resources,
        "rows": rows,
        "claim_boundary": (
            "Fresh public procedural diagnostic only. Representation distances are "
            "computed answer-blind before correctness is consulted. Pair-oracle coverage "
            "is a diagnostic of complementarity, not a deployable answer selector and not "
            "protected promotion or frontier evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seeds", default="20261025,20261026,20261027")
    parser.add_argument("--cases-per-family", type=int, default=6)
    parser.add_argument("--output", default="run-041-activation-diversity.json")
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
        "pair_policies": result["pair_policies"],
        "distance_vs_complementarity": result["distance_vs_complementarity"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
