"""Run 036: source-anchored synthesis.

Run 032 allowed a generated cross-view state to replace the task. This run keeps
the deterministic layout-normalized source intact and treats the generated state
only as an annotation. It also tests a simple answer-blind degeneration gate.
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
from run_032_cross_representation_invariant import (
    representations_for_case,
    score_values,
    summarize_view,
    synthesize_invariant,
)

RUN_VERSION = "helix-source-anchored-synthesis-001.0"


def strip_answer_marker(prompt: str) -> str:
    value = prompt.rstrip()
    suffix = "Answer value:"
    if not value.endswith(suffix):
        raise ValueError("prompt lacks answer marker")
    return value[:-len(suffix)].rstrip()


def quality_metrics(text: str) -> dict[str, Any]:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return {
            "tokens": 0,
            "unique_token_ratio": 0.0,
            "single_token_fraction": 1.0,
            "repeated_bigram_share": 1.0,
            "passes": False,
        }
    counts = Counter(tokens)
    bigrams = list(zip(tokens, tokens[1:]))
    bigram_counts = Counter(bigrams)
    unique_ratio = len(counts) / len(tokens)
    single_fraction = max(counts.values()) / len(tokens)
    repeated_bigram_share = (
        sum(v for v in bigram_counts.values() if v > 1) / max(1, len(bigrams))
    )
    passes = (
        len(tokens) >= 8
        and unique_ratio >= 0.50
        and single_fraction <= 0.15
        and repeated_bigram_share <= 0.25
    )
    return {
        "tokens": len(tokens),
        "unique_token_ratio": unique_ratio,
        "single_token_fraction": single_fraction,
        "repeated_bigram_share": repeated_bigram_share,
        "passes": passes,
    }


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


def pair_delta(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, int]:
    left_only = right_only = both = neither = 0
    for row in rows:
        a = int(row[left]) == int(row["correct_index"])
        b = int(row[right]) == int(row["correct_index"])
        if a and b:
            both += 1
        elif a:
            left_only += 1
        elif b:
            right_only += 1
        else:
            neither += 1
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "both_correct": both,
        "neither_correct": neither,
    }


def evaluate(model_dir: str, seeds: list[int], cases_per_family: int) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()
    started = time.time()
    rows: list[dict[str, Any]] = []
    public_specs = []

    for seed in seeds:
        cases = build_battery(seed, cases_per_family)
        seed_spec = []
        for local_index, case in enumerate(cases):
            reps = representations_for_case(case)
            layout_prompt, layout_choices = reps["layout-normalized"]
            layout_stem = strip_answer_marker(layout_prompt)
            source_score = score_values(scorer, layout_prompt, layout_choices)

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
            metrics = quality_metrics(invariant)

            invariant_prompt = (
                "Use the following derived shared state as the representation of the task.\n"
                f"Shared invariant state:\n{invariant}\n"
                "Answer value:"
            )
            invariant_score = score_values(
                scorer,
                invariant_prompt,
                layout_choices,
            )

            anchored_prompt = (
                "Solve the canonical task below. A derived annotation follows; use it only if "
                "it is compatible with the canonical task. The canonical task is authoritative.\n\n"
                "CANONICAL TASK:\n"
                f"{layout_stem}\n\n"
                "DERIVED ANNOTATION:\n"
                f"{invariant}\n\n"
                "Answer value:"
            )
            anchored_score = score_values(
                scorer,
                anchored_prompt,
                layout_choices,
            )

            gated_prediction = (
                int(anchored_score["prediction"])
                if metrics["passes"]
                else int(source_score["prediction"])
            )

            correct = int(case["correct_index"])
            rows.append({
                "uid": f"{seed}:{case['case_id']}",
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "source_prediction": int(source_score["prediction"]),
                "invariant_prediction": int(invariant_score["prediction"]),
                "anchored_prediction": int(anchored_score["prediction"]),
                "gated_anchored_prediction": gated_prediction,
                "quality": metrics,
                "invariant_sha256": hashlib.sha256(invariant.encode()).hexdigest(),
                "source_correct": int(source_score["prediction"]) == correct,
                "invariant_correct": int(invariant_score["prediction"]) == correct,
                "anchored_correct": int(anchored_score["prediction"]) == correct,
                "anchored_adds_new_correct_candidate": (
                    int(anchored_score["prediction"]) == correct
                    and int(source_score["prediction"]) != correct
                ),
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
            })
        public_specs.append({"seed": seed, "cases": seed_spec})

    pass_rows = [r for r in rows if r["quality"]["passes"]]
    fail_rows = [r for r in rows if not r["quality"]["passes"]]
    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started
    resources["summary_generations_per_case"] = 4
    resources["invariant_generations_per_case"] = 1

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-source-anchored-calibration",
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(rows),
        "source_battery_sha256": sha256_json(public_specs),
        "arms": {
            "layout-source-alone": summarize(rows, "source_prediction"),
            "generated-invariant-alone": summarize(rows, "invariant_prediction"),
            "layout-source-plus-invariant-annotation": summarize(rows, "anchored_prediction"),
            "syntax-gated-anchored-annotation": summarize(rows, "gated_anchored_prediction"),
        },
        "paired": {
            "anchored_vs_source": pair_delta(rows, "anchored_prediction", "source_prediction"),
            "invariant_vs_source": pair_delta(rows, "invariant_prediction", "source_prediction"),
            "gated_vs_source": pair_delta(rows, "gated_anchored_prediction", "source_prediction"),
        },
        "quality_gate": {
            "passes": len(pass_rows),
            "fails": len(fail_rows),
            "pass_fraction": len(pass_rows) / len(rows),
            "anchored_correct_when_pass": (
                sum(r["anchored_correct"] for r in pass_rows) / len(pass_rows)
                if pass_rows else None
            ),
            "anchored_correct_when_fail": (
                sum(r["anchored_correct"] for r in fail_rows) / len(fail_rows)
                if fail_rows else None
            ),
        },
        "candidate_effect": {
            "anchored_new_correct_candidates": sum(
                r["anchored_adds_new_correct_candidate"] for r in rows
            ),
            "source_correct_broken_by_anchor": sum(
                r["source_correct"] and not r["anchored_correct"] for r in rows
            ),
        },
        "resources": resources,
        "rows": rows,
        "claim_boundary": (
            "Fresh public procedural calibration. The canonical layout representation "
            "remains intact in anchored arms; generated annotations are advisory only. "
            "Syntax quality is not semantic verification. No protected promotion or "
            "frontier claim is authorized."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seeds", default="20261012,20261013")
    parser.add_argument("--cases-per-family", type=int, default=6)
    parser.add_argument("--output", default="run-036-anchored.json")
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
        "paired": result["paired"],
        "quality_gate": result["quality_gate"],
        "candidate_effect": result["candidate_effect"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
