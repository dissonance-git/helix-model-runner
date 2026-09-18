"""Run 037: deterministic representation composition.

Multiple obligation-preserving representations coexist in one prompt. No model-
generated intermediate state is used. This isolates whether A+B changes the
frozen actor's reasoning behavior beyond either singleton.
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
from offline_reasoning_content_probe import surface_task_and_choices
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import build_representations, _score_values

RUN_VERSION = "helix-deterministic-representation-composition-001.0"


def strip_answer_marker(prompt: str) -> str:
    value = prompt.rstrip()
    suffix = "Answer value:"
    if not value.endswith(suffix):
        raise ValueError("prompt lacks answer marker")
    return value[:-len(suffix)].rstrip()


def original_view(prompt: str) -> tuple[str, list[str]]:
    task, choices = surface_task_and_choices(prompt)
    return strip_answer_marker(task), choices


def case_views(case: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    names = sorted(case["surfaces"])
    transformed = build_representations(case["family"], case["surfaces"])
    layout_prompt, layout_choices = transformed["layout"]
    familiar_prompt, _familiar_choices = transformed["familiar"]
    primary_stem, _ = original_view(case["surfaces"][names[0]])
    secondary_stem, _ = original_view(case["surfaces"][names[1]])
    views = {
        "A": strip_answer_marker(layout_prompt),
        "B": strip_answer_marker(familiar_prompt),
        "P": primary_stem,
        "S": secondary_stem,
    }
    return views, layout_choices


def composite_prompt(order: list[str], views: dict[str, str]) -> str:
    blocks = []
    for index, name in enumerate(order, start=1):
        blocks.append(
            f"VIEW {index} [{name}]\n{views[name]}"
        )
    return (
        "The following are equivalent representations of the same reasoning task.\n"
        "Use all compatible structure. Preserve disagreements as uncertainty rather "
        "than deleting one view. Solve the single underlying obligation.\n\n"
        + "\n\n".join(blocks)
        + "\n\nAnswer value:"
    )


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
        "by_family": {k:sum(v)/len(v) for k,v in sorted(by_family.items())},
        "by_seed": {str(k):sum(v)/len(v) for k,v in sorted(by_seed.items())},
    }


def evaluate(model_dir: str, seeds: list[int], cases_per_family: int) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()
    started = time.time()
    rows = []
    public_spec = []

    for seed in seeds:
        cases = build_battery(seed, cases_per_family)
        seed_spec = []
        for case in cases:
            views, choices = case_views(case)
            prompts = {
                "A": views["A"] + "\nAnswer value:",
                "B": views["B"] + "\nAnswer value:",
                "AB": composite_prompt(["A","B"], views),
                "BA": composite_prompt(["B","A"], views),
                "PSAB": composite_prompt(["P","S","A","B"], views),
                "BASP": composite_prompt(["B","A","S","P"], views),
            }
            scored = {
                name:_score_values(scorer, prompt, choices)
                for name,prompt in prompts.items()
            }
            preds = {name:int(value["prediction"]) for name,value in scored.items()}
            correct = int(case["correct_index"])
            singleton_preds = [preds["A"],preds["B"],preds["P"] if "P" in preds else -1]
            # P/S are not separately scored in this run; source singleton coverage is A/B only.
            strict_ab = preds["A"] != correct and preds["B"] != correct and preds["AB"] == correct
            strict_ba = preds["A"] != correct and preds["B"] != correct and preds["BA"] == correct
            order_disagree = preds["AB"] != preds["BA"]
            order_one_correct = order_disagree and ((preds["AB"]==correct) != (preds["BA"]==correct))
            four_new = (
                preds["A"] != correct
                and preds["B"] != correct
                and preds["PSAB"] == correct
            )
            rows.append({
                "uid":f"{seed}:{case['case_id']}",
                "seed":seed,
                "case_id":case["case_id"],
                "family":case["family"],
                "difficulty":case["difficulty"],
                "correct_index":correct,
                "A_prediction":preds["A"],
                "B_prediction":preds["B"],
                "AB_prediction":preds["AB"],
                "BA_prediction":preds["BA"],
                "PSAB_prediction":preds["PSAB"],
                "BASP_prediction":preds["BASP"],
                "strict_AB_only_correct":strict_ab,
                "strict_BA_only_correct":strict_ba,
                "operator_order_disagrees":order_disagree,
                "operator_order_one_correct":order_one_correct,
                "four_view_composite_new_vs_AB_singletons":four_new,
                "prompt_sha256":{name:hashlib.sha256(p.encode()).hexdigest() for name,p in prompts.items()},
            })
            seed_spec.append({
                "case_id":case["case_id"],
                "family":case["family"],
                "difficulty":case["difficulty"],
                "correct_index":correct,
                "source_surface_sha256":{name:hashlib.sha256(prompt.encode()).hexdigest() for name,prompt in case["surfaces"].items()},
            })
        public_spec.append({"seed":seed,"cases":seed_spec})

    arms={
        "A":summarize(rows,"A_prediction"),
        "B":summarize(rows,"B_prediction"),
        "A+B":summarize(rows,"AB_prediction"),
        "B+A":summarize(rows,"BA_prediction"),
        "P+S+A+B":summarize(rows,"PSAB_prediction"),
        "B+A+S+P":summarize(rows,"BASP_prediction"),
    }
    seed_coverage=sum(
        r["A_prediction"]==r["correct_index"] or r["B_prediction"]==r["correct_index"]
        for r in rows
    )
    plus_pair=sum(
        r["A_prediction"]==r["correct_index"]
        or r["B_prediction"]==r["correct_index"]
        or r["AB_prediction"]==r["correct_index"]
        or r["BA_prediction"]==r["correct_index"]
        for r in rows
    )
    plus_all=sum(
        r["A_prediction"]==r["correct_index"]
        or r["B_prediction"]==r["correct_index"]
        or r["AB_prediction"]==r["correct_index"]
        or r["BA_prediction"]==r["correct_index"]
        or r["PSAB_prediction"]==r["correct_index"]
        or r["BASP_prediction"]==r["correct_index"]
        for r in rows
    )
    resources=scorer.counters()
    resources["wall_seconds"]=time.time()-started

    return {
        "schema_version":RUN_VERSION,
        "status":"fresh-public-deterministic-composition-calibration",
        "seeds":seeds,
        "cases_per_family":cases_per_family,
        "cases":len(rows),
        "source_battery_sha256":sha256_json(public_spec),
        "arms":arms,
        "emergence":{
            "A_or_B_coverage_correct":seed_coverage,
            "A_or_B_coverage":seed_coverage/len(rows),
            "through_ordered_pair_composites_correct":plus_pair,
            "through_ordered_pair_composites_coverage":plus_pair/len(rows),
            "through_all_composites_correct":plus_all,
            "through_all_composites_coverage":plus_all/len(rows),
            "strict_AB_only_correct_cases":sum(r["strict_AB_only_correct"] for r in rows),
            "strict_BA_only_correct_cases":sum(r["strict_BA_only_correct"] for r in rows),
            "operator_order_disagreement_cases":sum(r["operator_order_disagrees"] for r in rows),
            "operator_order_one_correct_cases":sum(r["operator_order_one_correct"] for r in rows),
            "four_view_composite_new_cases":sum(r["four_view_composite_new_vs_AB_singletons"] for r in rows),
        },
        "resources":resources,
        "rows":rows,
        "claim_boundary":(
            "Fresh public procedural diagnostic only. Composites contain only intact "
            "answer-blind representations and no generated intermediate text. Case-local "
            "new correctness is not yet a general representation-algebra claim."
        ),
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-dir",required=True)
    parser.add_argument("--model-file",required=True)
    parser.add_argument("--expected-sha256",required=True)
    parser.add_argument("--seeds",default="20261014,20261015,20261016")
    parser.add_argument("--cases-per-family",type=int,default=6)
    parser.add_argument("--output",default="run-037-deterministic-composition.json")
    args=parser.parse_args()

    model_path=Path(args.model_file)
    observed=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    result=evaluate(
        args.model_dir,
        [int(v) for v in args.seeds.split(",") if v.strip()],
        args.cases_per_family,
    )
    result["actor"]={
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
        "arms":result["arms"],
        "emergence":result["emergence"],
        "resources":result["resources"],
        "claim_boundary":result["claim_boundary"],
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
