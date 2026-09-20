"""Run 059: fresh structural replication of Run 058 family-specific routes."""
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
from run_028_canonical_representation import _score_values
from run_058_cross_family_compiler_portfolio import (
    _compact_program_record,
    _execute,
    _source_state,
)

RUN_VERSION = "helix-family-route-structure-replication-001.0"
SEEDS_DEFAULT = "20261231,20270101"
CASES_PER_FAMILY_DEFAULT = 4

TARGET_CONFIG = {
    "abstract-transformation": {
        "selected": ("first", "compose"),
        "negative": ("first", "second", "compose"),
        "selected_name": "first-stop",
        "negative_name": "forward",
    },
    "stack-language": {
        "selected": ("first", "second", "compose"),
        "negative": ("first", "compose"),
        "selected_name": "forward",
        "negative_name": "first-stop",
    },
}

GRID_SUBTYPE = {
    0: "rotate-90",
    1: "flip-horizontal",
    2: "flip-vertical",
    3: "transpose",
}


def _margin(scores: list[float]) -> float:
    values = sorted((float(x) for x in scores), reverse=True)
    return values[0] - values[1]


def _paired(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, int]:
    wins = regressions = ties = 0
    for row in rows:
        c = int(row["correct_index"])
        l = int(row["predictions"][left]) == c
        r = int(row["predictions"][right]) == c
        if l and not r:
            wins += 1
        elif r and not l:
            regressions += 1
        else:
            ties += 1
    return {"wins": wins, "regressions": regressions, "ties": ties}


def _summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    hits = sum(int(r["predictions"][arm] == r["correct_index"]) for r in rows)
    return {"cases": len(rows), "correct": hits, "accuracy": hits / len(rows) if rows else None}


def _subtype(case: dict[str, Any]) -> str | None:
    if case["family"] != "abstract-transformation":
        return None
    index = int(case["case_id"].split("-")[1])
    return GRID_SUBTYPE[index]


def self_test(helix_dir: str) -> dict[str, Any]:
    rows = []
    for case in build_battery(20261231, CASES_PER_FAMILY_DEFAULT):
        state = _source_state(case["family"], case["case_id"], case["surfaces"])
        row = {
            "case_id": case["case_id"],
            "family": case["family"],
            "difficulty": case["difficulty"],
            "subtype": _subtype(case),
            "correctness_visible_to_compiler": "correct_index" in state,
        }
        if case["family"] in TARGET_CONFIG:
            cfg = TARGET_CONFIG[case["family"]]
            selected, srec = _execute(
                helix_dir, state, cfg["selected"],
                {"run_id":"059-family-route-structure-replication","self_test":True,"arm":"selected"},
            )
            negative, nrec = _execute(
                helix_dir, state, cfg["negative"],
                {"run_id":"059-family-route-structure-replication","self_test":True,"arm":"negative"},
            )
            row["selected_basis"] = srec["basis_operators"]
            row["negative_basis"] = nrec["basis_operators"]
            row["selected_recovery_complete"] = srec.get("recovery_route") is not None
            row["negative_recovery_complete"] = nrec.get("recovery_route") is not None
            row["selected_prompt_sha256"] = hashlib.sha256(selected["final_prompt"].encode()).hexdigest()
            row["negative_prompt_sha256"] = hashlib.sha256(negative["final_prompt"].encode()).hexdigest()
        rows.append(row)
    if any(r["correctness_visible_to_compiler"] for r in rows):
        raise AssertionError("correctness leaked into compiler state")
    return {
        "schema_version": RUN_VERSION,
        "status": "self-test-pass",
        "cases": len(rows),
        "actual_execution_owner": "engine.ir.passes.execute_transform_program",
        "target_config": {k:{kk:list(vv) if isinstance(vv,tuple) else vv for kk,vv in v.items()} for k,v in TARGET_CONFIG.items()},
        "rows": rows,
    }


def evaluate(helix_dir: str, model_dir: str, seeds: list[int], cases_per_family: int) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()
    started = time.time()
    rows: list[dict[str, Any]] = []
    public_spec = []

    for seed in seeds:
        seed_spec = []
        for case in build_battery(seed, cases_per_family):
            state = _source_state(case["family"], case["case_id"], case["surfaces"])
            source = _score_values(scorer, state["source_prompt"], state["source_choices"])
            predictions = {"source": int(source["prediction"])}
            scores = {"source": [float(x) for x in source["scores"]]}
            program_records = {}

            if case["family"] in TARGET_CONFIG:
                cfg = TARGET_CONFIG[case["family"]]
                selected_state, selected_record = _execute(
                    helix_dir, state, cfg["selected"],
                    {
                        "run_id":"059-family-route-structure-replication",
                        "seed":seed,
                        "case_id":case["case_id"],
                        "arm":"selected",
                    },
                )
                negative_state, negative_record = _execute(
                    helix_dir, state, cfg["negative"],
                    {
                        "run_id":"059-family-route-structure-replication",
                        "seed":seed,
                        "case_id":case["case_id"],
                        "arm":"negative",
                    },
                )
                selected = _score_values(scorer, selected_state["final_prompt"], selected_state["final_choices"])
                negative = _score_values(scorer, negative_state["final_prompt"], negative_state["final_choices"])
                predictions["selected"] = int(selected["prediction"])
                predictions["negative"] = int(negative["prediction"])
                scores["selected"] = [float(x) for x in selected["scores"]]
                scores["negative"] = [float(x) for x in negative["scores"]]
                program_records["selected"] = _compact_program_record(selected_record)
                program_records["negative"] = _compact_program_record(negative_record)
            else:
                predictions["selected"] = predictions["source"]
                predictions["negative"] = predictions["source"]
                scores["selected"] = list(scores["source"])
                scores["negative"] = list(scores["source"])

            row = {
                "uid": f"{seed}:{case['case_id']}",
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "subtype": _subtype(case),
                "correct_index": int(case["correct_index"]),
                "predictions": predictions,
                "margins": {name:_margin(value) for name,value in scores.items()},
                "program_records": program_records,
            }
            rows.append(row)
            seed_spec.append({
                "case_id":case["case_id"],
                "family":case["family"],
                "difficulty":case["difficulty"],
                "subtype":_subtype(case),
                "correct_index":int(case["correct_index"]),
                "surface_sha256":{
                    name:hashlib.sha256(prompt.encode()).hexdigest()
                    for name,prompt in sorted(case["surfaces"].items())
                },
            })
        public_spec.append({"seed":seed,"cases":seed_spec})

    target_rows=[r for r in rows if r["family"] in TARGET_CONFIG]
    family_results={}
    for family,cfg in TARGET_CONFIG.items():
        fam=[r for r in rows if r["family"]==family]
        result={
            "source":_summary(fam,"source"),
            "selected":_summary(fam,"selected"),
            "negative":_summary(fam,"negative"),
            "selected_vs_source":_paired(fam,"selected","source"),
            "selected_vs_negative":_paired(fam,"selected","negative"),
        }
        if family=="abstract-transformation":
            result["by_subtype"]={}
            for subtype in GRID_SUBTYPE.values():
                sub=[r for r in fam if r["subtype"]==subtype]
                result["by_subtype"][subtype]={
                    "source":_summary(sub,"source"),
                    "selected":_summary(sub,"selected"),
                    "negative":_summary(sub,"negative"),
                    "selected_vs_source":_paired(sub,"selected","source"),
                }
        else:
            result["by_difficulty"]={}
            for difficulty in ("easy","hard"):
                sub=[r for r in fam if r["difficulty"]==difficulty]
                result["by_difficulty"][difficulty]={
                    "source":_summary(sub,"source"),
                    "selected":_summary(sub,"selected"),
                    "negative":_summary(sub,"negative"),
                    "selected_vs_source":_paired(sub,"selected","source"),
                }
        family_results[family]=result

    source_correct=sum(r["predictions"]["source"]==r["correct_index"] for r in rows)
    routed_correct=sum(r["predictions"]["selected"]==r["correct_index"] for r in rows)
    target_source=sum(r["predictions"]["source"]==r["correct_index"] for r in target_rows)
    target_selected=sum(r["predictions"]["selected"]==r["correct_index"] for r in target_rows)
    target_negative=sum(r["predictions"]["negative"]==r["correct_index"] for r in target_rows)
    coverage=sum(
        r["predictions"]["source"]==r["correct_index"] or r["predictions"]["selected"]==r["correct_index"]
        for r in target_rows
    )

    integrity={
        "programs":0,
        "all_exact":True,
        "all_recovery_complete":True,
        "all_preconditions_pass":True,
        "all_result_verifications_pass":True,
    }
    for row in target_rows:
        for record in row["program_records"].values():
            integrity["programs"]+=1
            integrity["all_exact"] &= not bool(record["contains_non_exact_step"])
            integrity["all_recovery_complete"] &= bool(record["recovery_complete"])
            for step in record["steps"]:
                integrity["all_preconditions_pass"] &= step["precondition_status"]=="pass"
                integrity["all_result_verifications_pass"] &= step["verification_status"]=="pass"

    selected_pair=_paired(target_rows,"selected","source")
    specificity_pair=_paired(target_rows,"selected","negative")
    resources=scorer.counters()
    resources["wall_seconds"]=time.time()-started

    return {
        "schema_version":RUN_VERSION,
        "status":"fresh-public-family-route-structure-replication",
        "seeds":seeds,
        "cases_per_family":cases_per_family,
        "cases":len(rows),
        "target_cases":len(target_rows),
        "source_battery_sha256":sha256_json(public_spec),
        "family_results":family_results,
        "overall":{
            "source":{"correct":source_correct,"accuracy":source_correct/len(rows)},
            "family_routed":{"correct":routed_correct,"accuracy":routed_correct/len(rows)},
            "target_source":{"correct":target_source,"accuracy":target_source/len(target_rows)},
            "target_selected":{"correct":target_selected,"accuracy":target_selected/len(target_rows)},
            "target_negative":{"correct":target_negative,"accuracy":target_negative/len(target_rows)},
            "selected_vs_source":selected_pair,
            "selected_vs_negative":specificity_pair,
            "target_source_plus_selected_coverage":{"correct":coverage,"accuracy":coverage/len(target_rows)},
        },
        "decisions":{
            "target_route_replication_earned":(
                target_selected>target_source
                and selected_pair["wins"]>selected_pair["regressions"]
            ),
            "route_specificity_earned":(
                target_selected>target_negative
                and specificity_pair["wins"]>specificity_pair["regressions"]
            ),
            "abstract_flip_horizontal_replicated":False,
            "stack_easy_replication_earned":False,
        },
        "compiler_integrity":integrity,
        "resources":resources,
        "rows":rows,
        "claim_boundary":(
            "Fresh public replication of routes frozen from Run 058. Family identity and visible "
            "task structure may select a route; correctness may not. This is not protected promotion "
            "evidence and cannot establish a universal routing law."
        ),
    }


def finalize_decisions(result: dict[str, Any]) -> None:
    abstract=result["family_results"]["abstract-transformation"]["by_subtype"]["flip-horizontal"]
    stack=result["family_results"]["stack-language"]["by_difficulty"]["easy"]
    result["decisions"]["abstract_flip_horizontal_replicated"]=(
        abstract["selected"]["correct"]>abstract["source"]["correct"]
        and abstract["selected_vs_source"]["wins"]>abstract["selected_vs_source"]["regressions"]
    )
    result["decisions"]["stack_easy_replication_earned"]=(
        stack["selected"]["correct"]>stack["source"]["correct"]
        and stack["selected_vs_source"]["wins"]>stack["selected_vs_source"]["regressions"]
    )


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--helix-dir",required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--seeds",default=SEEDS_DEFAULT)
    parser.add_argument("--cases-per-family",type=int,default=CASES_PER_FAMILY_DEFAULT)
    parser.add_argument("--output",default="run-059-family-route-structure-replication.json")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(args.helix_dir),indent=2,sort_keys=True))
        return

    if not args.model_dir or not args.model_file or not args.expected_sha256:
        raise SystemExit("model-dir, model-file and expected-sha256 are required")

    model_path=Path(args.model_file)
    digest=hashlib.sha256()
    with model_path.open("rb") as handle:
        while chunk:=handle.read(8*1024*1024):
            digest.update(chunk)
    observed=digest.hexdigest()
    if observed!=args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    result=evaluate(
        args.helix_dir,args.model_dir,
        [int(x) for x in args.seeds.split(",") if x.strip()],
        args.cases_per_family,
    )
    finalize_decisions(result)
    result["actor"]={
        "model":"HuggingFaceTB/SmolLM2-360M",
        "revision":"f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        "model_file_sha256":observed,
        "model_bytes":model_path.stat().st_size,
        "weights_frozen":True,
        "dtype":"float32",
        "device":"cpu",
    }
    Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({
        "schema_version":result["schema_version"],
        "family_results":result["family_results"],
        "overall":result["overall"],
        "decisions":result["decisions"],
        "compiler_integrity":result["compiler_integrity"],
        "resources":result["resources"],
        "claim_boundary":result["claim_boundary"],
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
