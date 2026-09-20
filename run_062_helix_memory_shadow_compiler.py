"""Run 062: Helix-memory shadow compiler, public actor side.

The public runner receives only learner-visible current Helix research consequences.
It has no retained importance labels. It freezes predictions for raw and two
actual guarded Helix compiler programs. Private judging happens outside this repo.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import _score_values

RUN_VERSION = "helix-memory-shadow-compiler-public-001.0"
OBLIGATION = "preserve-visible-research-consequence-and-tier-alternatives"
PRIVATE_KEYS = {
    "importance_status","impact_tier","article_interest_tier","correctness_status",
    "novelty_status","verification_status","publication_status","source_routes",
    "provenance","target_id","report_path","object_id","required_tier",
    "stored_importance_status",
}
PROGRAMS = {
    "reproject": ("reproject", "compose"),
    "factor": ("reproject", "factor", "compose"),
}


def _load_compiler(helix_dir: str):
    root = str(Path(helix_dir).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from engine.ir.passes import OperatorSpec, TransformationRegistry, execute_transform_program
    return OperatorSpec, TransformationRegistry, execute_transform_program


def _definitions(suite: dict[str, Any]) -> str:
    defs = suite["tier_definitions"]
    return "\n".join(
        f"- {name}: {defs[name]}" for name in suite["choices"]
    )


def _raw_prompt(case: dict[str, Any], suite: dict[str, Any]) -> str:
    visible = case["visible_consequence"]
    return (
        "Classify the retained research importance tier from learner-visible evidence only.\n"
        "Tier definitions:\n" + _definitions(suite) + "\n\n"
        "RESEARCH CONSEQUENCE:\n"
        + json.dumps(visible, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n\nReturn exactly one tier.\nAnswer value:"
    )


def _source_state(case: dict[str, Any], suite: dict[str, Any]) -> dict[str, Any]:
    visible = json.loads(json.dumps(case["visible_consequence"]))
    serialized = json.dumps(visible, ensure_ascii=False, sort_keys=True)
    lowered = serialized.casefold()
    for key in PRIVATE_KEYS:
        if f'"{key.casefold()}"' in lowered:
            raise AssertionError(f"private field leaked into public suite: {key}")
    return {
        "schema": "helix-memory-visible-state-001.0",
        "case_id": case["case_id"],
        "visible_consequence": visible,
        "source_prompt": _raw_prompt(case, suite),
        "source_choices": list(suite["choices"]),
        "tier_definitions": dict(suite["tier_definitions"]),
        "views": [],
    }


def _precondition(source: Any) -> dict[str, Any]:
    ok = (
        isinstance(source, dict)
        and isinstance(source.get("visible_consequence"), dict)
        and isinstance(source.get("source_prompt"), str)
        and len(source.get("source_choices") or []) == 3
        and isinstance(source.get("views"), list)
        and not any(key in source.get("visible_consequence", {}) for key in PRIVATE_KEYS)
    )
    return {
        "status": "pass" if ok else "fail",
        "applicable": ok,
        "exact": True,
        "verified_preconditions": ["learner-visible-consequence-only"] if ok else [],
        "reason": None if ok else "invalid visible research consequence state",
    }


def _verify(source: Any, target: Any) -> dict[str, Any]:
    ok = (
        isinstance(source, dict)
        and isinstance(target, dict)
        and target.get("case_id") == source.get("case_id")
        and target.get("visible_consequence") == source.get("visible_consequence")
        and target.get("source_prompt") == source.get("source_prompt")
        and target.get("source_choices") == source.get("source_choices")
        and target.get("tier_definitions") == source.get("tier_definitions")
        and not any(key in target.get("visible_consequence", {}) for key in PRIVATE_KEYS)
    )
    if "final_prompt" in target:
        ok = ok and target["final_prompt"].endswith("Answer value:")
        ok = ok and target.get("final_choices") == source.get("source_choices")
    return {
        "status": "pass" if ok else "fail",
        "exact": bool(ok),
        "source_recovery_preserved": bool(ok),
    }


def _append_view(source: dict[str, Any], operator_id: str, text: str):
    target = dict(source)
    target["views"] = [dict(row) for row in source["views"]]
    target["views"].append({"operator_id": operator_id, "text": text})
    return target, {
        "introduced_information": [f"deterministic learner-visible view: {operator_id}"],
        "discarded_information": [],
        "expected_observable": "actor tier-support change under exact structural re-expression",
        "known_risk": ["surface distribution shift"],
        "inverse_or_recovery_route": "visible_consequence and source_prompt retained verbatim",
        "peak_intermediate_size": len(text),
    }


def _reproject_impl(source: dict[str, Any]):
    v = source["visible_consequence"]
    lines = [
        "CANONICAL RESEARCH CONSEQUENCE",
        f"Title: {v.get('title') or ''}",
        f"Exact result: {v.get('exact_result') or ''}",
        f"Mechanism: {v.get('mechanism') or ''}",
        f"Contribution: {v.get('contribution') or ''}",
        f"Result type: {v.get('result_type') or ''}",
        f"Mathematical status: {v.get('mathematical_status') or ''}",
        "Limitations:",
        *[f"- {x}" for x in v.get("limitations") or []],
        "Next actions:",
        *[f"- {x}" for x in v.get("next_actions") or []],
    ]
    return _append_view(source, "canonical-field-reprojection", "\n".join(lines))


def _factor_impl(source: dict[str, Any]):
    v = source["visible_consequence"]
    lines = [
        "FACTORED RESEARCH VIEW",
        "[CONSEQUENCE]",
        str(v.get("exact_result") or v.get("title") or ""),
        "[MECHANISM]",
        str(v.get("mechanism") or ""),
        "[RESEARCH LEVERAGE]",
        str(v.get("contribution") or ""),
        *[f"- next: {x}" for x in v.get("next_actions") or []],
        "[BOUNDARY / RESIDUAL]",
        *[f"- {x}" for x in v.get("limitations") or []],
        "[FORM]",
        f"type={v.get('result_type') or ''}; status={v.get('mathematical_status') or ''}",
    ]
    return _append_view(source, "evidence-boundary-leverage-factorization", "\n".join(lines))


def _compose_impl(source: dict[str, Any]):
    target = dict(source)
    target["views"] = [dict(row) for row in source["views"]]
    blocks = [
        "Classify the retained research importance tier from learner-visible evidence only.",
        "The canonical source is authoritative. Derived views only reorganize the same visible content.",
        "Tier definitions:",
        *[
            f"- {name}: {source['tier_definitions'][name]}"
            for name in source["source_choices"]
        ],
        "",
        "CANONICAL SOURCE:",
        json.dumps(source["visible_consequence"], ensure_ascii=False, sort_keys=True, indent=2),
    ]
    for i, view in enumerate(source["views"], 1):
        blocks.extend(["", f"DERIVED VIEW {i} ({view['operator_id']}):", view["text"]])
    blocks.extend(["", "Return exactly one tier.", "Answer value:"])
    target["final_prompt"] = "\n".join(blocks)
    target["final_choices"] = list(source["source_choices"])
    return target, {
        "introduced_information": ["aligned composition of source and deterministic derived views"],
        "discarded_information": [],
        "expected_observable": "tier-support shift after obligation-preserving representation composition",
        "known_risk": ["longer prompt", "surface distribution shift"],
        "inverse_or_recovery_route": "drop derived blocks to recover visible_consequence exactly",
        "peak_intermediate_size": len(target["final_prompt"]),
    }


def _registry(helix_dir: str):
    OperatorSpec, TransformationRegistry, execute_transform_program = _load_compiler(helix_dir)
    reg = TransformationRegistry()
    common = dict(
        source_types=("research-consequence",),
        target_type="research-consequence",
        preconditions=("learner-visible-consequence-only",),
        preserved_obligations=(OBLIGATION,),
        state="exact",
        verifier=_verify,
        precondition_verifier=_precondition,
        certificate_requirements=("source recovery preserved", "tier alternatives unchanged"),
    )
    reg.register(OperatorSpec(
        name="reproject", basis_operator="REPROJECT",
        implementation=_reproject_impl,
        cheapest_falsifier="remove one visible field and require source-recovery verifier rejection",
        **common,
    ))
    reg.register(OperatorSpec(
        name="factor", basis_operator="FACTOR",
        implementation=_factor_impl,
        cheapest_falsifier="alter one limitation and require exact verifier rejection",
        **common,
    ))
    reg.register(OperatorSpec(
        name="compose", basis_operator="COMPOSE",
        implementation=_compose_impl,
        cheapest_falsifier="change a tier alternative and require exact verifier rejection",
        **common,
    ))
    return reg, execute_transform_program


def _execute(helix_dir: str, state: dict[str, Any], program: tuple[str, ...], arm: str):
    reg, execute_transform_program = _registry(helix_dir)
    target, record = execute_transform_program(
        reg,
        program,
        state,
        source_id="HELIX-MEMORY-" + hashlib.sha256(state["case_id"].encode()).hexdigest()[:20],
        source_type="research-consequence",
        required_obligations=(OBLIGATION,),
        provenance={"run_id":"062-helix-memory-shadow-compiler","case_id":state["case_id"],"arm":arm},
    )
    return target, record


def _compact(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "program_id": record["program_id"],
        "operators": record["operators"],
        "basis_operators": record["basis_operators"],
        "contains_non_exact_step": record["contains_non_exact_step"],
        "recovery_complete": record.get("recovery_route") is not None,
        "steps":[{
            "operator": row["operator"],
            "basis_operator": row["basis_operator"],
            "precondition_status": (row.get("precondition_verification") or {}).get("status"),
            "verification_status": (row.get("verification_state") or {}).get("status"),
            "discarded_information": row.get("discarded_information") or [],
            "inverse_or_recovery_route": row.get("inverse_or_recovery_route"),
        } for row in record["steps"]],
    }


def self_test(helix_dir: str, suite_path: str) -> dict[str, Any]:
    suite=json.loads(Path(suite_path).read_text(encoding="utf-8"))
    if any("required_tier" in json.dumps(case) for case in suite["cases"]):
        raise AssertionError("judge label leaked into public suite")
    rows=[]
    for case in suite["cases"][:3]:
        state=_source_state(case,suite)
        arms={}
        for arm,program in PROGRAMS.items():
            target,record=_execute(helix_dir,state,program,arm)
            arms[arm]={
                "basis":record["basis_operators"],
                "exact":not record["contains_non_exact_step"],
                "recovery_complete":record.get("recovery_route") is not None,
                "prompt_sha256":hashlib.sha256(target["final_prompt"].encode()).hexdigest(),
            }
        rows.append({"case_id":case["case_id"],"arms":arms})
    return {
        "schema_version":RUN_VERSION,
        "status":"self-test-pass",
        "cases":len(rows),
        "actual_execution_owner":"engine.ir.passes.execute_transform_program",
        "judge_visible_to_actor":False,
        "compiler_integrity_precheck":True,
        "rows":rows,
    }


def evaluate(helix_dir: str, model_dir: str, suite_path: str) -> dict[str, Any]:
    suite=json.loads(Path(suite_path).read_text(encoding="utf-8"))
    scorer=AttackScorer(model_dir)
    scorer.reset_counters()
    started=time.time()
    rows=[]
    integrity={
        "programs":0,"all_exact":True,"all_recovery_complete":True,
        "all_preconditions_pass":True,"all_result_verifications_pass":True,
    }
    for case in suite["cases"]:
        state=_source_state(case,suite)
        raw=_score_values(scorer,state["source_prompt"],state["source_choices"])
        predictions={"raw":int(raw["prediction"])}
        scores={"raw":[float(x) for x in raw["scores"]]}
        margins={"raw":float(raw["margin"])}
        records={}
        hashes={"raw":hashlib.sha256(state["source_prompt"].encode()).hexdigest()}
        for arm,program in PROGRAMS.items():
            target,record=_execute(helix_dir,state,program,arm)
            scored=_score_values(scorer,target["final_prompt"],target["final_choices"])
            predictions[arm]=int(scored["prediction"])
            scores[arm]=[float(x) for x in scored["scores"]]
            margins[arm]=float(scored["margin"])
            compact=_compact(record)
            records[arm]=compact
            hashes[arm]=hashlib.sha256(target["final_prompt"].encode()).hexdigest()
            integrity["programs"]+=1
            integrity["all_exact"] &= not compact["contains_non_exact_step"]
            integrity["all_recovery_complete"] &= compact["recovery_complete"]
            for step in compact["steps"]:
                integrity["all_preconditions_pass"] &= step["precondition_status"]=="pass"
                integrity["all_result_verifications_pass"] &= step["verification_status"]=="pass"
        rows.append({
            "case_id":case["case_id"],
            "predictions":predictions,
            "prediction_values":{arm:suite["choices"][idx] for arm,idx in predictions.items()},
            "scores":scores,
            "margins":margins,
            "program_records":records,
            "prompt_sha256":hashes,
        })
    resources=scorer.counters()
    resources["wall_seconds"]=time.time()-started
    return {
        "schema_version":RUN_VERSION,
        "status":"public-predictions-frozen-awaiting-private-judge",
        "source":{
            "helix_commit":suite["source_commit"],
            "research_generation_sha256":suite["research_generation_sha256"],
            "case_count":len(suite["cases"]),
        },
        "actor":{
            "model":"HuggingFaceTB/SmolLM2-360M",
            "revision":"f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
            "weights_frozen":True,
        },
        "choices":suite["choices"],
        "judge_visible_to_actor":False,
        "compiler_integrity":integrity,
        "arm_prediction_counts":{
            arm:dict(Counter(row["prediction_values"][arm] for row in rows))
            for arm in ("raw","reproject","factor")
        },
        "resources":resources,
        "rows":rows,
        "claim_boundary":"Public actor-side shadow predictions only. No labels or correctness are present in this artifact; private Helix Model judging is required.",
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--helix-dir",required=True)
    ap.add_argument("--suite",default="run-062-helix-memory-shadow-suite.json")
    ap.add_argument("--model-dir")
    ap.add_argument("--model-file")
    ap.add_argument("--expected-sha256")
    ap.add_argument("--output",default="run-062-helix-memory-shadow-compiler.json")
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.helix_dir,args.suite),indent=2,sort_keys=True))
        return
    if not args.model_dir or not args.model_file or not args.expected_sha256:
        raise SystemExit("model-dir, model-file and expected-sha256 are required")
    h=hashlib.sha256()
    p=Path(args.model_file)
    with p.open("rb") as f:
        while chunk:=f.read(8*1024*1024): h.update(chunk)
    observed=h.hexdigest()
    if observed!=args.expected_sha256: raise SystemExit(f"model digest mismatch: {observed}")
    result=evaluate(args.helix_dir,args.model_dir,args.suite)
    result["actor"].update({"model_file_sha256":observed,"model_bytes":p.stat().st_size,"dtype":"float32","device":"cpu"})
    Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({
        "schema_version":result["schema_version"],
        "status":result["status"],
        "source":result["source"],
        "arm_prediction_counts":result["arm_prediction_counts"],
        "compiler_integrity":result["compiler_integrity"],
        "resources":result["resources"],
        "judge_visible_to_actor":result["judge_visible_to_actor"],
        "claim_boundary":result["claim_boundary"],
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
