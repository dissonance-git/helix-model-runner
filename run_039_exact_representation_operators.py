"""Run 039: exact representation operators.

This run instantiates a bounded subset of Helix's representation-search operator
families as deterministic visible-input-only transforms. No model-generated
intermediate state is used.

Helix API contract:
  consumes research.representation-search / research.representation-frontier
Helix operator owner:
  engine/analyze/representations/space.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from offline_reasoning_battery import build_battery, sha256_json
from offline_reasoning_content_probe import surface_task_and_choices
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import (
    build_representations,
    _score_values,
    _grid_lines,
    _grid_inline,
    _brackets,
)

RUN_VERSION = "helix-exact-representation-operators-001.0"


def strip_answer_marker(prompt: str) -> str:
    value = prompt.rstrip()
    suffix = "Answer value:"
    if not value.endswith(suffix):
        raise ValueError("prompt lacks answer marker")
    return value[:-len(suffix)].rstrip()


def compose(source: str, derived: str) -> str:
    return (
        "These are equivalent, source-recoverable representations of one reasoning task.\n"
        "The canonical source remains authoritative. Use the derived operator view only "
        "to expose structure already present in the source.\n\n"
        "CANONICAL SOURCE:\n"
        f"{source}\n\n"
        "DERIVED OPERATOR VIEW:\n"
        f"{derived}\n\n"
        "Answer value:"
    )


def relation_operators(surfaces: dict[str, str]) -> list[tuple[str, str, list[str]]]:
    stem, choices = surface_task_and_choices(surfaces["prose"])
    facts = re.findall(
        r"([A-Za-z]+) is (north|south|east|west) of ([A-Za-z]+)\.",
        stem,
    )
    query = re.search(r"Where is ([A-Za-z]+) relative to ([A-Za-z]+)\?", stem)
    if not facts or query is None:
        raise ValueError("relation parse failed")
    left, right = query.groups()
    vectors = {
        "north": "(0,+1)",
        "south": "(0,-1)",
        "east": "(+1,0)",
        "west": "(-1,0)",
    }
    coord = (
        "Exact coordinate lift. Each visible relation becomes a displacement equation.\n"
        + "\n".join(
            f"{a} = {b} + {vectors[d]}" for a,d,b in facts
        )
        + f"\nTarget: direction of {left} - {right}."
    )
    graph = (
        "Topology/incidence re-encoding as a labeled directed graph.\n"
        + "\n".join(
            f"edge({b} -> {a}, label={d}); reverse edge({a} -> {b}, label="
            + {"north":"south","south":"north","east":"west","west":"east"}[d]
            + ")"
            for a,d,b in facts
        )
        + f"\nTarget nodes: source={right}, destination={left}. Determine their net compass relation."
    )
    return [
        ("coordinate_lift", coord, choices),
        ("topology_incidence_reencoding", graph, choices),
    ]


def grid_signature(value: str) -> str:
    rows = value.split("/")
    cols = ["".join(row[c] for row in rows) for c in range(3)]
    return f"rows={rows}; columns={cols}"


def grid_cells(value: str) -> str:
    rows = value.split("/")
    return " ".join(
        f"r{r+1}c{c+1}={rows[r][c]}"
        for r in range(3) for c in range(3)
    )


def grid_operators(surfaces: dict[str, str]) -> list[tuple[str, str, list[str]]]:
    stem, choices = surface_task_and_choices(surfaces["compact"])
    grids = re.findall(r"[0-2]{3}/[0-2]{3}/[0-2]{3}", stem)
    if len(grids) < 3 or len(grids) % 2 != 1:
        raise ValueError("grid parse failed")
    query = grids[-1]
    demos = list(zip(grids[:-1:2], grids[1:-1:2], strict=True))
    exact = (
        "Finite exact re-encoding. Every grid is represented by its nine coordinate/value facts.\n"
        + "\n".join(
            f"example {i}: input {{{grid_cells(a)}}} -> output {{{grid_cells(b)}}}"
            for i,(a,b) in enumerate(demos, start=1)
        )
        + f"\nquery input: {{{grid_cells(query)}}}. Apply the same exact transformation."
    )
    exact_choices = [grid_cells(choice) for choice in choices]

    factors = (
        "Factor/product decomposition. Preserve each grid exactly as separate row and column factors.\n"
        + "\n".join(
            f"example {i}: input {grid_signature(a)} -> output {grid_signature(b)}"
            for i,(a,b) in enumerate(demos, start=1)
        )
        + f"\nquery input: {grid_signature(query)}. Apply the same transformation."
    )
    factor_choices = [grid_signature(choice) for choice in choices]
    return [
        ("finite_exact_reencoding", exact, exact_choices),
        ("factor_product_decomposition", factors, factor_choices),
    ]


def bit_tuple(a: str, b: str, out: str | None = None) -> str:
    names = "pqrst"
    return " | ".join(
        f"{name}:({av},{bv})->{('?' if out is None else out[i])}"
        for i,(name,av,bv) in enumerate(zip(names,a,b,strict=True))
    )


def matrix_operators(surfaces: dict[str, str]) -> list[tuple[str, str, list[str]]]:
    stem, choices = surface_task_and_choices(surfaces["bits"])
    demos = re.findall(r"([01]{5}) \? ([01]{5}) = ([01]{5})", stem)
    queries = re.findall(r"([01]{5}) \? ([01]{5}) =", stem)
    if not demos or not queries:
        raise ValueError("matrix parse failed")
    qa,qb = queries[-1]

    exact = (
        "Finite exact Boolean re-encoding. Treat each bit position as one local input/output constraint.\n"
        + "\n".join(bit_tuple(a,b,c) for a,b,c in demos)
        + f"\nquery local inputs: {bit_tuple(qa,qb,None)}. Infer the one fixed local rule."
    )
    exact_choices = [" ".join(choice) for choice in choices]

    observations: dict[str, list[str]] = {"00":[],"01":[],"10":[],"11":[]}
    for a,b,c in demos:
        for i in range(5):
            observations[a[i]+b[i]].append(c[i])
    quotient_rows = []
    for pair, outs in observations.items():
        quotient_rows.append(
            f"equivalence class input={pair}: observed outputs={outs}"
        )
    query_pairs = [qa[i]+qb[i] for i in range(5)]
    quotient = (
        "Reversible quotient by repeated local input pattern. Examples are grouped only by "
        "their visible two-bit local input while retaining all observed outputs.\n"
        + "\n".join(quotient_rows)
        + f"\nquery local input classes by position p..t: {query_pairs}. "
        "Reconstruct the output bits from the same fixed rule."
    )
    return [
        ("finite_exact_reencoding", exact, exact_choices),
        ("reversible_quotient", quotient, exact_choices),
    ]


def planning_parse(surfaces: dict[str, str]) -> tuple[list[str], int, list[str]]:
    stem, choices = surface_task_and_choices(surfaces["grid"])
    match = re.search(r"Grid rows top-to-bottom: ([SG#.]{4}(?:/[SG#.]{4}){3})\.", stem)
    length = re.search(r"Choose the ([0-9]+)-move route", stem)
    if match is None or length is None:
        raise ValueError("planning parse failed")
    return match.group(1).split("/"), int(length.group(1)), choices


def planning_operators(surfaces: dict[str, str]) -> list[tuple[str, str, list[str]]]:
    rows, moves, choices = planning_parse(surfaces)
    n = len(rows)
    blocked=set()
    start=goal=None
    for y,row in enumerate(rows):
        for x,ch in enumerate(row):
            if ch=="#": blocked.add((x,y))
            elif ch=="S": start=(x,y)
            elif ch=="G": goal=(x,y)
    directions={"U":(0,-1),"R":(1,0),"D":(0,1),"L":(-1,0)}
    edges=[]
    for y in range(n):
        for x in range(n):
            if (x,y) in blocked: continue
            for label,(dx,dy) in directions.items():
                q=(x+dx,y+dy)
                if 0<=q[0]<n and 0<=q[1]<n and q not in blocked:
                    edges.append(f"({x},{y}) -{label}-> ({q[0]},{q[1]})")
    graph=(
        "Topology/incidence re-encoding. The board is the exact labeled legal-move graph.\n"
        f"start={start}; goal={goal}; required moves={moves}.\n"
        + "\n".join(edges)
    )
    coord=(
        "Coordinate lift. Legal state is (x,y); U=(0,-1), R=(+1,0), D=(0,+1), L=(-1,0).\n"
        f"start={start}; goal={goal}; blocked={sorted(blocked)}; board size={n}x{n}; "
        f"required moves={moves}. Apply candidate move vectors without entering blocked/outside cells."
    )
    rendered_choices=[" ".join(choice) for choice in choices]
    return [
        ("topology_incidence_reencoding", graph, rendered_choices),
        ("coordinate_lift", coord, rendered_choices),
    ]


PAIR={"ka":"ti","mo":"re","su":"va"}
CLOSE_TO_OPEN={v:k for k,v in PAIR.items()}
BRACKET={"ka":"(","ti":")","mo":"[","re":"]","su":"{","va":"}"}


def visible_stack_state(prefix: str) -> list[str]:
    stack=[]
    for token in prefix.split():
        if token in PAIR:
            stack.append(token)
        elif token in CLOSE_TO_OPEN:
            if not stack or stack[-1] != CLOSE_TO_OPEN[token]:
                return ["INVALID_VISIBLE_PREFIX"]
            stack.pop()
        else:
            raise ValueError(f"unknown token {token}")
    return stack


def stack_operators(surfaces: dict[str, str]) -> list[tuple[str, str, list[str]]]:
    stem, choices = surface_task_and_choices(surfaces["compact"])
    match = re.fullmatch(r"pairs ka/ti mo/re su/va; LIFO; prefix=(.*); completion=\?\nAnswer value:", stem)
    if match is None:
        # surface_task_and_choices keeps Answer value: in task for this battery
        match = re.search(r"prefix=(.*); completion=\?", stem)
    if match is None:
        raise ValueError("stack parse failed")
    prefix=match.group(1).strip()
    brackets="".join(BRACKET[t] for t in prefix.split())
    bracket_choices=["".join(BRACKET[t] for t in choice.split()) for choice in choices]
    exact=(
        "Finite exact re-encoding into ordinary bracket syntax. "
        "ka/ti=(), mo/re=[], su/va={}.\n"
        f"visible prefix={brackets}. Choose the completion that yields a balanced bracket sequence."
    )
    stack=visible_stack_state(prefix)
    lifted=(
        "Coordinate/state lift for a pushdown process. The visible prefix has already been "
        "processed by the explicit LIFO pair table.\n"
        f"remaining opener stack bottom->top={stack}; required closer for the current top, if any, "
        f"is {PAIR.get(stack[-1],'none') if stack and stack[-1] != 'INVALID_VISIBLE_PREFIX' else 'invalid'}. "
        "Continue applying the same LIFO rule to a candidate completion until the stack is empty."
    )
    return [
        ("finite_exact_reencoding", exact, bracket_choices),
        ("coordinate_lift", lifted, bracket_choices),
    ]


BUILDERS={
    "relational-composition":relation_operators,
    "abstract-transformation":grid_operators,
    "relational-matrix":matrix_operators,
    "grounded-planning":planning_operators,
    "stack-language":stack_operators,
}


def operators_for_case(family: str, surfaces: dict[str,str]) -> list[tuple[str,str,list[str]]]:
    return BUILDERS[family](surfaces)


def summarize(rows: list[dict[str,Any]], key: str) -> dict[str,Any]:
    hits=0
    by_family: dict[str,list[bool]]=defaultdict(list)
    by_seed: dict[int,list[bool]]=defaultdict(list)
    for row in rows:
        ok=int(row[key])==int(row["correct_index"])
        hits+=int(ok)
        by_family[row["family"]].append(ok)
        by_seed[int(row["seed"])].append(ok)
    return {
        "cases":len(rows),
        "correct":hits,
        "accuracy":hits/len(rows),
        "by_family":{k:sum(v)/len(v) for k,v in sorted(by_family.items())},
        "by_seed":{str(k):sum(v)/len(v) for k,v in sorted(by_seed.items())},
    }


def evaluate(model_dir: str, seeds: list[int], cases_per_family: int) -> dict[str,Any]:
    scorer=AttackScorer(model_dir)
    scorer.reset_counters()
    started=time.time()
    rows=[]
    spec=[]

    for seed in seeds:
        cases=build_battery(seed,cases_per_family)
        seed_spec=[]
        for case in cases:
            canonical=build_representations(case["family"],case["surfaces"])
            source_prompt, source_choices=canonical["layout"]
            source_stem=strip_answer_marker(source_prompt)
            ops=operators_for_case(case["family"],case["surfaces"])
            if len(ops)!=2:
                raise AssertionError("expected two exact operators")
            op1_id,op1_stem,op1_choices=ops[0]
            op2_id,op2_stem,op2_choices=ops[1]

            prompts={
                "source":(source_prompt,source_choices),
                "op1":(op1_stem+"\nAnswer value:",op1_choices),
                "op2":(op2_stem+"\nAnswer value:",op2_choices),
                "source_op1":(compose(source_stem,op1_stem),op1_choices),
                "source_op2":(compose(source_stem,op2_stem),op2_choices),
            }
            scores={
                name:_score_values(scorer,prompt,choices)
                for name,(prompt,choices) in prompts.items()
            }
            preds={name:int(v["prediction"]) for name,v in scores.items()}
            correct=int(case["correct_index"])
            source_correct=preds["source"]==correct
            op_only=[
                name for name in ("op1","op2")
                if preds[name]==correct and not source_correct
            ]
            composed_only=[
                name for name in ("source_op1","source_op2")
                if preds[name]==correct
                and preds["source"]!=correct
                and preds[name.replace("source_","")]!=correct
            ]
            rows.append({
                "uid":f"{seed}:{case['case_id']}",
                "seed":seed,
                "case_id":case["case_id"],
                "family":case["family"],
                "difficulty":case["difficulty"],
                "correct_index":correct,
                "operator_ids":[op1_id,op2_id],
                **{f"{name}_prediction":pred for name,pred in preds.items()},
                "operator_only_correct":op_only,
                "strict_source_operator_composition_correct":composed_only,
                "operator_predictions_disagree":preds["op1"]!=preds["op2"],
                "prompt_sha256":{
                    name:hashlib.sha256(prompt.encode()).hexdigest()
                    for name,(prompt,_choices) in prompts.items()
                },
            })
            seed_spec.append({
                "case_id":case["case_id"],
                "family":case["family"],
                "difficulty":case["difficulty"],
                "correct_index":correct,
                "source_surface_sha256":{
                    name:hashlib.sha256(prompt.encode()).hexdigest()
                    for name,prompt in case["surfaces"].items()
                },
                "operator_ids":[op1_id,op2_id],
            })
        spec.append({"seed":seed,"cases":seed_spec})

    arm_keys=["source","op1","op2","source_op1","source_op2"]
    arms={name:summarize(rows,f"{name}_prediction") for name in arm_keys}
    source_cov=sum(r["source_prediction"]==r["correct_index"] for r in rows)
    op_cov=sum(
        r["source_prediction"]==r["correct_index"]
        or r["op1_prediction"]==r["correct_index"]
        or r["op2_prediction"]==r["correct_index"]
        for r in rows
    )
    all_cov=sum(
        any(r[f"{name}_prediction"]==r["correct_index"] for name in arm_keys)
        for r in rows
    )
    resources=scorer.counters()
    resources["wall_seconds"]=time.time()-started

    return {
        "schema_version":RUN_VERSION,
        "status":"fresh-public-exact-operator-calibration",
        "api_contract":{
            "helix_api_commit":"8640aaf4",
            "consumes":["research.representation-search","research.representation-frontier"],
        },
        "helix_operator_owner":"engine/analyze/representations/space.py",
        "seeds":seeds,
        "cases_per_family":cases_per_family,
        "cases":len(rows),
        "source_battery_sha256":sha256_json(spec),
        "arms":arms,
        "coverage":{
            "source_correct":source_cov,
            "source_accuracy":source_cov/len(rows),
            "source_plus_exact_operators_correct":op_cov,
            "source_plus_exact_operators_coverage":op_cov/len(rows),
            "source_plus_operators_plus_compositions_correct":all_cov,
            "source_plus_operators_plus_compositions_coverage":all_cov/len(rows),
            "operator_only_correct_cases":sum(bool(r["operator_only_correct"]) for r in rows),
            "strict_source_operator_composition_cases":sum(bool(r["strict_source_operator_composition_correct"]) for r in rows),
            "operator_disagreement_cases":sum(r["operator_predictions_disagree"] for r in rows),
        },
        "resources":resources,
        "rows":rows,
        "claim_boundary":(
            "Fresh public procedural diagnostic. Exact operators are deterministic visible-input-only "
            "substrate transforms. Any case where an operator performs substantial symbolic computation "
            "is system/substrate capability, not actor-only reasoning. No protected promotion or frontier claim."
        ),
    }


def self_test(seed: int = 20261019, cases_per_family: int = 4) -> dict[str, Any]:
    cases = build_battery(seed, cases_per_family)
    counts: dict[str, int] = defaultdict(int)
    digests = []
    for case in cases:
        # Critical contract: operator builder receives family + visible surfaces only.
        ops = operators_for_case(case["family"], dict(case["surfaces"]))
        if len(ops) != 2:
            raise AssertionError((case["case_id"], len(ops)))
        canonical = build_representations(case["family"], case["surfaces"])
        source = strip_answer_marker(canonical["layout"][0])
        for operator_id, stem, choices in ops:
            if not stem.strip() or len(choices) != 4:
                raise AssertionError((case["case_id"], operator_id))
            composite = compose(source, stem)
            if not composite.endswith("Answer value:"):
                raise AssertionError((case["case_id"], operator_id, "composite"))
            counts[f"{case['family']}/{operator_id}"] += 1
            digests.append(hashlib.sha256(
                json.dumps({
                    "family": case["family"],
                    "operator_id": operator_id,
                    "stem": stem,
                    "choices": choices,
                }, sort_keys=True).encode()
            ).hexdigest())
    return {
        "cases": len(cases),
        "operator_views": len(digests),
        "counts": dict(sorted(counts.items())),
        "transform_digest": hashlib.sha256("|".join(digests).encode()).hexdigest(),
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--seeds",default="20261019,20261020,20261021")
    parser.add_argument("--cases-per-family",type=int,default=4)
    parser.add_argument("--output",default="run-039-exact-operators.json")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    if not args.model_dir or not args.model_file or not args.expected_sha256:
        raise SystemExit("model-dir, model-file and expected-sha256 are required")

    model_path=Path(args.model_file)
    observed=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if observed!=args.expected_sha256:
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
        "coverage":result["coverage"],
        "resources":result["resources"],
        "claim_boundary":result["claim_boundary"],
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
