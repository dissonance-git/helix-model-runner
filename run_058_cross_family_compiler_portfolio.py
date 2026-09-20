"""Run 058: actual Helix compiler across five reasoning geometries.

The frozen 360M base actor is held fixed.  Every transformation receives only
learner-visible task state.  The current Helix engine.ir.passes owner executes
all programs; this file supplies bounded domain adapters for the existing fresh
procedural battery.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from offline_reasoning_battery import build_battery, sha256_json
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import build_representations, _score_values
from run_039_exact_representation_operators import operators_for_case, strip_answer_marker

RUN_VERSION = "helix-cross-family-compiler-portfolio-001.0"
OBLIGATION = "preserve-visible-task-and-aligned-answer-alternatives"
BASIS_BY_OPERATOR = {
    "coordinate_lift": "LIFT",
    "finite_exact_reencoding": "REPROJECT",
    "reversible_quotient": "PROJECT",
    "topology_incidence_reencoding": "REPROJECT",
    "factor_product_decomposition": "FACTOR",
}


def _load_compiler(helix_dir: str):
    root = str(Path(helix_dir).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from engine.ir.passes import OperatorSpec, TransformationRegistry, execute_transform_program
    return OperatorSpec, TransformationRegistry, execute_transform_program


def _source_state(family: str, case_id: str, surfaces: dict[str, str]) -> dict[str, Any]:
    # Deliberately no correct_index, label, judge result or generator latent state.
    canonical = build_representations(family, surfaces)
    source_prompt, source_choices = canonical["layout"]
    return {
        "schema": "run058-visible-task-state-001.0",
        "family": family,
        "case_id": case_id,
        "surfaces": dict(surfaces),
        "source_prompt": source_prompt,
        "source_stem": strip_answer_marker(source_prompt),
        "source_choices": list(source_choices),
        "views": [],
    }


def _precondition(source: Any) -> dict[str, Any]:
    ok = (
        isinstance(source, dict)
        and "correct_index" not in source
        and isinstance(source.get("source_prompt"), str)
        and len(source.get("source_choices") or []) == 4
        and isinstance(source.get("views"), list)
    )
    return {
        "status": "pass" if ok else "fail",
        "applicable": ok,
        "exact": True,
        "verified_preconditions": ["visible-task-only"] if ok else [],
        "reason": None if ok else "state is not a four-choice visible-only task",
    }


def _verify(source: Any, target: Any) -> dict[str, Any]:
    ok = (
        isinstance(source, dict)
        and isinstance(target, dict)
        and "correct_index" not in target
        and target.get("source_prompt") == source.get("source_prompt")
        and target.get("source_choices") == source.get("source_choices")
        and all(len(row.get("choices") or []) == 4 for row in target.get("views") or [])
    )
    if "final_choices" in target:
        ok = ok and len(target["final_choices"]) == 4 and str(target.get("final_prompt") or "").endswith("Answer value:")
    return {
        "status": "pass" if ok else "fail",
        "exact": bool(ok),
        "source_recovery_preserved": bool(ok),
    }


def _append_view_impl(operator_id: str, stem: str, choices: list[str]):
    def implementation(source: dict[str, Any]):
        target = dict(source)
        target["views"] = [dict(row) for row in source["views"]]
        target["views"].append({
            "operator_id": operator_id,
            "stem": stem,
            "choices": list(choices),
        })
        return target, {
            "introduced_information": [f"exact visible-input view: {operator_id}"],
            "discarded_information": [],
            "expected_observable": "actor answer-support change under exact derived coordinates",
            "known_risk": ["surface distribution shift"],
            "inverse_or_recovery_route": "source_prompt and source_choices retained verbatim",
            "peak_intermediate_size": len(stem) + sum(len(x) for x in choices),
        }
    implementation.__name__ = "append_" + operator_id.replace("-", "_")
    return implementation


def _compose_impl(source: dict[str, Any]):
    target = dict(source)
    target["views"] = [dict(row) for row in source["views"]]
    blocks = [
        "The following are exact, source-recoverable views of one reasoning task.",
        "The canonical source remains authoritative. Derived views expose only structure already visible in it.",
        "",
        "CANONICAL SOURCE:",
        source["source_stem"],
    ]
    for index, view in enumerate(source["views"], 1):
        blocks.extend([
            "",
            f"DERIVED VIEW {index} ({view['operator_id']}):",
            view["stem"],
        ])
    blocks.extend([
        "",
        "The answer candidates below align the same candidate across every displayed view.",
        "Answer value:",
    ])
    final_choices = []
    for i, source_choice in enumerate(source["source_choices"]):
        parts = [f"source=[{source_choice}]"]
        for j, view in enumerate(source["views"], 1):
            parts.append(f"view{j}=[{view['choices'][i]}]")
        final_choices.append("; ".join(parts))
    target["final_prompt"] = "\n".join(blocks)
    target["final_choices"] = final_choices
    return target, {
        "introduced_information": ["aligned composition of source and exact derived views"],
        "discarded_information": [],
        "expected_observable": "route-order-sensitive actor support over aligned candidate encodings",
        "known_risk": ["longer candidate strings", "order-dependent language-model surface prior"],
        "inverse_or_recovery_route": "drop derived view blocks to recover canonical source exactly",
        "peak_intermediate_size": len(target["final_prompt"]) + sum(len(x) for x in final_choices),
    }


def _registry_for_case(helix_dir: str, family: str, surfaces: dict[str, str]):
    OperatorSpec, TransformationRegistry, execute_transform_program = _load_compiler(helix_dir)
    rows = operators_for_case(family, surfaces)
    if len(rows) != 2:
        raise AssertionError((family, len(rows)))
    registry = TransformationRegistry()

    for name, row in zip(("first", "second"), rows, strict=True):
        operator_id, stem, choices = row
        registry.register(OperatorSpec(
            name=name,
            source_types=("reasoning-task",),
            target_type="reasoning-task",
            preconditions=("visible-task-only",),
            preserved_obligations=(OBLIGATION,),
            state="exact",
            basis_operator=BASIS_BY_OPERATOR[operator_id],
            implementation=_append_view_impl(operator_id, stem, list(choices)),
            verifier=_verify,
            precondition_verifier=_precondition,
            certificate_requirements=("source recovery preserved", "four aligned alternatives"),
            cheapest_falsifier="drop or reorder one candidate and require exact verifier rejection",
        ))

    registry.register(OperatorSpec(
        name="compose",
        source_types=("reasoning-task",),
        target_type="reasoning-task",
        preconditions=("visible-task-only",),
        preserved_obligations=(OBLIGATION,),
        state="exact",
        basis_operator="COMPOSE",
        implementation=_compose_impl,
        verifier=_verify,
        precondition_verifier=_precondition,
        certificate_requirements=("source recovery preserved", "candidate alignment preserved"),
        cheapest_falsifier="mutate one aligned candidate and require exact verifier rejection",
    ))
    return registry, execute_transform_program, rows


def _execute(helix_dir: str, state: dict[str, Any], program: tuple[str, ...], provenance: dict[str, Any]):
    registry, execute_transform_program, _rows = _registry_for_case(helix_dir, state["family"], state["surfaces"])
    target, record = execute_transform_program(
        registry,
        program,
        state,
        source_id="TASK-" + hashlib.sha256(
            json.dumps({
                "family": state["family"],
                "case_id": state["case_id"],
                "source_prompt": state["source_prompt"],
                "source_choices": state["source_choices"],
            }, sort_keys=True).encode()
        ).hexdigest()[:20],
        source_type="reasoning-task",
        required_obligations=(OBLIGATION,),
        provenance=provenance,
    )
    return target, record


def _compact_program_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "program_id": record["program_id"],
        "operators": record["operators"],
        "basis_operators": record["basis_operators"],
        "recovery_complete": record.get("recovery_route") is not None,
        "contains_non_exact_step": record["contains_non_exact_step"],
        "steps": [
            {
                "operator": row["operator"],
                "basis_operator": row["basis_operator"],
                "verification_status": (row.get("verification_state") or {}).get("status"),
                "precondition_status": (row.get("precondition_verification") or {}).get("status"),
                "discarded_information": row.get("discarded_information") or [],
                "inverse_or_recovery_route": row.get("inverse_or_recovery_route"),
            }
            for row in record["steps"]
        ],
    }


PROGRAMS = {
    "first-stop": ("first", "compose"),
    "second-stop": ("second", "compose"),
    "forward": ("first", "second", "compose"),
    "reverse": ("second", "first", "compose"),
}


def self_test(helix_dir: str) -> dict[str, Any]:
    rows = []
    for case in build_battery(20261221, 1):
        state = _source_state(case["family"], case["case_id"], case["surfaces"])
        programs = {}
        for name, program in PROGRAMS.items():
            target, record = _execute(
                helix_dir, state, program,
                {"run_id": "058-cross-family-compiler-portfolio", "self_test": True, "program": name},
            )
            if "correct_index" in target:
                raise AssertionError("correctness leaked into compiler target")
            programs[name] = {
                "basis": record["basis_operators"],
                "recovery_complete": record.get("recovery_route") is not None,
                "final_prompt_sha256": hashlib.sha256(target["final_prompt"].encode()).hexdigest(),
                "final_choices_sha256": hashlib.sha256(json.dumps(target["final_choices"]).encode()).hexdigest(),
            }
        rows.append({"family": case["family"], "case_id": case["case_id"], "programs": programs})
    return {
        "schema_version": RUN_VERSION,
        "status": "self-test-pass",
        "cases": len(rows),
        "actual_execution_owner": "engine.ir.passes.execute_transform_program",
        "correctness_visible_to_compiler": False,
        "rows": rows,
    }


def _summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    hits = 0
    by_family: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        ok = int(row["predictions"][arm]) == int(row["correct_index"])
        hits += int(ok)
        by_family[row["family"]].append(ok)
    return {
        "cases": len(rows),
        "correct": hits,
        "accuracy": hits / len(rows),
        "by_family": {k: sum(v) / len(v) for k, v in sorted(by_family.items())},
    }


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


def evaluate(helix_dir: str, model_dir: str, seeds: list[int], cases_per_family: int) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()
    started = time.time()
    rows = []
    spec = []

    for seed in seeds:
        seed_spec = []
        for case in build_battery(seed, cases_per_family):
            # Remove correctness before compiler construction.
            state = _source_state(case["family"], case["case_id"], case["surfaces"])
            source_scored = _score_values(scorer, state["source_prompt"], state["source_choices"])
            predictions = {"source-layout": int(source_scored["prediction"])}
            score_vectors = {"source-layout": [float(x) for x in source_scored["scores"]]}
            program_records = {}
            route_hashes = {}

            for arm, program in PROGRAMS.items():
                target, record = _execute(
                    helix_dir, state, program,
                    {
                        "run_id": "058-cross-family-compiler-portfolio",
                        "seed": seed,
                        "case_id": case["case_id"],
                        "program": arm,
                    },
                )
                scored = _score_values(scorer, target["final_prompt"], target["final_choices"])
                predictions[arm] = int(scored["prediction"])
                score_vectors[arm] = [float(x) for x in scored["scores"]]
                program_records[arm] = _compact_program_record(record)
                route_hashes[arm] = {
                    "prompt_sha256": hashlib.sha256(target["final_prompt"].encode()).hexdigest(),
                    "choices_sha256": hashlib.sha256(json.dumps(target["final_choices"], sort_keys=True).encode()).hexdigest(),
                }

            correct = int(case["correct_index"])
            rows.append({
                "uid": f"{seed}:{case['case_id']}",
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "predictions": predictions,
                "scores": score_vectors,
                "program_records": program_records,
                "route_hashes": route_hashes,
            })
            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": correct,
                "source_sha256": hashlib.sha256(state["source_prompt"].encode()).hexdigest(),
                "route_hashes": route_hashes,
            })
        spec.append({"seed": seed, "cases": seed_spec})

    arms = ("source-layout", "first-stop", "second-stop", "forward", "reverse")
    summaries = {arm: _summary(rows, arm) for arm in arms}
    paired = {
        "forward-vs-source": _paired(rows, "forward", "source-layout"),
        "reverse-vs-source": _paired(rows, "reverse", "source-layout"),
        "forward-vs-reverse": _paired(rows, "forward", "reverse"),
        "first-stop-vs-source": _paired(rows, "first-stop", "source-layout"),
        "second-stop-vs-source": _paired(rows, "second-stop", "source-layout"),
    }

    source_only = sum(r["predictions"]["source-layout"] == r["correct_index"] for r in rows)
    any_compiler = sum(
        any(r["predictions"][arm] == r["correct_index"] for arm in arms[1:])
        for r in rows
    )
    any_arm = sum(
        any(r["predictions"][arm] == r["correct_index"] for arm in arms)
        for r in rows
    )
    strict_compiler_rescues = sum(
        r["predictions"]["source-layout"] != r["correct_index"]
        and any(r["predictions"][arm] == r["correct_index"] for arm in arms[1:])
        for r in rows
    )
    order_disagreements = sum(r["predictions"]["forward"] != r["predictions"]["reverse"] for r in rows)

    integrity = {
        "programs": 0,
        "all_recovery_complete": True,
        "all_exact": True,
        "all_preconditions_pass": True,
        "all_result_verifications_pass": True,
    }
    for row in rows:
        for record in row["program_records"].values():
            integrity["programs"] += 1
            integrity["all_recovery_complete"] &= bool(record["recovery_complete"])
            integrity["all_exact"] &= not bool(record["contains_non_exact_step"])
            for step in record["steps"]:
                integrity["all_preconditions_pass"] &= step["precondition_status"] == "pass"
                integrity["all_result_verifications_pass"] &= step["verification_status"] == "pass"

    forward_pair = paired["forward-vs-source"]
    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started
    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-cross-family-actual-compiler-pilot",
        "actual_execution_owner": "engine.ir.passes.execute_transform_program",
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(rows),
        "source_battery_sha256": sha256_json(spec),
        "arms": summaries,
        "paired": paired,
        "coverage": {
            "source_correct": source_only,
            "any_compiler_arm_correct": any_compiler,
            "any_arm_correct": any_arm,
            "strict_compiler_rescues_of_source_failure": strict_compiler_rescues,
            "forward_reverse_prediction_disagreements": order_disagreements,
        },
        "compiler_integrity": integrity,
        "decisions": {
            "global_forward_route_gain_earned": (
                summaries["forward"]["accuracy"] > summaries["source-layout"]["accuracy"]
                and forward_pair["wins"] > forward_pair["regressions"]
            ),
            "transform_order_changes_predictions": order_disagreements > 0,
            "compiler_candidate_expansion_established": strict_compiler_rescues > 0,
            "dynamic_route_selector_established": False,
        },
        "resources": resources,
        "rows": rows,
        "claim_boundary": (
            "Fresh public procedural pilot on five reasoning geometries. The actual Helix guarded "
            "execute_transform_program owner constructs every compiler arm from learner-visible state. "
            "Correctness is withheld until actor predictions are frozen. Candidate coverage is not "
            "deployable selection, one pilot does not establish a universal route, and no model promotion follows."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helix-dir", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--seeds", default="20261221,20261222")
    parser.add_argument("--cases-per-family", type=int, default=2)
    parser.add_argument("--output", default="run-058-cross-family-compiler-portfolio.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(args.helix_dir), indent=2, sort_keys=True))
        return

    if not args.model_dir or not args.model_file or not args.expected_sha256:
        raise SystemExit("model-dir, model-file and expected-sha256 are required")
    model_path = Path(args.model_file)
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    result = evaluate(
        args.helix_dir,
        args.model_dir,
        [int(x) for x in args.seeds.split(",") if x.strip()],
        args.cases_per_family,
    )
    result["actor"] = {
        "model": "HuggingFaceTB/SmolLM2-360M",
        "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        "model_file_sha256": observed,
        "model_bytes": model_path.stat().st_size,
        "weights_frozen": True,
        "dtype": "float32",
        "device": "cpu",
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema_version": result["schema_version"],
        "cases": result["cases"],
        "arms": result["arms"],
        "paired": result["paired"],
        "coverage": result["coverage"],
        "compiler_integrity": result["compiler_integrity"],
        "decisions": result["decisions"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
