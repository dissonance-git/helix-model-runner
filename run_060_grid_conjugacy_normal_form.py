"""Run 060: exact coordinate-conjugacy normal form over a frozen neural backend.

Helix changes the visible coordinate frame, executes the same semantic task in a
conjugate form, then relies on aligned candidate indices to recover the answer in
the native frame.  Correctness is never available to transformation or routing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable

from offline_reasoning_battery import build_battery, sha256_json
from offline_reasoning_content_probe import surface_task_and_choices
from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import build_representations, _score_values
from run_039_exact_representation_operators import grid_operators
from run_058_cross_family_compiler_portfolio import _compose_impl, _compact_program_record

RUN_VERSION = "helix-grid-conjugacy-normal-form-001.0"
OBLIGATION = "preserve-visible-grid-task-and-aligned-answer-index"
GRID_PATTERN = re.compile(r"[0-2]{3}/[0-2]{3}/[0-2]{3}")


def _grid(value: str) -> tuple[tuple[str, ...], ...]:
    rows = value.split("/")
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError(value)
    return tuple(tuple(row) for row in rows)


def _render(g: tuple[tuple[str, ...], ...]) -> str:
    return "/".join("".join(row) for row in g)


def _rot90_grid(value: str) -> str:
    g = _grid(value)
    return _render(tuple(tuple(row) for row in zip(*g[::-1])))


def _rot270_grid(value: str) -> str:
    out = value
    for _ in range(3):
        out = _rot90_grid(out)
    return out


def _flip_h_grid(value: str) -> str:
    g = _grid(value)
    return _render(tuple(tuple(reversed(row)) for row in g))


def _flip_v_grid(value: str) -> str:
    g = _grid(value)
    return _render(tuple(reversed(g)))


def _transpose_grid(value: str) -> str:
    g = _grid(value)
    return _render(tuple(tuple(row) for row in zip(*g)))


GRID_OPS: dict[str, Callable[[str], str]] = {
    "rotate-90": _rot90_grid,
    "flip-horizontal": _flip_h_grid,
    "flip-vertical": _flip_v_grid,
    "transpose": _transpose_grid,
}


def _transform_all_grids(text: str, fn: Callable[[str], str]) -> str:
    return GRID_PATTERN.sub(lambda m: fn(m.group(0)), text)


def _compact_only_surfaces(surfaces: dict[str, str]) -> dict[str, str]:
    # All grid representations/operators used in this experiment read compact
    # visible input only.  Keeping only that authoritative surface avoids an
    # inconsistent untouched prose sibling after coordinate re-projection.
    return {"compact": str(surfaces["compact"])}


def _infer_visible_grid_transform(surfaces: dict[str, str]) -> str:
    stem, _choices = surface_task_and_choices(surfaces["compact"])
    grids = GRID_PATTERN.findall(stem)
    if len(grids) < 3:
        raise ValueError("not enough visible grids")
    before, after = grids[0], grids[1]
    matches = [name for name, fn in GRID_OPS.items() if fn(before) == after]
    if len(matches) != 1:
        raise ValueError(f"visible demo does not identify one transform: {matches}")
    return matches[0]


def _source_state(case_id: str, surfaces: dict[str, str]) -> dict[str, Any]:
    visible = _compact_only_surfaces(surfaces)
    canonical = build_representations("abstract-transformation", visible)
    source_prompt, source_choices = canonical["layout"]
    return {
        "schema": "run060-visible-grid-state-001.0",
        "family": "abstract-transformation",
        "case_id": case_id,
        "surfaces": visible,
        "source_prompt": source_prompt,
        "source_stem": source_prompt.removesuffix("Answer value:").rstrip(),
        "source_choices": list(source_choices),
        "views": [],
        "frame_quarter_turns": 0,
        "visible_transform": _infer_visible_grid_transform(visible),
    }


def _load_compiler(helix_dir: str):
    root = str(Path(helix_dir).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from engine.ir.passes import OperatorSpec, TransformationRegistry, execute_transform_program
    return OperatorSpec, TransformationRegistry, execute_transform_program


def _precondition(source: Any) -> dict[str, Any]:
    ok = (
        isinstance(source, dict)
        and "correct_index" not in source
        and source.get("family") == "abstract-transformation"
        and isinstance((source.get("surfaces") or {}).get("compact"), str)
        and len(source.get("source_choices") or []) == 4
    )
    return {
        "status": "pass" if ok else "fail",
        "applicable": ok,
        "exact": True,
        "verified_preconditions": ["visible-grid-task-only"] if ok else [],
    }


def _reframe_impl(source: dict[str, Any]):
    old_compact = source["surfaces"]["compact"]
    new_compact = _transform_all_grids(old_compact, _rot90_grid)
    target = _source_state(source["case_id"], {"compact": new_compact})
    target["frame_quarter_turns"] = (int(source.get("frame_quarter_turns") or 0) + 1) % 4
    target["native_visible_transform"] = source.get("visible_transform")
    return target, {
        "introduced_information": ["exact 90-degree coordinate-frame change"],
        "discarded_information": [],
        "expected_observable": "conjugate semantic transform may align with a different neural execution regime",
        "known_risk": ["coordinate-sensitive actor may improve or degrade"],
        "inverse_or_recovery_route": "rotate every visible grid and aligned answer value 270 degrees",
        "peak_intermediate_size": len(new_compact),
    }


def _reframe_verify(source: Any, target: Any) -> dict[str, Any]:
    try:
        restored = _transform_all_grids(target["surfaces"]["compact"], _rot270_grid)
        source_transform = _infer_visible_grid_transform(source["surfaces"])
        target_transform = _infer_visible_grid_transform(target["surfaces"])
        # 90-degree conjugacy swaps horizontal/vertical and preserves rotate-90.
        expected = {
            "flip-horizontal": "flip-vertical",
            "flip-vertical": "flip-horizontal",
            "rotate-90": "rotate-90",
        }.get(source_transform)
        exact = restored == source["surfaces"]["compact"] and (
            expected is None or target_transform == expected
        )
        return {
            "status": "pass" if exact else "fail",
            "exact": exact,
            "source_recovery_preserved": restored == source["surfaces"]["compact"],
            "source_visible_transform": source_transform,
            "target_visible_transform": target_transform,
            "expected_target_transform": expected,
        }
    except Exception as exc:
        return {"status": "fail", "exact": False, "reason": repr(exc)}


def _cells_impl(source: dict[str, Any]):
    op_id, stem, choices = grid_operators(source["surfaces"])[0]
    target = dict(source)
    target["views"] = [dict(row) for row in source.get("views") or []]
    target["views"].append({
        "operator_id": op_id,
        "stem": stem,
        "choices": list(choices),
    })
    return target, {
        "introduced_information": ["nine explicit coordinate/value facts for every visible grid"],
        "discarded_information": [],
        "expected_observable": "make positional mapping explicit in current coordinate frame",
        "known_risk": ["surface expansion", "actor coordinate anisotropy"],
        "inverse_or_recovery_route": "source_prompt and source_choices retained verbatim",
        "peak_intermediate_size": len(stem) + sum(len(x) for x in choices),
    }


def _same_source_verify(source: Any, target: Any) -> dict[str, Any]:
    ok = (
        isinstance(source, dict) and isinstance(target, dict)
        and "correct_index" not in target
        and target.get("source_prompt") == source.get("source_prompt")
        and target.get("source_choices") == source.get("source_choices")
        and all(len(row.get("choices") or []) == 4 for row in target.get("views") or [])
    )
    if "final_choices" in target:
        ok = ok and len(target["final_choices"]) == 4 and str(target.get("final_prompt") or "").endswith("Answer value:")
    return {"status": "pass" if ok else "fail", "exact": bool(ok), "source_recovery_preserved": bool(ok)}


def _registry(helix_dir: str):
    OperatorSpec, TransformationRegistry, execute_transform_program = _load_compiler(helix_dir)
    registry = TransformationRegistry()
    registry.register(OperatorSpec(
        name="reframe",
        source_types=("reasoning-task",),
        target_type="reasoning-task",
        preconditions=("visible-grid-task-only",),
        preserved_obligations=(OBLIGATION,),
        state="exact",
        basis_operator="REPROJECT",
        implementation=_reframe_impl,
        verifier=_reframe_verify,
        precondition_verifier=_precondition,
        certificate_requirements=("exact inverse frame", "candidate index alignment"),
        cheapest_falsifier="rotate target back 270 degrees and compare visible compact source byte-for-byte",
    ))
    registry.register(OperatorSpec(
        name="cells",
        source_types=("reasoning-task",),
        target_type="reasoning-task",
        preconditions=("visible-grid-task-only",),
        preserved_obligations=(OBLIGATION,),
        state="exact",
        basis_operator="REPROJECT",
        implementation=_cells_impl,
        verifier=_same_source_verify,
        precondition_verifier=_precondition,
        certificate_requirements=("source retained", "four aligned alternatives"),
        cheapest_falsifier="mutate one coordinate/value fact and require exact mismatch",
    ))
    registry.register(OperatorSpec(
        name="compose",
        source_types=("reasoning-task",),
        target_type="reasoning-task",
        preconditions=("visible-grid-task-only",),
        preserved_obligations=(OBLIGATION,),
        state="exact",
        basis_operator="COMPOSE",
        implementation=_compose_impl,
        verifier=_same_source_verify,
        precondition_verifier=_precondition,
        certificate_requirements=("source retained", "candidate alignment retained"),
        cheapest_falsifier="mutate one aligned candidate and require exact mismatch",
    ))
    return registry, execute_transform_program


PROGRAMS = {
    "cells": ("cells", "compose"),
    "reframe": ("reframe", "compose"),
    "reframe_cells": ("reframe", "cells", "compose"),
}


def _execute(helix_dir: str, state: dict[str, Any], arm: str, provenance: dict[str, Any]):
    registry, execute_transform_program = _registry(helix_dir)
    return execute_transform_program(
        registry,
        PROGRAMS[arm],
        state,
        source_id="GRID-" + hashlib.sha256(
            json.dumps({
                "case_id": state["case_id"],
                "compact": state["surfaces"]["compact"],
            }, sort_keys=True).encode()
        ).hexdigest()[:20],
        source_type="reasoning-task",
        required_obligations=(OBLIGATION,),
        provenance={**provenance, "arm": arm},
    )


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


def _margin(scores: list[float]) -> float:
    ordered = sorted((float(x) for x in scores), reverse=True)
    return ordered[0] - ordered[1]


def self_test(helix_dir: str) -> dict[str, Any]:
    rows = []
    for seed in (20270110, 20270111):
        for case in build_battery(seed, 4):
            if case["family"] != "abstract-transformation":
                continue
            state = _source_state(case["case_id"], case["surfaces"])
            subtype = state["visible_transform"]
            if subtype not in {"flip-horizontal", "flip-vertical"}:
                continue
            if "correct_index" in state:
                raise AssertionError("correctness leaked into state")
            target, rec = _execute(
                helix_dir, state, "reframe_cells",
                {"run_id": "060-grid-conjugacy-normal-form", "self_test": True, "seed": seed},
            )
            expected = "flip-vertical" if subtype == "flip-horizontal" else "flip-horizontal"
            if target["visible_transform"] != expected:
                raise AssertionError((subtype, target["visible_transform"], expected))
            rows.append({
                "seed": seed,
                "case_id": case["case_id"],
                "native_transform": subtype,
                "compiled_transform": target["visible_transform"],
                "basis": rec["basis_operators"],
                "recovery_complete": rec.get("recovery_route") is not None,
                "contains_non_exact_step": rec["contains_non_exact_step"],
            })
    return {
        "schema_version": RUN_VERSION,
        "status": "self-test-pass",
        "actual_execution_owner": "engine.ir.passes.execute_transform_program",
        "correctness_visible_to_compiler": False,
        "rows": rows,
    }


def evaluate(helix_dir: str, model_dir: str, seeds: list[int]) -> dict[str, Any]:
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()
    started = time.time()
    rows: list[dict[str, Any]] = []
    public_spec = []

    for seed in seeds:
        seed_spec = []
        for case in build_battery(seed, 4):
            if case["family"] != "abstract-transformation":
                continue
            state = _source_state(case["case_id"], case["surfaces"])
            subtype = state["visible_transform"]
            if subtype not in {"flip-horizontal", "flip-vertical"}:
                continue

            source = _score_values(scorer, state["source_prompt"], state["source_choices"])
            predictions = {"source": int(source["prediction"])}
            scores = {"source": [float(x) for x in source["scores"]]}
            records = {}

            for arm in ("cells", "reframe", "reframe_cells"):
                target, record = _execute(
                    helix_dir, state, arm,
                    {
                        "run_id": "060-grid-conjugacy-normal-form",
                        "seed": seed,
                        "case_id": case["case_id"],
                        "native_transform": subtype,
                    },
                )
                scored = _score_values(scorer, target["final_prompt"], target["final_choices"])
                predictions[arm] = int(scored["prediction"])
                scores[arm] = [float(x) for x in scored["scores"]]
                records[arm] = _compact_program_record(record)

            compiled_arm = "cells" if subtype == "flip-horizontal" else "reframe_cells"
            predictions["compiled"] = predictions[compiled_arm]
            scores["compiled"] = list(scores[compiled_arm])

            correct = int(case["correct_index"])
            rows.append({
                "uid": f"{seed}:{case['case_id']}",
                "seed": seed,
                "case_id": case["case_id"],
                "native_transform": subtype,
                "compiled_arm": compiled_arm,
                "correct_index": correct,
                "predictions": predictions,
                "margins": {name: _margin(value) for name, value in scores.items()},
                "program_records": records,
            })
            seed_spec.append({
                "case_id": case["case_id"],
                "native_transform": subtype,
                "correct_index": correct,
                "compact_sha256": hashlib.sha256(state["surfaces"]["compact"].encode()).hexdigest(),
            })
        public_spec.append({"seed": seed, "cases": seed_spec})

    by_type = {}
    for subtype in ("flip-horizontal", "flip-vertical"):
        subset = [r for r in rows if r["native_transform"] == subtype]
        by_type[subtype] = {
            arm: _summary(subset, arm)
            for arm in ("source", "cells", "reframe", "reframe_cells", "compiled")
        }
        by_type[subtype]["reframe_cells_vs_cells"] = _paired(subset, "reframe_cells", "cells")
        by_type[subtype]["cells_vs_reframe_cells"] = _paired(subset, "cells", "reframe_cells")

    overall = {
        arm: _summary(rows, arm)
        for arm in ("source", "cells", "reframe", "reframe_cells", "compiled")
    }
    overall["compiled_vs_source"] = _paired(rows, "compiled", "source")
    overall["compiled_vs_cells"] = _paired(rows, "compiled", "cells")

    integrity = {
        "programs": 0,
        "all_exact": True,
        "all_recovery_complete": True,
        "all_preconditions_pass": True,
        "all_result_verifications_pass": True,
    }
    for row in rows:
        for record in row["program_records"].values():
            integrity["programs"] += 1
            integrity["all_exact"] &= not bool(record["contains_non_exact_step"])
            integrity["all_recovery_complete"] &= bool(record["recovery_complete"])
            for step in record["steps"]:
                integrity["all_preconditions_pass"] &= step["precondition_status"] == "pass"
                integrity["all_result_verifications_pass"] &= step["verification_status"] == "pass"

    v = by_type["flip-vertical"]
    h = by_type["flip-horizontal"]
    vertical_pair = v["reframe_cells_vs_cells"]
    horizontal_pair = h["cells_vs_reframe_cells"]
    compiled_source_pair = overall["compiled_vs_source"]
    compiled_cells_pair = overall["compiled_vs_cells"]

    resources = scorer.counters()
    resources["wall_seconds"] = time.time() - started

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-exact-conjugacy-test",
        "seeds": seeds,
        "cases": len(rows),
        "source_battery_sha256": sha256_json(public_spec),
        "mathematical_certificate": {
            "coordinate_change": "rotate-90",
            "inverse": "rotate-270",
            "conjugacy": {
                "flip-horizontal": "flip-vertical",
                "flip-vertical": "flip-horizontal",
            },
            "candidate_index_alignment": "same exact bijection applied to every visible grid including all answer values",
        },
        "by_native_transform": by_type,
        "overall": overall,
        "decisions": {
            "vertical_normal_form_gain_earned": (
                v["reframe_cells"]["correct"] > v["cells"]["correct"]
                and vertical_pair["wins"] > vertical_pair["regressions"]
            ),
            "horizontal_native_frame_advantage_earned": (
                h["cells"]["correct"] > h["reframe_cells"]["correct"]
                and horizontal_pair["wins"] > horizontal_pair["regressions"]
            ),
            "crossover_interaction_observed": (
                vertical_pair["wins"] > vertical_pair["regressions"]
                and horizontal_pair["wins"] > horizontal_pair["regressions"]
            ),
            "compiled_policy_gain_vs_source": (
                overall["compiled"]["correct"] > overall["source"]["correct"]
                and compiled_source_pair["wins"] > compiled_source_pair["regressions"]
            ),
            "compiled_policy_gain_vs_universal_cells": (
                overall["compiled"]["correct"] > overall["cells"]["correct"]
                and compiled_cells_pair["wins"] > compiled_cells_pair["regressions"]
            ),
        },
        "compiler_integrity": integrity,
        "resources": resources,
        "rows": rows,
        "claim_boundary": (
            "Fresh public exact-conjugacy test on one grid family and one frozen 360M actor. "
            "A positive result would establish only a bounded coordinate-normalization mechanism, "
            "not a universal compiler law, general reasoning gain, or model promotion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helix-dir", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--seeds", default="20270110,20270111,20270112,20270113,20270114,20270115,20270116,20270117")
    parser.add_argument("--output", default="run-060-grid-conjugacy-normal-form.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(args.helix_dir), indent=2, sort_keys=True))
        return

    if not args.model_dir or not args.model_file or not args.expected_sha256:
        raise SystemExit("model-dir, model-file and expected-sha256 are required")

    model_path = Path(args.model_file)
    h = hashlib.sha256()
    with model_path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            h.update(chunk)
    observed = h.hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    result = evaluate(
        args.helix_dir,
        args.model_dir,
        [int(x) for x in args.seeds.split(",") if x.strip()],
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
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema_version": result["schema_version"],
        "by_native_transform": result["by_native_transform"],
        "overall": result["overall"],
        "decisions": result["decisions"],
        "compiler_integrity": result["compiler_integrity"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
