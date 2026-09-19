"""Executable Rule 30 binding for Helix's first-class representation compiler.

The compiler transports the exact width-4 training object used by Helix Model
Run 053. It never generates a classifier and never computes held-out labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from engine.ir import OperatorSpec, TransformationRegistry, execute_transform_program
from engine.runtime.experiments.cellular_automata.rule30_077_zero_tail_basin import (
    derivative,
    forcing,
    shift,
    word_period,
    zero_tail_basin,
)

PROGRAM_VERSION = "rule30-transform-program-001.0"
TRAIN_WIDTH = 4
HELDOUT_WIDTHS = (8, 16, 32)
BRANCH_OBLIGATION = "classify-zero-tail-basin-branching-without-heldout-labels"

SOURCE_TYPE = "forward-rule30-history"
DUAL_TYPE = "reverse-zero-tail-ancestry"
FACTOR_TYPE = "branch-core-plus-forced-exterior"
LIFT_TYPE = "bounded-algebraic-feature-space"
PROJECT_TYPE = "small-boolean-dnf-classification-task"

DUAL_OPERATOR = "rule30-dualize-zero-tail"
FACTOR_OPERATOR = "rule30-factor-branching"
LIFT_OPERATOR = "rule30-lift-algebraic-features"
PROJECT_OPERATOR = "rule30-project-actor-task"
PROGRAM_OPERATORS = (DUAL_OPERATOR, FACTOR_OPERATOR, LIFT_OPERATOR, PROJECT_OPERATOR)
BASIS_OPERATORS = ("DUALIZE", "FACTOR", "LIFT", "PROJECT")


def state_features(state: tuple[int, int, int, int], n: int) -> dict[str, bool]:
    """Exact feature language frozen by Helix Model Run 053."""
    a, b, c, d = state
    mask = (1 << n) - 1
    h = forcing(a, b, c, d, n)
    words = {"a": a, "b": b, "c": c, "d": d, "h": h}
    features: dict[str, bool] = {}
    for name, word in words.items():
        period = word_period(word, n)
        for bound in (1, 2, 4):
            features[f"{name}.period<={bound}"] = period <= bound
        features[f"{name}.zero"] = word == 0
        features[f"{name}.ones"] = word == mask
        features[f"{name}.even-weight"] = word.bit_count() % 2 == 0
        if n % 2 == 0:
            features[f"{name}.half-repeat"] = shift(word, n, n // 2) == word
            features[f"{name}.half-complement"] = shift(word, n, n // 2) == (word ^ mask)

    names = tuple(words)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            features[f"{left}={right}"] = words[left] == words[right]
            features[f"{left}=~{right}"] = (words[left] ^ words[right]) == mask

    for name, word in words.items():
        dword = derivative(word, n)
        for right, right_word in words.items():
            features[f"D({name})={right}"] = dword == right_word
    return features


def feature_alias(name: str) -> str:
    suffixes = {
        ".period<=1": "p1", ".period<=2": "p2", ".period<=4": "p4",
        ".zero": "z", ".ones": "o", ".even-weight": "e",
        ".half-repeat": "hr", ".half-complement": "hc",
    }
    for suffix, short in suffixes.items():
        if name.endswith(suffix):
            return name[:-len(suffix)] + short
    if name.startswith("D(") and ")=" in name:
        left, right = name[2:].split(")=", 1)
        return f"D{left}={right}"
    if "=~" in name:
        return name.replace("=~", "~")
    return name


def _pass(**extra: Any) -> dict[str, Any]:
    return {"status": "pass", "exact": True, **extra}


def _dual_preconditions(source: Mapping[str, Any]) -> dict[str, Any]:
    verified = []
    width = source.get("width")
    if isinstance(width, int) and not isinstance(width, bool) and width >= 4 and width % 2 == 0:
        verified.append("cyclic-even-width-at-least-4")
    if source.get("semantics") == "rule30-right-history":
        verified.append("rule30-right-history-semantics")
    return {
        "status": "pass" if len(verified) == 2 else "fail",
        "applicable": len(verified) == 2,
        "exact": True,
        "verified_preconditions": verified,
    }


def _dualize(source: Mapping[str, Any]):
    width = int(source["width"])
    depth, edges = zero_tail_basin(width)
    states = tuple(sorted(depth))
    return {
        "width": width,
        "semantics": source["semantics"],
        "states": states,
        "depth_by_state": dict(depth),
        "edges": tuple(edges),
        "complete_reverse_basin": True,
    }, {
        "target_id": f"rule30:zero-tail-ancestry:w{width}",
        "introduced_information": [
            "exact reverse-basin membership", "exact reverse depth",
            "exact zero-tail continuation edges",
        ],
        "inverse_or_recovery_route": "replay-forward-right-history-from-reverse-basin",
        "expected_observable": "complete zero-tail ancestry rather than full state space",
        "peak_intermediate_size": len(states),
    }


def _verify_dualize(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    depth, edges = zero_tail_basin(int(source["width"]))
    ok = (
        target.get("complete_reverse_basin") is True
        and tuple(target.get("states") or ()) == tuple(sorted(depth))
        and dict(target.get("depth_by_state") or {}) == depth
        and tuple(target.get("edges") or ()) == tuple(edges)
    )
    return _pass(state_count=len(depth), edge_count=len(edges)) if ok else {
        "status": "fail", "exact": True, "reason": "reverse-basin replay mismatch"
    }


def _factor_preconditions(source: Mapping[str, Any]) -> dict[str, Any]:
    verified = []
    if source.get("complete_reverse_basin") is True:
        verified.append("complete-reverse-zero-tail-basin")
    states = set(source.get("states") or ())
    edges = tuple(source.get("edges") or ())
    if states and all(left in states and right in states for left, right in edges):
        verified.append("closed-reverse-basin-edge-set")
    return {
        "status": "pass" if len(verified) == 2 else "fail",
        "applicable": len(verified) == 2,
        "exact": True,
        "verified_preconditions": verified,
    }


def _factor(source: Mapping[str, Any]):
    states = tuple(source["states"])
    edges = tuple(source["edges"])
    outdegree = Counter(left for left, _right in edges)
    branch_set = {state for state in states if outdegree[state] > 1}
    branch = tuple(state for state in states if state in branch_set)
    nonbranch = tuple(state for state in states if state not in branch_set)
    rows = tuple({
        "state": state,
        "is_branch": state in branch_set,
        "zero_basin_outdegree": int(outdegree[state]),
    } for state in states)
    return {
        "width": int(source["width"]),
        "semantics": source["semantics"],
        "rows": rows,
        "branch_states": branch,
        "nonbranch_states": nonbranch,
        "branch_partition_complete": True,
    }, {
        "target_id": f"rule30:branch-partition:w{source['width']}",
        "introduced_information": ["branch versus forced-continuation partition"],
        "inverse_or_recovery_route": "recompose-branch-and-nonbranch-rows-by-state",
        "expected_observable": "branch iff reverse-basin outdegree exceeds one",
        "peak_intermediate_size": len(states),
    }


def _verify_factor(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    states = tuple(source["states"])
    outdegree = Counter(left for left, _right in source["edges"])
    expected = {state for state in states if outdegree[state] > 1}
    observed = set(target.get("branch_states") or ())
    nonbranch = set(target.get("nonbranch_states") or ())
    ok = (
        target.get("branch_partition_complete") is True
        and observed == expected
        and observed.isdisjoint(nonbranch)
        and observed | nonbranch == set(states)
    )
    return _pass(branch_state_count=len(expected)) if ok else {
        "status": "fail", "exact": True, "reason": "branch partition mismatch"
    }


def _lift_preconditions(source: Mapping[str, Any]) -> dict[str, Any]:
    states = [row.get("state") for row in source.get("rows") or ()]
    verified = []
    if source.get("branch_partition_complete") is True:
        verified.append("complete-branch-partition")
    if states and len(states) == len(set(states)):
        verified.append("unique-source-state-identity")
    return {
        "status": "pass" if len(verified) == 2 else "fail",
        "applicable": len(verified) == 2,
        "exact": True,
        "verified_preconditions": verified,
    }


def _lift(source: Mapping[str, Any]):
    width = int(source["width"])
    rows = tuple({
        "state": row["state"],
        "is_branch": bool(row["is_branch"]),
        "zero_basin_outdegree": int(row["zero_basin_outdegree"]),
        "features": state_features(row["state"], width),
    } for row in source["rows"])
    feature_names = tuple(sorted(rows[0]["features"])) if rows else ()
    return {
        "width": width,
        "semantics": source["semantics"],
        "rows": rows,
        "feature_names": feature_names,
        "feature_space_complete": True,
    }, {
        "target_id": f"rule30:algebraic-features:w{width}",
        "introduced_information": [
            "word period bounds", "zero/ones/even-weight",
            "half-repeat/half-complement", "word equality/complement",
            "cyclic derivative equality",
        ],
        "inverse_or_recovery_route": "drop-derived-features-and-recover-state-partition",
        "expected_observable": "exact Boolean relations over a,b,c,d,h",
        "peak_intermediate_size": len(rows) * max(1, len(feature_names)),
    }


def _verify_lift(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    width = int(source["width"])
    source_rows = {row["state"]: row for row in source["rows"]}
    target_rows = {row["state"]: row for row in target.get("rows") or ()}
    if set(source_rows) != set(target_rows):
        return {"status": "fail", "exact": True, "reason": "state identity mismatch"}
    for state, source_row in source_rows.items():
        row = target_rows[state]
        if bool(row["is_branch"]) != bool(source_row["is_branch"]):
            return {"status": "fail", "exact": True, "reason": "branch label mismatch"}
        if dict(row["features"]) != state_features(state, width):
            return {"status": "fail", "exact": True, "reason": "feature replay mismatch"}
    return _pass(row_count=len(target_rows), feature_count=len(target.get("feature_names") or ()))


def _project_preconditions(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = tuple(source.get("rows") or ())
    verified = []
    if source.get("feature_space_complete") is True:
        verified.append("complete-algebraic-feature-space")
    if rows and all(isinstance(row.get("features"), Mapping) for row in rows):
        verified.append("exact-feature-rows")
    if int(source.get("width") or 0) == TRAIN_WIDTH:
        verified.append("width-4-training-scope")
    return {
        "status": "pass" if len(verified) == 3 else "fail",
        "applicable": len(verified) == 3,
        "exact": True,
        "verified_preconditions": verified,
    }


def _aliases(features: tuple[str, ...]) -> dict[str, str]:
    aliases = {feature_alias(name): name for name in features}
    if len(aliases) != len(features):
        raise AssertionError("feature alias collision")
    return aliases


def _actor_projection(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = tuple(source["rows"])
    feature_names = tuple(source["feature_names"])
    variable = tuple(
        name for name in feature_names
        if len({bool(row["features"][name]) for row in rows}) > 1
    )
    aliases = _aliases(variable)
    full_to_alias = {full: alias for alias, full in aliases.items()}
    branch_rows = [row for row in rows if row["is_branch"]]
    nonbranch_rows = [row for row in rows if not row["is_branch"]]
    ordered_rows = branch_rows + nonbranch_rows

    truth: dict[str, str] = {}
    for name in variable:
        mask = 0
        for index, row in enumerate(ordered_rows):
            if row["features"][name]:
                mask |= 1 << index
        truth[full_to_alias[name]] = format(mask, "011x")

    payload = {
        "task": "Find 1-3 shortest DNF formulas whose exact 42-row truth mask equals target. Hidden widths 8/16/32 are tested unchanged.",
        "route": "DUALIZE>FACTOR>LIFT>PROJECT",
        "legend": "xp1/p2/p4=period(x)<=1/2/4;xz/xo/xe=zero/ones/even;xhr/xhc=half-repeat/half-complement;x=y equal;x~y complement;Dx=y cyclic-derivative(x)=y",
        "bits": "hex bit i=row i; rows0..9=B,10..41=N; !alias=42-bit complement; AND literals per clause, OR clauses",
        "target": format((1 << len(branch_rows)) - 1, "011x"),
        "truth": truth,
        "output": {"candidates": [{"id": "x", "dnf": [["alias", "!alias"]]}]},
        "rules": "Use aliases exactly. No row/state IDs in formulas. Prefer fewest literals. JSON only. No all-width claim.",
    }
    prompt = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    recovery_index = [{
        "row": index,
        "state": [format(word, f"0{TRAIN_WIDTH}b") for word in row["state"]],
        "label": "BRANCH" if row["is_branch"] else "NONBRANCH",
    } for index, row in enumerate(ordered_rows)]
    return {
        "width": TRAIN_WIDTH,
        "heldout_widths": list(HELDOUT_WIDTHS),
        "heldout_labels_visible_to_actor": False,
        "row_count": len(ordered_rows),
        "branch_row_count": len(branch_rows),
        "feature_count": len(variable),
        "feature_aliases": aliases,
        "actor_payload": payload,
        "actor_prompt": prompt,
        "actor_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "actor_prompt_bytes": len(prompt.encode()),
        "recovery_index": recovery_index,
        "actor_visible_fields": ["actor_prompt"],
        "classifier_proposal_present": False,
        "projection_complete": True,
    }


def _project(source: Mapping[str, Any]):
    target = _actor_projection(source)
    return target, {
        "target_id": "rule30:actor-branch-classification:w4",
        "discarded_information": [],
        "introduced_information": [
            "label-blind variable-feature pruning",
            "reversible compact feature aliases",
            "42-row transposed truth masks",
        ],
        "inverse_or_recovery_route": "recovery-index-plus-feature-alias-map",
        "expected_observable": "actor-visible exact task without state identifiers or held-out labels",
        "peak_intermediate_size": target["actor_prompt_bytes"],
    }


def _verify_project(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    replay = _actor_projection(source)
    fields = (
        "row_count", "branch_row_count", "feature_count", "feature_aliases",
        "actor_payload", "actor_prompt", "actor_prompt_sha256",
        "actor_prompt_bytes", "recovery_index",
    )
    ok = all(target.get(field) == replay.get(field) for field in fields)
    prompt = str(target.get("actor_prompt") or "")
    ok = ok and "state" not in prompt and target.get("classifier_proposal_present") is False
    return _pass(
        actor_prompt_sha256=target.get("actor_prompt_sha256"),
        actor_prompt_bytes=target.get("actor_prompt_bytes"),
        variable_feature_count=target.get("feature_count"),
    ) if ok else {
        "status": "fail", "exact": True,
        "reason": "actor projection replay or blinding mismatch",
    }


def build_rule30_transform_registry() -> TransformationRegistry:
    registry = TransformationRegistry()
    registry.register(OperatorSpec(
        DUAL_OPERATOR, (SOURCE_TYPE,), DUAL_TYPE,
        ("cyclic-even-width-at-least-4", "rule30-right-history-semantics"),
        (BRANCH_OBLIGATION,), basis_operator="DUALIZE",
        implementation=_dualize, verifier=_verify_dualize,
        precondition_verifier=_dual_preconditions,
        certificate_requirements=("exact reverse-basin replay",),
        cheapest_falsifier="reverse-basin state or edge mismatch",
    ))
    registry.register(OperatorSpec(
        FACTOR_OPERATOR, (DUAL_TYPE,), FACTOR_TYPE,
        ("complete-reverse-zero-tail-basin", "closed-reverse-basin-edge-set"),
        (BRANCH_OBLIGATION,), basis_operator="FACTOR",
        implementation=_factor, verifier=_verify_factor,
        precondition_verifier=_factor_preconditions,
        certificate_requirements=("exact branch partition",),
        cheapest_falsifier="outdegree/branch mismatch",
    ))
    registry.register(OperatorSpec(
        LIFT_OPERATOR, (FACTOR_TYPE,), LIFT_TYPE,
        ("complete-branch-partition", "unique-source-state-identity"),
        (BRANCH_OBLIGATION,), basis_operator="LIFT",
        implementation=_lift, verifier=_verify_lift,
        precondition_verifier=_lift_preconditions,
        certificate_requirements=("feature replay from canonical Rule 30 words",),
        cheapest_falsifier="single feature row replay mismatch",
    ))
    registry.register(OperatorSpec(
        PROJECT_OPERATOR, (LIFT_TYPE,), PROJECT_TYPE,
        ("complete-algebraic-feature-space", "exact-feature-rows", "width-4-training-scope"),
        (BRANCH_OBLIGATION,), basis_operator="PROJECT",
        implementation=_project, verifier=_verify_project,
        precondition_verifier=_project_preconditions,
        certificate_requirements=("actor projection replay", "held-out-label blinding"),
        cheapest_falsifier="prompt mismatch or leaked hidden information",
    ))
    return registry


def compile_rule30_branch_task() -> dict[str, Any]:
    """Execute Run 053's route through the actual generic transform runtime."""
    source = {
        "width": TRAIN_WIDTH,
        "semantics": "rule30-right-history",
        "question": "Classify exact zero-tail-basin branch states from width-agnostic Boolean features without exposing held-out labels.",
        "heldout_widths": list(HELDOUT_WIDTHS),
    }
    target, program = execute_transform_program(
        build_rule30_transform_registry(),
        PROGRAM_OPERATORS,
        source,
        source_id="rule30:forward-history:w4",
        source_type=SOURCE_TYPE,
        required_obligations=(BRANCH_OBLIGATION,),
        provenance={
            "owner": "engine.runtime.experiments.cellular_automata.rule30_transform_program",
            "source_exact_owner": "engine.runtime.experiments.cellular_automata.rule30_077_zero_tail_basin",
            "model_experiment_owner": "dissonance-git/helix-model",
            "historical_model_run": "053-rule30-four-operator-weave",
        },
    )
    return {
        "schema_version": PROGRAM_VERSION,
        "status": "compiled-and-executed-exact-transform-program",
        "task": target,
        "program": program,
        "claim_boundary": [
            "The compiler performs exact representation transport only.",
            "No classifier candidate is generated by Helix.",
            "No held-out width label is computed or exposed by this artifact.",
            "Held-out evaluation remains a separate Helix/Helix Model judgment step.",
            "Finite width-4 compilation does not establish an all-dyadic theorem.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compile_rule30_branch_task()
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "basis_operators": result["program"]["basis_operators"],
        "row_count": result["task"]["row_count"],
        "branch_row_count": result["task"]["branch_row_count"],
        "feature_count": result["task"]["feature_count"],
        "actor_prompt_sha256": result["task"]["actor_prompt_sha256"],
        "actor_prompt_bytes": result["task"]["actor_prompt_bytes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
