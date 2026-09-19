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
MECHANISM_TYPE = "branch-mechanism-coordinates"
MECHANISM_TASK_TYPE = "two-coordinate-boolean-classification-task"

DUAL_OPERATOR = "rule30-dualize-zero-tail"
FACTOR_OPERATOR = "rule30-factor-branching"
LIFT_OPERATOR = "rule30-lift-algebraic-features"
PROJECT_OPERATOR = "rule30-project-actor-task"
REPROJECT_OPERATOR = "rule30-reproject-branch-mechanism"
MECHANISM_PROJECT_OPERATOR = "rule30-project-mechanism-task"
PROGRAM_OPERATORS = (DUAL_OPERATOR, FACTOR_OPERATOR, LIFT_OPERATOR, PROJECT_OPERATOR)
BASIS_OPERATORS = ("DUALIZE", "FACTOR", "LIFT", "PROJECT")
MECHANISM_PROGRAM_OPERATORS = (DUAL_OPERATOR, FACTOR_OPERATOR, LIFT_OPERATOR, REPROJECT_OPERATOR)
MECHANISM_BASIS_OPERATORS = ("DUALIZE", "FACTOR", "LIFT", "REPROJECT")
MECHANISM_TASK_PROGRAM_OPERATORS = (*MECHANISM_PROGRAM_OPERATORS, MECHANISM_PROJECT_OPERATOR)
MECHANISM_TASK_BASIS_OPERATORS = (*MECHANISM_BASIS_OPERATORS, "PROJECT")


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
        "reverse_depth": int(source["depth_by_state"][state]),
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
        "reverse_depth": int(row["reverse_depth"]),
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


def _reproject_preconditions(source: Mapping[str, Any]) -> dict[str, Any]:
    required_features = {"D(a)=c", "b=d", "D(b)=c", "D(c)=d"}
    rows = tuple(source.get("rows") or ())
    verified = []
    if source.get("feature_space_complete") is True:
        verified.append("complete-algebraic-feature-space")
    if rows and all(required_features.issubset(set(row.get("features") or {})) for row in rows):
        verified.append("branch-mechanism-literals-available")
    return {
        "status": "pass" if len(verified) == 2 else "fail",
        "applicable": len(verified) == 2,
        "exact": True,
        "verified_preconditions": verified,
    }


def _branch_mechanism_view(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for row in source["rows"]:
        features = row["features"]
        exceptional = (not bool(features["D(a)=c"])) and bool(features["b=d"])
        derivative_chain = bool(features["D(b)=c"]) and (not bool(features["D(c)=d"]))
        rows.append({
            "state": row["state"],
            "reverse_depth": int(row["reverse_depth"]),
            "is_branch": bool(row["is_branch"]),
            "exceptional_pair": exceptional,
            "derivative_chain": derivative_chain,
            "branch_by_mechanism": exceptional or derivative_chain,
        })
    branch_rows = [row for row in rows if row["is_branch"]]
    exceptional_rows = [row for row in rows if row["exceptional_pair"]]
    derivative_rows = [row for row in rows if row["derivative_chain"]]
    overlap_rows = [row for row in rows if row["exceptional_pair"] and row["derivative_chain"]]
    false_positive = [row for row in rows if row["branch_by_mechanism"] and not row["is_branch"]]
    false_negative = [row for row in rows if row["is_branch"] and not row["branch_by_mechanism"]]
    depth_counts = Counter(row["reverse_depth"] for row in branch_rows)
    max_branch_period = max(
        (
            max(word_period(word, int(source["width"])) for word in row["state"])
            for row in branch_rows
        ),
        default=0,
    )
    return {
        "width": int(source["width"]),
        "state_count": len(rows),
        "branch_count": len(branch_rows),
        "exceptional_pair_count": len(exceptional_rows),
        "derivative_chain_count": len(derivative_rows),
        "overlap_count": len(overlap_rows),
        "false_positive_count": len(false_positive),
        "false_negative_count": len(false_negative),
        "branch_depth_counts": {str(depth): int(count) for depth, count in sorted(depth_counts.items())},
        "maximum_reverse_depth": max((row["reverse_depth"] for row in rows), default=0),
        "maximum_branch_component_period": max_branch_period,
        "rows": tuple(rows),
        "mechanism_complete": not false_positive and not false_negative,
        "mechanism_disjoint": not overlap_rows,
    }


def _reproject(source: Mapping[str, Any]):
    target = _branch_mechanism_view(source)
    return target, {
        "target_id": f"rule30:branch-mechanism:w{source['width']}",
        "discarded_information": [
            "feature columns outside D(a)=c, b=d, D(b)=c, and D(c)=d",
        ],
        "introduced_information": [
            "exceptional-pair coordinate",
            "derivative-chain coordinate",
            "exact branch-equivalence diagnostic",
        ],
        "inverse_or_recovery_route": "recompute-complete-feature-space-from-retained-state-identity",
        "expected_observable": (
            "within the exact zero-tail basin, branch iff exceptional_pair OR derivative_chain"
        ),
        "peak_intermediate_size": len(target["rows"]),
    }


def _verify_reproject(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    replay = _branch_mechanism_view(source)
    exact_fields = (
        "width", "state_count", "branch_count", "exceptional_pair_count",
        "derivative_chain_count", "overlap_count", "false_positive_count",
        "false_negative_count", "branch_depth_counts", "maximum_reverse_depth",
        "maximum_branch_component_period", "rows", "mechanism_complete",
        "mechanism_disjoint",
    )
    ok = all(target.get(field) == replay.get(field) for field in exact_fields)
    ok = ok and target.get("mechanism_complete") is True and target.get("mechanism_disjoint") is True
    return _pass(
        branch_count=target.get("branch_count"),
        exceptional_pair_count=target.get("exceptional_pair_count"),
        derivative_chain_count=target.get("derivative_chain_count"),
        branch_depth_counts=target.get("branch_depth_counts"),
        maximum_branch_component_period=target.get("maximum_branch_component_period"),
    ) if ok else {
        "status": "fail",
        "exact": True,
        "reason": "branch-mechanism reprojection does not exactly classify the source basin",
    }


def _mechanism_project_preconditions(source: Mapping[str, Any]) -> dict[str, Any]:
    verified = []
    if source.get("mechanism_complete") is True and source.get("mechanism_disjoint") is True:
        verified.append("exact-disjoint-branch-mechanism")
    if int(source.get("width") or 0) == TRAIN_WIDTH:
        verified.append("width-4-training-scope")
    return {
        "status": "pass" if len(verified) == 2 else "fail",
        "applicable": len(verified) == 2,
        "exact": True,
        "verified_preconditions": verified,
    }


def _mechanism_actor_projection(source: Mapping[str, Any]) -> dict[str, Any]:
    counts = Counter(
        (
            bool(row["exceptional_pair"]),
            bool(row["derivative_chain"]),
            bool(row["is_branch"]),
        )
        for row in source["rows"]
    )
    training = [
        {
            "E": exceptional,
            "S": derivative_chain,
            "branch": branch,
            "count": int(count),
        }
        for (exceptional, derivative_chain, branch), count in sorted(counts.items())
    ]
    payload = {
        "task": (
            "Infer the shortest exact Boolean rule for branch from the two compiler "
            "coordinates using only width-4 training rows. Hidden widths 8/16/32 "
            "are tested unchanged after your answer is frozen."
        ),
        "coordinates": {
            "E": "D(a)!=c AND b=d",
            "S": "D(b)=c AND D(c)!=d",
        },
        "training": training,
        "output": {
            "formula": {
                "op": "or|and|not|var",
                "args": "for or/and: two expressions; for not: one expression; for var: one of E,S",
            }
        },
        "rules": (
            "JSON only. Use only E,S and op values or,and,not,var. "
            "Return one shortest formula. Do not claim an all-width theorem."
        ),
    }
    prompt = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return {
        "width": TRAIN_WIDTH,
        "heldout_widths": list(HELDOUT_WIDTHS),
        "heldout_labels_visible_to_actor": False,
        "training_patterns": training,
        "actor_payload": payload,
        "actor_prompt": prompt,
        "actor_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "actor_prompt_bytes": len(prompt.encode()),
        "actor_visible_fields": ["actor_prompt"],
        "projection_complete": True,
    }


def _mechanism_project(source: Mapping[str, Any]):
    target = _mechanism_actor_projection(source)
    return target, {
        "target_id": "rule30:mechanism-actor-task:w4",
        "discarded_information": [
            "individual state identities",
            "reverse depths",
            "full 70-feature rows",
        ],
        "introduced_information": [
            "count-compressed two-coordinate truth table",
        ],
        "inverse_or_recovery_route": "recompute-mechanism-view-from-canonical-width-4-basin",
        "expected_observable": "smallest exact Boolean rule over E and S",
        "peak_intermediate_size": target["actor_prompt_bytes"],
    }


def _verify_mechanism_project(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    replay = _mechanism_actor_projection(source)
    fields = (
        "width", "heldout_widths", "heldout_labels_visible_to_actor",
        "training_patterns", "actor_payload", "actor_prompt",
        "actor_prompt_sha256", "actor_prompt_bytes",
    )
    ok = all(target.get(field) == replay.get(field) for field in fields)
    payload = target.get("actor_payload")
    actor_safe = isinstance(payload, Mapping) and "rows" not in payload and "state" not in payload
    return _pass(
        actor_prompt_sha256=target.get("actor_prompt_sha256"),
        actor_prompt_bytes=target.get("actor_prompt_bytes"),
        training_pattern_count=len(target.get("training_patterns") or ()),
    ) if ok and actor_safe else {
        "status": "fail",
        "exact": True,
        "reason": "mechanism actor projection replay or blinding mismatch",
    }


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
    actor_payload = target.get("actor_payload")
    actor_safe = (
        isinstance(actor_payload, Mapping)
        and "state" not in actor_payload
        and "recovery_index" not in actor_payload
    )
    ok = ok and actor_safe and target.get("classifier_proposal_present") is False
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
        REPROJECT_OPERATOR, (LIFT_TYPE,), MECHANISM_TYPE,
        ("complete-algebraic-feature-space", "branch-mechanism-literals-available"),
        (BRANCH_OBLIGATION,), basis_operator="REPROJECT",
        implementation=_reproject, verifier=_verify_reproject,
        precondition_verifier=_reproject_preconditions,
        certificate_requirements=("exact branch equivalence", "disjoint mechanism partition"),
        cheapest_falsifier="one basin state misclassified by the two mechanism coordinates",
    ))
    registry.register(OperatorSpec(
        MECHANISM_PROJECT_OPERATOR, (MECHANISM_TYPE,), MECHANISM_TASK_TYPE,
        ("exact-disjoint-branch-mechanism", "width-4-training-scope"),
        (BRANCH_OBLIGATION,), basis_operator="PROJECT",
        implementation=_mechanism_project, verifier=_verify_mechanism_project,
        precondition_verifier=_mechanism_project_preconditions,
        certificate_requirements=("count-compressed projection replay", "held-out-label blinding"),
        cheapest_falsifier="actor task differs from exact two-coordinate width-4 truth table",
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


def compile_rule30_branch_mechanism(width: int) -> dict[str, Any]:
    """Compile the exact basin into the two-coordinate branch mechanism view."""
    if width < 4 or width % 2:
        raise ValueError("width must be an even integer at least 4")
    source = {
        "width": int(width),
        "semantics": "rule30-right-history",
        "question": "Why is every exact zero-tail-basin fork confined to the small branch core?",
    }
    target, program = execute_transform_program(
        build_rule30_transform_registry(),
        MECHANISM_PROGRAM_OPERATORS,
        source,
        source_id=f"rule30:forward-history:w{width}",
        source_type=SOURCE_TYPE,
        required_obligations=(BRANCH_OBLIGATION,),
        provenance={
            "owner": "engine.runtime.experiments.cellular_automata.rule30_transform_program",
            "source_exact_owner": "engine.runtime.experiments.cellular_automata.rule30_077_zero_tail_basin",
            "finite_candidate_source": "dissonance-git/helix-model:runs/053-rule30-four-operator-weave/Result.json",
        },
    )
    return {
        "schema_version": PROGRAM_VERSION,
        "status": "compiled-and-verified-branch-mechanism",
        "mechanism": target,
        "program": program,
        "claim_boundary": [
            "The two-coordinate branch identity is established only for the exact finite width executed.",
            "Matching results at several widths do not establish the all-dyadic theorem.",
            "The reprojection is fail-closed: a single false positive, false negative, or overlap rejects the exact transform.",
        ],
    }


def compile_rule30_branch_mechanism_task() -> dict[str, Any]:
    """Compile the exact width-4 basin all the way to the two-coordinate actor task."""
    source = {
        "width": TRAIN_WIDTH,
        "semantics": "rule30-right-history",
        "question": "Can a compressed branch-mechanism view expose the finite invariant to a frozen proposal actor?",
        "heldout_widths": list(HELDOUT_WIDTHS),
    }
    target, program = execute_transform_program(
        build_rule30_transform_registry(),
        MECHANISM_TASK_PROGRAM_OPERATORS,
        source,
        source_id="rule30:forward-history:w4",
        source_type=SOURCE_TYPE,
        required_obligations=(BRANCH_OBLIGATION,),
        provenance={
            "owner": "engine.runtime.experiments.cellular_automata.rule30_transform_program",
            "comparison_run": "dissonance-git/helix-model:runs/053-rule30-four-operator-weave",
            "next_model_run": "dissonance-git/helix-model:runs/054-rule30-compiler-reprojection",
        },
    )
    return {
        "schema_version": PROGRAM_VERSION,
        "status": "compiled-mechanism-actor-task",
        "task": target,
        "program": program,
        "claim_boundary": [
            "The actor receives only the count-compressed width-4 E/S truth table.",
            "Widths 8, 16, and 32 remain hidden until the actor answer is frozen.",
            "The E/S coordinates were retained from prior exact Run 053 control evidence; this run tests representation access, not independent discovery of those coordinates.",
            "Finite transfer does not establish an all-dyadic theorem.",
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
