from __future__ import annotations

import hashlib

from engine.runtime.experiments.cellular_automata.rule30_transform_program import (
    BASIS_OPERATORS,
    BRANCH_OBLIGATION,
    PROGRAM_OPERATORS,
    PROJECT_TYPE,
    MECHANISM_BASIS_OPERATORS,
    MECHANISM_PROGRAM_OPERATORS,
    MECHANISM_TASK_BASIS_OPERATORS,
    MECHANISM_TASK_PROGRAM_OPERATORS,
    build_rule30_transform_registry,
    compile_rule30_branch_mechanism,
    compile_rule30_branch_mechanism_task,
    compile_rule30_branch_task,
)

RUN053_PROMPT_SHA256 = "63da596d6a0510ec2f845849c30bc482c78b404afd2ea963a52c1e486c2eb3a4"


def test_rule30_compiler_route_executes_through_first_class_ir_runtime() -> None:
    result = compile_rule30_branch_task()
    program = result["program"]
    task = result["task"]

    assert result["status"] == "compiled-and-executed-exact-transform-program"
    assert tuple(program["operators"]) == PROGRAM_OPERATORS
    assert tuple(program["basis_operators"]) == BASIS_OPERATORS
    assert program["required_obligations"] == [BRANCH_OBLIGATION]
    assert program["target_type"] == PROJECT_TYPE
    assert program["execution_owner"] == "engine.ir.passes.execute_transform_program"
    assert program["contains_non_exact_step"] is False
    assert len(program["steps"]) == 4
    assert all(step["verification_state"]["status"] == "pass" for step in program["steps"])
    assert all(step["verification_state"]["exact"] is True for step in program["steps"])
    assert all(step["inverse_or_recovery_route"] for step in program["steps"])
    assert program["recovery_route"] is not None

    assert task["row_count"] == 42
    assert task["branch_row_count"] == 10
    assert task["feature_count"] == 70
    assert task["heldout_widths"] == [8, 16, 32]
    assert task["heldout_labels_visible_to_actor"] is False
    assert task["classifier_proposal_present"] is False


def test_rule30_compiler_projection_is_byte_identical_to_run053_actor_task() -> None:
    task = compile_rule30_branch_task()["task"]
    prompt = task["actor_prompt"]

    assert task["actor_prompt_bytes"] == 2054
    assert task["actor_prompt_sha256"] == RUN053_PROMPT_SHA256
    assert hashlib.sha256(prompt.encode()).hexdigest() == RUN053_PROMPT_SHA256
    assert task["actor_payload"]["route"] == "DUALIZE>FACTOR>LIFT>PROJECT"
    assert len(task["feature_aliases"]) == 70
    assert len(task["actor_payload"]["truth"]) == 70
    assert '"state":' not in prompt
    assert '"recovery_index":' not in prompt


def test_rule30_compiler_registry_exposes_honest_basis_bindings() -> None:
    by_name = {
        row["operator"]: row
        for row in build_rule30_transform_registry().discover()
    }

    assert tuple(by_name[name]["basis_operator"] for name in PROGRAM_OPERATORS) == BASIS_OPERATORS
    assert all(by_name[name]["implemented"] for name in PROGRAM_OPERATORS)
    assert all(by_name[name]["guarded_execution_required"] for name in PROGRAM_OPERATORS)
    assert all(by_name[name]["precondition_verifier_available"] for name in PROGRAM_OPERATORS)


def test_rule30_branch_mechanism_reprojection_is_stable_through_width32() -> None:
    expected_states = {4: 42, 8: 74, 16: 2122, 32: 171786}
    for width, state_count in expected_states.items():
        result = compile_rule30_branch_mechanism(width)
        mechanism = result["mechanism"]
        program = result["program"]

        assert result["status"] == "compiled-and-verified-branch-mechanism"
        assert tuple(program["operators"]) == MECHANISM_PROGRAM_OPERATORS
        assert tuple(program["basis_operators"]) == MECHANISM_BASIS_OPERATORS
        assert all(step["verification_state"]["status"] == "pass" for step in program["steps"])
        assert mechanism["state_count"] == state_count
        assert mechanism["branch_count"] == 10
        assert mechanism["exceptional_pair_count"] == 2
        assert mechanism["derivative_chain_count"] == 8
        assert mechanism["overlap_count"] == 0
        assert mechanism["false_positive_count"] == 0
        assert mechanism["false_negative_count"] == 0
        assert mechanism["mechanism_complete"] is True
        assert mechanism["mechanism_disjoint"] is True
        assert mechanism["branch_depth_counts"] == {"3": 2, "4": 4, "5": 4}
        assert mechanism["maximum_branch_component_period"] == 4


def test_rule30_mechanism_actor_task_is_compiler_generated_and_hidden_width_blind() -> None:
    result = compile_rule30_branch_mechanism_task()
    task = result["task"]
    program = result["program"]

    assert result["status"] == "compiled-mechanism-actor-task"
    assert tuple(program["operators"]) == MECHANISM_TASK_PROGRAM_OPERATORS
    assert tuple(program["basis_operators"]) == MECHANISM_TASK_BASIS_OPERATORS
    assert all(step["verification_state"]["status"] == "pass" for step in program["steps"])
    assert task["heldout_widths"] == [8, 16, 32]
    assert task["heldout_labels_visible_to_actor"] is False
    assert task["training_patterns"] == [
        {"E": False, "S": False, "branch": False, "count": 32},
        {"E": False, "S": True, "branch": True, "count": 8},
        {"E": True, "S": False, "branch": True, "count": 2},
    ]
    assert len(task["training_patterns"]) == 3
    assert '"state":' not in task["actor_prompt"]
    assert '"rows":' not in task["actor_prompt"]
    assert task["actor_prompt_bytes"] < 1200
