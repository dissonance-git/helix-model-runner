from engine.runtime.experiments.cellular_automata.rule30_080_derivative_basin_filtration import (
    build_result,
)


def test_derivative_basin_filtration_exact_lift_and_functional_quotient() -> None:
    result = build_result()
    assert result["status"] == "PASS"

    expected_basin_states = {1: 12, 2: 18, 4: 42, 8: 74, 16: 2122, 32: 171786}
    expected_branches = {1: 4, 2: 6, 4: 10, 8: 10, 16: 10, 32: 10}

    for row in result["rows"]:
        width = row["width"]
        assert row["basin_states"] == expected_basin_states[width]
        assert row["branch_states"] == expected_branches[width]
        for scale in row["scales"]:
            k = scale["scale"]
            assert scale["zero_fiber_states"] == expected_basin_states[k]
            assert scale["lifted_width_k_basin_states"] == expected_basin_states[k]
            assert scale["zero_fiber_equals_lifted_width_k_basin"] is True
            assert scale["projected_continuation_outdegree_values"] == [1]
            assert scale["projected_graph_functional"] is True
            assert scale["branch_states_inside_zero_fiber"] == expected_branches[k]

    assert result["branch_scale_ladder"] == {
        "width_1": 4,
        "width_2": 6,
        "width_4": 10,
        "width_8": 10,
        "width_16": 10,
        "width_32": 10,
        "new_branch_species_after_scale_4_in_tested_widths": 0,
    }
