from engine.runtime.experiments.cellular_automata.rule30_079_dyadic_derivative_filtration import (
    build_result,
    verify_algebra_width,
    verify_basin_width,
)


def test_dyadic_derivative_filtration_has_fixed_d4_kernel() -> None:
    for width in (4, 8, 16, 32, 64):
        row = verify_algebra_width(width)
        assert row["nullity_D4"] == 4
        assert row["kernel_D4_words"] == 16
        assert row["kernel_D4_equals_period_dividing_4"] is True
        assert row["D_to_width_is_zero"] is True


def test_exact_branching_stays_inside_d4_core_through_width32() -> None:
    expected_states = {4: 42, 8: 74, 16: 2122, 32: 171786}
    for width, states in expected_states.items():
        row = verify_basin_width(width)
        assert row["basin_states"] == states
        assert row["branch_states"] == 10
        assert row["branch_outside_D4_core"] == 0
        assert row["zero_basin_core_escape_edges"] == 0
        assert row["branch_depth_counts"] == {"3": 2, "4": 4, "5": 4}


def test_result_keeps_the_all_dyadic_implications_open() -> None:
    result = build_result()
    assert result["status"] == "PASS"
    assert result["proved_algebraic_reduction"]["four_column_D4_core_tuple_count"] == 65536
    reduction = result["all_dyadic_branch_theorem_reduction"]
    assert reduction["status"] == "open"
    assert "branch state" in reduction["obligation_A"]
    assert "continuation" in reduction["obligation_B"]
