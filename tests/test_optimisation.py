import numpy as np
import pytest

from lms_optimizer.optimizer import DynamicProgram, PortfolioOptimizer, PortfolioWeights
from lms_optimizer.milp import milp_optimize
from lms_optimizer.simulation import exact_current_round, formal_cvar, simulate_portfolio


def test_bellman_toy_schedule_and_save_strong_team():
    forecasts = {1: {"A": .9, "B": .8}, 2: {"A": .99, "B": .1}}
    available = {1: ["A", "B"], 2: ["A", "B"]}
    values = DynamicProgram(forecasts, available).solve()
    assert values[0].team == "B"  # A is more valuable in round two.
    assert values[0].dynamic_value == pytest.approx(.8 * .99)


def test_simulation_same_team_is_perfectly_correlated_and_reproducible():
    allocation = {"e1": "A", "e2": "A"}
    summary = simulate_portfolio(allocation, {"f": np.array([.5, .2, .3])}, {"f": ("A", "B")}, simulations=5000, seed=4)
    repeated = simulate_portfolio(allocation, {"f": np.array([.5, .2, .3])}, {"f": ("A", "B")}, simulations=5000, seed=4)
    assert np.array_equal(summary.survivor_counts, repeated.survivor_counts)
    assert set(summary.survivor_counts).issubset({0, 2})


def test_portfolio_exposure_cap_and_cvar():
    scenarios = [{"A": True, "B": False}, {"A": False, "B": True}]
    optimizer = PortfolioOptimizer({"e1": ["A", "B"], "e2": ["A", "B"]}, scenarios, PortfolioWeights(cvar=1), exposure_cap=1)
    allocation, score = optimizer.optimize()
    assert len(allocation) == 2 and len(set(allocation.values())) == 2
    assert score["cvar_eliminated"] == 1


def test_infeasible_portfolio_is_reported():
    with pytest.raises(ValueError, match="no feasible"):
        PortfolioOptimizer({"e1": ["A"], "e2": ["A"]}, [{"A": True}], exposure_cap=1).optimize()


def test_formal_cvar_matches_hand_calculated_discrete_loss():
    assert formal_cvar(np.array([0.0, 1.0, 2.0, 2.0]), alpha=.5) == pytest.approx(2.0)


def test_exact_current_round_distribution_and_wipeout():
    result = exact_current_round({"e1": "A", "e2": "B"}, {"f": np.array([.5, .2, .3])}, {"f": ("A", "B")})
    assert result["expected_survivors"] == pytest.approx(.8)
    assert result["wipeout_probability"] == pytest.approx(.2)
    assert result["probability_at_least_one"] == pytest.approx(.8)


def test_complete_objective_exact_and_milp_agree_with_soft_cap_and_nonuniform_probabilities():
    candidates = {"e1": ["A", "B"], "e2": ["A", "B"]}
    scenarios = [{"A": True}, {"B": True}, {}]
    weights = PortfolioWeights(expected_survivors=1, at_least_one=2, wipeout=3, future_value=1, concentration=.1, cvar=1, soft_cap=.5, cvar_alpha=.5)
    future = {"e1": {"A": .2, "B": .1}, "e2": {"A": .2, "B": .1}}
    kwargs = dict(weights=weights, soft_exposure_cap=1, soft_penalty=.5, scenario_probabilities=[.6, .3, .1], future_values=future)
    exact_allocation, exact_score = PortfolioOptimizer(candidates, scenarios, **kwargs).optimize()
    result = milp_optimize(candidates, scenarios, **kwargs)
    assert result.feasible
    assert result.components["objective"] == pytest.approx(exact_score["objective"], abs=1e-5)
    assert result.components["expected_survivors"] == pytest.approx(exact_score["expected_survivors"])
