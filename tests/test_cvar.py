import numpy as np
import pytest

from lms_optimizer.cvar import formal_cvar
from lms_optimizer.historical_evaluator import (
    _DECISION_CVAR_CACHE, _aggregate_cvar, _exact_decision_cvar,
    clustered_cvar_bootstrap, evaluate_cohort,
)
from lms_optimizer.simulation import exact_current_round


def test_formal_cvar_uniform_discrete_and_threshold_mass():
    result = formal_cvar([0, 1, 2, 3], alpha=.5)
    assert result.var_threshold == 1 and result.cvar == pytest.approx(2.5)
    result = formal_cvar([0, 1, 1, 4], alpha=.75)
    assert result.var_threshold == 1 and result.cvar == pytest.approx(4)


def test_formal_cvar_weights_zero_and_complete_loss():
    assert formal_cvar([0, 10], [.9, .1], .95).cvar == pytest.approx(10)
    assert formal_cvar([0, 0], alpha=.95).cvar == 0
    assert formal_cvar([3, 3], alpha=.95).cvar == 3
    assert formal_cvar([0, 10], [.9, .1], .95).expected_loss == pytest.approx(1)
    assert formal_cvar([0, 10], [.9, .1], .95, "eliminated-entry fraction").loss_definition == "eliminated-entry fraction"


@pytest.mark.parametrize("losses, weights, alpha", [([1], None, 1), ([1], [-1], .95), ([1], [0], .95), ([np.nan], None, .95), ([np.inf], None, .95)])
def test_formal_cvar_rejects_invalid_inputs(losses, weights, alpha):
    with pytest.raises(ValueError):
        formal_cvar(losses, weights, alpha)


def test_exact_scenarios_preserve_shared_and_opposing_team_correlation():
    probabilities = {"f": np.array([.5, .2, .3])}
    teams = {"f": ("A", "B")}
    shared = exact_current_round({"e1": "A", "e2": "A"}, probabilities, teams)
    opposing = exact_current_round({"e1": "A", "e2": "B"}, probabilities, teams)
    assert dict(zip(shared["survivor_counts"], shared["probabilities"])) == {0: .5, 2: .5}
    assert dict(zip(opposing["survivor_counts"], opposing["probabilities"])) == {0: .2, 1: .8}


def test_predicted_cvar_is_pre_result_and_cache_is_equivalent():
    from tests.test_historical_evaluator import match
    rounds = [[match(1, 0, home="A", away="B"), match(1, 1, home="C", away="D")]]
    _DECISION_CVAR_CACHE.clear()
    first = _exact_decision_cvar(rounds[0], {"e1": "A", "e2": "D"}, ["e1", "e2"])
    second = _exact_decision_cvar(rounds[0], {"e1": "A", "e2": "D"}, ["e1", "e2"])
    assert first == second and first["scenario_count"] == 9 and first["scenario_probability_total"] == pytest.approx(1)
    decisions, metrics, _ = evaluate_cohort(rounds, 0, 2, "independent_greedy")
    assert decisions[0]["information_cutoff"] == decisions[0]["cutoff"]
    assert "cvar_eliminated" not in decisions[0] and "cvar_eliminated" not in metrics[0]


def test_realised_aggregate_cvar_and_clustered_bootstrap_require_comparable_data():
    rows = [{"season": season, "strategy": "s", "cartel_size": 2, "realised_loss": loss, "active_entries_before": 2} for season, loss in [("a", 0), ("a", 2), ("b", 1), ("b", 2)]]
    result = _aggregate_cvar(rows, .5, 50, 7)
    assert len(result) == 2 and all(summary["observation_count"] == 2 for summary in result.values())
    assert clustered_cvar_bootstrap(rows, repetitions=50, seed=7)[0] <= clustered_cvar_bootstrap(rows, repetitions=50, seed=7)[1]
    with pytest.raises(ValueError, match="two comparable"):
        _aggregate_cvar(rows[:1], .95, 10, 7)
