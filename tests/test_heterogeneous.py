from datetime import datetime, timedelta, timezone

import pytest

from lms_optimizer.heterogeneous import (
    ENTRY_LIMIT, TOTAL_ENTRIES, EntryHistory, HeterogeneousCohort,
    HistoricalSelection, STRATEGIES, construct_heterogeneous_cohort,
    evaluate_heterogeneous, validate_heterogeneous_cohort,
)
from lms_optimizer.historical_evaluator import construct_rounds
from lms_optimizer.models import HistoricalMatch
from lms_optimizer.milp import milp_optimize
from lms_optimizer.optimizer import DynamicProgram, PortfolioOptimizer, PortfolioWeights


def match(day, index, home_goals=2, away_goals=0):
    start = datetime(2020, 8, 1, tzinfo=timezone.utc) + timedelta(days=day - 1)
    return HistoricalMatch(season="2020/21", match_date=start, home_team=f"H{index}", away_team=f"A{index}", full_time_home_goals=home_goals, full_time_away_goals=away_goals, closing_home_odds=2, closing_draw_odds=3.5, closing_away_odds=4, data_source="synthetic", collected_at=start, is_sample=True)


def archive():
    return [match(1, i) for i in range(6)] + [match(8, i + 6) for i in range(6)]


def test_two_players_ten_entries_and_valid_winning_histories():
    cohort = construct_heterogeneous_cohort(archive(), "2020/21", 2)
    assert cohort.feasible and len(cohort.entries) == TOTAL_ENTRIES
    assert {entry.player_id for entry in cohort.entries} == {"player-1", "player-2"}
    assert all(len([entry for entry in cohort.entries if entry.player_id == player]) == ENTRY_LIMIT for player in ("player-1", "player-2"))
    assert cohort.validation["valid"] is True
    assert cohort.heterogeneity["distinct_used_team_sets"] >= 2


def test_validator_rejects_losing_reused_future_and_duplicate_entries():
    rows = archive()
    valid_match = rows[0]
    selection = HistoricalSelection(1, f"2020/21|{valid_match.match_date.isoformat()}|{valid_match.home_team}|{valid_match.away_team}", valid_match.home_team, valid_match.match_date.isoformat())
    bad = EntryHistory("duplicate", "player-1", [selection, selection], True)
    bad2 = EntryHistory("duplicate", "player-1", [HistoricalSelection(1, selection.fixture_id, valid_match.away_team, selection.selected_at)], True)
    cohort = HeterogeneousCohort("2020/21", 2, rows[6].match_date.isoformat(), [bad, bad2], True, None, {}, {})
    result = validate_heterogeneous_cohort(cohort, rows)
    codes = {failure["code"] for failure in result["failures"]}
    assert {"duplicate entry identifier", "reused team", "losing historical selection"} <= codes


def test_validator_rejects_future_result_cutoff_and_player_limit():
    rows = archive(); selected = rows[0]
    selection = HistoricalSelection(1, f"2020/21|{selected.match_date.isoformat()}|{selected.home_team}|{selected.away_team}", selected.home_team, (selected.match_date + timedelta(days=10)).isoformat())
    entries = [EntryHistory(f"e{i}", "player-1", [selection], True) for i in range(11)]
    cohort = HeterogeneousCohort("2020/21", 2, rows[1].match_date.isoformat(), entries, True, None, {}, {})
    codes = {failure["code"] for failure in validate_heterogeneous_cohort(cohort, rows)["failures"]}
    assert "selection after the cutoff" in codes and "more than ten entries for one player" in codes


def test_early_point_feasible_late_point_infeasible_and_deterministic():
    rows = archive()
    first = construct_heterogeneous_cohort(rows, "2020/21", 2)
    repeated = construct_heterogeneous_cohort(rows, "2020/21", 2)
    assert first.feasible and [(e.entry_id, [s.team for s in e.selections]) for e in first.entries] == [(e.entry_id, [s.team for s in e.selections]) for e in repeated.entries]
    assert construct_heterogeneous_cohort(rows, "2020/21", 1).feasible is False


def test_all_seven_strategies_and_milp_twenty_entry_evaluation():
    report = evaluate_heterogeneous(archive(), seed=7, bootstrap_repetitions=20)
    assert set(row["strategy"] for row in report["evaluations"]) == set(STRATEGIES)
    assert any(row["feasible"] for row in report["cohort_construction"])
    assert all(item["status"] == "success" for evaluation in report["evaluations"] for item in evaluation["milp"])
    assert all(row["cartel_size"] == TOTAL_ENTRIES for row in report["evaluations"])


def test_expected_survivors_concentrates_and_protect_one_diversifies():
    scenarios = [{"A": True, "B": True}, {"A": True}, {"B": True}, {}]
    probabilities = [.56, .24, .14, .06]
    candidates = {"e1": ["A", "B"], "e2": ["A", "B"]}
    maximum = milp_optimize(candidates, scenarios, PortfolioWeights(expected_survivors=1, at_least_one=0, wipeout=0), scenario_probabilities=probabilities)
    protected = milp_optimize(candidates, scenarios, PortfolioWeights(expected_survivors=0, at_least_one=1, wipeout=0), scenario_probabilities=probabilities)
    assert maximum.feasible and set(maximum.allocation.values()) == {"A"}
    assert protected.feasible and set(protected.allocation.values()) == {"A", "B"}
    assert protected.components["probability_at_least_one"] == pytest.approx(.94)
    assert maximum.components["expected_survivors"] == pytest.approx(1.6)


def test_same_team_correlation_and_concentration_penalty():
    scenarios = [{"A": True, "B": True}, {"A": True}, {"B": True}, {}]
    oracle = PortfolioOptimizer({"e1": ["A", "B"], "e2": ["A", "B"]}, scenarios, PortfolioWeights(expected_survivors=1, at_least_one=0, wipeout=0), scenario_probabilities=[.56, .24, .14, .06])
    assert oracle.components({"e1": "A", "e2": "A"}).expected_survivors == pytest.approx(1.6)
    diversified = PortfolioOptimizer({"e1": ["A", "B"], "e2": ["A", "B"]}, scenarios, PortfolioWeights(expected_survivors=1, at_least_one=0, wipeout=0, concentration=.1), scenario_probabilities=[.56, .24, .14, .06])
    assert set(diversified.optimize()[0].values()) == {"A", "B"}


def test_bellman_preserves_stronger_future_team():
    program = DynamicProgram({1: {"A": .8, "B": .79}, 2: {"A": .1, "B": .95}}, {1: ["A", "B"], 2: ["A", "B"]}, horizon=2)
    # A is used now because Bellman preserves the stronger future B.
    assert program.solve()[0].team == "A"
