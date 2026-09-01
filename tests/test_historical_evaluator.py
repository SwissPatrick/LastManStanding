from datetime import datetime, timedelta, timezone
import numpy as np
from lms_optimizer.historical_evaluator import construct_rounds, clustered_bootstrap, clustered_paired_bootstrap, evaluate_cohort
from lms_optimizer.models import HistoricalMatch

def match(day, i, home=None, away=None, home_goals=1, away_goals=0):
    return HistoricalMatch(season="2020/21", match_date=datetime(2020,8,day,tzinfo=timezone.utc), home_team=home or f"H{i}", away_team=away or f"A{i}", full_time_home_goals=home_goals, full_time_away_goals=away_goals, closing_home_odds=2, closing_draw_odds=3.5, closing_away_odds=4, data_source="synthetic", collected_at=datetime.now(timezone.utc), is_sample=True)

def test_round_constructor_prevents_team_duplicates_and_keeps_six_match_rounds():
    rows = [match(1,i) for i in range(6)] + [match(2,6,"H0","New")]
    rounds, audits = construct_rounds(rows, "2020/21")
    assert len(rounds) == 1 and len(rounds[0]) == 6
    assert audits[0].eligible and audits[1].match_count == 1

def test_midweek_and_postponed_rescheduled_window_is_audited():
    rows = [match(1,i) for i in range(6)] + [match(5,i+6) for i in range(6)]
    rounds, audits = construct_rounds(rows, "2020/21")
    assert len(rounds) == 2
    assert any("multi-day" in warning for audit in audits for warning in audit.warnings)

def test_clustered_bootstrap_is_reproducible_by_season():
    rows = [{"season":"a","metric":1.0},{"season":"a","metric":3.0},{"season":"b","metric":5.0}]
    assert clustered_bootstrap(rows,"metric",1000,7) == clustered_bootstrap(rows,"metric",1000,7)

def test_all_strategy_interfaces_and_first_round_elimination_count_is_zero():
    rows = [match(1, i, home_goals=0, away_goals=1) for i in range(6)] + [match(8, i + 10) for i in range(6)]
    rounds, _ = construct_rounds(rows, "2020/21")
    strategies = ("concentrated_favourite", "equal_diversification", "independent_greedy", "bellman", "max_expected_survivors", "protect_one", "balanced")
    for strategy in strategies:
        decisions, metrics, survival = evaluate_cohort(rounds, 0, 1, strategy, all_matches=rows)
        assert decisions and len(metrics) == 1
        assert survival["e1"] == 0
        assert metrics[0]["surviving_after"] == 0

def test_paired_bootstrap_keeps_strategy_comparisons_clustered_by_season():
    rows = [
        {"season": "a", "start_round": 1, "cartel_size": 1, "strategy": "concentrated_favourite", "metric": 1.0},
        {"season": "a", "start_round": 1, "cartel_size": 1, "strategy": "bellman", "metric": 2.0},
        {"season": "b", "start_round": 1, "cartel_size": 1, "strategy": "concentrated_favourite", "metric": 3.0},
        {"season": "b", "start_round": 1, "cartel_size": 1, "strategy": "bellman", "metric": 4.0},
    ]
    assert clustered_paired_bootstrap(rows, "metric", "bellman", repetitions=1000, seed=11) == (1.0, 1.0)
