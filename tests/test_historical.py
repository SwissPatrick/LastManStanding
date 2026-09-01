from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from lms_optimizer.data import import_historical_csv, import_football_data_csv
from lms_optimizer.backtest import expanding_backtest, market_predictor
from lms_optimizer.elo import EloModel
from lms_optimizer.modeling import DixonColesModel
from lms_optimizer.models import HistoricalMatch


def synthetic_matches(n=30):
    start = datetime(2020, 8, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        home, away = ("A", "B") if i % 2 == 0 else ("B", "A")
        rows.append(HistoricalMatch(season="2020/21", match_date=start + timedelta(days=i), home_team=home, away_team=away, full_time_home_goals=2 if home == "A" else 0, full_time_away_goals=0 if home == "A" else 1, closing_home_odds=2.0, closing_draw_odds=3.5, closing_away_odds=4.0, data_source="synthetic", collected_at=start, is_sample=True))
    return rows


def test_dixon_coles_probability_matrix_is_normalized():
    model = DixonColesModel(decay_rate=0.01).fit(synthetic_matches())
    prediction = model.predict("A", "B")
    assert np.all(np.isfinite(prediction.scoreline))
    assert np.all(prediction.scoreline >= 0)
    assert prediction.scoreline.sum() == pytest.approx(1.0, abs=1e-12)
    assert prediction.outcome.sum() == pytest.approx(1.0, abs=1e-12)


def test_elo_is_strictly_chronological_and_as_of_excludes_future():
    matches = synthetic_matches()
    full = EloModel().fit(matches)
    partial = EloModel().fit(matches, as_of=matches[10].match_date)
    assert len(partial.snapshots) == 10
    assert full.ratings != partial.ratings


def test_parameter_recovery_direction_and_expanding_backtest_no_leakage():
    matches = synthetic_matches(35)
    model = DixonColesModel().fit(matches)
    assert model.predict("A", "B").home_goals > model.predict("B", "A").home_goals
    seen_lengths = []
    def predictor(train, test):
        seen_lengths.append((len(train), max(m.match_date for m in train), test.match_date))
        return market_predictor(train, test)
    result = expanding_backtest(matches, predictor, min_train=10)
    assert result.metrics["log_loss"] >= 0
    assert all(length == index + 10 and train_date < test_date for index, (length, train_date, test_date) in enumerate(seen_lengths))


def test_csv_import_normalizes_duplicates_and_preserves_raw(tmp_path):
    frame = pd.DataFrame([
        {"season": "2020/21", "date": "2020-08-01", "home_team": "Man Utd", "away_team": "Spurs", "home_goals": 1, "away_goals": 0, "home_odds": 2, "draw_odds": 3.5, "away_odds": 4},
        {"season": "2020/21", "date": "2020-08-01", "home_team": "Man Utd", "away_team": "Spurs", "home_goals": 1, "away_goals": 0, "home_odds": 2, "draw_odds": 3.5, "away_odds": 4},
    ])
    source = tmp_path / "history.csv"
    frame.to_csv(source, index=False)
    report = import_historical_csv(source, tmp_path / "raw")
    assert len(report.matches) == 1
    assert report.matches[0].home_team == "Manchester United"
    assert len(report.duplicate_rows) == 1
    assert (tmp_path / "raw" / "history.csv").exists()


def test_blank_trailing_football_data_row_is_reported_not_imputed(tmp_path):
    frame = pd.DataFrame([{"Date": "01/08/20", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 0, "AvgH": 2, "AvgD": 3.5, "AvgA": 4}, {}])
    source = tmp_path / "2014-15_E0.csv"; frame.to_csv(source, index=False)
    report = import_football_data_csv(source, "2014/15", tmp_path / "raw")
    assert len(report.matches) == 1
    assert len(report.missing_rows) == 1
    assert report.missing_rows[0]["field"] == "result"
