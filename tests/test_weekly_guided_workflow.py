from datetime import datetime, timedelta, timezone

import pytest

from lms_optimizer.models import Entry, Fixture, FixtureStatus, OddsQuote, Player, Round, Season
from lms_optimizer.storage import Repository
from lms_optimizer.workflow import LMSWorkflow


def setup_service(tmp_path):
    service = LMSWorkflow(Repository(tmp_path / "lms.sqlite3"))
    now = datetime.now(timezone.utc)
    service.create_season(Season(season="2026/27", name="Test season"))
    service.create_round(Round(season="2026/27", round_number=1, selection_deadline=now + timedelta(days=1)))
    service.add_player(Player(player_id="p1", name="Player One"))
    service.add_entry(Entry(entry_id="e1", player="p1", season="2026/27"))
    fixtures = [Fixture(fixture_id=f"f{i}", season="2026/27", round_number=1, home_team=f"H{i}", away_team=f"A{i}", kickoff=now + timedelta(days=2, minutes=i), collected_at=now) for i in range(6)]
    service.add_fixtures(fixtures)
    service.add_odds([OddsQuote(fixture_id=f.fixture_id, bookmaker="manual", home=1.5, draw=4, away=6, collected_at=now, market_timestamp=now) for f in fixtures])
    return service


def test_guided_validation_and_exact_analysis(tmp_path):
    service = setup_service(tmp_path)
    gate = service.validate_round("2026/27", 1)
    assert gate["valid"] and gate["six_match_rule"]
    analysis = service.analyse_round("2026/27", 1)
    assert analysis["allocation"]["e1"] == "H0"
    assert float(analysis["risk"]["probabilities"].sum()) == pytest.approx(1)


def test_guided_missing_odds_blocks_analysis(tmp_path):
    service = setup_service(tmp_path)
    service.repo.connection.execute("DELETE FROM odds_quotes WHERE fixture_id = ?", ("f0",)); service.repo.connection.commit()
    gate = service.validate_round("2026/27", 1)
    assert not gate["valid"] and not gate["odds_complete"]
    with pytest.raises(ValueError): service.analyse_round("2026/27", 1)


def test_guided_lock_result_and_survivor_advancement(tmp_path):
    service = setup_service(tmp_path)
    analysis = service.analyse_round("2026/27", 1)
    service.save_recommendation_selections(analysis["allocation"], analysis["backups"], 1)
    assert service.repo.count("selections") >= 1
    result = service.record_results_and_advance("f0", FixtureStatus.PLAYED, 2, 0)
    assert result["e1"] == "surviving"
