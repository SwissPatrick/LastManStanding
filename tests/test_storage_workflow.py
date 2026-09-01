from datetime import datetime, timedelta, timezone

import pytest

from lms_optimizer.models import Entry, Fixture, FixtureStatus, OddsQuote, Player, Round, Season
from lms_optimizer.storage import Repository
from lms_optimizer.workflow import LMSWorkflow


def make_workflow(tmp_path):
    repo = Repository(tmp_path / "lms.sqlite3")
    workflow = LMSWorkflow(repo)
    now = datetime.now(timezone.utc)
    workflow.create_season(Season(season="2026/27", name="Test season", is_sample=True))
    workflow.create_round(Round(season="2026/27", round_number=1, selection_deadline=now + timedelta(days=1), is_sample=True))
    fixtures = [Fixture(fixture_id=str(i), season="2026/27", round_number=1, home_team=f"Home{i}", away_team=f"Away{i}", kickoff=now + timedelta(days=2), collected_at=now, is_sample=True) for i in range(6)]
    workflow.add_fixtures(fixtures)
    return repo, workflow, fixtures, now


def test_database_persists_domain_records_and_audit(tmp_path):
    repo, workflow, fixtures, now = make_workflow(tmp_path)
    workflow.add_player(Player(player_id="p1", name="Alex", is_sample=True))
    workflow.add_entry(Entry(entry_id="e1", player="p1", season="2026/27", is_sample=True))
    workflow.add_odds([OddsQuote(fixture_id="0", bookmaker="Book A", home=2.0, draw=3.5, away=4.0, collected_at=now, market_timestamp=now, is_sample=True)])
    assert repo.count("seasons") == 1
    assert repo.count("rounds") == 1
    assert repo.count("fixtures") == 6
    assert repo.count("players") == 1
    assert repo.count("entries") == 1
    assert repo.count("odds_quotes") == 1
    assert repo.count("audit_log") >= 10
    repo.close()


def test_entry_limit_is_ten(tmp_path):
    repo, workflow, _, _ = make_workflow(tmp_path)
    workflow.add_player(Player(player_id="p1", name="Alex"))
    for index in range(10):
        workflow.add_entry(Entry(entry_id=f"e{index}", player="p1", season="2026/27"))
    with pytest.raises(ValueError, match="at most ten"):
        workflow.add_entry(Entry(entry_id="e10", player="p1", season="2026/27"))


def test_selection_reuse_and_fallback_preview(tmp_path):
    repo, workflow, _, now = make_workflow(tmp_path)
    workflow.add_player(Player(player_id="p1", name="Alex"))
    workflow.add_entry(Entry(entry_id="e1", player="p1", season="2026/27"))
    workflow.add_odds([OddsQuote(fixture_id=str(i), bookmaker="Book A", home=2.0 + i / 10, draw=3.5, away=4.0, collected_at=now, market_timestamp=now) for i in range(6)])
    workflow.record_selection("e1", 1, "Home0", selected_at=now)
    with pytest.raises(ValueError, match="already been used"):
        workflow.record_selection("e1", 1, "Home0", selected_at=now)
    fallback = workflow.fallback_preview("e1", 1)
    assert fallback in workflow.available_teams("e1", 1)
    assert repo.count("selections") == 1


def test_played_result_eliminates_and_postponed_fixture_stays_pending(tmp_path):
    repo, workflow, fixtures, now = make_workflow(tmp_path)
    workflow.add_player(Player(player_id="p1", name="Alex"))
    workflow.add_entry(Entry(entry_id="e1", player="p1", season="2026/27"))
    workflow.add_entry(Entry(entry_id="e2", player="p1", season="2026/27"))
    workflow.record_selection("e1", 1, "Home0", selected_at=now)
    workflow.record_selection("e2", 1, "Away1", selected_at=now)
    workflow.record_fixture_status("0", FixtureStatus.POSTPONED)
    assert workflow.survival() == {"e1": "surviving", "e2": "surviving"}
    workflow.record_fixture_status("0", FixtureStatus.PLAYED, 0, 0)
    assert workflow.survival()["e1"] == "eliminated"
    assert workflow.survival()["e2"] == "surviving"
    assert repo.count("audit_log") >= 12
