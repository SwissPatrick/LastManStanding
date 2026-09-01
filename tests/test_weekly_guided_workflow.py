from datetime import datetime, timedelta, timezone
import json

import pytest

from lms_optimizer.models import Entry, Fixture, FixtureStatus, OddsQuote, Player, Round, Season
from lms_optimizer.storage import Repository
from lms_optimizer.workflow import LMSWorkflow, select_current_provider_events
from lms_optimizer.forecast_snapshot import ForecastStore
from lms_optimizer.weekly import RecommendationSnapshot, WeeklyStore
from lms_optimizer.providers import ProviderBookmaker, ProviderEvent, ProviderOutcome, ProviderResponse


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


def test_forecast_required_creation_and_stale_gate(tmp_path):
    service = setup_service(tmp_path)
    assert not service.validate_round("2026/27", 1, "bellman")["valid"]
    store = ForecastStore(tmp_path / "forecasts")
    old = store.create_manual(datetime(2020, 1, 1, tzinfo=timezone.utc), datetime(2019, 1, 1, tzinfo=timezone.utc), "old", "elo", "1", [])
    store.save(old)
    assert not service.validate_round("2026/27", 1, "bellman", old.version)["valid"]
    fresh = store.create_manual(datetime.now(timezone.utc) + timedelta(days=1), datetime.now(timezone.utc), "fresh", "elo", "1", [])
    store.save(fresh)
    service.forecast_snapshots = lambda: [old, fresh]
    assert service.validate_round("2026/27", 1, "bellman", fresh.version)["valid"]


def test_recommendation_compare_lock_unlock_preserves_versions_and_audit(tmp_path):
    store = WeeklyStore(tmp_path / "recommendations")
    base = RecommendationSnapshot(version="v1", created_at=datetime.now(timezone.utc), season="2026/27", round_number=1, odds_snapshot_version="o1", forecast_snapshot_version="f1", active_entries=["e1"], used_teams={"e1": []}, objective_weights={"expected": 1}, exposure_limits={}, simulation_settings={}, seed=7, optimiser_version="test", allocation={"e1": "A"}, backups={"e1": "B"}, odds_snapshot={"f1": {"home": 2.0}}, probabilities={"A": .5}, exact_risk={"cvar": 1.0}, risk_estimates={"cvar": 1.0})
    store.save(base); locked_path = store.lock("v1"); locked = RecommendationSnapshot.model_validate_json(locked_path.read_text())
    with pytest.raises(ValueError): store.unlock(locked.version, "")
    changed = base.model_copy(update={"version": "v2", "allocation": {"e1": "C"}, "odds_snapshot": {"f1": {"home": 3.0}}}); store.save(changed)
    comparison = store.compare(base, changed)
    assert comparison["allocations"] and comparison["odds"]
    unlocked = store.unlock(locked.version, "odds changed", datetime.now(timezone.utc))
    assert not unlocked.locked and unlocked.previous_version == locked.version
    assert locked_path.exists() and (store.directory / f"{unlocked.version}.json").exists()


def test_provider_refresh_matches_event_id_and_preserves_provenance(tmp_path):
    service = setup_service(tmp_path); now = datetime.now(timezone.utc)
    event = ProviderEvent("provider-1", "Home 0", "Away 0", now + timedelta(days=2), (ProviderBookmaker("uk", "UK Book", now, (ProviderOutcome("Home 0", 1.5), ProviderOutcome("Draw", 4.0), ProviderOutcome("Away 0", 6.0))),))
    response = ProviderResponse("The Odds API v4", "current_odds", now, 200, "abc", {"regions": "uk", "markets": "h2h"}, {"x-requests-remaining": "1"}, str(tmp_path / "raw.json"), (event,))
    class FakeProvider:
        def current_odds(self, force_refresh=False): return response
    result = service.refresh_provider_odds(FakeProvider(), "2026/27", 1)
    assert result["events"] == 1 and result["bookmakers"] == 1
    assert service.fixtures()[-1].provider_event_id == "provider-1"
    assert "abc" in json.dumps(service.repo.list_payloads("raw_imports"))


def test_provider_refresh_selects_current_matchweek_independent_of_response_order(tmp_path):
    service = setup_service(tmp_path)
    now = datetime.now(timezone.utc)
    events = []
    for week, offset in (("current", 3), ("later", 11)):
        for index in range(10):
            kickoff = now + timedelta(days=offset, minutes=index)
            home = f"{week.title()} Home {index}"
            away = f"{week.title()} Away {index}"
            events.append(ProviderEvent(
                f"{week}-{index}", home, away, kickoff,
                (ProviderBookmaker("uk-book", "UK Book", now, (
                    ProviderOutcome(home, 1.5),
                    ProviderOutcome("Draw", 4.0),
                    ProviderOutcome(away, 6.0),
                )),),
            ))
    response = ProviderResponse("The Odds API v4", "current_odds", now, 200, "twenty", {"regions": "uk", "markets": "h2h"}, {}, str(tmp_path / "raw.json"), tuple(reversed(events)))

    class FakeProvider:
        def current_odds(self, force_refresh=False): return response

    result = service.refresh_provider_odds(FakeProvider(), "2026/27", 1)
    selected = {fixture.provider_event_id for fixture in service.fixtures() if fixture.provider_event_id}
    assert result["events"] == 10
    assert result["provider_events"] == 20
    assert selected == {f"current-{index}" for index in range(10)}
    assert set(result["deferred_events"]) == {f"later-{index}" for index in range(10)}
    assert result["warnings"] == []
    assert result["provenance"]["round_clusters"] == [[f"current-{index}" for index in range(10)], [f"later-{index}" for index in range(10)]]


def test_provider_event_selection_flags_started_team_conflicts_and_caps():
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    def event(event_id, kickoff, home, away):
        return ProviderEvent(event_id, home, away, kickoff, ())
    result = select_current_provider_events((
        event("started", now - timedelta(hours=1), "Old Home", "Old Away"),
        event("first", now + timedelta(days=1), "A", "B"),
        event("conflict", now + timedelta(days=1, minutes=1), "B", "C"),
        event("second", now + timedelta(days=1, minutes=2), "D", "E"),
    ), now=now)
    assert [item.event_id for item in result["started"]] == ["started"]
    assert [item.event_id for item in result["selected"]] == ["first", "second"]
    assert "team_conflict:conflict" in result["warnings"]


def test_provider_event_selection_enforces_cap_and_reports_ambiguous_boundary():
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    def event(event_id, kickoff, home, away):
        return ProviderEvent(event_id, home, away, kickoff, ())
    first = [event(f"f{i}", now + timedelta(days=1, minutes=i), f"FH{i}", f"FA{i}") for i in range(10)]
    near_boundary = event("near", now + timedelta(days=5, hours=1), "NH", "NA")
    result = select_current_provider_events(tuple(first + [near_boundary]), now=now)
    assert len(result["selected"]) == 10
    assert result["deferred"] == ("near",)
    assert "ambiguous_matchweek_boundary" in result["warnings"]
