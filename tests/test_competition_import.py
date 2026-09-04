from datetime import datetime, timedelta, timezone

import pytest

from lms_optimizer.competition_import import parse_competition_csv
from lms_optimizer.models import Entry, FamilyMember, Fixture, OddsQuote, Round, Season
from lms_optimizer.storage import Repository
from lms_optimizer.workflow import LMSWorkflow


def make_import_workflow(tmp_path):
    repo = Repository(tmp_path / "lms.sqlite3")
    workflow = LMSWorkflow(repo)
    now = datetime.now(timezone.utc)
    workflow.create_season(Season(season="2026/27", name="Fictional season", is_sample=True))
    workflow.create_round(Round(season="2026/27", round_number=1, selection_deadline=now + timedelta(days=2), is_sample=True))
    workflow.create_round(Round(season="2026/27", round_number=2, selection_deadline=now + timedelta(days=9), is_sample=True))
    for index, name in enumerate(("Alex", "Blair", "Casey", "Devon", "Emery"), 1):
        member_id, entry_id = f"member-{index}", f"family-entry-{index}"
        workflow.add_family_member(FamilyMember(member_id=member_id, name=name, position=index, is_sample=True))
        workflow.add_entry(Entry(entry_id=entry_id, member_id=member_id, season="2026/27", is_sample=True))
    fixtures = []
    teams = [("Arsenal", "Everton"), ("Manchester City", "Chelsea"), ("Liverpool", "Fulham"), ("Tottenham", "West Ham"), ("Newcastle", "Sunderland"), ("Wolverhampton Wanderers", "Burnley")]
    for round_number in (1, 2):
        for index, (home, away) in enumerate(teams):
            fixtures.append(Fixture(fixture_id=f"{round_number}-{index}", season="2026/27", round_number=round_number, home_team=home, away_team=away, kickoff=now + timedelta(days=round_number * 4), collected_at=now, is_sample=True))
    workflow.add_fixtures(fixtures)
    workflow.add_odds([OddsQuote(fixture_id=fixture.fixture_id, bookmaker="Fictional", home=1.8, draw=3.4, away=5.0, collected_at=now, market_timestamp=now, is_sample=True) for fixture in fixtures])
    return repo, workflow


def test_parser_supports_bom_semicolons_aliases_and_reports_unknowns():
    raw = "\ufeffEntry;Round 1;Round 2\nAlex One; Man Utd ; Spurs \n\n".encode()
    parsed = parse_competition_csv(raw, 2)
    assert not parsed.errors
    assert parsed.rows[0].picks == {1: "Manchester United", 2: "Tottenham"}
    bad = parse_competition_csv(b"Entry,Round 1\nAlex,Made Up FC\n", 1)
    assert bad.errors[0].row == 2
    assert "Unknown" in bad.errors[0].message


def test_parser_rejects_duplicate_headers_reuse_and_future_pick():
    parsed = parse_competition_csv(b"Entry,Round 1,round 1,Round 3\nAlex,Arsenal,Everton,Liverpool\n", 2)
    messages = " ".join(issue.message for issue in parsed.errors)
    assert "duplicated" in messages and "beyond" in messages


def test_two_round_complete_import_persists_links_picks_eliminations_and_counts(tmp_path):
    repo, workflow = make_import_workflow(tmp_path)
    first = b"Entry,Round 1\nAlex One,Arsenal\nBlair One,Man City\nCasey One,Liverpool\nDevon One,Spurs\nEmery One,Wolves\nOutside A,Chelsea\nOutside B,Everton\n"
    links = {"alex one": "family-entry-1", "blair one": "family-entry-2", "casey one": "family-entry-3", "devon one": "family-entry-4", "emery one": "family-entry-5"}
    preview = workflow.competition_import_preview(first, "2026/27", 1, True, links)
    assert not preview["has_errors"]
    assert preview["summary"]["total_competition_entries_alive"] == 7
    outcome = workflow.apply_competition_import(first, preview)
    assert outcome["summary"]["outside_entries_alive"] == 2
    assert workflow.used_teams("family-entry-1") == ["Arsenal"]
    assert workflow.available_teams("family-entry-1", 2)

    second = b"Entry,Round 1,Round 2\nAlex One,Arsenal,Everton\nBlair One,Manchester City,Arsenal\nCasey One,Liverpool,Everton\nDevon One,Tottenham,Manchester City\nEmery One,Wolverhampton Wanderers,Liverpool\nOutside A,Chelsea,Arsenal\n"
    preview = workflow.competition_import_preview(second, "2026/27", 2, True)
    assert preview["summary"]["proposed_eliminations"] == ["Outside B"]
    with pytest.raises(ValueError, match="Confirm"):
        workflow.apply_competition_import(second, preview)
    outcome = workflow.apply_competition_import(second, preview, confirm_eliminations=True)
    assert outcome["summary"]["total_competition_entries_alive"] == 6
    assert workflow.wider_field("2026/27", 2).surviving_entries == 1
    assert next(entry for entry in repo.competition_entries("2026/27") if entry["display_label"] == "Outside B")["active"] is False
    assert workflow.used_teams("family-entry-1") == ["Arsenal", "Everton"]
    repo.close()
    restarted = LMSWorkflow(Repository(tmp_path / "lms.sqlite3"))
    assert len(restarted.repo.competition_entries("2026/27")) == 7
    assert restarted.used_teams("family-entry-1") == ["Arsenal", "Everton"]


def test_partial_never_eliminates_and_stale_preview_is_rejected(tmp_path):
    _, workflow = make_import_workflow(tmp_path)
    raw = b"Entry,Round 1\nOutside A,Arsenal\nOutside B,Chelsea\n"
    preview = workflow.competition_import_preview(raw, "2026/27", 1, True)
    workflow.apply_competition_import(raw, preview)
    partial = b"Entry,Round 2\nOutside A,Everton\n"
    preview = workflow.competition_import_preview(partial, "2026/27", 2, False)
    assert not preview["summary"]["proposed_eliminations"]
    workflow.apply_competition_import(partial, preview)
    assert all(entry["active"] for entry in workflow.repo.competition_entries("2026/27"))
    stale = workflow.competition_import_preview(b"Entry,Round 2\nOutside B,Everton\n", "2026/27", 2, False)
    workflow.repo.audit("unrelated_change", {})
    with pytest.raises(ValueError, match="changed after preview"):
        workflow.apply_competition_import(b"Entry,Round 2\nOutside B,Everton\n", stale)


def test_backup_is_not_consumed(tmp_path):
    _, workflow = make_import_workflow(tmp_path)
    workflow.save_recommendation_selections({"family-entry-1": "Arsenal"}, {"family-entry-1": "Chelsea"}, 1)
    assert "Chelsea" not in workflow.used_teams("family-entry-1")
