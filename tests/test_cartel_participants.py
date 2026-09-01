from datetime import datetime, timedelta, timezone

from lms_optimizer.models import Entry, FamilyMember, Fixture, FixtureStatus, OddsQuote, Round, Season, WiderFieldSnapshot
from lms_optimizer.storage import Repository
from lms_optimizer.workflow import LMSWorkflow


def setup_cartel(tmp_path):
    repo = Repository(tmp_path / "cartel.sqlite3")
    service = LMSWorkflow(repo)
    now = datetime.now(timezone.utc)
    service.create_season(Season(season="2026/27", name="Cartel", is_sample=True))
    service.create_round(Round(season="2026/27", round_number=1, selection_deadline=now + timedelta(days=1), is_sample=True))
    fixtures = [Fixture(fixture_id=str(i), season="2026/27", round_number=1, home_team=f"Home{i}", away_team=f"Away{i}", kickoff=now + timedelta(days=2), collected_at=now, is_sample=True) for i in range(6)]
    service.add_fixtures(fixtures)
    service.add_odds([OddsQuote(fixture_id=str(i), bookmaker="Book", home=2, draw=4, away=6, collected_at=now, market_timestamp=now, is_sample=True) for i in range(6)])
    return repo, service, now


def test_five_members_can_have_different_entry_counts_and_independent_used_teams(tmp_path):
    repo, service, now = setup_cartel(tmp_path)
    counts = [1, 2, 3, 4, 10]
    for position, count in enumerate(counts, 1):
        member = FamilyMember(member_id=f"m{position}", name=["Me", "Brother", "Dad", "Uncle", "Cousin"][position - 1], position=position)
        service.add_family_member(member)
        for index in range(count): service.add_entry(Entry(entry_id=f"e{position}-{index}", member_id=member.member_id, season="2026/27"))
    assert [len(service.entries_by_member("2026/27")[f"m{i}"]) for i in range(1, 6)] == counts
    assert len(service.family_members()) == 5
    service.record_selection("e1-0", 1, "Home0", selected_at=now)
    service.record_selection("e2-0", 1, "Home0", selected_at=now)
    assert service.used_teams("e1-0") == ["Home0"]
    assert service.used_teams("e2-0") == ["Home0"]
    repo.close()


def test_member_can_have_partial_elimination_and_field_snapshots_are_round_specific(tmp_path):
    repo, service, now = setup_cartel(tmp_path)
    service.add_family_member(FamilyMember(member_id="dad", name="Dad", position=3))
    service.add_entry(Entry(entry_id="d1", member_id="dad", season="2026/27"))
    service.add_entry(Entry(entry_id="d2", member_id="dad", season="2026/27"))
    service.record_selection("d1", 1, "Home0", selected_at=now)
    service.record_selection("d2", 1, "Away1", selected_at=now)
    service.record_fixture_status("0", FixtureStatus.PLAYED, 0, 0)
    assert service.survival()["d1"] == "eliminated"
    assert service.survival()["d2"] == "surviving"
    assert service.member_survival("2026/27")["Dad"] == "surviving"
    service.save_wider_field(WiderFieldSnapshot(season="2026/27", round_number=1, starting_entries=100, surviving_entries=86, recorded_at=now))
    service.save_wider_field(WiderFieldSnapshot(season="2026/27", round_number=2, starting_entries=100, surviving_entries=70, recorded_at=now))
    assert service.wider_field("2026/27", 1).surviving_entries == 86
    assert service.wider_field("2026/27", 2).surviving_entries == 70
    repo.close()
