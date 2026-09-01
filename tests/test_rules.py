from datetime import datetime, timezone, timedelta
from lms_optimizer.models import Fixture, Selection
from lms_optimizer.rules import round_is_open, validate_selection

def fixtures():
    now = datetime.now(timezone.utc)
    return [Fixture(fixture_id=f"{round_number}-{i}", season="2026/27", round_number=round_number, home_team=f"Home{i}", away_team=f"Away{i}", kickoff=now + timedelta(days=2), collected_at=now) for round_number in (1, 2) for i in range(6)]

def test_round_requires_six_matches():
    assert round_is_open(fixtures(), 1)
    assert not round_is_open(fixtures()[:5], 1)

def test_team_cannot_be_reused_by_entry():
    now = datetime.now(timezone.utc)
    selection = Selection(entry_id="e1", round_number=2, team="Home0", selected_at=now)
    previous = [Selection(entry_id="e1", round_number=1, team="Home0", selected_at=now)]
    decision = validate_selection(selection, fixtures(), previous, now=now)
    assert not decision.eligible
    assert "already" in decision.reason
