"""Competition rules and deterministic selection validation."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from .models import Fixture, FixtureStatus, Selection

@dataclass(frozen=True)
class RuleDecision:
    eligible: bool
    reason: str

def eligible_fixtures(fixtures: list[Fixture], round_number: int) -> list[Fixture]:
    return [f for f in fixtures if f.round_number == round_number and f.status == FixtureStatus.SCHEDULED]

def round_is_open(fixtures: list[Fixture], round_number: int, minimum_matches: int = 6) -> bool:
    return len(eligible_fixtures(fixtures, round_number)) >= minimum_matches

def validate_selection(selection: Selection, fixtures: list[Fixture], previous: list[Selection], now: datetime | None = None, minimum_matches: int = 6, deadline: datetime | None = None) -> RuleDecision:
    now = now or datetime.now(timezone.utc)
    eligible = eligible_fixtures(fixtures, selection.round_number)
    if len(eligible) < minimum_matches:
        return RuleDecision(False, f"round has {len(eligible)} eligible matches; {minimum_matches} required")
    if selection.team not in {team for f in eligible for team in (f.home_team, f.away_team)}:
        return RuleDecision(False, "team is not playing in an eligible fixture")
    if any(s.entry_id == selection.entry_id and s.team == selection.team for s in previous):
        return RuleDecision(False, "team has already been used by this entry")
    deadline = deadline or min(f.kickoff for f in eligible)
    if selection.selected_at > deadline or now > deadline:
        return RuleDecision(False, "selection deadline has passed")
    return RuleDecision(True, "selection is valid")

def apply_result(selection: Selection, fixture: Fixture) -> Selection:
    if fixture.status != FixtureStatus.PLAYED or fixture.home_goals is None or fixture.away_goals is None:
        return selection.model_copy(update={"result": "pending"})
    winner = fixture.home_team if fixture.home_goals > fixture.away_goals else fixture.away_team if fixture.away_goals > fixture.home_goals else None
    return selection.model_copy(update={"result": "survived" if winner == selection.team else "eliminated"})

def entry_survival(selection_results: list[Selection]) -> dict[str, bool]:
    results: dict[str, bool] = defaultdict(lambda: True)
    for selection in selection_results:
        if selection.result == "eliminated":
            results[selection.entry_id] = False
    return dict(results)
