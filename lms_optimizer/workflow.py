"""Business workflow for manual cartel operation."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from .models import Entry, Fixture, FixtureStatus, OddsQuote, Player, Round, Season
from .probability import additive, market_disagreement, proportional, power_method, shin
from .rules import eligible_fixtures, round_is_open, validate_selection
from .storage import Repository

@dataclass(frozen=True)
class TeamProbability:
    team: str
    fixture_id: str
    bookmaker_count: int
    overround: float
    proportional: float
    additive: float
    power: float
    shin: float
    disagreement: float

class LMSWorkflow:
    def __init__(self, repository: Repository) -> None:
        self.repo = repository

    def create_season(self, season: Season) -> None:
        self.repo.save_season(season)
        self.repo.audit("season_created", season.model_dump())

    def create_round(self, round_: Round) -> None:
        self.repo.save_round(round_)
        self.repo.audit("round_created", round_.model_dump())

    def add_fixtures(self, fixtures: list[Fixture]) -> None:
        if len({f.fixture_id for f in fixtures}) != len(fixtures):
            raise ValueError("duplicate fixture identifiers")
        self.repo.save_fixtures(fixtures)
        for fixture in fixtures:
            self.repo.audit("fixture_saved", fixture.model_dump())

    def add_odds(self, quotes: list[OddsQuote]) -> None:
        self.repo.save_odds(quotes)
        for quote in quotes:
            self.repo.audit("odds_saved", quote.model_dump())

    def add_player(self, player: Player) -> None:
        self.repo.save_player(player)
        self.repo.audit("player_created", player.model_dump())

    def add_entry(self, entry: Entry) -> None:
        entries = [Entry.model_validate(x) for x in self.repo.list_payloads("entries") if x.get("player") == entry.player]
        if len(entries) >= 10:
            raise ValueError("an individual may have at most ten entries")
        self.repo.save_entry(entry)
        self.repo.audit("entry_created", entry.model_dump())

    def fixtures(self) -> list[Fixture]:
        return [Fixture.model_validate(x) for x in self.repo.list_payloads("fixtures")]

    def entries(self) -> list[Entry]:
        return [Entry.model_validate(x) for x in self.repo.list_payloads("entries")]

    def odds(self) -> list[OddsQuote]:
        return [OddsQuote.model_validate(x) for x in self.repo.list_payloads("odds_quotes")]

    def available_teams(self, entry_id: str, round_number: int) -> list[str]:
        used = {x["team"] for x in self.repo.list_payloads("selections") if x["entry_id"] == entry_id}
        return sorted({t for f in eligible_fixtures(self.fixtures(), round_number) for t in (f.home_team, f.away_team) if t not in used})

    def used_teams(self, entry_id: str) -> list[str]:
        return sorted({x["team"] for x in self.repo.list_payloads("selections") if x["entry_id"] == entry_id})

    def record_selection(self, entry_id: str, round_number: int, team: str, is_backup: bool = False, selected_at: datetime | None = None) -> None:
        selected_at = selected_at or datetime.now(timezone.utc)
        from .models import Selection
        previous = [Selection.model_validate(x) for x in self.repo.list_payloads("selections")]
        round_rows = [Round.model_validate(x) for x in self.repo.list_payloads("rounds") if x["round_number"] == round_number]
        deadline = round_rows[-1].selection_deadline if round_rows else None
        decision = validate_selection(Selection(entry_id=entry_id, round_number=round_number, team=team, selected_at=selected_at), self.fixtures(), previous, now=selected_at, deadline=deadline)
        if not decision.eligible:
            raise ValueError(decision.reason)
        payload = {"entry_id": entry_id, "round_number": round_number, "team": team, "is_backup": is_backup, "selected_at": selected_at.isoformat(), "result": "pending"}
        self.repo.save_selection(payload)
        self.repo.audit("selection_recorded", payload)

    def fallback_preview(self, entry_id: str, round_number: int) -> str | None:
        candidates = self.available_teams(entry_id, round_number)
        scored = {row.team: row.shin for row in self.team_probabilities(round_number)}
        return min(candidates, key=lambda team: 1 - scored.get(team, 0.0)) if candidates else None

    def team_probabilities(self, round_number: int) -> list[TeamProbability]:
        fixture_list = eligible_fixtures(self.fixtures(), round_number)
        quotes = self.odds()
        output: list[TeamProbability] = []
        for fixture in fixture_list:
            market = [q for q in quotes if q.fixture_id == fixture.fixture_id]
            if not market: continue
            arrays = [[q.home, q.draw, q.away] for q in market]
            consensus = [sum(row[i] for row in arrays) / len(arrays) for i in range(3)]
            raw = [1 / x for x in consensus]
            probs = [proportional(consensus), additive(consensus), power_method(consensus), shin(consensus)]
            disagreement = market_disagreement([[1 / q.home, 1 / q.draw, 1 / q.away] for q in market])
            for team, index in ((fixture.home_team, 0), (fixture.away_team, 2)):
                output.append(TeamProbability(team, fixture.fixture_id, len(market), sum(raw) - 1, *(float(p[index]) for p in probs), disagreement))
        return output

    def exposure(self, round_number: int) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for selection in self.repo.list_payloads("selections"):
            if selection["round_number"] == round_number and not selection.get("is_backup", False):
                counts[selection["team"]] += 1
        return dict(sorted(counts.items()))

    def record_fixture_status(self, fixture_id: str, status: FixtureStatus, home_goals: int | None = None, away_goals: int | None = None) -> None:
        fixtures = self.fixtures()
        target = next((f for f in fixtures if f.fixture_id == fixture_id), None)
        if target is None: raise ValueError("unknown fixture")
        updated = target.model_copy(update={"status": status, "home_goals": home_goals, "away_goals": away_goals})
        self.repo.save_fixtures([updated])
        self.repo.audit("fixture_status_recorded", updated.model_dump())

    def survival(self) -> dict[str, str]:
        fixture_by_id = {f.fixture_id: f for f in self.fixtures()}
        results = {e.entry_id: "surviving" for e in self.entries()}
        for item in self.repo.list_payloads("selections"):
            if item.get("is_backup", False): continue
            fixture = next((f for f in fixture_by_id.values() if item["team"] in (f.home_team, f.away_team) and f.round_number == item["round_number"]), None)
            if fixture and fixture.status == FixtureStatus.PLAYED:
                winner = fixture.home_team if fixture.home_goals > fixture.away_goals else fixture.away_team if fixture.away_goals > fixture.home_goals else None
                if winner != item["team"]: results[item["entry_id"]] = "eliminated"
        return results
