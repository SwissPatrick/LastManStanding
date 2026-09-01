"""Business workflow for manual cartel operation."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from .models import Entry, Fixture, FixtureStatus, OddsQuote, Player, Round, Season
from .probability import additive, market_disagreement, proportional, power_method, shin
from .rules import eligible_fixtures, round_is_open, validate_selection
from .storage import Repository
from .simulation import exact_current_round, adaptive_multi_round_simulation
from .forecast_snapshot import ForecastSnapshot, ForecastStore
from .providers import Provider, ProviderEvent, ProviderError, normalise_team
import numpy as np

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

    def forecast_snapshots(self, directory: str | None = None) -> list[ForecastSnapshot]:
        return [ForecastSnapshot.model_validate_json(path.read_text()) for path in sorted(ForecastStore(directory or "data/forecasts").directory.glob("*.json"))]

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

    def refresh_provider_odds(self, provider: Provider, season: str, round_number: int, tolerance_hours: int = 3, force_refresh: bool = False) -> dict[str, object]:
        response = provider.current_odds(force_refresh=force_refresh)
        existing = self.fixtures(); created, matched, ambiguous, quotes = [], [], [], []
        now = datetime.now(timezone.utc)
        for event in response.events:
            candidates = [fixture for fixture in existing if fixture.provider_event_id == event.event_id]
            if not candidates:
                candidates = [fixture for fixture in existing if fixture.season == season and fixture.round_number == round_number and normalise_team(fixture.home_team) == event.home_team and normalise_team(fixture.away_team) == event.away_team and abs((fixture.kickoff - event.kickoff).total_seconds()) <= tolerance_hours * 3600]
            if len(candidates) > 1: ambiguous.append({"event_id": event.event_id, "reason": "multiple fixture matches"}); continue
            fixture = candidates[0] if candidates else None
            if fixture is None:
                fixture = Fixture(fixture_id=f"odds-{event.event_id}", provider_event_id=event.event_id, season=season, round_number=round_number, home_team=event.home_team, away_team=event.away_team, kickoff=event.kickoff, collected_at=response.retrieved_at, market_timestamp=response.retrieved_at, data_source=response.provider)
                created.append(fixture); existing.append(fixture)
            elif fixture.provider_event_id != event.event_id:
                fixture = fixture.model_copy(update={"provider_event_id": event.event_id, "collected_at": response.retrieved_at})
                self.repo.save_fixtures([fixture]); matched.append(fixture.fixture_id)
            for bookmaker in event.bookmakers:
                if bookmaker.included:
                    outcome_map = {normalise_team(outcome.name): outcome.price for outcome in bookmaker.outcomes}
                    quotes.append(OddsQuote(fixture_id=fixture.fixture_id, bookmaker=bookmaker.key, home=outcome_map[event.home_team], draw=outcome_map["Draw"], away=outcome_map[event.away_team], collected_at=response.retrieved_at, market_timestamp=bookmaker.last_update or response.retrieved_at, data_source=response.provider))
        if created: self.add_fixtures(created)
        if quotes: self.add_odds(quotes)
        provenance = {"provider": response.provider, "endpoint_type": response.endpoint_type, "retrieval_timestamp": response.retrieved_at.isoformat(), "http_status": response.http_status, "response_checksum": response.checksum, "request_parameters": response.request_parameters, "quota_headers": response.quota_headers, "provider_event_ids": [event.event_id for event in response.events], "raw_response_storage_reference": response.raw_storage_reference}
        self.repo.record_raw("provider_response_metadata", response.retrieved_at.isoformat(), provenance)
        self.repo.audit("provider_odds_refresh", provenance)
        return {"events": len(response.events), "created": len(created), "matched": len(matched), "ambiguous": ambiguous, "bookmakers": len(quotes), "provenance": provenance, "from_cache": response.from_cache, "stale": response.stale}

    def propose_provider_results(self, provider: Provider, force_refresh: bool = False) -> dict[str, object]:
        response = provider.recent_scores(force_refresh=force_refresh); fixtures = {fixture.provider_event_id: fixture for fixture in self.fixtures() if fixture.provider_event_id}; proposals, unmatched = [], []
        for event in response.events:
            fixture = fixtures.get(event.event_id)
            if fixture is None: unmatched.append({"event_id": event.event_id, "home_team": event.home_team, "away_team": event.away_team}); continue
            if event.home_score is not None and event.away_score is not None: proposals.append({"fixture_id": fixture.fixture_id, "provider_event_id": event.event_id, "status": FixtureStatus.PLAYED.value, "home_goals": event.home_score, "away_goals": event.away_score, "provenance": {"provider": response.provider, "checksum": response.checksum, "retrieved_at": response.retrieved_at.isoformat()}})
        return {"proposals": proposals, "unmatched": unmatched, "provenance": {"provider": response.provider, "endpoint_type": response.endpoint_type, "checksum": response.checksum, "retrieved_at": response.retrieved_at.isoformat()}}

    def confirm_provider_results(self, proposals: list[dict[str, object]]) -> dict[str, str]:
        survival = {}
        for proposal in proposals:
            survival = self.record_results_and_advance(proposal["fixture_id"], FixtureStatus.PLAYED, int(proposal["home_goals"]), int(proposal["away_goals"]))
        self.repo.audit("provider_results_confirmed", {"proposals": proposals, "survival": survival})
        return survival

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
            per_bookmaker = [[proportional(odds), additive(odds), power_method(odds), shin(odds)] for odds in arrays]
            probs = [np.mean([bookmaker[index] for bookmaker in per_bookmaker], axis=0) for index in range(4)]
            overround = float(np.mean([sum(1 / price for price in odds) - 1 for odds in arrays]))
            disagreement = market_disagreement([[1 / q.home, 1 / q.draw, 1 / q.away] for q in market])
            for team, index in ((fixture.home_team, 0), (fixture.away_team, 2)):
                output.append(TeamProbability(team, fixture.fixture_id, len(market), overround, *(float(p[index]) for p in probs), disagreement))
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

    def validate_round(self, season: str, round_number: int, strategy: str = "concentrated_favourite", forecast_version: str | None = None) -> dict[str, object]:
        rounds = [Round.model_validate(x) for x in self.repo.list_payloads("rounds") if x["season"] == season and x["round_number"] == round_number]
        fixtures = [f for f in self.fixtures() if f.season == season and f.round_number == round_number]
        eligible = eligible_fixtures(fixtures, round_number)
        errors = []
        if not rounds: errors.append("Create the selected round first.")
        else:
            deadline = rounds[-1].selection_deadline
            if deadline.tzinfo is None: errors.append("Selection deadline must include a timezone.")
        if len(eligible) < 6: errors.append(f"Only {len(eligible)} eligible fixtures; six are required.")
        quotes = self.odds(); by_fixture = {f.fixture_id: [q for q in quotes if q.fixture_id == f.fixture_id] for f in eligible}
        if any(not rows for rows in by_fixture.values()): errors.append("Every eligible fixture needs at least one odds quote.")
        entries = [e for e in self.entries() if e.season == season and e.active]
        if not entries: errors.append("Create at least one active entry.")
        if any(not self.available_teams(e.entry_id, round_number) for e in entries): errors.append("Each active entry needs an eligible unused team.")
        forecast = next((item for item in self.forecast_snapshots() if item.version == forecast_version), None) if forecast_version else None
        if strategy in {"bellman", "balanced"}:
            if forecast is None: errors.append("Future-value strategies require a selected forecast snapshot.")
            elif forecast.validation_status != "validated": errors.append("Selected forecast snapshot is not validated.")
            elif forecast.information_cutoff < max((quote.market_timestamp for quote in quotes), default=forecast.information_cutoff): errors.append("Selected forecast snapshot is stale because odds were observed after its cutoff.")
        return {"valid": not errors, "errors": errors, "eligible_fixture_count": len(eligible), "six_match_rule": len(eligible) >= 6, "active_entry_count": len(entries), "odds_complete": all(bool(rows) for rows in by_fixture.values()), "timezone": str(rounds[-1].selection_deadline.tzinfo) if rounds else "", "forecast_version": forecast.version if forecast else None, "forecast_state": forecast.validation_status if forecast else "absent", "forecast_cutoff": forecast.information_cutoff.isoformat() if forecast else None}

    def analyse_round(self, season: str, round_number: int, strategy: str = "concentrated_favourite") -> dict[str, object]:
        gate = self.validate_round(season, round_number, strategy)
        if not gate["valid"]: raise ValueError("; ".join(gate["errors"]))
        entries = [e for e in self.entries() if e.season == season and e.active]
        probabilities = self.team_probabilities(round_number)
        scored = {row.team: row.proportional for row in probabilities}
        allocation, backups = {}, {}
        for entry in entries:
            teams = sorted(self.available_teams(entry.entry_id, round_number), key=lambda team: (-scored.get(team, 0.0), team))
            allocation[entry.entry_id] = teams[0]; backups[entry.entry_id] = teams[1] if len(teams) > 1 else None
        fixture_probabilities = {}; fixture_teams = {}
        for fixture in eligible_fixtures(self.fixtures(), round_number):
            quotes = [q for q in self.odds() if q.fixture_id == fixture.fixture_id]
            consensus = [sum(getattr(q, side) for q in quotes) / len(quotes) for side in ("home", "draw", "away")]
            fixture_probabilities[fixture.fixture_id] = proportional(consensus); fixture_teams[fixture.fixture_id] = (fixture.home_team, fixture.away_team)
        risk = exact_current_round(allocation, fixture_probabilities, fixture_teams)
        return {"allocation": allocation, "backups": backups, "probabilities": [row.__dict__ for row in probabilities], "exposure": self.exposure(round_number), "risk": risk, "strategy": strategy, "objective_weights": {"expected_survivors": 0.0, "at_least_one": 0.0, "wipeout": 0.0, "future_value": 0.0, "concentration": 0.0, "cvar": 0.0}}

    def save_recommendation_selections(self, allocation: dict[str, str], backups: dict[str, str | None], round_number: int) -> None:
        """Persist reviewed picks through the service boundary before locking."""
        existing = {(row["entry_id"], row["round_number"], row["is_backup"]) for row in self.repo.list_payloads("selections")}
        for entry_id, team in allocation.items():
            key = (entry_id, round_number, False)
            if key not in existing:
                payload = {"entry_id": entry_id, "round_number": round_number, "team": team, "is_backup": False, "selected_at": datetime.now(timezone.utc).isoformat(), "result": "pending"}
                self.repo.save_selection(payload); self.repo.audit("recommendation_selection_saved", payload)
            backup = backups.get(entry_id)
            if backup and (entry_id, round_number, True) not in existing:
                payload = {"entry_id": entry_id, "round_number": round_number, "team": backup, "is_backup": True, "selected_at": datetime.now(timezone.utc).isoformat(), "result": "pending"}
                self.repo.save_selection(payload); self.repo.audit("recommendation_backup_saved", payload)

    def record_results_and_advance(self, fixture_id: str, status: FixtureStatus, home_goals: int | None = None, away_goals: int | None = None) -> dict[str, str]:
        self.record_fixture_status(fixture_id, status, home_goals, away_goals)
        survival = self.survival()
        for entry_id, state in survival.items():
            if state == "eliminated":
                for entry in self.entries():
                    if entry.entry_id == entry_id:
                        self.repo.save_entry(entry.model_copy(update={"active": False}))
        self.repo.audit("results_finalised", {"fixture_id": fixture_id, "survival": survival})
        return survival
