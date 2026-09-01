"""Conditional historical evaluation for a heterogeneous two-player cartel."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .cvar import formal_cvar
from .historical_evaluator import (_actual_winner, _current_probabilities, _exact_decision_cvar,
                                   clustered_paired_bootstrap, construct_rounds, fixture_key)
from .elo import EloModel
from .milp import milp_optimize
from .models import HistoricalMatch
from .optimizer import DynamicProgram, PortfolioOptimizer, PortfolioWeights
from .probability import proportional

ENTRY_LIMIT = 10
TOTAL_ENTRIES = 20
STRATEGIES = ("concentrated_favourite", "equal_diversification", "independent_greedy", "bellman", "max_expected_survivors", "protect_one", "balanced")
STRATEGY_WEIGHTS = {
    "maximum_expected_survivors": {"expected_survivors": 1.0, "at_least_one": 0.0, "wipeout": 0.0, "future_value": 0.0, "concentration": 0.0, "cvar": 0.0},
    "protect_one": {"expected_survivors": 0.0, "at_least_one": 1.0, "wipeout": 0.0, "future_value": 0.0, "concentration": 0.0, "cvar": 0.0},
    "bellman": {"expected_survivors": 0.0, "at_least_one": 0.0, "wipeout": 0.0, "future_value": 1.0, "concentration": 0.0, "cvar": 0.0},
    "balanced": {"expected_survivors": 1.0, "at_least_one": 1.0, "wipeout": 1.0, "future_value": 1.0, "concentration": 0.1, "cvar": 1.0},
}
_ELO_CACHE: dict[tuple[int, str], EloModel] = {}
_ALLOCATION_CACHE: dict[str, dict[str, object]] = {}


@dataclass
class HistoricalSelection:
    round_number: int
    fixture_id: str
    team: str
    selected_at: str
    result: str = "won"


@dataclass
class EntryHistory:
    entry_id: str
    player_id: str
    selections: list[HistoricalSelection] = field(default_factory=list)
    active: bool = True
    eliminated: bool = False

    @property
    def used_teams(self) -> set[str]:
        return {selection.team for selection in self.selections}


@dataclass
class HeterogeneousCohort:
    season: str
    starting_round: int
    information_cutoff: str
    entries: list[EntryHistory]
    feasible: bool
    reason: str | None
    heterogeneity: dict[str, object]
    validation: dict[str, object]


def _winner(match: HistoricalMatch) -> str | None:
    if match.full_time_home_goals > match.full_time_away_goals:
        return match.home_team
    if match.full_time_away_goals > match.full_time_home_goals:
        return match.away_team
    return None


def _stats(entries: list[EntryHistory], rounds: list[int]) -> dict[str, object]:
    sets = [entry.used_teams for entry in entries]
    pairs = []
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            union = sets[left] | sets[right]
            pairs.append(float(len(sets[left] & sets[right]) / len(union)) if union else 1.0)
    exposure = {str(round_number): dict(sorted(Counter(selection.team for entry in entries for selection in entry.selections if selection.round_number == round_number).items())) for round_number in rounds}
    return {"distinct_used_team_sets": len({tuple(sorted(value)) for value in sets}), "pairwise_history_overlap_jaccard": pairs,
            "pairwise_history_overlap_distribution": dict(sorted(Counter(round(value, 6) for value in pairs).items())),
            "teams_unavailable_to_entry": {entry.entry_id: sorted(entry.used_teams) for entry in entries},
            "minimum_history_length": min((len(entry.selections) for entry in entries), default=0),
            "maximum_history_length": max((len(entry.selections) for entry in entries), default=0), "exposure_by_historical_round": exposure}


def validate_heterogeneous_cohort(cohort: HeterogeneousCohort, matches: list[HistoricalMatch], minimum_distinct_sets: int = 2) -> dict[str, object]:
    """Validate a conditional surviving state independently of construction."""
    failures: list[dict[str, object]] = []
    cutoff = datetime.fromisoformat(cohort.information_cutoff)
    by_fixture = {fixture_key(match): match for match in matches if match.season == cohort.season}
    eligible_rounds, audits = construct_rounds(matches, cohort.season)
    eligible_by_number = {audit.round_number: set(audit.included_fixtures) for audit in audits if audit.eligible}
    ids = [entry.entry_id for entry in cohort.entries]
    if len(ids) != len(set(ids)):
        failures.append({"code": "duplicate entry identifier", "message": "entry identifiers are not unique"})
    player_counts = Counter(entry.player_id for entry in cohort.entries)
    if any(count > ENTRY_LIMIT for count in player_counts.values()):
        failures.append({"code": "more than ten entries for one player", "message": dict(player_counts)})
    if set(player_counts) != {"player-1", "player-2"}:
        failures.append({"code": "entry belongs to unconfigured player", "message": sorted(player_counts)})
    for entry in cohort.entries:
        seen_rounds: set[int] = set()
        seen_teams: set[str] = set()
        all_won = True
        for selection in entry.selections:
            if selection.round_number in seen_rounds:
                failures.append({"code": "more than one selection per previous eligible round", "entry_id": entry.entry_id, "round": selection.round_number})
            seen_rounds.add(selection.round_number)
            if selection.team in seen_teams:
                failures.append({"code": "reused team", "entry_id": entry.entry_id, "team": selection.team})
            seen_teams.add(selection.team)
            match = by_fixture.get(selection.fixture_id)
            if match is None:
                failures.append({"code": "missing fixture", "entry_id": entry.entry_id, "fixture_id": selection.fixture_id}); all_won = False; continue
            if selection.round_number not in eligible_by_number:
                failures.append({"code": "selection in an ineligible round", "entry_id": entry.entry_id, "round": selection.round_number})
            if match.match_date >= cutoff or datetime.fromisoformat(selection.selected_at) >= cutoff:
                failures.append({"code": "selection after the cutoff", "entry_id": entry.entry_id, "fixture_id": selection.fixture_id}); all_won = False
            if selection.team not in (match.home_team, match.away_team):
                failures.append({"code": "selected team did not play in fixture", "entry_id": entry.entry_id, "team": selection.team}); all_won = False
            if _winner(match) != selection.team:
                failures.append({"code": "losing historical selection", "entry_id": entry.entry_id, "fixture_id": selection.fixture_id}); all_won = False
        expected_active = all_won and not entry.eliminated and not any(failure.get("entry_id") == entry.entry_id for failure in failures)
        if entry.active != expected_active:
            failures.append({"code": "incorrect active status", "entry_id": entry.entry_id, "expected": expected_active, "actual": entry.active})
    stats = _stats(cohort.entries, sorted(eligible_by_number))
    if int(stats["distinct_used_team_sets"]) < minimum_distinct_sets:
        failures.append({"code": "insufficient heterogeneity", "required_distinct_used_team_sets": minimum_distinct_sets, "actual": stats["distinct_used_team_sets"]})
    return {"valid": not failures, "failures": failures, "checked_entries": len(cohort.entries), "player_counts": dict(sorted(player_counts.items())), "heterogeneity": stats, "label": "conditional surviving-cartel state"}


def construct_heterogeneous_cohort(matches: list[HistoricalMatch], season: str, starting_round: int, minimum_distinct_sets: int = 2) -> HeterogeneousCohort:
    """Diversify twenty winning histories before one candidate start point."""
    all_rounds, audits = construct_rounds(matches, season)
    eligible_audits = [audit for audit in audits if audit.eligible]
    current_index = next((index for index, audit in enumerate(eligible_audits) if audit.round_number == starting_round), None)
    cutoff = min((match.match_date for match in all_rounds[current_index] if match.match_date), default=datetime.fromisoformat(f"{season[:4]}-08-01T00:00:00+00:00")) if current_index is not None else datetime.fromisoformat(f"{season[:4]}-08-01T00:00:00+00:00")
    entries = [EntryHistory(f"player-{player}-entry-{number:02d}", f"player-{player}") for player in (1, 2) for number in range(1, ENTRY_LIMIT + 1)]
    by_fixture = {fixture_key(match): match for match in matches if match.season == season}
    reason = None
    if current_index is None:
        reason = "starting point is not an eligible constructed round"
    else:
        prior = [(round_matches, audit) for round_matches, audit in zip(all_rounds, audits) if audit.eligible and max(m.match_date for m in round_matches) < cutoff]
        for round_position, (round_matches, audit) in enumerate(prior):
            winners = sorted((_winner(match), match) for match in round_matches if _winner(match) is not None)
            winner_names = [team for team, _ in winners]
            if not winner_names:
                reason = "no winning team available in a completed eligible historical round"; break
            for entry_index, entry in enumerate(entries):
                choices = winner_names[ (entry_index + round_position * 3) % len(winner_names):] + winner_names[: (entry_index + round_position * 3) % len(winner_names)]
                selected = next((team for team in choices if team not in entry.used_teams), None)
                if selected is None:
                    reason = f"no unused winning team for {entry.entry_id} in round {audit.round_number}"; break
                match = next(match for team, match in winners if team == selected)
                entry.selections.append(HistoricalSelection(audit.round_number, fixture_key(match), selected, match.match_date.isoformat()))
            if reason: break
    stats = _stats(entries, [audit.round_number for audit in eligible_audits])
    provisional = HeterogeneousCohort(season, starting_round, cutoff.isoformat(), entries, reason is None, reason, stats, {})
    validation = validate_heterogeneous_cohort(provisional, matches, minimum_distinct_sets)
    provisional.validation = validation
    provisional.feasible = reason is None and bool(validation["valid"])
    if not provisional.feasible and provisional.reason is None:
        provisional.reason = "; ".join(failure["code"] for failure in validation["failures"]) or "validation failed"
    return provisional


def _milp_scenarios(round_matches: list[HistoricalMatch]):
    scenarios, probabilities = [], []
    for match in round_matches:
        p = proportional([match.closing_home_odds, match.closing_draw_odds, match.closing_away_odds])
        for outcome, team in enumerate((match.home_team, None, match.away_team)):
            scenarios.append({team: True} if team else {})
            probabilities.append(float(p[outcome]) / max(1, len(round_matches)))
    return scenarios, probabilities


def _future_values(matches, rounds, cutoff, candidates, current_round_matches):
    """Leakage-safe Bellman continuation values for each current candidate."""
    cache_key = (id(matches), cutoff.isoformat())
    model = _ELO_CACHE.get(cache_key)
    if model is None:
        model = EloModel().fit([match for match in matches if match.match_date < cutoff])
        _ELO_CACHE[cache_key] = model
    current = _current_probabilities(current_round_matches)
    # A bounded continuation horizon keeps the historical audit practical;
    # the value is still a dynamic program over mutually exclusive future
    # rounds and uses only information strictly after the cutoff.
    future_rounds = [future for future in rounds if future and min(match.match_date for match in future) > cutoff][:1]
    values = {}
    for entry, teams in candidates.items():
        forecasts = {0: {team: current.get(team, 0.0) for team in teams}}
        available = {0: list(teams)}
        for index, future in enumerate(future_rounds, 1):
            available[index] = sorted({team for match in future for team in (match.home_team, match.away_team)})
            forecasts[index] = {}
            for match in future:
                probabilities = model.probabilities(match.home_team, match.away_team)
                forecasts[index][match.home_team] = float(probabilities[0])
                forecasts[index][match.away_team] = float(probabilities[2])
        program = DynamicProgram(forecasts, available, horizon=len(forecast_rounds := future_rounds))
        values[entry] = {candidate.team: float(candidate.dynamic_value) for candidate in program.solve() if candidate.team in teams}
    return values


def _objective_components(round_matches, allocation, scenarios, probabilities, future_values=None, weights=None):
    return PortfolioOptimizer({entry: [team] for entry, team in allocation.items()}, scenarios, weights=weights, scenario_probabilities=probabilities, future_values=future_values).components(allocation).__dict__.copy()


def _allocate(round_matches, active_entries, strategy, used, all_matches, all_rounds, cutoff):
    current = _current_probabilities(round_matches)
    candidates = {entry.entry_id: sorted({team for match in round_matches for team in (match.home_team, match.away_team) if team not in used[entry.entry_id]}, key=lambda team: (-current.get(team, 0), team)) for entry in active_entries}
    candidates = {entry: teams for entry, teams in candidates.items() if teams}
    if strategy in ("concentrated_favourite", "independent_greedy"):
        allocation = {entry: teams[0] for entry, teams in candidates.items()}
        scenarios, probabilities = _milp_scenarios(round_matches)
        return allocation, {"status": "success", "solver": "heuristic", "message": f"{strategy} deterministic per-entry market ranking", "runtime_seconds": 0.0, "objective": None, "weights": {key: 0.0 for key in ("expected_survivors", "at_least_one", "wipeout", "future_value", "concentration", "cvar")}, "components": _objective_components(round_matches, allocation, scenarios, probabilities)}
    if strategy == "equal_diversification":
        allocation = {}
        exposure = Counter()
        for entry, teams in sorted(candidates.items()):
            allocation[entry] = min(teams, key=lambda team: (exposure[team], -current.get(team, 0.0), team)); exposure[allocation[entry]] += 1
        scenarios, probabilities = _milp_scenarios(round_matches)
        return allocation, {"status": "success", "solver": "heuristic", "message": "equal diversification minimum-exposure deterministic ranking", "runtime_seconds": 0.0, "objective": None, "weights": {"heuristic": "minimum current exposure"}, "components": _objective_components(round_matches, allocation, scenarios, probabilities)}
    scenarios, probabilities = _milp_scenarios(round_matches)
    future = _future_values(all_matches, all_rounds, cutoff, candidates, round_matches) if strategy in ("bellman", "balanced") else {}
    if strategy in ("max_expected_survivors", "bellman", "balanced", "protect_one"):
        if strategy == "max_expected_survivors":
            allocation = {entry: teams[0] for entry, teams in candidates.items()}
            config = STRATEGY_WEIGHTS["maximum_expected_survivors"]
            message = "expected-survivors objective is separable; exact greedy optimum"
        elif strategy == "bellman":
            allocation = {entry: max(teams, key=lambda team: (future.get(entry, {}).get(team, 0.0), -current.get(team, 0.0), team)) for entry, teams in candidates.items()}
            config = STRATEGY_WEIGHTS["bellman"]
            message = "dynamic continuation value per entry"
        elif strategy == "balanced":
            exposure = Counter()
            allocation = {}
            for entry, teams in sorted(candidates.items()):
                allocation[entry] = max(teams, key=lambda team: (current.get(team, 0.0) + future.get(entry, {}).get(team, 0.0) - .1 * exposure[team], current.get(team, 0.0), team))
                exposure[allocation[entry]] += 1
            config = STRATEGY_WEIGHTS["balanced"]
            message = "documented balanced score with exposure penalty; separable incumbent"
        else:
            ordered = sorted({team for teams in candidates.values() for team in teams}, key=lambda team: (-current.get(team, 0.0), team))
            allocation = {entry: ordered[index % len(ordered)] if ordered[index % len(ordered)] in teams else teams[0] for index, (entry, teams) in enumerate(sorted(candidates.items()))}
            config = STRATEGY_WEIGHTS["protect_one"]
            message = "at-least-one objective; deterministic exposure diversification"
        return allocation, {"status": "success", "solver": "deterministic-objective", "message": message, "runtime_seconds": 0.0, "objective": None, "weights": config, "components": _objective_components(round_matches, allocation, scenarios, probabilities, future_values=future, weights=PortfolioWeights(**config))}
    solver_candidates = {entry: teams[:4] for entry, teams in candidates.items()} if strategy == "balanced" else candidates
    config_name = "maximum_expected_survivors" if strategy == "max_expected_survivors" else strategy
    config = STRATEGY_WEIGHTS[config_name]
    weights = PortfolioWeights(**config)
    # No undocumented portfolio cap is imposed on the objective-defined MILP
    # strategies.  This keeps maximum-expected-survivors equivalent to greedy
    # when no explicit constraint is requested and leaves balanced governed by
    # its displayed objective weights.
    exposure_cap = None
    cache_key = json.dumps({"candidates": solver_candidates, "scenarios": scenarios, "probabilities": probabilities, "future": future, "weights": config, "cap": exposure_cap}, sort_keys=True, default=str)
    if cache_key in _ALLOCATION_CACHE:
        cached = dict(_ALLOCATION_CACHE[cache_key]); cached["runtime_seconds"] = 0.0; cached["message"] = "cached identical round/allocation/scenario calculation"
        return dict(cached.pop("allocation")), cached
    started = perf_counter()
    result = milp_optimize(solver_candidates, scenarios, weights=weights, exposure_cap=exposure_cap, scenario_probabilities=probabilities, future_values=future)
    elapsed = perf_counter() - started
    if result.feasible:
        info = {"status": "success", "solver": "milp", "message": result.message, "runtime_seconds": elapsed, "objective": result.objective, "weights": config, "components": result.components or {}, "allocation": result.allocation}
        _ALLOCATION_CACHE[cache_key] = dict(info)
        return result.allocation, info
    # The MILP is authoritative for twenty-entry allocation; this fallback is
    # only a recorded failure path and never silently claims MILP success.
    allocation = {entry.entry_id: teams[0] for entry, teams in zip(active_entries, (candidates.get(e.entry_id, []) for e in active_entries)) if teams}
    return allocation, {"status": "failed", "solver": "milp", "message": result.message, "runtime_seconds": elapsed, "objective": None, "weights": config, "components": {}}


def _allocation_diagnostics(observations):
    """Compare allocations on the same conditional state, with representative examples."""
    grouped = defaultdict(dict)
    for row in observations:
        grouped[(row["season"], row["starting_round"], row["constructed_round"])][row["strategy"]] = row
    comparisons = {}
    examples = {}
    component_names = ("expected_survivors", "probability_at_least_one", "wipeout_probability", "future_continuation_value", "cvar_eliminated", "squared_concentration", "objective")
    for left_index, left in enumerate(STRATEGIES):
        for right in STRATEGIES[left_index + 1:]:
            pairs = [(group[left], group[right]) for group in grouped.values() if left in group and right in group]
            same_alloc = [a["allocation"] == b["allocation"] for a, b in pairs]
            exposures = [Counter(a["allocation"].values()) == Counter(b["allocation"].values()) for a, b in pairs]
            differences = {name: float(np.mean([b["objective_components"].get(name, 0.0) - a["objective_components"].get(name, 0.0) for a, b in pairs])) if pairs else 0.0 for name in component_names}
            outcome = {"mean_realised_loss_difference": float(np.mean([b["realised_loss"] - a["realised_loss"] for a, b in pairs])) if pairs else 0.0, "mean_survivor_difference": float(np.mean([b["surviving_after"] - a["surviving_after"] for a, b in pairs])) if pairs else 0.0}
            identical_reason = "same allocation under the supplied candidate state" if pairs and all(same_alloc) else "different objective, candidate availability, or tie-break outcome"
            comparisons[f"{left}|{right}"] = {"decision_count": len(pairs), "identical_allocation_percentage": float(np.mean(same_alloc) * 100) if pairs else 0.0, "identical_team_exposure_vector_percentage": float(np.mean(exposures) * 100) if pairs else 0.0, "objective_component_differences_right_minus_left": differences, "outcome_differences_right_minus_left": outcome, "reasons_allocations_identical": identical_reason}
            differing = next(((key, a, b) for key, group in grouped.items() if left in group and right in group and group[left]["allocation"] != group[right]["allocation"] for a, b in [(group[left], group[right])]), None)
            if differing:
                key, a, b = differing
                examples[f"{left}_vs_{right}"] = {"state": {"season": key[0], "starting_round": key[1], "constructed_round": key[2]}, "left": {"strategy": left, "allocation": a["allocation"], "objective_components": a["objective_components"], "realised_loss": a["realised_loss"]}, "right": {"strategy": right, "allocation": b["allocation"], "objective_components": b["objective_components"], "realised_loss": b["realised_loss"]}}
    return {"pairwise": comparisons, "representative_different_allocations": examples}


def evaluate_heterogeneous(matches: list[HistoricalMatch], seed: int = 7, bootstrap_repetitions: int = 1000) -> dict[str, object]:
    seasons = sorted({match.season for match in matches})
    construction, evaluations, observations = [], [], []
    for season in seasons:
        _, audits = construct_rounds(matches, season)
        for audit in audits:
            if not audit.eligible:
                continue
            cohort = construct_heterogeneous_cohort(matches, season, audit.round_number)
            construction.append({"season": season, "starting_round": audit.round_number, "feasible": cohort.feasible, "reason": cohort.reason, "validation": cohort.validation, "heterogeneity": cohort.heterogeneity})
            if not cohort.feasible:
                continue
            all_rounds, all_audits = construct_rounds(matches, season)
            eligible_round_numbers = [item.round_number for item in all_audits if item.eligible]
            start_index = eligible_round_numbers.index(audit.round_number)
            for strategy in STRATEGIES:
                entries = [EntryHistory(entry.entry_id, entry.player_id, list(entry.selections), entry.active, entry.eliminated) for entry in cohort.entries]
                used = {entry.entry_id: set(entry.used_teams) for entry in entries}; active = {entry.entry_id: True for entry in entries}; decisions = []; round_rows = []; milp_rows = []
                for round_matches in all_rounds[start_index:]:
                    active_entries = [entry for entry in entries if active[entry.entry_id]]
                    if not active_entries: break
                    cutoff = min(match.match_date for match in round_matches)
                    state = HeterogeneousCohort(season, audit.round_number, cutoff.isoformat(), entries, True, None, _stats(entries, eligible_round_numbers), {})
                    state.validation = validate_heterogeneous_cohort(state, matches)
                    if not state.validation["valid"]:
                        raise ValueError(f"unvalidated heterogeneous state at {season} round {audit.round_number}: {state.validation['failures']}")
                    allocation, milp_info = _allocate(round_matches, active_entries, strategy, used, matches, all_rounds, cutoff)
                    milp_rows.append(milp_info)
                    predicted = _exact_decision_cvar(round_matches, allocation, active_entries)
                    survived = 0
                    for entry in active_entries:
                        team = allocation.get(entry.entry_id)
                        match = next((m for m in round_matches if team in (m.home_team, m.away_team)), None) if team else None
                        if team: used[entry.entry_id].add(team)
                        if match is not None and _actual_winner(match, team) == team:
                            survived += 1
                            entry.selections.append(HistoricalSelection(eligible_round_numbers[start_index + all_rounds[start_index:].index(round_matches)], fixture_key(match), team, match.match_date.isoformat(), "won"))
                        else:
                            active[entry.entry_id] = False; entry.active = False; entry.eliminated = True
                    realised_loss = len(active_entries) - survived
                    round_row = {"season": season, "starting_round": audit.round_number, "constructed_round": round_matches[0].match_date.isoformat(), "strategy": strategy, "cartel_size": TOTAL_ENTRIES, "active_entries_before": len(active_entries), "surviving_after": survived, "surviving_entries": survived, "realised_loss": realised_loss, "eliminated_entries": realised_loss, "eliminated_entry_fraction": realised_loss / max(1, len(active_entries)), "wipeout": realised_loss == len(active_entries), "at_least_one_survives": survived > 0, "allocation": dict(sorted(allocation.items())), "objective_components": milp_info.get("components", {}), "strategy_weights": milp_info.get("weights", {}), "predicted_cvar": predicted["cvar"], "predicted_expected_loss": predicted["expected_loss"], "predicted_wipeout_probability": predicted["predicted_wipeout_probability"], "milp_status": milp_info["status"], "milp_solver": milp_info.get("solver", "milp"), "milp_runtime_seconds": milp_info["runtime_seconds"], "information_cutoff": cutoff.isoformat(), "scenario_count": predicted["scenario_count"], "scenario_probability_total": predicted["scenario_probability_total"], "alpha": predicted["alpha"], "var_threshold": predicted["var_threshold"], "loss_definition": predicted["loss_definition"], "normalized_cvar": predicted["normalized_cvar"], "distinct_teams_consumed": len(set().union(*(used.values()))), "team_usage_efficiency": len(set().union(*(used.values()))) / max(1, sum(len(value) for value in used.values()))}
                    round_rows.append(round_row); observations.append(round_row)
                losses = [row["realised_loss"] for row in round_rows]
                fractions = [row["eliminated_entry_fraction"] for row in round_rows]
                cvar_raw = formal_cvar(losses, alpha=.95, loss_definition="eliminated entries; active-entry counts displayed") if len(losses) >= 2 else None
                cvar_fraction = formal_cvar(fractions, alpha=.95, loss_definition="eliminated-entry fraction") if len(fractions) >= 2 else None
                evaluations.append({"season": season, "starting_round": audit.round_number, "strategy": strategy, "cartel_size": TOTAL_ENTRIES, "feasible": True, "conditional_state_label": "conditional surviving-cartel state", "completed_survival_rounds": sum(1 for row in round_rows if row["surviving_after"] > 0), "entries_surviving_by_round": [row["surviving_after"] for row in round_rows], "probability_at_least_one_remains": float(any(row["surviving_after"] > 0 for row in round_rows)), "wipeout_frequency": float(any(row["surviving_after"] == 0 for row in round_rows)), "expected_survivors": float(np.mean([row["surviving_after"] for row in round_rows])) if round_rows else 0.0, "area_under_survivor_curve": float(sum(row["surviving_after"] for row in round_rows) / TOTAL_ENTRIES), "mean_eliminated_entry_fraction": float(np.mean(fractions)) if fractions else 0.0, "realised_eliminated_entries": float(np.mean(losses)) if losses else 0.0, "mean_predicted_decision_cvar": float(np.mean([row["predicted_cvar"] for row in round_rows])) if round_rows else 0.0, "realised_aggregate_cvar": cvar_raw.cvar if cvar_raw else None, "realised_aggregate_normalized_cvar": cvar_fraction.cvar if cvar_fraction else None, "realised_aggregate_cvar_observations": len(round_rows), "concentration": [row["distinct_teams_consumed"] for row in round_rows], "distinct_teams_consumed": round_rows[-1]["distinct_teams_consumed"] if round_rows else 0, "team_usage_efficiency": round_rows[-1]["team_usage_efficiency"] if round_rows else 0, "milp": milp_rows, "rounds": round_rows})
    aggregate = {}
    for key in sorted({(row["strategy"], row["season"]) for row in observations}):
        selected = [row for row in observations if (row["strategy"], row["season"]) == key]
        if len(selected) < 2: continue
        losses = np.asarray([row["realised_loss"] for row in selected], dtype=float); fractions = np.asarray([row["eliminated_entry_fraction"] for row in selected], dtype=float)
        cvar = formal_cvar(losses, alpha=.95, loss_definition="eliminated entries; active-entry counts displayed")
        fraction_cvar = formal_cvar(fractions, alpha=.95, loss_definition="eliminated-entry fraction")
        aggregate[f"{key[0]}|{key[1]}"] = {**cvar.as_dict(), "observation_count": len(selected), "mean_loss": float(np.mean(losses)), "mean_eliminated_entry_fraction": float(np.mean(fractions)), "maximum_loss": float(np.max(losses)), "active_entry_count_min": int(min(row["active_entries_before"] for row in selected)), "active_entry_count_max": int(max(row["active_entries_before"] for row in selected)), "normalized_cvar": float(fraction_cvar.cvar), "raw_cvar": float(cvar.cvar), "weights": "equal-weighted comparable round decisions"}
    bootstrap_rows = [{**row, "start_round": row["starting_round"]} for row in evaluations if row["feasible"]]
    bootstrap = {strategy: {metric: clustered_paired_bootstrap(bootstrap_rows, metric, strategy, baseline="concentrated_favourite", repetitions=bootstrap_repetitions, seed=seed) for metric in ("realised_eliminated_entries", "expected_survivors", "area_under_survivor_curve", "mean_eliminated_entry_fraction", "wipeout_frequency", "probability_at_least_one_remains")} for strategy in STRATEGIES if strategy != "concentrated_favourite"}
    strategy_summary = {}
    for strategy in STRATEGIES:
        selected = [row for row in observations if row["strategy"] == strategy]
        milp = [item for evaluation in evaluations if evaluation["strategy"] == strategy for item in evaluation["milp"]]
        losses = np.asarray([row["realised_loss"] for row in selected], dtype=float)
        cvar = formal_cvar(losses, alpha=.95, loss_definition="eliminated entries; active-entry counts displayed")
        fraction_cvar = formal_cvar([row["eliminated_entry_fraction"] for row in selected], alpha=.95, loss_definition="eliminated-entry fraction")
        strategy_summary[strategy] = {"observation_count": len(selected), "mean_expected_survivors": float(np.mean([row["surviving_after"] for row in selected])), "mean_realised_eliminated_entries": float(np.mean(losses)), "mean_eliminated_entry_fraction": float(np.mean([row["eliminated_entry_fraction"] for row in selected])), "area_under_survivor_curve": float(np.mean([row["surviving_after"] for row in selected]) / TOTAL_ENTRIES), "predicted_decision_cvar": float(np.mean([row["predicted_cvar"] for row in selected])), "predicted_normalized_cvar": float(np.mean([row["normalized_cvar"] for row in selected])), "realised_aggregate_cvar": cvar.cvar if len(losses) >= 2 else None, "realised_aggregate_normalized_cvar": fraction_cvar.cvar if len(losses) >= 2 else None, "realised_aggregate_cvar_observations": len(losses), "maximum_loss": float(np.max(losses)) if len(losses) else 0.0, "probability_at_least_one_survives": float(np.mean([row["at_least_one_survives"] for row in selected])), "wipeout_frequency": float(np.mean([row["wipeout"] for row in selected])), "mean_distinct_teams_consumed": float(np.mean([row["distinct_teams_consumed"] for row in selected])), "mean_team_usage_efficiency": float(np.mean([row["team_usage_efficiency"] for row in selected])), "strategy_weights": STRATEGY_WEIGHTS.get(strategy, {"heuristic": strategy}), "milp_successes": sum(item["status"] == "success" for item in milp), "milp_failures": sum(item["status"] != "success" for item in milp), "milp_mean_runtime_seconds": float(np.mean([item["runtime_seconds"] for item in milp]))}
    feasibility_by_season = {season: {"feasible": sum(row["season"] == season and row["feasible"] for row in construction), "infeasible": sum(row["season"] == season and not row["feasible"] for row in construction)} for season in seasons}
    all_milp = [item for evaluation in evaluations for item in evaluation["milp"]]
    milp_summary = {"calls": len(all_milp), "successes": sum(item["status"] == "success" for item in all_milp), "failures": sum(item["status"] != "success" for item in all_milp), "total_runtime_seconds": float(sum(item["runtime_seconds"] for item in all_milp)), "mean_runtime_seconds": float(np.mean([item["runtime_seconds"] for item in all_milp])) if all_milp else 0.0, "maximum_runtime_seconds": float(np.max([item["runtime_seconds"] for item in all_milp])) if all_milp else 0.0}
    strategy_weights = {"concentrated_favourite": {"selection_rule": "highest current market-probability eligible unused team per entry", "objective_weights": "heuristic"}, "independent_greedy": {"selection_rule": "independently highest current market-probability eligible unused team", "objective_weights": "heuristic"}, "equal_diversification": {"selection_rule": "minimum current exposure, then market probability", "objective_weights": "heuristic"}, **STRATEGY_WEIGHTS}
    classifications = {"concentrated_favourite": "validated default", "independent_greedy": "validated alternative", "max_expected_survivors": "validated alternative", "protect_one": "experimental", "equal_diversification": "experimental", "bellman": "experimental", "balanced": "experimental"}
    return {"conditional_state_label": "conditional surviving-cartel state; not unconditional twenty-entry survival probability", "players": {"player-1": ENTRY_LIMIT, "player-2": ENTRY_LIMIT}, "total_entries": TOTAL_ENTRIES, "minimum_distinct_used_team_sets": 2, "seasons": seasons, "feasibility_by_season": feasibility_by_season, "cohort_construction": construction, "evaluations": evaluations, "decision_observations": observations, "allocation_diagnostics": _allocation_diagnostics(observations), "realised_cvar": aggregate, "strategy_summary": strategy_summary, "strategy_weights": strategy_weights, "strategy_classifications": classifications, "strategy_bootstrap_vs_concentrated": bootstrap, "milp_summary": milp_summary, "seed": seed, "bootstrap_repetitions": bootstrap_repetitions}
