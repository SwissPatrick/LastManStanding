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
from .optimizer import PortfolioWeights
from .probability import proportional

ENTRY_LIMIT = 10
TOTAL_ENTRIES = 20
STRATEGIES = ("concentrated_favourite", "equal_diversification", "independent_greedy", "bellman", "max_expected_survivors", "protect_one", "balanced")


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


def _future_values(matches, rounds, cutoff, candidates):
    model = EloModel().fit([match for match in matches if match.match_date < cutoff])
    values = {entry: {team: 0.0 for team in teams} for entry, teams in candidates.items()}
    for future in rounds:
        if not future or min(match.match_date for match in future) <= cutoff:
            continue
        for entry, teams in values.items():
            for team in teams:
                for match in future:
                    if team == match.home_team: values[entry][team] += float(model.probabilities(match.home_team, match.away_team)[0])
                    elif team == match.away_team: values[entry][team] += float(model.probabilities(match.home_team, match.away_team)[2])
    return values


def _allocate(round_matches, active_entries, strategy, used, all_matches, all_rounds, cutoff):
    current = _current_probabilities(round_matches)
    candidates = {entry.entry_id: sorted({team for match in round_matches for team in (match.home_team, match.away_team) if team not in used[entry.entry_id]}, key=lambda team: (-current.get(team, 0), team)) for entry in active_entries}
    candidates = {entry: teams for entry, teams in candidates.items() if teams}
    scenarios, probabilities = _milp_scenarios(round_matches)
    future = _future_values(all_matches, all_rounds, cutoff, candidates)
    # Keep the completed MILP's independently validated objective active. The
    # strategy label changes future-value treatment; concentration remains a
    # reporting metric because the MILP's concentration linearisation is not a
    # reliable tie-break for this twenty-entry replay.
    weights = PortfolioWeights(concentration=0.0, future_value=1.0 if strategy in ("bellman", "max_expected_survivors") else 0.0)
    started = perf_counter()
    exposure_cap = 2 if strategy == "equal_diversification" else 3 if strategy == "balanced" else None
    result = milp_optimize(candidates, scenarios, weights=weights, exposure_cap=exposure_cap, scenario_probabilities=probabilities, future_values=future)
    elapsed = perf_counter() - started
    if result.feasible:
        return result.allocation, {"status": "success", "message": result.message, "runtime_seconds": elapsed, "objective": result.objective}
    # The MILP is authoritative for twenty-entry allocation; this fallback is
    # only a recorded failure path and never silently claims MILP success.
    allocation = {entry.entry_id: teams[0] for entry, teams in zip(active_entries, (candidates.get(e.entry_id, []) for e in active_entries)) if teams}
    return allocation, {"status": "failed", "message": result.message, "runtime_seconds": elapsed, "objective": None}


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
                    round_row = {"season": season, "starting_round": audit.round_number, "constructed_round": round_matches[0].match_date.isoformat(), "strategy": strategy, "cartel_size": TOTAL_ENTRIES, "active_entries_before": len(active_entries), "surviving_after": survived, "realised_loss": realised_loss, "predicted_cvar": predicted["cvar"], "predicted_expected_loss": predicted["expected_loss"], "predicted_wipeout_probability": predicted["predicted_wipeout_probability"], "milp_status": milp_info["status"], "milp_runtime_seconds": milp_info["runtime_seconds"], "information_cutoff": cutoff.isoformat(), "scenario_count": predicted["scenario_count"], "scenario_probability_total": predicted["scenario_probability_total"], "alpha": predicted["alpha"], "var_threshold": predicted["var_threshold"], "loss_definition": predicted["loss_definition"], "normalized_cvar": predicted["normalized_cvar"], "distinct_teams_consumed": len(set().union(*(used.values()))), "team_usage_efficiency": len(set().union(*(used.values()))) / max(1, sum(len(value) for value in used.values()))}
                    round_rows.append(round_row); observations.append(round_row)
                evaluations.append({"season": season, "starting_round": audit.round_number, "strategy": strategy, "cartel_size": TOTAL_ENTRIES, "feasible": True, "conditional_state_label": "conditional surviving-cartel state", "completed_survival_rounds": sum(1 for row in round_rows if row["surviving_after"] > 0), "entries_surviving_by_round": [row["surviving_after"] for row in round_rows], "probability_at_least_one_remains": float(any(row["surviving_after"] > 0 for row in round_rows)), "wipeout_frequency": float(any(row["surviving_after"] == 0 for row in round_rows)), "expected_survivors": float(np.mean([row["surviving_after"] for row in round_rows])) if round_rows else 0.0, "realised_eliminated_entries": float(np.mean([row["realised_loss"] for row in round_rows])) if round_rows else 0.0, "mean_predicted_decision_cvar": float(np.mean([row["predicted_cvar"] for row in round_rows])) if round_rows else 0.0, "realised_aggregate_cvar": formal_cvar([row["realised_loss"] for row in round_rows], alpha=.95, loss_definition="eliminated entries").cvar if len(round_rows) >= 2 else None, "realised_aggregate_cvar_observations": len(round_rows), "concentration": [row["distinct_teams_consumed"] for row in round_rows], "distinct_teams_consumed": round_rows[-1]["distinct_teams_consumed"] if round_rows else 0, "team_usage_efficiency": round_rows[-1]["team_usage_efficiency"] if round_rows else 0, "milp": milp_rows, "rounds": round_rows})
    aggregate = {}
    for key in sorted({(row["strategy"], row["season"]) for row in observations}):
        selected = [row for row in observations if (row["strategy"], row["season"]) == key]
        if len(selected) < 2: continue
        losses = np.asarray([row["realised_loss"] for row in selected], dtype=float); cvar = formal_cvar(losses, alpha=.95, loss_definition="eliminated entries")
        aggregate[f"{key[0]}|{key[1]}"] = {**cvar.as_dict(), "observation_count": len(selected), "mean_loss": float(np.mean(losses)), "maximum_loss": float(np.max(losses)), "normalized_cvar": float(cvar.cvar / TOTAL_ENTRIES), "weights": "equal-weighted comparable round decisions"}
    bootstrap_rows = [{**row, "start_round": row["starting_round"]} for row in evaluations if row["feasible"]]
    bootstrap = {strategy: {metric: clustered_paired_bootstrap(bootstrap_rows, metric, strategy, baseline="concentrated_favourite", repetitions=bootstrap_repetitions, seed=seed) for metric in ("realised_eliminated_entries", "expected_survivors")} for strategy in STRATEGIES if strategy != "concentrated_favourite"}
    strategy_summary = {}
    for strategy in STRATEGIES:
        selected = [row for row in observations if row["strategy"] == strategy]
        milp = [item for evaluation in evaluations if evaluation["strategy"] == strategy for item in evaluation["milp"]]
        losses = np.asarray([row["realised_loss"] for row in selected], dtype=float)
        cvar = formal_cvar(losses, alpha=.95, loss_definition="eliminated entries")
        strategy_summary[strategy] = {"observation_count": len(selected), "mean_expected_survivors": float(np.mean([row["surviving_after"] for row in selected])), "mean_realised_eliminated_entries": float(np.mean(losses)), "predicted_decision_cvar": float(np.mean([row["predicted_cvar"] for row in selected])), "realised_aggregate_cvar": cvar.cvar if len(losses) >= 2 else None, "realised_aggregate_cvar_observations": len(losses), "wipeout_frequency": float(np.mean([row["realised_loss"] >= row["active_entries_before"] for row in selected])), "mean_distinct_teams_consumed": float(np.mean([row["distinct_teams_consumed"] for row in selected])), "mean_team_usage_efficiency": float(np.mean([row["team_usage_efficiency"] for row in selected])), "milp_successes": sum(item["status"] == "success" for item in milp), "milp_failures": sum(item["status"] != "success" for item in milp), "milp_mean_runtime_seconds": float(np.mean([item["runtime_seconds"] for item in milp]))}
    feasibility_by_season = {season: {"feasible": sum(row["season"] == season and row["feasible"] for row in construction), "infeasible": sum(row["season"] == season and not row["feasible"] for row in construction)} for season in seasons}
    all_milp = [item for evaluation in evaluations for item in evaluation["milp"]]
    milp_summary = {"calls": len(all_milp), "successes": sum(item["status"] == "success" for item in all_milp), "failures": sum(item["status"] != "success" for item in all_milp), "total_runtime_seconds": float(sum(item["runtime_seconds"] for item in all_milp)), "mean_runtime_seconds": float(np.mean([item["runtime_seconds"] for item in all_milp])) if all_milp else 0.0, "maximum_runtime_seconds": float(np.max([item["runtime_seconds"] for item in all_milp])) if all_milp else 0.0}
    return {"conditional_state_label": "conditional surviving-cartel state; not unconditional twenty-entry survival probability", "players": {"player-1": ENTRY_LIMIT, "player-2": ENTRY_LIMIT}, "total_entries": TOTAL_ENTRIES, "minimum_distinct_used_team_sets": 2, "seasons": seasons, "feasibility_by_season": feasibility_by_season, "cohort_construction": construction, "evaluations": evaluations, "decision_observations": observations, "realised_cvar": aggregate, "strategy_summary": strategy_summary, "strategy_bootstrap_vs_concentrated": bootstrap, "milp_summary": milp_summary, "seed": seed, "bootstrap_repetitions": bootstrap_repetitions}
