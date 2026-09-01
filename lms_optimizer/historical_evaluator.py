"""Leakage-safe historical LMS cohort evaluator."""
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from pathlib import Path
import hashlib, json, time
import numpy as np
from .models import HistoricalMatch
from .probability import proportional
from .optimizer import DynamicProgram
from .elo import EloModel
from .market_strength import MarketStrengthModel
from .cvar import formal_cvar

_FORECAST_CACHE = {}
_DP_CACHE = {}
_DECISION_CVAR_CACHE = {}
_DECISION_CVAR_CACHE_HITS = 0
_DECISION_CVAR_CACHE_MISSES = 0

@dataclass
class ConstructedRound:
    season: str
    round_number: int
    start_date: str
    end_date: str
    included_fixtures: list[str]
    excluded_fixtures: list[dict[str, str]]
    match_count: int
    eligible: bool
    warnings: list[str]
    cutoff: str

def fixture_key(match: HistoricalMatch) -> str:
    return f"{match.season}|{match.match_date.isoformat()}|{match.home_team}|{match.away_team}"

def construct_rounds(matches: list[HistoricalMatch], season: str, minimum_matches: int = 6) -> tuple[list[list[HistoricalMatch]], list[ConstructedRound]]:
    rows = sorted((m for m in matches if m.season == season), key=lambda m: (m.match_date, m.home_team, m.away_team))
    groups: list[list[HistoricalMatch]] = []; excluded: list[dict[str, str]] = []
    for match in rows:
        placed = False
        for group in reversed(groups[-2:]):
            first = group[0]; teams = {t for m in group for t in (m.home_team, m.away_team)}
            if (match.match_date.date() - first.match_date.date()).days <= 3 and len(group) < 10 and not ({match.home_team, match.away_team} & teams):
                group.append(match); placed = True; break
        if not placed:
            groups.append([match])
    audits = []
    for index, group in enumerate(groups, 1):
        start, end = min(m.match_date for m in group), max(m.match_date for m in group)
        eligible = len(group) >= minimum_matches
        warnings = []
        if len(group) == 10: warnings.append("ten-fixture cap reached")
        if end.date() - start.date() > timedelta(days=2): warnings.append("multi-day/rescheduled window")
        if index > 1: warnings.append("multi-day/rescheduled separation from previous group: calendar gap, team collision or ten-fixture cap")
        audits.append(ConstructedRound(season, index, start.isoformat(), end.isoformat(), [fixture_key(m) for m in group], [], len(group), eligible, warnings, start.isoformat()))
    return [group for group in groups if len(group) >= minimum_matches], audits

def _current_probabilities(round_matches: list[HistoricalMatch]) -> dict[str, float]:
    values = {}
    for match in round_matches:
        p = proportional([match.closing_home_odds, match.closing_draw_odds, match.closing_away_odds])
        values[match.home_team] = max(values.get(match.home_team, 0), float(p[0]))
        values[match.away_team] = max(values.get(match.away_team, 0), float(p[2]))
    return values

def _choose(round_matches, used, probabilities, mode):
    candidates = []
    for match in round_matches:
        p = proportional([match.closing_home_odds, match.closing_draw_odds, match.closing_away_odds])
        candidates += [(match.home_team, float(p[0])), (match.away_team, float(p[2]))]
    candidates = [(team, p) for team, p in candidates if team not in used]
    if not candidates: return None
    if mode == "shortest": return min(candidates, key=lambda x: next(m.closing_home_odds if m.home_team == x[0] else m.closing_away_odds for m in round_matches if x[0] in (m.home_team, m.away_team)))
    return max(candidates, key=lambda x: (x[1], x[0]))

def _actual_winner(match, team):
    return match.home_team if match.full_time_home_goals > match.full_time_away_goals and team == match.home_team else match.away_team if match.full_time_away_goals > match.full_time_home_goals and team == match.away_team else None

def _exact_decision_cvar(round_matches, allocation, active, alpha=.95):
    """Exact loss distribution for one allocation, using fixture-level outcomes.

    Entries selecting the same team share one outcome. Opposing selections in a
    fixture are mutually exclusive because only H/D/A can occur. Convolution of
    fixture loss polynomials is equivalent to enumerating all match scenarios,
    without materialising 3**fixtures scenario rows.
    """
    key = (tuple((fixture_key(m), m.closing_home_odds, m.closing_draw_odds, m.closing_away_odds) for m in round_matches), tuple(sorted(allocation.items())), len(active), alpha)
    global _DECISION_CVAR_CACHE_HITS, _DECISION_CVAR_CACHE_MISSES
    if key in _DECISION_CVAR_CACHE:
        _DECISION_CVAR_CACHE_HITS += 1
        return _DECISION_CVAR_CACHE[key]
    _DECISION_CVAR_CACHE_MISSES += 1
    distribution = {0: 1.0}
    relevant_fixtures = 0
    for match in round_matches:
        assigned = [team for team in allocation.values() if team in (match.home_team, match.away_team)]
        if not assigned:
            continue
        relevant_fixtures += 1
        probabilities = proportional([match.closing_home_odds, match.closing_draw_odds, match.closing_away_odds])
        local_losses = [sum(team != match.home_team for team in assigned), len(assigned), sum(team != match.away_team for team in assigned)]
        next_distribution = {}
        for previous, previous_probability in distribution.items():
            for loss, probability in zip(local_losses, probabilities):
                next_distribution[previous + loss] = next_distribution.get(previous + loss, 0.0) + previous_probability * float(probability)
        distribution = next_distribution
    # A recorded allocation should have one fixture per active entry. Defensive
    # handling keeps malformed historical rows finite and auditable.
    if not distribution:
        distribution = {len(active): 1.0}
    losses = np.asarray(sorted(distribution), dtype=float)
    weights = np.asarray([distribution[int(loss)] for loss in losses], dtype=float)
    result = formal_cvar(losses, weights, alpha, "eliminated entries")
    result_data = {**result.as_dict(), "scenario_count": int(3 ** relevant_fixtures), "scenario_probability_total": 1.0,
                   "normalized_cvar": float(result.cvar / max(1, len(active))), "predicted_wipeout_probability": float(distribution.get(len(active), 0.0))}
    _DECISION_CVAR_CACHE[key] = result_data
    return result_data

def evaluate_cohort(rounds: list[list[HistoricalMatch]], start_index: int, size: int, strategy: str, seed: int = 7, all_matches: list[HistoricalMatch] | None = None) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    entries = {f"e{i+1}": True for i in range(size)}; used = {entry: set() for entry in entries}; survival_rounds = {entry: 0 for entry in entries}; decisions=[]; round_metrics=[]
    for relative, round_matches in enumerate(rounds[start_index:], 0):
        active = [entry for entry, alive in entries.items() if alive]
        if not active: break
        current = _current_probabilities(round_matches)
        allocation = {}
        for entry in active:
            allocation[entry] = _choose(round_matches, used[entry], current, "shortest" if strategy == "concentrated_favourite" else "market")
        if strategy == "equal_diversification":
            ranked = sorted(current, key=lambda t: (-current[t], t)); allocation = {entry: ranked[i % len(ranked)] for i, entry in enumerate(active)}
            allocation = {entry: team if team not in used[entry] else _choose(round_matches, used[entry], current, "market") for entry, team in allocation.items()}
        elif strategy in ("bellman", "max_expected_survivors", "protect_one", "balanced"):
            forecast_rounds = {start_index+relative: current}
            available = {start_index+relative: list(current)}
            prior_matches = [m for m in (all_matches or []) if m.match_date < min(x.match_date for x in round_matches)]
            cache_key = min(x.match_date for x in round_matches)
            if strategy == "bellman" and cache_key not in _FORECAST_CACHE: _FORECAST_CACHE[cache_key] = EloModel().fit(prior_matches) if prior_matches else None
            future_model = _FORECAST_CACHE.get(cache_key)
            if strategy == "bellman":
                for future_index, future in enumerate(rounds[start_index+relative+1:], start_index+relative+1):
                    future_probs = {}
                    for match in future:
                        p = np.asarray(future_model.probabilities(match.home_team, match.away_team)) if future_model else np.ones(3)/3
                        future_probs[match.home_team] = max(future_probs.get(match.home_team, 0.), float(p[0])); future_probs[match.away_team] = max(future_probs.get(match.away_team, 0.), float(p[2]))
                    forecast_rounds[future_index] = future_probs
                    # Horizon pruning: retain the six strongest future candidates.
                    available[future_index] = sorted(future_probs, key=lambda t: (-future_probs[t], t))[:6]
            if strategy == "bellman":
                for entry in active:
                    # A six-round horizon keeps the all-season replay tractable;
                    # the pruning is recorded in the report configuration.
                    dp_key = (cache_key, start_index + relative, tuple(sorted(used[entry])))
                    if dp_key not in _DP_CACHE:
                        _DP_CACHE[dp_key] = DynamicProgram(forecast_rounds, available, frozenset(used[entry]), horizon=start_index+relative+1).solve()
                    values = _DP_CACHE[dp_key]
                    allocation[entry] = values[0].team if values else allocation[entry]
            else:
                candidates = {entry: [t for t in current if t not in used[entry]] for entry in active}
                ranked = sorted(current, key=lambda t: (-current[t], t))
                if strategy == "protect_one":
                    allocation = {entry: next((t for t in ranked if t in candidates[entry]), candidates[entry][0]) for entry in active}
                elif strategy == "balanced":
                    allocation = {entry: next((t for t in ranked[i % len(ranked):] + ranked[:i % len(ranked)] if t in candidates[entry]), candidates[entry][0]) for i, entry in enumerate(active)}
                else:
                    allocation = {entry: max(candidates[entry], key=lambda t: (current[t], t)) for entry in active if candidates[entry]}
        cutoff = min(m.match_date for m in round_matches).isoformat()
        predicted = _exact_decision_cvar(round_matches, allocation, active)
        survived = 0
        for entry, team in allocation.items():
            used[entry].add(team); match = next((m for m in round_matches if team in (m.home_team,m.away_team)), None)
            if match is None or _actual_winner(match, team) is None: entries[entry] = False
            else: survived += 1; survival_rounds[entry] += 1
        realised_loss = len(active) - survived
        decision_fields = {"season": round_matches[0].season, "start_round": start_index+1, "constructed_round": start_index+relative+1, "cutoff": cutoff, "strategy": strategy, "cartel_size": size,
                           "active_entries_before": len(active), "scenario_count": predicted["scenario_count"], "scenario_probability_total": predicted["scenario_probability_total"],
                           "alpha": predicted["alpha"], "var_threshold": predicted["var_threshold"], "cvar": predicted["cvar"], "normalized_cvar": predicted["normalized_cvar"],
                           "expected_loss": predicted["expected_loss"], "realised_loss": realised_loss, "information_cutoff": cutoff, "loss_definition": predicted["loss_definition"],
                           "predicted_wipeout_probability": predicted["predicted_wipeout_probability"]}
        decisions.extend({**decision_fields, "entry": entry, "team": team, "odds_timing": next((m.odds_timing for m in round_matches if team in (m.home_team,m.away_team)), "timestamp-unknown")} for entry, team in allocation.items() if team)
        round_metrics.append({"season": round_matches[0].season, "start_round": start_index+1, "constructed_round": start_index+relative+1, "strategy": strategy, "cartel_size": size, "cutoff": cutoff, "active_before": len(active), "surviving_after": survived, "survival_fraction": survived/len(active), **decision_fields, "realised_eliminated_entries": realised_loss})
    return decisions, round_metrics, survival_rounds

def clustered_bootstrap(rows: list[dict[str, object]], metric: str, repetitions: int = 1000, seed: int = 7) -> tuple[float, float]:
    seasons = sorted({row["season"] for row in rows})
    if not seasons: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); values=[]
    for _ in range(repetitions):
        sample = rng.choice(seasons, len(seasons), replace=True)
        selected = [row for season in sample for row in rows if row["season"] == season]
        values.append(float(np.mean([row[metric] for row in selected])))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

def clustered_paired_bootstrap(rows: list[dict[str, object]], metric: str, strategy: str, baseline: str = "concentrated_favourite", repetitions: int = 1000, seed: int = 7) -> tuple[float, float]:
    """Season-clustered percentile interval for strategy minus baseline."""
    keys = sorted({(r["season"], r["start_round"], r["cartel_size"]) for r in rows})
    seasons = sorted({r["season"] for r in rows}); by_key = defaultdict(dict)
    for row in rows: by_key[(row["season"], row["start_round"], row["cartel_size"])][row["strategy"]] = float(row[metric])
    rng = np.random.default_rng(seed); values = []
    for _ in range(repetitions):
        sampled = rng.choice(seasons, len(seasons), replace=True); diffs = []
        for season in sampled:
            for key in keys:
                if key[0] == season and strategy in by_key[key] and baseline in by_key[key]:
                    diffs.append(by_key[key][strategy] - by_key[key][baseline])
        values.append(float(np.mean(diffs)) if diffs else float("nan"))
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))

def clustered_cvar_bootstrap(rows: list[dict[str, object]], loss_key: str = "realised_loss", fraction_key: str | None = None, alpha: float = .95, repetitions: int = 1000, seed: int = 7) -> tuple[float, float]:
    """Season-clustered percentile interval for empirical CVaR."""
    seasons = sorted({row["season"] for row in rows})
    if not rows or not seasons: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); values = []
    for _ in range(repetitions):
        sampled = rng.choice(seasons, len(seasons), replace=True)
        selected = [row for season in sampled for row in rows if row["season"] == season]
        losses = np.asarray([float(row[loss_key]) / float(row[fraction_key]) if fraction_key else float(row[loss_key]) for row in selected])
        values.append(formal_cvar(losses, alpha=alpha).cvar)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

def _aggregate_cvar(rows: list[dict[str, object]], alpha: float, bootstrap_repetitions: int, seed: int, fraction: bool = False) -> dict[str, dict[str, object]]:
    """Formal realised CVaR summaries; each observation has equal weight."""
    grouped = defaultdict(list)
    for row in rows:
        key = (row["strategy"],) if fraction else (row["strategy"], row["cartel_size"], row["season"])
        grouped[key].append(row)
    result = {}
    for key, observations in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        if len(observations) < 2:
            raise ValueError("aggregate CVaR requires at least two comparable observations")
        losses = np.asarray([float(x["realised_loss"]) / float(x["cartel_size"]) if fraction else float(x["realised_loss"]) for x in observations])
        summary = formal_cvar(losses, alpha=alpha, loss_definition="eliminated-entry fraction" if fraction else "eliminated entries")
        bootstrap = clustered_cvar_bootstrap(observations, "realised_loss", "cartel_size" if fraction else None, alpha, bootstrap_repetitions, seed)
        label = "|".join(map(str, key))
        denominator = max(1, int(observations[0]["cartel_size"])) if not fraction else 1
        summary_data = summary.as_dict() | {"observation_count": len(observations), "mean_loss": float(np.mean(losses)), "maximum_loss": float(np.max(losses)), "normalized_cvar": float(summary.cvar / denominator), "clustered_season_bootstrap_interval": bootstrap, "weights": "equal-weighted observations"}
        result[label] = summary_data
    return result

def _aggregate_cvar_overall(rows, alpha, bootstrap_repetitions, seed):
    if len(rows) < 2:
        raise ValueError("aggregate CVaR requires at least two comparable observations")
    losses = np.asarray([float(x["realised_loss"]) for x in rows])
    summary = formal_cvar(losses, alpha=alpha, loss_definition="eliminated entries")
    return summary.as_dict() | {"observation_count": len(rows), "mean_loss": float(np.mean(losses)), "maximum_loss": float(np.max(losses)), "normalized_cvar": float(summary.cvar / max(1, int(rows[0]["cartel_size"]))), "clustered_season_bootstrap_interval": clustered_cvar_bootstrap(rows, alpha=alpha, repetitions=bootstrap_repetitions, seed=seed), "weights": "equal-weighted observations"}

def _diagnostics(rows, aggregate, alpha):
    diagnostics = {}
    for key in sorted({(x["strategy"], x["cartel_size"]) for x in rows}, key=lambda x: (str(x[0]), int(x[1]))):
        strategy, size = key; selected = [x for x in rows if x["strategy"] == strategy and x["cartel_size"] == size]
        aggregate_row = aggregate[f"{strategy}|{size}"]
        diagnostics[f"{strategy}|{size}"] = {"observation_count": len(selected), "mean_predicted_expected_loss": float(np.mean([x["expected_loss"] for x in selected])), "mean_realised_loss": float(np.mean([x["realised_loss"] for x in selected])), "mean_predicted_cvar": float(np.mean([x["cvar"] for x in selected])), "realised_aggregate_cvar": aggregate_row["cvar"], "predicted_wipeout_probability": float(np.mean([x["predicted_wipeout_probability"] for x in selected])), "realised_wipeout_frequency": float(np.mean([x["realised_loss"] >= x["active_entries_before"] for x in selected])), "alpha": alpha}
    return diagnostics

def evaluate_all_seasons(matches: list[HistoricalMatch], seed: int = 7, bootstrap_repetitions: int = 1000) -> dict[str, object]:
    global _DECISION_CVAR_CACHE_HITS, _DECISION_CVAR_CACHE_MISSES
    _DECISION_CVAR_CACHE.clear(); _DECISION_CVAR_CACHE_HITS = 0; _DECISION_CVAR_CACHE_MISSES = 0
    seasons = sorted({m.season for m in matches}); strategies = ("concentrated_favourite","equal_diversification","independent_greedy","bellman","max_expected_survivors","protect_one","balanced")
    rounds_audit=[]; decisions=[]; metrics=[]; cvar_observations=[]
    for season in seasons:
        rounds, audit = construct_rounds(matches, season); rounds_audit.extend(asdict(a) for a in audit)
        for start in range(len(rounds)):
            for size in (1,3,5,10):
                for strategy in strategies:
                    d, r, survival_by_entry = evaluate_cohort(rounds, start, size, strategy, seed, matches); decisions.extend(d)
                    cvar_observations.extend(r)
                    completed_values = list(survival_by_entry.values()); losses = np.asarray([size - v for v in completed_values], dtype=float)
                    survivor_path = [int(x["surviving_after"]) for x in r]
                    metrics.append({"season":season,"start_round":start+1,"cartel_size":size,"strategy":strategy,
                        "completed_survival_rounds":float(np.sum(completed_values)),"mean_survival_rounds":float(np.mean(completed_values)),"median_survival_rounds":float(np.median(completed_values)),
                        "survival_distribution":json.dumps(completed_values),"survivors_by_round":json.dumps(survivor_path),
                        "probability_at_least_one_by_round":json.dumps([float(x > 0) for x in survivor_path]),
                        "wipeout_frequency":float(any(x == 0 for x in survivor_path)),"expected_surviving_entries":float(np.mean(survivor_path)) if survivor_path else 0.,
                        "eliminated_entry_loss":float(np.mean(losses)) if losses.size else 0.,"realised_eliminated_entries":float(np.mean(losses)) if losses.size else 0.,
                        "distinct_teams_consumed":len({x["team"] for x in d}),"team_usage_efficiency":len({x["team"] for x in d}) / max(1, len(d)),
                        "concentration_by_round":json.dumps([len({x["team"] for x in d if x["constructed_round"] == rm["constructed_round"]}) for rm in r]),
                        "decision_points":len(r),"cohorts":1})
    bootstrap = {strategy: {metric: clustered_bootstrap([m for m in metrics if m["strategy"] == strategy], metric, bootstrap_repetitions, seed) for metric in ("mean_survival_rounds","wipeout_frequency","expected_surviving_entries")} for strategy in strategies}
    paired = {strategy: {metric: clustered_paired_bootstrap(metrics, metric, strategy, repetitions=bootstrap_repetitions, seed=seed) for metric in ("mean_survival_rounds","wipeout_frequency","expected_surviving_entries")} for strategy in strategies if strategy != "concentrated_favourite"}
    cvar_alpha = .95
    aggregate_by_season = _aggregate_cvar(cvar_observations, cvar_alpha, bootstrap_repetitions, seed)
    aggregate_overall = {}
    for strategy in strategies:
        for size in (1, 3, 5, 10):
            selected = [x for x in cvar_observations if x["strategy"] == strategy and x["cartel_size"] == size]
            if selected:
                aggregate_overall[f"{strategy}|{size}"] = _aggregate_cvar(selected, cvar_alpha, bootstrap_repetitions, seed, False).pop(f"{strategy}|{size}|{selected[0]['season']}", None) if len({x["season"] for x in selected}) == 1 else _aggregate_cvar_overall(selected, cvar_alpha, bootstrap_repetitions, seed)
    diagnostics = _diagnostics(cvar_observations, aggregate_overall, cvar_alpha)
    return {"seasons":seasons,"starting_points":len({(m["season"],m["start_round"]) for m in metrics}),"decisions":decisions,"cohort_metrics":metrics,"round_construction_audit":rounds_audit,"clustered_bootstrap":bootstrap,"paired_bootstrap_vs_concentrated":paired,"predicted_decision_cvar":cvar_observations,"realised_cvar_by_strategy_cartel_season":aggregate_by_season,"realised_cvar_overall":aggregate_overall,"realised_cvar_cross_size_fraction":_aggregate_cvar(cvar_observations, cvar_alpha, bootstrap_repetitions, seed, True),"predicted_vs_realised_diagnostics":diagnostics,"cvar_alpha":cvar_alpha,"cvar_weights":"equal-weighted comparable decisions; scenario weights normalised per decision","cvar_cache":{"entries":len(_DECISION_CVAR_CACHE),"hits":_DECISION_CVAR_CACHE_HITS,"misses":_DECISION_CVAR_CACHE_MISSES},"bootstrap_repetitions":bootstrap_repetitions,"seed":seed,"model_versions":{"current_probability":"proportional-market-consensus","future_probability":"chronological-elo","evaluator":"historical-cartel-v2"}}
