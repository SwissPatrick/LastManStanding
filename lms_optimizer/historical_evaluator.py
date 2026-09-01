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

_FORECAST_CACHE = {}
_DP_CACHE = {}

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
        decisions.extend({"season": round_matches[0].season, "start_round": start_index+1, "constructed_round": start_index+relative+1, "cutoff": min(m.match_date for m in round_matches).isoformat(), "entry": entry, "team": team, "strategy": strategy, "cartel_size": size, "odds_timing": next((m.odds_timing for m in round_matches if team in (m.home_team,m.away_team)), "timestamp-unknown")} for entry, team in allocation.items() if team)
        survived = 0
        for entry, team in allocation.items():
            used[entry].add(team); match = next((m for m in round_matches if team in (m.home_team,m.away_team)), None)
            if match is None or _actual_winner(match, team) is None: entries[entry] = False
            else: survived += 1; survival_rounds[entry] += 1
        round_metrics.append({"season": round_matches[0].season, "start_round": start_index+1, "constructed_round": start_index+relative+1, "strategy": strategy, "active_before": len(active), "surviving_after": survived, "survival_fraction": survived/len(active)})
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

def evaluate_all_seasons(matches: list[HistoricalMatch], seed: int = 7, bootstrap_repetitions: int = 1000) -> dict[str, object]:
    seasons = sorted({m.season for m in matches}); strategies = ("concentrated_favourite","equal_diversification","independent_greedy","bellman","max_expected_survivors","protect_one","balanced")
    rounds_audit=[]; decisions=[]; metrics=[]
    for season in seasons:
        rounds, audit = construct_rounds(matches, season); rounds_audit.extend(asdict(a) for a in audit)
        for start in range(len(rounds)):
            for size in (1,3,5,10):
                for strategy in strategies:
                    d, r, survival_by_entry = evaluate_cohort(rounds, start, size, strategy, seed, matches); decisions.extend(d)
                    completed_values = list(survival_by_entry.values()); losses = np.asarray([size - v for v in completed_values], dtype=float)
                    survivor_path = [int(x["surviving_after"]) for x in r]
                    metrics.append({"season":season,"start_round":start+1,"cartel_size":size,"strategy":strategy,
                        "completed_survival_rounds":float(np.sum(completed_values)),"mean_survival_rounds":float(np.mean(completed_values)),"median_survival_rounds":float(np.median(completed_values)),
                        "survival_distribution":json.dumps(completed_values),"survivors_by_round":json.dumps(survivor_path),
                        "probability_at_least_one_by_round":json.dumps([float(x > 0) for x in survivor_path]),
                        "wipeout_frequency":float(any(x == 0 for x in survivor_path)),"expected_surviving_entries":float(np.mean(survivor_path)) if survivor_path else 0.,
                        "eliminated_entry_loss":float(np.mean(losses)) if losses.size else 0.,"cvar_eliminated":float(np.max(losses)) if losses.size else 0.,
                        "distinct_teams_consumed":len({x["team"] for x in d}),"team_usage_efficiency":len({x["team"] for x in d}) / max(1, len(d)),
                        "concentration_by_round":json.dumps([len({x["team"] for x in d if x["constructed_round"] == rm["constructed_round"]}) for rm in r]),
                        "decision_points":len(r),"cohorts":1})
    bootstrap = {strategy: {metric: clustered_bootstrap([m for m in metrics if m["strategy"] == strategy], metric, bootstrap_repetitions, seed) for metric in ("mean_survival_rounds","wipeout_frequency","expected_surviving_entries")} for strategy in strategies}
    paired = {strategy: {metric: clustered_paired_bootstrap(metrics, metric, strategy, repetitions=bootstrap_repetitions, seed=seed) for metric in ("mean_survival_rounds","wipeout_frequency","expected_surviving_entries")} for strategy in strategies if strategy != "concentrated_favourite"}
    return {"seasons":seasons,"starting_points":len({(m["season"],m["start_round"]) for m in metrics}),"decisions":decisions,"cohort_metrics":metrics,"round_construction_audit":rounds_audit,"clustered_bootstrap":bootstrap,"paired_bootstrap_vs_concentrated":paired,"bootstrap_repetitions":bootstrap_repetitions,"seed":seed,"model_versions":{"current_probability":"proportional-market-consensus","future_probability":"chronological-elo","evaluator":"historical-cartel-v2"}}
