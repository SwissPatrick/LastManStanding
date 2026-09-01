"""Retrospective LMS policy evaluation on real matchdays."""
from collections import defaultdict
import numpy as np
from .models import HistoricalMatch
from .probability import proportional
from .modeling import DixonColesModel
from .elo import EloModel

def retrospective_rounds(matches: list[HistoricalMatch], season: str) -> list[list[HistoricalMatch]]:
    rows = sorted((m for m in matches if m.season == season), key=lambda m: m.match_date)
    grouped = defaultdict(list)
    for match in rows: grouped[(match.match_date.isocalendar().year, match.match_date.isocalendar().week)].append(match)
    return [group for group in grouped.values() if len(group) >= 6]

def evaluate_lms_policies(matches: list[HistoricalMatch], season: str, seed: int = 7) -> dict[str, dict[str, float]]:
    rounds = retrospective_rounds(matches, season)
    if not rounds: raise ValueError("no retrospective rounds with at least six matches")
    training = [m for m in matches if m.season < season]
    dc = DixonColesModel().fit(training) if len(training) >= 100 else None
    elo = EloModel().fit(training) if len(training) >= 1 else None
    rng = np.random.default_rng(seed)
    policies = ("shortest-priced", "highest-market-win", "highest-dixon-coles-win", "highest-elo-win", "random", "save-elite")
    results = {policy: [] for policy in policies}
    for _ in range(250):
        used = {policy: set() for policy in policies}; alive = {policy: True for policy in policies}; survived = {policy: 0 for policy in policies}
        for round_matches in rounds:
            for policy in policies:
                if not alive[policy]: continue
                candidates = [(match, 0 if match.full_time_home_goals > match.full_time_away_goals else 2 if match.full_time_away_goals > match.full_time_home_goals else 1) for match in round_matches]
                candidates = [(m, result) for m, result in candidates if m.home_team not in used[policy] and m.away_team not in used[policy]]
                if not candidates: continue
                def value(item):
                    match, _ = item
                    market = proportional([match.closing_home_odds, match.closing_draw_odds, match.closing_away_odds])
                    p = market
                    if policy == "highest-dixon-coles-win" and dc is not None:
                        try: p = dc.predict(match.home_team, match.away_team).outcome
                        except ValueError: pass
                    elif policy == "highest-elo-win" and elo is not None: p = np.asarray(elo.probabilities(match.home_team, match.away_team))
                    elif policy == "save-elite":
                        p = market
                        if max(market[0], market[2]) > .55: return -1.0
                    return min(match.closing_home_odds, match.closing_away_odds) if policy == "shortest-priced" else max(p[0], p[2])
                if policy == "random": chosen, result = candidates[int(rng.integers(len(candidates)))]
                elif policy == "shortest-priced": chosen, result = min(candidates, key=value)
                else: chosen, result = max(candidates, key=value)
                chosen_prob = proportional([chosen.closing_home_odds, chosen.closing_draw_odds, chosen.closing_away_odds])
                if policy == "highest-dixon-coles-win" and dc is not None:
                    try: chosen_prob = dc.predict(chosen.home_team, chosen.away_team).outcome
                    except ValueError: pass
                elif policy == "highest-elo-win" and elo is not None: chosen_prob = np.asarray(elo.probabilities(chosen.home_team, chosen.away_team))
                chosen_team = chosen.home_team if (policy == "shortest-priced" and chosen.closing_home_odds <= chosen.closing_away_odds) or (policy != "shortest-priced" and chosen_prob[0] >= chosen_prob[2]) else chosen.away_team
                used[policy].add(chosen_team)
                actual_team = chosen.home_team if result == 0 else chosen.away_team if result == 2 else None
                if actual_team != chosen_team: alive[policy] = False
                else: survived[policy] += 1
        for policy in policies: results[policy].append(survived[policy])
    return {policy: {"median_survival_rounds": float(np.median(values)), "mean_survival_rounds": float(np.mean(values)), "probability_season_completion": float(np.mean(np.asarray(values) == len(rounds))), "round_count": len(rounds)} for policy, values in results.items()}
