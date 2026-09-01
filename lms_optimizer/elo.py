"""Strictly chronological Elo ratings."""
from dataclasses import dataclass
from math import exp
from .models import HistoricalMatch

@dataclass(frozen=True)
class EloSnapshot:
    date: object
    ratings: dict[str, float]

class EloModel:
    def __init__(self, initial: float = 1500.0, home_advantage: float = 60.0, k_factor: float = 20.0, goal_margin: float = 0.15, regression: float = 0.25) -> None:
        self.initial, self.home_advantage, self.k_factor, self.goal_margin, self.regression = initial, home_advantage, k_factor, goal_margin, regression
        self.ratings: dict[str, float] = {}
        self.snapshots: list[EloSnapshot] = []

    def _rating(self, team: str) -> float:
        return self.ratings.setdefault(team, self.initial)

    def expected(self, home_team: str, away_team: str) -> float:
        return 1 / (1 + 10 ** (-(self._rating(home_team) + self.home_advantage - self._rating(away_team)) / 400))

    def fit(self, matches: list[HistoricalMatch], as_of=None) -> "EloModel":
        self.ratings, self.snapshots = {}, []
        previous_season = None
        for match in sorted(matches, key=lambda m: m.match_date):
            if as_of is not None and match.match_date >= as_of: break
            if previous_season is not None and match.season != previous_season:
                self.ratings = {team: self.initial + self.regression * (rating - self.initial) for team, rating in self.ratings.items()}
            previous_season = match.season
            expected = self.expected(match.home_team, match.away_team)
            actual = 1.0 if match.full_time_home_goals > match.full_time_away_goals else 0.0 if match.full_time_home_goals < match.full_time_away_goals else 0.5
            margin = 1 + self.goal_margin * max(0, abs(match.full_time_home_goals - match.full_time_away_goals) - 1)
            delta = self.k_factor * margin * (actual - expected)
            self.ratings[match.home_team] = self._rating(match.home_team) + delta
            self.ratings[match.away_team] = self._rating(match.away_team) - delta
            self.snapshots.append(EloSnapshot(match.match_date, dict(self.ratings)))
        return self

    def probabilities(self, home_team: str, away_team: str) -> list[float]:
        p_home = self.expected(home_team, away_team)
        p_draw = 0.25 * (1 - abs(p_home - 0.5) * 2)
        return [p_home * (1 - p_draw), p_draw, (1 - p_home) * (1 - p_draw)]

