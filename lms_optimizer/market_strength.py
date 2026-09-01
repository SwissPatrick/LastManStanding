"""Market-implied Bradley-Terry team strength model."""
from math import exp, log
import numpy as np
from scipy.optimize import minimize
from .models import HistoricalMatch
from .probability import proportional

class MarketStrengthModel:
    def __init__(self, decay_rate: float = .003, home_advantage: float = .2, season_regression: float = .75):
        self.decay_rate, self.home_advantage, self.season_regression = decay_rate, home_advantage, season_regression
        self.strength: dict[str, float] = {}; self.teams: list[str] = []

    def fit(self, matches: list[HistoricalMatch], as_of=None) -> "MarketStrengthModel":
        data = sorted([m for m in matches if as_of is None or m.match_date < as_of], key=lambda m: m.match_date)
        if not data: raise ValueError("historical market data is required")
        self.teams = sorted({t for m in data for t in (m.home_team, m.away_team)}); index = {t:i for i,t in enumerate(self.teams)}; latest = data[-1].match_date
        weights = np.array([exp(-self.decay_rate * max(0, (latest - m.match_date).days)) for m in data])
        def unpack(x):
            strengths = np.r_[x[:-1], -np.sum(x[:-1])]
            return strengths, x[-1]
        def objective(x):
            strengths, home = unpack(x); value = 0.
            for m, weight in zip(data, weights):
                observed = proportional([m.closing_home_odds, m.closing_draw_odds, m.closing_away_odds])
                home_p = 1 / (1 + exp(-(strengths[index[m.home_team]] - strengths[index[m.away_team]] + home)))
                target = observed[0] + .5 * observed[1]
                value -= weight * (target * log(max(home_p, 1e-12)) + (1-target) * log(max(1-home_p, 1e-12)))
            return value
        result = minimize(objective, np.zeros(len(self.teams)), method="BFGS")
        strengths, self.home_advantage = unpack(result.x)
        self.strength = dict(zip(self.teams, strengths))
        return self

    def probabilities(self, home_team: str, away_team: str) -> np.ndarray:
        if home_team not in self.strength or away_team not in self.strength: return np.ones(3) / 3
        home = 1 / (1 + exp(-(self.strength[home_team] - self.strength[away_team] + self.home_advantage)))
        draw = .25 * (1 - abs(home - .5) * 2)
        return np.array([home * (1-draw), draw, (1-home) * (1-draw)])

