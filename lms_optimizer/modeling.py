"""Time-decayed Dixon-Coles model and scoreline calculations."""
from dataclasses import dataclass
from math import exp, factorial, log
import numpy as np
from scipy.optimize import minimize
from .models import HistoricalMatch

@dataclass
class MatchPrediction:
    home_goals: float
    away_goals: float
    scoreline: np.ndarray
    outcome: np.ndarray
    uncertainty: np.ndarray | None = None

def poisson_matrix(home_lambda: float, away_lambda: float, max_goals: int = 10) -> np.ndarray:
    if not (np.isfinite(home_lambda) and np.isfinite(away_lambda) and home_lambda >= 0 and away_lambda >= 0):
        raise ValueError("goal rates must be finite and non-negative")
    h = np.array([exp(-home_lambda) * home_lambda**i / factorial(i) for i in range(max_goals + 1)])
    a = np.array([exp(-away_lambda) * away_lambda**i / factorial(i) for i in range(max_goals + 1)])
    result = np.outer(h, a)
    return result / result.sum()

def dixon_coles_tau(home_goals: int, away_goals: int, home_lambda: float, away_lambda: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0: return 1 - home_lambda * away_lambda * rho
    if home_goals == 0 and away_goals == 1: return 1 + home_lambda * rho
    if home_goals == 1 and away_goals == 0: return 1 + away_lambda * rho
    if home_goals == 1 and away_goals == 1: return 1 - rho
    return 1.0

def outcome_probabilities(scoreline: np.ndarray) -> np.ndarray:
    scoreline = np.asarray(scoreline, dtype=float)
    home = np.tril(scoreline, -1).sum()
    draw = np.trace(scoreline)
    away = np.triu(scoreline, 1).sum()
    result = np.array([home, draw, away])
    return result / result.sum()

class DixonColesModel:
    def __init__(self, decay_rate: float = 0.003, max_goals: int = 10) -> None:
        self.decay_rate, self.max_goals = decay_rate, max_goals
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.home_advantage = 0.0
        self.rho = 0.0
        self.fitted_at: object = None

    def fit(self, matches: list[HistoricalMatch], as_of=None) -> "DixonColesModel":
        ordered = sorted(matches, key=lambda m: m.match_date)
        if as_of is not None: ordered = [m for m in ordered if m.match_date < as_of]
        if not ordered: raise ValueError("at least one historical match is required")
        self.teams = sorted({t for m in ordered for t in (m.home_team, m.away_team)})
        n = len(self.teams); index = {t:i for i,t in enumerate(self.teams)}
        latest = ordered[-1].match_date
        weights = np.array([exp(-self.decay_rate * max(0, (latest - m.match_date).days)) for m in ordered])
        def unpack(x):
            attack = np.r_[x[:n-1], -np.sum(x[:n-1])]
            defence = x[n-1:2*n-1]
            return attack, defence, x[-2], x[-1]
        def objective(x):
            attack, defence, home, rho = unpack(x)
            value = 0.0
            for m, weight in zip(ordered, weights):
                lam, mu = exp(home + attack[index[m.home_team]] - defence[index[m.away_team]]), exp(attack[index[m.away_team]] - defence[index[m.home_team]])
                tau = dixon_coles_tau(m.full_time_home_goals, m.full_time_away_goals, lam, mu, rho)
                if tau <= 0: return 1e12
                value -= weight * (log(tau) - lam + m.full_time_home_goals*log(lam) - log(factorial(m.full_time_home_goals)) - mu + m.full_time_away_goals*log(mu) - log(factorial(m.full_time_away_goals)))
            return value
        x0 = np.zeros(2*n+1); x0[-2] = 0.2
        result = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-7})
        if not result.success and not np.isfinite(result.fun): raise ValueError("Dixon-Coles optimisation failed")
        attack, defence, self.home_advantage, self.rho = unpack(result.x)
        self.attack = dict(zip(self.teams, attack)); self.defence = dict(zip(self.teams, defence)); self.fitted_at = latest
        return self

    def predict(self, home_team: str, away_team: str) -> MatchPrediction:
        if home_team not in self.attack or away_team not in self.attack: raise ValueError("team was not seen in training data")
        home_lambda = exp(self.home_advantage + self.attack[home_team] - self.defence[away_team])
        away_lambda = exp(self.attack[away_team] - self.defence[home_team])
        matrix = poisson_matrix(home_lambda, away_lambda, self.max_goals)
        for h in range(2):
            for a in range(2):
                matrix[h,a] *= dixon_coles_tau(h, a, home_lambda, away_lambda, self.rho)
        matrix = np.maximum(matrix, 0)
        matrix /= matrix.sum()
        return MatchPrediction(home_lambda, away_lambda, matrix, outcome_probabilities(matrix))

    def bootstrap(self, matches: list[HistoricalMatch], home_team: str, away_team: str, samples: int = 30, seed: int = 7) -> np.ndarray:
        rng = np.random.default_rng(seed); predictions = []
        for sample in rng.integers(0, len(matches), size=samples):
            model = DixonColesModel(self.decay_rate, self.max_goals).fit([matches[i] for i in sample])
            predictions.append(model.predict(home_team, away_team).outcome)
        return np.percentile(np.asarray(predictions), [2.5, 97.5], axis=0)

