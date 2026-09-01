"""Exact portfolio oracle and single-entry Bellman optimisation."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import numpy as np
from .simulation import formal_cvar

@dataclass(frozen=True)
class CandidateValue:
    team: str
    win_probability: float
    continuation_value: float
    opportunity_cost: float
    dynamic_value: float
    rank: int
    explanation: str

class DynamicProgram:
    def __init__(self, forecasts: dict[int, dict[str, float]], available_teams: dict[int, list[str]], used: frozenset[str] = frozenset(), horizon: int | None = None):
        self.forecasts, self.available, self.used, self.horizon = forecasts, available_teams, used, horizon or (max(forecasts) if forecasts else 0)
    def _value(self, round_number: int, used: frozenset[str]) -> float:
        if round_number > self.horizon: return 1.0
        candidates = [t for t in self.available.get(round_number, []) if t not in used]
        if not candidates: return 0.0
        return max(self.forecasts.get(round_number, {}).get(t, 0) * self._value(round_number + 1, used | {t}) for t in candidates)
    def solve(self) -> list[CandidateValue]:
        self._value = lru_cache(maxsize=None)(self._value)
        round_number = min(self.forecasts) if self.forecasts else 1
        candidates = [t for t in self.available.get(round_number, []) if t not in self.used]
        values = []
        for team in candidates:
            p = self.forecasts.get(round_number, {}).get(team, 0.)
            continuation = self._value(round_number + 1, self.used | {team})
            value = p * continuation
            alternatives = [self.forecasts.get(round_number, {}).get(other, 0.) * self._value(round_number + 1, self.used | {other}) for other in candidates if other != team]
            opportunity = max(alternatives, default=0.) - value
            values.append((team, p, continuation, opportunity, value))
        values.sort(key=lambda row: row[4], reverse=True)
        return [CandidateValue(row[0], row[1], row[2], row[3], row[4], i+1, "use now" if i == 0 else "preserve for future value") for i, row in enumerate(values)]

@dataclass
class PortfolioWeights:
    expected_survivors: float = 1.
    at_least_one: float = 1.
    wipeout: float = 1.
    future_value: float = 0.
    concentration: float = 0.
    cvar: float = 0.
    soft_cap: float = 0.
    cvar_alpha: float = .95

@dataclass
class ObjectiveComponents:
    expected_survivors: float
    probability_at_least_one: float
    wipeout_probability: float
    future_continuation_value: float
    cvar_eliminated: float
    soft_cap_excess: float
    squared_concentration: float
    objective: float

class PortfolioOptimizer:
    """Exact oracle. Scenarios are team->win indicators, never independent entries."""
    def __init__(self, entry_candidates: dict[str, list[str]], scenarios: list[dict[str, bool]], weights: PortfolioWeights | None = None, exposure_cap: int | dict[str, int] | None = None, soft_cap: int | dict[str, int] | None = None, soft_penalty: float | None = None, scenario_probabilities: list[float] | None = None, future_values: dict[str, dict[str, float]] | None = None, soft_exposure_cap: int | dict[str, int] | None = None):
        self.entry_candidates = entry_candidates
        self.scenarios = scenarios
        self.weights = weights or PortfolioWeights()
        self.hard_cap = {team: exposure_cap for team in set(t for values in entry_candidates.values() for t in values)} if isinstance(exposure_cap, int) else (exposure_cap or {})
        soft_cap = soft_exposure_cap if soft_exposure_cap is not None else soft_cap
        self.soft_cap = {team: soft_cap for team in set(t for values in entry_candidates.values() for t in values)} if isinstance(soft_cap, int) else (soft_cap or {})
        self.soft_penalty = self.weights.soft_cap if soft_penalty is None else soft_penalty
        raw = np.asarray(scenario_probabilities if scenario_probabilities is not None else np.ones(len(scenarios)), dtype=float)
        if len(raw) != len(scenarios) or np.any(raw < 0) or raw.sum() <= 0: raise ValueError("scenario probabilities must be non-negative and non-zero")
        self.scenario_probabilities = raw / raw.sum()
        self.future_values = future_values or {}

    def _allocations(self):
        entries = list(self.entry_candidates)
        for values in product(*(self.entry_candidates[e] for e in entries)):
            allocation = dict(zip(entries, values))
            exposure = {t: list(allocation.values()).count(t) for t in set(values)}
            if all(exposure.get(t, 0) <= cap for t, cap in self.hard_cap.items()):
                yield allocation

    def components(self, allocation: dict[str, str]) -> ObjectiveComponents:
        survivors = np.asarray([sum(bool(scenario.get(team, False)) for team in allocation.values()) for scenario in self.scenarios], dtype=float)
        probabilities = self.scenario_probabilities
        expected = float(np.dot(probabilities, survivors))
        any_survive = float(np.dot(probabilities, survivors > 0))
        wipeout = float(np.dot(probabilities, survivors == 0))
        future = float(sum(self.future_values.get(entry, {}).get(team, 0.) for entry, team in allocation.items()))
        soft = float(sum(max(0, sum(value == team for value in allocation.values()) - cap) for team, cap in self.soft_cap.items()))
        concentration = float(sum(sum(value == team for value in allocation.values()) ** 2 for team in set(allocation.values())))
        cvar = _weighted_cvar(len(allocation) - survivors, probabilities, self.weights.cvar_alpha)
        objective = (self.weights.expected_survivors * expected + self.weights.at_least_one * any_survive - self.weights.wipeout * wipeout + self.weights.future_value * future - self.weights.cvar * cvar - self.soft_penalty * soft - self.weights.concentration * concentration)
        return ObjectiveComponents(expected, any_survive, wipeout, future, cvar, soft, concentration, objective)

    def score(self, allocation: dict[str, str]) -> dict[str, float]:
        c = self.components(allocation)
        return c.__dict__.copy()

    def optimize(self) -> tuple[dict[str, str], dict[str, float]]:
        allocations = list(self._allocations())
        if not allocations: raise ValueError("no feasible portfolio allocation")
        scored = [(self.components(a), a) for a in allocations]
        component, allocation = max(scored, key=lambda row: row[0].objective)
        return allocation, component.__dict__.copy()

    def efficient_frontier(self):
        points = [(a, self.components(a)) for a in self._allocations()]
        return [(a, c.__dict__.copy()) for a, c in points if not any(o.expected_survivors >= c.expected_survivors and o.probability_at_least_one >= c.probability_at_least_one and o.wipeout_probability <= c.wipeout_probability and o.__dict__ != c.__dict__ for _, o in points)]

def _weighted_cvar(losses: np.ndarray, probabilities: np.ndarray, alpha: float) -> float:
    if not 0 <= alpha < 1: raise ValueError("alpha must be in [0, 1)")
    candidates = np.unique(losses)
    return float(min(eta + np.sum(probabilities * np.maximum(losses - eta, 0)) / (1 - alpha) for eta in candidates))
