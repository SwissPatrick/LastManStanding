"""Correlated match-outcome Monte Carlo for LMS portfolios."""
from dataclasses import dataclass
from itertools import product
import numpy as np

def formal_cvar(losses: np.ndarray, alpha: float = .95) -> float:
    """Empirical Rockafellar-Uryasev CVaR: min_eta eta+E[(L-eta)+]/(1-alpha)."""
    if not 0 <= alpha < 1: raise ValueError("alpha must be in [0, 1)")
    values = np.asarray(losses, dtype=float)
    if values.size == 0: return 0.0
    return float(min(eta + np.maximum(values - eta, 0).mean() / (1-alpha) for eta in np.unique(values)))

def exact_current_round(allocation: dict[str, str], fixture_probabilities: dict[str, np.ndarray], fixture_teams: dict[str, tuple[str, str]], alpha: float = .95) -> dict[str, object]:
    """Enumerate mutually-exclusive H/D/A outcomes for a practical current round."""
    fixtures = list(fixture_probabilities)
    if len(fixtures) > 10: raise ValueError("exact enumeration is limited to ten fixtures")
    distribution = {}
    for outcomes in product(range(3), repeat=len(fixtures)):
        probability = 1.0
        winners = set()
        for fixture_id, outcome in zip(fixtures, outcomes):
            p = np.asarray(fixture_probabilities[fixture_id], dtype=float); p = p / p.sum(); probability *= p[outcome]
            home, away = fixture_teams[fixture_id]
            if outcome == 0: winners.add(home)
            elif outcome == 2: winners.add(away)
        survivors = sum(team in winners for team in allocation.values())
        distribution[survivors] = distribution.get(survivors, 0.0) + probability
    counts = np.array(sorted(distribution), dtype=int); probabilities = np.array([distribution[c] for c in counts])
    expected = float(np.sum(counts * probabilities)); any_survive = float(np.sum(probabilities[counts > 0])); wipeout = float(distribution.get(0, 0.0))
    losses = len(allocation) - counts
    return {"survivor_counts": counts, "probabilities": probabilities, "expected_survivors": expected, "probability_at_least_one": any_survive, "wipeout_probability": wipeout, "cvar_eliminated": formal_cvar(np.repeat(losses, np.maximum(1, np.rint(probabilities*100000).astype(int))), alpha)}

@dataclass
class AdaptiveSimulationSummary:
    survivor_counts: np.ndarray
    probability_at_least_one: float
    wipeout_probability: float
    expected_survivors: float
    cvar_eliminated: float
    standard_errors: dict[str, float]
    confidence_interval_widths: dict[str, float]
    simulations: int
    converged: bool
    stopping_reason: str
    diagnostics: list[dict[str, float]]

def adaptive_multi_round_simulation(allocation: dict[str, str], rounds: list[tuple[dict[str, np.ndarray], dict[str, tuple[str, str]]]], minimum_runs: int = 10000, maximum_runs: int = 100000, batch_size: int = 5000, target_standard_error: float = .005, target_ci_width: float = .02, seed: int = 7, alpha: float = .95) -> AdaptiveSimulationSummary:
    """Adaptive multi-round simulation; convergence requires every metric."""
    if minimum_runs < 1 or maximum_runs < minimum_runs or batch_size < 1: raise ValueError("invalid simulation limits")
    rng = np.random.default_rng(seed); all_counts: list[int] = []; diagnostics = []
    while len(all_counts) < maximum_runs:
        n = min(batch_size, maximum_runs - len(all_counts)); batch_counts = np.zeros(n, dtype=int)
        for probabilities, teams in rounds:
            for fixture_id, raw_p in probabilities.items():
                p = np.asarray(raw_p, dtype=float); p = p / p.sum(); outcomes = rng.choice(3, size=n, p=p); home, away = teams[fixture_id]
                for selected in allocation.values(): batch_counts += (((selected == home) & (outcomes == 0)) | ((selected == away) & (outcomes == 2))).astype(int)
        all_counts.extend(batch_counts.tolist())
        values = np.asarray(all_counts, dtype=float); any_values = (values > 0).astype(float); wipe_values = (values == 0).astype(float); losses = len(allocation) - values
        metrics = {"probability_at_least_one": any_values, "wipeout_probability": wipe_values, "expected_survivors": values, "cvar_eliminated": np.asarray([formal_cvar(losses[:i], alpha) for i in range(max(1, len(losses)-min(batch_size, len(losses))+1), len(losses)+1)])}
        standard_errors = {name: float(np.std(series, ddof=1) / np.sqrt(len(series))) for name, series in metrics.items()}
        widths = {name: 3.92 * standard_errors[name] for name in metrics}
        diagnostics.append({"simulations": float(len(values)), **{f"{name}_se": standard_errors[name] for name in standard_errors}, **{f"{name}_ci_width": widths[name] for name in widths}})
        if len(values) >= minimum_runs and all(standard_errors[name] <= target_standard_error and widths[name] <= target_ci_width for name in metrics):
            return AdaptiveSimulationSummary(values.astype(int), float(any_values.mean()), float(wipe_values.mean()), float(values.mean()), float(formal_cvar(losses, alpha)), standard_errors, widths, len(values), True, "all metrics reached both targets", diagnostics)
    values = np.asarray(all_counts, dtype=int); losses = len(allocation) - values; any_values = values > 0; wipe_values = values == 0
    series = {"probability_at_least_one": any_values.astype(float), "wipeout_probability": wipe_values.astype(float), "expected_survivors": values.astype(float), "cvar_eliminated": np.asarray([formal_cvar(losses, alpha)])}
    se = {name: float(np.std(value, ddof=1 if len(value) > 1 else 0) / np.sqrt(len(value))) for name, value in series.items()}; widths = {name: 3.92 * se[name] for name in se}
    return AdaptiveSimulationSummary(values, float(any_values.mean()), float(wipe_values.mean()), float(values.mean()), float(formal_cvar(losses, alpha)), se, widths, len(values), False, "maximum simulation count reached before all metrics converged", diagnostics)

@dataclass
class SimulationSummary:
    survivor_counts: np.ndarray
    probability_at_least_one: float
    wipeout_probability: float
    standard_error_at_least_one: float
    confidence_interval_at_least_one: tuple[float, float]
    simulations: int

def simulate_portfolio(allocation: dict[str, str], fixture_probabilities: dict[str, np.ndarray], fixture_teams: dict[str, tuple[str, str]], simulations: int = 10000, seed: int = 7) -> SimulationSummary:
    if simulations < 1: raise ValueError("simulations must be positive")
    rng = np.random.default_rng(seed); survivors = np.zeros(simulations, dtype=int)
    for fixture_id, probabilities in fixture_probabilities.items():
        p = np.asarray(probabilities, dtype=float); p = p / p.sum()
        outcomes = rng.choice(3, size=simulations, p=p)
        home, away = fixture_teams[fixture_id]
        for entry, team in allocation.items():
            wins = ((team == home) & (outcomes == 0)) | ((team == away) & (outcomes == 2))
            survivors += wins.astype(int)
    any_survive = survivors > 0
    probability = float(any_survive.mean()); se = float(np.sqrt(probability * (1-probability) / simulations))
    return SimulationSummary(survivors, probability, float(np.mean(survivors == 0)), se, (max(0., probability-1.96*se), min(1., probability+1.96*se)), simulations)
