"""Correlated match-outcome Monte Carlo for LMS portfolios."""
from dataclasses import dataclass
from itertools import product
import numpy as np
from .cvar import formal_cvar as _formal_cvar
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import threading

def formal_cvar(losses: np.ndarray, alpha: float = .95) -> float:
    """Deprecated scalar compatibility wrapper; use ``cvar.formal_cvar`` for details."""
    return _formal_cvar(losses, alpha=alpha).cvar

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
    cvar = _formal_cvar(losses, probabilities, alpha, "eliminated entries")
    return {"survivor_counts": counts, "probabilities": probabilities, "expected_survivors": expected, "probability_at_least_one": any_survive, "wipeout_probability": wipeout, "cvar_eliminated": cvar.cvar, "cvar": cvar.as_dict()}

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
    survivor_count_histogram: np.ndarray | None = None
    worker_count: int = 1

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


def deterministic_child_seed(master_seed: int, batch_index: int) -> int:
    """Stable seed derivation independent of worker allocation or completion."""
    return int(np.random.SeedSequence([int(master_seed), int(batch_index)]).generate_state(1, dtype=np.uint64)[0])


def _simulate_batch_worker(payload: tuple[dict[str, str], list[tuple[dict[str, np.ndarray], dict[str, tuple[str, str]]]], int, int, int]) -> tuple[int, np.ndarray]:
    allocation, rounds, batch_index, batch_size, seed = payload
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=1)
    except ImportError:
        limiter = None
    if limiter is not None:
        limiter.__enter__()
    try:
        rng = np.random.default_rng(seed)
        survivors = np.zeros(batch_size, dtype=np.int16 if len(allocation) < 32767 else np.int32)
        for probabilities, teams in rounds:
            for fixture_id in sorted(probabilities):
                p = np.asarray(probabilities[fixture_id], dtype=float)
                p = p / p.sum()
                outcomes = rng.choice(3, size=batch_size, p=p)
                home, away = teams[fixture_id]
                for selected in allocation.values():
                    survivors += (((selected == home) & (outcomes == 0)) | ((selected == away) & (outcomes == 2))).astype(survivors.dtype)
        return batch_index, np.bincount(survivors, minlength=len(allocation) + 1)
    finally:
        if limiter is not None:
            limiter.__exit__(None, None, None)


def _summary_from_histogram(histogram: np.ndarray, allocation_size: int, batch_metrics: list[dict[str, float]], diagnostics: list[dict[str, float]], simulations: int, converged: bool, stopping_reason: str, worker_count: int, retain_samples: bool = True) -> AdaptiveSimulationSummary:
    if simulations < 1:
        raise ValueError("at least one simulation batch is required")
    survivor_values = np.repeat(np.arange(len(histogram), dtype=int), histogram.astype(int)) if retain_samples else np.empty(0, dtype=int)
    probabilities = histogram.astype(float) / simulations
    survivors = np.arange(len(histogram), dtype=float)
    any_probability = float(probabilities[1:].sum())
    wipeout = float(probabilities[0])
    expected = float(np.dot(survivors, probabilities))
    losses = allocation_size - survivors
    cvar_result = _formal_cvar(losses, probabilities, .95, "eliminated entries")
    mean_se = float(np.sqrt(max(0.0, expected * 0 + np.dot((survivors - expected) ** 2, probabilities)) / simulations))
    any_se = float(np.sqrt(max(0.0, any_probability * (1 - any_probability)) / simulations))
    wipe_se = float(np.sqrt(max(0.0, wipeout * (1 - wipeout)) / simulations))
    cvar_values = np.asarray([item["cvar_eliminated"] for item in batch_metrics], dtype=float)
    cvar_se = float(np.std(cvar_values, ddof=1) / np.sqrt(len(cvar_values))) if len(cvar_values) > 1 else 0.0
    errors = {"probability_at_least_one": any_se, "wipeout_probability": wipe_se, "expected_survivors": mean_se, "cvar_eliminated": cvar_se}
    widths = {name: 3.92 * value for name, value in errors.items()}
    return AdaptiveSimulationSummary(survivor_values, any_probability, wipeout, expected, cvar_result.cvar, errors, widths, simulations, converged, stopping_reason, diagnostics, histogram.copy(), worker_count)


def adaptive_multi_round_simulation_parallel(allocation: dict[str, str], rounds: list[tuple[dict[str, np.ndarray], dict[str, tuple[str, str]]]], config=None, cancel_event: threading.Event | None = None, progress_callback=None) -> AdaptiveSimulationSummary:
    """Run deterministic bounded batches, optionally using Windows processes.

    Each batch has a seed derived solely from ``(master seed, batch index)``;
    aggregation is by batch index, so worker count cannot change results.
    """
    if config is None:
        from .performance import performance_profiles
        config = performance_profiles()["Standard"]
    config.validate()
    histogram = np.zeros(len(allocation) + 1, dtype=np.int64)
    batch_metrics: list[dict[str, float]] = []
    diagnostics: list[dict[str, float]] = []
    submitted = 0
    executor = None
    if config.workers > 1:
        executor = ProcessPoolExecutor(max_workers=config.workers, mp_context=mp.get_context("spawn"))
    try:
        batch_index = 0
        while submitted < config.maximum_runs:
            if cancel_event is not None and cancel_event.is_set():
                if submitted:
                    return _summary_from_histogram(histogram, len(allocation), batch_metrics, diagnostics, submitted, False, "cancelled", config.workers)
                raise ValueError("simulation cancelled before first complete batch")
            size = min(config.batch_size, config.maximum_runs - submitted)
            payload = (allocation, rounds, batch_index, size, deterministic_child_seed(config.seed, batch_index))
            if executor is None:
                index, batch_hist = _simulate_batch_worker(payload)
            else:
                future = executor.submit(_simulate_batch_worker, payload)
                index, batch_hist = future.result()
            if index != batch_index:
                raise RuntimeError("simulation batch ordering invariant failed")
            histogram += batch_hist
            submitted += size
            batch_prob = float(batch_hist[1:].sum() / size)
            batch_wipe = float(batch_hist[0] / size)
            batch_survivors = np.arange(len(batch_hist), dtype=float)
            batch_losses = len(allocation) - batch_survivors
            batch_cvar = _formal_cvar(batch_losses, batch_hist / size, .95, "eliminated entries").cvar
            batch_expected = float(np.dot(batch_survivors, batch_hist / size))
            batch_metrics.append({"probability_at_least_one": batch_prob, "wipeout_probability": batch_wipe, "expected_survivors": batch_expected, "cvar_eliminated": batch_cvar})
            provisional = _summary_from_histogram(histogram, len(allocation), batch_metrics, diagnostics, submitted, False, "running", config.workers, retain_samples=False)
            diagnostics.append({"simulations": float(submitted), **{f"{name}_se": value for name, value in provisional.standard_errors.items()}, **{f"{name}_ci_width": value for name, value in provisional.confidence_interval_widths.items()}})
            if progress_callback is not None:
                progress_callback({"simulations": submitted, "maximum_runs": config.maximum_runs, "converged": False, "standard_errors": dict(provisional.standard_errors), "confidence_interval_widths": dict(provisional.confidence_interval_widths)})
            if submitted >= config.minimum_runs and all(provisional.standard_errors[name] <= config.target_standard_error and provisional.confidence_interval_widths[name] <= config.target_ci_width for name in provisional.standard_errors):
                return _summary_from_histogram(histogram, len(allocation), batch_metrics, diagnostics, submitted, True, "all metrics reached both targets", config.workers)
            batch_index += 1
        return _summary_from_histogram(histogram, len(allocation), batch_metrics, diagnostics, submitted, False, "maximum simulation count reached before all metrics converged", config.workers)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
