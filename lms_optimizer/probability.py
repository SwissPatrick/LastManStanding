"""Market probability methods; odds are never treated as fair directly."""
import numpy as np

def _validate(odds: list[float]) -> np.ndarray:
    values = np.asarray(odds, dtype=float)
    if values.size < 2 or np.any(~np.isfinite(values)) or np.any(values <= 1):
        raise ValueError("decimal odds must all be greater than 1")
    return values

def implied_probabilities(odds: list[float]) -> np.ndarray:
    return 1.0 / _validate(odds)

def proportional(odds: list[float]) -> np.ndarray:
    raw = implied_probabilities(odds)
    return raw / raw.sum()

def additive(odds: list[float]) -> np.ndarray:
    raw = implied_probabilities(odds)
    return raw - (raw.sum() - 1.0) / len(raw)

def power_method(odds: list[float], tolerance: float = 1e-10) -> np.ndarray:
    raw = implied_probabilities(odds)
    lo, hi = 0.01, 10.0
    for _ in range(100):
        exponent = (lo + hi) / 2
        total = np.power(raw, exponent).sum()
        if total > 1: lo = exponent
        else: hi = exponent
        if abs(total - 1) < tolerance: break
    return np.power(raw, (lo + hi) / 2)

def shin(odds: list[float]) -> np.ndarray:
    """Remove overround with Shin's insider-share model.

    For bookmaker-implied values ``q_i = 1 / odds_i`` and ``Q = sum(q_i)``,
    Shin's equation is

    ``p_i(z) = (sqrt(z² + 4(1-z) q_i² / Q) - z) / (2(1-z))``

    with ``z`` chosen so that ``sum(p_i(z)) = 1``.  The square on ``q_i``
    and the division by ``Q`` are essential parts of the equation.
    """
    raw = implied_probabilities(odds)
    overround = raw.sum()
    if overround <= 1:
        return raw / overround

    def probabilities(insider_share: float) -> np.ndarray:
        numerator = np.sqrt(insider_share**2 + 4 * (1 - insider_share) * raw**2 / overround) - insider_share
        denominator = 2 * (1 - insider_share)
        return numerator / denominator

    # The sum is >= 1 at z=0 and falls below 1 near z=1 for an overround
    # market, so bisection gives the unique root in this interval.
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(100):
        midpoint = (lo + hi) / 2
        if probabilities(midpoint).sum() > 1:
            lo = midpoint
        else:
            hi = midpoint
    z = (lo + hi) / 2
    result = probabilities(z)
    # Only round-off correction: the returned vector already solves Shin's
    # equation; this is not a substitute for solving it.
    return result / result.sum()

def market_disagreement(probability_sets: list[list[float]]) -> float:
    values = np.asarray(probability_sets, dtype=float)
    if values.ndim != 2 or len(values) < 2: return 0.0
    normalized = values / values.sum(axis=1, keepdims=True)
    return float(np.mean(np.std(normalized, axis=0)))
