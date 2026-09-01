"""Formal discrete Rockafellar--Uryasev CVaR utilities."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class CVaRResult:
    alpha: float
    var_threshold: float
    cvar: float
    loss_definition: str
    expected_loss: float
    probability_total: float

    def as_dict(self) -> dict[str, object]:
        return {"alpha": self.alpha, "var_threshold": self.var_threshold, "cvar": self.cvar,
                "loss_definition": self.loss_definition, "expected_loss": self.expected_loss,
                "probability_total": self.probability_total}


def formal_cvar(losses, weights=None, alpha: float = .95,
                loss_definition: str = "eliminated entries") -> CVaRResult:
    """Return weighted discrete CVaR using the RU objective.

    Evaluating the convex RU objective at every distinct loss is exact for a
    finite distribution.  In particular, it retains probability mass at the
    VaR threshold rather than taking a conditional mean of only larger losses.
    """
    if not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or not 0 <= float(alpha) < 1:
        raise ValueError("alpha must be finite and in [0, 1)")
    values = np.asarray(losses, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("losses must be non-empty and finite")
    if weights is None:
        probabilities = np.full(values.size, 1.0 / values.size)
    else:
        probabilities = np.asarray(weights, dtype=float).reshape(-1)
        if probabilities.size != values.size or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("weights must be finite, non-negative and match losses")
        total = float(probabilities.sum())
        if total <= 0:
            raise ValueError("weights must have positive total")
        probabilities = probabilities / total
    unique_losses = np.unique(values)
    denominator = 1.0 - float(alpha)
    objectives = [float(eta + np.sum(probabilities * np.maximum(values - eta, 0.0)) / denominator) for eta in unique_losses]
    minimum = min(objectives)
    # The smallest minimising threshold is the conventional lower VaR threshold.
    eta = float(unique_losses[next(i for i, value in enumerate(objectives) if math.isclose(value, minimum, rel_tol=1e-12, abs_tol=1e-12))])
    cvar = float(eta + np.sum(probabilities * np.maximum(values - eta, 0.0)) / denominator)
    cumulative = 0.0
    for loss in unique_losses:
        cumulative += float(probabilities[values == loss].sum())
        if cumulative >= float(alpha) - 1e-12:
            var = float(loss)
            break
    return CVaRResult(float(alpha), var, cvar, str(loss_definition), float(np.sum(values * probabilities)), float(probabilities.sum()))

