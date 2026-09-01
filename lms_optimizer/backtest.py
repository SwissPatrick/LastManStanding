"""Chronological expanding-window backtesting and leakage-safe metrics."""
from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
from .models import HistoricalMatch
from .probability import proportional

@dataclass
class BacktestResult:
    predictions: np.ndarray
    outcomes: np.ndarray
    dates: list[object]
    metrics: dict[str, float]

def outcome_index(match: HistoricalMatch) -> int:
    return 0 if match.full_time_home_goals > match.full_time_away_goals else 1 if match.full_time_home_goals == match.full_time_away_goals else 2

def score_metrics(predictions: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(predictions, dtype=float), 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    y = np.asarray(outcomes, dtype=int)
    one_hot = np.eye(3)[y]
    log_loss = float(-np.mean(np.log(p[np.arange(len(y)), y])))
    brier = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
    accuracy = float(np.mean(np.argmax(p, axis=1) == y))
    rps = float(np.mean((np.cumsum(p, axis=1)[:, :2] - np.cumsum(one_hot, axis=1)[:, :2]) ** 2))
    confidence = p.max(axis=1)
    calibration_errors = []
    for threshold in np.arange(0.5, 1.0, 0.1):
        mask = confidence >= threshold
        if np.any(mask):
            calibration_errors.append(abs(float(np.mean((np.argmax(p, axis=1) == y)[mask])) - float(np.mean(confidence[mask]))))
    calibration = float(np.mean(calibration_errors)) if calibration_errors else 0.0
    return {"log_loss": log_loss, "brier_score": brier, "accuracy": accuracy, "ranked_probability_score": rps, "calibration_error": calibration}

def expanding_backtest(matches: list[HistoricalMatch], predictor: Callable[[list[HistoricalMatch], HistoricalMatch], np.ndarray], min_train: int = 20) -> BacktestResult:
    ordered = sorted(matches, key=lambda m: m.match_date)
    predictions, outcomes, dates = [], [], []
    for index in range(min_train, len(ordered)):
        train, test = ordered[:index], ordered[index]
        prediction = np.asarray(predictor(train, test), dtype=float)
        if prediction.shape != (3,): raise ValueError("predictor must return three probabilities")
        predictions.append(prediction); outcomes.append(outcome_index(test)); dates.append(test.match_date)
    if not predictions: raise ValueError("not enough matches for the requested training window")
    return BacktestResult(np.asarray(predictions), np.asarray(outcomes), dates, score_metrics(np.asarray(predictions), np.asarray(outcomes)))

def market_predictor(_: list[HistoricalMatch], match: HistoricalMatch) -> np.ndarray:
    return proportional([match.closing_home_odds, match.closing_draw_odds, match.closing_away_odds])
