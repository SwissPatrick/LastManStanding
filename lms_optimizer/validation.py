"""Leakage-safe genuine-data validation reports."""
from dataclasses import dataclass
import numpy as np
from .backtest import score_metrics, outcome_index
from .elo import EloModel
from .ensemble import MarketEnsemble
from .models import HistoricalMatch
from .modeling import DixonColesModel
from .probability import power_method, proportional, shin

@dataclass
class ValidationReport:
    target_season: str
    rows: list[dict[str, object]]
    metrics: dict[str, dict[str, float]]
    bootstrap_differences: dict[str, dict[str, tuple[float, float]]]

def _market(match, method):
    odds = [match.closing_home_odds, match.closing_draw_odds, match.closing_away_odds]
    return {"proportional": proportional(odds), "power": power_method(odds), "shin": shin(odds)}[method]

def _safe_dc(model, match):
    try: return model.predict(match.home_team, match.away_team).outcome
    except ValueError: return np.ones(3) / 3

def validate_season(matches: list[HistoricalMatch], target_season: str, seed: int = 7, bootstrap_samples: int = 300) -> ValidationReport:
    ordered = sorted(matches, key=lambda m: m.match_date)
    test = [m for m in ordered if m.season == target_season]
    train = [m for m in ordered if m.season < target_season]
    if len(train) < 100 or not test: raise ValueError("insufficient chronological training or target data")
    dc = DixonColesModel().fit(train); elo = EloModel().fit(train)
    rows = []
    for match in test:
        components = [_market(match, "proportional"), _safe_dc(dc, match), np.asarray(elo.probabilities(match.home_team, match.away_team))]
        row = {"season": target_season, "date": match.match_date.isoformat(), "home_team": match.home_team, "away_team": match.away_team, "actual": outcome_index(match), "market": components[0].tolist(), "dixon_coles": components[1].tolist(), "elo": components[2].tolist(), "proportional": components[0].tolist(), "power": _market(match, "power").tolist(), "shin": _market(match, "shin").tolist(), "model_version": "dc-elo-market-v1", "training_cutoff": max(m.match_date for m in train).isoformat(), "data_source": match.data_source}
        rows.append(row)
    # Calibration is fit on a later slice inside training, never on target matches.
    split = max(50, int(len(train) * .8)); fit_part, calibration_part = train[:split], train[split:]
    if calibration_part:
        fit_dc, fit_elo = DixonColesModel().fit(fit_part), EloModel().fit(fit_part)
        features, labels = [], []
        for match in calibration_part:
            features.append(np.concatenate([_market(match, "proportional"), _safe_dc(fit_dc, match), fit_elo.probabilities(match.home_team, match.away_team)])); labels.append(outcome_index(match))
        ensemble = MarketEnsemble().fit(np.asarray(features), np.asarray(labels))
        for row, match in zip(rows, test):
            row["ensemble"] = ensemble.predict(np.asarray([np.concatenate([row["market"], row["dixon_coles"], row["elo"]])])).probabilities[0].tolist()
    else:
        for row in rows: row["ensemble"] = row["market"]
    metrics = {}
    for name in ("proportional", "power", "shin", "dixon_coles", "elo", "ensemble"):
        metrics[name] = score_metrics(np.asarray([row[name] for row in rows]), np.asarray([row["actual"] for row in rows]))
    rng = np.random.default_rng(seed); differences = {}
    actual = np.asarray([row["actual"] for row in rows])
    base = np.asarray([row["proportional"] for row in rows])
    for name in ("power", "shin", "dixon_coles", "elo", "ensemble"):
        candidate = np.asarray([row[name] for row in rows]); diffs = []
        for _ in range(bootstrap_samples):
            sample = rng.integers(0, len(rows), len(rows))
            diffs.append(score_metrics(candidate[sample], actual[sample])["log_loss"] - score_metrics(base[sample], actual[sample])["log_loss"])
        differences[name] = {"log_loss_difference_95ci": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))}
    return ValidationReport(target_season, rows, metrics, differences)

