"""Run final real-data holdouts and persist prediction-level provenance."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from lms_optimizer.data import import_football_data_csv
from lms_optimizer.validation import validate_season

raw = Path("data/raw/football-data")
matches = []
for path in sorted(raw.glob("*_E0.csv")):
    season = path.name.split("_")[0].replace("-", "/")
    matches.extend(import_football_data_csv(path, season, raw).matches)
results = {}
for season in ("2024/25", "2025/26"):
    report = validate_season(matches, season)
    results[season] = {"metrics": report.metrics, "bootstrap_differences": report.bootstrap_differences, "prediction_count": len(report.rows), "predictions": report.rows}
Path("data/real_model_validation.json").write_text(json.dumps(results, indent=2, default=str))
print(json.dumps({s: {"prediction_count": x["prediction_count"], "metrics": x["metrics"], "bootstrap_differences": x["bootstrap_differences"]} for s, x in results.items()}, indent=2))

