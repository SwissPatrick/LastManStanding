"""Generate compact real-data audit, holdout, and LMS policy reports."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from lms_optimizer.data import audit_historical, import_football_data_csv
from lms_optimizer.lms_eval import evaluate_lms_policies

raw = Path("data/raw/football-data")
matches = []
for path in sorted(raw.glob("*_E0.csv")):
    season = path.name.split("_")[0].replace("-", "/")
    matches.extend(import_football_data_csv(path, season, raw).matches)
report = json.loads(Path("data/real_model_validation.json").read_text())
report["lms_policies"] = {season: evaluate_lms_policies(matches, season) for season in ("2024/25", "2025/26")}
report["data_audit"] = audit_historical(matches)
Path("data/real_validation_report.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps({"seasons": len({m.season for m in matches}), "matches": len(matches), "holdouts": list(report["lms_policies"]), "report": "data/real_validation_report.json"}, indent=2))

