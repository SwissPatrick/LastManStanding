"""Refresh, import, audit, and validate the genuine Football-Data archive."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from lms_optimizer.data import audit_historical, import_football_data_csv

raw = Path("data/raw/football-data")
all_matches = []
reports = []
for path in sorted(raw.glob("*_E0.csv")):
    season = path.name.split("_")[0].replace("-", "/")
    report = import_football_data_csv(path, season, raw)
    reports.append(report)
    all_matches.extend(report.matches)
audit = audit_historical(all_matches)
audit["source_rows"] = {season: len(report.matches) + len(report.missing_rows) + len(report.duplicate_rows) + len(report.invalid_rows) for season, report in zip([p.name.split("_")[0].replace("-", "/") for p in sorted(raw.glob("*_E0.csv"))], reports)}
audit["missing_results"] = sum(1 for report in reports for row in report.missing_rows if row.get("field") == "result")
audit["missing_odds"] = sum(1 for report in reports for row in report.missing_rows if row.get("field") == "odds")
audit["import_reports"] = [{"season": path.name.split("_")[0].replace("-", "/"), "matches": len(report.matches), "missing": len(report.missing_rows), "duplicates": len(report.duplicate_rows), "invalid": len(report.invalid_rows)} for path, report in zip(sorted(raw.glob("*_E0.csv")), reports)]
Path("data/real_historical_audit.json").write_text(json.dumps(audit, indent=2, default=str))
print(json.dumps({"seasons": len(reports), "matches": len(all_matches), "audit": "data/real_historical_audit.json"}, indent=2))
