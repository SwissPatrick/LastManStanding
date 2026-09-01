"""Run the deterministic historical heterogeneous twenty-entry evaluation."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lms_optimizer.data import import_football_data_csv
from lms_optimizer.heterogeneous import evaluate_heterogeneous
from lms_optimizer.reproducibility import canonical_json_bytes


def load_matches(root: Path):
    matches = []
    for path in sorted((root / "data/raw/football-data").glob("*_E0.csv")):
        matches.extend(import_football_data_csv(path, path.name.split("_E0")[0].replace("-", "/"), root / "data/raw/football-data").matches)
    return matches


def main():
    root = Path(__file__).resolve().parents[1]
    report = evaluate_heterogeneous(load_matches(root), seed=7, bootstrap_repetitions=1000)
    output = root / "data/heterogeneous_cartel_evaluation"; output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    analytical = canonical_json_bytes(report); (output / "report.analytical.json").write_bytes(analytical)
    print(json.dumps({"seasons": len(report["seasons"]), "feasible_points": sum(x["feasible"] for x in report["cohort_construction"]), "infeasible_points": sum(not x["feasible"] for x in report["cohort_construction"]), "evaluations": len(report["evaluations"]), "decision_observations": len(report["decision_observations"]), "hash": hashlib.sha256(analytical).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
