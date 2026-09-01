"""Run the deterministic historical heterogeneous twenty-entry evaluation."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lms_optimizer.data import import_football_data_csv
from lms_optimizer.heterogeneous import evaluate_heterogeneous
from lms_optimizer.reproducibility import canonical_json_bytes, write_canonical_csv


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
    decision_columns = ["season", "starting_round", "constructed_round", "strategy", "cartel_size", "active_entries_before", "eliminated_entries", "eliminated_entry_fraction", "surviving_entries", "wipeout", "at_least_one_survives", "predicted_expected_loss", "predicted_cvar", "normalized_cvar", "realised_loss", "scenario_count", "scenario_probability_total", "information_cutoff", "milp_status", "milp_runtime_seconds"]
    write_canonical_csv(output / "decision_observations.csv", report["decision_observations"], decision_columns, ["season", "starting_round", "constructed_round", "strategy"])
    summary_columns = ["strategy", "observation_count", "mean_expected_survivors", "mean_realised_eliminated_entries", "mean_eliminated_entry_fraction", "area_under_survivor_curve", "predicted_decision_cvar", "predicted_normalized_cvar", "realised_aggregate_cvar", "realised_aggregate_normalized_cvar", "probability_at_least_one_survives", "wipeout_frequency", "maximum_loss"]
    write_canonical_csv(output / "strategy_summary.csv", [{"strategy": key, **value} for key, value in report["strategy_summary"].items()], summary_columns, ["strategy"])
    (output / "allocation_diagnostics.json").write_text(json.dumps(report["allocation_diagnostics"], indent=2, default=str), encoding="utf-8")
    print(json.dumps({"seasons": len(report["seasons"]), "feasible_points": sum(x["feasible"] for x in report["cohort_construction"]), "infeasible_points": sum(not x["feasible"] for x in report["cohort_construction"]), "evaluations": len(report["evaluations"]), "decision_observations": len(report["decision_observations"]), "hash": hashlib.sha256(analytical).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
