"""Rebuild historical evaluation and canonical analytical artefacts."""
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from lms_optimizer.data import import_football_data_csv
from lms_optimizer.historical_evaluator import evaluate_all_seasons
from lms_optimizer.reproducibility import canonical_json_bytes, write_canonical_csv, first_difference

def load_matches():
    result = []
    raw = Path("data/raw/football-data")
    for path in sorted(raw.glob("*_E0.csv")):
        result.extend(import_football_data_csv(path, path.name.split("_")[0].replace("-", "/"), raw).matches)
    return result

def build_report(matches, seed):
    started = time.perf_counter()
    report = evaluate_all_seasons(matches, seed=seed, bootstrap_repetitions=1000)
    report["configuration"] = {"minimum_eligible_matches": 6, "maximum_fixtures_per_constructed_round": 10, "round_grouping": "chronological greedy grouping within three calendar days, no team collision, ten-fixture cap", "current_probability": "proportional de-vigged market consensus; timestamp-unknown prices are decision-time proxies", "future_probability": "chronological Elo, fitted only before each decision cutoff", "bellman_horizon_rounds": 1, "bootstrap": {"method": "whole-season clustered percentile", "repetitions": 1000, "seed": seed}, "strategies": ["concentrated_favourite", "equal_diversification", "independent_greedy", "bellman", "max_expected_survivors", "protect_one", "balanced"], "cartel_sizes": [1, 3, 5, 10]}
    report["seed"] = seed; report["runtime_seconds"] = time.perf_counter() - started
    report["data_manifest_checksum"] = hashlib.sha256(Path("data/football_data_manifest.json").read_bytes()).hexdigest()
    return report

def write_report(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (directory / "report.analytical.json").write_bytes(canonical_json_bytes(report))
    hashes = {"report": hashlib.sha256((directory / "report.analytical.json").read_bytes()).hexdigest()}
    for name, rows in (("cohorts.csv", report["cohort_metrics"]), ("decisions.csv", report["decisions"]), ("round_audit.csv", report["round_construction_audit"])):
        if rows:
            columns = sorted(rows[0]); sort_columns = [c for c in ("season", "start_round", "constructed_round", "strategy", "cartel_size", "entry", "team", "round_number") if c in columns]
            hashes[name] = write_canonical_csv(directory / name, rows, columns, sort_columns)
    (directory / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    return hashes

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=Path("data/historical_cartel_evaluation")); parser.add_argument("--seed", type=int, default=7); parser.add_argument("--compare", nargs=2, type=Path); args = parser.parse_args()
    if args.compare:
        one = json.loads((args.compare[0] / "report.json").read_text(encoding="utf-8")); two = json.loads((args.compare[1] / "report.json").read_text(encoding="utf-8")); one.pop("runtime_seconds", None); two.pop("runtime_seconds", None); print(json.dumps(first_difference(one, two), indent=2, default=str)); return 0
    report = build_report(load_matches(), args.seed); hashes = write_report(report, args.output_dir); print(json.dumps({"seasons": len(report["seasons"]), "cohort_rows": len(report["cohort_metrics"]), "decisions": len(report["decisions"]), "bootstrap_repetitions": report["bootstrap_repetitions"], "runtime_seconds": report["runtime_seconds"], "hashes": hashes}, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
