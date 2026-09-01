"""Benchmark deterministic CPU simulation execution locally."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lms_optimizer.performance import detect_hardware, PerformanceConfig
from lms_optimizer.simulation import adaptive_multi_round_simulation_parallel


def workload():
    probabilities = {f"fixture-{i}": np.array([.55, .20, .25]) for i in range(6)}
    teams = {f"fixture-{i}": (f"Home-{i}", f"Away-{i}") for i in range(6)}
    return {"e1": "Home-0", "e2": "Home-1", "e3": "Home-2", "e4": "Home-3"}, [(probabilities, teams)]


def main() -> None:
    hardware = detect_hardware()
    candidates = [1, 4, 8, hardware.physical_cpus or hardware.default_workers]
    if hardware.default_workers + 2 <= hardware.safe_logical_limit:
        candidates.append(hardware.default_workers + 2)
    workers = sorted({max(1, min(value, hardware.safe_logical_limit)) for value in candidates})
    allocation, rounds = workload(); rows = []
    for worker_count in workers:
        config = PerformanceConfig("benchmark", 4_000, 4_000, 500, .0001, .0004, worker_count, 2718)
        started = time.perf_counter(); summary = adaptive_multi_round_simulation_parallel(allocation, rounds, config); elapsed = time.perf_counter() - started
        canonical = json.dumps({"histogram": summary.survivor_count_histogram.tolist(), "simulations": summary.simulations, "converged": summary.converged}, sort_keys=True).encode()
        rows.append({"workers": worker_count, "elapsed_seconds": elapsed, "simulations_per_second": summary.simulations / elapsed, "speedup_vs_one": None, "efficiency": None, "analytical_hash": hashlib.sha256(canonical).hexdigest(), "converged": summary.converged, "stopping_reason": summary.stopping_reason})
    baseline = rows[0]["elapsed_seconds"]
    for row in rows:
        row["speedup_vs_one"] = baseline / row["elapsed_seconds"]
        row["efficiency"] = row["speedup_vs_one"] / row["workers"]
    output = {"hardware": hardware.__dict__, "workload": {"simulations": 4000, "seed": 2718}, "results": rows, "recommended_workers": min(rows, key=lambda row: row["elapsed_seconds"])["workers"]}
    path = Path("data/benchmarks"); path.mkdir(parents=True, exist_ok=True)
    target = path / "simulation_benchmark.json"; target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
