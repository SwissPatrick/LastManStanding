from datetime import datetime, timezone
import hashlib
import time
import threading

import numpy as np
import pytest

from lms_optimizer.performance import PerformanceConfig, SimulationJobService, detect_hardware, performance_profiles
from lms_optimizer.simulation import adaptive_multi_round_simulation_parallel, deterministic_child_seed


def workload():
    probabilities = {"f0": np.array([.6, .2, .2]), "f1": np.array([.5, .2, .3])}
    teams = {"f0": ("A", "B"), "f1": ("C", "D")}
    return {"e1": "A", "e2": "C"}, [(probabilities, teams)]


def config(workers):
    return PerformanceConfig("test", 40, 80, 20, .000001, .000001, workers, 123)


def test_hardware_fallback_and_profiles(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    hardware = detect_hardware()
    assert hardware.logical_cpus == 8 and hardware.safe_logical_limit == 6
    profiles = performance_profiles(hardware)
    assert profiles["Quick"].maximum_runs == 50_000 and profiles["Maximum"].maximum_runs == 10_000_000
    assert profiles["Standard"].workers == hardware.default_workers


def test_worker_limit_validation():
    with pytest.raises(ValueError): PerformanceConfig("bad", 1, 1, 1, .1, .1, 10_000, 1).validate()


def test_child_seeds_are_stable_and_batch_specific():
    assert deterministic_child_seed(7, 4) == deterministic_child_seed(7, 4)
    assert deterministic_child_seed(7, 4) != deterministic_child_seed(7, 5)


def test_serial_and_process_results_are_identical():
    allocation, rounds = workload()
    serial = adaptive_multi_round_simulation_parallel(allocation, rounds, config=config(1))
    parallel = adaptive_multi_round_simulation_parallel(allocation, rounds, config=config(2))
    assert np.array_equal(serial.survivor_count_histogram, parallel.survivor_count_histogram)
    assert serial.expected_survivors == parallel.expected_survivors
    assert serial.cvar_eliminated == parallel.cvar_eliminated
    assert serial.diagnostics == parallel.diagnostics


def test_batch_aggregation_is_worker_count_invariant():
    allocation, rounds = workload()
    hashes = []
    for workers in (1, 2):
        result = adaptive_multi_round_simulation_parallel(allocation, rounds, config=config(workers))
        hashes.append(hashlib.sha256(result.survivor_count_histogram.tobytes()).hexdigest())
    assert len(set(hashes)) == 1


def test_job_service_duplicate_prevention_progress_and_completion():
    allocation, rounds = workload(); service = SimulationJobService()
    job_id = service.start(allocation, rounds, config(1))
    with pytest.raises(ValueError): service.start(allocation, rounds, config(1))
    for _ in range(100):
        status = service.status(job_id)
        if status["state"] != "running": break
        time.sleep(.01)
    assert status["state"] == "completed" and status["simulations"] == 80
    assert status["diagnostics"]
    service.close()


def test_job_service_worker_failure_is_safe():
    allocation = {"e1": "A"}; rounds = [({"f": np.array([0., 0., 0.])}, {"f": ("A", "B")})]
    service = SimulationJobService(); job_id = service.start(allocation, rounds, config(1))
    for _ in range(100):
        status = service.status(job_id)
        if status["state"] != "running": break
        time.sleep(.01)
    assert status["state"] == "failed" and "zero-size" not in str(status["error"]).lower()
    service.close()


def test_cancellation_stops_before_work_and_reports_cancelled():
    allocation, rounds = workload(); service = SimulationJobService()
    job_id = service.start(allocation, rounds, PerformanceConfig("cancel", 1, 100, 20, .000001, .000001, 1, 9))
    service.cancel(job_id)
    for _ in range(100):
        status = service.status(job_id)
        if status["state"] != "running": break
        time.sleep(.01)
    assert status["state"] == "cancelled"
    service.close()
