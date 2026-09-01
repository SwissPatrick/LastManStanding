"""CPU execution controls for adaptive LMS simulation.

This module owns execution policy only. Probability generation and risk
definitions remain in :mod:`lms_optimizer.simulation`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import threading
import uuid


@dataclass(frozen=True)
class HardwareInfo:
    logical_cpus: int
    physical_cpus: int | None
    available_memory_bytes: int | None
    safe_logical_limit: int
    default_workers: int
    library_threads: int = 1


def detect_hardware() -> HardwareInfo:
    logical = max(1, os.cpu_count() or 1)
    physical = None
    memory = None
    try:
        import psutil  # optional local enhancement
        physical = psutil.cpu_count(logical=False)
        memory = int(psutil.virtual_memory().available)
    except (ImportError, AttributeError, OSError):
        if platform.system() == "Windows":
            try:
                command = "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"
                physical = int(subprocess.check_output(["powershell", "-NoProfile", "-Command", command], text=True, stderr=subprocess.DEVNULL, timeout=2).strip())
            except (OSError, ValueError, subprocess.SubprocessError):
                physical = None
            try:
                import ctypes
                class MemoryStatus(ctypes.Structure):
                    _fields_ = [("length", ctypes.c_uint32), ("memory_load", ctypes.c_uint32), ("total", ctypes.c_uint64), ("available", ctypes.c_uint64), ("page_total", ctypes.c_uint64), ("page_available", ctypes.c_uint64), ("virtual_total", ctypes.c_uint64), ("virtual_available", ctypes.c_uint64), ("extended", ctypes.c_uint64), ("virtual_extended", ctypes.c_uint64)]
                status = MemoryStatus(); status.length = ctypes.sizeof(MemoryStatus)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    memory = int(status.available)
            except (AttributeError, OSError):
                memory = None
    if physical is not None:
        physical = max(1, min(int(physical), logical))
    safe_limit = max(1, logical - 2) if logical > 2 else logical
    # Parallelism is an opt-in optimization.  The verified local benchmark is
    # authoritative for automatic selection; safe hardware limits remain
    # available for explicit advanced use.
    default = 1
    return HardwareInfo(logical, physical, memory, safe_limit, max(1, default))


@dataclass(frozen=True)
class PerformanceConfig:
    name: str
    minimum_runs: int
    maximum_runs: int
    batch_size: int
    target_standard_error: float
    target_ci_width: float
    workers: int
    seed: int

    def validate(self, hardware: HardwareInfo | None = None) -> "PerformanceConfig":
        limit = (hardware or detect_hardware()).safe_logical_limit
        if not 1 <= self.workers <= limit:
            raise ValueError(f"worker count must be between 1 and {limit}")
        if self.minimum_runs < 1 or self.maximum_runs < self.minimum_runs or self.batch_size < 1:
            raise ValueError("invalid simulation limits")
        if self.target_standard_error <= 0 or self.target_ci_width <= 0:
            raise ValueError("convergence targets must be positive")
        return self

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def performance_profiles(hardware: HardwareInfo | None = None) -> dict[str, PerformanceConfig]:
    workers = (hardware or detect_hardware()).default_workers
    return {
        "Quick": PerformanceConfig("Quick", 5_000, 50_000, 5_000, .02, .08, workers, 7),
        "Standard": PerformanceConfig("Standard", 10_000, 500_000, 10_000, .005, .02, workers, 7),
        "Deep": PerformanceConfig("Deep", 25_000, 2_000_000, 25_000, .002, .01, workers, 7),
        "Maximum": PerformanceConfig("Maximum", 50_000, 10_000_000, 50_000, .001, .005, workers, 7),
    }


def benchmark_fingerprint(hardware: HardwareInfo) -> str:
    payload = {
        "logical": hardware.logical_cpus,
        "physical": hardware.physical_cpus,
        "safe_limit": hardware.safe_logical_limit,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": __import__("numpy").__version__,
        "workload": "lms-adaptive-worker-benchmark-v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def benchmark_worker_counts(hardware: HardwareInfo | None = None, benchmark_runner=None) -> dict[str, object]:
    """Measure a representative workload at stable hardware-derived counts."""
    hardware = hardware or detect_hardware()
    candidates = sorted({max(1, min(value, hardware.safe_logical_limit)) for value in (1, 4, 8, hardware.safe_logical_limit)})
    if benchmark_runner is None:
        from .simulation import adaptive_multi_round_simulation_parallel
        import numpy as np
        probabilities = {f"fixture-{i}": np.array([.55, .20, .25]) for i in range(6)}
        teams = {f"fixture-{i}": (f"Home-{i}", f"Away-{i}") for i in range(6)}
        allocation = {"e1": "Home-0", "e2": "Home-1", "e3": "Home-2", "e4": "Home-3"}
        rounds = [(probabilities, teams)]
        def benchmark_runner(worker_count):
            started = time.perf_counter()
            summary = adaptive_multi_round_simulation_parallel(allocation, rounds, PerformanceConfig("benchmark", 4_000, 4_000, 500, .0001, .0004, worker_count, 2718))
            elapsed = time.perf_counter() - started
            return {"elapsed_seconds": elapsed, "simulations_per_second": summary.simulations / elapsed, "analytical_hash": hashlib.sha256(summary.survivor_count_histogram.tobytes()).hexdigest(), "converged": summary.converged}
    results = []
    for worker_count in candidates:
        measured = dict(benchmark_runner(worker_count))
        measured["workers"] = worker_count
        results.append(measured)
    return {"fingerprint": benchmark_fingerprint(hardware), "hardware": hardware.__dict__, "results": results}


def select_worker_configuration(hardware: HardwareInfo | None = None, cache_dir: str | os.PathLike = "data/benchmarks", benchmark_runner=None) -> dict[str, object]:
    """Select automatic workers only when parallelism beats serial by 10%."""
    hardware = hardware or detect_hardware()
    fingerprint = benchmark_fingerprint(hardware)
    path = __import__("pathlib").Path(cache_dir) / f"worker-selection-{fingerprint}.json"
    try:
        if path.exists():
            benchmark = json.loads(path.read_text(encoding="utf-8"))
            if benchmark.get("fingerprint") != fingerprint:
                raise ValueError("benchmark fingerprint mismatch")
        else:
            benchmark = benchmark_worker_counts(hardware, benchmark_runner)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(benchmark, sort_keys=True, indent=2), encoding="utf-8")
        results = benchmark["results"]
        serial = next(row for row in results if int(row["workers"]) == 1)
        fastest = max(results, key=lambda row: float(row["simulations_per_second"]))
        if int(fastest["workers"]) != 1 and float(fastest["simulations_per_second"]) >= float(serial["simulations_per_second"]) * 1.10:
            selected = int(fastest["workers"]); reason = "benchmark selected parallel configuration with at least 10% speedup"
        else:
            selected = 1; reason = "one worker selected because no parallel configuration beat it by 10%"
        return {"workers": selected, "reason": reason, "measured_simulations_per_second": float(next(row for row in results if int(row["workers"]) == selected)["simulations_per_second"]), "benchmark": benchmark, "benchmark_path": str(path), "fallback": False}
    except Exception as exc:
        return {"workers": 1, "reason": f"one worker fallback because benchmark failed: {exc}", "measured_simulations_per_second": None, "benchmark": None, "benchmark_path": str(path), "fallback": True}


def effective_thread_configuration() -> dict[str, int | str]:
    try:
        from threadpoolctl import threadpool_info
        libraries = threadpool_info()
        return {"process_workers": 1, "math_library_threads": 1, "libraries": ",".join(item.get("internal_api", "unknown") for item in libraries) or "none"}
    except ImportError:
        return {"process_workers": 1, "math_library_threads": 1, "libraries": "threadpoolctl unavailable"}


def resource_snapshot() -> dict[str, float | int | None]:
    """Best-effort local resource metrics; unavailable values are explicit."""
    try:
        import psutil
        return {"cpu_percent": float(psutil.cpu_percent(interval=None)), "memory_percent": float(psutil.virtual_memory().percent)}
    except (ImportError, AttributeError, OSError):
        return {"cpu_percent": None, "memory_percent": None}


class SimulationJobService:
    """Non-blocking job coordinator for Streamlit and other local clients."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lms-simulation")
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, object]] = {}

    def start(self, allocation, rounds, config: PerformanceConfig) -> str:
        config.validate()
        with self._lock:
            if any(job["state"] == "running" for job in self._jobs.values()):
                raise ValueError("a simulation job is already running")
            job_id = uuid.uuid4().hex
            job = {"job_id": job_id, "state": "running", "started_at": datetime.now(timezone.utc), "cancel": threading.Event(), "summary": None, "error": None, "config": config, "progress": {"simulations": 0, "maximum_runs": config.maximum_runs}}
            self._jobs[job_id] = job
            job["future"] = self._executor.submit(self._run, job, allocation, rounds, config)
            return job_id

    def _run(self, job, allocation, rounds, config) -> None:
        try:
            from .simulation import adaptive_multi_round_simulation_parallel
            def update(progress):
                with self._lock:
                    job["progress"] = progress
            summary = adaptive_multi_round_simulation_parallel(allocation, rounds, config=config, cancel_event=job["cancel"], progress_callback=update)
            with self._lock:
                job["summary"] = summary
                job["state"] = "cancelled" if summary.stopping_reason == "cancelled" else "completed"
        except Exception as exc:  # safe message is exposed by status()
            with self._lock:
                job["state"] = "cancelled" if job["cancel"].is_set() else "failed"
                job["error"] = None if job["cancel"].is_set() else str(exc)

    def cancel(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job["state"] != "running":
                raise ValueError("job is not running")
            job["cancel"].set()

    def status(self, job_id: str) -> dict[str, object]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise ValueError("unknown simulation job")
            summary = job["summary"]
            elapsed = (datetime.now(timezone.utc) - job["started_at"]).total_seconds()
            progress = dict(job["progress"])
            result = {"job_id": job_id, "state": job["state"], "elapsed_seconds": max(0.0, elapsed), "error": job["error"], "progress": progress, "simulations": progress.get("simulations", 0), "maximum_runs": progress.get("maximum_runs"), "simulations_per_second": float(progress.get("simulations", 0)) / max(elapsed, .001), "active_workers": job["config"].workers if job["state"] == "running" else 0, "resources": resource_snapshot()}
            if summary is not None:
                result.update({"simulations": summary.simulations, "converged": summary.converged, "stopping_reason": summary.stopping_reason, "diagnostics": summary.diagnostics, "summary": summary})
            return result

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
