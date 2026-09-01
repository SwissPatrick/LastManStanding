from datetime import datetime, timezone
import numpy as np
import pytest
from lms_optimizer.milp import milp_optimize
from lms_optimizer.optimizer import PortfolioOptimizer, PortfolioWeights
from lms_optimizer.simulation import adaptive_multi_round_simulation
from lms_optimizer.forecast_snapshot import ForecastSnapshot, ForecastStore
from lms_optimizer.weekly import RecommendationSnapshot, WeeklyStore

def test_adaptive_simulation_converges_reproducibly():
    args = ({"e1":"A"}, [({"f":np.array([1.,0.,0.])}, {"f":("A","B")})])
    a = adaptive_multi_round_simulation(*args, minimum_runs=100, maximum_runs=200, batch_size=50, target_standard_error=.01, target_ci_width=.04)
    b = adaptive_multi_round_simulation(*args, minimum_runs=100, maximum_runs=200, batch_size=50, target_standard_error=.01, target_ci_width=.04)
    assert a.converged and a.stopping_reason.startswith("all metrics")
    assert np.array_equal(a.survivor_counts, b.survivor_counts)

def test_adaptive_simulation_reports_non_convergence():
    summary = adaptive_multi_round_simulation({"e1":"A"}, [({"f":np.array([.5,.0,.5])}, {"f":("A","B")})], minimum_runs=10, maximum_runs=20, batch_size=10, target_standard_error=1e-12, target_ci_width=1e-12)
    assert not summary.converged and "maximum" in summary.stopping_reason

def test_milp_matches_exact_expected_survivor_oracle():
    candidates = {"e1":["A","B"],"e2":["A","B"]}; scenarios = [{"A":True,"B":False},{"A":False,"B":True}]
    exact = PortfolioOptimizer(candidates, scenarios, PortfolioWeights(at_least_one=0, wipeout=0)).optimize()[1]["expected_survivors"]
    result = milp_optimize(candidates, scenarios, PortfolioWeights(at_least_one=0, wipeout=0))
    assert result.feasible and result.allocation and result.objective == pytest.approx(exact)

def test_forecast_snapshot_is_immutable_and_versioned(tmp_path):
    manifest = tmp_path / "manifest.json"; manifest.write_text("{}")
    store = ForecastStore(tmp_path / "forecasts"); snap = store.create(datetime.now(timezone.utc), datetime.now(timezone.utc), manifest, "elo", "1", [])
    store.save(snap)
    with pytest.raises(FileExistsError): store.save(snap)

def test_weekly_lock_and_whatsapp_message(tmp_path):
    store = WeeklyStore(tmp_path); snap = RecommendationSnapshot(version="v1", created_at=datetime.now(timezone.utc), season="2026/27", round_number=1, odds_snapshot_version="o1", forecast_snapshot_version="f1", active_entries=["e1"], used_teams={"e1":[]}, objective_weights={}, exposure_limits={}, simulation_settings={}, seed=7, optimiser_version="1", allocation={"e1":"A"}, risk_estimates={})
    store.save(snap); locked = store.lock("v1")
    assert locked.exists() and "e1: A" in store.whatsapp_message(snap)
