"""Scenario MILP optimiser with exact objective components and validation."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from .optimizer import PortfolioOptimizer, PortfolioWeights

@dataclass
class MILPResult:
    allocation: dict[str, str]
    objective: float
    feasible: bool
    message: str
    raw_result: object
    components: dict[str, float] | None = None

def milp_optimize(entry_candidates: dict[str, list[str]], scenarios: list[dict[str, bool]], weights: PortfolioWeights | None = None, exposure_cap: int | dict[str, int] | None = None, soft_exposure_cap: int | dict[str, int] | None = None, soft_penalty: float | None = None, scenario_probabilities: list[float] | None = None, future_values: dict[str, dict[str, float]] | None = None) -> MILPResult:
    weights = weights or PortfolioWeights(); entries = list(entry_candidates); teams = sorted({team for values in entry_candidates.values() for team in values}); pairs = [(entry, team) for entry in entries for team in entry_candidates[entry]]
    n_x = len(pairs); n_s = len(scenarios)
    if not pairs: return MILPResult({}, float("nan"), False, "no eligible entry/team variables", None)
    oracle = PortfolioOptimizer(entry_candidates, scenarios, weights, exposure_cap, soft_exposure_cap, soft_penalty, scenario_probabilities, future_values)
    probabilities = oracle.scenario_probabilities; n = n_x + n_s + 1 + n_s + len(teams) + sum(len(entry_candidates[e]) for e in teams if False)
    y_keys = [(team, k) for team in teams for k in range(1, len(entries)+1)]
    eta_i = n_x+n_s; u_start=eta_i+1; slack_start=u_start+n_s; y_start=slack_start+len(teams); n=y_start+len(y_keys)
    c=np.zeros(n); integrality=np.zeros(n); integrality[:n_x+n_s]=1; lower=np.zeros(n); upper=np.full(n, np.inf); upper[:n_x+n_s]=1; lower[eta_i]=0; upper[eta_i]=len(entries)
    soft_caps=oracle.soft_cap; hard_caps=oracle.hard_cap; soft_penalty=oracle.soft_penalty
    # Minimise negative benefits plus penalties. Wipeout contributes a constant -w_wipeout.
    for i, (entry, team) in enumerate(pairs):
        c[i] = -weights.expected_survivors * sum(probabilities[s] * bool(scenario.get(team, False)) for s, scenario in enumerate(scenarios)) - weights.future_value * future_values.get(entry, {}).get(team, 0.) if future_values else -weights.expected_survivors * sum(probabilities[s] * bool(scenario.get(team, False)) for s, scenario in enumerate(scenarios))
    for s in range(n_s): c[n_x+s] = -(weights.at_least_one + weights.wipeout) * probabilities[s]
    c[eta_i] = weights.cvar; c[u_start:u_start+n_s] = weights.cvar * probabilities / (1-weights.cvar_alpha)
    c[slack_start:slack_start+len(teams)] = soft_penalty
    c[y_start:] = [weights.concentration * (2*k-1) for _, k in y_keys]
    rows=[]; lows=[]; highs=[]
    for entry in entries:
        row=np.zeros(n); row[[i for i,(e,_) in enumerate(pairs) if e==entry]]=1; rows.append(row); lows.append(1); highs.append(1)
    for s, scenario in enumerate(scenarios):
        row=np.zeros(n); row[n_x+s]=-1
        for i,(_,team) in enumerate(pairs): row[i] += bool(scenario.get(team, False))
        rows.append(row); lows.append(0); highs.append(np.inf) # z <= survivors
        row=np.zeros(n)
        for i,(_,team) in enumerate(pairs): row[i] += bool(scenario.get(team, False))
        row[n_x+s] -= len(entries)
        rows.append(row); lows.append(-np.inf); highs.append(0) # survivors <= N z
        row=np.zeros(n); row[eta_i]=-1; row[u_start+s]=-1
        for i,(_,team) in enumerate(pairs): row[i] += -bool(scenario.get(team, False))
        rows.append(row); lows.append(-np.inf); highs.append(-len(entries)) # u >= loss-eta
    for team in teams:
        exposure=np.zeros(n); exposure[[i for i,(_,t) in enumerate(pairs) if t==team]]=1
        if team in hard_caps:
            rows.append(exposure); lows.append(-np.inf); highs.append(hard_caps[team])
        if team in soft_caps:
            row=exposure.copy(); row[slack_start+teams.index(team)] -= 1; rows.append(row); lows.append(-np.inf); highs.append(soft_caps[team])
        y_indices=[y_start+j for j,(t,_) in enumerate(y_keys) if t==team]
        row=exposure.copy(); row[y_indices] -= 1; rows.append(row); lows.append(0); highs.append(0)
        for prior, current in zip(y_indices, y_indices[1:]):
            row=np.zeros(n); row[current]=1; row[prior]=-1; rows.append(row); lows.append(-np.inf); highs.append(0)
    result=milp(c, integrality=integrality, bounds=Bounds(lower,upper), constraints=LinearConstraint(np.asarray(rows), np.asarray(lows), np.asarray(highs)), options={"time_limit":30})
    if not result.success: return MILPResult({}, float("nan"), False, result.message, result)
    if not all(abs(result.x[i]-round(result.x[i])) <= 1e-6 for i in range(n_x+n_s)): return MILPResult({}, float("nan"), False, "solver returned non-integral decision variables", result)
    allocation={entry:team for i,(entry,team) in enumerate(pairs) if result.x[i] > .5}
    if len(allocation)!=len(entries) or any(entry not in allocation for entry in entries): return MILPResult({}, float("nan"), False, "returned allocation violates entry allocation constraints", result)
    components=oracle.components(allocation); objective=components.objective
    if not np.isfinite(objective) or abs((-result.fun - weights.wipeout) - objective) > 1e-5: return MILPResult({}, float("nan"), False, "solver objective failed independent allocation validation", result)
    return MILPResult(allocation, objective, True, result.message, result, components.__dict__.copy())

