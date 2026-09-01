# Agent instructions: Last Man Standing Optimiser

## Objective

Build a local Python application that recommends coordinated Premier League Last Man Standing selections for multiple related entries. It combines bookmaker odds, football models, future fixture value, portfolio optimisation, Monte Carlo risk analysis, and optional separate financial hedge calculations. It must remain a local PyCharm project, not a hosted SaaS product.

## Competition rules

- Only Premier League matches are eligible.
- A team can be used once per entry for a season; reuse across different entries is allowed.
- A round is selectable only when at least six eligible matches are scheduled.
- Track selection deadlines, postponements, cancellations, and abandoned matches.
- Model automatic selection and progression rules explicitly when those rules are confirmed.
- Support multiple players and a maximum of ten paid entries per individual.
- Support rollover into a new season.
- Keep a complete audit trail of inputs, selections, results, and state changes.
- Never infer that a postponed/cancelled match is a win; preserve its unresolved state until competition rules determine the outcome.

## Mathematical and data requirements

- Never treat inverse decimal odds as fair probabilities without removing overround.
- Support proportional, additive, power, Shin, weighted consensus, exchange weighting, reliability weighting, stale-price detection, market disagreement, and favourite-longshot correction.
- Football models must include time-decayed attack/defence ratings, home advantage, Poisson scorelines, Dixon-Coles low-score correction, Elo, optional xG, rest/congestion, promoted-team uncertainty, and documented uncertainty intervals.
- Use chronological training/testing. No future results, closing prices unavailable at prediction time, end-of-season ratings, or random splits may leak into a prediction.
- Learn ensemble weights from training data. Do not hard-code permanent production weights.
- All scoreline and outcome probabilities must be finite, non-negative, and normalised.
- Reproducible simulations must use explicit random seeds and report Monte Carlo standard errors.

## Architecture

- models.py: Pydantic domain and historical data models.
- rules.py: deterministic competition rules.
- data.py: validated CSV import, normalisation, duplicate/missing checks, raw preservation.
- probability.py: market probability conversion.
- modeling.py: time-decayed Dixon-Coles and scoreline maths.
- elo.py: chronological Elo ratings and probabilities.
- ensemble.py: training-only learned calibrated stacking.
- backtest.py: expanding-window evaluation and benchmark comparison.
- market_strength.py: market-implied Bradley-Terry-style future strength forecasts.
- validation.py: leakage-safe final-season model validation and bootstrap differences.
- optimizer.py: memoised Bellman recursion and exact small-cartel allocation.
- simulation.py: correlated outcome Monte Carlo with standard errors.
- lms_eval.py: retrospective LMS policy evaluation.
- storage.py: SQLite persistence and audit log.
- workflow.py: application use cases.
- app.py: Streamlit presentation only.

Preserve raw imports separately from processed data. All records need source, collection time, and market timestamp where relevant. Clearly label synthetic/demo data.

## Testing standards

Run python -m pytest -q after meaningful changes. Add unit, integration, mathematical, parameter-recovery, no-leakage, database rollback, competition-rule, and streamlit.testing.v1.AppTest UI tests. Tests must cover five/six/seven fixtures, deadline boundaries, UTC/local conversion, duplicate players/entries, ten-entry limit, reuse, independent availability, postponed/cancelled/abandoned states, fallback preview, transaction rollback, audit logs, and survival transitions. Do not count health endpoints as UI tests.

## Security

- Keep API credentials in local .env; maintain a safe .env.example.
- Never commit secrets, session tokens, banking information, private reports, or local databases.
- Do not place bets automatically.
- Do not present any result as guaranteed profit; show uncertainty and worst-case exposure.

## Version control

- Review `git status` before and after every task.
- Never commit secrets, session tokens, banking information, or personal data.
- Run `python -m pytest -q` before commits.
- Make milestone-focused commits.
- Never force-push.
- Never rewrite shared history without explicit permission.
- Push completed, passing milestones to GitHub.
- Leave unrelated user changes untouched.

## Staged delivery plan

1. Project structure and competition-rule engine.
2. Manual fixtures and odds input.
3. Fair-odds probability engine.
4. Weekly recommendation dashboard.
5. Entry and used-team tracking.
6. Poisson, Dixon-Coles, and Elo models.
7. Chronological backtesting and calibration.
8. Future fixture-value engine.
9. Multi-entry portfolio optimiser.
10. Monte Carlo risk analysis.
11. External fixture/odds integrations with caching.
12. Optional separate financial hedge calculator.

Complete and verify each stage before later stages depend on it. Do not add future-team value, cartel optimisation, live APIs, Monte Carlo season simulation, or financial hedging until explicitly staged.
