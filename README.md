# Last Man Standing Optimiser

Local Python 3.12+ application for coordinated Premier League Last Man Standing analysis. It is not a hosted service and does not place bets.

## Purpose

This is a local Python application for coordinated Premier League Last Man
Standing analysis. It is not a hosted service and does not place bets.

## Setup and run

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run main.py
python -m pytest -q
```

Python 3.12 or later is supported. Copy `.env.example` to `.env` only when
local credentials are needed; real credentials are never committed.

Refresh the documented Football-Data archive with:

```powershell
python scripts\refresh_football_data.py
python scripts\import_real_history.py
python scripts\evaluate_historical_cartels.py
```

Raw datasets, downloaded archives, local databases and large generated
cohort/decision outputs are intentionally excluded from version control.

The verified foundation provides validated fixture, odds, entry and selection models; deterministic six-match/deadline/reuse rules; SQLite storage with raw-import separation and audit logging; and four margin-removal methods. The dashboard supports manual input and an optional The Odds API v4 provider for current EPL fixtures, UK h2h odds and recent scores. Provider data is cached locally for at least five minutes, raw responses are immutable and ignored by Git, and a force refresh consumes a new provider request.

## Optional automatic EPL refresh

Obtain a key from [The Odds API](https://the-odds-api.com/) and set it only in a local `.env` or process environment:

```powershell
$env:ODDS_API_KEY = "your-local-key"
streamlit run main.py
```

The normal odds refresh requests one `soccer_epl` UK `h2h` decimal market and uses one request credit; cached refreshes do not make a request. The dashboard never displays or stores the key. Without the key, use manual fixture entry, CSV paste and manual results. Provider timestamps mean retrieval time, bookmaker market-update time, and the event kickoff time; they are distinct and are shown separately.

Provider result history is limited by the provider's recent-score window. Unmatched or ambiguous results are proposals only and require manual resolution and confirmation before LMS advancement. If authentication, quota, timeout or network errors occur, the last successful local response remains available where present.

For a read-only smoke test (no bets and no locked-selection changes), run:

```powershell
python scripts\smoke_odds_api.py
```

Troubleshooting: confirm `ODDS_API_KEY` is set in the process running Streamlit, wait for the cache window or use the explicit force-refresh control, and check quota headers in the refresh details. Never paste keys into fixture CSV, screenshots, reports or issue descriptions.

All imported records must retain source and collection/market timestamps. Sample
or demonstration data must be labelled as such. Current limitations include
manual/local data operation, no live API integration, no automated betting,
and no financial hedge execution. Models should be extended only when
chronological out-of-sample evaluation supports them.

This software is for research and decision support only. Football outcomes and
model probabilities are uncertain; users are responsible for complying with
applicable laws and avoiding unaffordable risk.
