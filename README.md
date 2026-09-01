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

The verified foundation provides validated fixture, odds, entry and selection models; deterministic six-match/deadline/reuse rules; SQLite storage with raw-import separation and audit logging; and four margin-removal methods. The dashboard accepts manual three-way odds and labels them as analysis input, not live data.

All imported records must retain source and collection/market timestamps. Sample
or demonstration data must be labelled as such. Current limitations include
manual/local data operation, no live API integration, no automated betting,
and no financial hedge execution. Models should be extended only when
chronological out-of-sample evaluation supports them.

This software is for research and decision support only. Football outcomes and
model probabilities are uncertain; users are responsible for complying with
applicable laws and avoiding unaffordable risk.
