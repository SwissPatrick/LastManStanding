"""Read-only minimum-credit The Odds API smoke test."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lms_optimizer.providers import OddsApiProvider, ProviderError, live_smoke_test


if __name__ == "__main__":
    if not OddsApiProvider().api_key:
        print("ODDS_API_KEY is not configured; smoke test skipped (manual mode remains available).")
    else:
        try:
            print(json.dumps(live_smoke_test(), indent=2))
        except ProviderError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1)
