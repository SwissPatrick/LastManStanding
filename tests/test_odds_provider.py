import json
from datetime import datetime, timezone
from urllib.error import HTTPError

import pytest

from lms_optimizer.providers import OddsApiProvider, ProviderError, normalise_team


class Response:
    status = 200
    def __init__(self, payload, headers=None): self.payload = payload; self.headers = headers or {"x-requests-remaining": "499"}
    def read(self): return json.dumps(self.payload).encode()


def payload():
    return [{"id": "event-1", "sport_key": "soccer_epl", "commence_time": "2030-08-10T15:00:00Z", "home_team": "Man Utd", "away_team": "Spurs", "bookmakers": [{"key": "uk-book", "title": "UK Book", "last_update": "2030-08-01T12:00:00Z", "markets": [{"key": "h2h", "outcomes": [{"name": "Draw", "price": 3.5}, {"name": "Tottenham Hotspur", "price": 4.0}, {"name": "Manchester United", "price": 1.8}]}]}]}]


def test_missing_key_and_secret_redaction(tmp_path):
    with pytest.raises(ProviderError, match="ODDS_API_KEY"):
        OddsApiProvider(api_key="", cache_dir=tmp_path).current_odds()
    provider = OddsApiProvider(api_key="secret-value", cache_dir=tmp_path, opener=lambda *args, **kwargs: Response(payload()))
    result = provider.current_odds(force_refresh=True)
    assert "secret-value" not in json.dumps(result.request_parameters)
    assert result.events[0].event_id == "event-1"


def test_aliases_outcome_names_and_bookmaker_coverage(tmp_path):
    result = OddsApiProvider(api_key="x", cache_dir=tmp_path, opener=lambda *args, **kwargs: Response(payload())).current_odds(force_refresh=True)
    event = result.events[0]; book = event.bookmakers[0]
    assert normalise_team("Man Utd") == "Manchester United" and normalise_team("Spurs") == "Tottenham"
    assert book.included and {item.name for item in book.outcomes} == {"Draw", "Tottenham Hotspur", "Manchester United"}
    assert result.quota_headers["x-requests-remaining"] == "499"


def test_incomplete_market_excluded_and_cache_hit(tmp_path):
    bad = payload(); bad[0]["bookmakers"][0]["markets"][0]["outcomes"] = [{"name": "Manchester United", "price": 1.8}, {"name": "Tottenham Hotspur", "price": 4.0}]
    calls = []
    def opener(*args, **kwargs): calls.append(1); return Response(bad)
    provider = OddsApiProvider(api_key="x", cache_dir=tmp_path, opener=opener)
    first = provider.current_odds(force_refresh=True); second = provider.current_odds()
    assert not first.events[0].bookmakers[0].included and second.from_cache and len(calls) == 1


def test_authentication_failure_is_redacted(tmp_path):
    def opener(*args, **kwargs): raise HTTPError("https://api.the-odds-api.com", 401, "secret-value", {}, None)
    with pytest.raises(ProviderError, match="authentication") as error:
        OddsApiProvider(api_key="secret-value", cache_dir=tmp_path, opener=opener, sleep=lambda _: None).current_odds(force_refresh=True)
    assert "secret-value" not in str(error.value)


def test_transient_retry_and_immutable_raw_response(tmp_path):
    calls = []
    def opener(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1: raise TimeoutError()
        return Response(payload())
    provider = OddsApiProvider(api_key="x", cache_dir=tmp_path, opener=opener, sleep=lambda _: None)
    result = provider.current_odds(force_refresh=True)
    assert len(calls) == 2 and result.raw_storage_reference.startswith(str(tmp_path))
    raw_files = list(tmp_path.glob("raw-*.json")); assert len(raw_files) == 1 and raw_files[0].exists()
