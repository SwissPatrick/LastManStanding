"""Provider boundary for The Odds API v4; secrets never cross this module's result types."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


TEAM_ALIASES = {"manchester united": "Manchester United", "man utd": "Manchester United", "manchester city": "Manchester City", "man city": "Manchester City", "tottenham hotspur": "Tottenham", "spurs": "Tottenham", "west ham united": "West Ham", "newcastle united": "Newcastle", "brighton and hove albion": "Brighton", "nottingham forest": "Nottingham Forest", "wolves": "Wolverhampton Wanderers"}


def normalise_team(name: str) -> str:
    clean = " ".join(name.strip().lower().split())
    return TEAM_ALIASES.get(clean, " ".join(part.capitalize() for part in clean.split()))


@dataclass(frozen=True)
class ProviderOutcome:
    name: str
    price: float


@dataclass(frozen=True)
class ProviderBookmaker:
    key: str
    title: str
    last_update: datetime | None
    outcomes: tuple[ProviderOutcome, ...]
    included: bool = True
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class ProviderEvent:
    event_id: str
    home_team: str
    away_team: str
    kickoff: datetime
    bookmakers: tuple[ProviderBookmaker, ...]
    home_score: int | None = None
    away_score: int | None = None
    status: str | None = None


@dataclass(frozen=True)
class ProviderResponse:
    provider: str
    endpoint_type: str
    retrieved_at: datetime
    http_status: int
    checksum: str
    request_parameters: dict[str, str]
    quota_headers: dict[str, str]
    raw_storage_reference: str
    events: tuple[ProviderEvent, ...]
    from_cache: bool = False
    stale: bool = False


class Provider(Protocol):
    def current_odds(self, force_refresh: bool = False) -> ProviderResponse: ...
    def recent_scores(self, force_refresh: bool = False) -> ProviderResponse: ...


class ProviderError(RuntimeError):
    pass


class OddsApiProvider:
    provider_name = "The Odds API v4"

    def __init__(self, api_key: str | None = None, cache_dir: str | Path = "data/provider_cache", cache_seconds: int = 300, timeout: float = 10.0, opener: Callable = urlopen, sleep: Callable = time.sleep):
        self.api_key = api_key if api_key is not None else os.getenv("ODDS_API_KEY", "").strip()
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_seconds = max(300, int(cache_seconds)); self.timeout = timeout; self.opener = opener; self.sleep = sleep

    def _request(self, endpoint_type: str, path: str, parameters: dict[str, str], force_refresh: bool) -> ProviderResponse:
        safe_parameters = dict(parameters); safe_parameters["apiKey"] = "[REDACTED]"
        cache_key = hashlib.sha256(json.dumps({"path": path, "parameters": parameters}, sort_keys=True).encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if not force_refresh and cache_path.exists() and time.time() - cache_path.stat().st_mtime < self.cache_seconds:
            payload = json.loads(cache_path.read_text(encoding="utf-8")); return self._parse(endpoint_type, payload, True)
        if not self.api_key: raise ProviderError("ODDS_API_KEY is not configured; manual mode remains available.")
        url = "https://api.the-odds-api.com/v4" + path + "?" + urlencode({**parameters, "apiKey": self.api_key})
        error = None
        for attempt in range(3):
            try:
                response = self.opener(Request(url, headers={"Accept": "application/json"}), timeout=self.timeout)
                raw = response.read(); status = int(getattr(response, "status", 200)); headers = {key.lower(): value for key, value in response.headers.items()}; data = json.loads(raw.decode("utf-8")); break
            except HTTPError as exc:
                if exc.code in (401, 403): raise ProviderError("Odds provider authentication failed; check ODDS_API_KEY.")
                if exc.code == 429: error = ProviderError("Odds provider rate limit reached; use cached data or try later.")
                else: error = ProviderError(f"Odds provider HTTP error {exc.code}.")
            except (TimeoutError, URLError, OSError, json.JSONDecodeError) as exc:
                error = ProviderError("Odds provider request failed; cached/manual data may still be used.")
            if attempt < 2: self.sleep(2 ** attempt)
        else:
            if cache_path.exists():
                cached = self._parse(endpoint_type, json.loads(cache_path.read_text(encoding="utf-8")), True)
                return replace(cached, stale=True)
            raise error or ProviderError("Odds provider request failed.")
        checksum = hashlib.sha256(raw).hexdigest(); raw_ref = self.cache_dir / f"raw-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{checksum[:16]}.json"
        if not raw_ref.exists(): raw_ref.write_bytes(raw)
        metadata = {"provider": self.provider_name, "endpoint_type": endpoint_type, "retrieved_at": datetime.now(timezone.utc).isoformat(), "http_status": status, "checksum": checksum, "request_parameters": safe_parameters, "quota_headers": {key: value for key, value in headers.items() if "request" in key or "remaining" in key or "used" in key}, "raw_storage_reference": str(raw_ref), "payload": data}
        cache_path.write_text(json.dumps(metadata, default=str), encoding="utf-8")
        return self._parse(endpoint_type, metadata, False)

    def _parse(self, endpoint_type: str, payload: dict, from_cache: bool) -> ProviderResponse:
        data = payload.get("payload", payload); events = []
        for event in data:
            if not isinstance(event, dict) or not event.get("id"): continue
            books = []
            for bookmaker in event.get("bookmakers", []):
                market = next((market for market in bookmaker.get("markets", []) if market.get("key") == "h2h"), None)
                outcomes = tuple(ProviderOutcome(str(item.get("name", "")), float(item["price"])) for item in (market or {}).get("outcomes", []) if item.get("name") and item.get("price") is not None)
                names = {normalise_team(item.name) for item in outcomes}; expected = {normalise_team(event.get("home_team", "")), normalise_team(event.get("away_team", "")), "Draw"}
                included = len(outcomes) == 3 and expected.issubset(names) and all(item.price > 1 for item in outcomes)
                books.append(ProviderBookmaker(str(bookmaker.get("key", "")), str(bookmaker.get("title", bookmaker.get("key", ""))), datetime.fromisoformat(bookmaker["last_update"].replace("Z", "+00:00")) if bookmaker.get("last_update") else None, outcomes, included, None if included else "incomplete or invalid three-way h2h market"))
            scores = {normalise_team(item.get("name", "")): item.get("score") for item in event.get("scores", [])}
            home = normalise_team(event.get("home_team", "")); away = normalise_team(event.get("away_team", ""))
            events.append(ProviderEvent(str(event["id"]), home, away, datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00")), tuple(books), int(scores[home]) if scores.get(home) is not None else None, int(scores[away]) if scores.get(away) is not None else None, str(event.get("status", "completed")) if event.get("scores") else None))
        return ProviderResponse(self.provider_name, endpoint_type, datetime.fromisoformat(payload.get("retrieved_at", datetime.now(timezone.utc).isoformat())), int(payload.get("http_status", 200)), str(payload.get("checksum", hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest())), payload.get("request_parameters", {"apiKey": "[REDACTED]"}), payload.get("quota_headers", {}), str(payload.get("raw_storage_reference", "cache")), tuple(events), from_cache, False)

    def current_odds(self, force_refresh: bool = False) -> ProviderResponse:
        return self._request("current_odds", "/sports/soccer_epl/odds", {"regions": "uk", "markets": "h2h", "oddsFormat": "decimal"}, force_refresh)

    def recent_scores(self, force_refresh: bool = False) -> ProviderResponse:
        return self._request("recent_scores", "/sports/soccer_epl/scores", {"daysFrom": "3"}, force_refresh)


def live_smoke_test() -> dict[str, object]:
    response = OddsApiProvider().current_odds(force_refresh=True)
    return {"provider": response.provider, "events": len(response.events), "bookmakers": sum(len(event.bookmakers) for event in response.events), "quota_headers": response.quota_headers, "checksum": response.checksum}
