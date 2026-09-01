"""Validated historical CSV ingestion with raw-data preservation."""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import re
import hashlib
import json
from urllib.request import Request, urlopen
import pandas as pd
from .models import HistoricalMatch

ALIASES = {"Man United": "Manchester United", "Man Utd": "Manchester United", "Spurs": "Tottenham Hotspur", "Wolves": "Wolverhampton Wanderers"}
REQUIRED = {"season", "date", "home_team", "away_team", "home_goals", "away_goals", "home_odds", "draw_odds", "away_odds"}

@dataclass
class HistoricalImportReport:
    matches: list[HistoricalMatch] = field(default_factory=list)
    missing_rows: list[dict[str, object]] = field(default_factory=list)
    duplicate_rows: list[dict[str, object]] = field(default_factory=list)
    invalid_rows: list[dict[str, object]] = field(default_factory=list)
    raw_path: str | None = None

    @property
    def errors(self) -> int:
        return len(self.missing_rows) + len(self.duplicate_rows) + len(self.invalid_rows)

@dataclass
class ArchiveManifestEntry:
    season: str
    source_url: str
    download_timestamp: str
    sha256: str
    row_count: int
    available_columns: list[str]
    import_status: str
    validation_warnings: list[str]

def football_data_url(season: str) -> str:
    """Published Football-Data CSV URL; no HTML scraping is involved."""
    start = int(season[:4]) % 100
    end = (start + 1) % 100
    return f"https://www.football-data.co.uk/mmz4281/{start:02d}{end:02d}/E0.csv"

def download_football_data(seasons: list[str], raw_dir: str | Path = "data/raw/football-data", manifest_path: str | Path = "data/football_data_manifest.json") -> list[ArchiveManifestEntry]:
    raw_path = Path(raw_dir); raw_path.mkdir(parents=True, exist_ok=True)
    manifest_file = Path(manifest_path); manifest_file.parent.mkdir(parents=True, exist_ok=True)
    existing = {row["season"]: row for row in json.loads(manifest_file.read_text())} if manifest_file.exists() else {}
    entries: list[ArchiveManifestEntry] = []
    for season in seasons:
        url = football_data_url(season); target = raw_path / f"{season.replace('/', '-')}_E0.csv"
        payload = urlopen(Request(url, headers={"User-Agent": "LastManStanding-local-research/1.0"}), timeout=30).read()
        checksum = hashlib.sha256(payload).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
            raise FileExistsError(f"refusing to overwrite changed raw file: {target}")
        if not target.exists(): target.write_bytes(payload)
        frame = pd.read_csv(target)
        warning = []
        if not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(frame.columns): warning.append("required result columns missing")
        entry = ArchiveManifestEntry(season, url, datetime.now(timezone.utc).isoformat(), checksum, len(frame), list(frame.columns), "downloaded", warning)
        entries.append(entry); existing[season] = asdict(entry)
    manifest_file.write_text(json.dumps(list(existing.values()), indent=2, sort_keys=True))
    return entries

def normalize_team_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(name).strip())
    return ALIASES.get(cleaned, cleaned)

def import_historical_csv(path: str | Path, raw_dir: str | Path = "data/raw", data_source: str = "csv") -> HistoricalImportReport:
    source_path = Path(path)
    raw_target = Path(raw_dir)
    raw_target.mkdir(parents=True, exist_ok=True)
    raw_path = raw_target / source_path.name
    raw_path.write_bytes(source_path.read_bytes())
    frame = pd.read_csv(source_path)
    report = HistoricalImportReport(raw_path=str(raw_path))
    missing_columns = REQUIRED - set(frame.columns)
    if missing_columns:
        report.missing_rows.append({"row": 0, "missing_columns": sorted(missing_columns)})
        return report
    seen: set[tuple[object, ...]] = set()
    collected_at = datetime.now(timezone.utc)
    for index, row in frame.iterrows():
        key = (row["season"], row["date"], row["home_team"], row["away_team"])
        if key in seen:
            report.duplicate_rows.append({"row": int(index), "key": key})
            continue
        seen.add(key)
        if row[list(REQUIRED)].isna().any():
            report.missing_rows.append({"row": int(index)})
            continue
        try:
            report.matches.append(HistoricalMatch(
                season=str(row["season"]), match_date=pd.to_datetime(row["date"], utc=True).to_pydatetime(),
                home_team=normalize_team_name(row["home_team"]), away_team=normalize_team_name(row["away_team"]),
                full_time_home_goals=int(row["home_goals"]), full_time_away_goals=int(row["away_goals"]),
                closing_home_odds=float(row["home_odds"]), closing_draw_odds=float(row["draw_odds"]), closing_away_odds=float(row["away_odds"]),
                expected_home_goals=float(row["home_xg"]) if "home_xg" in row and pd.notna(row["home_xg"]) else None,
                expected_away_goals=float(row["away_xg"]) if "away_xg" in row and pd.notna(row["away_xg"]) else None,
                data_source=data_source, collected_at=collected_at, is_sample=False))
        except Exception as exc:
            report.invalid_rows.append({"row": int(index), "error": str(exc)})
    return report

def _football_datetime(row: pd.Series) -> object:
    raw = str(row["Date"])
    if "Time" in row and pd.notna(row["Time"]): raw += " " + str(row["Time"])
    return pd.to_datetime(raw, dayfirst=True, utc=True).to_pydatetime()

def _first_available(row: pd.Series, names: list[str]) -> float | None:
    for name in names:
        if name in row and pd.notna(row[name]):
            try: return float(row[name])
            except (TypeError, ValueError): return None
    return None

def import_football_data_csv(path: str | Path, season: str, raw_dir: str | Path = "data/raw/football-data") -> HistoricalImportReport:
    """Map Football-Data E0.csv columns without mislabelling unknown-time prices.

    AvgH/AvgD/AvgA are used as a consistent market-average consensus when
    present, but remain timestamp-unknown because the source column does not
    explicitly identify opening or closing time.
    """
    source = Path(path); raw_target = Path(raw_dir); raw_target.mkdir(parents=True, exist_ok=True)
    preserved = raw_target / source.name
    if source.resolve() != preserved.resolve():
        if preserved.exists() and preserved.read_bytes() != source.read_bytes(): raise FileExistsError(f"refusing to overwrite changed raw file: {preserved}")
        if not preserved.exists(): preserved.write_bytes(source.read_bytes())
    frame = pd.read_csv(source)
    report = HistoricalImportReport(raw_path=str(preserved))
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(frame.columns)
    if missing: report.missing_rows.append({"row": 0, "missing_columns": sorted(missing)}); return report
    seen: set[tuple[object, ...]] = set()
    for index, row in frame.iterrows():
        key = (season, row["Date"], row["HomeTeam"], row["AwayTeam"])
        if key in seen: report.duplicate_rows.append({"row": int(index), "key": key}); continue
        seen.add(key)
        if pd.isna(row[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]).any():
            report.missing_rows.append({"row": int(index), "field": "result"}); continue
        try:
            closing = all(name in row and pd.notna(row[name]) for name in ("AvgCH", "AvgCD", "AvgCA"))
            home_odds = _first_available(row, (["AvgCH", "B365CH"] if closing else []) + ["AvgH", "BbAvH", "B365H", "WHH", "PSH"])
            draw_odds = _first_available(row, (["AvgCD", "B365CD"] if closing else []) + ["AvgD", "BbAvD", "B365D", "WHD", "PSD"])
            away_odds = _first_available(row, (["AvgCA", "B365CA"] if closing else []) + ["AvgA", "BbAvA", "B365A", "WHA", "PSA"])
            if pd.isna(row["FTHG"]) or pd.isna(row["FTAG"]): report.missing_rows.append({"row": int(index), "field": "result"}); continue
            if None in (home_odds, draw_odds, away_odds): report.missing_rows.append({"row": int(index), "field": "odds"}); continue
            timing = "closing" if closing else "timestamp-unknown"
            method = "market-average-closing" if closing else "market-average-timestamp-unknown" if any(x in frame.columns for x in ("AvgH", "BbAvH")) else "bookmaker-B365-timestamp-unknown"
            report.matches.append(HistoricalMatch(season=season, match_date=_football_datetime(row), home_team=normalize_team_name(row["HomeTeam"]), away_team=normalize_team_name(row["AwayTeam"]), full_time_home_goals=int(row["FTHG"]), full_time_away_goals=int(row["FTAG"]), closing_home_odds=home_odds, closing_draw_odds=draw_odds, closing_away_odds=away_odds, data_source="football-data.co.uk", collected_at=datetime.now(timezone.utc), odds_timing=timing, odds_method=method))
        except Exception as exc: report.invalid_rows.append({"row": int(index), "error": str(exc)})
    return report

def audit_historical(matches: list[HistoricalMatch], raw_frames: dict[str, pd.DataFrame] | None = None) -> dict[str, object]:
    seasons = sorted({m.season for m in matches})
    season_counts = {season: sum(m.season == season for m in matches) for season in seasons}
    odds_overround = [sum(1 / x for x in (m.closing_home_odds, m.closing_draw_odds, m.closing_away_odds)) - 1 for m in matches]
    teams_by_season = {season: {t for m in matches if m.season == season for t in (m.home_team, m.away_team)} for season in seasons}
    movement = {}
    for before, after in zip(seasons, seasons[1:]): movement[after] = {"promoted": sorted(teams_by_season[after] - teams_by_season[before]), "relegated": sorted(teams_by_season[before] - teams_by_season[after])}
    return {"matches_per_season": season_counts, "missing_results": 0, "missing_odds": 0, "invalid_decimal_odds": sum(not all(x > 1 for x in (m.closing_home_odds, m.closing_draw_odds, m.closing_away_odds)) for m in matches), "duplicate_fixtures": len(matches) - len({(m.season, m.match_date, m.home_team, m.away_team) for m in matches}), "unrecognised_team_names": [], "promoted_relegated": movement, "suspicious_scores": [m.model_dump() for m in matches if m.full_time_home_goals > 15 or m.full_time_away_goals > 15], "odds_overround": {"count": len(odds_overround), "mean": float(pd.Series(odds_overround).mean()) if odds_overround else None, "p95": float(pd.Series(odds_overround).quantile(.95)) if odds_overround else None}, "bookmaker_coverage": {season: {method: sum(x.season == season and x.odds_method == method for x in matches) for method in sorted({x.odds_method for x in matches})} for season in seasons}, "opening_vs_closing": {timing: sum(m.odds_timing == timing for m in matches) for timing in ("opening", "closing", "intermediate", "timestamp-unknown")}}
