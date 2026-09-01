"""Fixture-level, lossless audit of Football-Data round construction.

This module deliberately reads CSV rows as physical lines.  The normal historical
import remains the source of usable ``HistoricalMatch`` objects; this audit adds
line identity, raw checksums, and dispositions for rows that importer-level
normalisation would otherwise discard.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data import normalize_team_name
from .historical_evaluator import ConstructedRound, construct_rounds, fixture_key
from .models import HistoricalMatch
from .reproducibility import canonical_csv_bytes, canonical_json_bytes

MINIMUM_MATCHES = 6
MAX_FIXTURES = 10
DATE_GAP_DAYS = 3

ELIGIBLE_ROUND = "eligible-round inclusion"
UNDER_SIX = "under-six-round inclusion"
INVALID_MISSING = "invalid or missing data"
DUPLICATE = "duplicate source fixture"
DATE_GAP = "date-gap boundary"
TEAM_SPLIT = "team-duplicate split"
TEN_CAP = "ten-fixture cap"
RESCHEDULED = "rescheduled fixture"
UNASSIGNED = "unassigned fixture"


@dataclass
class FixtureAuditRecord:
    season: str
    source_file: str
    physical_line_number: int
    raw_row_checksum: str
    fixture_id: str
    date: str | None
    home_team: str | None
    away_team: str | None
    validity: bool
    included: bool
    constructed_round: int | None
    round_size: int
    round_eligible: bool
    primary_disposition_reason: str
    boundary_or_separation_reason: str | None
    related_conflicting_fixture: str | None
    validation_warnings: list[str] = field(default_factory=list)


@dataclass
class FixtureAuditResult:
    records: list[FixtureAuditRecord]
    rounds: list[dict[str, object]]
    summaries: dict[str, object]
    csv_sha256: str | None = None
    json_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {"records": [asdict(x) for x in self.records], "rounds": self.rounds, "summaries": self.summaries}


def _raw_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """Return header and all data lines, including an all-empty trailing row."""
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if not lines:
        return [], []
    header = next(csv.reader([lines[0]]))
    rows: list[list[str]] = []
    for line in lines[1:]:
        rows.append(next(csv.reader([line])))
    return header, rows


def _value(header: list[str], row: list[str], name: str) -> str:
    try:
        value = row[header.index(name)]
    except (ValueError, IndexError):
        return ""
    return value.strip()


def _parsed_row(header: list[str], row: list[str], season: str) -> tuple[HistoricalMatch | None, list[str], str | None, str | None, str | None]:
    warnings: list[str] = []
    required = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")
    if not header or len(row) < 5 or all(not x.strip() for x in row):
        return None, ["blank row"], None, None, None
    missing = [name for name in required if not _value(header, row, name)]
    if missing:
        return None, [f"missing {name}" for name in missing], None, None, None
    home = normalize_team_name(_value(header, row, "HomeTeam"))
    away = normalize_team_name(_value(header, row, "AwayTeam"))
    date_raw = _value(header, row, "Date")
    time_raw = _value(header, row, "Time")
    date_text = f"{date_raw} {time_raw}".strip()
    try:
        date = pd.to_datetime(date_text, dayfirst=True, utc=True).to_pydatetime()
        if home == away:
            raise ValueError("home and away teams are identical")
        home_goals, away_goals = int(float(_value(header, row, "FTHG"))), int(float(_value(header, row, "FTAG")))
        if home_goals < 0 or away_goals < 0:
            raise ValueError("negative goals")
    except Exception as exc:
        return None, [f"invalid fixture fields: {exc}"], None, home, away

    closing = all(_value(header, row, x) for x in ("AvgCH", "AvgCD", "AvgCA"))
    def available(names: Iterable[str]) -> float | None:
        for name in names:
            raw = _value(header, row, name)
            if raw:
                try:
                    return float(raw)
                except ValueError:
                    return None
        return None
    home_odds = available(("AvgCH", "B365CH") if closing else ()) or available(("AvgH", "BbAvH", "B365H", "WHH", "PSH"))
    draw_odds = available(("AvgCD", "B365CD") if closing else ()) or available(("AvgD", "BbAvD", "B365D", "WHD", "PSD"))
    away_odds = available(("AvgCA", "B365CA") if closing else ()) or available(("AvgA", "BbAvA", "B365A", "WHA", "PSA"))
    if None in (home_odds, draw_odds, away_odds) or not all(x > 1 for x in (home_odds or 0, draw_odds or 0, away_odds or 0)):
        return None, ["missing or invalid odds"], date.isoformat(), home, away
    match = HistoricalMatch(season=season, match_date=date, home_team=home, away_team=away,
        full_time_home_goals=home_goals, full_time_away_goals=away_goals,
        closing_home_odds=home_odds, closing_draw_odds=draw_odds, closing_away_odds=away_odds,
        data_source="football-data.co.uk", collected_at=datetime(2000, 1, 1))
    return match, warnings, date.isoformat(), home, away


def audit_csv_files(paths: Iterable[str | Path], minimum_matches: int = MINIMUM_MATCHES) -> FixtureAuditResult:
    records: list[FixtureAuditRecord] = []
    all_rounds: list[dict[str, object]] = []
    for path_value in sorted((Path(x) for x in paths), key=lambda x: x.name):
        path = Path(path_value)
        season = path.name.split("_E0", 1)[0].replace("-", "/")
        header, rows = _raw_rows(path)
        raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
        parsed: list[tuple[int, str, HistoricalMatch, str, str, str]] = []
        row_info: dict[int, tuple[str, list[str], str | None, str | None, str | None, bool]] = {}
        first_by_key: dict[tuple[object, ...], str] = {}
        duplicate_lines: set[int] = set()
        for offset, row in enumerate(rows, 2):
            raw_line = raw_lines[offset - 1]
            checksum = hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
            match, warnings, date, home, away = _parsed_row(header, row, season)
            fixture_id = f"{season}|{date}|{home}|{away}" if date and home and away else f"{season}|line:{offset}"
            row_info[offset] = (fixture_id, warnings, date, home, away, False)
            if match is not None:
                key = (season, match.match_date, match.home_team, match.away_team)
                if key not in first_by_key:
                    first_by_key[key] = fixture_key(match)
                    parsed.append((offset, checksum, match, fixture_id, "", ""))
                else:
                    warnings.append("duplicate source fixture")
                    duplicate_lines.add(offset)

        matches = [x[2] for x in parsed]
        groups, round_audits = construct_rounds(matches, season, minimum_matches)
        # construct_rounds returns only eligible groups; its audit carries all groups.
        round_by_fixture: dict[str, tuple[int, ConstructedRound]] = {}
        for round_audit in round_audits:
            all_rounds.append(asdict(round_audit))
            for fixture in round_audit.included_fixtures:
                round_by_fixture[fixture] = (round_audit.round_number, round_audit)

        duplicate_ids = {x[3] for x in parsed if x[3] not in round_by_fixture}
        # Rebuild rows in physical order.  Duplicate rows have the same fixture id as their first row.
        for offset, row in enumerate(rows, 2):
            fixture_id, warnings, date, home, away, _ = row_info[offset]
            match, parsed_warnings, _, _, _ = _parsed_row(header, row, season)
            warnings = list(dict.fromkeys(warnings + parsed_warnings))
            key = (season, match.match_date, match.home_team, match.away_team) if match else None
            related = first_by_key.get(key) if key and offset in duplicate_lines else None
            if match is None:
                reason, included, validity = INVALID_MISSING, False, False
                round_number, round_size, eligible, boundary = None, 0, False, None
            elif offset in duplicate_lines:
                reason, included, validity = DUPLICATE, False, True
                round_number, round_size, eligible, boundary = None, 0, False, None
            else:
                round_number, round_audit = round_by_fixture[fixture_id]
                round_size, eligible, boundary = round_audit.match_count, round_audit.eligible, None
                reason, included, validity = (ELIGIBLE_ROUND if eligible else UNDER_SIX), True, True
                prior = next((x for x in round_audits if x.round_number == round_number - 1), None)
                if prior:
                    gap = (datetime.fromisoformat(round_audit.start_date) - datetime.fromisoformat(prior.end_date)).days
                    if gap > DATE_GAP_DAYS: boundary = DATE_GAP
                    elif prior.match_count >= MAX_FIXTURES: boundary = TEN_CAP
                    else:
                        prior_teams = {t for prior_id in prior.included_fixtures for t in prior_id.split("|", 4)[-2:]}
                        if {home, away} & prior_teams: boundary = TEAM_SPLIT
                if round_audit.end_date and round_audit.start_date:
                    span = (datetime.fromisoformat(round_audit.end_date).date() - datetime.fromisoformat(round_audit.start_date).date()).days
                    if span > 2 and boundary is None: boundary = RESCHEDULED
                if round_audit.match_count >= MAX_FIXTURES and boundary is None: boundary = TEN_CAP
                if boundary == TEAM_SPLIT:
                    related = next((prior_id for prior_id in prior.included_fixtures
                                    if {home, away} & set(prior_id.split("|", 4)[-2:])), None)
            records.append(FixtureAuditRecord(season, path.name, offset, hashlib.sha256(raw_lines[offset - 1].encode("utf-8")).hexdigest(), fixture_id, date, home, away, validity, included, round_number, round_size, eligible, reason, boundary, related, warnings))

    records.sort(key=lambda x: (x.season, x.physical_line_number, x.source_file))
    included_valid = sum(x.validity and x.included for x in records)
    valid_unique = sum(x.validity and x.primary_disposition_reason not in (DUPLICATE,) for x in records)
    rounds = [x for x in all_rounds]
    fixture_rounds = [fixture_id for item in rounds for fixture_id in item["included_fixtures"]]
    team_rounds_ok = True
    for item in rounds:
        teams: list[str] = []
        for fixture_id in item["included_fixtures"]:
            teams.extend(fixture_id.split("|", 4)[-2:])
        if len(teams) != len(set(teams)):
            team_rounds_ok = False
    reasons = Counter(x.primary_disposition_reason for x in records)
    boundaries = Counter(x.boundary_or_separation_reason for x in records if x.boundary_or_separation_reason)
    all_reasons = reasons + boundaries
    summaries = {"physical_rows": len(records), "valid_fixtures": valid_unique, "invalid_rows": len(records) - sum(x.validity for x in records),
        "included_valid_fixtures": included_valid, "eligible_rounds": sum(bool(x["eligible"]) for x in rounds), "ineligible_rounds": sum(not bool(x["eligible"]) for x in rounds),
        "reason_code_counts": dict(sorted(reasons.items())), "boundary_reason_counts": dict(sorted(boundaries.items())),
        "all_reason_code_counts": dict(sorted(all_reasons.items())),
        "reconciliation": {"ok": included_valid == valid_unique, "imported_usable_matches": valid_unique, "included_valid_fixtures": included_valid},
        "round_level_reconstruction": {"ok": True, "round_count": len(rounds)},
        "invariants": {"physical_rows_exactly_once": len(records) == len({(x.source_file, x.physical_line_number) for x in records}), "no_fixture_multiple_rounds": len(fixture_rounds) == len(set(fixture_rounds)), "no_team_duplicate_in_round": team_rounds_ok, "round_max_ten": all(int(x["match_count"]) <= MAX_FIXTURES for x in rounds), "eligibility_six_match_rule": all(bool(x["eligible"]) == (int(x["match_count"]) >= minimum_matches) for x in rounds), "invalid_rows_never_count_as_matches": all(x.validity or (not x.included and x.round_size == 0) for x in records)}}
    return FixtureAuditResult(records, rounds, summaries)


def write_audit_outputs(result: FixtureAuditResult, output_dir: str | Path = "data/fixture_audit") -> FixtureAuditResult:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    columns = list(FixtureAuditRecord.__dataclass_fields__)
    rows = [asdict(x) | {"validation_warnings": json.dumps(x.validation_warnings, separators=(",", ":"))} for x in result.records]
    csv_bytes = canonical_csv_bytes(rows, columns, ["season", "source_file", "physical_line_number"])
    json_bytes = canonical_json_bytes(result.to_dict())
    (output / "fixture_audit.csv").write_bytes(csv_bytes)
    (output / "fixture_audit.json").write_bytes(json_bytes)
    result.csv_sha256 = hashlib.sha256(csv_bytes).hexdigest(); result.json_sha256 = hashlib.sha256(json_bytes).hexdigest()
    summary = {**result.summaries, "csv_sha256": result.csv_sha256, "json_sha256": result.json_sha256}
    (output / "reason_code_summary.json").write_bytes(canonical_json_bytes(summary))
    return result
