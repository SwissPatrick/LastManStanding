"""Audited CSV parsing for organiser survivor lists.

Parsing is deliberately side-effect free.  Applying a preview lives in the
workflow/repository transaction so an upload can always be reviewed first.
"""
from __future__ import annotations

import csv
import difflib
import hashlib
import io
import re
from dataclasses import dataclass, field

from .providers import TEAM_ALIASES

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 5_000
ROUND_RE = re.compile(r"^round\s+(\d+)$", re.I)

# Provider spellings plus aliases.  This is intentionally a closed set: a
# typo must be fixed by the user rather than quietly becoming a new team.
CANONICAL_TEAMS = {
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Leeds United", "Leicester City", "Liverpool", "Manchester City",
    "Manchester United", "Newcastle", "Nottingham Forest", "Sunderland",
    "Tottenham", "West Ham", "Wolverhampton Wanderers", "Sheffield United",
    "Ipswich Town", "Southampton", "Luton Town", "Watford", "Norwich City",
    "West Bromwich Albion", "Stoke City", "Hull City", "Cardiff City",
    "Swansea City", "Blackburn Rovers", "Blackpool", "Queens Park Rangers",
    "Reading", "Wigan Athletic", "Bolton Wanderers", "Middlesbrough",
    "Portsmouth", "Birmingham City", "Charlton Athletic", "Derby County",
    "Barnsley", "Huddersfield Town",
}
_TEAM_INDEX = {" ".join(team.lower().split()): team for team in CANONICAL_TEAMS}
_TEAM_INDEX.update(TEAM_ALIASES)
_TEAM_INDEX.update({"man united": "Manchester United", "tottenham hotspur": "Tottenham"})


def normalise_label(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def canonical_team(value: str) -> str | None:
    return _TEAM_INDEX.get(" ".join(value.strip().casefold().split()))


@dataclass(frozen=True)
class ImportIssue:
    severity: str
    message: str
    row: int | None = None
    column: str | None = None


@dataclass(frozen=True)
class CompetitionRow:
    row_number: int
    label: str
    normalised_label: str
    picks: dict[int, str]


@dataclass
class ParsedCompetitionCSV:
    checksum: str
    rows: list[CompetitionRow] = field(default_factory=list)
    issues: list[ImportIssue] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[ImportIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]


def csv_template() -> bytes:
    return ("Entry,Round 1,Round 2,Round 3\n"
            "Alex Example 1,Everton,Manchester City,Liverpool\n"
            "Alex Example 2,Arsenal,Manchester United,\n"
            "Morgan Example,Chelsea,Sunderland,Manchester City\n").encode("utf-8")


def parse_competition_csv(raw: bytes, reporting_round: int) -> ParsedCompetitionCSV:
    result = ParsedCompetitionCSV(checksum=hashlib.sha256(raw).hexdigest())
    if len(raw) > MAX_BYTES:
        result.issues.append(ImportIssue("error", f"File is larger than the {MAX_BYTES // 1024 // 1024} MB limit."))
        return result
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        result.issues.append(ImportIssue("error", "The file must be UTF-8 CSV (Excel UTF-8 is supported)."))
        return result
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;")
    except csv.Error:
        dialect = csv.excel
    try:
        table = list(csv.reader(io.StringIO(text), dialect, strict=True))
    except csv.Error as exc:
        result.issues.append(ImportIssue("error", f"Malformed CSV: {exc}."))
        return result
    while table and not any(cell.strip() for cell in table[-1]):
        table.pop()
    if not table:
        result.issues.append(ImportIssue("error", "The CSV is empty."))
        return result
    if len(table) - 1 > MAX_ROWS:
        result.issues.append(ImportIssue("error", f"The CSV has more than {MAX_ROWS} rows."))
        return result
    headers = [cell.strip() for cell in table[0]]
    result.headers = headers
    seen_headers: set[str] = set()
    rounds: dict[int, int] = {}
    entry_column: int | None = None
    for index, header in enumerate(headers):
        key = normalise_label(header)
        if not header:
            result.issues.append(ImportIssue("error", "Header is blank.", 1, f"column {index + 1}")); continue
        if key in seen_headers:
            result.issues.append(ImportIssue("error", "Header is duplicated.", 1, header))
        seen_headers.add(key)
        if key == "entry":
            if entry_column is not None: result.issues.append(ImportIssue("error", "Entry header is duplicated.", 1, header))
            entry_column = index
        else:
            match = ROUND_RE.match(header)
            if not match:
                result.issues.append(ImportIssue("error", "Columns must be Entry or Round N.", 1, header)); continue
            number = int(match.group(1))
            if number in rounds: result.issues.append(ImportIssue("error", "Round column is duplicated.", 1, header))
            rounds[number] = index
    if entry_column != 0:
        result.issues.append(ImportIssue("error", "Entry must be the first column.", 1, "Entry"))
    labels: set[str] = set()
    for line_number, row in enumerate(table[1:], 2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != len(headers):
            result.issues.append(ImportIssue("error", "Row has a different number of columns from the header.", line_number)); continue
        label = row[0].strip()
        normalised = normalise_label(label)
        if not label:
            result.issues.append(ImportIssue("error", "Entry label is blank.", line_number, "Entry")); continue
        if normalised in labels:
            result.issues.append(ImportIssue("error", "Entry label is duplicated (matching is case-insensitive).", line_number, "Entry")); continue
        labels.add(normalised)
        picks: dict[int, str] = {}
        for number, column in rounds.items():
            value = row[column].strip()
            if not value: continue
            if number > reporting_round:
                result.issues.append(ImportIssue("error", "Pick is beyond the selected reporting round.", line_number, headers[column])); continue
            team = canonical_team(value)
            if team is None:
                result.issues.append(ImportIssue("error", f"Unknown or ambiguous team '{value}'.", line_number, headers[column])); continue
            picks[number] = team
        used: dict[str, int] = {}
        for number, team in picks.items():
            if team in used:
                result.issues.append(ImportIssue("error", f"{team} is reused from Round {used[team]}.", line_number, f"Round {number}"))
            used[team] = number
        if picks:
            last = max(picks)
            gaps = [number for number in range(1, last) if number in rounds and number not in picks]
            if gaps:
                result.issues.append(ImportIssue("warning", f"Missing historical pick(s): {', '.join('Round ' + str(x) for x in gaps)}. Blanks will not erase history.", line_number))
        result.rows.append(CompetitionRow(line_number, label, normalised, picks))
    return result


def suspected_rename(new_label: str, existing_labels: list[str]) -> str | None:
    """Return a review-only hint; never use fuzzy matching for identity."""
    for existing in existing_labels:
        score = difflib.SequenceMatcher(None, new_label, existing).ratio()
        if score >= 0.88:
            return existing
    return None
