import csv
import hashlib
from pathlib import Path

from lms_optimizer.fixture_audit import (
    DATE_GAP, DUPLICATE, ELIGIBLE_ROUND, INVALID_MISSING, TEAM_SPLIT,
    TEN_CAP, UNDER_SIX, audit_csv_files, write_audit_outputs,
)


def write_source(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "AvgH", "AvgD", "AvgA"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture(date: str, home: str, away: str) -> dict[str, object]:
    return {"Date": date, "HomeTeam": home, "AwayTeam": away, "FTHG": 1, "FTAG": 0, "AvgH": 2, "AvgD": 3.5, "AvgA": 4}


def test_fixture_audit_keeps_blank_and_duplicate_rows_with_stable_hash(tmp_path):
    source = tmp_path / "2014-15_E0.csv"
    rows = [fixture("01/08/14", "A", "B"), fixture("01/08/14", "A", "B"), fixture("01/08/14", "C", "D")]
    write_source(source, rows)
    with source.open("a", encoding="utf-8", newline="") as stream:
        stream.write(",,,,,,,\n")
    result = audit_csv_files([source])
    assert len(result.records) == 4
    assert result.records[-1].physical_line_number == 5
    assert result.records[-1].primary_disposition_reason == INVALID_MISSING
    duplicate = next(x for x in result.records if x.primary_disposition_reason == DUPLICATE)
    assert duplicate.related_conflicting_fixture == result.records[0].fixture_id
    first = write_audit_outputs(audit_csv_files([source]), tmp_path / "one")
    second = write_audit_outputs(audit_csv_files([source]), tmp_path / "two")
    assert first.csv_sha256 == second.csv_sha256
    assert first.json_sha256 == second.json_sha256


def test_round_eligibility_and_under_six_inclusion(tmp_path):
    source = tmp_path / "2020-21_E0.csv"
    rows = [fixture("01/08/20", f"H{i}", f"A{i}") for i in range(6)]
    rows += [fixture("20/08/20", "H6", "A6")]
    write_source(source, rows)
    result = audit_csv_files([source])
    assert result.summaries["eligible_rounds"] == 1
    assert result.summaries["ineligible_rounds"] == 1
    assert sum(x.primary_disposition_reason == ELIGIBLE_ROUND for x in result.records) == 6
    assert sum(x.primary_disposition_reason == UNDER_SIX for x in result.records) == 1


def test_date_gap_team_split_and_ten_fixture_cap_are_audited(tmp_path):
    source = tmp_path / "2021-22_E0.csv"
    rows = [fixture("01/08/21", f"H{i}", f"A{i}") for i in range(10)]
    rows += [fixture("01/08/21", "H10", "A10"), fixture("10/08/21", "H11", "A11")]
    write_source(source, rows)
    result = audit_csv_files([source])
    assert any(x.boundary_or_separation_reason == TEN_CAP for x in result.records)
    assert any(x.boundary_or_separation_reason == DATE_GAP for x in result.records)

    conflict = tmp_path / "2022-23_E0.csv"
    write_source(conflict, [fixture("01/08/22", "A", "B"), fixture("01/08/22", "A", "C")])
    conflict_result = audit_csv_files([conflict])
    assert conflict_result.records[1].boundary_or_separation_reason == TEAM_SPLIT


def test_reconciliation_and_raw_row_checksums(tmp_path):
    source = tmp_path / "2023-24_E0.csv"
    write_source(source, [fixture("01/08/23", "A", "B")])
    result = audit_csv_files([source])
    raw = source.read_text(encoding="utf-8").splitlines()[1].encode("utf-8")
    assert result.summaries["reconciliation"]["ok"] is True
    assert result.records[0].raw_row_checksum == hashlib.sha256(raw).hexdigest()
    assert result.summaries["invariants"]["physical_rows_exactly_once"] is True
