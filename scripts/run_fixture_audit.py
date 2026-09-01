"""Build the canonical fixture-level audit for the local Football-Data archive."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lms_optimizer.fixture_audit import audit_csv_files, write_audit_outputs


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "data/raw/football-data").glob("*_E0.csv"))
    if len(paths) != 16:
        raise SystemExit(f"expected 16 Football-Data seasons, found {len(paths)}")
    result = write_audit_outputs(audit_csv_files(paths), root / "data/fixture_audit")
    print(result.summaries)
    print({"csv_sha256": result.csv_sha256, "json_sha256": result.json_sha256})


if __name__ == "__main__":
    main()
