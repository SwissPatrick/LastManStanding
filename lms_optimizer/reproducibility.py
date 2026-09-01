"""Canonical analytical artefacts and first-difference diagnostics.

Volatile metadata is deliberately limited to fields that describe execution,
not the information used by a historical decision.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

VOLATILE_FIELDS = frozenset({
    "generated_at", "generation_timestamp", "runtime_seconds", "output_directory",
    "temporary_path", "temp_path", "process_id", "pid",
})
MISSING_VALUE = ""
FLOAT_DIGITS = 12

def _number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("analytical output contains non-finite value")
    return format(value, f".{FLOAT_DIGITS}g")

def _sort_key(item: Any, path: str) -> tuple:
    if not isinstance(item, dict):
        return (repr(item),)
    fields = {
        "decisions": ("season", "start_round", "constructed_round", "strategy", "cartel_size", "entry", "team"),
        "cohort_metrics": ("season", "start_round", "cartel_size", "strategy"),
        "round_construction_audit": ("season", "round_number", "fixture_id", "source_row"),
        "included_fixtures": ("fixture_id",),
    }
    names = fields.get(path.rsplit(".", 1)[-1], ())
    return tuple((str(item.get(name, "")) for name in names)) or (json.dumps(item, sort_keys=True, default=str),)

def canonical_value(value: Any, path: str = "") -> Any:
    if isinstance(value, dict):
        return {key: canonical_value(value[key], f"{path}.{key}".strip("."))
                for key in sorted(value) if key not in VOLATILE_FIELDS}
    if isinstance(value, (list, tuple)):
        values = [canonical_value(item, path) for item in value]
        if values and all(isinstance(item, dict) for item in values):
            values.sort(key=lambda item: _sort_key(item, path))
        return values
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported analytical value at {path}: {type(value).__name__}")

def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(canonical_value(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")

def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()

def canonical_csv_bytes(rows: list[dict[str, Any]], columns: list[str], sort_columns: list[str]) -> bytes:
    ordered = sorted(rows, key=lambda row: tuple(str(row.get(column, MISSING_VALUE)) for column in sort_columns))
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in ordered:
        writer.writerow({column: (MISSING_VALUE if row.get(column) is None else _number(row[column]) if isinstance(row.get(column), float) else row.get(column, MISSING_VALUE)) for column in columns})
    return buffer.getvalue().encode("utf-8")

def write_canonical_csv(path: Path, rows: list[dict[str, Any]], columns: list[str], sort_columns: list[str]) -> str:
    data = canonical_csv_bytes(rows, columns, sort_columns)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()

def first_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"path": path, "value_one": left, "value_two": right}
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return {"path": f"{path}.{key}", "value_one": left.get(key), "value_two": right.get(key)}
            found = first_difference(left[key], right[key], f"{path}.{key}")
            if found: return found
        return None
    if isinstance(left, list):
        for index, (one, two) in enumerate(zip(left, right)):
            found = first_difference(one, two, f"{path}[{index}]")
            if found: return found
        if len(left) != len(right):
            return {"path": path, "value_one": len(left), "value_two": len(right)}
        return None
    if left != right:
        return {"path": path, "value_one": left, "value_two": right}
    return None
