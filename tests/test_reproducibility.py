import pytest
from lms_optimizer.reproducibility import canonical_value, canonical_json_bytes, canonical_csv_bytes, first_difference

def test_volatile_fields_removed_but_cutoff_retained():
    value = {"runtime_seconds": 1, "generated_at": "now", "cutoff": "2020-01-01", "x": {"pid": 2, "v": 1.2345678901234}}
    result = canonical_value(value)
    assert "runtime_seconds" not in result and "generated_at" not in result and result["cutoff"] == "2020-01-01"
    assert result["x"]["v"] == "1.23456789012"

def test_nested_dict_and_record_lists_are_stable():
    one = {"decisions": [{"season": "b", "entry": "e2"}, {"season": "a", "entry": "e1"}], "z": {"b": 2, "a": 1}}
    two = {"z": {"a": 1, "b": 2}, "decisions": [{"entry": "e1", "season": "a"}, {"entry": "e2", "season": "b"}]}
    assert canonical_json_bytes(one) == canonical_json_bytes(two)

def test_csv_has_explicit_order_and_unix_newlines():
    data = canonical_csv_bytes([{"id": 2, "value": 1.2}, {"id": 1, "value": None}], ["id", "value"], ["id"])
    assert data == b"id,value\n1,\n2,1.2\n"

def test_first_difference_is_actionable():
    result = first_difference({"rows": [{"id": "a", "p": 1}]}, {"rows": [{"id": "a", "p": 2}]})
    assert result == {"path": "$.rows[0].p", "value_one": 1, "value_two": 2}

def test_non_finite_values_rejected():
    with pytest.raises(ValueError):
        canonical_json_bytes({"p": float("nan")})
