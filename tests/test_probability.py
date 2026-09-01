import numpy as np
import pytest
from lms_optimizer.probability import additive, power_method, proportional, shin

@pytest.mark.parametrize("method", [proportional, additive, power_method, shin])
def test_margin_methods_return_probability_vector(method):
    result = method([2.0, 3.5, 4.0])
    assert np.all(result >= 0)
    assert sum(result) == pytest.approx(1.0, abs=1e-7)

def test_invalid_odds_rejected():
    for odds in ([1.0, 2.0, 3.0], [float("nan"), 2.0, 3.0], [float("inf"), 2.0, 3.0]):
        with pytest.raises(ValueError):
            shin(odds)


@pytest.mark.parametrize("odds", [
    [2.10, 3.40, 3.60],  # normal football market
    [2.00, 3.00, 5.90],  # near-fair market
    [1.50, 3.00, 4.00],  # relatively high overround
])
def test_shin_is_finite_non_negative_and_normalized(odds):
    result = shin(odds)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)
    assert np.sum(result) == pytest.approx(1.0, abs=1e-10)
