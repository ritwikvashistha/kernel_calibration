import numpy as np
import pytest

from kernel_calibration import (
    brier_score,
    expected_calibration_error,
    kernel_calibration_error,
    maximum_calibration_error,
)


def test_brier_score_perfect_is_zero():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y, y) == pytest.approx(0.0)


def test_brier_score_half():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.full(4, 0.5)
    assert brier_score(y, p) == pytest.approx(0.25)


def test_ece_perfect_predictions_zero():
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    p = y.copy()  # predictions exactly 0/1 matching labels
    assert expected_calibration_error(y, p) == pytest.approx(0.0)


def test_mce_at_least_ece():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=200) > 0.5).astype(float)
    p = np.clip(rng.uniform(size=200), 0.02, 0.98)
    ece = expected_calibration_error(y, p)
    mce = maximum_calibration_error(y, p)
    assert mce >= ece - 1e-9


def test_kce_zero_residual_is_zero():
    rng = np.random.default_rng(1)
    p = np.clip(rng.uniform(size=20), 0.05, 0.95)
    y = p.copy()  # zero residuals
    assert kernel_calibration_error(y, p, 0.1) == pytest.approx(0.0, abs=1e-8)


def test_kce_finite():
    rng = np.random.default_rng(2)
    p = np.clip(rng.uniform(size=30), 0.05, 0.95)
    y = (rng.uniform(size=30) > 0.5).astype(float)
    val = kernel_calibration_error(y, p, 0.1)
    assert np.isfinite(val)
