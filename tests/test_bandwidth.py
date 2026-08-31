import numpy as np
import pytest

from kernel_calibration import median_heuristic, select_bandwidths


def test_median_heuristic_known_value():
    # Pairwise distances of {0,1,2,3}: [1,2,3,1,2,1] -> sorted [1,1,1,2,2,3],
    # median = 1.5.
    assert median_heuristic([0, 1, 2, 3]) == pytest.approx(1.5)


def test_median_heuristic_positive():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    w = median_heuristic(X)
    assert w > 0.0


def test_median_heuristic_identical_points_fallback():
    assert median_heuristic(np.ones((10, 2))) == 1.0


def test_median_heuristic_subsampling_runs():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 2))
    w = median_heuristic(X, max_samples=50, seed=3)
    assert w > 0.0


def test_select_bandwidths_returns_two_positive():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 2))
    f = np.clip(rng.uniform(size=40), 0.05, 0.95)
    pw, xw = select_bandwidths(X, f)
    assert pw > 0.0 and xw > 0.0


def test_select_bandwidths_feeds_klce_test():
    from kernel_calibration import KLCE_test

    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 2))
    f = np.clip(rng.uniform(size=40), 0.05, 0.95)
    y = (rng.uniform(size=40) > 0.5).astype(float)
    pw, xw = select_bandwidths(X, f)
    stat, pval = KLCE_test(X, y, f, pw, 100, key=0, x_kernel_width=xw)
    assert 0.0 < float(pval) <= 1.0
