import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import numpy as np

from kernel_calibration.plots import plot_local_calibration_bias, reliability_diagram


def _data(n=100, seed=0):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.uniform(size=n), 0.02, 0.98)
    y = (rng.uniform(size=n) < p).astype(float)
    return y, p


def test_reliability_diagram_returns_axes():
    y, p = _data()
    ax = reliability_diagram(y, p, n_bins=10)
    assert ax is not None
    assert len(ax.lines) >= 2  # diagonal + model curve


def test_plot_local_calibration_bias_returns_axes():
    rng = np.random.default_rng(1)
    feature = rng.normal(size=50)
    bias = rng.normal(size=50) * 0.1
    ax = plot_local_calibration_bias(feature, bias, xlabel="age")
    assert ax is not None
    assert ax.get_xlabel() == "age"
