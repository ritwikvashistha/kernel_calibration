"""kernel_calibration (KiTE) — kernel local calibration testing and recalibration.

Implements the Kernel Local Calibration Error (KLCE) from
Vashistha & Farahi, *I-trustworthy Models* (AISTATS 2025, arXiv:2501.15617):

- ``KLCE_test`` — a hypothesis test for whether a binary classifier is locally
  calibrated (the null is local calibration; a small p-value rejects it).
- ``local_calibration_bias`` — the "error-witness" diagnostic that localises
  *where* in feature space the model is miscalibrated.
- ``recalibrated_model`` — a kernel-penalised recalibrator that fixes it.
- ``select_bandwidths`` / ``median_heuristic`` — default kernel bandwidths.
- ``brier_score`` / ``expected_calibration_error`` / ``maximum_calibration_error``
  / ``kernel_calibration_error`` — baseline calibration metrics for comparison.

Plotting helpers live in ``kernel_calibration.plots`` (optional; needs matplotlib).
"""

from importlib.metadata import PackageNotFoundError, version

from .bandwidth import median_heuristic, select_bandwidths
from .datasets import make_calibration_data
from .kite import (
    KLCE2_boosting,
    KLCE2_estimator,
    KLCE_test,
    KLCETestResult,
    compute_null_distribution,
    create_kernel,
    rbf_kernel,
    recalibrated_model,
)
from .lcb import local_calibration_bias
from .metrics import (
    brier_score,
    expected_calibration_error,
    kernel_calibration_error,
    maximum_calibration_error,
)

try:
    __version__ = version("kernel_calibration")
except PackageNotFoundError:  # package is not installed (e.g. run from source tree)
    __version__ = "0.0.0+unknown"

__all__ = [
    # test + statistic
    "KLCE_test",
    "KLCETestResult",
    "KLCE2_estimator",
    "KLCE2_boosting",
    "compute_null_distribution",
    "create_kernel",
    "rbf_kernel",
    # diagnostic
    "local_calibration_bias",
    # recalibration
    "recalibrated_model",
    # bandwidth selection
    "median_heuristic",
    "select_bandwidths",
    # baseline metrics
    "brier_score",
    "expected_calibration_error",
    "maximum_calibration_error",
    "kernel_calibration_error",
    # example data
    "make_calibration_data",
    "__version__",
]
