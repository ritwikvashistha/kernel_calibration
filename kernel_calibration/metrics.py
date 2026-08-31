"""Calibration metrics: Brier score, ECE, MCE, and KCE.

These are baseline (global) calibration measures, useful for comparison alongside
the local KLCE statistic in examples and reports. ``ece`` and ``mce`` are the
familiar binning-based measures; ``brier_score`` is a proper scoring rule; and
``kernel_calibration_error`` (KCE) is the kernel calibration error of Widmann et al.
(2019) — precisely KLCE **without** covariate localisation (the feature kernel set
to a constant), which makes concrete the paper's point that KLCE generalises KCE.
"""

import jax.numpy as jnp
import numpy as np

from .kite import KLCE2_estimator, rbf_kernel

__all__ = [
    "brier_score",
    "expected_calibration_error",
    "kernel_calibration_error",
    "maximum_calibration_error",
]


def brier_score(y, p) -> float:
    """Brier score: mean squared error between probabilities and outcomes.

    Parameters
    ----------
    y : array_like
        Binary labels of shape (n_samples,).
    p : array_like
        Predicted probabilities of shape (n_samples,).

    Returns
    -------
    float
        Mean of ``(p - y) ** 2``. Lower is better.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def _bin_stats(y: np.ndarray, p: np.ndarray, n_bins: int):
    """Yield (weight, accuracy, confidence) for each non-empty equal-width bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with the interior edges maps p into bins 0..n_bins-1.
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    n = p.shape[0]
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        yield count / n, float(np.mean(y[mask])), float(np.mean(p[mask]))


def expected_calibration_error(y, p, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) with equal-width bins.

    ECE = sum over bins of ``(bin_weight) * |accuracy - confidence|``.

    Parameters
    ----------
    y : array_like
        Binary labels of shape (n_samples,).
    p : array_like
        Predicted probabilities of shape (n_samples,).
    n_bins : int, optional
        Number of equal-width bins in [0, 1]. Default is 15.

    Returns
    -------
    float
        The ECE. Lower is better; 0 means perfectly calibrated on this binning.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(sum(w * abs(acc - conf) for w, acc, conf in _bin_stats(y, p, n_bins)))


def maximum_calibration_error(y, p, n_bins: int = 15) -> float:
    """Maximum Calibration Error (MCE): the worst per-bin calibration gap.

    Parameters
    ----------
    y, p : array_like
        Labels and predicted probabilities of shape (n_samples,).
    n_bins : int, optional
        Number of equal-width bins in [0, 1]. Default is 15.

    Returns
    -------
    float
        ``max_bin |accuracy - confidence|``; 0.0 if there are no samples.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    gaps = [abs(acc - conf) for _, acc, conf in _bin_stats(y, p, n_bins)]
    return float(max(gaps)) if gaps else 0.0


def kernel_calibration_error(y, p, prob_kernel_width: float) -> float:
    """Squared Kernel Calibration Error (KCE) of Widmann et al. (2019).

    This is the global (non-local) counterpart of KLCE: the same U-statistic on
    residuals ``y - p`` but with only the probability kernel and no feature kernel
    (equivalently, the feature kernel set to a constant). It measures whether the
    model is calibrated on average, without localising in feature space.

    Parameters
    ----------
    y : array_like
        Binary labels of shape (n_samples,).
    p : array_like
        Predicted probabilities of shape (n_samples,).
    prob_kernel_width : float
        Bandwidth for the RBF probability kernel.

    Returns
    -------
    float
        The squared KCE estimate (an unbiased U-statistic, diagonal removed).
    """
    y = jnp.asarray(y)
    p = jnp.asarray(p)
    err = y - p
    gamma_p = 1.0 / (prob_kernel_width**2)
    K = rbf_kernel(p.reshape(-1, 1), p.reshape(-1, 1), gamma_p)
    return float(KLCE2_estimator(K, err))
