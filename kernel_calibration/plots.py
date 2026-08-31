"""Plotting helpers (optional).

These require matplotlib, which is an optional dependency. Install it with::

    pip install "kernel_calibration[viz]"

The helpers return the matplotlib ``Axes`` so callers can further customise them.
"""

from typing import Optional

import numpy as np

from .metrics import _bin_stats

__all__ = ["plot_local_calibration_bias", "reliability_diagram"]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            'Plotting requires matplotlib. Install it with `pip install "kernel_calibration[viz]"`.'
        ) from exc
    return plt


def reliability_diagram(y, p, n_bins: int = 15, ax=None, label: Optional[str] = None):
    """Plot a reliability diagram (accuracy vs. confidence per bin).

    Parameters
    ----------
    y : array_like
        Binary labels of shape (n_samples,).
    p : array_like
        Predicted probabilities of shape (n_samples,).
    n_bins : int, optional
        Number of equal-width bins in [0, 1]. Default is 15.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if omitted.
    label : str, optional
        Legend label for the model curve.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    plt = _require_matplotlib()
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    confidences, accuracies = [], []
    for _, acc, conf in _bin_stats(y, p, n_bins):
        confidences.append(conf)
        accuracies.append(acc)

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfectly calibrated")
    ax.plot(confidences, accuracies, marker="o", label=label or "model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability diagram")
    ax.legend(loc="best")
    return ax


def plot_local_calibration_bias(feature, bias, ax=None, xlabel: str = "feature"):
    """Scatter the local calibration bias against a single feature.

    Useful for the "where is the model miscalibrated" view: points far from zero
    mark regions of feature space where predictions are over- or under-confident.

    Parameters
    ----------
    feature : array_like
        A 1-D feature value per sample, shape (n_samples,).
    bias : array_like
        Local calibration bias per sample (e.g. the output of
        :func:`~kernel_calibration.lcb.local_calibration_bias`), shape (n_samples,).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if omitted.
    xlabel : str, optional
        Label for the feature axis. Default is "feature".

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    plt = _require_matplotlib()
    feature = np.asarray(feature, dtype=float)
    bias = np.asarray(bias, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.5))

    order = np.argsort(feature)
    ax.axhline(0.0, linestyle="--", color="grey")
    ax.scatter(feature, bias, s=14, alpha=0.6)
    ax.plot(feature[order], bias[order], color="tab:blue", alpha=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Local calibration bias")
    ax.set_title("Local calibration bias vs. " + xlabel)
    return ax
