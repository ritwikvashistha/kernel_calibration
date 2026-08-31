"""Kernel bandwidth selection.

The KLCE test and the LCB diagnostic both depend on RBF kernel bandwidths (the
``prob_kernel_width`` / ``x_kernel_width`` arguments). Power and localisation are
sensitive to these, and guessing them is the main usability pitfall. This module
provides the **median heuristic** — a robust, parameter-free default that sets the
bandwidth to the median pairwise Euclidean distance of the data (Gretton et al.,
2012). It is a solid starting point; the paper also discusses a greedy
Type-II-error-minimising search, which is left for a future release.
"""

import numpy as np

__all__ = ["median_heuristic", "select_bandwidths"]


def median_heuristic(
    values,
    max_samples: int = 2000,
    seed: int = 0,
) -> float:
    """Median-heuristic bandwidth: the median pairwise Euclidean distance.

    Parameters
    ----------
    values : array_like
        Data of shape (n_samples, n_features) or (n_samples,) for a single feature
        (e.g. predicted probabilities).
    max_samples : int, optional
        If ``n_samples`` exceeds this, a random subsample of this size is used to
        keep the pairwise computation O(max_samples^2). Default is 2000.
    seed : int, optional
        Seed for the subsampling RNG (only used when subsampling). Default is 0.

    Returns
    -------
    float
        The median non-zero pairwise distance, usable directly as a kernel width.
        Falls back to ``1.0`` if all points coincide (median distance is zero).
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n = values.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        values = values[idx]
        n = max_samples

    # Pairwise squared distances, then the upper triangle (i < j).
    sq = (
        np.sum(values**2, axis=1)[:, None]
        + np.sum(values**2, axis=1)[None, :]
        - 2.0 * values @ values.T
    )
    sq = np.clip(sq, 0.0, None)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(sq[iu])
    dists = dists[dists > 0]
    if dists.size == 0:
        return 1.0
    return float(np.median(dists))


def select_bandwidths(
    X,
    f,
    max_samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Median-heuristic bandwidths for the probability and feature kernels.

    Convenience wrapper that returns sensible defaults for both kernels used by
    :func:`~kernel_calibration.kite.KLCE_test` and
    :func:`~kernel_calibration.lcb.local_calibration_bias`.

    Parameters
    ----------
    X : array_like
        Feature matrix of shape (n_samples, n_features) or (n_samples,).
    f : array_like
        Predicted probabilities of shape (n_samples,).
    max_samples, seed
        Passed through to :func:`median_heuristic`.

    Returns
    -------
    Tuple[float, float]
        ``(prob_kernel_width, x_kernel_width)``.

    Examples
    --------
    >>> pw, xw = select_bandwidths(X, f)
    >>> stat, pval = KLCE_test(X, y, f, prob_kernel_width=pw, iterations=200,
    ...                        key=0, x_kernel_width=xw)
    """
    prob_width = median_heuristic(f, max_samples=max_samples, seed=seed)
    x_width = median_heuristic(X, max_samples=max_samples, seed=seed)
    return prob_width, x_width
