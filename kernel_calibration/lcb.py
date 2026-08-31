"""Local Calibration Bias (LCB) — the "error-witness" diagnostic.

Where :func:`~kernel_calibration.kite.KLCE_test` answers *whether* a model is
locally calibrated, the LCB diagnostic answers *where* it is miscalibrated. It is
the kernel-weighted average of residuals ``y - f`` around a query point in the
joint (prediction, feature) space, i.e. an estimate of the local calibration bias

    LCB(x', a) = P(y = 1 | X = x', f(z) = a) - a,

the signed gap between the true class frequency and the predicted probability in
the neighbourhood of ``(x', a)``. Positive values mean the model is under-confident
there (outcomes happen more often than predicted); negative values mean it is
over-confident.

This implements Equation (7) / Proposition 3.8 of Vashistha & Farahi,
*I-trustworthy Models* (AISTATS 2025, arXiv:2501.15617).
"""

from typing import Optional

import jax.numpy as jnp

from .kite import rbf_kernel

__all__ = ["local_calibration_bias"]


def local_calibration_bias(
    X: jnp.ndarray,
    y: jnp.ndarray,
    f: jnp.ndarray,
    prob_kernel_width: float,
    x_kernel_width: Optional[float] = None,
    *,
    X_query: Optional[jnp.ndarray] = None,
    f_query: Optional[jnp.ndarray] = None,
    leave_one_out: bool = False,
) -> jnp.ndarray:
    """Estimate the local calibration bias at each query point.

    For a query point ``(x', a')`` the estimator is the Nadaraya–Watson-style
    kernel-weighted average of residuals (Eq. 7 of the paper)::

        LCB(x', a') = sum_i (y_i - f_i) k(f_i, a') l(x_i, x')
                      -------------------------------------------
                             sum_i k(f_i, a') l(x_i, x')

    with ``k`` an RBF kernel on predicted probabilities and ``l`` an RBF kernel on
    features. Setting the feature kernel aside (a single feature value) this reduces
    to the usual reliability-curve bias; with the feature kernel it localises the
    bias in feature space, revealing e.g. subgroups where the model is miscalibrated.

    Parameters
    ----------
    X : array_like
        Reference feature matrix of shape (n_samples, n_features). NumPy arrays are
        accepted.
    y : array_like
        Reference labels of shape (n_samples,).
    f : array_like
        Reference predicted probabilities of shape (n_samples,).
    prob_kernel_width : float
        Bandwidth for the probability kernel ``k``.
    x_kernel_width : float, optional
        Bandwidth for the feature kernel ``l``. If omitted, ``prob_kernel_width``
        is used for both kernels.
    X_query : array_like, optional
        Feature points at which to evaluate the bias, shape (m_samples, n_features).
        Defaults to ``X`` (evaluate at every reference point).
    f_query : array_like, optional
        Predicted probabilities at the query points, shape (m_samples,). Required
        if ``X_query`` is given; defaults to ``f`` when evaluating at ``X``.
    leave_one_out : bool, optional
        Only used when evaluating at the reference points themselves (no explicit
        ``X_query``). If True, each point's own residual is excluded from its
        estimate, giving an honest (leave-one-out) bias. Default is False.

    Returns
    -------
    jnp.ndarray
        Estimated local calibration bias at each query point, shape (m_samples,).

    Examples
    --------
    >>> bias = local_calibration_bias(X, y, f, prob_kernel_width=0.1)
    >>> # bias[i] > 0  -> model under-confident near sample i
    >>> # bias[i] < 0  -> model over-confident near sample i
    """
    if x_kernel_width is None:
        x_kernel_width = prob_kernel_width

    X = jnp.asarray(X)
    y = jnp.asarray(y)
    f = jnp.asarray(f)
    err = y - f

    evaluating_on_self = X_query is None and f_query is None
    if X_query is None:
        X_query = X
    if f_query is None:
        if not evaluating_on_self:
            raise ValueError("f_query must be provided when X_query is given.")
        f_query = f
    X_query = jnp.asarray(X_query)
    f_query = jnp.asarray(f_query)
    if X_query.shape[0] != f_query.shape[0]:
        raise ValueError(
            f"X_query and f_query must have the same number of samples. "
            f"Got {X_query.shape[0]} and {f_query.shape[0]}."
        )

    gamma_p = 1.0 / (prob_kernel_width**2)
    gamma_x = 1.0 / (x_kernel_width**2)

    # Rectangular kernels between the n reference points and m query points.
    K_p = rbf_kernel(f.reshape(-1, 1), f_query.reshape(-1, 1), gamma_p)  # (n, m)
    K_x = rbf_kernel(X, X_query, gamma_x)  # (n, m)
    weights = K_p * K_x  # (n, m)

    if leave_one_out and evaluating_on_self:
        weights = weights * (1.0 - jnp.eye(weights.shape[0]))

    numerator = weights.T @ err  # (m,)
    denominator = jnp.sum(weights, axis=0)  # (m,)
    return numerator / denominator
