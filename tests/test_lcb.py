import jax.numpy as jnp
import numpy as np
import pytest

from kernel_calibration import local_calibration_bias


def _data(n=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    f = np.clip(rng.uniform(size=n), 0.05, 0.95)
    y = (rng.uniform(size=n) > 0.5).astype(float)
    return X, y, f


def test_lcb_self_shape():
    X, y, f = _data()
    bias = local_calibration_bias(X, y, f, prob_kernel_width=0.2)
    assert bias.shape == (X.shape[0],)
    assert jnp.all(jnp.isfinite(bias))


def test_lcb_constant_residual_recovers_constant():
    # If y - f is a constant c everywhere, the kernel-weighted average is c,
    # regardless of the kernel weights.
    X, _, _ = _data()
    f = jnp.full((X.shape[0],), 0.3)
    y = f + 0.2
    bias = local_calibration_bias(X, y, f, prob_kernel_width=0.2)
    assert jnp.allclose(bias, 0.2, atol=1e-5)


def test_lcb_zero_residual_is_zero():
    X, _, f = _data()
    y = jnp.asarray(f)  # residual exactly zero
    bias = local_calibration_bias(X, y, f, prob_kernel_width=0.2)
    assert jnp.allclose(bias, 0.0, atol=1e-6)


def test_lcb_query_points_shape():
    X, y, f = _data()
    Xq = np.zeros((5, 2))
    fq = np.linspace(0.1, 0.9, 5)
    bias = local_calibration_bias(X, y, f, prob_kernel_width=0.2, X_query=Xq, f_query=fq)
    assert bias.shape == (5,)


def test_lcb_query_requires_f_query():
    X, y, f = _data()
    with pytest.raises(ValueError, match="f_query"):
        local_calibration_bias(X, y, f, prob_kernel_width=0.2, X_query=np.zeros((3, 2)))


def test_lcb_leave_one_out_runs():
    X, y, f = _data()
    bias = local_calibration_bias(X, y, f, prob_kernel_width=0.2, leave_one_out=True)
    assert bias.shape == (X.shape[0],)
    assert jnp.all(jnp.isfinite(bias))
