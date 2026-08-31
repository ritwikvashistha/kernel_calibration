import jax.numpy as jnp
import numpy as np
import pytest

from kernel_calibration import create_kernel, rbf_kernel
from kernel_calibration.kite import KLCE2_estimator


def test_rbf_kernel_symmetric_and_unit_diagonal():
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.normal(size=(10, 3)))
    K = rbf_kernel(X, X, 0.5)
    assert jnp.allclose(K, K.T, atol=1e-6)
    assert jnp.allclose(jnp.diag(K), 1.0, atol=1e-6)


def test_rbf_kernel_rectangular_shape():
    rng = np.random.default_rng(1)
    X = jnp.asarray(rng.normal(size=(7, 2)))
    Y = jnp.asarray(rng.normal(size=(4, 2)))
    assert rbf_kernel(X, Y, 1.0).shape == (7, 4)


def test_rbf_kernel_feature_mismatch_raises():
    with pytest.raises(ValueError, match="same number of features"):
        rbf_kernel(jnp.zeros((5, 2)), jnp.zeros((5, 3)), 1.0)


def test_create_kernel_is_psd():
    # A product of two PSD RBF Gram matrices is PSD (Schur product theorem).
    rng = np.random.default_rng(2)
    X = jnp.asarray(rng.normal(size=(15, 2)))
    p = jnp.asarray(np.clip(rng.uniform(size=15), 0.05, 0.95))
    K = create_kernel(X, p, 0.3, 1.0)
    eig = jnp.linalg.eigvalsh((K + K.T) / 2)
    assert float(eig.min()) > -1e-6


def test_create_kernel_validation():
    X = jnp.zeros((5, 2))
    with pytest.raises(ValueError, match="1D array"):
        create_kernel(X, jnp.zeros((5, 1)), 0.2, 1.0)  # p not 1D
    with pytest.raises(ValueError, match="must match"):
        create_kernel(X, jnp.zeros((4,)), 0.2, 1.0)  # sample-count mismatch


def test_klce2_estimator_minimal_and_validation():
    # minimal n = 2 works
    K = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    err = jnp.array([1.0, -1.0])
    assert np.isfinite(float(KLCE2_estimator(K, err)))
    with pytest.raises(ValueError, match="2D array"):
        KLCE2_estimator(jnp.zeros((3,)), jnp.zeros((3,)))
    with pytest.raises(ValueError, match="square"):
        KLCE2_estimator(jnp.zeros((3, 4)), jnp.zeros((3,)))
