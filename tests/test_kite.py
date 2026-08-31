import jax.numpy as jnp
import pytest
from jax import random

from kernel_calibration import KLCE_test, recalibrated_model
from kernel_calibration.kite import KLCE2_estimator, create_kernel


def _synthetic_data(n=16, seed=0):
    key = random.PRNGKey(seed)
    k1, k2, k3 = random.split(key, 3)
    X = random.normal(k1, (n, 2))
    y = (random.uniform(k2, (n,)) > 0.5).astype(jnp.float32)
    p = jnp.clip(random.uniform(k3, (n,)), 0.05, 0.95)
    return X, y, p


def test_klce2_estimator_identity_kernel_zero_error():
    n = 8
    K = jnp.ones((n, n))
    err = jnp.zeros((n,))
    assert float(KLCE2_estimator(K, err)) == 0.0


def test_klce2_estimator_off_diagonal_formula():
    K = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    err = jnp.array([1.0, -1.0])
    # Only the two off-diagonal terms: 0.5 * (1)(-1) twice, / (2*1)
    expected = (0.5 * -1.0 + 0.5 * -1.0) / 2.0
    assert float(KLCE2_estimator(K, err)) == pytest.approx(expected)


def test_create_kernel_shape():
    X, _, p = _synthetic_data()
    K = create_kernel(X, p, 0.2, 1.0)
    assert K.shape == (X.shape[0], X.shape[0])


def test_klce_test_returns_finite_pvalue():
    X, y, p = _synthetic_data()
    key = random.PRNGKey(1)
    stat, p_value = KLCE_test(X, y, p, 0.2, 50, key)
    assert jnp.isfinite(stat)
    assert 0.0 < float(p_value) <= 1.0


def test_predict_proba_before_fit_raises():
    model = recalibrated_model(num_steps=1)
    X, y, p = _synthetic_data(n=8)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_proba(p, X)


def test_recalibrated_model_fit_predict():
    X, y, p = _synthetic_data(n=12)
    model = recalibrated_model(
        num_steps=5,
        hidden_layer_sizes=(8,),
        seed=0,
    )
    model.fit(p, X, y)
    p_hat = model.predict_proba(p, X)
    assert p_hat.shape == p.shape
    assert jnp.all(p_hat > 0.0)
    assert jnp.all(p_hat < 1.0)


def test_klce_test_result_object():
    X, y, p = _synthetic_data()
    result = KLCE_test(X, y, p, 0.2, 50, key=0)
    # SciPy-style attribute access
    assert jnp.isfinite(result.statistic)
    assert 0.0 < float(result.pvalue) <= 1.0
    assert result.null_distribution.shape == (50,)
    # ...and still unpacks as a 2-tuple for backward compatibility
    stat, pval = result
    assert float(stat) == float(result.statistic)
    assert float(pval) == float(result.pvalue)


def test_klce_test_accepts_int_key_and_numpy():
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    p = np.clip(rng.uniform(size=20), 0.05, 0.95)
    y = (rng.uniform(size=20) > 0.5).astype(float)
    stat, pval = KLCE_test(X, y, p, 0.2, 50, key=7)
    assert 0.0 < float(pval) <= 1.0


def test_klce_test_uncorrected_pvalue_path():
    X, y, p = _synthetic_data()
    _, pval = KLCE_test(X, y, p, 0.2, 50, key=0, add_one_correction=False)
    assert 0.0 < float(pval) <= 1.0


def test_recalibrated_model_fit_returns_self():
    X, y, p = _synthetic_data(n=12)
    model = recalibrated_model(num_steps=3, hidden_layer_sizes=(8,), seed=0)
    assert model.fit(p, X, y) is model


def test_recalibrated_model_get_set_params():
    model = recalibrated_model(alpha=0.3)
    assert model.get_params()["alpha"] == 0.3
    assert model.set_params(alpha=0.9) is model
    assert model.alpha == 0.9


def test_recalibrated_model_set_params_invalid_raises():
    model = recalibrated_model()
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(not_a_param=1)
