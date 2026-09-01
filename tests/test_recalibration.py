import numpy as np

from kernel_calibration import (
    KLCE_test,
    expected_calibration_error,
    make_calibration_data,
    recalibrated_model,
    select_bandwidths,
)


def test_recalibration_reduces_local_miscalibration():
    """On planted local miscalibration, the recalibrator should lower both the
    KLCE statistic (toward zero) and the ECE."""
    X, y, f = make_calibration_data(n=800, miscalibration=0.3, seed=3)
    pw, xw = select_bandwidths(X, f)

    stat_before, _ = KLCE_test(X, y, f, pw, 100, key=0, x_kernel_width=xw)
    ece_before = expected_calibration_error(y, f)

    model = recalibrated_model(
        sigma_k=pw,
        sigma_l=xw,
        alpha=0.02,
        beta=1.0,
        num_steps=800,
        learning_rate=0.03,
        seed=0,
    )
    model.fit(f, X, y)
    f_hat = np.asarray(model.predict_proba(f, X))

    stat_after, _ = KLCE_test(X, y, f_hat, pw, 100, key=0, x_kernel_width=xw)

    assert abs(float(stat_after)) < abs(float(stat_before))
    assert expected_calibration_error(y, f_hat) < ece_before


def test_recalibrated_model_auto_bandwidth_defaults():
    """recalibrated_model() with no params set auto-selects kernel widths (median
    heuristic) and reduces ECE out of the box."""
    X, y, f = make_calibration_data(n=1200, miscalibration=0.25, seed=5)
    ece_before = expected_calibration_error(y, f)

    model = recalibrated_model(num_steps=600, seed=0)  # all other defaults
    model.fit(f, X, y)

    assert model._sigma_k is not None and model._sigma_l is not None
    f_hat = np.asarray(model.predict_proba(f, X))
    assert expected_calibration_error(y, f_hat) < ece_before
