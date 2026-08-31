import numpy as np

from kernel_calibration import make_calibration_data


def test_make_calibration_data_shapes_and_ranges():
    X, y, f = make_calibration_data(n=200, seed=0)
    assert X.shape == (200, 1)
    assert y.shape == (200,)
    assert f.shape == (200,)
    assert set(np.unique(y)).issubset({0.0, 1.0})
    assert np.all((f > 0) & (f < 1))


def test_make_calibration_data_zero_miscalibration_is_calibrated():
    # With miscalibration=0 the model equals the true probability everywhere,
    # so the local calibration bias should be essentially zero.
    from kernel_calibration import local_calibration_bias, select_bandwidths

    X, y, f = make_calibration_data(n=1500, miscalibration=0.0, seed=1)
    pw, xw = select_bandwidths(X, f)
    bias = np.asarray(local_calibration_bias(X, y, f, pw, xw))
    # honest global average of residuals is near zero for a calibrated model
    assert abs(float(np.mean(bias))) < 0.05


def test_make_calibration_data_miscalibration_is_detectable():
    from kernel_calibration import KLCE_test, select_bandwidths

    X, y, f = make_calibration_data(n=1500, miscalibration=0.3, seed=2)
    pw, xw = select_bandwidths(X, f)
    _, pval = KLCE_test(X, y, f, pw, 300, key=0, x_kernel_width=xw)
    assert float(pval) < 0.05  # the planted miscalibration should be rejected
