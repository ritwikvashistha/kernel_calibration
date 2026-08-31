# API reference

Everything below is importable from the top-level `kernel_calibration` package
(e.g. `from kernel_calibration import KLCE_test`). Plotting helpers live in the
`kernel_calibration.plots` submodule.

## The test and statistic

::: kernel_calibration.kite.KLCE_test

::: kernel_calibration.kite.KLCETestResult

::: kernel_calibration.kite.KLCE2_estimator

::: kernel_calibration.kite.create_kernel

::: kernel_calibration.kite.rbf_kernel

## The diagnostic

::: kernel_calibration.lcb.local_calibration_bias

## Recalibration

::: kernel_calibration.kite.recalibrated_model

## Bandwidth selection

::: kernel_calibration.bandwidth.select_bandwidths

::: kernel_calibration.bandwidth.median_heuristic

## Baseline metrics

::: kernel_calibration.metrics.expected_calibration_error

::: kernel_calibration.metrics.maximum_calibration_error

::: kernel_calibration.metrics.brier_score

::: kernel_calibration.metrics.kernel_calibration_error

## Plotting

::: kernel_calibration.plots.reliability_diagram

::: kernel_calibration.plots.plot_local_calibration_bias

## Example data

::: kernel_calibration.datasets.make_calibration_data

::: kernel_calibration.datasets.fetch_compas
