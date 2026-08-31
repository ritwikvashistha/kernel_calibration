# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `local_calibration_bias` — the LCB "error-witness" diagnostic that localizes
  *where* in feature space a model is miscalibrated (paper Eq. 7).
- `select_bandwidths` and `median_heuristic` for automatic kernel bandwidth
  selection.
- Baseline calibration metrics: `expected_calibration_error`,
  `maximum_calibration_error`, `brier_score`, and `kernel_calibration_error` (KCE,
  the global special case of KLCE).
- Plotting helpers `reliability_diagram` and `plot_local_calibration_bias` in
  `kernel_calibration.plots` (optional `viz` extra).
- Example data: `make_calibration_data` (synthetic) and `datasets.fetch_compas`.
- `KLCETestResult` — `KLCE_test` now returns a SciPy-style result object exposing
  `.statistic`, `.pvalue`, and `.null_distribution` (still unpacks as a 2-tuple).
- Example notebooks (quickstart, recalibration, COMPAS diagnostic, Type-I error
  check) and a `mkdocs-material` documentation site.
- Continuous integration (GitHub Actions) across Python 3.9–3.12.

### Changed
- Package renamed to `kernel_calibration` (imported as `kernel_calibration`).
- `KLCE_test` accepts NumPy arrays and integer seeds, and defaults to the
  corrected permutation p-value `(1 + #{null >= observed}) / (1 + iterations)`.
- `recalibrated_model` follows the scikit-learn estimator conventions:
  `verbose` is silent by default, `fit` returns `self`, and `get_params` /
  `set_params` are available.
- `rbf_kernel` now supports rectangular kernels (different sample counts).

## [0.1.0]

- Initial release: `KLCE_test` (kernel local calibration test) and
  `recalibrated_model` (kernel-penalized recalibration).
