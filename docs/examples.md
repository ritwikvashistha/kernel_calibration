# Examples

Runnable Jupyter notebooks live in the [`examples/`](https://github.com/ritwikvashistha/kernel_calibration/tree/main/examples)
directory of the repository (they render with outputs directly on GitHub).

### 1. Quickstart

[`01_quickstart.ipynb`](https://github.com/ritwikvashistha/kernel_calibration/blob/main/examples/01_quickstart.ipynb)
— the core workflow in a few lines: generate data, pick bandwidths, run
`KLCE_test`, read the result, and visualize the reliability diagram and the null
distribution. Includes the sanity check that a calibrated model is *not* rejected.

### 2. Recalibration

[`02_recalibration.ipynb`](https://github.com/ritwikvashistha/kernel_calibration/blob/main/examples/02_recalibration.ipynb)
— fix a miscalibrated model with `recalibrated_model`. The KLCE p-value climbs
above the significance threshold, ECE and Brier drop, and AUC is preserved.

### 3. Auditing a real model (COMPAS)

[`03_diagnostic_compas.ipynb`](https://github.com/ritwikvashistha/kernel_calibration/blob/main/examples/03_diagnostic_compas.ipynb)
— the flagship diagnostic. Train a recidivism-risk model on criminal-history
features, then audit its local calibration with respect to **protected attributes
the model never saw** (age and race). The global reliability diagram looks fine, but
the LCB diagnostic reveals where and for whom the risk scores are systematically off.

### 4. Type-I error check

[`04_type_i_error.ipynb`](https://github.com/ritwikvashistha/kernel_calibration/blob/main/examples/04_type_i_error.ipynb)
— empirical evidence that the test controls its false-positive rate: under a locally
calibrated model the p-values are approximately uniform and the rejection rate sits
near the nominal level.

## Reproducing the README figures

```bash
python docs/generate_figures.py
```
