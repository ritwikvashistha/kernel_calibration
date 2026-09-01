# Experiment harness

Validation and benchmark studies for `kernel_calibration`, designed to prototype
locally at `--scale smoke` and then run at `--scale full` on a server.

## Install

```bash
pip install -e ".[dev,experiments]"     # from the repo root
```

`experiments` adds `pandas`, `scikit-learn`, `matplotlib`, and `scipy`.

## Run

```bash
# fast local sanity check (seconds):
python experiments/run_all.py --scale smoke

# full study on a server (GPU auto-detected by JAX):
python experiments/run_all.py --scale full --x64 --out runs/2026-08-run1

# a single experiment:
python experiments/exp_type_i.py --scale full --x64
```

Common flags (all experiments): `--scale {smoke,small,full}`, `--out DIR`,
`--seed N`, `--x64` (enable JAX float64), `--no-plots`.

## The experiments

| Script | Question | Key outputs |
| --- | --- | --- |
| `exp_type_i.py` | Does the test hold its nominal level under the null? | rejection rate ≈ α across (n, d); p-value uniformity (KS); corrected vs. uncorrected |
| `exp_power.py` | How does power behave? | Type-II vs. dimension (paper Fig. 3); power vs. effect size |
| `exp_bandwidth.py` | How sensitive is power to bandwidth, and is the median heuristic good? | power heatmap over bandwidth grid with the median-heuristic marker |
| `exp_recalibration.py` | Which (α, β) fix local miscalibration without hurting AUC? | ECE/p-value/AUC before→after; ECE heatmap; a recommended default |
| `exp_baselines.py` | Does KLCE flag local miscalibration that ECE/Brier/KCE miss? | per-dataset table on COMPAS and Adult |

## Notes

- **`--x64`**: the KLCE² U-statistic sums O(n²) terms with cancellation; on large
  n, float32 (JAX's default) can lose precision. Compare runs with and without
  `--x64` — differences flag where double precision matters.
- **GPU**: JAX uses an available GPU automatically; the harness prints the backend
  and devices at startup.
- **Memory**: the kernel matrix is O(n²). `exp_baselines.py` caps the evaluation
  set (`max_eval`) for this reason — a reminder that large-n support is future work.
- **Long runs**: use `nohup python experiments/run_all.py --scale full --x64 &`
  or `tmux`/`screen`. Outputs (CSV + PNG) land in `--out`.
