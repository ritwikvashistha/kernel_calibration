"""Precision study: float32 (shipped default) vs float64 for the KLCE^2 statistic.

The estimator sums O(n^2) terms with cancellation, so at large n the float32 XLA
path may lose precision and slightly bias the test. For identical null datasets we
compare, at each (n, d):

- the KLCE^2 statistic in float32 (the package) vs a float64 NumPy reference, and
- the Type-I rejection rate under each precision.

If the large-n Type-I drift seen in exp_type_i is a precision artifact, the float64
rejection rate should sit closer to alpha and the statistic relative error should be
non-trivial at n=5000.

    python experiments/exp_precision.py --scale smoke
    python experiments/exp_precision.py --scale full        # float32 path (shipped)
"""

from __future__ import annotations

import argparse

import numpy as np
from _common import Timer, add_common_args, calibrated_data, configure, save_table

SWEEPS = {
    "smoke": {"ns": [2000], "ds": [1], "trials": 15, "iters": 100},
    "small": {"ns": [2000, 5000], "ds": [1, 2], "trials": 50, "iters": 200},
    "full": {"ns": [2000, 5000], "ds": [1, 2], "trials": 200, "iters": 300},
}
ALPHA = 0.05


def _rbf_f64(A, B, gamma):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * A @ B.T
    np.clip(sq, 0.0, None, out=sq)
    return np.exp(-gamma * sq)


def _stat_f64(K, err):
    # off-diagonal U-statistic == (err' K err - sum_i K_ii err_i^2) / (n(n-1))
    n = err.shape[0]
    quad = float(err @ (K @ err))
    diag = float((np.diag(K) * err**2).sum())
    return (quad - diag) / (n * (n - 1))


def _null_f64(K, err, iters, rng):
    n = err.shape[0]
    diagK = np.diag(K)
    denom = n * (n - 1)
    out = np.empty(iters)
    for b in range(iters):
        e = err[rng.permutation(n)]
        out[b] = (float(e @ (K @ e)) - float((diagK * e**2).sum())) / denom
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    out = configure(args)
    cfg = SWEEPS[args.scale]

    import kernel_calibration as kc

    rows = []
    with Timer(f"precision study ({args.scale})"):
        for n in cfg["ns"]:
            for d in cfg["ds"]:
                for t in range(cfg["trials"]):
                    seed = args.seed + 1000 * t
                    X, y, f = calibrated_data(n, d, seed=seed)
                    pw, xw = kc.select_bandwidths(X, f)

                    # float32 (or x64 if --x64) via the package
                    res = kc.KLCE_test(X, y, f, pw, cfg["iters"], key=seed, x_kernel_width=xw)
                    stat_pkg = float(res.statistic)
                    pval_pkg = float(res.pvalue)

                    # float64 reference via NumPy on identical data
                    err = (y - f).astype(np.float64)
                    K64 = _rbf_f64(f, f, 1.0 / pw**2) * _rbf_f64(X, X, 1.0 / xw**2)
                    stat_f64 = _stat_f64(K64, err)
                    null_f64 = _null_f64(K64, err, cfg["iters"], np.random.default_rng(seed))
                    pval_f64 = (1.0 + np.sum(null_f64 >= stat_f64)) / (1.0 + cfg["iters"])

                    rel_err = abs(stat_pkg - stat_f64) / (abs(stat_f64) + 1e-12)
                    rows.append(
                        {
                            "n": n,
                            "d": d,
                            "trial": t,
                            "stat_pkg": stat_pkg,
                            "stat_f64": stat_f64,
                            "rel_err": rel_err,
                            "pval_pkg": pval_pkg,
                            "pval_f64": pval_f64,
                            "reject_pkg": pval_pkg < ALPHA,
                            "reject_f64": pval_f64 < ALPHA,
                        }
                    )
    save_table(rows, out / "precision_raw.csv")

    import pandas as pd

    df = pd.DataFrame(rows)
    summary = []
    for (n, d), g in df.groupby(["n", "d"]):
        summary.append(
            {
                "n": n,
                "d": d,
                "reject_pkg": g["reject_pkg"].mean(),
                "reject_f64": g["reject_f64"].mean(),
                "mean_rel_err": g["rel_err"].mean(),
                "median_rel_err": g["rel_err"].median(),
                "max_rel_err": g["rel_err"].max(),
                "n_trials": len(g),
            }
        )
    save_table(summary, out / "precision_summary.csv")
    print(f"\n[precision] backend precision vs float64 reference (alpha={ALPHA}):")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
